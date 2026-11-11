/*
 * fpgate-accel: classifies the images the proxy sends over a Unix socket.
 *
 *   fpgate-accel SOCKET --pkg DIR --index FILE [--verify] [--quiet]
 *   fpgate-accel --selftest [N] --pkg DIR --index FILE [--stages DIR]
 *
 * Wire format: proxy/src/accel_proto.h (source of truth). Behaviour follows mock_accel.c:
 * one client and one image at a time, UNSUPPORTED for anything but 448x448 RGB8 HWC,
 * STATUS requests answered OK. The classification runs on the CPU backend for now.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "accel_drv.h"
#include "accel_proto.h"
#include "model.h"
#include "ops.h"
#include "pkg.h"
#include "sha256.h"

static struct {
    int fpga, fpga_attn, check, verbose, blocks;
    int sim;                    /* software engine instead of the FPGA (tests on a PC) */
    int chunk;                  /* rows per engine operation, -1 = model default, 0 = no overlap */
    int pipeline;               /* row-chunk pipeline between attentions */
    int logits;                 /* print the logits with 9 digits (compare two runs bit by bit) */
    accel_hw_t hw;
} g_opt = {.chunk = -1, .pipeline = 1};

#define FPGA_REGION_MAP 0x6000000u      /* 96 MiB, see model.c */

/* connect the model to the FPGA when requested; 0 on success */
static int setup_backend(model_t *model)
{
    model_limit_blocks(model, g_opt.blocks);
    if (!g_opt.fpga)
        return 0;
    if (g_opt.sim) {
        if (g_opt.fpga_attn) {
            fprintf(stderr, "--sim simulates the linear engine only; it cannot be combined with --fpga-attn\n");
            return -1;
        }
        if (accel_hw_open_sim(&g_opt.hw, FPGA_REGION_MAP) != 0)
            return -1;
    } else if (accel_hw_open(&g_opt.hw, FPGA_REGION_MAP) != 0)
        return -1;
    if (model_attach_fpga(model, &g_opt.hw, g_opt.check, g_opt.verbose, g_opt.fpga_attn) != 0)
        return -1;
    if (g_opt.chunk >= 0)
        model_set_chunk(model, g_opt.chunk);
    model_set_pipeline(model, g_opt.pipeline);
    return 0;
}

static void print_stats(const model_t *model, double total)
{
    const model_stats_t *s = model_stats(model);
    /* with overlap the three parts run at the same time, so the real cost is the wall time of the layers */
    const double fpga_hps = s->t_lin_wall > 0 ? s->t_lin_wall : s->t_pack + s->t_run + s->t_read;
    const double listed = s->t_patch + s->t_ln + s->t_rope + s->t_attn + s->t_swiglu + s->t_quant;
    if (g_opt.fpga)
        printf("    FPGA: %d operations, %llu cycles (%.3f s at %.0f MHz); HPS: pack %.3f s, wait %.3f s, read %.3f s "
               "(layers wall %.3f s)%s\n",
               s->ops, s->cycles, s->cycles / (g_opt.hw.mhz * 1e6), g_opt.hw.mhz, s->t_pack, s->t_run, s->t_read,
               fpga_hps,
               g_opt.check ? (s->mismatches ? "  -- OUTPUTS DIFFER FROM THE CPU" : "  -- every output identical to the CPU") : "");
    if (g_opt.fpga && s->pl_wall > 0)
        printf("    pipeline between attentions: wall %.3f s; ARM busy: LayerNorm %.2f, quantise %.2f, pack %.2f, read %.2f, "
               "SwiGLU %.2f, RoPE %.2f, other %.2f; ARM waiting for the engine %.2f s\n",
               s->pl_wall, s->pl_ln, s->pl_quant, s->pl_pack, s->pl_read, s->pl_swiglu, s->pl_rope, s->pl_other, s->pl_idle);
    if (g_opt.fpga_attn)
        printf("    attention on the FPGA: %llu + %llu + %llu cycles (scores, softmax, context; %.3f s at %.0f MHz); HPS: pack Q/K/V %.3f s, "
               "scores %.3f s, softmax %.3f s, context %.3f s, read %.3f s%s\n",
               s->att_cyc_s, s->att_cyc_sm, s->att_cyc_pv, (s->att_cyc_s + s->att_cyc_sm + s->att_cyc_pv) / (g_opt.hw.mhz * 1e6), g_opt.hw.mhz,
               s->t_att_pack, s->t_att_s, s->t_att_sm, s->t_att_pv, s->t_att_read,
               g_opt.check ? (s->att_mismatches ? "  -- DIFFERS FROM THE CPU" : "  -- identical to the CPU") : "");
    printf("    time by stage (s): attention %.2f, LayerNorm %.2f, patch_embed %.2f, RoPE %.2f, SwiGLU %.2f, quantise %.2f, "
           "%s %.2f, rest %.2f, total %.2f\n",
           s->t_attn, s->t_ln, s->t_patch, s->t_rope, s->t_swiglu, s->t_quant,
           s->pl_wall > 0 ? "PIPELINE (LayerNorm, quantise, linear layers, SwiGLU, RoPE, sums, all overlapped)" : "FPGA path",
           fpga_hps, total - listed - fpga_hps, total);
}

static volatile sig_atomic_t g_stop;
static void on_signal(int s) { (void)s; g_stop = 1; }

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ socket */

static int read_full(int fd, void *buf, size_t n)
{
    char *p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r == 0)
            return -1;
        if (r < 0) {
            if (errno == EINTR && !g_stop)
                continue;
            return -1;
        }
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n)
{
    const char *p = buf;
    while (n) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

static void reply(int fd, const accel_req_hdr_t *rq, uint8_t status, float score, uint32_t cycles, uint32_t us)
{
    accel_resp_hdr_t rs;
    memset(&rs, 0, sizeof rs);
    memcpy(rs.magic, ACCEL_RESP_MAGIC, 4);
    rs.version = ACCEL_PROTO_VERSION;
    rs.type = rq->type;
    rs.status = status;
    rs.request_id = rq->request_id;
    rs.score_nsfw = score;
    rs.hw_cycles = cycles;
    rs.latency_us = us;
    (void)write_full(fd, &rs, sizeof rs);
}

static int serve(const char *path, const pkg_t *pkg, int quiet)
{
    model_t *model = model_create(pkg);
    float *img = malloc((size_t)pkg->size * pkg->size * 3 * sizeof(float));
    if (!img) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    if (setup_backend(model) != 0)
        return 1;
    unlink(path);
    int ls = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a;
    memset(&a, 0, sizeof a);
    a.sun_family = AF_UNIX;
    snprintf(a.sun_path, sizeof a.sun_path, "%s", path);
    if (ls < 0 || bind(ls, (struct sockaddr *)&a, sizeof a) != 0 || listen(ls, 4) != 0) {
        perror("fpgate-accel");
        return 1;
    }
    fprintf(stderr, "fpgate-accel on %s: %s backend, model %dx%d\n", path, g_opt.fpga ? "FPGA linear layers" : "CPU",
            pkg->size, pkg->size);

    while (!g_stop) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0)
            continue;
        for (;;) {
            accel_req_hdr_t rq;
            if (read_full(fd, &rq, sizeof rq) != 0)
                break;
            if (memcmp(rq.magic, ACCEL_REQ_MAGIC, 4) != 0)
                break;
            if (rq.version != ACCEL_PROTO_VERSION || rq.payload_len > ACCEL_MAX_PAYLOAD) {
                reply(fd, &rq, ACCEL_UNSUPPORTED, 0, 0, 0);
                break;
            }
            unsigned char *px = rq.payload_len ? malloc(rq.payload_len) : NULL;
            if (rq.payload_len && (!px || read_full(fd, px, rq.payload_len) != 0)) {
                free(px);
                break;
            }
            if (rq.type == ACCEL_MSG_STATUS) {
                reply(fd, &rq, ACCEL_OK, 0, 0, 0);
                free(px);
                continue;
            }
            if (rq.type != ACCEL_MSG_CLASSIFY || rq.channels != 3 || rq.pixfmt != ACCEL_PIXFMT_RGB8_HWC ||
                rq.payload_len != (uint32_t)rq.width * rq.height * 3u || !rq.payload_len ||
                rq.width != pkg->size || rq.height != pkg->size) {
                reply(fd, &rq, ACCEL_UNSUPPORTED, 0, 0, 0);
                free(px);
                continue;
            }
            double t0 = now_s();
            float logits[4];
            model_prepare_rgb8(px, pkg->size, img);
            free(px);
            model_stats_reset(model);
            int rc = model_forward(model, img, logits, NULL, NULL);
            uint32_t us = (uint32_t)((now_s() - t0) * 1e6);
            if (rc != 0) {
                reply(fd, &rq, ACCEL_HW_ERROR, 0, 0, us);
                continue;
            }
            float score = model_score(logits);
            if (!quiet)
                fprintf(stderr, "classify id=%u %ux%u -> %.4f (%.2f s)\n", rq.request_id, rq.width, rq.height,
                        (double)score, us / 1e6);
            reply(fd, &rq, ACCEL_OK, score, (uint32_t)model_stats(model)->cycles, us);
        }
        close(fd);
    }
    unlink(path);
    free(img);
    model_destroy(model);
    return 0;
}

/* ---------------------------------------------------------------- selftest */

typedef struct {
    const char *dir;
    int checked;
    const char *patch_sha;      /* expected SHA-256 of the patch-embedding output, or NULL */
} stage_ctx_t;

static void stage_cb(void *vctx, const char *name, const void *data, size_t count, int elem_bytes)
{
    stage_ctx_t *c = vctx;
    if (!strcmp(name, "patch_embed") && c->patch_sha) {
        sha256_t h;
        char hex[65];
        sha256_init(&h);
        sha256_update(&h, data, count * sizeof(float));
        sha256_hex(&h, hex);
        printf("    patch_embed (exact stage): SHA-256 %s\n", !strcmp(hex, c->patch_sha) ? "MATCHES the reference" : "DIFFERS");
        return;
    }
    if (!c->dir || elem_bytes < 0 || (strncmp(name, "b0.", 3) && strncmp(name, "b11.", 4)))
        return;
    char path[600];
    snprintf(path, sizeof path, "%s/%s.i64", c->dir, name);
    FILE *f = fopen(path, "rb");
    if (!f)
        return;
    int64_t *ref = malloc(count * sizeof *ref);
    size_t got = fread(ref, sizeof *ref, count, f);
    fclose(f);
    if (got != count) {
        printf("    stage %-12s: reference has %zu values, expected %zu\n", name, got, count);
        free(ref);
        return;
    }
    size_t diff = 0;
    int64_t maxd = 0;
    for (size_t i = 0; i < count; i++) {
        int64_t v = elem_bytes == 1 ? ((const int8_t *)data)[i] : elem_bytes == 2 ? ((const int16_t *)data)[i]
                                                                                  : ((const int32_t *)data)[i];
        int64_t d = v > ref[i] ? v - ref[i] : ref[i] - v;
        if (d) {
            diff++;
            if (d > maxd)
                maxd = d;
        }
    }
    printf("    stage %-12s: %9zu values, %8zu differ (%.4f %%), max |diff| = %lld\n", name, count, diff,
           100.0 * diff / count, (long long)maxd);
    c->checked++;
    free(ref);
}


static int load_i64(const char *dir, const char *name, int64_t **out, size_t count)
{
    char path[600];
    snprintf(path, sizeof path, "%s/%s.i64", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    *out = malloc(count * sizeof **out);
    size_t got = fread(*out, sizeof **out, count, f);
    fclose(f);
    return got == count ? 0 : -1;
}

static size_t count_diff(const int64_t *ref, const int64_t *got, size_t n)
{
    size_t d = 0;
    for (size_t i = 0; i < n; i++)
        d += ref[i] != got[i];
    return d;
}

/* Integer kernels with the reference's own inputs: they must match the reference exactly. */
static int oracle(const pkg_t *p, int bi, const char *dir)
{
    const block_t *B = &p->blocks[bi];
    const int T = p->tokens, A = B->q->out, D = p->patch_embed->out, N = B->neurons, H = B->heads, hd = A / H;
    char nm[48];
    int bad = 0;
    int64_t *rq, *rk, *rv, *rctx, *rproj, *rg, *ru, *rsw;
    snprintf(nm, sizeof nm, "b%d.q_w", bi);     if (load_i64(dir, nm, &rq, (size_t)T * A)) return 1;
    snprintf(nm, sizeof nm, "b%d.k_w", bi);     if (load_i64(dir, nm, &rk, (size_t)T * A)) return 1;
    snprintf(nm, sizeof nm, "b%d.v_w", bi);     if (load_i64(dir, nm, &rv, (size_t)T * A)) return 1;
    snprintf(nm, sizeof nm, "b%d.ctx_i", bi);   if (load_i64(dir, nm, &rctx, (size_t)T * A)) return 1;
    snprintf(nm, sizeof nm, "b%d.proj_w", bi);  if (load_i64(dir, nm, &rproj, (size_t)T * D)) return 1;
    snprintf(nm, sizeof nm, "b%d.fc1g_w", bi);  if (load_i64(dir, nm, &rg, (size_t)T * N)) return 1;
    snprintf(nm, sizeof nm, "b%d.fc1x_w", bi);  if (load_i64(dir, nm, &ru, (size_t)T * N)) return 1;
    snprintf(nm, sizeof nm, "b%d.swiglu_i", bi); if (load_i64(dir, nm, &rsw, (size_t)T * N)) return 1;

    /* (1) RoPE + attention from the reference q, k, v */
    int16_t *q16 = malloc((size_t)T * A * 2), *k16 = malloc((size_t)T * A * 2);
    int8_t *q8 = malloc((size_t)T * A), *k8 = malloc((size_t)T * A), *v8 = malloc((size_t)T * A), *ctx = malloc((size_t)T * A);
    for (size_t i = 0; i < (size_t)T * A; i++) {
        q16[i] = (int16_t)rq[i];
        k16[i] = (int16_t)rk[i];
        v8[i] = (int8_t)rv[i];
    }
    rope_int(q16, T, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, q8);
    rope_int(k16, T, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, k8);
    attention_int(p, B, q8, k8, v8, T, A, H, hd, ctx);
    int64_t *got = malloc((size_t)T * A * sizeof *got);
    for (size_t i = 0; i < (size_t)T * A; i++)
        got[i] = ctx[i];
    size_t d = count_diff(rctx, got, (size_t)T * A);
    printf("    oracle b%-2d RoPE + softmax + P.V from reference q,k,v : %zu of %zu differ\n", bi, d, (size_t)T * A);
    bad |= d != 0;

    /* (2) proj GEMM from the reference INT8 context */
    float *ctxf = malloc((size_t)T * A * sizeof *ctxf);
    for (size_t i = 0; i < (size_t)T * A; i++)
        ctxf[i] = (float)((double)rctx[i] * B->s_ctx);
    int8_t *a8 = malloc((size_t)T * A);
    int32_t *acc = malloc((size_t)T * (D > N ? D : N) * sizeof *acc);
    int16_t *y16 = malloc((size_t)T * (D > N ? D : N) * 2);
    quantize_f32(ctxf, (size_t)T * A, B->proj->s_x, 8, a8);
    gemm_s8(a8, B->proj->w, acc, T, D, A);
    epilogue_int(acc, T, D, B->proj->bias, B->proj->m0, B->proj->shift, B->proj->out_bits, y16);
    got = realloc(got, (size_t)T * D * sizeof *got);
    for (size_t i = 0; i < (size_t)T * D; i++)
        got[i] = y16[i];
    d = count_diff(rproj, got, (size_t)T * D);
    printf("    oracle b%-2d proj GEMM + epilogue from reference ctx    : %zu of %zu differ\n", bi, d, (size_t)T * D);
    bad |= d != 0;

    /* (3) SwiGLU from the reference gate and up outputs */
    got = realloc(got, (size_t)T * N * sizeof *got);
    const int qm_in = qmax_of(p->silu_in_bits);
    const int64_t rnd = (int64_t)1 << (B->sw_sh - 1);
    for (size_t i = 0; i < (size_t)T * N; i++) {
        int64_t y = (int64_t)B->silu[rg[i] + qm_in] * ru[i];
        y = (y * B->sw_m0 + rnd) >> B->sw_sh;
        got[i] = y > 32767 ? 32767 : y < -32767 ? -32767 : y;
    }
    d = count_diff(rsw, got, (size_t)T * N);
    printf("    oracle b%-2d SwiGLU from reference gate, up             : %zu of %zu differ\n", bi, d, (size_t)T * N);
    bad |= d != 0;

    free(rq); free(rk); free(rv); free(rctx); free(rproj); free(rg); free(ru); free(rsw);
    free(q16); free(k16); free(q8); free(k8); free(v8); free(ctx); free(got); free(ctxf); free(a8); free(acc); free(y16);
    return bad;
}

/* Reference data for the attention RTL tests: for the chosen heads of blocks 0 and 11 write the INT8 head slices of
 * q (after RoPE), k, v, the INT32 scores S, the UQ0.16 probabilities P and the INT8 context, all from the reference
 * q/k/v stages, and check the context against the reference stage (b<N>.ctx_i). */
static int write_file(const char *dir, const char *name, const void *data, size_t n)
{
    char path[700];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, 1, n, f) != n) {
        fprintf(stderr, "cannot write %s\n", path);
        return -1;
    }
    fclose(f);
    return 0;
}

static int dump_attn(const pkg_t *p, const char *stages, const char *out)
{
    static const int blocks[2] = {0, 11};
    static const int heads_dumped[2] = {0, 5};
    const int T = p->tokens;
    int bad = 0;
    for (int bn = 0; bn < 2; bn++) {
        const int bi = blocks[bn];
        const block_t *B = &p->blocks[bi];
        const int A = B->q->out, hd = A / B->heads;
        char nm[48];
        int64_t *rq, *rk, *rv, *rctx;
        snprintf(nm, sizeof nm, "b%d.q_w", bi);   if (load_i64(stages, nm, &rq, (size_t)T * A)) return 1;
        snprintf(nm, sizeof nm, "b%d.k_w", bi);   if (load_i64(stages, nm, &rk, (size_t)T * A)) return 1;
        snprintf(nm, sizeof nm, "b%d.v_w", bi);   if (load_i64(stages, nm, &rv, (size_t)T * A)) return 1;
        snprintf(nm, sizeof nm, "b%d.ctx_i", bi); if (load_i64(stages, nm, &rctx, (size_t)T * A)) return 1;
        int16_t *q16 = malloc((size_t)T * A * 2), *k16 = malloc((size_t)T * A * 2);
        int8_t *q8 = malloc((size_t)T * A), *k8 = malloc((size_t)T * A), *v8 = malloc((size_t)T * A);
        for (size_t i = 0; i < (size_t)T * A; i++) {
            q16[i] = (int16_t)rq[i];
            k16[i] = (int16_t)rk[i];
            v8[i] = (int8_t)rv[i];
        }
        rope_int(q16, T, A, B->heads, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, q8);
        rope_int(k16, T, A, B->heads, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, k8);
        for (int hn = 0; hn < 2; hn++) {
            const int h = heads_dumped[hn];
            char dir[600], path[700];
            snprintf(dir, sizeof dir, "%s/attn_b%d_h%d", out, bi, h);
            mkdir(out, 0755);
            mkdir(dir, 0755);
            int8_t *qh = malloc((size_t)T * hd), *kh = malloc((size_t)T * hd), *vh = malloc((size_t)T * hd);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < hd; d++) {
                    qh[(size_t)t * hd + d] = q8[(size_t)t * A + (size_t)h * hd + d];
                    kh[(size_t)t * hd + d] = k8[(size_t)t * A + (size_t)h * hd + d];
                    vh[(size_t)t * hd + d] = v8[(size_t)t * A + (size_t)h * hd + d];
                }
            int32_t *S = malloc((size_t)T * T * sizeof *S);
            uint16_t *P = malloc((size_t)T * T * sizeof *P);
            int8_t *ctx = malloc((size_t)T * hd);
            attention_head_ref(p, B, qh, kh, vh, T, hd, hd, S, P, ctx);
            size_t diff = 0;
            for (int t = 0; t < T; t++)
                for (int d = 0; d < hd; d++)
                    diff += (int64_t)ctx[(size_t)t * hd + d] != rctx[(size_t)t * A + (size_t)h * hd + d];
            printf("  block %2d head %d: context from the reference q,k,v vs %s.ctx_i: %zu of %zu differ\n", bi, h, "stage",
                   diff, (size_t)T * hd);
            bad |= diff != 0;
            int32_t smin = S[0], smax = S[0];
            for (size_t i = 1; i < (size_t)T * T; i++) {
                if (S[i] < smin) smin = S[i];
                if (S[i] > smax) smax = S[i];
            }
            write_file(dir, "q8.bin", qh, (size_t)T * hd);
            write_file(dir, "k8.bin", kh, (size_t)T * hd);
            write_file(dir, "v8.bin", vh, (size_t)T * hd);
            write_file(dir, "s.bin", S, (size_t)T * T * sizeof *S);
            write_file(dir, "p.bin", P, (size_t)T * T * sizeof *P);
            write_file(dir, "ctx.bin", ctx, (size_t)T * hd);
            snprintf(path, sizeof path, "%s/params.txt", dir);
            FILE *f = fopen(path, "w");
            fprintf(f, "%d %d %lld %lld %lld %lld\n", T, hd, (long long)B->sm_m0, (long long)B->sm_sh,
                    (long long)B->av_m0, (long long)B->av_sh);
            fclose(f);
            printf("    wrote %s (S range %d .. %d)\n", dir, smin, smax);
            free(qh); free(kh); free(vh); free(S); free(P); free(ctx);
        }
        free(rq); free(rk); free(rv); free(rctx); free(q16); free(k16); free(q8); free(k8); free(v8);
    }
    return bad;
}

static float expected_logit(const vector_ref_t *v, int i)
{
    float f;
    memcpy(&f, &v->logits_hex[i], sizeof f);
    return f;
}

static int selftest(const pkg_t *pkg, int n, const char *stages)
{
    model_t *model = model_create(pkg);
    float *img = malloc((size_t)pkg->size * pkg->size * 3 * sizeof(float));
    if (setup_backend(model) != 0)
        return 1;
    stage_ctx_t sc = {stages, 0, NULL};
    int same_decision = 0, total = 0, partial_runs = 0, partial_bad = 0;
    double worst = 0.0;
    for (int i = 0; i < n && i < pkg->n_vectors; i++) {
        const vector_ref_t *v = &pkg->vectors[i];
        char path[600];
        snprintf(path, sizeof path, "%s/%s", pkg->dir, v->path);
        FILE *f = fopen(path, "rb");
        size_t cnt = (size_t)pkg->size * pkg->size * 3;
        if (!f || fread(img, sizeof(float), cnt, f) != cnt) {
            fprintf(stderr, "cannot read %s\n", path);
            return 1;
        }
        fclose(f);
        if (i == 0) {
            /* the proxy sends uint8 RGB: rebuild it from the reference tensor and normalise again */
            static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
            static const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
            const size_t plane = (size_t)pkg->size * pkg->size;
            uint8_t *u8 = malloc(plane * 3);
            float *again = malloc(plane * 3 * sizeof(float));
            for (size_t k = 0; k < plane; k++)
                for (int c = 0; c < 3; c++) {
                    float v = rintf((img[c * plane + k] * stdv[c] + mean[c]) * 255.0f);
                    u8[k * 3 + c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            model_prepare_rgb8(u8, pkg->size, again);
            size_t bad = 0;
            for (size_t k = 0; k < plane * 3; k++)
                bad += memcmp(&again[k], &img[k], sizeof(float)) != 0;
            printf("normalisation uint8 RGB -> float32 CHW: %zu of %zu values differ from the reference tensor\n", bad,
                   plane * 3);
            free(u8);
            free(again);
        }
        float lg[4];
        double t0 = now_s();
        sc.patch_sha = v->patch_sha[0] ? v->patch_sha : NULL;
        model_stats_reset(model);
        if (g_opt.fpga)
            model_set_pipeline(model, g_opt.pipeline && !(i == 0 && stages));   /* the stage table needs whole layers */
        model_forward(model, img, lg, i == 0 ? stage_cb : NULL, &sc);
        double dt = now_s() - t0;
        if (g_opt.blocks > 0) {
            /* only the first blocks ran: the logits mean nothing, what matters is the per-layer check */
            printf("vector %2d: partial run of %d block(s), %.1f s\n", i, g_opt.blocks, dt);
            print_stats(model, dt);
            partial_bad |= model_stats(model)->mismatches != 0;
            partial_runs++;
            fflush(stdout);
            continue;
        }
        double md = 0.0;
        float ex[4];
        for (int c = 0; c < 4; c++) {
            ex[c] = expected_logit(v, c);
            double d = fabs((double)lg[c] - (double)ex[c]);
            if (d > md)
                md = d;
        }
        if (g_opt.logits)
            printf("    logits %.9g %.9g %.9g %.9g\n", lg[0], lg[1], lg[2], lg[3]);
        float s_got = model_score(lg), s_ref = model_score(ex);
        int same = (s_got >= 0.5f) == (s_ref >= 0.5f);
        same_decision += same;
        total++;
        if (md > worst)
            worst = md;
        printf("vector %2d (%s): max |logit diff| = %.5f  score %.5f (ref %.5f)  decision %s  %.1f s\n", i, v->label,
               md, (double)s_got, (double)s_ref, same ? "same" : "DIFFERENT", dt);
        print_stats(model, dt);
        fflush(stdout);
    }
    if (partial_runs) {
        free(img);
        model_destroy(model);
        return partial_bad ? 1 : 0;
    }
    int exact = 0;
    if (stages) {
        printf("\nInteger kernels fed with the reference inputs (must be exact):\n");
        exact |= oracle(pkg, 0, stages);
        exact |= oracle(pkg, 11, stages);
        printf("  -> %s\n", exact ? "NOT exact" : "bit-exact");
    }
    printf("\n%d/%d decisions identical, worst logit difference %.5f\n", same_decision, total, worst);
    free(img);
    model_destroy(model);
    return (same_decision == total && !exact) ? 0 : 1;
}

/* --------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    const char *sock = NULL, *pkg_dir = NULL, *index = NULL, *stages = NULL, *dump_dir = NULL;
    int do_selftest = 0, n_self = 20, verify = 0, quiet = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pkg") && i + 1 < argc)
            pkg_dir = argv[++i];
        else if (!strcmp(argv[i], "--index") && i + 1 < argc)
            index = argv[++i];
        else if (!strcmp(argv[i], "--stages") && i + 1 < argc)
            stages = argv[++i];
        else if (!strcmp(argv[i], "--dump-attn") && i + 1 < argc)
            dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--verify"))
            verify = 1;
        else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
        else if (!strcmp(argv[i], "--fpga"))
            g_opt.fpga = 1;
        else if (!strcmp(argv[i], "--fpga-attn"))
            g_opt.fpga = g_opt.fpga_attn = 1;
        else if (!strcmp(argv[i], "--sim"))
            g_opt.fpga = g_opt.sim = 1;
        else if (!strcmp(argv[i], "--no-overlap"))
            g_opt.chunk = 0;
        else if (!strcmp(argv[i], "--no-pipeline"))
            g_opt.pipeline = 0;
        else if (!strcmp(argv[i], "--logits"))
            g_opt.logits = 1;
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc)
            g_opt.chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check"))
            g_opt.check = 1;
        else if (!strcmp(argv[i], "--verbose"))
            g_opt.verbose = 1;
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc)
            g_opt.blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selftest")) {
            do_selftest = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                n_self = atoi(argv[++i]);
        } else if (argv[i][0] != '-' && !sock)
            sock = argv[i];
        else {
            fprintf(stderr, "usage: %s SOCKET --pkg DIR --index FILE [--verify] [--quiet]\n"
                            "       %s --selftest [N] --pkg DIR --index FILE [--stages DIR]\n", argv[0], argv[0]);
            return 2;
        }
    }
    if (!pkg_dir || !index || (!sock && !do_selftest && !dump_dir)) {
        fprintf(stderr, "a package directory, an index and a socket path (or --selftest) are required\n");
        return 2;
    }
    if (sock && strlen(sock) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        fprintf(stderr, "socket path too long\n");
        return 2;
    }

    pkg_t *pkg = malloc(sizeof *pkg);
    if (pkg_load(pkg, pkg_dir, index) != 0)
        return 1;
    if (verify) {
        double t0 = now_s();
        if (pkg_verify(pkg) != 0)
            return 1;
        fprintf(stderr, "package verified (SHA-256) in %.1f s\n", now_s() - t0);
    }
    if (dump_dir) {
        if (!stages) {
            fprintf(stderr, "--dump-attn needs --stages DIR\n");
            return 2;
        }
        printf("Attention reference data -> %s\n", dump_dir);
        return dump_attn(pkg, stages, dump_dir);
    }
    if (do_selftest)
        return selftest(pkg, n_self, stages);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    int rc = serve(sock, pkg, quiet);
    pkg_unload(pkg);
    free(pkg);
    return rc;
}
