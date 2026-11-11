#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include "accel_drv.h"
#include "fpga_pack.h"
#include "ops.h"

/* layout of the reserved DDR3 region used by the FPGA backend (offsets from ACCEL_REGION_PHYS) */
#define FPGA_A_OFF 0x000000u        /* activations, tile-major */
#define FPGA_O_OFF 0x800000u        /* outputs, row-major */
#define FPGA_W_OFF 0x1000000u       /* all packed weights */
#define FPGA_REGION_LEN 0x6000000u  /* 96 MiB: 16 MiB of buffers + 56 MiB of weights + attention buffers */
/* attention buffers of one head (offsets from the region base); S and P never leave the DDR3 */
#define ATT_OFF 0x5000000u
#define ATT_Q (ATT_OFF + 0x000000u)         /* 128 KiB slot: Q as activations */
#define ATT_K (ATT_OFF + 0x020000u)         /* 128 KiB slot: K^T tiles */
#define ATT_V (ATT_OFF + 0x040000u)         /* 128 KiB slot: V tiles */
#define ATT_S (ATT_OFF + 0x060000u)         /* 5 MiB slot: INT32 scores, 1040 columns per row */
#define ATT_P (ATT_OFF + 0x560000u)         /* 3 MiB slot: probabilities, activation format */
#define ATT_C (ATT_OFF + 0x860000u)         /* 1 MiB slot: context [tokens][heads * hd] INT8 */
#define ATT_END (ATT_OFF + 0x960000u)

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* run a statement and add its wall time to a counter of model_stats_t */
#define TIMED(field, stmt) do { double _t0 = now_s(); stmt; m->st.field += now_s() - _t0; } while (0)

struct model {
    const pkg_t *p;
    int tokens, dim, max_neurons;
    float *x;           /* residual stream [tokens][dim] */
    float *n;           /* LayerNorm output [tokens][max(dim, neurons)] */
    float *hh;          /* SwiGLU output [tokens][neurons] */
    float *ctxf;        /* context as float [tokens][512] */
    float *cols;        /* im2col [patches][588] */
    float *patch_out;   /* patch embedding output [patches][dim] */
    int8_t *a8;         /* quantised activations */
    int16_t *a16;
    int32_t *acc;       /* GEMM accumulators [tokens][max_out] */
    int16_t *q16, *k16, *v16, *y16, *g16, *u16, *h16;
    int8_t *q8, *k8, *v8, *ctx8;

    /* FPGA backend (NULL hw = everything on the CPU) */
    accel_hw_t *hw;
    int check, verbose, max_blocks, fpga_attn;
    uint8_t *att_stage;                 /* staging for the Q / K^T / V images of one head */
    size_t att_stage_len;
    int8_t *att_chk;                    /* CPU context used by --check */
    uint32_t w_off[PKG_MAX_LAYERS];     /* byte offset of each layer's packed weights in the region, or 0 */
    uint8_t *stage;                     /* cached staging for packing */
    size_t stage_len;
    uint8_t *rdbuf;                     /* copy of the output matrix read back from the region */
    int16_t *chk16;                     /* CPU result used by --check */
    int a_ok, a_rows, a_k, a_bits;      /* activations currently packed in the region */
    model_stats_t st;
};

static int attention_fpga(model_t *m, const block_t *B, int T, int A, int H, int hd);

static void *xalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "model: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

model_t *model_create(const pkg_t *p)
{
    model_t *m = calloc(1, sizeof *m);
    m->p = p;
    m->tokens = p->tokens;
    m->dim = p->patch_embed->out;
    int attn = 0, neu = 0;
    for (int i = 0; i < p->n_blocks; i++) {
        if (p->blocks[i].q->out > attn) attn = p->blocks[i].q->out;
        if (p->blocks[i].neurons > neu) neu = p->blocks[i].neurons;
    }
    m->max_neurons = neu;
    const size_t T = (size_t)m->tokens;
    const size_t widest = (size_t)(neu > m->dim ? neu : m->dim);
    m->x = xalloc(T * m->dim * sizeof(float));
    m->n = xalloc(T * widest * sizeof(float));
    m->hh = xalloc(T * neu * sizeof(float));
    m->ctxf = xalloc(T * attn * sizeof(float));
    m->cols = xalloc((size_t)(T - 1) * p->patch_embed->in * sizeof(float));
    m->patch_out = xalloc((size_t)(T - 1) * m->dim * sizeof(float));
    m->a8 = xalloc(T * widest);
    m->a16 = xalloc(T * widest * 2);
    m->acc = xalloc(T * widest * sizeof(int32_t));
    m->q16 = xalloc(T * attn * 2); m->k16 = xalloc(T * attn * 2); m->v16 = xalloc(T * attn * 2);
    m->y16 = xalloc(T * widest * 2);
    m->g16 = xalloc(T * neu * 2); m->u16 = xalloc(T * neu * 2); m->h16 = xalloc(T * neu * 2);
    m->q8 = xalloc(T * attn); m->k8 = xalloc(T * attn); m->v8 = xalloc(T * attn); m->ctx8 = xalloc(T * attn);
    return m;
}

void model_destroy(model_t *m)
{
    if (!m)
        return;
    free(m->x); free(m->n); free(m->hh); free(m->ctxf); free(m->cols); free(m->patch_out);
    free(m->a8); free(m->a16); free(m->acc);
    free(m->q16); free(m->k16); free(m->v16); free(m->y16); free(m->g16); free(m->u16); free(m->h16);
    free(m->q8); free(m->k8); free(m->v8); free(m->ctx8);
    free(m->stage); free(m->rdbuf); free(m->chk16); free(m->att_stage); free(m->att_chk);
    free(m);
}

/* the whole GEMM + epilogue on the CPU, from the quantised input already in a8 / a16 */
static void linear_cpu(model_t *m, const layer_t *L, int rows, int16_t *out)
{
    if (L->in_bits == 8)
        gemm_s8(m->a8, L->w, m->acc, rows, L->out, L->in);
    else
        gemm_s16(m->a16, L->w, m->acc, rows, L->out, L->in);
    epilogue_int(m->acc, rows, L->out, L->bias, L->m0, L->shift, L->out_bits, out);
}

/* the same operation on the FPGA: pack the input, run, read the output matrix back */
static int linear_fpga(model_t *m, const layer_t *L, int rows, int16_t *out)
{
    const int K = L->in, N = L->out, planes = L->in_bits > 8 ? 2 : 1;
    const int out16 = L->out_bits > 8;
    double t0 = now_s();
    if (!(m->a_ok && m->a_rows == rows && m->a_k == K && m->a_bits == L->in_bits)) {
        const size_t bytes = fpga_acts_bytes(rows, K, planes);
        if (planes == 1)
            fpga_pack_acts8(m->a8, rows, K, m->stage);
        else
            fpga_pack_acts16(m->a16, rows, K, m->stage);
        memcpy(m->hw->region + FPGA_A_OFF, m->stage, bytes);
        m->a_ok = 1; m->a_rows = rows; m->a_k = K; m->a_bits = L->in_bits;
    }
    double t1 = now_s();
    accel_op_t op = {.M = (uint32_t)rows, .N = (uint32_t)N, .K = (uint32_t)K, .in_bits = L->in_bits,
                     .out_bits = L->out_bits, .spp = planes == 2 ? 4 : 16,
                     .a_phys = ACCEL_REGION_PHYS + FPGA_A_OFF, .w_phys = ACCEL_REGION_PHYS + m->w_off[L - m->p->layers],
                     .out_phys = ACCEL_REGION_PHYS + FPGA_O_OFF};
    accel_stat_t st;
    if (accel_hw_run(m->hw, &op, &st, 60.0) != 0) {
        m->a_ok = 0;
        return -1;
    }
    double t2 = now_s();
    const size_t n = (size_t)rows * N;
    memcpy(m->rdbuf, m->hw->region + FPGA_O_OFF, n * (out16 ? 2 : 1));
    if (out16)
        memcpy(out, m->rdbuf, n * 2);
    else
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
            out[i] = ((const int8_t *)m->rdbuf)[i];
    double t3 = now_s();
    m->st.ops++;
    m->st.cycles += st.cycles;
    m->st.t_pack += t1 - t0;
    m->st.t_run += t2 - t1;
    m->st.t_read += t3 - t2;
    size_t bad = 0;
    if (m->check) {
        linear_cpu(m, L, rows, m->chk16);
        for (size_t i = 0; i < n; i++)
            bad += m->chk16[i] != out[i];
        m->st.mismatches += bad;
    }
    if (m->verbose)
        fprintf(stderr, "  fpga %-10s M=%d N=%-4d K=%-4d: %9llu cycles (%.1f ms at %.0f MHz), pack %.1f ms, run %.1f ms, "
                        "read %.1f ms%s\n", L->name, rows, N, K, (unsigned long long)st.cycles, st.cycles / (m->hw->mhz * 1e3), m->hw->mhz,
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, m->check ? (bad ? "  MISMATCH" : "  identical to CPU") : "");
    if (m->check && bad)
        fprintf(stderr, "  *** %s: %zu of %zu outputs differ from the CPU result\n", L->name, bad, n);
    return 0;
}

/* quantise the float input, run the INT GEMM and the epilogue; result in int16 container.
 * Returns 0, or -1 if the FPGA failed. reuse_quant: the input is the one quantised by the previous call. */
static int linear_int(model_t *m, const layer_t *L, const float *in, int rows, int16_t *out, int reuse_quant)
{
    if (!reuse_quant) {
        m->a_ok = 0;                    /* a new input: whatever is packed in the region is stale */
        if (L->in_bits == 8)
            TIMED(t_quant, quantize_f32(in, (size_t)rows * L->in, L->s_x, 8, m->a8));
        else
            TIMED(t_quant, quantize_f32(in, (size_t)rows * L->in, L->s_x, 16, m->a16));
    }
    if (m->hw && fpga_supports(L) && m->w_off[L - m->p->layers])
        return linear_fpga(m, L, rows, out);
    linear_cpu(m, L, rows, out);
    return 0;
}

static void dbg16(model_dbg_fn f, void *c, int b, const char *what, const int16_t *d, size_t n)
{
    char nm[48];
    if (!f)
        return;
    snprintf(nm, sizeof nm, "b%d.%s", b, what);
    f(c, nm, d, n, 2);
}

int model_forward(model_t *m, const float *img, float logits[4], model_dbg_fn dbg, void *dbg_ctx)
{
    const pkg_t *p = m->p;
    const int T = m->tokens, D = m->dim, NP = T - 1;
    const int side = p->size / p->patch;           /* 32 */
    const int pk = p->patch, ps = p->patch_embed->in;   /* 14, 588 */

    const double t_patch0 = now_s();
    /* ---- patch embedding: unfold (channel, ky, kx), INT8 GEMM, float de-quantisation ---- */
#pragma omp parallel for schedule(static) collapse(2)
    for (int py = 0; py < side; py++)
        for (int px = 0; px < side; px++) {
            float *c = m->cols + (size_t)(py * side + px) * ps;
            for (int ch = 0; ch < p->channels; ch++)
                for (int ky = 0; ky < pk; ky++)
                    for (int kx = 0; kx < pk; kx++)
                        c[(ch * pk + ky) * pk + kx] =
                            img[((size_t)ch * p->size + (size_t)(py * pk + ky)) * p->size + (size_t)(px * pk + kx)];
        }
    {
        const layer_t *L = p->patch_embed;
        quantize_f32(m->cols, (size_t)NP * ps, L->s_x, 8, m->a8);
        gemm_s8(m->a8, L->w, m->acc, NP, L->out, L->in);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < NP; t++)
            for (int c = 0; c < D; c++) {
                int64_t a = m->acc[(size_t)t * D + c];
                if (L->bias)
                    a += L->bias[c];
                m->patch_out[(size_t)t * D + c] = (float)((double)a * L->deq[c]);
            }
        if (dbg)
            dbg(dbg_ctx, "patch_embed", m->patch_out, (size_t)NP * D, -4);
        for (int c = 0; c < D; c++)
            m->x[c] = p->cls[c] + p->pos[c];
        for (int t = 0; t < NP; t++)
            for (int c = 0; c < D; c++)
                m->x[(size_t)(t + 1) * D + c] = m->patch_out[(size_t)t * D + c] + p->pos[(size_t)(t + 1) * D + c];
    }

    m->st.t_patch += now_s() - t_patch0;

    /* ---- transformer blocks ---- */
    const int nb = (m->max_blocks > 0 && m->max_blocks < p->n_blocks) ? m->max_blocks : p->n_blocks;
    for (int bi = 0; bi < nb; bi++) {
        const block_t *B = &p->blocks[bi];
        const int A = B->q->out, H = B->heads, hd = A / H, N = B->neurons;

        TIMED(t_ln, layer_norm(m->x, T, D, B->n1_g, B->n1_b, p->ln_eps, m->n));
        if (linear_int(m, B->q, m->n, T, m->q16, 0)) return -1;
        if (linear_int(m, B->k, m->n, T, m->k16, B->k->s_x == B->q->s_x)) return -1;
        if (linear_int(m, B->v, m->n, T, m->v16, B->v->s_x == B->k->s_x)) return -1;
        dbg16(dbg, dbg_ctx, bi, "q_w", m->q16, (size_t)T * A);
        dbg16(dbg, dbg_ctx, bi, "k_w", m->k16, (size_t)T * A);
        dbg16(dbg, dbg_ctx, bi, "v_w", m->v16, (size_t)T * A);

        TIMED(t_rope, rope_int(m->q16, T, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, m->q8));
        TIMED(t_rope, rope_int(m->k16, T, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, m->k8));
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * A; i++)
            m->v8[i] = (int8_t)m->v16[i];

        if (m->fpga_attn) {
            TIMED(t_attn, if (attention_fpga(m, B, T, A, H, hd)) return -1);
        } else {
            TIMED(t_attn, attention_int(p, B, m->q8, m->k8, m->v8, T, A, H, hd, m->ctx8));
        }
        if (dbg) {
            char nm[48];
            snprintf(nm, sizeof nm, "b%d.ctx_i", bi);
            dbg(dbg_ctx, nm, m->ctx8, (size_t)T * A, 1);
        }
        /* proj quantises ctx * s_ctx with its own input scale. When both scales are the same, the
         * float round trip returns the INT8 value unchanged (|c| <= 127, relative error ~1e-5 of one
         * step, far from a rounding boundary), so ctx8 is used directly. Otherwise do it the long way. */
        const int ctx_direct = B->proj->in_bits == 8 && B->proj->in == A &&
                               fabs(B->s_ctx - B->proj->s_x) <= 1e-9 * fabs(B->proj->s_x);
        if (ctx_direct) {
            TIMED(t_quant, memcpy(m->a8, m->ctx8, (size_t)T * A));
            m->a_ok = 0;
            if (linear_int(m, B->proj, NULL, T, m->y16, 1)) return -1;
        } else {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < (ptrdiff_t)T * A; i++)
                m->ctxf[i] = (float)((double)m->ctx8[i] * B->s_ctx);
            if (linear_int(m, B->proj, m->ctxf, T, m->y16, 0)) return -1;
        }
        dbg16(dbg, dbg_ctx, bi, "proj_w", m->y16, (size_t)T * D);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * D; i++)
            m->x[i] += (float)((double)m->y16[i] * B->s_proj);

        TIMED(t_ln, layer_norm(m->x, T, D, B->n2_g, B->n2_b, p->ln_eps, m->n));
        if (linear_int(m, B->fc1g, m->n, T, m->g16, 0)) return -1;
        if (linear_int(m, B->fc1x, m->n, T, m->u16, B->fc1x->s_x == B->fc1g->s_x)) return -1;
        dbg16(dbg, dbg_ctx, bi, "fc1g_w", m->g16, (size_t)T * N);
        dbg16(dbg, dbg_ctx, bi, "fc1x_w", m->u16, (size_t)T * N);

        const double t_sw0 = now_s();
        const int64_t sw_rnd = (int64_t)1 << (B->sw_sh - 1);
        const int qm_in = qmax_of(p->silu_in_bits);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * N; i++) {
            int64_t y = (int64_t)B->silu[m->g16[i] + qm_in] * m->u16[i];
            y = (y * B->sw_m0 + sw_rnd) >> B->sw_sh;
            m->h16[i] = (int16_t)(y > 32767 ? 32767 : y < -32767 ? -32767 : y);
            m->hh[i] = (float)((double)m->h16[i] * B->s_h);
        }
        m->st.t_swiglu += now_s() - t_sw0;
        dbg16(dbg, dbg_ctx, bi, "swiglu_i", m->h16, (size_t)T * N);

        TIMED(t_ln, layer_norm(m->hh, T, N, B->mlpn_g, B->mlpn_b, p->ln_eps, m->n));
        if (linear_int(m, B->fc2, m->n, T, m->y16, 0)) return -1;
        dbg16(dbg, dbg_ctx, bi, "fc2_w", m->y16, (size_t)T * D);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * D; i++)
            m->x[i] += (float)((double)m->y16[i] * B->s_fc2);
    }

    /* ---- pooling, final LayerNorm, head ---- */
    float pooled[1024], pn[1024];
    for (int c = 0; c < D; c++) {
        double s = 0.0;
        for (int t = 1; t < T; t++)
            s += m->x[(size_t)t * D + c];
        pooled[c] = (float)(s / (T - 1));
    }
    TIMED(t_ln, layer_norm(pooled, 1, D, p->fcn_g, p->fcn_b, p->ln_eps, pn));
    {
        const layer_t *L = p->head;
        int8_t q[1024];
        int32_t acc[16];
        quantize_f32(pn, (size_t)D, L->s_x, 8, q);
        gemm_s8(q, L->w, acc, 1, L->out, L->in);
        for (int c = 0; c < L->out; c++) {
            int64_t a = acc[c];
            if (L->bias)
                a += L->bias[c];
            logits[c] = (float)((double)a * L->deq[c]);
        }
    }
    if (dbg)
        dbg(dbg_ctx, "logits", logits, 4, -4);
    return 0;
}

/* ---- attention on the FPGA: per head Q.K^T (RAW32) -> softmax unit -> P.V, S and P stay in the DDR3 ---- */

/* --check, head 0 only: the scores and probabilities of rows 0..15 and of the last row against the CPU */
static size_t att_check_stages(model_t *m, const block_t *B, int T, int A, int NK, int KP, int h, int hd)
{
    const uint8_t *reg = m->hw->region;
    size_t bad_s = 0, bad_p = 0;
    int32_t *S = malloc((size_t)T * sizeof *S);
    uint16_t *P = malloc((size_t)T * sizeof *P);
    for (int n = 0; n < 17; n++) {
        const int r = n < 16 ? n : T - 1;
        const int8_t *qi = m->q8 + (size_t)r * A + (size_t)h * hd;
        for (int j = 0; j < T; j++) {
            const int8_t *kj = m->k8 + (size_t)j * A + (size_t)h * hd;
            int32_t s = 0;
            for (int d = 0; d < hd; d++)
                s += (int32_t)qi[d] * (int32_t)kj[d];
            S[j] = s;
        }
        const int32_t *hw_s = (const int32_t *)(reg + ATT_S + (size_t)r * NK * 4);
        for (int j = 0; j < T; j++)
            bad_s += hw_s[j] != S[j];
        softmax_row_int(m->p, B, S, T, P);
        const int sub = r / 16, i = r % 16;
        for (int k = 0; k < KP; k++) {
            const uint8_t hi = reg[ATT_P + (size_t)sub * 2 * KP * 16 + (size_t)k * 16 + i];
            const uint8_t lo = reg[ATT_P + (size_t)sub * 2 * KP * 16 + (size_t)KP * 16 + (size_t)k * 16 + i];
            const uint16_t got = (uint16_t)((hi << 8) | lo);
            bad_p += got != (k < T ? P[k] : 0);
        }
    }
    free(S);
    free(P);
    if (m->verbose || bad_s || bad_p)
        fprintf(stderr, "  fpga attention head %d: scores of 17 rows %s (%zu differ), probabilities %s (%zu differ)\n", h,
                bad_s ? "MISMATCH" : "identical to the CPU", bad_s, bad_p ? "MISMATCH" : "identical to the CPU", bad_p);
    return bad_s + bad_p;
}

static int attention_fpga(model_t *m, const block_t *B, int T, int A, int H, int hd)
{
    accel_hw_t *hw = m->hw;
    const int NK = ((T + 15) / 16) * 16;            /* keys padded to whole 16-key tiles */
    const int KP = (T + 1) & ~1;                    /* k extent of P and V: even */
    const int subs = (T + 15) / 16;
    const int spp_s = subs < 255 ? subs : 255;
    const int spp_pv = 16384 / (2 * KP) < 255 ? 16384 / (2 * KP) : 255;
    for (int h = 0; h < H; h++) {
        double t0 = now_s();
        size_t n;
        n = fpga_acts_bytes(T, hd, 1);
        fpga_pack_acts8_strided(m->q8 + (size_t)h * hd, T, hd, A, m->att_stage);
        memcpy(hw->region + ATT_Q, m->att_stage, n);
        n = fpga_kt_bytes(T, hd);
        fpga_pack_kt(m->k8 + (size_t)h * hd, T, A, hd, m->att_stage);
        memcpy(hw->region + ATT_K, m->att_stage, n);
        n = fpga_v_bytes(hd, KP);
        fpga_pack_v(m->v8 + (size_t)h * hd, T, A, hd, KP, (int32_t)B->av_m0, (int)B->av_sh, m->att_stage);
        memcpy(hw->region + ATT_V, m->att_stage, n);
        double t1 = now_s();

        accel_stat_t st;
        accel_op_t s_op = {.M = (uint32_t)T, .N = (uint32_t)NK, .K = (uint32_t)hd, .in_bits = 8, .out_bits = 32,
                           .spp = spp_s, .a_phys = ACCEL_REGION_PHYS + ATT_Q, .w_phys = ACCEL_REGION_PHYS + ATT_K,
                           .out_phys = ACCEL_REGION_PHYS + ATT_S, .raw32 = 1, .stride_out = (uint32_t)NK * 4};
        if (accel_hw_run(hw, &s_op, &st, 60.0) != 0)
            return -1;
        m->st.att_cyc_s += st.cycles;
        double t2 = now_s();

        accel_sm_op_t sm_op = {.rows = (uint32_t)T, .cols = (uint32_t)T, .kp = (uint32_t)KP,
                               .s_phys = ACCEL_REGION_PHYS + ATT_S, .s_stride = (uint32_t)NK * 4,
                               .p_phys = ACCEL_REGION_PHYS + ATT_P, .m0 = (uint32_t)B->sm_m0, .sh = (int)B->sm_sh};
        if (accel_hw_run_sm(hw, &sm_op, &st, 60.0) != 0)
            return -1;
        m->st.att_cyc_sm += st.cycles;
        double t3 = now_s();

        if (m->check && h == 0)
            m->st.att_mismatches += att_check_stages(m, B, T, A, NK, KP, h, hd);

        accel_op_t pv_op = {.M = (uint32_t)T, .N = (uint32_t)hd, .K = (uint32_t)KP, .in_bits = 16, .out_bits = 8,
                            .spp = spp_pv, .a_phys = ACCEL_REGION_PHYS + ATT_P, .w_phys = ACCEL_REGION_PHYS + ATT_V,
                            .out_phys = ACCEL_REGION_PHYS + ATT_C + (uint32_t)h * hd, .in_unsigned = 1,
                            .stride_out = (uint32_t)A};
        if (accel_hw_run(hw, &pv_op, &st, 60.0) != 0)
            return -1;
        m->st.att_cyc_pv += st.cycles;
        double t4 = now_s();
        m->st.t_att_pack += t1 - t0;
        m->st.t_att_s += t2 - t1;
        m->st.t_att_sm += t3 - t2;
        m->st.t_att_pv += t4 - t3;
    }
    double t5 = now_s();
    memcpy(m->ctx8, hw->region + ATT_C, (size_t)T * A);
    m->st.t_att_read += now_s() - t5;

    if (m->check) {
        const pkg_t *p = m->p;
        attention_int(p, B, m->q8, m->k8, m->v8, T, A, H, hd, m->att_chk);
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)T * A; i++)
            bad += m->att_chk[i] != m->ctx8[i];
        m->st.att_mismatches += bad;
        if (m->verbose || bad)
            fprintf(stderr, "  fpga attention, whole block: context %s (%zu of %zu differ)\n",
                    bad ? "MISMATCH" : "identical to the CPU", bad, (size_t)T * A);
    }
    return 0;
}

int model_attach_fpga(model_t *m, accel_hw_t *hw, int check, int verbose, int fpga_attn)
{
    const pkg_t *p = m->p;
    if (hw->region_len < FPGA_REGION_LEN) {
        fprintf(stderr, "model: the mapped region (%zu bytes) is smaller than the %u needed\n", hw->region_len,
                FPGA_REGION_LEN);
        return -1;
    }
    size_t max_stage = 0, max_out = 0;
    uint32_t off = FPGA_W_OFF;
    double t0 = now_s();
    int n_fpga = 0;
    uint8_t *tile = NULL;
    size_t tile_len = 0;
    for (int i = 0; i < p->n_layers; i++) {
        const layer_t *L = &p->layers[i];
        if (!fpga_supports(L))
            continue;
        const size_t bytes = fpga_weights_bytes(L);
        if (off + bytes > FPGA_REGION_LEN) {
            fprintf(stderr, "model: the packed weights do not fit the reserved region\n");
            free(tile);
            return -1;
        }
        if (bytes > tile_len) {
            free(tile);
            tile = malloc(bytes);
            tile_len = bytes;
        }
        fpga_pack_weights(L, tile);
        memcpy(hw->region + off, tile, bytes);
        m->w_off[i] = off;
        off = (off + (uint32_t)bytes + 63u) & ~63u;
        n_fpga++;
        const size_t a = fpga_acts_bytes(m->tokens, L->in, L->in_bits > 8 ? 2 : 1);
        const size_t o = (size_t)m->tokens * L->out * (L->out_bits > 8 ? 2 : 1);
        if (a > max_stage) max_stage = a;
        if (o > max_out) max_out = o;
    }
    free(tile);
    if (max_stage > FPGA_O_OFF - FPGA_A_OFF || max_out > FPGA_W_OFF - FPGA_O_OFF) {
        fprintf(stderr, "model: activation or output buffers do not fit their region slots\n");
        return -1;
    }
    m->stage = malloc(max_stage);
    m->stage_len = max_stage;
    m->rdbuf = malloc(max_out);
    m->chk16 = malloc((size_t)m->tokens * (m->max_neurons > m->dim ? m->max_neurons : m->dim) * sizeof(int16_t));
    m->hw = hw;
    m->check = check;
    m->verbose = verbose;
    m->a_ok = 0;
    if (fpga_attn) {
        if (!hw->has_softmax) {
            fprintf(stderr, "model: --fpga-attn needs a bitstream with the softmax unit (registers at 0x100)\n");
            return -1;
        }
        if (ATT_END > FPGA_REGION_LEN || m->tokens > 1040) {
            fprintf(stderr, "model: attention buffers do not fit the region\n");
            return -1;
        }
        const int hd = p->blocks[0].q->out / p->blocks[0].heads;
        size_t st = fpga_acts_bytes(m->tokens, hd, 1), k = fpga_kt_bytes(m->tokens, hd), v = fpga_v_bytes(hd, (m->tokens + 1) & ~1);
        m->att_stage_len = st > k ? (st > v ? st : v) : (k > v ? k : v);
        m->att_stage = malloc(m->att_stage_len);
        m->att_chk = malloc((size_t)m->tokens * p->blocks[0].q->out);
        m->fpga_attn = 1;
    }
    fprintf(stderr, "model: %d layers packed into the FPGA region (%.1f MiB of weights) in %.1f s\n", n_fpga,
            (off - FPGA_W_OFF) / 1048576.0, now_s() - t0);
    return 0;
}

void model_limit_blocks(model_t *m, int n) { m->max_blocks = n; }
const model_stats_t *model_stats(const model_t *m) { return &m->st; }
void model_stats_reset(model_t *m) { memset(&m->st, 0, sizeof m->st); }

float model_score(const float logits[4])
{
    double mx = logits[0];
    for (int i = 1; i < 4; i++)
        if (logits[i] > mx)
            mx = logits[i];
    double e[4], s = 0.0;
    for (int i = 0; i < 4; i++) {
        e[i] = exp((double)logits[i] - mx);
        s += e[i];
    }
    return (float)((e[1] + e[2] + e[3]) / s);
}

void model_prepare_rgb8(const uint8_t *rgb, int size, float *out)
{
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    const size_t plane = (size_t)size * size;
    for (size_t i = 0; i < plane; i++)
        for (int c = 0; c < 3; c++) {
            float v = (float)rgb[i * 3 + c] / 255.0f;
            out[c * plane + i] = (v - mean[c]) / stdv[c];
        }
}
