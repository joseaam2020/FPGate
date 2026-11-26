/*
 * Software model of the linear engine (see accel_drv.h, accel_hw_open_sim). For tests on a PC only:
 * it decodes the packed activation / weight formats of accel/DECISIONES.md (H2), computes the GEMM and
 * the epilogue with the CPU kernels and writes the output matrix, on a worker thread, with a delay.
 * The output is written in slices with a pause between them so that a caller that reads or reuses a
 * buffer before the operation is done gets wrong data, which the comparison with the CPU backend shows.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "accel_drv.h"
#include "ops.h"

#define SIM_MACS_PER_S 25.0e9       /* about the engine: 256 MAC per cycle at 100 MHz */
#define SIM_SLICES 8
#define SIM_SM_ELEMS_PER_S 100.0e6  /* about the softmax unit: one score per cycle at 100 MHz */

struct accel_sim {
    pthread_t th;
    int running;
    volatile int done;          /* set by the worker when the output is completely written */
    accel_op_t op;
    int failed;
    uint64_t cycles;
    accel_hw_t *hw;
    /* softmax unit: its own thread and state, so it can run while the engine does */
    pthread_t sm_th;
    int sm_running;
    volatile int sm_done;
    accel_sm_op_t sm_op;
    accel_sm_sim_ctx_t sm_ctx;
    int sm_failed;
    uint64_t sm_cycles;
    /* tables of the RoPE epilogue and of the fused SwiGLU, as loaded through accel_hw_load_rope / the driver */
    int16_t rope_tab[2][2][32][16];
    int16_t silu[ACCEL_SILU_ENTRIES];
};

void accel_sim_load_rope(accel_hw_t *h, const int16_t tab[2][2][32][16])
{
    memcpy(h->sim->rope_tab, tab, sizeof h->sim->rope_tab);
}

void accel_sim_load_silu(accel_hw_t *h, const int16_t *silu)
{
    memcpy(h->sim->silu, silu, sizeof h->sim->silu);
}

static void pause_s(double s)
{
    if (s <= 0)
        return;
    struct timespec t = {.tv_sec = (time_t)s, .tv_nsec = (long)((s - (time_t)s) * 1e9)};
    nanosleep(&t, NULL);
}

static void *sim_thread(void *arg)
{
    struct accel_sim *sim = arg;
    accel_hw_t *h = sim->hw;
    const accel_op_t *op = &sim->op;
    const int M = (int)op->M, N = (int)op->N, K = (int)op->K;
    const int planes = op->in_bits > 8 ? 2 : 1;
    const int raw = op->raw32;
    const int out16 = !raw && op->out_bits > 8;
    const size_t esz = raw ? 4 : out16 ? 2 : 1;
    const uint32_t base = ACCEL_REGION_PHYS;
    const int rope = op->rope_hd != 0, sw = op->swiglu;
    sim->failed = 1;
    if (rope && (op->rope_hd != 64 || N % 64 || raw || out16 || sw || op->row0 + (unsigned)M > 1025))
        return NULL;
    if (sw && (N % 32 || raw || out16 || !op->stride_out))
        return NULL;
    if (N % 16 || (op->in_unsigned && planes != 2) || (raw && planes != 1) || op->a_phys < base || op->w_phys < base || op->out_phys < base)
        return NULL;
    const size_t stride = op->stride_out ? op->stride_out : (size_t)N * esz;
    const size_t a_bytes = (size_t)((M + 15) / 16) * planes * K * 16;
    const size_t w_bytes = (size_t)(N / 16) * (256 + 16 * (size_t)K);
    const size_t o_bytes = (size_t)(M - 1) * stride + (size_t)N * esz;
    if (op->a_phys - base + a_bytes > h->region_len || op->w_phys - base + w_bytes > h->region_len ||
        op->out_phys - base + o_bytes > h->region_len)
        return NULL;
    const uint8_t *A = h->region + (op->a_phys - base);
    const uint8_t *W = h->region + (op->w_phys - base);
    uint8_t *O = h->region + (op->out_phys - base);

    int8_t *a8 = NULL, *w8 = malloc((size_t)N * K);
    int16_t *a16 = NULL, *y = malloc((size_t)M * N * sizeof *y);
    int32_t *acc = malloc((size_t)M * N * sizeof *acc), *bias = malloc(N * 4), *m0 = malloc(N * 4), *sh = malloc(N * 4);
    if (planes == 1)
        a8 = malloc((size_t)M * K);
    else
        a16 = malloc((size_t)M * K * 2);
    if (!w8 || !y || !acc || !bias || !m0 || !sh || (!a8 && !a16))
        goto done;

    for (int r = 0; r < M; r++) {
        const size_t sub = (size_t)r / 16, i = (size_t)r % 16;
        for (int k = 0; k < K; k++) {
            if (planes == 1)
                a8[(size_t)r * K + k] = (int8_t)A[sub * K * 16 + (size_t)k * 16 + i];
            else {
                const uint8_t *hi = A + sub * 2 * K * 16, *lo = hi + (size_t)K * 16;
                a16[(size_t)r * K + k] = (int16_t)(((unsigned)hi[(size_t)k * 16 + i] << 8) | lo[(size_t)k * 16 + i]);
            }
        }
    }
    for (int t = 0; t < N / 16; t++) {
        const uint8_t *c = W + (size_t)t * (256 + 16 * (size_t)K);
        for (int j = 0; j < 16; j++) {
            const uint8_t *e = c + 16 * j;
            const int n = t * 16 + j;
            bias[n] = (int32_t)((uint32_t)e[0] | (uint32_t)e[1] << 8 | (uint32_t)e[2] << 16 | (uint32_t)e[3] << 24);
            m0[n] = (int32_t)((uint32_t)e[4] | (uint32_t)e[5] << 8 | (uint32_t)e[6] << 16 | (uint32_t)e[7] << 24);
            sh[n] = e[8];
            for (int k = 0; k < K; k++)
                w8[(size_t)n * K + k] = (int8_t)c[256 + (size_t)k * 16 + j];
        }
    }
    if (planes == 1)
        gemm_s8(a8, w8, acc, M, N, K);
    else if (op->in_unsigned) {
        /* UQ0.16 probabilities: same order and int32 wrap-around as attention_head_ref */
        for (int r = 0; r < M; r++)
            for (int n = 0; n < N; n++) {
                uint32_t s = 0;
                for (int k = 0; k < K; k++)
                    s += (uint32_t)(uint16_t)a16[(size_t)r * K + k] * (uint32_t)(int32_t)w8[(size_t)n * K + k];
                acc[(size_t)r * N + n] = (int32_t)s;
            }
    } else
        gemm_s16(a16, w8, acc, M, N, K);
    if (!raw)
        epilogue_int(acc, M, N, bias, m0, sh, rope ? 16 : op->out_bits, y);
    if (rope) {
        /* the RoPE unit of the epilogue (rope_unit.sv): interleaved pairs, separable Q1.14 tables, one rounding shift,
         * saturation to +-127; token 0 is only shifted */
        const int rsh = (int)(h->rope_cfg & 63), pre = (int)((h->rope_cfg >> 8) & 63);
        const int64_t rnd = (int64_t)1 << (rsh - 1);
        int16_t *y2 = malloc((size_t)M * N * sizeof *y2);
        if (!y2)
            goto done;
        for (int r = 0; r < M; r++) {
            const int t = (int)op->row0 + r;
            for (int n = 0; n < N; n++) {
                const int d = n % 64, half = d >> 5, pi = (d & 31) >> 1;
                const int16_t *x = y + (size_t)r * N + n - d;
                int64_t v;
                if (t == 0)
                    v = (x[d] + ((int64_t)1 << (pre - 1))) >> pre;
                else {
                    const int u = t - 1, yx = half ? (u & 31) : (u >> 5);
                    const int64_t c = sim->rope_tab[0][half][yx][pi], s = sim->rope_tab[1][half][yx][pi];
                    const int64_t rot = (d & 1) ? x[d - 1] : -(int64_t)x[d + 1];
                    v = ((int64_t)x[d] * c + rot * s + rnd) >> rsh;
                }
                y2[(size_t)r * N + n] = (int16_t)(v > 127 ? 127 : v < -127 ? -127 : v);
            }
        }
        free(y);
        y = y2;
    }
    if (sw) {
        /* swiglu_stage: tiles alternate gate(j), up(j); h = sat16((silu[g] * u * M0 + 2^(shift-1)) >> shift) */
        const int64_t rnd = (int64_t)1 << (op->sw_shift - 1);
        int16_t *hbuf = malloc((size_t)M * (N / 2) * sizeof *hbuf);
        if (!hbuf)
            goto done;
        for (int r = 0; r < M; r++)
            for (int jt = 0; jt < N / 32; jt++)
                for (int c = 0; c < 16; c++) {
                    const int g = y[(size_t)r * N + jt * 32 + c], u = y[(size_t)r * N + jt * 32 + 16 + c];
                    int64_t v = (int64_t)sim->silu[g + 127] * u;
                    v = (v * (int64_t)op->sw_m0 + rnd) >> op->sw_shift;
                    hbuf[(size_t)r * (N / 2) + jt * 16 + c] = (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
                }
        free(y);
        y = hbuf;
    }

    const double macs = (double)M * N * K;
    const double dur = macs / SIM_MACS_PER_S;
    sim->cycles = (uint64_t)(macs / 256.0) + 1000;
    pause_s(dur / 2);                           /* compute, then write the rows progressively */
    for (int s = 0; s < SIM_SLICES; s++) {
        const int r0 = (int)((int64_t)M * s / SIM_SLICES), r1 = (int)((int64_t)M * (s + 1) / SIM_SLICES);
        for (int r = r0; r < r1; r++) {
            uint8_t *o = O + (size_t)r * stride;
            if (raw)
                memcpy(o, acc + (size_t)r * N, (size_t)N * 4);
            else if (sw)
                memcpy(o, y + (size_t)r * (N / 2), (size_t)N);
            else if (out16)
                memcpy(o, y + (size_t)r * N, (size_t)N * 2);
            else
                for (int n = 0; n < N; n++)
                    o[n] = (uint8_t)(int8_t)y[(size_t)r * N + n];
        }
        pause_s(dur / 2 / SIM_SLICES);
    }
    sim->failed = 0;
done:
    free(a8); free(a16); free(w8); free(y); free(acc); free(bias); free(m0); free(sh);
    return NULL;
}

/* softmax unit: reads S row by row, writes P (two 16-bit planes, KP words, zero padded) row by row with pauses */
static void *sm_thread(void *arg)
{
    struct accel_sim *sim = arg;
    accel_hw_t *h = sim->hw;
    const accel_sm_op_t *op = &sim->sm_op;
    const uint32_t base = ACCEL_REGION_PHYS;
    const int rows = (int)op->rows, cols = (int)op->cols, kp = (int)op->kp;
    sim->sm_failed = 1;
    if (!sim->sm_ctx.pkg || !sim->sm_ctx.block || rows <= 0 || cols <= 0 || kp < cols || (op->s_stride & 3) ||
        op->s_phys < base || op->p_phys < base)
        return NULL;
    const size_t s_bytes = (size_t)(rows - 1) * op->s_stride + (size_t)cols * 4;
    const size_t p_bytes = (size_t)((rows + 15) / 16) * 2 * kp * 16;
    if (op->s_phys - base + s_bytes > h->region_len || op->p_phys - base + p_bytes > h->region_len)
        return NULL;
    const uint8_t *S = h->region + (op->s_phys - base);
    uint8_t *P = h->region + (op->p_phys - base);
    int32_t *srow = malloc((size_t)cols * sizeof *srow);
    uint16_t *prow = malloc((size_t)cols * sizeof *prow);
    if (!srow || !prow)
        goto done;
    const double macs = (double)rows * cols;
    sim->sm_cycles = (uint64_t)macs + 1000;
    const double dur = macs / SIM_SM_ELEMS_PER_S;
    for (int s = 0; s < SIM_SLICES; s++) {
        const int r0 = (int)((int64_t)rows * s / SIM_SLICES), r1 = (int)((int64_t)rows * (s + 1) / SIM_SLICES);
        pause_s(dur / SIM_SLICES);
        for (int r = r0; r < r1; r++) {
            memcpy(srow, S + (size_t)r * op->s_stride, (size_t)cols * 4);
            softmax_row_int(sim->sm_ctx.pkg, sim->sm_ctx.block, srow, cols, prow);
            const size_t sub = (size_t)r / 16, i = (size_t)r % 16;
            uint8_t *hi = P + sub * 2 * kp * 16, *lo = hi + (size_t)kp * 16;
            for (int k = 0; k < kp; k++) {
                const uint16_t v = k < cols ? prow[k] : 0;
                hi[(size_t)k * 16 + i] = (uint8_t)(v >> 8);
                lo[(size_t)k * 16 + i] = (uint8_t)v;
            }
        }
    }
    /* padding rows of the last subtile are zero */
    for (int r = rows; r % 16; r++) {
        const size_t sub = (size_t)r / 16, i = (size_t)r % 16;
        uint8_t *hi = P + sub * 2 * kp * 16, *lo = hi + (size_t)kp * 16;
        for (int k = 0; k < kp; k++)
            hi[(size_t)k * 16 + i] = lo[(size_t)k * 16 + i] = 0;
    }
    sim->sm_failed = 0;
done:
    free(srow);
    free(prow);
    return NULL;
}

static void *sm_entry(void *arg)
{
    struct accel_sim *sim = arg;
    sm_thread(arg);
    __sync_synchronize();
    sim->sm_done = 1;
    return NULL;
}

int accel_sim_sm_busy(const accel_hw_t *h)
{
    return h->sim->sm_running && !h->sim->sm_done;
}

int accel_sim_sm_start(accel_hw_t *h, const accel_sm_op_t *op)
{
    struct accel_sim *s = h->sim;
    if (s->sm_running) {
        fprintf(stderr, "accel(sim): softmax start while a softmax is still in flight\n");
        return -1;
    }
    s->sm_op = *op;
    if (op->sim_ctx)
        s->sm_ctx = *(const accel_sm_sim_ctx_t *)op->sim_ctx;
    else
        memset(&s->sm_ctx, 0, sizeof s->sm_ctx);
    s->sm_failed = 1;
    s->sm_done = 0;
    if (pthread_create(&s->sm_th, NULL, sm_entry, s) != 0)
        return -1;
    s->sm_running = 1;
    return 0;
}

int accel_sim_sm_wait(accel_hw_t *h, accel_stat_t *st)
{
    struct accel_sim *s = h->sim;
    if (!s->sm_running) {
        fprintf(stderr, "accel(sim): softmax wait without a started softmax\n");
        return -1;
    }
    pthread_join(s->sm_th, NULL);
    s->sm_running = 0;
    if (st) {
        st->wall_s = 0;
        st->cycles = s->sm_cycles;
        st->bytes_rd = st->bytes_wr = 0;
    }
    if (s->sm_failed) {
        fprintf(stderr, "accel(sim): softmax rows=%u cols=%u failed (bad descriptor or no sim_ctx)\n", s->sm_op.rows,
                s->sm_op.cols);
        return -1;
    }
    return 0;
}

/* thread entry: run the operation, then publish completion */
static void *sim_entry(void *arg)
{
    struct accel_sim *sim = arg;
    sim_thread(arg);
    __sync_synchronize();
    sim->done = 1;
    return NULL;
}

int accel_sim_busy(const accel_hw_t *h)
{
    return h->sim->running && !h->sim->done;
}

int accel_hw_open_sim(accel_hw_t *h, size_t region_len)
{
    memset(h, 0, sizeof *h);
    h->fd = -1;
    h->region = calloc(1, region_len);
    h->sim = calloc(1, sizeof *h->sim);
    if (!h->region || !h->sim) {
        fprintf(stderr, "accel: no memory for the simulated engine (%zu bytes)\n", region_len);
        free(h->region);
        free(h->sim);
        memset(h, 0, sizeof *h);
        h->fd = -1;
        return -1;
    }
    h->region_len = region_len;
    h->mhz = 100.0;
    h->hw_version = ACCEL_HW_VERSION_H5B;   /* models the h5b engine: concurrency is safe */
    h->has_softmax = 1;
    h->sim->hw = h;
    return 0;
}

int accel_sim_start(accel_hw_t *h, const accel_op_t *op)
{
    struct accel_sim *s = h->sim;
    if (s->running) {
        fprintf(stderr, "accel(sim): start while an operation is still in flight\n");
        return -1;
    }
    s->op = *op;
    s->failed = 1;
    s->done = 0;
    if (pthread_create(&s->th, NULL, sim_entry, s) != 0)
        return -1;
    s->running = 1;
    return 0;
}

int accel_sim_wait(accel_hw_t *h, accel_stat_t *st)
{
    struct accel_sim *s = h->sim;
    if (!s->running) {
        fprintf(stderr, "accel(sim): wait without a started operation\n");
        return -1;
    }
    pthread_join(s->th, NULL);
    s->running = 0;
    if (st) {
        st->wall_s = 0;
        st->cycles = s->cycles;
        st->bytes_rd = st->bytes_wr = 0;
    }
    if (s->failed) {
        fprintf(stderr, "accel(sim): operation M=%u N=%u K=%u failed or is not supported by the simulated engine\n",
                s->op.M, s->op.N, s->op.K);
        return -1;
    }
    return 0;
}

void accel_sim_free(accel_hw_t *h)
{
    if (h->sim && h->sim->sm_running)
        pthread_join(h->sim->sm_th, NULL);
    if (h->sim && h->sim->running)
        pthread_join(h->sim->th, NULL);
    free(h->sim);
    h->sim = NULL;
}
