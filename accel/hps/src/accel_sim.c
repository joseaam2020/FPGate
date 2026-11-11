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

struct accel_sim {
    pthread_t th;
    int running;
    volatile int done;          /* set by the worker when the output is completely written */
    accel_op_t op;
    int failed;
    uint64_t cycles;
    accel_hw_t *hw;
};

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
    const int out16 = op->out_bits > 8;
    const size_t esz = out16 ? 2 : 1;
    const uint32_t base = ACCEL_REGION_PHYS;
    sim->failed = 1;
    if (op->raw32 || op->in_unsigned || N % 16 || op->a_phys < base || op->w_phys < base || op->out_phys < base)
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
    else
        gemm_s16(a16, w8, acc, M, N, K);
    epilogue_int(acc, M, N, bias, m0, sh, op->out_bits, y);

    const double macs = (double)M * N * K;
    const double dur = macs / SIM_MACS_PER_S;
    sim->cycles = (uint64_t)(macs / 256.0) + 1000;
    pause_s(dur / 2);                           /* compute, then write the rows progressively */
    for (int s = 0; s < SIM_SLICES; s++) {
        const int r0 = (int)((int64_t)M * s / SIM_SLICES), r1 = (int)((int64_t)M * (s + 1) / SIM_SLICES);
        for (int r = r0; r < r1; r++) {
            uint8_t *o = O + (size_t)r * stride;
            if (out16)
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
    h->has_softmax = 0;
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
    if (h->sim && h->sim->running)
        pthread_join(h->sim->th, NULL);
    free(h->sim);
    h->sim = NULL;
}
