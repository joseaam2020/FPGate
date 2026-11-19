/*
 * step_bench: cost of each ARM piece of one pipeline chunk (192 rows), with the real functions of accel/hps/src.
 * Tells which part of the fused MLP step (SwiGLU, LayerNorm 3072, 16-bit quantisation, 16-bit packing) and of
 * the 8-bit PREP (quantisation, packing, copy) costs what on the Cortex-A9. Run with OMP_NUM_THREADS=1 and =2.
 *
 * Build (PC):
 *   docker run --rm -v "${PWD}:/repo" -w /repo/accel/tools/board fpgate-proxy-dev arm-linux-musleabihf-gcc \
 *     -std=gnu11 -O3 -Wall -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -ffp-contract=off -fopenmp -static \
 *     -I../../hps/src -I../../../proxy/src step_bench.c ../../hps/src/ops.c ../../hps/src/fpga_pack.c \
 *     ../../hps/src/pkg.c ../../hps/src/sha256.c -lm -pthread -o step_bench
 */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fpga_pack.h"
#include "ops.h"

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#define ROWS 192
#define NEU 3072
#define DIM 768
#define REPS 20

static void result(const char *name, double secs)
{
    printf("RESULT bench: %-34s %7.2f ms\n", name, secs * 1e3 / REPS);
}

static float frand(void) { return (float)rand() / RAND_MAX * 8.0f - 4.0f; }

int main(void)
{
    printf("threads: %d\n", omp_get_max_threads());
    srand(1);
    /* MLP tail inputs */
    int16_t *g16 = malloc((size_t)ROWS * NEU * 2), *u16 = malloc((size_t)ROWS * NEU * 2);
    int8_t *u8 = malloc((size_t)ROWS * NEU);
    float *hh = malloc((size_t)ROWS * NEU * 4), *nrm = malloc((size_t)ROWS * NEU * 4);
    int16_t *a16 = malloc((size_t)ROWS * NEU * 2);
    uint8_t *stage = malloc((size_t)ROWS * NEU * 2 + 4096);
    float *gam = malloc(NEU * 4), *bet = malloc(NEU * 4);
    const int qm_in = 32767;
    int16_t *silu = malloc((size_t)(2 * qm_in + 1) * 2);
    for (int i = 0; i < 2 * qm_in + 1; i++)
        silu[i] = (int16_t)(rand() % 65536 - 32768);
    for (size_t i = 0; i < (size_t)ROWS * NEU; i++) {
        g16[i] = (int16_t)(rand() % 20000 - 10000);
        u16[i] = (int16_t)(rand() % 20000 - 10000);
        u8[i] = (int8_t)(rand() % 256 - 128);
        hh[i] = frand();
    }
    for (int i = 0; i < NEU; i++) { gam[i] = 1.0f + frand() * 0.01f; bet[i] = frand() * 0.01f; }
    const int32_t sw_m0 = 1234567;
    const int sw_sh = 31;
    const int64_t sw_rnd = (int64_t)1 << (sw_sh - 1);
    const double s_h = 3.1e-4;
    double t;

    /* 1. SwiGLU per element (the PS_MLP formula), u as INT8 like this model */
    t = now_s();
    for (int r = 0; r < REPS; r++) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ROWS; i++) {
            const int16_t *gr = g16 + (size_t)i * NEU;
            const int8_t *ur = u8 + (size_t)i * NEU;
            float *hr = hh + (size_t)i * NEU;
            for (int k = 0; k < NEU; k++) {
                int64_t y = (int64_t)silu[gr[k] + qm_in] * ur[k];
                y = (y * sw_m0 + sw_rnd) >> sw_sh;
                const int16_t h = (int16_t)(y > 32767 ? 32767 : y < -32767 ? -32767 : y);
                hr[k] = (float)((double)h * s_h);
            }
        }
    }
    result("swiglu_192x3072", now_s() - t);

    /* 2. LayerNorm 192 x 3072 */
    t = now_s();
    for (int r = 0; r < REPS; r++)
        layer_norm(hh, ROWS, NEU, gam, bet, 1e-6, nrm);
    result("layernorm_192x3072", now_s() - t);

    /* 3. 16-bit quantisation, one call (OpenMP inside) and per row (as PS_MLP does) */
    t = now_s();
    for (int r = 0; r < REPS; r++)
        quantize_f32(nrm, (size_t)ROWS * NEU, 2.7e-4, 16, a16);
    result("quantize16_192x3072_onecall", now_s() - t);
    t = now_s();
    for (int r = 0; r < REPS; r++) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ROWS; i++)
            quantize_f32(nrm + (size_t)i * NEU, NEU, 2.7e-4, 16, a16 + (size_t)i * NEU);
    }
    result("quantize16_192x3072_perrow", now_s() - t);

    /* 4. 16-bit packing of the chunk */
    t = now_s();
    for (int r = 0; r < REPS; r++)
        fpga_pack_acts16(a16, ROWS, NEU, stage);
    result("pack16_192x3072", now_s() - t);

    /* 5. the 8-bit PREP of a 768-wide layer: quantise + pack + copy */
    float *x = malloc((size_t)ROWS * DIM * 4);
    int8_t *a8 = malloc((size_t)ROWS * DIM);
    uint8_t *st8 = malloc((size_t)ROWS * DIM + 4096), *dst8 = malloc((size_t)ROWS * DIM + 4096);
    for (size_t i = 0; i < (size_t)ROWS * DIM; i++) x[i] = frand();
    t = now_s();
    for (int r = 0; r < REPS; r++)
        quantize_f32(x, (size_t)ROWS * DIM, 0.031, 8, a8);
    result("quantize8_192x768", now_s() - t);
    t = now_s();
    for (int r = 0; r < REPS; r++)
        fpga_pack_acts8(a8, ROWS, DIM, st8);
    result("pack8_192x768", now_s() - t);
    t = now_s();
    for (int r = 0; r < REPS; r++)
        memcpy(dst8, st8, (size_t)ROWS * DIM);
    result("memcpy_147KB", now_s() - t);

    /* 6. LayerNorm 192 x 768 and the residual add */
    float *y768 = malloc((size_t)ROWS * DIM * 4);
    int16_t *y16 = malloc((size_t)ROWS * DIM * 2);
    for (size_t i = 0; i < (size_t)ROWS * DIM; i++) y16[i] = (int16_t)(rand() % 20000 - 10000);
    t = now_s();
    for (int r = 0; r < REPS; r++)
        layer_norm(x, ROWS, DIM, gam, bet, 1e-6, y768);
    result("layernorm_192x768", now_s() - t);
    t = now_s();
    for (int r = 0; r < REPS; r++) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)ROWS * DIM; i++)
            x[i] += (float)((double)y16[i] * 1.7e-4);
    }
    result("residual_192x768", now_s() - t);

    /* 7. an empty OpenMP region, to see the fork/join cost */
    volatile int sink = 0;
    t = now_s();
    for (int r = 0; r < REPS * 50; r++) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < 2; i++)
            sink += i;
    }
    printf("RESULT bench: %-34s %7.1f us\n", "omp_region_empty", (now_s() - t) * 1e6 / (REPS * 50));
    return 0;
}
