/*
 * hps_bench: microbenchmarks for the work the HPS would take over from the accelerator.
 *
 *   hps_bench mem      bandwidth of the reserved DDR3 region from the HPS (uncached and cached mappings)
 *   hps_bench cpu      CPU cost: INT8 GEMM with NEON, LayerNorm (float32), integer softmax
 *   hps_bench micro    region reads with ldm (8 registers) and vld1.8 {d0-d7}; LayerNorm in three timed phases
 *   hps_bench all
 *
 * Reads and writes the first 32 MiB of the reserved region (0x30000000), which nothing else uses
 * while the accelerator is idle. Static musl binary for the Cortex-A9 (NEON).
 */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define REGION_PHYS 0x30000000u
#define TEST_BYTES  (32u << 20)

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void report_bw(const char *what, size_t bytes, int reps, double secs)
{
    printf("  %-46s %8.1f MB/s\n", what, (double)bytes * reps / secs / 1e6);
    fflush(stdout);
}

/* ---------------------------------------------------------------- memory */

static uint32_t sink;

static void mem_tests(const char *label, volatile uint8_t *r, uint8_t *heap)
{
    const size_t n = TEST_BYTES;
    double t;
    printf("[mem] %s\n", label);

    t = now_s();
    for (int rep = 0; rep < 2; rep++) {
        volatile uint32_t *p = (volatile uint32_t *)r;
        for (size_t i = 0; i < n / 4; i++)
            p[i] = (uint32_t)i;
    }
    report_bw("write, 32-bit stores", n, 2, now_s() - t);

    t = now_s();
    for (int rep = 0; rep < 2; rep++) {
        volatile uint32_t *p = (volatile uint32_t *)r;
        uint32_t s = 0;
        for (size_t i = 0; i < n / 4; i++)
            s += p[i];
        sink += s;
    }
    report_bw("read, 32-bit loads", n, 2, now_s() - t);

    t = now_s();
    for (int rep = 0; rep < 4; rep++) {
        const uint8_t *p = (const uint8_t *)r;
        uint32x4_t acc = vdupq_n_u32(0);
        for (size_t i = 0; i < n; i += 64) {
            acc = vaddq_u32(acc, vreinterpretq_u32_u8(vld1q_u8(p + i)));
            acc = vaddq_u32(acc, vreinterpretq_u32_u8(vld1q_u8(p + i + 16)));
            acc = vaddq_u32(acc, vreinterpretq_u32_u8(vld1q_u8(p + i + 32)));
            acc = vaddq_u32(acc, vreinterpretq_u32_u8(vld1q_u8(p + i + 48)));
        }
        sink += vgetq_lane_u32(acc, 0);
    }
    report_bw("read, NEON 4 x 16 bytes per step", n, 4, now_s() - t);

    t = now_s();
    for (int rep = 0; rep < 4; rep++)
        memcpy(heap, (const void *)r, n);
    report_bw("read, memcpy region -> heap", n, 4, now_s() - t);

    t = now_s();
    for (int rep = 0; rep < 4; rep++) {
        uint8_t *p = (uint8_t *)r;
        uint8x16_t v = vdupq_n_u8(0x5A);
        for (size_t i = 0; i < n; i += 64) {
            vst1q_u8(p + i, v);
            vst1q_u8(p + i + 16, v);
            vst1q_u8(p + i + 32, v);
            vst1q_u8(p + i + 48, v);
        }
    }
    report_bw("write, NEON 4 x 16 bytes per step", n, 4, now_s() - t);

    t = now_s();
    for (int rep = 0; rep < 4; rep++)
        memcpy((void *)r, heap, n);
    report_bw("write, memcpy heap -> region", n, 4, now_s() - t);

    t = now_s();
    for (int rep = 0; rep < 4; rep++)
        memcpy(heap + n, heap, n);
    report_bw("(reference) memcpy heap -> heap", n, 4, now_s() - t);
}

static int run_mem(void)
{
    uint8_t *heap = malloc(2 * (size_t)TEST_BYTES);
    if (!heap) { puts("out of memory"); return 1; }
    memset(heap, 0x33, 2 * (size_t)TEST_BYTES);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }
    void *u = mmap(NULL, TEST_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, REGION_PHYS);
    if (u == MAP_FAILED) { perror("mmap uncached"); return 1; }
    mem_tests("uncached mapping (/dev/mem, O_SYNC): what accel_test uses", u, heap);
    munmap(u, TEST_BYTES);
    close(fd);

    fd = open("/dev/mem", O_RDWR);
    void *c = mmap(NULL, TEST_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, REGION_PHYS);
    if (c == MAP_FAILED) { perror("mmap cached"); return 1; }
    mem_tests("cached mapping (no O_SYNC): NOT coherent with the FPGA, for reference only", c, heap);
    munmap(c, TEST_BYTES);
    close(fd);
    free(heap);
    return 0;
}

/* ------------------------------------------------------------------- cpu */

static inline int32_t hsum_s32(int32x4_t v)
{
    int32x2_t p = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(p, p), 0);
}

/* C[M][N] int32 = A[M][K] int8 * W[N][K]^T int8 (what a linear layer needs). 2 x 2 register block. */
static void gemm_s8(const int8_t *A, const int8_t *W, int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i += 2) {
        for (int j = 0; j < N; j += 2) {
            int32x4_t a00 = vdupq_n_s32(0), a01 = a00, a10 = a00, a11 = a00;
            const int8_t *x0 = A + (size_t)i * K, *x1 = x0 + K;
            const int8_t *w0 = W + (size_t)j * K, *w1 = w0 + K;
            for (int k = 0; k < K; k += 8) {
                int8x8_t va0 = vld1_s8(x0 + k), va1 = vld1_s8(x1 + k);
                int8x8_t vb0 = vld1_s8(w0 + k), vb1 = vld1_s8(w1 + k);
                a00 = vpadalq_s16(a00, vmull_s8(va0, vb0));
                a01 = vpadalq_s16(a01, vmull_s8(va0, vb1));
                a10 = vpadalq_s16(a10, vmull_s8(va1, vb0));
                a11 = vpadalq_s16(a11, vmull_s8(va1, vb1));
            }
            C[(size_t)i * N + j]         = hsum_s32(a00);
            C[(size_t)i * N + j + 1]     = hsum_s32(a01);
            C[(size_t)(i + 1) * N + j]     = hsum_s32(a10);
            C[(size_t)(i + 1) * N + j + 1] = hsum_s32(a11);
        }
    }
}

static void bench_gemm(void)
{
    /* patch_embed-like shape on a slice, then scaled: M rows x N outputs x K inputs */
    const int M = 128, N = 256, K = 592;           /* K rounded up to a multiple of 8 */
    int8_t *A = aligned_alloc(16, (size_t)M * K), *W = aligned_alloc(16, (size_t)N * K);
    int32_t *C = malloc((size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)(rand() & 0xFF);
    for (size_t i = 0; i < (size_t)N * K; i++) W[i] = (int8_t)(rand() & 0xFF);
    gemm_s8(A, W, C, M, N, K);                      /* warm up */
    double t = now_s();
    const int reps = 5;
    for (int r = 0; r < reps; r++)
        gemm_s8(A, W, C, M, N, K);
    double s = (now_s() - t) / reps;
    double macs = (double)M * N * K;
    printf("  INT8 GEMM, NEON 2x2 block, one core: %.0f MMAC/s  (%d x %d x %d in %.1f ms)\n",
           macs / s / 1e6, M, N, K, s * 1e3);
    printf("    -> patch_embed 1024 x 768 x 588 (0.46 GMAC): about %.2f s on one core\n", 0.4628e9 / (macs / s));
    printf("    -> all linear layers of one image (about 54 GMAC): about %.0f s on one core\n", 54e9 / (macs / s));
    sink += (uint32_t)C[0];
    free(A); free(W); free(C);
}

static void bench_layernorm(void)
{
    const int rows = 1025, d = 768;
    float *x = malloc((size_t)rows * d * 4), *y = malloc((size_t)rows * d * 4);
    float *g = malloc(d * 4), *b = malloc(d * 4);
    for (int i = 0; i < rows * d; i++) x[i] = (float)(rand() % 2000 - 1000) * 0.01f;
    for (int i = 0; i < d; i++) { g[i] = 1.0f; b[i] = 0.0f; }
    double t = now_s();
    const int reps = 5;
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < rows; i++) {
            const float *xr = x + (size_t)i * d;
            float *yr = y + (size_t)i * d;
            float mean = 0.0f;
            for (int j = 0; j < d; j++) mean += xr[j];
            mean /= d;
            float var = 0.0f;
            for (int j = 0; j < d; j++) { float e = xr[j] - mean; var += e * e; }
            float inv = 1.0f / sqrtf(var / d + 1e-6f);
            for (int j = 0; j < d; j++) yr[j] = (xr[j] - mean) * inv * g[j] + b[j];
        }
    double s = (now_s() - t) / reps;
    printf("  LayerNorm float32, scalar: %.1f ms for 1025 x 768  (%.0f ns per row)\n", s * 1e3, s / rows * 1e9);
    printf("    -> 3 per block, 12 blocks (+ the 1232-wide one): about %.2f s per image\n", s * 36 * 1.0);
    sink += (uint32_t)y[5];
    free(x); free(y); free(g); free(b);
}

static void bench_softmax(void)
{
    const int rows = 1025, cols = 1025;
    uint16_t etab[1025], rtab[256];
    for (int i = 0; i < 1025; i++) etab[i] = (uint16_t)(65535.0 * exp(-i / 64.0));
    for (int i = 0; i < 256; i++) rtab[i] = (uint16_t)(65535.0 * 256.0 / (256.0 + i));
    int32_t *S = malloc((size_t)rows * cols * 4);
    uint16_t *P = malloc((size_t)rows * cols * 2);
    for (size_t i = 0; i < (size_t)rows * cols; i++) S[i] = (rand() % 200000) - 100000;
    const int32_t M0 = 1266079258;
    const int sh = 35;
    const int Pb = 16;
    double t = now_s();
    for (int i = 0; i < rows; i++) {
        const int32_t *s = S + (size_t)i * cols;
        uint16_t *p = P + (size_t)i * cols;
        int32_t mx = s[0];
        for (int j = 1; j < cols; j++) if (s[j] > mx) mx = s[j];
        uint32_t sum = 0;
        for (int j = 0; j < cols; j++) {
            int64_t d = (int64_t)(mx - s[j]);
            int64_t idx = (d * M0 + ((int64_t)1 << (sh - 1))) >> sh;
            if (idx > 1024) idx = 1024;
            uint16_t e = etab[idx];
            p[j] = e;
            sum += e;
        }
        int k = 31 - __builtin_clz(sum);
        int mant = (int)((sum >> (k - 8)) - 256);
        int psh = 15 + k - Pb;
        uint64_t r = rtab[mant];
        for (int j = 0; j < cols; j++) {
            uint64_t v = ((uint64_t)p[j] * r + ((uint64_t)1 << (psh - 1))) >> psh;
            p[j] = v > 65535 ? 65535 : (uint16_t)v;
        }
    }
    double s = now_s() - t;
    double per_el = s / ((double)rows * cols);
    printf("  Integer softmax, scalar, one head: %.1f ms for 1025 x 1025  (%.1f ns per element, %.0f cycles at 925 MHz)\n",
           s * 1e3, per_el * 1e9, per_el * 925e6);
    printf("    -> 8 heads x 12 blocks: about %.1f s per image on one core (about half with two cores)\n", s * 96);
    sink += P[7];
    free(S); free(P);
}

static int run_cpu(void)
{
    puts("[cpu] work the HPS would take over (one core; the HPS has two)");
    bench_gemm();
    bench_layernorm();
    bench_softmax();
    return 0;
}

/* ---------------------------------------------------------------- micro */

static void result_bw(const char *name, size_t bytes, double secs)
{
    printf("RESULT bench: %s %.1f MB/s\n", name, (double)bytes / secs / 1e6);
    fflush(stdout);
}

static void result_ms(const char *name, double secs)
{
    printf("RESULT bench: %s %.2f ms\n", name, secs * 1e3);
    fflush(stdout);
}

/* Four ldm of 8 registers per step: 128 bytes. r7 is left out so it can stay the frame pointer. */
static void micro_ldm(const uint8_t *r, size_t n)
{
    const int reps = 4;
    double t = now_s();
    for (int rep = 0; rep < reps; rep++) {
        const uint32_t *p = (const uint32_t *)r;
        for (size_t i = 0; i < n; i += 128) {
            __asm__ volatile(
                "ldmia %[p]!, {r0, r1, r2, r3, r4, r5, r6, r8}\n\t"
                "ldmia %[p]!, {r0, r1, r2, r3, r4, r5, r6, r8}\n\t"
                "ldmia %[p]!, {r0, r1, r2, r3, r4, r5, r6, r8}\n\t"
                "ldmia %[p]!, {r0, r1, r2, r3, r4, r5, r6, r8}\n\t"
                : [p] "+r" (p)
                :
                : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r8", "memory");
        }
    }
    result_bw("region_read_ldm8_128B", n * reps, now_s() - t);
}

/* 128 bytes per step: two groups of vld1.8 {d0-d3} + {d4-d7} (a single vld1 takes at most four D registers). */
static void micro_vld1(const uint8_t *r, size_t n)
{
    const int reps = 4;
    double t = now_s();
    for (int rep = 0; rep < reps; rep++) {
        const uint8_t *p = r;
        for (size_t i = 0; i < n; i += 128) {
            __asm__ volatile(
                "vld1.8 {d0-d3}, [%[p]]!\n\t"
                "vld1.8 {d4-d7}, [%[p]]!\n\t"
                "vld1.8 {d0-d3}, [%[p]]!\n\t"
                "vld1.8 {d4-d7}, [%[p]]!\n\t"
                : [p] "+r" (p)
                :
                : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "memory");
        }
    }
    result_bw("region_read_vld1_d0-d7_128B", n * reps, now_s() - t);
}

static int run_micro(void)
{
    puts("[micro] ARM microbenchmarks (one core)");
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }
    void *u = mmap(NULL, TEST_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, REGION_PHYS);
    if (u == MAP_FAILED) { perror("mmap uncached"); return 1; }
    micro_ldm((const uint8_t *)u, TEST_BYTES);
    micro_vld1((const uint8_t *)u, TEST_BYTES);
    munmap(u, TEST_BYTES);
    close(fd);

    /* LayerNorm of 1025 x 768 (same arithmetic as layer_norm() in ops.c), one timed pass per phase */
    const int rows = 1025, d = 768, reps = 5;
    const float epsf = 1e-6f;
    float *x = malloc((size_t)rows * d * 4), *y = malloc((size_t)rows * d * 4);
    float *g = malloc(d * 4), *b = malloc(d * 4);
    float *mu = malloc(rows * 4), *den = malloc(rows * 4);
    for (int i = 0; i < rows * d; i++) x[i] = (float)(rand() % 2000 - 1000) * 0.01f;
    for (int i = 0; i < d; i++) { g[i] = 1.0f; b[i] = 0.0f; }

    double t_sum = 0.0, t_var = 0.0, t_norm = 0.0;
    for (int rep = 0; rep < reps; rep++) {
        double t = now_s();
        for (int r = 0; r < rows; r++) {                     /* phase 1: sums -> mean */
            const float *xr = x + (size_t)r * d;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                s0 += xr[j]; s1 += xr[j + 1]; s2 += xr[j + 2]; s3 += xr[j + 3];
            }
            for (; j < d; j++)
                s0 += xr[j];
            mu[r] = (float)(((s0 + s1) + (s2 + s3)) / d);
        }
        t_sum += now_s() - t;

        t = now_s();
        for (int r = 0; r < rows; r++) {                     /* phase 2: variance -> 1/denominator */
            const float *xr = x + (size_t)r * d;
            const float m = mu[r];
            double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                const float e0 = xr[j] - m, e1 = xr[j + 1] - m, e2 = xr[j + 2] - m, e3 = xr[j + 3] - m;
                v0 += (double)(e0 * e0); v1 += (double)(e1 * e1); v2 += (double)(e2 * e2); v3 += (double)(e3 * e3);
            }
            for (; j < d; j++) {
                const float e = xr[j] - m;
                v0 += (double)(e * e);
            }
            const float var = (float)(((v0 + v1) + (v2 + v3)) / d);
            den[r] = sqrtf(var + epsf);
        }
        t_var += now_s() - t;

        t = now_s();
        for (int r = 0; r < rows; r++) {                     /* phase 3: normalise, with a division per element */
            const float *xr = x + (size_t)r * d;
            float *yr = y + (size_t)r * d;
            const float m = mu[r], dn = den[r];
            for (int k = 0; k < d; k++) {
                float t2 = (xr[k] - m) / dn;
                t2 = t2 * g[k];
                yr[k] = t2 + b[k];
            }
        }
        t_norm += now_s() - t;
    }
    result_ms("layernorm_1025x768_phase1_sums", t_sum / reps);
    result_ms("layernorm_1025x768_phase2_variance", t_var / reps);
    result_ms("layernorm_1025x768_phase3_normalise_div", t_norm / reps);
    sink += (uint32_t)(int)y[5];
    free(x); free(y); free(g); free(b); free(mu); free(den);
    return 0;
}

int main(int argc, char **argv)
{
    const char *what = argc > 1 ? argv[1] : "all";
    const int all = !strcmp(what, "all");
    int rc = 0;
    if (!strcmp(what, "mem") || all) rc |= run_mem();
    if (!strcmp(what, "cpu") || all) rc |= run_cpu();
    if (!strcmp(what, "micro") || all) rc |= run_micro();
    if (strcmp(what, "mem") && strcmp(what, "cpu") && strcmp(what, "micro") && !all) {
        puts("usage: hps_bench [mem|cpu|micro|all]");
        return 2;
    }
    return rc + (sink == 0x12345678u);
}
