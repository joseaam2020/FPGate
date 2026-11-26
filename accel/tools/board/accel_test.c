/*
 * Board test of the FPGate accelerator (H4). Runs as root on the HPS.
 *
 *   accel_test regs                    registers, error codes, soft reset
 *   accel_test mem                     read/write test of the reserved DDR3 region
 *   accel_test layer DIR [DIR...]      run packed layers and compare with the model, bit by bit
 *   accel_test faults DIR [--bad-addr] error injection on a small layer; --bad-addr also points
 *                                      the DMA outside the SDRAM (expects a read timeout)
 *   accel_test all DIR [DIR...]        regs + mem + layers + faults (on the first DIR)
 *   accel_test concurrent [rows]       scores engine and softmax unit alone, then at the same time, then at
 *                                      the same time with a memcpy thread (default rows 1025; try 32 first)
 *
 * A layer directory holds params6.txt or params.txt ("M N K IN_BITS OUT_BITS [SPP]"), mem_a.bin
 * (tile-major activations), mem_w.bin (weight tiles with epilogue constants) and y.bin (expected
 * output, row-major). Formats: see accel/DECISIONES.md, section "H2".
 *
 * The reserved DDR3 region must exist (Linux booted with mem=768M): see README.md.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#include "accel_drv.h"
#include "fpga_pack.h"

#define LW_BASE      0xFF200000u
#define LW_LEN       0x1000u
#define REGION_PHYS  0x30000000u          /* top 256 MB of the 1 GB DDR3 (Linux limited to 768 MB) */
#define REGION_LEN   0x01000000u          /* the first 16 MB are enough for the test layers */
#define A_OFF        0x000000u
#define W_OFF        0x400000u
#define O_OFF        0x600000u

#define R_VERSION    0x00
#define R_CTRL       0x04
#define R_STATUS     0x08
#define R_ERR_CODE   0x0C
#define R_OP         0x10
#define R_DIM_M      0x14
#define R_DIM_N      0x18
#define R_DIM_K      0x1C
#define R_ADDR_A     0x20
#define R_ADDR_W     0x24
#define R_ADDR_OUT   0x28
#define R_CYCLES_LO  0x38
#define R_CYCLES_HI  0x3C
#define R_BYTES_RD   0x40
#define R_BYTES_WR   0x44
#define R_HEARTBEAT  0x48
#define R_STRIDE_OUT 0x4C
#define R_SCRATCH    0x50

#define CTRL_START      (1u << 0)
#define CTRL_RESET_SOFT (1u << 1)
#define CTRL_CLEAR_ERR  (1u << 3)
#define ST_BUSY  1u
#define ST_DONE  2u
#define ST_ERROR 4u

static volatile uint8_t  *csr;
static volatile uint32_t *region;
static int checks, fails;

static uint32_t rd(unsigned off) { return *(volatile uint32_t *)(csr + off); }
static void wr(unsigned off, uint32_t v) { *(volatile uint32_t *)(csr + off) = v; }

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void check(int ok, const char *what, uint32_t got, uint32_t exp)
{
    checks++;
    if (ok) {
        printf("  PASS  %s\n", what);
    } else {
        fails++;
        printf("  FAIL  %s (got 0x%08x, expected 0x%08x)\n", what, got, exp);
    }
}

/* the region is mapped uncached (device-like): only aligned 32-bit accesses */
static void copy_to_dev(uint32_t off, const uint8_t *src, size_t n)
{
    volatile uint32_t *d = region + off / 4;
    for (size_t i = 0; i < n; i += 4) {
        uint32_t w = 0;
        memcpy(&w, src + i, (n - i) < 4 ? (n - i) : 4);
        d[i / 4] = w;
    }
    __sync_synchronize();
}

static void copy_from_dev(uint8_t *dst, uint32_t off, size_t n)
{
    volatile uint32_t *s = region + off / 4;
    for (size_t i = 0; i < n; i += 4) {
        uint32_t w = s[i / 4];
        memcpy(dst + i, &w, (n - i) < 4 ? (n - i) : 4);
    }
}

static void fill_dev(uint32_t off, uint8_t v, size_t n)
{
    volatile uint32_t *d = region + off / 4;
    uint32_t w = v * 0x01010101u;
    for (size_t i = 0; i < n; i += 4)
        d[i / 4] = w;
    __sync_synchronize();
}

static uint8_t *read_file(const char *dir, const char *name, size_t *len)
{
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(n + 4);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len = n;
    return buf;
}

struct layer {
    char name[128];
    int M, N, K, ib, ob, spp;
    uint8_t *a, *w, *y;
    size_t alen, wlen, ylen;
};

static int load_layer(const char *dir, struct layer *L)
{
    char path[512];
    FILE *f;
    memset(L, 0, sizeof *L);
    const char *slash = strrchr(dir, '/');
    snprintf(L->name, sizeof L->name, "%s", slash ? slash + 1 : dir);
    snprintf(path, sizeof path, "%s/params6.txt", dir);
    f = fopen(path, "r");
    if (!f) {
        snprintf(path, sizeof path, "%s/params.txt", dir);
        f = fopen(path, "r");
    }
    if (!f) {
        printf("cannot open params in %s\n", dir);
        return -1;
    }
    L->spp = 2;
    if (fscanf(f, "%d %d %d %d %d %d", &L->M, &L->N, &L->K, &L->ib, &L->ob, &L->spp) < 5) {
        printf("bad params in %s\n", dir);
        fclose(f);
        return -1;
    }
    fclose(f);
    L->a = read_file(dir, "mem_a.bin", &L->alen);
    L->w = read_file(dir, "mem_w.bin", &L->wlen);
    L->y = read_file(dir, "y.bin", &L->ylen);
    if (!L->a || !L->w || !L->y) {
        printf("missing mem_a.bin / mem_w.bin / y.bin in %s\n", dir);
        return -1;
    }
    if (L->alen > W_OFF - A_OFF || L->wlen > O_OFF - W_OFF || L->ylen > REGION_LEN - O_OFF) {
        printf("layer %s does not fit the test layout\n", L->name);
        return -1;
    }
    return 0;
}

static void free_layer(struct layer *L)
{
    free(L->a);
    free(L->w);
    free(L->y);
}

static void upload(const struct layer *L)
{
    copy_to_dev(A_OFF, L->a, L->alen);
    copy_to_dev(W_OFF, L->w, L->wlen);
}

static void program(const struct layer *L)
{
    unsigned stride = L->N * (L->ob > 8 ? 2 : 1);
    wr(R_ADDR_A, REGION_PHYS + A_OFF);
    wr(R_ADDR_W, REGION_PHYS + W_OFF);
    wr(R_ADDR_OUT, REGION_PHYS + O_OFF);
    wr(R_STRIDE_OUT, stride);
    wr(R_DIM_M, L->M);
    wr(R_DIM_N, L->N);
    wr(R_DIM_K, L->K);
    wr(R_OP, ((uint32_t)L->spp << 8) | ((L->ob > 8) << 2) | (1u << 1) | (L->ib > 8));
}

/* wait until BUSY drops; returns elapsed ms, or -1 on timeout */
static double wait_idle(double timeout_ms)
{
    double t0 = now_ms();
    while (rd(R_STATUS) & ST_BUSY)
        if (now_ms() - t0 > timeout_ms)
            return -1;
    return now_ms() - t0;
}

static int compare_output(const struct layer *L, int verbose)
{
    uint8_t *got = malloc(L->ylen + 4);
    int bad = 0;
    copy_from_dev(got, O_OFF, L->ylen);
    for (size_t i = 0; i < L->ylen; i++)
        if (got[i] != L->y[i]) {
            if (verbose && bad < 5)
                printf("    mismatch at output byte %zu: got 0x%02x expected 0x%02x (row %zu)\n", i, got[i], L->y[i],
                       i / (L->N * (L->ob > 8 ? 2 : 1)));
            bad++;
        }
    free(got);
    return bad;
}

static void test_regs(void)
{
    uint32_t v, s;
    printf("[regs] identification and registers\n");
    v = rd(R_VERSION);
    printf("  VERSION = 0x%08x\n", v);
    check(v == 0x46500001u || v == 0x46500002u, "VERSION is 'FP' 0001/0002", v, 0x46500002u);
    check(rd(0x60) == 0xDEADBEEFu, "unmapped register reads 0xDEADBEEF", rd(0x60), 0xDEADBEEFu);

    uint32_t h0 = rd(R_HEARTBEAT);
    double t0 = now_ms();
    usleep(20000);
    uint32_t h1 = rd(R_HEARTBEAT);
    double dt = now_ms() - t0;
    printf("  HEARTBEAT: %u counts in %.2f ms -> accelerator clock about %.1f MHz\n", h1 - h0, dt,
           (uint32_t)(h1 - h0) / (dt * 1000.0));
    check(h1 != h0, "HEARTBEAT advances", h1, h0 + 1);

    wr(R_SCRATCH, 0xA5A55A5Au);
    check(rd(R_SCRATCH) == 0xA5A55A5Au, "SCRATCH write/read", rd(R_SCRATCH), 0xA5A55A5Au);
    wr(R_VERSION, 0xFFFFFFFFu);
    check(rd(R_VERSION) == v, "VERSION is read-only", rd(R_VERSION), v);

    /* bad parameters are rejected by the real engine (error code 2) */
    wr(R_CTRL, CTRL_RESET_SOFT);
    wr(R_DIM_M, 16); wr(R_DIM_N, 16); wr(R_DIM_K, 7); wr(R_OP, 0x0200);
    wr(R_CTRL, CTRL_START);
    usleep(1000);
    s = rd(R_STATUS);
    check(s == ST_ERROR, "odd K -> STATUS is ERROR", s, ST_ERROR);
    check(rd(R_ERR_CODE) == 2, "ERR_CODE is 2 (bad parameters)", rd(R_ERR_CODE), 2);
    wr(R_CTRL, CTRL_CLEAR_ERR);
    check(rd(R_STATUS) == 0, "idle after CLEAR_ERR", rd(R_STATUS), 0);
    wr(R_DIM_M, 0); wr(R_DIM_N, 16); wr(R_DIM_K, 8);
    wr(R_CTRL, CTRL_START);
    usleep(1000);
    check(rd(R_STATUS) == ST_ERROR && rd(R_ERR_CODE) == 2, "M = 0 -> error 2", rd(R_STATUS), ST_ERROR);
    wr(R_CTRL, CTRL_CLEAR_ERR);
}

static void test_mem(void)
{
    printf("[mem] reserved DDR3 region at 0x%08x\n", REGION_PHYS);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < 0x10000 / 4; i++)
        region[i] = 0x9E3779B9u * (i + 1);
    __sync_synchronize();
    for (uint32_t i = 0; i < 0x10000 / 4; i++)
        if (region[i] != 0x9E3779B9u * (i + 1))
            bad++;
    check(bad == 0, "64 KiB write/read pattern in the region", bad, 0);
}

static int run_layer(struct layer *L, int verbose, double *ms_out, uint32_t *cyc_out)
{
    upload(L);
    fill_dev(O_OFF, 0xCC, L->ylen);
    program(L);
    wr(R_CTRL, CTRL_RESET_SOFT);          /* clean state, clears counters */
    program(L);
    double t0 = now_ms();
    wr(R_CTRL, CTRL_START);
    double el = wait_idle(20000);
    double ms = now_ms() - t0;
    uint32_t s = rd(R_STATUS);
    if (el < 0 || s != ST_DONE) {
        printf("    operation did not finish cleanly: STATUS=0x%x ERR_CODE=%u\n", s, rd(R_ERR_CODE));
        return -1;
    }
    uint32_t cyc = rd(R_CYCLES_LO);
    int bad = compare_output(L, verbose);
    uint32_t brd = rd(R_BYTES_RD), bwr = rd(R_BYTES_WR);
    double macs = (double)L->M * L->N * L->K;
    double mhz = cyc / (ms * 1000.0);
    printf("  %-22s M=%d N=%d K=%d in=%d out=%d spp=%d: %u cycles, %.2f ms, read %u B, written %u B, "
           "%.1f MAC/cycle (true MACs), polled clock ~%.0f MHz\n",
           L->name, L->M, L->N, L->K, L->ib, L->ob, L->spp, cyc, ms, brd, bwr, macs / cyc, mhz);
    if (ms_out) *ms_out = ms;
    if (cyc_out) *cyc_out = cyc;
    return bad;
}

static void test_layers(int n, char **dirs)
{
    printf("[layer] real layers of the e1_r7 model through the accelerator\n");
    for (int i = 0; i < n; i++) {
        struct layer L;
        if (load_layer(dirs[i], &L)) {
            fails++;
            checks++;
            continue;
        }
        int bad = run_layer(&L, 1, NULL, NULL);
        char what[256];
        snprintf(what, sizeof what, "%s output is bit-exact with the model", L.name);
        check(bad == 0, what, (uint32_t)bad, 0);
        free_layer(&L);
    }
}

static void test_faults(const char *dir, int bad_addr)
{
    struct layer L;
    printf("[faults] error injection on %s\n", dir);
    if (load_layer(dir, &L)) {
        fails++; checks++;
        return;
    }
    upload(&L);
    program(&L);
    wr(R_CTRL, CTRL_RESET_SOFT);
    program(&L);

    /* START while busy: error 1, the running operation still completes */
    fill_dev(O_OFF, 0xCC, L.ylen);
    wr(R_CTRL, CTRL_START);
    wr(R_CTRL, CTRL_START);
    uint32_t s = rd(R_STATUS);
    check((s & (ST_BUSY | ST_ERROR)) == (ST_BUSY | ST_ERROR), "START while BUSY sets ERROR", s, ST_BUSY | ST_ERROR);
    check(rd(R_ERR_CODE) == 1, "ERR_CODE is 1", rd(R_ERR_CODE), 1);
    wr(R_CTRL, CTRL_CLEAR_ERR);
    check(wait_idle(20000) >= 0, "first operation finishes", 0, 1);
    check(compare_output(&L, 1) == 0, "its output is still correct", 0, 0);

    /* soft reset in the middle of an operation, then run again */
    fill_dev(O_OFF, 0xCC, L.ylen);
    wr(R_CTRL, CTRL_START);
    usleep(5);
    wr(R_CTRL, CTRL_RESET_SOFT);
    usleep(1000);
    check(rd(R_STATUS) == 0, "idle after RESET_SOFT in the middle of an operation", rd(R_STATUS), 0);
    program(&L);
    fill_dev(O_OFF, 0xCC, L.ylen);
    wr(R_CTRL, CTRL_START);
    check(wait_idle(20000) >= 0, "operation after the reset finishes", 0, 1);
    check(rd(R_STATUS) == ST_DONE && compare_output(&L, 1) == 0, "and is bit-exact", rd(R_STATUS), ST_DONE);

    if (bad_addr) {
        /* point the activations outside the SDRAM: the read DMA must time out (error 3) */
        wr(R_CTRL, CTRL_RESET_SOFT);
        program(&L);
        wr(R_ADDR_A, 0x50000000u);
        wr(R_CTRL, CTRL_START);
        double el = wait_idle(2000);
        s = rd(R_STATUS);
        printf("  bad address: STATUS=0x%x ERR_CODE=%u after %.2f ms\n", s, rd(R_ERR_CODE), el);
        check(el >= 0 && s == ST_ERROR, "bad address ends in ERROR (no hang)", s, ST_ERROR);
        check(rd(R_ERR_CODE) == 3 || rd(R_ERR_CODE) == 6, "ERR_CODE is 3 (read timeout) or 6 (watchdog)", rd(R_ERR_CODE), 3);
        wr(R_CTRL, CTRL_CLEAR_ERR);
        program(&L);
        fill_dev(O_OFF, 0xCC, L.ylen);
        wr(R_CTRL, CTRL_START);
        check(wait_idle(20000) >= 0 && rd(R_STATUS) == ST_DONE && compare_output(&L, 1) == 0,
              "a correct operation works right after the error", rd(R_STATUS), ST_DONE);
    }
    free_layer(&L);
}

/* ---- concurrent: engine (scores S = Q K^T, RAW32) and softmax unit at the same time ----
 * Everything lives in the window 0x30000000 + [0x6000000, 0x8000000). Fixed softmax constants: scores are
 * random in [-30000, 30000), so idx = (d * M0) >> SH with M0 / 2^SH = 1/64 spans the 1025-entry exp table. */
#define C_OFF   0x6000000u
#define C_Q     (C_OFF + 0x000000u)       /* Q, 128 KiB slot */
#define C_K     (C_OFF + 0x020000u)       /* K^T tiles, 128 KiB slot */
/* slot sizes cover the largest case (1025 rows): S = 1025 x 1040 x 4 = 0x411040 B, P = 0x209040 B.
 * The slots must not overlap: an earlier layout put the S slots on top of the P slots, so the concurrent
 * runs overwrote each other's outputs and the comparison reported DIFFER. */
#define C_SIN   (C_OFF + 0x040000u)       /* synthetic scores for the softmax unit, one S slot */
#define C_ES_SZ 0x420000u
#define C_P_SZ  0x210000u
#define C_MC_SZ 0x400000u
#define C_ES    (C_SIN + C_ES_SZ)         /* engine output S: 3 slots (alone, conc, conc+memcpy) */
#define C_P     (C_ES + 3 * C_ES_SZ)      /* softmax output P: 3 slots */
#define C_MC    (C_P + 3 * C_P_SZ)        /* memcpy destination */
_Static_assert(C_MC + C_MC_SZ <= C_OFF + 0x2000000u, "concurrent test buffers leave the window");
#define C_SM_M0 (1u << 20)
#define C_SM_SH 26

static uint32_t rng_state = 12345u;
static uint32_t rng(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static volatile int mc_stop;
static struct { uint8_t *dst, *src; double bytes, t; } mc;

static void *memcpy_thread(void *arg)
{
    (void)arg;
    double t0 = now_ms();
    mc.bytes = 0;
    do {
        memcpy(mc.dst, mc.src, C_MC_SZ);
        mc.bytes += C_MC_SZ;
    } while (!mc_stop);
    mc.t = now_ms() - t0;
    return NULL;
}

/* starts the selected units back to back, waits for all of them; optionally with the memcpy thread running */
static int conc_run(accel_hw_t *hw, const accel_op_t *eo, const accel_sm_op_t *so, int do_e, int do_s,
                    int with_mc, accel_stat_t *es, accel_stat_t *ss, double *mcmbs)
{
    pthread_t th;
    if (with_mc) {
        mc_stop = 0;
        if (pthread_create(&th, NULL, memcpy_thread, NULL) != 0)
            return -1;
    }
    int rc = 0;
    if (do_e && accel_hw_start(hw, eo) != 0)
        rc = -1;
    if (rc == 0 && do_s && accel_hw_sm_start(hw, so) != 0)
        rc = -1;
    if (do_e && accel_hw_wait(hw, eo, es, 60.0) != 0)
        rc = -1;
    if (do_s && accel_hw_sm_wait(hw, so, ss, 60.0) != 0)
        rc = -1;
    if (with_mc) {
        mc_stop = 1;
        pthread_join(th, NULL);
        *mcmbs = mc.bytes / 1048576.0 / (mc.t / 1000.0);
    }
    return rc;
}

static int test_concurrent(int rows, int reps)
{
    const int hd = 64;
    const int NK = ((rows + 15) / 16) * 16, KP = (rows + 1) & ~1;
    const int spp_s = (NK / 16) < 255 ? NK / 16 : 255;
    accel_hw_t hw;
    if (rows < 1 || rows > 1025) {
        fprintf(stderr, "rows must be in 1..1025\n");
        return 2;
    }
    accel_hw_set_cached(0);     /* this tool copies into the region itself, without cache maintenance */
    if (accel_hw_open(&hw, 0x8000000u) != 0)
        return 2;
    if (!hw.has_softmax) {
        fprintf(stderr, "this bitstream has no softmax unit\n");
        return 2;
    }

    int8_t *q = malloc((size_t)rows * hd), *k = malloc((size_t)rows * hd);
    int32_t *sin_ = malloc((size_t)rows * NK * 4);
    uint8_t *stage = malloc(C_ES_SZ), *a = malloc(C_ES_SZ), *b = malloc(C_ES_SZ);
    mc.src = malloc(C_MC_SZ);
    mc.dst = hw.region + C_MC;
    if (!q || !k || !sin_ || !stage || !a || !b || !mc.src) {
        perror("malloc");
        return 2;
    }
    for (int i = 0; i < rows * hd; i++) {
        q[i] = (int8_t)(rng() % 255 - 127);
        k[i] = (int8_t)(rng() % 255 - 127);
    }
    for (int i = 0; i < rows * NK; i++)
        sin_[i] = (int32_t)(rng() % 60000) - 30000;
    for (size_t i = 0; i < C_MC_SZ; i++)
        mc.src[i] = (uint8_t)rng();

    size_t n = fpga_acts_bytes(rows, hd, 1);
    fpga_pack_acts8_strided(q, rows, hd, hd, stage);
    memcpy(hw.region + C_Q, stage, n);
    n = fpga_kt_bytes(rows, hd);
    fpga_pack_kt(k, rows, hd, hd, stage);
    memcpy(hw.region + C_K, stage, n);
    memcpy(hw.region + C_SIN, sin_, (size_t)rows * NK * 4);

    const size_t es_bytes = (size_t)rows * NK * 4, p_bytes = fpga_acts_bytes(rows, KP, 2);
    accel_op_t eo[3];
    accel_sm_op_t so[3];
    for (int i = 0; i < 3; i++) {
        eo[i] = (accel_op_t){.M = (uint32_t)rows, .N = (uint32_t)NK, .K = hd, .in_bits = 8, .out_bits = 32,
                             .spp = spp_s, .a_phys = REGION_PHYS + C_Q, .w_phys = REGION_PHYS + C_K,
                             .out_phys = REGION_PHYS + C_ES + i * C_ES_SZ, .raw32 = 1,
                             .stride_out = (uint32_t)NK * 4};
        so[i] = (accel_sm_op_t){.rows = (uint32_t)rows, .cols = (uint32_t)rows, .kp = (uint32_t)KP,
                                .s_phys = REGION_PHYS + C_SIN, .s_stride = (uint32_t)NK * 4,
                                .p_phys = REGION_PHYS + C_P + i * C_P_SZ, .m0 = C_SM_M0, .sh = C_SM_SH};
    }
    accel_stat_t e_alone = {0}, s_alone = {0}, e_c = {0}, s_c = {0}, e_d = {0}, s_d = {0};
    double mb_d = 0;
    int rc = 0;
    /* output slots start with a known pattern so an unwritten byte cannot match by luck */
    memset(stage, 0xA5, C_ES_SZ);
    for (int i = 0; i < 3; i++) {
        memcpy(hw.region + C_ES + i * C_ES_SZ, stage, es_bytes);
        memcpy(hw.region + C_P + i * C_P_SZ, stage, p_bytes);
    }
    /* each phase runs `reps` times; a failed run (e.g. ERR_CODE 5, engine output FIFO overflow) is counted and
     * the test goes on (the driver resets the unit). Times and outputs are those of the last good run. */
    int fail[4] = {0, 0, 0, 0};
    double ta = 0, tb = 0, tc = 0, td = 0, t0;
    for (int r = 0; r < reps; r++) {
        t0 = now_ms();
        if (conc_run(&hw, &eo[0], &so[0], 1, 0, 0, &e_alone, NULL, NULL)) fail[0]++; else ta = now_ms() - t0; /* (a) S alone */
        t0 = now_ms();
        if (conc_run(&hw, &eo[0], &so[0], 0, 1, 0, NULL, &s_alone, NULL)) fail[1]++; else tb = now_ms() - t0; /* (b) SM alone */
        t0 = now_ms();
        if (conc_run(&hw, &eo[1], &so[1], 1, 1, 0, &e_c, &s_c, NULL)) fail[2]++; else tc = now_ms() - t0;     /* (c) both */
        t0 = now_ms();
        if (conc_run(&hw, &eo[2], &so[2], 1, 1, 1, &e_d, &s_d, &mb_d)) fail[3]++; else td = now_ms() - t0;    /* (d) +memcpy */
    }
    printf("RESULT conc_fail: reps %d; failed runs: S alone %d, SM alone %d, conc %d, conc+memcpy %d\n", reps, fail[0],
           fail[1], fail[2], fail[3]);
    if (fail[0] == reps || fail[1] == reps || fail[2] == reps || fail[3] == reps) {
        fprintf(stderr, "a phase never succeeded: no comparison\n");
        accel_hw_close(&hw);
        return 1;
    }
    (void)rc;

    /* (a) and (b) wrote slot 0; (c) slot 1; (d) slot 2 */
    int same_c = 1, same_d = 1;
    memcpy(a, hw.region + C_ES, es_bytes);
    memcpy(b, hw.region + C_ES + 1 * C_ES_SZ, es_bytes);
    same_c &= !memcmp(a, b, es_bytes);
    memcpy(b, hw.region + C_ES + 2 * C_ES_SZ, es_bytes);
    same_d &= !memcmp(a, b, es_bytes);
    memcpy(a, hw.region + C_P, p_bytes);
    memcpy(b, hw.region + C_P + 1 * C_P_SZ, p_bytes);
    same_c &= !memcmp(a, b, p_bytes);
    memcpy(b, hw.region + C_P + 2 * C_P_SZ, p_bytes);
    same_d &= !memcmp(a, b, p_bytes);

    const double mb = 1048576.0;
    printf("  rows %d: S alone %.2f ms rd %u wr %u; SM alone %.2f ms rd %u wr %u; both %.2f ms; both+memcpy %.2f ms\n",
           rows, ta, e_alone.bytes_rd, e_alone.bytes_wr, tb, s_alone.bytes_rd, s_alone.bytes_wr, tc, td);
    printf("RESULT conc: S %llu alone / %llu conc / %llu conc+memcpy; SM %llu alone / %llu conc / %llu conc+memcpy; "
           "MB/s engine+SM alone %.1f, conc %.1f, conc+memcpy %.1f, memcpy thread %.1f; outputs %s\n",
           (unsigned long long)e_alone.cycles, (unsigned long long)e_c.cycles, (unsigned long long)e_d.cycles,
           (unsigned long long)s_alone.cycles, (unsigned long long)s_c.cycles, (unsigned long long)s_d.cycles,
           (e_alone.bytes_rd + e_alone.bytes_wr + s_alone.bytes_rd + s_alone.bytes_wr) / mb / ((ta + tb) / 1000.0),
           (e_c.bytes_rd + e_c.bytes_wr + s_c.bytes_rd + s_c.bytes_wr) / mb / (tc / 1000.0),
           (e_d.bytes_rd + e_d.bytes_wr + s_d.bytes_rd + s_d.bytes_wr) / mb / (td / 1000.0), mb_d,
           same_c && same_d ? "identical" : "DIFFER");
    if (!same_c || !same_d)
        printf("  DIFFER: conc %s, conc+memcpy %s\n", same_c ? "ok" : "differs", same_d ? "ok" : "differs");
    accel_hw_close(&hw);
    return (same_c && same_d) ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s regs | mem | layer DIR... | faults DIR [--bad-addr] | all DIR... | concurrent [rows]\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "concurrent"))
        return test_concurrent(argc > 2 ? atoi(argv[2]) : 1025, argc > 3 && atoi(argv[3]) > 0 ? atoi(argv[3]) : 1);
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem (run as root)");
        return 2;
    }
    csr = mmap(NULL, LW_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, LW_BASE);
    if (csr == MAP_FAILED) {
        perror("mmap csr");
        return 2;
    }
    const char *cmd = argv[1];
    int need_region = strcmp(cmd, "regs") != 0;
    if (need_region) {
        region = mmap(NULL, REGION_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, REGION_PHYS);
        if (region == MAP_FAILED) {
            perror("mmap reserved region (is Linux limited with mem=768M?)");
            return 2;
        }
    }

    if (!strcmp(cmd, "regs")) {
        test_regs();
    } else if (!strcmp(cmd, "mem")) {
        test_mem();
    } else if (!strcmp(cmd, "layer")) {
        test_layers(argc - 2, argv + 2);
    } else if (!strcmp(cmd, "faults")) {
        if (argc < 3) { fprintf(stderr, "faults needs a layer directory\n"); return 2; }
        test_faults(argv[2], argc > 3 && !strcmp(argv[3], "--bad-addr"));
    } else if (!strcmp(cmd, "all")) {
        if (argc < 3) { fprintf(stderr, "all needs layer directories\n"); return 2; }
        test_regs();
        test_mem();
        test_layers(argc - 2, argv + 2);
        test_faults(argv[2], 0);
    } else {
        fprintf(stderr, "unknown command %s\n", cmd);
        return 2;
    }
    printf("\n%s: %d of %d checks passed\n", fails ? "FAILED" : "ALL PASSED", checks - fails, checks);
    return fails ? 1 : 0;
}
