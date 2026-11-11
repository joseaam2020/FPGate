/*
 * Board test of the FPGate accelerator (H4). Runs as root on the HPS.
 *
 *   accel_test regs                    registers, error codes, soft reset
 *   accel_test mem                     read/write test of the reserved DDR3 region
 *   accel_test layer DIR [DIR...]      run packed layers and compare with the model, bit by bit
 *   accel_test faults DIR [--bad-addr] error injection on a small layer; --bad-addr also points
 *                                      the DMA outside the SDRAM (expects a read timeout)
 *   accel_test all DIR [DIR...]        regs + mem + layers + faults (on the first DIR)
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
    check(v == 0x46500001u, "VERSION is 'FP' 0001", v, 0x46500001u);
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
    check(rd(R_VERSION) == 0x46500001u, "VERSION is read-only", rd(R_VERSION), 0x46500001u);

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

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s regs | mem | layer DIR... | faults DIR [--bad-addr] | all DIR...\n", argv[0]);
        return 2;
    }
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
