#define _GNU_SOURCE
#include "accel_drv.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "../../tools/kmod/fpgate_mem.h"

static int g_cached_mode = -1;      /* accel_hw_set_cached */

#define LW_BASE 0xFF200000u
#define LW_LEN 0x1000u

#define R_VERSION 0x00
#define R_CTRL 0x04
#define R_STATUS 0x08
#define R_ERR_CODE 0x0C
#define R_OP 0x10
#define R_DIM_M 0x14
#define R_DIM_N 0x18
#define R_DIM_K 0x1C
#define R_ADDR_A 0x20
#define R_ADDR_W 0x24
#define R_ADDR_OUT 0x28
#define R_CYCLES_LO 0x38
#define R_CYCLES_HI 0x3C
#define R_BYTES_RD 0x40
#define R_BYTES_WR 0x44
#define R_HEARTBEAT 0x48
#define R_STRIDE_OUT 0x4C
/* output layout writer, RoPE and fused SwiGLU (csr.sv); 0x98 FEATURES reads 0xDEADBEEF on a bitstream without them */
#define R_OUT_TSTRIDE 0x80
#define R_HEAD_TILES 0x84
#define R_HEAD_STRIDE 0x88
#define R_ROW0 0x8C
#define R_ROPE_CFG 0x90
#define R_ROPE_TAB 0x94
#define R_FEATURES 0x98
#define R_SW_M0 0xA0
#define R_SW_SHIFT 0xA4
#define R_SW_TAB 0xA8

/* softmax unit register block (csr_sm.sv) */
#define SM_BASE 0x100
#define SM_VERSION (SM_BASE + 0x00)
#define SM_CTRL (SM_BASE + 0x04)
#define SM_STATUS (SM_BASE + 0x08)
#define SM_ERR_CODE (SM_BASE + 0x0C)
#define SM_ROWS (SM_BASE + 0x10)
#define SM_COLS (SM_BASE + 0x14)
#define SM_KP (SM_BASE + 0x18)
#define SM_ADDR_S (SM_BASE + 0x1C)
#define SM_STRIDE_S (SM_BASE + 0x20)
#define SM_ADDR_P (SM_BASE + 0x24)
#define SM_M0 (SM_BASE + 0x28)
#define SM_SH (SM_BASE + 0x2C)
#define SM_CYCLES (SM_BASE + 0x30)
#define SM_BYTES_RD (SM_BASE + 0x34)
#define SM_BYTES_WR (SM_BASE + 0x38)

#define CTRL_START (1u << 0)
#define CTRL_RESET_SOFT (1u << 1)
#define CTRL_CLEAR_ERR (1u << 3)
#define ST_BUSY 1u
#define ST_DONE 2u
#define ST_ERROR 4u

static inline uint32_t rd(const accel_hw_t *h, unsigned off) { return *(volatile uint32_t *)(h->csr + off); }
static inline void wr(const accel_hw_t *h, unsigned off, uint32_t v) { *(volatile uint32_t *)(h->csr + off) = v; }

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int fpga_operating(void)
{
    char buf[64] = {0};
    FILE *f = fopen("/sys/class/fpga_manager/fpga0/state", "r");
    if (!f)
        return 0;
    if (!fgets(buf, sizeof buf, f))
        buf[0] = 0;
    fclose(f);
    return strncmp(buf, "operating", 9) == 0;
}

int accel_hw_open(accel_hw_t *h, size_t region_len)
{
    memset(h, 0, sizeof *h);
    h->fd = -1;
    if (!fpga_operating()) {
        fprintf(stderr, "accel: the FPGA is not configured (fpga_manager state is not 'operating'); "
                        "touching its registers would hang the board. See accel/tools/board/README.md\n");
        return -1;
    }
    h->mem_fd = -1;
    h->fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (h->fd < 0) {
        perror("accel: open /dev/mem (run as root)");
        return -1;
    }
    void *c = mmap(NULL, LW_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, LW_BASE);
    if (c == MAP_FAILED) {
        perror("accel: mmap registers");
        return -1;
    }
    h->csr = c;
    void *r = MAP_FAILED;
    if (g_cached_mode != 0) {
        h->mem_fd = open(FPGATE_MEM_PATH, O_RDWR);
        if (h->mem_fd >= 0) {
            r = mmap(NULL, region_len, PROT_READ | PROT_WRITE, MAP_SHARED, h->mem_fd, 0);
            if (r == MAP_FAILED) {
                perror("accel: mmap " FPGATE_MEM_PATH);
                return -1;
            }
            h->cached = 1;
        } else if (g_cached_mode > 0) {
            perror("accel: open " FPGATE_MEM_PATH " (insmod fpgate_mem.ko)");
            return -1;
        }
    }
    if (r == MAP_FAILED) {
        r = mmap(NULL, region_len, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, ACCEL_REGION_PHYS);
        if (r == MAP_FAILED) {
            perror("accel: mmap DDR3 region");
            return -1;
        }
    }
    h->region = r;
    h->region_len = region_len;
    uint32_t v = rd(h, R_VERSION);
    if ((v >> 16) != 0x4650u || ((v & 0xFFFFu) != 1u && (v & 0xFFFFu) != 2u)) {
        fprintf(stderr, "accel: unexpected VERSION 0x%08x (expected 0x46500001 or 0x46500002)\n", v);
        return -1;
    }
    h->hw_version = (int)(v & 0xFFFFu);
    {
        const uint32_t f = rd(h, R_FEATURES);
        h->features = (f & ~7u) == 0 ? f : 0;       /* 0xDEADBEEF: the register does not exist */
    }
    wr(h, R_CTRL, CTRL_RESET_SOFT);
    wr(h, R_CTRL, CTRL_CLEAR_ERR);
    {
        uint32_t h0 = rd(h, R_HEARTBEAT);
        double t0 = now_s();
        usleep(20000);
        uint32_t h1 = rd(h, R_HEARTBEAT);
        h->mhz = (uint32_t)(h1 - h0) / ((now_s() - t0) * 1e6);
    }
    {
        uint32_t sv = rd(h, SM_VERSION);
        h->has_softmax = (sv == 0x534D0001u || sv == 0x534D0002u);
    }
    if (h->has_softmax) {
        wr(h, SM_CTRL, CTRL_RESET_SOFT);
        wr(h, SM_CTRL, CTRL_CLEAR_ERR);
    }
    return 0;
}

void accel_hw_close(accel_hw_t *h)
{
    if (h->sim) {
        accel_sim_free(h);
        free(h->region);
        memset(h, 0, sizeof *h);
        h->fd = -1;
        return;
    }
    if (h->region)
        munmap(h->region, h->region_len);
    if (h->csr)
        munmap((void *)h->csr, LW_LEN);
    if (h->mem_fd >= 0)
        close(h->mem_fd);
    if (h->fd >= 0)
        close(h->fd);
    memset(h, 0, sizeof *h);
    h->fd = -1;
    h->mem_fd = -1;
}

void accel_hw_set_cached(int mode)
{
    g_cached_mode = mode;
}

int accel_hw_is_cached(const accel_hw_t *h)
{
    return h->cached;
}

static void cache_op(const accel_hw_t *h, unsigned long req, size_t off, size_t len, const char *what)
{
    if (!h->cached || h->sim || len == 0)
        return;
    if (off > h->region_len || len > h->region_len - off) {
        fprintf(stderr, "accel: cache %s outside the region (off %zu len %zu)\n", what, off, len);
        return;
    }
    struct fpgate_mem_range r = {.offset = (uint32_t)off, .len = (uint32_t)len};
    if (ioctl(h->mem_fd, req, &r) != 0)
        perror(what);
}

void accel_hw_clean(const accel_hw_t *h, size_t off, size_t len)
{
    cache_op(h, FPGATE_MEM_CLEAN, off, len, "accel: cache clean");
}

void accel_hw_inval(const accel_hw_t *h, size_t off, size_t len)
{
    cache_op(h, FPGATE_MEM_INVAL, off, len, "accel: cache invalidate");
}

/* SiLU table of the fused SwiGLU: index = gate + 127. Only while the engine is idle. */
static void load_silu(accel_hw_t *h, const int16_t *tab)
{
    if (h->sim)
        accel_sim_load_silu(h, tab);
    else
        for (uint32_t i = 0; i < ACCEL_SILU_ENTRIES; i++)
            wr(h, R_SW_TAB, (i << 16) | (uint16_t)tab[i]);
    h->silu_loaded = tab;
}

/* the hardware keeps the RoPE tables of a head of 64 as four RAMs of 512 x 16 bit (cos / sin x pair lane), indexed
 * {half = d[5], yx (y when half = 0, x when half = 1), pair in half}: tab[sin][half][yx][pair]. The class token's
 * row is not in the tables (token 0 is not rotated), the token t = 1 + 32 * y + x uses row t - 1. Returns -1 when the
 * real tables are not of that form. */
static int rope_separate(const int16_t *cos_tab, const int16_t *sin_tab, int16_t tab[2][2][32][16])
{
    for (int sn = 0; sn < 2; sn++) {
        const int16_t *t = sn ? sin_tab : cos_tab;
        for (int u = 0; u < 1024; u++) {
            const int y = u >> 5, x = u & 31;
            for (int d = 0; d < 64; d++) {
                const int16_t v = t[(size_t)u * 64 + d];
                const int src = d < 32 ? y * 32 : x;                 /* the y-only half, the x-only half */
                if (v != t[(size_t)src * 64 + d] || ((d & 1) && v != t[(size_t)u * 64 + d - 1]))
                    return -1;
                if (d < 32 ? x == 0 : y == 0)
                    tab[sn][d >> 5][d < 32 ? y : x][(d & 31) >> 1] = v;
            }
        }
    }
    return 0;
}

int accel_hw_load_rope(accel_hw_t *h, const int16_t *cos_tab, const int16_t *sin_tab, int frac, int pre_shift)
{
    static int16_t tab[2][2][32][16];
    if (!(h->features & ACCEL_FEAT_ROPE)) {
        fprintf(stderr, "accel: this bitstream has no RoPE epilogue\n");
        return -1;
    }
    if (frac + pre_shift < 1 || frac + pre_shift > 63 || pre_shift < 1 || pre_shift > 63 || rope_separate(cos_tab, sin_tab, tab) != 0) {
        fprintf(stderr, "accel: the RoPE tables are not separable (or the shifts are out of range)\n");
        return -1;
    }
    h->rope_cfg = ((uint32_t)pre_shift << 8) | (uint32_t)(frac + pre_shift);
    if (h->sim) {
        accel_sim_load_rope(h, tab);
        return 0;
    }
    for (uint32_t sn = 0; sn < 2; sn++)
        for (uint32_t half = 0; half < 2; half++)
            for (uint32_t yx = 0; yx < 32; yx++)
                for (uint32_t pi = 0; pi < 16; pi++)
                    wr(h, R_ROPE_TAB, (sn << 26) | (((half << 9) | (yx << 4) | pi) << 16) | (uint16_t)tab[sn][half][yx][pi]);
    return 0;
}

void accel_hw_sim_set_features(accel_hw_t *h, uint32_t mask)
{
    h->features = mask & 7u;
}

int accel_hw_start(accel_hw_t *h, const accel_op_t *op)
{
    if (h->in_flight) {
        /* on the real engine a second START would reset it mid-operation and lose the first result */
        fprintf(stderr, "accel: start while an operation is still in flight (call accel_hw_wait first)\n");
        return -1;
    }
    if (h->sim) {
        if ((op->rope_hd && !(h->features & ACCEL_FEAT_ROPE)) || (op->swiglu && !(h->features & ACCEL_FEAT_SWIGLU))) {
            fprintf(stderr, "accel(sim): RoPE / SwiGLU requested but the engine does not announce it\n");
            return -1;
        }
        if (op->swiglu && h->silu_loaded != op->sw_tab && op->sw_tab)
            load_silu(h, op->sw_tab);
        if (accel_sim_start(h, op) != 0)
            return -1;
        h->in_flight = 1;
        return 0;
    }
    const int out16 = op->out_bits > 8 && !op->raw32;
    const uint32_t elem = op->raw32 ? 4u : (out16 ? 2u : 1u);
    if ((op->rope_hd && (!(h->features & ACCEL_FEAT_ROPE) || out16 || op->raw32 || op->swiglu)) ||
        (op->swiglu && (!(h->features & ACCEL_FEAT_SWIGLU) || out16 || op->raw32 || !op->sw_tab || (op->N & 31) ||
                        !op->stride_out))) {
        fprintf(stderr, "accel: operation M=%u N=%u K=%u asks for a RoPE / SwiGLU mode this bitstream or descriptor "
                        "does not support (FEATURES 0x%x)\n", op->M, op->N, op->K, h->features);
        return -1;
    }
    wr(h, R_CTRL, CTRL_RESET_SOFT);          /* clean state, clears the counters */
    if (h->features & (ACCEL_FEAT_OUTFMT | ACCEL_FEAT_ROPE)) {
        /* these registers survive the soft reset and also shape row-major outputs (the head jump), so every
         * operation sets them, not only the ones that use them */
        wr(h, R_OUT_TSTRIDE, 0);
        wr(h, R_HEAD_TILES, op->rope_hd ? op->rope_hd / 16 : 0);
        wr(h, R_HEAD_STRIDE, op->rope_hd);
        wr(h, R_ROW0, op->row0);
        if (op->rope_hd)
            wr(h, R_ROPE_CFG, h->rope_cfg);
    }
    if (op->swiglu) {
        wr(h, R_SW_M0, op->sw_m0);
        wr(h, R_SW_SHIFT, op->sw_shift);
        if (h->silu_loaded != op->sw_tab)
            load_silu(h, op->sw_tab);
    }
    wr(h, R_ADDR_A, op->a_phys);
    wr(h, R_ADDR_W, op->w_phys);
    wr(h, R_ADDR_OUT, op->out_phys);
    wr(h, R_STRIDE_OUT, op->stride_out ? op->stride_out : op->N * elem);
    wr(h, R_DIM_M, op->M);
    wr(h, R_DIM_N, op->N);
    wr(h, R_DIM_K, op->K);
    wr(h, R_OP, ((uint32_t)op->spp << 8) | ((uint32_t)(op->swiglu != 0) << 7) | ((uint32_t)(op->rope_hd != 0) << 6) |
                    ((uint32_t)op->raw32 << 3) | ((uint32_t)out16 << 2) | ((uint32_t)(!op->in_unsigned) << 1) |
                    (op->in_bits > 8));
    __sync_synchronize();
    h->t_start = now_s();
    wr(h, R_CTRL, CTRL_START);
    (void)rd(h, R_STATUS);                   /* read back: the START write has reached the engine (BUSY is up) */
    h->in_flight = 1;
    return 0;
}

int accel_hw_busy(const accel_hw_t *h)
{
    if (h->sim)
        return accel_sim_busy(h);
    return (rd(h, R_STATUS) & ST_BUSY) != 0;
}

int accel_hw_wait(accel_hw_t *h, const accel_op_t *op, accel_stat_t *st, double timeout_s)
{
    if (!h->in_flight) {
        fprintf(stderr, "accel: wait without a started operation\n");
        return -1;
    }
    h->in_flight = 0;
    if (h->sim)
        return accel_sim_wait(h, st);
    const double t0 = h->t_start;
    uint32_t s;
    do {
        s = rd(h, R_STATUS);
        if (!(s & ST_BUSY))
            break;
    } while (now_s() - t0 < timeout_s);
    const double dt = now_s() - t0;
    s = rd(h, R_STATUS);
    if (st) {
        st->wall_s = dt;
        st->cycles = ((uint64_t)rd(h, R_CYCLES_HI) << 32) | rd(h, R_CYCLES_LO);
        st->bytes_rd = rd(h, R_BYTES_RD);
        st->bytes_wr = rd(h, R_BYTES_WR);
    }
    if (s != ST_DONE) {
        fprintf(stderr, "accel: operation M=%u N=%u K=%u did not finish cleanly: STATUS=0x%x ERR_CODE=%u\n", op->M,
                op->N, op->K, s, rd(h, R_ERR_CODE));
        wr(h, R_CTRL, CTRL_RESET_SOFT);
        wr(h, R_CTRL, CTRL_CLEAR_ERR);
        return -1;
    }
    return 0;
}

int accel_hw_run(accel_hw_t *h, const accel_op_t *op, accel_stat_t *st, double timeout_s)
{
    if (accel_hw_start(h, op) != 0)
        return -1;
    return accel_hw_wait(h, op, st, timeout_s);
}

int accel_hw_sm_start(accel_hw_t *h, const accel_sm_op_t *op)
{
    if (h->sm_in_flight) {
        fprintf(stderr, "accel: softmax start while a softmax is still in flight (call accel_hw_sm_wait first)\n");
        return -1;
    }
    if (h->sim) {
        if (accel_sim_sm_start(h, op) != 0)
            return -1;
        h->sm_in_flight = 1;
        return 0;
    }
    if (!h->has_softmax) {
        fprintf(stderr, "accel: this bitstream has no softmax unit\n");
        return -1;
    }
    wr(h, SM_CTRL, CTRL_RESET_SOFT);
    wr(h, SM_ROWS, op->rows);
    wr(h, SM_COLS, op->cols);
    wr(h, SM_KP, op->kp);
    wr(h, SM_ADDR_S, op->s_phys);
    wr(h, SM_STRIDE_S, op->s_stride);
    wr(h, SM_ADDR_P, op->p_phys);
    wr(h, SM_M0, op->m0);
    wr(h, SM_SH, (uint32_t)op->sh);
    __sync_synchronize();
    h->sm_t_start = now_s();
    wr(h, SM_CTRL, CTRL_START);
    h->sm_in_flight = 1;
    return 0;
}

int accel_hw_sm_busy(const accel_hw_t *h)
{
    if (h->sim)
        return accel_sim_sm_busy(h);
    return (rd(h, SM_STATUS) & ST_BUSY) != 0;
}

int accel_hw_sm_wait(accel_hw_t *h, const accel_sm_op_t *op, accel_stat_t *st, double timeout_s)
{
    if (!h->sm_in_flight) {
        fprintf(stderr, "accel: softmax wait without a started softmax\n");
        return -1;
    }
    h->sm_in_flight = 0;
    if (h->sim)
        return accel_sim_sm_wait(h, st);
    const double t0 = h->sm_t_start;
    uint32_t s;
    do {
        s = rd(h, SM_STATUS);
        if (!(s & ST_BUSY))
            break;
    } while (now_s() - t0 < timeout_s);
    const double dt = now_s() - t0;
    s = rd(h, SM_STATUS);
    if (st) {
        st->wall_s = dt;
        st->cycles = rd(h, SM_CYCLES);
        st->bytes_rd = rd(h, SM_BYTES_RD);
        st->bytes_wr = rd(h, SM_BYTES_WR);
    }
    if (s != ST_DONE) {
        fprintf(stderr, "accel: softmax rows=%u cols=%u did not finish cleanly: STATUS=0x%x ERR_CODE=%u\n", op->rows,
                op->cols, s, rd(h, SM_ERR_CODE));
        wr(h, SM_CTRL, CTRL_RESET_SOFT);
        wr(h, SM_CTRL, CTRL_CLEAR_ERR);
        return -1;
    }
    return 0;
}

int accel_hw_run_sm(accel_hw_t *h, const accel_sm_op_t *op, accel_stat_t *st, double timeout_s)
{
    if (accel_hw_sm_start(h, op) != 0)
        return -1;
    return accel_hw_sm_wait(h, op, st, timeout_s);
}
