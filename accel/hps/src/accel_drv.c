#define _GNU_SOURCE
#include "accel_drv.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

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
    void *r = mmap(NULL, region_len, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, ACCEL_REGION_PHYS);
    if (r == MAP_FAILED) {
        perror("accel: mmap DDR3 region");
        return -1;
    }
    h->region = r;
    h->region_len = region_len;
    uint32_t v = rd(h, R_VERSION);
    if (v != 0x46500001u) {
        fprintf(stderr, "accel: unexpected VERSION 0x%08x (expected 0x46500001)\n", v);
        return -1;
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
    h->has_softmax = rd(h, SM_VERSION) == 0x534D0001u;
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
    if (h->fd >= 0)
        close(h->fd);
    memset(h, 0, sizeof *h);
    h->fd = -1;
}

int accel_hw_start(accel_hw_t *h, const accel_op_t *op)
{
    if (h->in_flight) {
        /* on the real engine a second START would reset it mid-operation and lose the first result */
        fprintf(stderr, "accel: start while an operation is still in flight (call accel_hw_wait first)\n");
        return -1;
    }
    if (h->sim) {
        if (accel_sim_start(h, op) != 0)
            return -1;
        h->in_flight = 1;
        return 0;
    }
    const int out16 = op->out_bits > 8 && !op->raw32;
    const uint32_t elem = op->raw32 ? 4u : (out16 ? 2u : 1u);
    wr(h, R_CTRL, CTRL_RESET_SOFT);          /* clean state, clears the counters */
    wr(h, R_ADDR_A, op->a_phys);
    wr(h, R_ADDR_W, op->w_phys);
    wr(h, R_ADDR_OUT, op->out_phys);
    wr(h, R_STRIDE_OUT, op->stride_out ? op->stride_out : op->N * elem);
    wr(h, R_DIM_M, op->M);
    wr(h, R_DIM_N, op->N);
    wr(h, R_DIM_K, op->K);
    wr(h, R_OP, ((uint32_t)op->spp << 8) | ((uint32_t)op->raw32 << 3) | ((uint32_t)out16 << 2) |
                    ((uint32_t)(!op->in_unsigned) << 1) | (op->in_bits > 8));
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
