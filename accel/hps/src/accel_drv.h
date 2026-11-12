#ifndef FPGATE_ACCEL_DRV_H
#define FPGATE_ACCEL_DRV_H
/*
 * Access to the accelerator from Linux on the HPS: register file through the lightweight bridge
 * (/dev/mem, 0xFF200000) and the reserved DDR3 region (0x30000000, uncached through /dev/mem).
 * Same register sequence as accel/tools/board/accel_test.c, which validated it on the board.
 *
 * Preconditions (accel/tools/board/README.md): the FPGA is configured by U-Boot (MSEL = 01010,
 * soc_system.rbf on the SD) and the region is outside Linux RAM. Reading the registers with the
 * FPGA unconfigured hangs the board, so accel_hw_open() refuses to touch them unless
 * /sys/class/fpga_manager/fpga0/state says "operating".
 */
#include <stddef.h>
#include <stdint.h>

#include "pkg.h"

#define ACCEL_REGION_PHYS 0x30000000u

typedef struct accel_hw {
    int fd;
    volatile uint8_t *csr;
    uint8_t *region;
    size_t region_len;
    double mhz;                 /* accelerator clock measured with HEARTBEAT at open */
    int has_softmax;            /* the bitstream has the attention softmax unit (registers at 0x100) */
    struct accel_sim *sim;      /* software engine (accel_hw_open_sim); NULL on the real hardware */
    double t_start;             /* wall time of the last accel_hw_start */
    int in_flight;              /* an operation was started and not yet collected with accel_hw_wait */
    double sm_t_start;          /* wall time of the last accel_hw_sm_start */
    int sm_in_flight;           /* a softmax was started and not yet collected with accel_hw_sm_wait */
} accel_hw_t;

typedef struct {
    uint32_t M, N, K;
    int in_bits, out_bits, spp;
    uint32_t a_phys, w_phys, out_phys;
    int in_unsigned;            /* 16-bit input is UQ (attention probabilities); default: signed */
    int raw32;                  /* output the INT32 accumulators unchanged (attention scores) */
    uint32_t stride_out;        /* bytes between output rows; 0 = N * element size */
} accel_op_t;

typedef struct {
    uint64_t cycles;
    uint32_t bytes_rd, bytes_wr;
    double wall_s;
} accel_stat_t;

/* returns 0 on success; messages go to stderr */
int accel_hw_open(accel_hw_t *h, size_t region_len);
void accel_hw_close(accel_hw_t *h);

typedef struct {
    uint32_t rows, cols, kp;
    uint32_t s_phys, s_stride, p_phys;
    uint32_t m0;
    int sh;
    const void *sim_ctx;        /* ignored by the hardware; the software model reads an accel_sm_sim_ctx_t */
} accel_sm_op_t;

/* what the software softmax unit needs to run softmax_row_int (the hardware has the tables built in) */
typedef struct {
    const pkg_t *pkg;
    const block_t *block;
} accel_sm_sim_ctx_t;

/* softmax unit: scores (RAW32 output of the engine) -> probabilities in the activation format. 0 on DONE. */
int accel_hw_run_sm(accel_hw_t *h, const accel_sm_op_t *op, accel_stat_t *st, double timeout_s);

/* Asynchronous softmax: sm_start fails if a softmax is already in flight (independent of the linear engine's
 * in_flight); sm_wait collects it (op = the same descriptor); sm_busy is non-blocking. */
int accel_hw_sm_start(accel_hw_t *h, const accel_sm_op_t *op);
int accel_hw_sm_busy(const accel_hw_t *h);
int accel_hw_sm_wait(accel_hw_t *h, const accel_sm_op_t *op, accel_stat_t *st, double timeout_s);

/* Software model of the linear engine, for testing on a PC (no FPGA, no /dev/mem): the region is
 * ordinary memory and an operation runs on a worker thread that decodes the same packed formats and
 * writes the output progressively after a delay proportional to its size, so ordering mistakes
 * (reading or overwriting a buffer the "engine" is still using) show up as wrong results.
 * Supports RAW32, unsigned 16-bit input, stride_out, and a softmax unit on a second thread (accel_sim_sm_*)
 * that can be in flight at the same time as the engine. */
int accel_hw_open_sim(accel_hw_t *h, size_t region_len);

/* Starts one operation without waiting: the engine works while the caller does something else.
 * Exactly one operation may be in flight; accel_hw_wait must be called before the next start. */
int accel_hw_start(accel_hw_t *h, const accel_op_t *op);
/* Waits for the operation started with accel_hw_start (op = the same descriptor). Returns 0 on DONE,
 * -1 on error or timeout (the engine is reset so the next operation starts clean). */
int accel_hw_wait(accel_hw_t *h, const accel_op_t *op, accel_stat_t *st, double timeout_s);

/* Non-blocking: 1 while the operation started with accel_hw_start is still running, 0 when it has finished
 * (accel_hw_wait still has to be called to collect the result). */
int accel_hw_busy(const accel_hw_t *h);

/* internal: software engine (accel_sim.c) */
int accel_sim_busy(const accel_hw_t *h);
int accel_sim_start(accel_hw_t *h, const accel_op_t *op);
int accel_sim_wait(accel_hw_t *h, accel_stat_t *st);
void accel_sim_free(accel_hw_t *h);
/* software softmax unit: one operation in flight; sm_op.sim_ctx must point to an accel_sm_sim_ctx_t */
int accel_sim_sm_start(accel_hw_t *h, const accel_sm_op_t *op);
int accel_sim_sm_wait(accel_hw_t *h, accel_stat_t *st);
int accel_sim_sm_busy(const accel_hw_t *h);

/* start + wait */
int accel_hw_run(accel_hw_t *h, const accel_op_t *op, accel_stat_t *st, double timeout_s);

#endif
