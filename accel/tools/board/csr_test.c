/*
 * Board bring-up test for the accelerator registers (H0).
 * Maps the lightweight HPS-to-FPGA bridge at 0xFF200000 through /dev/mem and checks
 * VERSION, HEARTBEAT, read/write registers, byte enables, the START/BUSY/DONE flow,
 * the error paths and the soft reset. Run as root on the board.
 *
 * Build (static, musl):  see accel/tools/board/README.md
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define LW_BASE 0xFF200000u
#define MAP_LEN 0x1000u

#define R_VERSION   0x00
#define R_CTRL      0x04
#define R_STATUS    0x08
#define R_ERR_CODE  0x0C
#define R_OP        0x10
#define R_DIM_M     0x14
#define R_ADDR_A    0x20
#define R_CYCLES_LO 0x38
#define R_CYCLES_HI 0x3C
#define R_HEARTBEAT 0x48
#define R_STRIDE_OUT 0x4C
#define R_SCRATCH   0x50

#define CTRL_START      (1u << 0)
#define CTRL_RESET_SOFT (1u << 1)
#define CTRL_CLEAR_ERR  (1u << 3)
#define ST_BUSY  (1u << 0)
#define ST_DONE  (1u << 1)
#define ST_ERROR (1u << 2)

#define VERSION_EXPECTED 0x46500001u

static volatile uint8_t *base;
static int fails, checks;

static uint32_t rd(unsigned off) { return *(volatile uint32_t *)(base + off); }
static void wr(unsigned off, uint32_t v) { *(volatile uint32_t *)(base + off) = v; }
static void wr8(unsigned off, uint8_t v) { *(volatile uint8_t *)(base + off) = v; }

static void check(int ok, const char *what, uint32_t got, uint32_t expected)
{
    checks++;
    if (ok) {
        printf("  PASS  %s\n", what);
    } else {
        fails++;
        printf("  FAIL  %s (got 0x%08x, expected 0x%08x)\n", what, got, expected);
    }
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* poll STATUS until BUSY drops; returns elapsed ms or -1 on timeout */
static double wait_idle(double timeout_ms)
{
    double t0 = now_ms();
    while (rd(R_STATUS) & ST_BUSY)
        if (now_ms() - t0 > timeout_ms)
            return -1;
    return now_ms() - t0;
}

int main(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem (run as root)");
        return 2;
    }
    base = mmap(NULL, MAP_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, LW_BASE);
    if (base == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    printf("FPGate accelerator register test, base 0x%08x\n", LW_BASE);

    printf("[1] identification\n");
    uint32_t v = rd(R_VERSION);
    printf("  VERSION = 0x%08x\n", v);
    check(v == VERSION_EXPECTED, "VERSION is 'FP' 0001", v, VERSION_EXPECTED);
    if (v != VERSION_EXPECTED) {
        printf("The design is not what was expected at this address.\n"
               "Check that the .sof was programmed and that the HPS-FPGA bridges are enabled.\n");
        return 1;
    }
    check(rd(0x60) == 0xDEADBEEFu, "unmapped register reads 0xDEADBEEF", rd(0x60), 0xDEADBEEFu);

    printf("[2] heartbeat\n");
    uint32_t h0 = rd(R_HEARTBEAT);
    double t0 = now_ms();
    usleep(20000);
    uint32_t h1 = rd(R_HEARTBEAT);
    double dt = now_ms() - t0;
    double mhz = (uint32_t)(h1 - h0) / (dt * 1000.0);
    printf("  advanced by %u in %.2f ms -> fabric clock about %.1f MHz\n", h1 - h0, dt, mhz);
    check(h1 != h0, "HEARTBEAT advances", h1, h0 + 1);

    printf("[3] read/write registers\n");
    wr(R_SCRATCH, 0xA5A55A5Au);
    check(rd(R_SCRATCH) == 0xA5A55A5Au, "SCRATCH word write/read", rd(R_SCRATCH), 0xA5A55A5Au);
    wr(R_SCRATCH, 0x00000000u);
    wr8(R_SCRATCH + 1, 0xC3);
    check(rd(R_SCRATCH) == 0x0000C300u, "SCRATCH byte write (byte enable)", rd(R_SCRATCH), 0x0000C300u);
    wr(R_ADDR_A, 0x3F000000u);
    wr(R_STRIDE_OUT, 0x12345678u);
    check(rd(R_ADDR_A) == 0x3F000000u, "ADDR_A write/read", rd(R_ADDR_A), 0x3F000000u);
    check(rd(R_STRIDE_OUT) == 0x12345678u, "STRIDE_OUT write/read", rd(R_STRIDE_OUT), 0x12345678u);
    wr(R_VERSION, 0xFFFFFFFFu);
    check(rd(R_VERSION) == VERSION_EXPECTED, "VERSION is read-only", rd(R_VERSION), VERSION_EXPECTED);

    printf("[4] START -> BUSY -> DONE\n");
    wr(R_OP, 0);
    wr(R_DIM_M, 500000); /* placeholder engine: busy for DIM_M cycles */
    wr(R_CTRL, CTRL_START);
    uint32_t s = rd(R_STATUS);
    check(s & ST_BUSY, "BUSY right after START", s, ST_BUSY);
    double el = wait_idle(2000);
    check(el >= 0, "operation finishes (no timeout)", 0, 1);
    s = rd(R_STATUS);
    check(s == ST_DONE, "STATUS is DONE only", s, ST_DONE);
    uint32_t cyc = rd(R_CYCLES_LO);
    printf("  CYCLES = %u (about %.1f ms at the measured clock), polled %.2f ms\n",
           cyc, mhz > 0 ? cyc / (mhz * 1000.0) : 0.0, el);
    check(cyc >= 500000 && cyc <= 500100, "CYCLES close to DIM_M", cyc, 500000);
    check(rd(R_CYCLES_HI) == 0, "CYCLES_HI is 0", rd(R_CYCLES_HI), 0);

    printf("[5] START while BUSY\n");
    wr(R_DIM_M, 5000000);
    wr(R_CTRL, CTRL_START);
    wr(R_CTRL, CTRL_START);
    s = rd(R_STATUS);
    check((s & (ST_BUSY | ST_ERROR)) == (ST_BUSY | ST_ERROR), "BUSY and ERROR set", s, ST_BUSY | ST_ERROR);
    check(rd(R_ERR_CODE) == 1, "ERR_CODE is 1", rd(R_ERR_CODE), 1);
    wr(R_CTRL, CTRL_CLEAR_ERR);
    check((rd(R_STATUS) & ST_ERROR) == 0 && rd(R_ERR_CODE) == 0, "CLEAR_ERR clears the error", rd(R_STATUS), ST_BUSY);
    check(wait_idle(3000) >= 0, "first operation still completes", 0, 1);

    printf("[6] error reported by the engine\n");
    wr(R_OP, 0x80000000u);
    wr(R_DIM_M, 10000);
    wr(R_CTRL, CTRL_START);
    check(wait_idle(2000) >= 0, "operation ends", 0, 1);
    s = rd(R_STATUS);
    check(s == ST_ERROR, "STATUS is ERROR only", s, ST_ERROR);
    check(rd(R_ERR_CODE) == 0x10, "ERR_CODE is 0x10", rd(R_ERR_CODE), 0x10);
    wr(R_CTRL, CTRL_CLEAR_ERR);
    check(rd(R_STATUS) == 0, "idle after CLEAR_ERR", rd(R_STATUS), 0);

    printf("[7] soft reset while busy\n");
    wr(R_OP, 0);
    wr(R_DIM_M, 50000000);
    wr(R_CTRL, CTRL_START);
    check(rd(R_STATUS) & ST_BUSY, "busy before reset", rd(R_STATUS), ST_BUSY);
    wr(R_CTRL, CTRL_RESET_SOFT);
    check(rd(R_STATUS) == 0, "idle after RESET_SOFT", rd(R_STATUS), 0);
    usleep(10000);
    check(rd(R_STATUS) == 0, "still idle later (engine stopped)", rd(R_STATUS), 0);

    printf("\n%s: %d of %d checks passed\n", fails ? "FAILED" : "ALL PASSED", checks - fails, checks);
    munmap((void *)base, MAP_LEN);
    close(fd);
    return fails ? 1 : 0;
}
