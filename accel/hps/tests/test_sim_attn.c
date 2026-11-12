/*
 * Software engine + softmax unit (accel_sim.c) against attention_head_ref: two heads with the geometry of block 0
 * of the package go through Q.K^T (RAW32, stride) -> softmax unit -> P.V (unsigned UQ0.16 input, stride) and must
 * give the same S, P and ctx as the CPU reference. The softmax of head 0 runs while the engine computes the scores
 * of head 1 (both in flight at once). Negative control: reading P before the unit is done gives differences.
 * usage: test_sim_attn PACKAGE_DIR INDEX
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "accel_drv.h"
#include "fpga_pack.h"
#include "ops.h"
#include "pkg.h"

#define NH 2
#define PG(x) (((x) + 4095) & ~(size_t)4095)

typedef struct {
    size_t o[6];                 /* region offsets: q, k, v, s, p, c */
    int8_t *q8, *k8, *v8;        /* [T][A] */
} head_t;

static int T, A, hd, NK, KP;
static accel_hw_t hw;
static const block_t *blk;

static void fill(int8_t *x, size_t n, unsigned *seed)
{
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        x[i] = (int8_t)((int)((*seed >> 24) % 41) - 20);
    }
}

static accel_op_t s_op(const head_t *h)
{
    accel_op_t op = {.M = (uint32_t)T, .N = (uint32_t)NK, .K = (uint32_t)hd, .in_bits = 8, .out_bits = 32, .spp = 1,
                     .a_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[0], .w_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[1],
                     .out_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[3], .raw32 = 1, .stride_out = (uint32_t)NK * 4};
    return op;
}

static accel_sm_op_t sm_op(const head_t *h, const accel_sm_sim_ctx_t *ctx)
{
    accel_sm_op_t op = {.rows = (uint32_t)T, .cols = (uint32_t)T, .kp = (uint32_t)KP,
                        .s_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[3], .s_stride = (uint32_t)NK * 4,
                        .p_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[4], .m0 = (uint32_t)blk->sm_m0,
                        .sh = (int)blk->sm_sh, .sim_ctx = ctx};
    return op;
}

static accel_op_t pv_op(const head_t *h)
{
    accel_op_t op = {.M = (uint32_t)T, .N = (uint32_t)hd, .K = (uint32_t)KP, .in_bits = 16, .out_bits = 8, .spp = 1,
                     .a_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[4], .w_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[2],
                     .out_phys = ACCEL_REGION_PHYS + (uint32_t)h->o[5], .in_unsigned = 1, .stride_out = (uint32_t)A};
    return op;
}

static uint16_t p_at(const head_t *h, int r, int k)
{
    const uint8_t *hi = hw.region + h->o[4] + (size_t)(r / 16) * 2 * KP * 16;
    return (uint16_t)((hi[(size_t)k * 16 + r % 16] << 8) | hi[(size_t)KP * 16 + (size_t)k * 16 + r % 16]);
}

static long count_p_diff(const head_t *h, const uint16_t *Pr)
{
    long bad = 0;
    for (int r = 0; r < T; r++)
        for (int k = 0; k < KP; k++)
            bad += p_at(h, r, k) != (k < T ? Pr[(size_t)r * T + k] : 0);
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s PACKAGE_DIR INDEX\n", argv[0]);
        return 2;
    }
    pkg_t *pkg = malloc(sizeof *pkg);
    if (pkg_load(pkg, argv[1], argv[2]) != 0)
        return 2;
    blk = &pkg->blocks[0];
    T = pkg->tokens;
    A = blk->q->out;
    hd = A / blk->heads;
    NK = ((T + 15) / 16) * 16;
    KP = (T + 1) & ~1;
    printf("block 0: tokens %d, heads %d, head dim %d\n", T, blk->heads, hd);

    const size_t sz[6] = {PG(fpga_acts_bytes(T, hd, 1)), PG(fpga_kt_bytes(T, hd)), PG(fpga_v_bytes(hd, KP)),
                          PG((size_t)T * NK * 4), PG(fpga_acts_bytes(T, KP, 2)), PG((size_t)T * A)};
    size_t per = 0;
    for (int i = 0; i < 6; i++)
        per += sz[i];
    if (accel_hw_open_sim(&hw, per * NH) != 0)
        return 2;

    head_t hs[NH];
    unsigned seed = 12345;
    for (int h = 0; h < NH; h++) {
        size_t off = per * h;
        for (int i = 0; i < 6; i++) {
            hs[h].o[i] = off;
            off += sz[i];
        }
        hs[h].q8 = malloc((size_t)T * A);
        hs[h].k8 = malloc((size_t)T * A);
        hs[h].v8 = malloc((size_t)T * A);
        fill(hs[h].q8, (size_t)T * A, &seed);
        fill(hs[h].k8, (size_t)T * A, &seed);
        fill(hs[h].v8, (size_t)T * A, &seed);
        /* the head is columns [0, hd) of the [T][A] matrices, as head 0 in the model */
        fpga_pack_acts8_strided(hs[h].q8, T, hd, A, hw.region + hs[h].o[0]);
        fpga_pack_kt(hs[h].k8, T, A, hd, hw.region + hs[h].o[1]);
        fpga_pack_v(hs[h].v8, T, A, hd, KP, (int32_t)blk->av_m0, (int)blk->av_sh, hw.region + hs[h].o[2]);
    }

    const accel_sm_sim_ctx_t ctx = {pkg, blk};
    int fail = 0;
    accel_op_t a0 = s_op(&hs[0]), a1 = s_op(&hs[1]);
    accel_sm_op_t m0 = sm_op(&hs[0], &ctx), m1 = sm_op(&hs[1], &ctx);

    /* scores of head 0; then its softmax together with the scores of head 1 */
    if (accel_hw_run(&hw, &a0, NULL, 60) != 0)
        return 1;
    if (accel_sim_sm_start(&hw, &m0) != 0 || accel_hw_start(&hw, &a1) != 0)
        return 1;
    const int overlapped = accel_sim_sm_busy(&hw) && accel_hw_busy(&hw);
    if (accel_hw_wait(&hw, &a1, NULL, 60) != 0 || accel_sim_sm_wait(&hw, NULL) != 0)
        return 1;
    printf("engine and softmax unit in flight at the same time: %s\n", overlapped ? "yes" : "NO");
    fail |= !overlapped;
    if (accel_sim_sm_start(&hw, &m1) != 0 || accel_sim_sm_wait(&hw, NULL) != 0)
        return 1;
    for (int h = 0; h < NH; h++) {
        accel_op_t pv = pv_op(&hs[h]);
        if (accel_hw_run(&hw, &pv, NULL, 60) != 0)
            return 1;
    }

    int32_t *Sr = malloc((size_t)T * T * 4);
    uint16_t *Pr = malloc((size_t)T * T * 2);
    int8_t *cr = malloc((size_t)T * hd);
    for (int h = 0; h < NH; h++) {
        attention_head_ref(pkg, blk, hs[h].q8, hs[h].k8, hs[h].v8, T, A, hd, Sr, Pr, cr);
        const int32_t *s_hw = (const int32_t *)(hw.region + hs[h].o[3]);
        long bad_s = 0, bad_c = 0;
        for (int r = 0; r < T; r++)
            for (int j = 0; j < T; j++)
                bad_s += s_hw[(size_t)r * NK + j] != Sr[(size_t)r * T + j];
        const long bad_p = count_p_diff(&hs[h], Pr);
        for (int r = 0; r < T; r++)
            for (int d = 0; d < hd; d++)
                bad_c += (int8_t)hw.region[hs[h].o[5] + (size_t)r * A + d] != cr[(size_t)r * hd + d];
        printf("head %d: S differ %ld, P differ %ld, ctx differ %ld\n", h, bad_s, bad_p, bad_c);
        fail |= bad_s || bad_p || bad_c;
    }

    /* negative control: clear P, start the unit and read P at once (before it is done) */
    attention_head_ref(pkg, blk, hs[0].q8, hs[0].k8, hs[0].v8, T, A, hd, Sr, Pr, cr);
    memset(hw.region + hs[0].o[4], 0, sz[4]);
    if (accel_sim_sm_start(&hw, &m0) != 0)
        return 1;
    const int busy = accel_sim_sm_busy(&hw);
    const long early = count_p_diff(&hs[0], Pr);
    accel_sim_sm_wait(&hw, NULL);
    printf("negative control: unit %s when P was read, %ld values differ from the reference\n",
           busy ? "still busy" : "already done", early);
    if (!busy || early == 0) {
        printf("negative control FAILED: reading P early must give differences\n");
        fail = 1;
    }
    printf("after the wait: P differ %ld\n", count_p_diff(&hs[0], Pr));

    accel_hw_close(&hw);
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}
