#include "fpga_pack.h"

#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

/* In-place 16x16 byte transpose: on entry x[r] holds row r, on exit x[c] holds column c
 * (stages swap row/column bit 0, 1, 2, 3 using vtrn on 8/16/32-bit units and a 64-bit half swap). */
static inline void transpose16(uint8x16_t x[16])
{
    for (int i = 0; i < 16; i += 2) {
        uint8x16x2_t t = vtrnq_u8(x[i], x[i + 1]);
        x[i] = t.val[0]; x[i + 1] = t.val[1];
    }
    for (int i = 0; i < 16; i += 4)
        for (int j = 0; j < 2; j++) {
            uint16x8x2_t t = vtrnq_u16(vreinterpretq_u16_u8(x[i + j]), vreinterpretq_u16_u8(x[i + j + 2]));
            x[i + j] = vreinterpretq_u8_u16(t.val[0]); x[i + j + 2] = vreinterpretq_u8_u16(t.val[1]);
        }
    for (int i = 0; i < 16; i += 8)
        for (int j = 0; j < 4; j++) {
            uint32x4x2_t t = vtrnq_u32(vreinterpretq_u32_u8(x[i + j]), vreinterpretq_u32_u8(x[i + j + 4]));
            x[i + j] = vreinterpretq_u8_u32(t.val[0]); x[i + j + 4] = vreinterpretq_u8_u32(t.val[1]);
        }
    for (int i = 0; i < 8; i++) {
        uint8x16_t a = x[i], b = x[i + 8];
        x[i] = vcombine_u8(vget_low_u8(a), vget_low_u8(b));
        x[i + 8] = vcombine_u8(vget_high_u8(a), vget_high_u8(b));
    }
}

/* one full subtile (16 rows) of INT8 activations, columns [0, K & ~15) */
static void pack8_sub_neon(const int8_t *a, int K, int stride, uint8_t *d)
{
    uint8x16_t x[16];
    for (int k0 = 0; k0 + 16 <= K; k0 += 16) {
        for (int i = 0; i < 16; i++)
            x[i] = vld1q_u8((const uint8_t *)a + (size_t)i * stride + k0);
        transpose16(x);
        for (int c = 0; c < 16; c++)
            vst1q_u8(d + (size_t)(k0 + c) * 16, x[c]);
    }
}

/* same for INT16: hi plane then lo plane */
static void pack16_sub_neon(const int16_t *a, int K, uint8_t *hi, uint8_t *lo)
{
    uint8x16_t h[16], l[16];
    for (int k0 = 0; k0 + 16 <= K; k0 += 16) {
        for (int i = 0; i < 16; i++) {
            uint8x16x2_t v = vld2q_u8((const uint8_t *)(a + (size_t)i * K + k0)); /* val[0] = low bytes */
            l[i] = v.val[0]; h[i] = v.val[1];
        }
        transpose16(h);
        transpose16(l);
        for (int c = 0; c < 16; c++) {
            vst1q_u8(hi + (size_t)(k0 + c) * 16, h[c]);
            vst1q_u8(lo + (size_t)(k0 + c) * 16, l[c]);
        }
    }
}
#endif

size_t fpga_weights_bytes(const layer_t *L)
{
    return (size_t)(L->out / 16) * (256 + 16 * (size_t)L->in);
}

/* tile t (16 output channels) of L: 256 bytes of epilogue constants, then K words of 16 weights */
static void pack_tile(const layer_t *L, int t, uint8_t *c)
{
    const int K = L->in;
    memset(c, 0, 256);
    for (int j = 0; j < 16; j++) {
        const int n = t * 16 + j;
        const int32_t bias = L->bias ? L->bias[n] : 0;
        const uint32_t b = (uint32_t)bias, m = (uint32_t)L->m0[n];
        uint8_t *e = c + 16 * j;
        for (int i = 0; i < 4; i++) {
            e[i] = (uint8_t)(b >> (8 * i));
            e[4 + i] = (uint8_t)(m >> (8 * i));
        }
        e[8] = (uint8_t)L->shift[n];
    }
    uint8_t *w = c + 256;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < 16; j++)
            w[(size_t)k * 16 + j] = (uint8_t)L->w[(size_t)(t * 16 + j) * K + k];
}

void fpga_pack_weights(const layer_t *L, uint8_t *dst)
{
    for (int t = 0; t < L->out / 16; t++)
        pack_tile(L, t, dst + (size_t)t * (256 + 16 * (size_t)L->in));
}

size_t fpga_weights_pair_bytes(const layer_t *G, const layer_t *U)
{
    return fpga_weights_bytes(G) + fpga_weights_bytes(U);
}

void fpga_pack_weights_pair(const layer_t *G, const layer_t *U, uint8_t *dst)
{
    const size_t tb = 256 + 16 * (size_t)G->in;
    for (int t = 0; t < G->out / 16; t++) {
        pack_tile(G, t, dst + (size_t)(2 * t) * tb);
        pack_tile(U, t, dst + (size_t)(2 * t + 1) * tb);
    }
}

size_t fpga_acts_bytes(int rows, int K, int planes)
{
    return (size_t)((rows + 15) / 16) * planes * (size_t)K * 16;
}

void fpga_pack_acts8(const int8_t *a, int rows, int K, uint8_t *dst)
{
    const int subs = (rows + 15) / 16;
#pragma omp parallel for schedule(static)
    for (int s = 0; s < subs; s++) {
        uint8_t *d = dst + (size_t)s * K * 16;
        int k0 = 0;
#ifdef __ARM_NEON
        if (s * 16 + 16 <= rows) {
            pack8_sub_neon(a + (size_t)s * 16 * K, K, K, d);
            k0 = K & ~15;
        }
#endif
        for (int k = k0; k < K; k++)
            for (int i = 0; i < 16; i++) {
                const int r = s * 16 + i;
                d[(size_t)k * 16 + i] = r < rows ? (uint8_t)a[(size_t)r * K + k] : 0;
            }
    }
}

void fpga_pack_acts16(const int16_t *a, int rows, int K, uint8_t *dst)
{
    const int subs = (rows + 15) / 16;
#pragma omp parallel for schedule(static)
    for (int s = 0; s < subs; s++) {
        uint8_t *hi = dst + (size_t)s * 2 * K * 16;
        uint8_t *lo = hi + (size_t)K * 16;
        int k0 = 0;
#ifdef __ARM_NEON
        if (s * 16 + 16 <= rows) {
            pack16_sub_neon(a + (size_t)s * 16 * K, K, hi, lo);
            k0 = K & ~15;
        }
#endif
        for (int k = k0; k < K; k++)
            for (int i = 0; i < 16; i++) {
                const int r = s * 16 + i;
                const uint16_t v = r < rows ? (uint16_t)a[(size_t)r * K + k] : 0;
                hi[(size_t)k * 16 + i] = (uint8_t)(v >> 8);
                lo[(size_t)k * 16 + i] = (uint8_t)(v & 0xFF);
            }
    }
}

void fpga_pack_acts8_sub(const int8_t *a, int rows, int K, uint8_t *d)
{
    int k0 = 0;
#ifdef __ARM_NEON
    if (rows == 16) {
        pack8_sub_neon(a, K, K, d);
        k0 = K & ~15;
    }
#endif
    for (int k = k0; k < K; k++)
        for (int i = 0; i < 16; i++)
            d[(size_t)k * 16 + i] = i < rows ? (uint8_t)a[(size_t)i * K + k] : 0;
}

void fpga_pack_acts16_sub(const int16_t *a, int rows, int K, uint8_t *hi, uint8_t *lo)
{
    int k0 = 0;
#ifdef __ARM_NEON
    if (rows == 16) {
        pack16_sub_neon(a, K, hi, lo);
        k0 = K & ~15;
    }
#endif
    for (int k = k0; k < K; k++)
        for (int i = 0; i < 16; i++) {
            const uint16_t v = i < rows ? (uint16_t)a[(size_t)i * K + k] : 0;
            hi[(size_t)k * 16 + i] = (uint8_t)(v >> 8);
            lo[(size_t)k * 16 + i] = (uint8_t)(v & 0xFF);
        }
}

void fpga_pack_acts8_strided(const int8_t *a, int rows, int K, int stride, uint8_t *dst)
{
    const int subs = (rows + 15) / 16;
#pragma omp parallel for schedule(static)
    for (int s = 0; s < subs; s++) {
        uint8_t *d = dst + (size_t)s * K * 16;
        int k0 = 0;
#ifdef __ARM_NEON
        if (s * 16 + 16 <= rows) {
            pack8_sub_neon(a + (size_t)s * 16 * stride, K, stride, d);
            k0 = K & ~15;
        }
#endif
        for (int k = k0; k < K; k++)
            for (int i = 0; i < 16; i++) {
                const int r = s * 16 + i;
                d[(size_t)k * 16 + i] = r < rows ? (uint8_t)a[(size_t)r * stride + k] : 0;
            }
    }
}

size_t fpga_kt_bytes(int tokens, int hd)
{
    return (size_t)((tokens + 15) / 16) * (256 + 16 * (size_t)hd);
}

void fpga_pack_kt(const int8_t *k, int tokens, int stride, int hd, uint8_t *dst)
{
    const int nt = (tokens + 15) / 16;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        uint8_t *c = dst + (size_t)t * (256 + 16 * (size_t)hd);
        memset(c, 0, 256);
        uint8_t *w = c + 256;
        for (int d = 0; d < hd; d++)
            for (int j = 0; j < 16; j++) {
                const int key = t * 16 + j;
                w[(size_t)d * 16 + j] = key < tokens ? (uint8_t)k[(size_t)key * stride + d] : 0;
            }
    }
}

size_t fpga_v_bytes(int hd, int kp)
{
    return (size_t)(hd / 16) * (256 + 16 * (size_t)kp);
}

void fpga_pack_v(const int8_t *v, int tokens, int stride, int hd, int kp, int32_t av_m0, int av_sh, uint8_t *dst)
{
    for (int t = 0; t < hd / 16; t++) {
        uint8_t *c = dst + (size_t)t * (256 + 16 * (size_t)kp);
        memset(c, 0, 256);
        for (int j = 0; j < 16; j++) {
            uint8_t *e = c + 16 * j;
            for (int i = 0; i < 4; i++)
                e[4 + i] = (uint8_t)((uint32_t)av_m0 >> (8 * i));
            e[8] = (uint8_t)av_sh;
        }
        uint8_t *w = c + 256;
        for (int k = 0; k < kp; k++) {
            if (k < tokens)
                memcpy(w + (size_t)k * 16, v + (size_t)k * stride + t * 16, 16);
            else
                memset(w + (size_t)k * 16, 0, 16);
        }
    }
}
