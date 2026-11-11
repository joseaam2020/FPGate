#include "fpga_pack.h"

#include <string.h>

size_t fpga_weights_bytes(const layer_t *L)
{
    return (size_t)(L->out / 16) * (256 + 16 * (size_t)L->in);
}

void fpga_pack_weights(const layer_t *L, uint8_t *dst)
{
    const int K = L->in;
    for (int t = 0; t < L->out / 16; t++) {
        uint8_t *c = dst + (size_t)t * (256 + 16 * (size_t)K);
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
}

size_t fpga_acts_bytes(int rows, int K, int planes)
{
    return (size_t)((rows + 15) / 16) * planes * (size_t)K * 16;
}

void fpga_pack_acts8(const int8_t *a, int rows, int K, uint8_t *dst)
{
    const int subs = (rows + 15) / 16;
    for (int s = 0; s < subs; s++) {
        uint8_t *d = dst + (size_t)s * K * 16;
        for (int k = 0; k < K; k++)
            for (int i = 0; i < 16; i++) {
                const int r = s * 16 + i;
                d[(size_t)k * 16 + i] = r < rows ? (uint8_t)a[(size_t)r * K + k] : 0;
            }
    }
}

void fpga_pack_acts16(const int16_t *a, int rows, int K, uint8_t *dst)
{
    const int subs = (rows + 15) / 16;
    for (int s = 0; s < subs; s++) {
        uint8_t *hi = dst + (size_t)s * 2 * K * 16;
        uint8_t *lo = hi + (size_t)K * 16;
        for (int k = 0; k < K; k++)
            for (int i = 0; i < 16; i++) {
                const int r = s * 16 + i;
                const uint16_t v = r < rows ? (uint16_t)a[(size_t)r * K + k] : 0;
                hi[(size_t)k * 16 + i] = (uint8_t)(v >> 8);
                lo[(size_t)k * 16 + i] = (uint8_t)(v & 0xFF);
            }
    }
}

void fpga_pack_acts8_strided(const int8_t *a, int rows, int K, int stride, uint8_t *dst)
{
    const int subs = (rows + 15) / 16;
    for (int s = 0; s < subs; s++) {
        uint8_t *d = dst + (size_t)s * K * 16;
        for (int k = 0; k < K; k++)
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
