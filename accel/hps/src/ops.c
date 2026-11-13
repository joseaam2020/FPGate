#include "ops.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
/* horizontal sum of an int32x4 (ARMv7 has no vaddvq_s32) */
static inline int32_t vaddvq_s32_compat(int32x4_t v)
{
    int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(s, s), 0);
}
#endif

/* Scalar reference element: exact rintf(x / sc) clamped to +-qm (kept as the fallback of the NEON path). */
static inline int quant_one(float x, float sc, float inv, int qm)
{
    const float magic = 12582912.0f; /* 1.5 * 2^23 */
    const float t = x * inv;
    float r = (t + magic) - magic;
    const float f = fabsf(t - truncf(t));
    if (fabsf(f - 0.5f) < 1e-5f * fabsf(t) + 1e-6f)
        r = rintf(x / sc);
    int v = (int)r;
    return v > qm ? qm : v < -qm ? -qm : v;
}

void quantize_f32(const float *x, size_t n, double scale, int bits, void *q)
{
    const float sc = (float)scale;
    const float inv = 1.0f / sc;
    const int qm = qmax_of(bits);
    /* Exact replacement for rintf(x / sc): the product x * inv differs from the quotient by a few
     * ulp, which can only change the rounded value when the quotient is within that error of a
     * half-integer. Those (rare) elements are recomputed with the real division. Values with
     * |q| >= 2^22 are clamped anyway (qm <= 32767), so the magic-number rounding is valid. */
    const ptrdiff_t nn = (ptrdiff_t)n;
    ptrdiff_t i0 = 0;
#ifdef __ARM_NEON
    /* NEON path, 4 floats per step. Exactness: vmulq_f32 / vaddq_f32 / vsubq_f32 round to nearest
     * even on normal numbers, exactly like the scalar code. If t is denormal the result is 0 in both
     * paths (|t| < 1 rounds to 0 or +-1 identically and no tie candidate arises). d = |t - r| equals
     * min(f, 1 - f) with f = |t - trunc(t)|, so |d - 0.5| is the scalar tie test. Any lane that is a
     * tie candidate, or has |t| >= 2^22 (or NaN), is recomputed with the exact scalar code. */
    const float32x4_t vinv = vdupq_n_f32(inv), vmagic = vdupq_n_f32(12582912.0f);
    const float32x4_t vhalf = vdupq_n_f32(0.5f), vk = vdupq_n_f32(1e-5f), ve = vdupq_n_f32(1e-6f);
    const float32x4_t vbig = vdupq_n_f32(4194304.0f);
    const int32x4_t vqm = vdupq_n_s32(qm), vnqm = vdupq_n_s32(-qm);
    const ptrdiff_t nv = nn & ~(ptrdiff_t)3;
#pragma omp parallel for schedule(static) if (n > 4096)
    for (ptrdiff_t i = 0; i < nv; i += 4) {
        const float32x4_t xv = vld1q_f32(x + i);
        const float32x4_t t = vmulq_f32(xv, vinv);
        const float32x4_t r = vsubq_f32(vaddq_f32(t, vmagic), vmagic);
        const float32x4_t at = vabsq_f32(t);
        const float32x4_t d = vabsq_f32(vsubq_f32(t, r));
        /* tie candidate: |d - 0.5| < 1e-5*|t| + 1e-6 (separate mul and add, no FMA) */
        const float32x4_t lim = vaddq_f32(vmulq_f32(vk, at), ve);
        const uint32x4_t tie = vcltq_f32(vabsq_f32(vsubq_f32(d, vhalf)), lim);
        const uint32x4_t ok = vcltq_f32(at, vbig); /* false for NaN as well */
        int32x4_t v = vcvtq_s32_f32(r);
        v = vminq_s32(vmaxq_s32(v, vnqm), vqm);
        const uint32x4_t bad = vmvnq_u32(vandq_u32(ok, vmvnq_u32(tie)));
        const uint32x2_t b2 = vorr_u32(vget_low_u32(bad), vget_high_u32(bad));
        if (vget_lane_u32(vpmax_u32(b2, b2), 0)) {
            int32_t tmp[4];
            for (int k = 0; k < 4; k++)
                tmp[k] = quant_one(x[i + k], sc, inv, qm);
            v = vld1q_s32(tmp);
        }
        const int16x4_t v16 = vmovn_s32(v);
        if (bits == 8)
            vst1_lane_u32((uint32_t *)((int8_t *)q + i), vreinterpret_u32_s8(vqmovn_s16(vcombine_s16(v16, v16))), 0);
        else
            vst1_s16((int16_t *)q + i, v16);
    }
    i0 = nv;
#endif
#pragma omp parallel for schedule(static) if (n > 4096)
    for (ptrdiff_t i = i0; i < nn; i++) {
        const int v = quant_one(x[i], sc, inv, qm);
        if (bits == 8)
            ((int8_t *)q)[i] = (int8_t)v;
        else
            ((int16_t *)q)[i] = (int16_t)v;
    }
}

void gemm_s8(const int8_t *A, const int8_t *W, int32_t *C, int M, int N, int K)
{
#ifdef __ARM_NEON
    /* 1 row x 4 columns per step: each 8-byte load of the row feeds four dot products
     * (vmull_s8 -> int16 products, vpadalq_s16 -> int32 accumulators). Integer, so exact. */
    const int K8 = K & ~7;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int8_t *a = A + (size_t)i * K;
        int32_t *c = C + (size_t)i * N;
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            const int8_t *w0 = W + (size_t)j * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
            int32x4_t s0 = vdupq_n_s32(0), s1 = s0, s2 = s0, s3 = s0;
            int k = 0;
            for (; k < K8; k += 8) {
                const int8x8_t av = vld1_s8(a + k);
                s0 = vpadalq_s16(s0, vmull_s8(av, vld1_s8(w0 + k)));
                s1 = vpadalq_s16(s1, vmull_s8(av, vld1_s8(w1 + k)));
                s2 = vpadalq_s16(s2, vmull_s8(av, vld1_s8(w2 + k)));
                s3 = vpadalq_s16(s3, vmull_s8(av, vld1_s8(w3 + k)));
            }
            int32_t t0 = vaddvq_s32_compat(s0), t1 = vaddvq_s32_compat(s1);
            int32_t t2 = vaddvq_s32_compat(s2), t3 = vaddvq_s32_compat(s3);
            for (; k < K; k++) {
                const int32_t av = a[k];
                t0 += av * w0[k]; t1 += av * w1[k]; t2 += av * w2[k]; t3 += av * w3[k];
            }
            c[j] = t0; c[j + 1] = t1; c[j + 2] = t2; c[j + 3] = t3;
        }
        for (; j < N; j++) {
            const int8_t *w = W + (size_t)j * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)a[k] * (int32_t)w[k];
            c[j] = s;
        }
    }
#else
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int8_t *a = A + (size_t)i * K;
        int32_t *c = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const int8_t *w = W + (size_t)j * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)a[k] * (int32_t)w[k];
            c[j] = s;
        }
    }
#endif
}

void gemm_s16(const int16_t *A, const int8_t *W, int32_t *C, int M, int N, int K)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int16_t *a = A + (size_t)i * K;
        int32_t *c = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const int8_t *w = W + (size_t)j * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)a[k] * (int32_t)w[k];
            c[j] = s;
        }
    }
}

void epilogue_int(const int32_t *acc, int M, int N, const int32_t *bias, const int32_t *m0, const int32_t *shift,
                  int bits, int16_t *out)
{
    const int64_t qm = qmax_of(bits);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int64_t a = acc[(size_t)i * N + j];
            if (bias)
                a += bias[j];
            int64_t y = (a * (int64_t)m0[j] + ((int64_t)1 << (shift[j] - 1))) >> shift[j];
            out[(size_t)i * N + j] = (int16_t)(y > qm ? qm : y < -qm ? -qm : y);
        }
}

#ifdef LN_FAST_DIV
/* a / den, bit-identical to the IEEE division. t = a * rden is a candidate; the residual
 * e = a - t * den is exact in double (24 x 24 bit product, and a ~ t * den so the subtraction
 * is exact). t is the correctly rounded quotient when |e| < half the distance from t to its
 * nearest neighbour times den (strict: ties and anything doubtful use the real division). */
static inline float ln_div(float a, float den, float rden)
{
    const float t = a * rden;
    uint32_t u;
    memcpy(&u, &t, sizeof u);
    const uint32_t E = (u >> 23) & 0xff;
    if (E <= 24 || E == 255)
        return a / den; /* zero, subnormal, tiny, inf, nan */
    /* spacing below t is half the spacing above when t is a power of two */
    const uint32_t Ep = (u & 0x7fffff) ? E : E - 1;
    const uint32_t hb = (Ep - 24) << 23; /* 2^(Ep-127-24) = half ulp, as float bits */
    float half;
    memcpy(&half, &hb, sizeof half);
    const double e = (double)a - (double)t * (double)den;
    const double lim = (double)half * (double)den;
    return (e < lim && e > -lim) ? t : a / den;
}
#endif

void layer_norm(const float *x, int rows, int d, const float *g, const float *b, double eps, float *y)
{
    const float epsf = (float)eps;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * d;
        float *yr = y + (size_t)r * d;
        /* four independent double accumulators break the add-latency chain (the VFP double add
         * has a latency of several cycles on the Cortex-A9). Sums of float values in double are
         * exact for the ranges seen here, so the result does not depend on the order. */
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = 0;
        for (; j + 4 <= d; j += 4) {
            s0 += xr[j]; s1 += xr[j + 1]; s2 += xr[j + 2]; s3 += xr[j + 3];
        }
        for (; j < d; j++)
            s0 += xr[j];
        const float mu = (float)(((s0 + s1) + (s2 + s3)) / d);
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
        j = 0;
        for (; j + 4 <= d; j += 4) {
            const float e0 = xr[j] - mu, e1 = xr[j + 1] - mu, e2 = xr[j + 2] - mu, e3 = xr[j + 3] - mu;
            v0 += (double)(e0 * e0); v1 += (double)(e1 * e1); v2 += (double)(e2 * e2); v3 += (double)(e3 * e3);
        }
        for (; j < d; j++) {
            const float e = xr[j] - mu;
            v0 += (double)(e * e);
        }
        const float var = (float)(((v0 + v1) + (v2 + v3)) / d);
        const float den = sqrtf(var + epsf);
#ifndef LN_FAST_DIV
        for (int k = 0; k < d; k++) {
            float t = (xr[k] - mu) / den;
#else
        const float rden = 1.0f / den;
        for (int k = 0; k < d; k++) {
            float t = ln_div(xr[k] - mu, den, rden);
#endif
            t = t * g[k];
            yr[k] = t + b[k];
        }
    }
}

void rope_int(const int16_t *x, int tokens, int stride, int heads, int hd, const int16_t *cos_tab,
              const int16_t *sin_tab, int frac, int pre_shift, int8_t *out)
{
    rope_int_range(x, 0, tokens, stride, heads, hd, cos_tab, sin_tab, frac, pre_shift, out);
}

/* x and out point at the first of `tokens` rows, which are tokens t0 .. t0 + tokens - 1 of the sequence
 * (token 0 is the class token: no rotation) */
void rope_int_range(const int16_t *x, int t0, int tokens, int stride, int heads, int hd, const int16_t *cos_tab,
                    const int16_t *sin_tab, int frac, int pre_shift, int8_t *out)
{
    const int sh = frac + pre_shift;
    const int64_t rnd = (int64_t)1 << (sh - 1);
#pragma omp parallel for schedule(static)
    for (int tl = 0; tl < tokens; tl++) {
        const int t = t0 + tl;
        for (int h = 0; h < heads; h++) {
            const int16_t *xi = x + (size_t)tl * stride + (size_t)h * hd;
            int8_t *o = out + (size_t)tl * stride + (size_t)h * hd;
            if (t == 0) {
                for (int d = 0; d < hd; d++) {
                    int64_t v = (xi[d] + ((int64_t)1 << (pre_shift - 1))) >> pre_shift;
                    o[d] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
                }
                continue;
            }
            const int16_t *c = cos_tab + (size_t)(t - 1) * hd;
            const int16_t *s = sin_tab + (size_t)(t - 1) * hd;
            for (int d = 0; d < hd; d++) {
                /* rot_interleaved: [-x1, x0, -x3, x2, ...] */
                int64_t rot = (d & 1) ? xi[d - 1] : -(int64_t)xi[d + 1];
                int64_t v = ((int64_t)xi[d] * c[d] + rot * s[d] + rnd) >> sh;
                o[d] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
            }
        }
    }
}

/* Integer softmax of one row of scores (3 passes: max, exponential by table + sum, reciprocal by table).
 * Shared by the production kernel and the reference dumps so both are the same code. */
void softmax_row_int(const pkg_t *p, const block_t *b, const int32_t *S, int n, uint16_t *P)
{
    const int n_entries = 1024;                 /* exp table has 1025 entries; index clamps at 1024 */
    const int pbits = p->softmax_bits;
    const int64_t sm_rnd = (int64_t)1 << (b->sm_sh - 1);
    int32_t mx = S[0];
    for (int j = 1; j < n; j++)
        if (S[j] > mx)
            mx = S[j];
    uint32_t sum = 0;
    for (int j = 0; j < n; j++) {
        int64_t d = (int64_t)mx - S[j];
        int64_t idx = (d * b->sm_m0 + sm_rnd) >> b->sm_sh;
        if (idx > n_entries)
            idx = n_entries;
        uint16_t e = p->exp_tab[idx];
        P[j] = e;
        sum += e;
    }
    int kb = 31 - __builtin_clz(sum);
    int mant = (int)((sum >> (kb - 8)) - 256);
    int sh = 15 + kb - pbits;
    uint64_t r = p->recip_tab[mant];
    uint64_t rnd = (uint64_t)1 << (sh - 1);
    const uint64_t pmax = ((uint64_t)1 << pbits) - 1;
    for (int j = 0; j < n; j++) {
        uint64_t pv = ((uint64_t)P[j] * r + rnd) >> sh;
        P[j] = (uint16_t)(pv > pmax ? pmax : pv);
    }
}

void attention_head_ref(const pkg_t *p, const block_t *b, const int8_t *q, const int8_t *k, const int8_t *v,
                        int tokens, int stride, int hd, int32_t *S, uint16_t *P, int8_t *ctx)
{
    const int64_t av_rnd = (int64_t)1 << (b->av_sh - 1);
    for (int i = 0; i < tokens; i++) {
        const int8_t *qi = q + (size_t)i * stride;
        int32_t *Si = S + (size_t)i * tokens;
        uint16_t *Pi = P + (size_t)i * tokens;
        for (int j = 0; j < tokens; j++) {
            const int8_t *kj = k + (size_t)j * stride;
            int32_t s = 0;
            for (int d = 0; d < hd; d++)
                s += (int32_t)qi[d] * (int32_t)kj[d];
            Si[j] = s;
        }
        softmax_row_int(p, b, Si, tokens, Pi);
        for (int d = 0; d < hd; d++) {
            int32_t acc = 0;
            for (int j = 0; j < tokens; j++)
                acc += (int32_t)Pi[j] * (int32_t)v[(size_t)j * stride + d];
            int64_t y = ((int64_t)acc * b->av_m0 + av_rnd) >> b->av_sh;
            ctx[(size_t)i * hd + d] = (int8_t)(y > 127 ? 127 : y < -127 ? -127 : y);
        }
    }
}

void attention_int(const pkg_t *p, const block_t *b, const int8_t *q, const int8_t *k, const int8_t *v, int tokens,
                   int stride, int heads, int hd, int8_t *ctx)
{
    const int64_t av_rnd = (int64_t)1 << (b->av_sh - 1);

#pragma omp parallel
    {
        int32_t *S = malloc((size_t)tokens * sizeof *S);
        uint16_t *P = malloc((size_t)tokens * sizeof *P);
        int32_t acc[128];
#pragma omp for schedule(static) collapse(2)
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < tokens; i++) {
                const int8_t *qi = q + (size_t)i * stride + (size_t)h * hd;
                for (int j = 0; j < tokens; j++) {
                    const int8_t *kj = k + (size_t)j * stride + (size_t)h * hd;
                    int32_t s = 0;
                    for (int d = 0; d < hd; d++)
                        s += (int32_t)qi[d] * (int32_t)kj[d];
                    S[j] = s;
                }
                softmax_row_int(p, b, S, tokens, P);
                /* context = P @ V, requantised to INT8 */
                memset(acc, 0, sizeof(int32_t) * (size_t)hd);
                for (int j = 0; j < tokens; j++) {
                    const int32_t pj = P[j];
                    if (!pj)
                        continue;
                    const int8_t *vj = v + (size_t)j * stride + (size_t)h * hd;
                    for (int d = 0; d < hd; d++)
                        acc[d] += pj * (int32_t)vj[d];
                }
                int8_t *o = ctx + (size_t)i * stride + (size_t)h * hd;
                for (int d = 0; d < hd; d++) {
                    int64_t y = ((int64_t)acc[d] * b->av_m0 + av_rnd) >> b->av_sh;
                    o[d] = (int8_t)(y > 127 ? 127 : y < -127 ? -127 : y);
                }
            }
        free(S);
        free(P);
    }
}
