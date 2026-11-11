#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void quantize_f32(const float *x, size_t n, double scale, int bits, void *q)
{
    const float sc = (float)scale;
    const int qm = qmax_of(bits);
    if (bits == 8) {
        int8_t *o = q;
        for (size_t i = 0; i < n; i++) {
            int v = (int)rintf(x[i] / sc);
            o[i] = (int8_t)(v > qm ? qm : v < -qm ? -qm : v);
        }
    } else {
        int16_t *o = q;
        for (size_t i = 0; i < n; i++) {
            int v = (int)rintf(x[i] / sc);
            o[i] = (int16_t)(v > qm ? qm : v < -qm ? -qm : v);
        }
    }
}

void gemm_s8(const int8_t *A, const int8_t *W, int32_t *C, int M, int N, int K)
{
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

void layer_norm(const float *x, int rows, int d, const float *g, const float *b, double eps, float *y)
{
    const float epsf = (float)eps;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * d;
        float *yr = y + (size_t)r * d;
        double s = 0.0;
        for (int j = 0; j < d; j++)
            s += xr[j];
        const float mu = (float)(s / d);
        double v = 0.0;
        for (int j = 0; j < d; j++) {
            float e = xr[j] - mu;
            v += (double)(e * e);
        }
        const float var = (float)(v / d);
        const float den = sqrtf(var + epsf);
        for (int j = 0; j < d; j++) {
            float t = (xr[j] - mu) / den;
            t = t * g[j];
            yr[j] = t + b[j];
        }
    }
}

void rope_int(const int16_t *x, int tokens, int stride, int heads, int hd, const int16_t *cos_tab,
              const int16_t *sin_tab, int frac, int pre_shift, int8_t *out)
{
    const int sh = frac + pre_shift;
    const int64_t rnd = (int64_t)1 << (sh - 1);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; t++) {
        for (int h = 0; h < heads; h++) {
            const int16_t *xi = x + (size_t)t * stride + (size_t)h * hd;
            int8_t *o = out + (size_t)t * stride + (size_t)h * hd;
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
