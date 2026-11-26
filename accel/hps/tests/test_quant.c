#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ops.h"

static uint32_t s = 12345;
static uint32_t rnd(void) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

int main(void)
{
    const double scales[] = {0.0123, 0.5, 1.0 / 127, 3.7e-4, 0.0317, 2.0, 0.1, 1e-3, 0.0009765625};
    size_t bad = 0, total = 0;
    enum { N = 1 << 20 };
    float *x = malloc(N * sizeof *x);
    int16_t *q = malloc(N * sizeof *q);
    int8_t *q8 = malloc(N);
    for (int si = 0; si < 9; si++)
        for (int bits = 8; bits <= 16; bits += 8)
            for (int rep = 0; rep < 12; rep++) {
                float sc = (float)scales[si];
                for (int i = 0; i < N; i++) {
                    int mode = rnd() % 3;
                    float v;
                    if (mode == 0) v = ((float)(rnd() >> 8) / 16777216.0f - 0.5f) * 2.0f * sc * 40000.0f;
                    else if (mode == 1) v = ((int)(rnd() % 65536) - 32768) * sc + ((rnd() & 1) ? 0.5f : -0.5f) * sc; /* near ties */
                    else { float t = ((int)(rnd() % 4001) - 2000) + 0.5f; v = t * sc; }
                    if (rnd() % 7 == 0) v = nextafterf(v, (rnd() & 1) ? 1e30f : -1e30f);
                    x[i] = v;
                }
                if (bits == 8) quantize_f32(x, N, scales[si], 8, q8);
                else quantize_f32(x, N, scales[si], 16, q);
                const int qm = bits == 8 ? 127 : 32767;
                for (int i = 0; i < N; i++) {
                    int v = (int)rintf(x[i] / sc);
                    v = v > qm ? qm : v < -qm ? -qm : v;
                    int got = bits == 8 ? q8[i] : q[i];
                    total++;
                    if (v != got) bad++;
                }
            }
    /* 50 M extra random values: scales 1e-4..10 (log-uniform), values near +-0.5*k, saturation, tiny/huge */
    for (int rep = 0; rep < 50; rep++) {
        const int bits = (rep & 1) ? 16 : 8;
        const float sc = powf(10.0f, -4.0f + 5.0f * ((float)(rnd() >> 8) / 16777216.0f));
        for (int i = 0; i < N; i++) {
            int mode = rnd() % 6;
            float v;
            const float u = (float)(rnd() >> 8) / 16777216.0f - 0.5f;
            if (mode == 0) v = u * 2.0f * sc * 300.0f;                       /* mostly in range for 8 bit */
            else if (mode == 1) v = u * 2.0f * sc * 70000.0f;                /* saturates */
            else if (mode == 2) v = (((int)(rnd() % 4001) - 2000) + 0.5f) * sc; /* exact-ish ties */
            else if (mode == 3) v = (((int)(rnd() % 200001) - 100000) + ((rnd() & 1) ? 0.5f : -0.5f)) * sc;
            else if (mode == 4) v = u * 1e-37f;                              /* tiny / denormal products */
            else v = u * 2.0f * sc * 1e7f;                                   /* beyond 2^22 */
            if (rnd() % 5 == 0) v = nextafterf(v, (rnd() & 1) ? 1e30f : -1e30f);
            x[i] = v;
        }
        if (bits == 8) quantize_f32(x, N, sc, 8, q8);
        else quantize_f32(x, N, sc, 16, q);
        const float fsc = (float)(double)sc;
        const int qm = bits == 8 ? 127 : 32767;
        for (int i = 0; i < N; i++) {
            int v = (int)rintf(x[i] / fsc);
            v = v > qm ? qm : v < -qm ? -qm : v;
            int got = bits == 8 ? q8[i] : q[i];
            total++;
            if (v != got) bad++;
        }
    }
    printf("%zu values, %zu mismatches\n", total, bad);
    return bad != 0;
}
