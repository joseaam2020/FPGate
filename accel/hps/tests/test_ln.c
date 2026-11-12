#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ops.h"

static void ln_old(const float *x, int rows, int d, const float *g, const float *b, double eps, float *y)
{
    const float epsf = (float)eps;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * d;
        float *yr = y + (size_t)r * d;
        double s = 0.0;
        for (int j = 0; j < d; j++) s += xr[j];
        const float mu = (float)(s / d);
        double v = 0.0;
        for (int j = 0; j < d; j++) { float e = xr[j] - mu; v += (double)(e * e); }
        const float var = (float)(v / d);
        const float den = sqrtf(var + epsf);
        for (int j = 0; j < d; j++) { float t = (xr[j] - mu) / den; t = t * g[j]; yr[j] = t + b[j]; }
    }
}

static float rf(void) { return (float)rand() / RAND_MAX - 0.5f; }

int main(int argc, char **argv)
{
    const int dims[] = {768, 2048, 1232, 1024, 13};
    srand(11);
    long bad = 0, tot = 0, rowbad = 0;
    for (int di = 0; di < 5; di++) {
        int d = dims[di], rows = 300;
        float *x = malloc((size_t)rows * d * 4), *g = malloc(d * 4), *b = malloc(d * 4);
        float *y0 = malloc((size_t)rows * d * 4), *y1 = malloc((size_t)rows * d * 4);
        for (int j = 0; j < d; j++) { g[j] = 1.0f + rf(); b[j] = rf(); }
        for (int r = 0; r < rows; r++) {
            float sc = powf(10.0f, (float)(r % 5) - 2.0f);
            for (int j = 0; j < d; j++) {
                float v = rf() * 4.0f * sc;
                if (rand() % 200 == 0) v *= 300.0f;           /* outliers */
                if (rand() % 50 == 0) v = rf() * 1e-4f;       /* tiny values */
                x[(size_t)r * d + j] = v + 3.0f * sc;
            }
        }
        ln_old(x, rows, d, g, b, 1e-6, y0);
        layer_norm(x, rows, d, g, b, 1e-6, y1);
        for (int r = 0; r < rows; r++) {
            long rb = 0;
            for (int j = 0; j < d; j++) rb += memcmp(&y0[(size_t)r * d + j], &y1[(size_t)r * d + j], 4) != 0;
            bad += rb; rowbad += rb != 0; tot += d;
        }
    }
    printf("layer_norm: %ld of %ld values differ (%ld rows)\n", bad, tot, rowbad);
    long nrows = argc > 1 ? atol(argv[1]) : 200000;
    const int sdims[] = {1, 2, 3, 7, 13, 64, 96, 768, 1024};
    enum { D = 1024, B = 64 };
    float *x = malloc((size_t)D * B * 4), *g = malloc((size_t)D * 4), *b = malloc((size_t)D * 4);
    float *y0 = malloc((size_t)D * B * 4), *y1 = malloc((size_t)D * B * 4);
    uint64_t st = 0x9e3779b97f4a7c15ull;
    long sbad = 0, stot = 0;
#define XS (st ^= st << 13, st ^= st >> 7, st ^= st << 17, st)
#define UF ((float)(XS >> 40) / 16777216.0f)
    for (long done = 0; done < nrows; done += B) {
        int d = sdims[XS % 9];
        int rows = nrows - done < B ? (int)(nrows - done) : B;
        for (int j = 0; j < d; j++) { g[j] = 1.0f + (UF - 0.5f); b[j] = UF - 0.5f; }
        for (int r = 0; r < rows; r++) {
            float sc = powf(10.0f, (float)(int)(XS % 13) - 6.0f);   /* 1e-6 .. 1e6 */
            float off = (XS & 1) ? 0.0f : sc * (float)(int)(XS % 2001 - 1000) * 0.01f;
            float tiny = (XS % 4 == 0) ? 1e-5f : 1.0f;              /* small variance around a big mean */
            for (int j = 0; j < d; j++) {
                float v = (UF - 0.5f) * 4.0f * sc * tiny;
                if (XS % 200 == 0) v *= 300.0f;
                x[(size_t)r * d + j] = v + off;
            }
        }
        ln_old(x, rows, d, g, b, 1e-6, y0);
        layer_norm(x, rows, d, g, b, 1e-6, y1);
        for (size_t j = 0; j < (size_t)rows * d; j++) sbad += memcmp(&y0[j], &y1[j], 4) != 0;
        stot += (long)rows * d;
    }
    printf("layer_norm stress: %ld of %ld values differ (%ld rows)\n", sbad, stot, nrows);
    return bad != 0 || sbad != 0;
}
