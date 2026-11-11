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

int main(void)
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
    return bad != 0;
}
