#define _GNU_SOURCE
#include "resample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PRECISION_BITS (32 - 8 - 2)     /* same as Pillow's Resample.c */
#define MAX_DIM        65535

struct resizer {
    int in_w, in_h, size;
    int ow, oh;                 /* output window (size x size) */
    int ksize_h, ksize_v;
    int *hb, *vb;               /* [start, count] per output column / row */
    int32_t *hk, *vk;           /* fixed-point coefficients */
    uint8_t *tmp;               /* in_h rows of ow RGB pixels */
    int rows;
};

static double bicubic(double x)
{
    const double a = -0.5;
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/*
 * Pillow's precompute_coeffs + normalize_coeffs_8bpc for the output indices
 * [first, first + count) of a conceptual resize from in_size to new_size.
 */
static int make_coeffs(int in_size, int new_size, int first, int count,
                       int **bounds, int32_t **kk, int *ksize_out)
{
    double scale = (double)in_size / new_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = 2.0 * filterscale;
    int ksize = (int)ceil(support) * 2 + 1;

    *bounds = malloc(sizeof(int) * 2 * (size_t)count);
    *kk = malloc(sizeof(int32_t) * (size_t)ksize * (size_t)count);
    double *tmp = malloc(sizeof(double) * (size_t)ksize);
    if (!*bounds || !*kk || !tmp) {
        free(*bounds);
        free(*kk);
        free(tmp);
        return -1;
    }

    for (int i = 0; i < count; i++) {
        double center = ((first + i) + 0.5) * scale;
        double ww = 0.0, ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        int xmax = (int)(center + support + 0.5);
        if (xmin < 0)
            xmin = 0;
        if (xmax > in_size)
            xmax = in_size;
        xmax -= xmin;
        if (xmax < 0)
            xmax = 0;
        if (xmax > ksize)
            xmax = ksize;

        int x;
        for (x = 0; x < xmax; x++) {
            double w = bicubic((x + xmin - center + 0.5) * ss);
            tmp[x] = w;
            ww += w;
        }
        int32_t *k = *kk + (size_t)i * (size_t)ksize;
        for (x = 0; x < xmax; x++) {
            double v = ww != 0.0 ? tmp[x] / ww : tmp[x];
            k[x] = (int32_t)(v < 0 ? v * (1 << PRECISION_BITS) - 0.5
                                   : v * (1 << PRECISION_BITS) + 0.5);
        }
        for (; x < ksize; x++)
            k[x] = 0;
        (*bounds)[i * 2] = xmin;
        (*bounds)[i * 2 + 1] = xmax;
    }
    free(tmp);
    *ksize_out = ksize;
    return 0;
}

static inline uint8_t clip8(int32_t v)
{
    v >>= PRECISION_BITS;
    return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

resizer_t *rs_new(int in_w, int in_h, int size, int center_crop)
{
    if (in_w < 1 || in_h < 1 || in_w > MAX_DIM || in_h > MAX_DIM || size < 1 || size > 4096)
        return NULL;

    int new_w = size, new_h = size, x0 = 0, y0 = 0;
    if (center_crop) {
        /* torchvision: shorter side -> size, longer = int(size * long / short) */
        if (in_w <= in_h) {
            new_h = (int)((long long)size * in_h / in_w);
        } else {
            new_w = (int)((long long)size * in_w / in_h);
        }
        /* CenterCrop: int(round((dim - size) / 2.0)), Python rounds half to even */
        x0 = (int)lrint((new_w - size) / 2.0);
        y0 = (int)lrint((new_h - size) / 2.0);
    }

    resizer_t *r = calloc(1, sizeof *r);
    if (!r)
        return NULL;
    r->in_w = in_w;
    r->in_h = in_h;
    r->size = size;
    r->ow = r->oh = size;

    if (make_coeffs(in_w, new_w, x0, r->ow, &r->hb, &r->hk, &r->ksize_h) != 0 ||
        make_coeffs(in_h, new_h, y0, r->oh, &r->vb, &r->vk, &r->ksize_v) != 0)
        goto fail;
    r->tmp = malloc((size_t)in_h * (size_t)r->ow * 3);
    if (!r->tmp)
        goto fail;
    return r;

fail:
    rs_free(r);
    return NULL;
}

void rs_push_row(resizer_t *r, const uint8_t *row)
{
    if (r->rows >= r->in_h)
        return;
    uint8_t *dst = r->tmp + (size_t)r->rows * (size_t)r->ow * 3;
    for (int i = 0; i < r->ow; i++) {
        const int32_t *k = r->hk + (size_t)i * (size_t)r->ksize_h;
        const uint8_t *s = row + (size_t)r->hb[i * 2] * 3;
        int n = r->hb[i * 2 + 1];
        int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
        for (int x = 0; x < n; x++, s += 3) {
            s0 += s[0] * k[x];
            s1 += s[1] * k[x];
            s2 += s[2] * k[x];
        }
        dst[i * 3] = clip8(s0);
        dst[i * 3 + 1] = clip8(s1);
        dst[i * 3 + 2] = clip8(s2);
    }
    r->rows++;
}

int rs_finish(resizer_t *r, uint8_t *out)
{
    if (r->rows != r->in_h)
        return -1;
    size_t rb = (size_t)r->ow * 3;
    int32_t *acc = malloc(rb * sizeof *acc);
    if (!acc)
        return -1;
    for (int oy = 0; oy < r->oh; oy++) {
        const int32_t *k = r->vk + (size_t)oy * (size_t)r->ksize_v;
        int y0 = r->vb[oy * 2], n = r->vb[oy * 2 + 1];
        for (size_t c = 0; c < rb; c++)
            acc[c] = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < n; y++) {
            const uint8_t *src = r->tmp + (size_t)(y0 + y) * rb;
            int32_t kv = k[y];
            for (size_t c = 0; c < rb; c++)
                acc[c] += src[c] * kv;
        }
        uint8_t *dst = out + (size_t)oy * rb;
        for (size_t c = 0; c < rb; c++)
            dst[c] = clip8(acc[c]);
    }
    free(acc);
    return 0;
}

void rs_free(resizer_t *r)
{
    if (!r)
        return;
    if (r->tmp)
        explicit_bzero(r->tmp, (size_t)r->in_h * (size_t)r->ow * 3);
    free(r->tmp);
    free(r->hb);
    free(r->hk);
    free(r->vb);
    free(r->vk);
    free(r);
}
