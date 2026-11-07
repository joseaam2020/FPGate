#define _GNU_SOURCE
#include "substitute.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

#include "image_detect.h"
#include "image_prep.h"

#define MAX_SIDE      800
#define MIN_SIDE      16
#define UNKNOWN_SIDE  256
#define CUSTOM_MAX    (2u * 1024 * 1024)

static uint8_t *g_custom;
static size_t g_custom_len;
static const char *g_custom_type;

void substitute_free(void)
{
    free(g_custom);
    g_custom = NULL;
    g_custom_len = 0;
}

int substitute_load(const char *path, char *err, size_t err_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, err_len, "cannot open %s", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || (size_t)n > CUSTOM_MAX) {
        fclose(f);
        snprintf(err, err_len, "%s must be between 1 byte and %u KiB", path, CUSTOM_MAX / 1024);
        return -1;
    }
    uint8_t *data = malloc((size_t)n);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(data);
        snprintf(err, err_len, "cannot read %s", path);
        return -1;
    }
    fclose(f);

    img_kind_t kind = image_sniff(data, (size_t)n);
    const char *type = kind == IMG_JPEG ? "image/jpeg" : kind == IMG_PNG ? "image/png"
                       : kind == IMG_WEBP ? "image/webp" : NULL;
    if (!type) {
        free(data);
        snprintf(err, err_len, "%s is not a JPEG, PNG or WebP image", path);
        return -1;
    }
    prep_cfg_t cfg = { .size = 64, .center_crop = 1, .dct_scaling = 0, .max_pixels = 16000000ul };
    prep_result_t r;
    prep_status_t st = image_prep(kind, data, (size_t)n, &cfg, &r);
    if (st != PREP_OK) {
        free(data);
        snprintf(err, err_len, "%s does not decode (%s)", path, prep_status_name(st));
        return -1;
    }
    prep_result_free(&r);

    substitute_free();
    g_custom = data;
    g_custom_len = (size_t)n;
    g_custom_type = type;
    return 0;
}

/* ---- the generated placeholder ------------------------------------------------ */

static void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static int put_chunk(buf_t *b, const char tag[4], const void *data, size_t n)
{
    uint8_t hdr[8];
    put_be32(hdr, (uint32_t)n);
    memcpy(hdr + 4, tag, 4);
    uLong crc = crc32(0L, (const Bytef *)tag, 4);
    if (n)
        crc = crc32(crc, (const Bytef *)data, (uInt)n);
    uint8_t tail[4];
    put_be32(tail, (uint32_t)crc);
    return buf_append(b, hdr, 8) || (n && buf_append(b, data, n)) || buf_append(b, tail, 4) ? -1 : 0;
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* A grey picture with a border and a cross: obviously a stand-in, tiny once compressed. */
static int make_png(buf_t *png, int w, int h)
{
    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    uint8_t ihdr[13];
    put_be32(ihdr, (uint32_t)w);
    put_be32(ihdr + 4, (uint32_t)h);
    ihdr[8] = 8;        /* bit depth */
    ihdr[9] = 0;        /* grayscale */
    ihdr[10] = ihdr[11] = ihdr[12] = 0;

    z_stream zs;
    memset(&zs, 0, sizeof zs);
    if (deflateInit(&zs, 6) != Z_OK)
        return -1;
    uint8_t *row = malloc((size_t)w + 1);
    buf_t z = {0};
    uint8_t out[4096];
    int rc = -1;
    if (!row)
        goto done;

    for (int y = 0; y < h; y++) {
        row[0] = 0;                                         /* filter: none */
        int edge_y = y < 2 || y >= h - 2;
        for (int x = 0; x < w; x++) {
            int v = 0xC8;
            if (edge_y || x < 2 || x >= w - 2)
                v = 0x70;
            else {
                long d1 = (long)x * h - (long)y * w;        /* distance to the diagonals, scaled */
                long d2 = (long)(w - 1 - x) * h - (long)y * w;
                long tol = (long)(w > h ? w : h) * 2;
                if ((d1 < 0 ? -d1 : d1) < tol || (d2 < 0 ? -d2 : d2) < tol)
                    v = 0x90;
            }
            row[x + 1] = (uint8_t)v;
        }
        zs.next_in = row;
        zs.avail_in = (uInt)w + 1;
        do {
            zs.next_out = out;
            zs.avail_out = sizeof out;
            deflate(&zs, Z_NO_FLUSH);
            if (buf_append(&z, out, sizeof out - zs.avail_out) != 0)
                goto done;
        } while (zs.avail_out == 0);
    }
    int ret;
    do {
        zs.next_out = out;
        zs.avail_out = sizeof out;
        ret = deflate(&zs, Z_FINISH);
        if (buf_append(&z, out, sizeof out - zs.avail_out) != 0)
            goto done;
    } while (ret != Z_STREAM_END);

    if (buf_append(png, sig, sizeof sig) == 0 && put_chunk(png, "IHDR", ihdr, sizeof ihdr) == 0 &&
        put_chunk(png, "IDAT", z.p, z.len) == 0 && put_chunk(png, "IEND", NULL, 0) == 0)
        rc = 0;
done:
    deflateEnd(&zs);
    free(row);
    buf_free(&z);
    return rc;
}

int substitute_response(buf_t *out, int width, int height, int close_connection, const char *reason)
{
    buf_t png = {0};
    const void *body;
    size_t body_len;
    const char *type;

    if (g_custom) {
        body = g_custom;
        body_len = g_custom_len;
        type = g_custom_type;
    } else {
        int w = width > 0 ? width : UNKNOWN_SIDE;
        int h = height > 0 ? height : UNKNOWN_SIDE;
        if (w > MAX_SIDE || h > MAX_SIDE) {         /* shrink both sides together: keep the shape */
            int big = w > h ? w : h;
            w = (int)((long long)w * MAX_SIDE / big);
            h = (int)((long long)h * MAX_SIDE / big);
        }
        w = clampi(w, MIN_SIDE, MAX_SIDE);
        h = clampi(h, MIN_SIDE, MAX_SIDE);
        if (make_png(&png, w, h) != 0) {
            buf_free(&png);
            return -1;
        }
        body = png.p;
        body_len = png.len;
        type = "image/png";
    }
    int rc = buf_printf(out,
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %zu\r\n"
                        "Cache-Control: no-store\r\n"
                        "X-FPGate-Block: %s\r\n"
                        "%s\r\n",
                        type, body_len, reason, close_connection ? "Connection: close\r\n" : "");
    if (rc == 0)
        rc = buf_append(out, body, body_len);
    buf_free(&png);
    return rc;
}
