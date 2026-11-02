/*
 * prep_tool FILE SIZE crop|squash DCT OUT.rgb
 *
 * Runs the proxy's image preprocessing on a file and writes the resulting
 * SIZE x SIZE x 3 bytes. Used by test_prep.py to compare against Pillow.
 * Exit code: 0 ok, 3 corrupt, 4 unsupported, 5 too large, 6 not an image.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "image_detect.h"
#include "image_prep.h"

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: %s FILE SIZE crop|squash DCT OUT.rgb\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "read error\n");
        return 2;
    }
    fclose(f);

    prep_cfg_t cfg = {
        .size = atoi(argv[2]),
        .center_crop = strcmp(argv[3], "crop") == 0,
        .dct_scaling = atoi(argv[4]),
        .max_pixels = getenv("PREP_MAX_PIXELS") ? strtoul(getenv("PREP_MAX_PIXELS"), NULL, 10)
                                                : 16000000ul,
    };
    img_kind_t kind = image_sniff(buf, (size_t)n);
    if (kind != IMG_JPEG && kind != IMG_PNG) {
        fprintf(stderr, "not a decodable image (%s)\n", image_kind_name(kind));
        free(buf);
        return 6;
    }
    struct timespec t0, t1;
    prep_result_t r;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    prep_status_t st = image_prep(kind, buf, (size_t)n, &cfg, &r);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (st != PREP_OK) {
        fprintf(stderr, "%s\n", prep_status_name(st));
        free(buf);
        return st == PREP_CORRUPT ? 3 : st == PREP_UNSUPPORTED ? 4 : st == PREP_TOO_LARGE ? 5 : 7;
    }
    FILE *o = fopen(argv[5], "wb");
    if (!o || fwrite(r.rgb, 1, r.rgb_len, o) != r.rgb_len) {
        fprintf(stderr, "write error\n");
        return 2;
    }
    fclose(o);
    printf("ok %dx%d %.1fms\n", r.src_w, r.src_h,
           (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
    prep_result_free(&r);
    free(buf);
    return 0;
}
