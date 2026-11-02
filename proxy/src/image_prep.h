#ifndef FPGATE_IMAGE_PREP_H
#define FPGATE_IMAGE_PREP_H

#include <stddef.h>
#include <stdint.h>

#include "image_detect.h"

/*
 * Decodes a JPEG or PNG and produces what fpgate-accel expects: size x size
 * RGB, uint8, row-major (see accel_proto.h). Normalisation and quantisation
 * are the accelerator's job.
 *
 * The pipeline mirrors PIL (Image.open(...).convert("RGB") followed by the
 * timm evaluation transform), including: alpha dropped without compositing,
 * gamma/ICC/EXIF orientation ignored, 16-bit PNG truncated to 8 bits.
 */

typedef struct {
    int size;                   /* output side in pixels (224, 336, 448...) */
    int center_crop;            /* 1: resize shorter side + centre crop; 0: squash */
    int dct_scaling;            /* 1: let libjpeg decode at 1/2, 1/4, 1/8 when the image is
                                 *    much larger than needed (faster, not bit-exact with PIL) */
    unsigned long max_pixels;   /* decompression-bomb guard */
} prep_cfg_t;

typedef enum {
    PREP_OK = 0,
    PREP_CORRUPT,               /* damaged or truncated data */
    PREP_UNSUPPORTED,           /* valid image we cannot convert (CMYK, 12-bit...) */
    PREP_TOO_LARGE,             /* more than cfg->max_pixels */
    PREP_NOMEM,
} prep_status_t;

typedef struct {
    uint8_t *rgb;               /* size * size * 3, caller releases with prep_result_free */
    size_t rgb_len;
    int src_w, src_h;           /* dimensions of the original image */
} prep_result_t;

prep_status_t image_prep(img_kind_t kind, const uint8_t *data, size_t len,
                         const prep_cfg_t *cfg, prep_result_t *out);
/* Zeroes the pixels before freeing them (RS-01). */
void prep_result_free(prep_result_t *r);
const char *prep_status_name(prep_status_t s);

#endif
