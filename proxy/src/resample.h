#ifndef FPGATE_RESAMPLE_H
#define FPGATE_RESAMPLE_H

#include <stdint.h>

/*
 * Streaming image resizer that reproduces, bit for bit, what the training
 * pipeline does with PIL:
 *
 *   crop mode  (timm eval transform: Resize(shorter side, BICUBIC) + CenterCrop)
 *   squash mode (Resize((size, size), BICUBIC))
 *
 * It follows Pillow's ImagingResample: bicubic (a = -0.5) with the support
 * widened by the downscale factor, 22-bit fixed-point coefficients, a
 * horizontal pass first and a vertical pass second, each rounded to 8 bits.
 * Rows are pushed as the decoder produces them, so the full-resolution image
 * is never held in memory: only in_height x size x 3 bytes.
 */

typedef struct resizer resizer_t;

/* Returns NULL on invalid geometry or out of memory. */
resizer_t *rs_new(int in_w, int in_h, int size, int center_crop);
/* `row` holds in_w RGB pixels. Rows must be pushed top to bottom. */
void rs_push_row(resizer_t *r, const uint8_t *row);
/* Writes size x size x 3 bytes. Returns 0, or -1 if rows are missing. */
int rs_finish(resizer_t *r, uint8_t *out);
void rs_free(resizer_t *r);

#endif
