#ifndef FPGATE_IMAGE_DETECT_H
#define FPGATE_IMAGE_DETECT_H

#include <stddef.h>
#include <stdint.h>

/* What the bytes actually are, decided by magic numbers (they win over headers). */
typedef enum {
    IMG_UNKNOWN = 0,    /* not a recognised image */
    IMG_JPEG,
    IMG_PNG,
    IMG_OTHER,          /* recognised but not decodable here: GIF, WebP, AVIF, BMP... */
} img_kind_t;

/* What a Content-Type header suggests. */
typedef enum {
    CT_NOT_IMAGE = 0,   /* text, json, video, svg and anything else */
    CT_IMAGE,           /* any image type except svg */
    CT_AMBIGUOUS,       /* missing, application/octet-stream, binary/octet-stream */
} ct_class_t;

/* Number of leading body bytes image_sniff() needs to be conclusive. */
#define IMG_SNIFF_BYTES 12

ct_class_t image_classify_content_type(const char *value, size_t len);
img_kind_t image_sniff(const uint8_t *data, size_t len);
const char *image_kind_name(img_kind_t k);

#endif
