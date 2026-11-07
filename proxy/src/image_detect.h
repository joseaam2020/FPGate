#ifndef FPGATE_IMAGE_DETECT_H
#define FPGATE_IMAGE_DETECT_H

#include <stddef.h>
#include <stdint.h>

/* What the bytes actually are, decided by magic numbers (they win over headers). */
typedef enum {
    IMG_UNKNOWN = 0,    /* not a recognised image */
    IMG_JPEG,
    IMG_PNG,
    IMG_WEBP,
    IMG_OTHER,          /* recognised but not decodable here: GIF, AVIF, BMP... */
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

/*
 * Copies an Accept header value without its image/avif entry, so that servers
 * negotiating the format fall back to WebP, which we can decode, instead of AVIF,
 * which we cannot. Everything else, including order and q-values, is kept.
 * Returns 1 and sets *outn when something was removed, 0 when the value has no
 * AVIF entry (out is then not meaningful), -1 if `cap` (needs n + 4) is too small.
 */
int accept_drop_avif(const char *v, size_t n, char *out, size_t cap, size_t *outn);

#endif
