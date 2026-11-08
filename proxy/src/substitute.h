#ifndef FPGATE_SUBSTITUTE_H
#define FPGATE_SUBSTITUTE_H

#include "buf.h"

/*
 * What the browser gets instead of a blocked image: a fresh 200 response with
 * an image, never the original's headers (no ETag, no Content-Encoding, no
 * cache validators) and `Cache-Control: no-store`, so the replacement is not
 * remembered under the original URL.
 *
 * By default the image is a grey PNG with a cross, generated at the original's
 * dimensions (capped) so page layout does not jump. A custom JPEG/PNG/WebP can
 * be loaded instead; it is sent as it is.
 */

/* Loads and validates a custom image. Returns 0, or -1 with a message in err. */
int substitute_load(const char *path, char *err, size_t err_len);
void substitute_free(void);
/* Content-Type of the custom image in use, or NULL when the generated PNG is used. */
const char *substitute_custom_type(void);

/*
 * Appends a complete HTTP response to `out`. `width` and `height` are the
 * original's dimensions when known (0 otherwise). `reason` goes in the
 * X-FPGate-Block header. Returns 0, or -1 on out of memory.
 */
int substitute_response(buf_t *out, int width, int height, int close_connection, const char *reason);

#endif
