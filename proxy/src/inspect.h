#ifndef FPGATE_INSPECT_H
#define FPGATE_INSPECT_H

#include <stddef.h>

#include "buf.h"
#include "image_prep.h"

/*
 * HTTP-aware relay for one client/server pair. Sits between the two sockets
 * (plain HTTP, or the decrypted side of an intercepted CONNECT):
 *
 *  - everything that is not an image streams through untouched and at once;
 *  - a 200 response that is an image is held back until complete, decoded and
 *    preprocessed, then delivered unchanged. In DES-3.3 the verdict of
 *    fpgate-accel decides between that and a substitute.
 *
 * Held images are bounded per image and globally, and zeroed when released.
 */

typedef struct {
    int enabled;                /* 0: pure forwarding, no HTTP parsing at all */
    size_t max_bytes;           /* largest image analysed; bigger ones stream */
    size_t budget_bytes;        /* total held across all connections */
    prep_cfg_t prep;
} inspect_cfg_t;

typedef struct {
    unsigned long seen;         /* image responses held for analysis */
    unsigned long decoded;
    unsigned long corrupt, unsupported_format, too_many_pixels, not_an_image;
    unsigned long skipped_oversize, skipped_budget, skipped_encoded, skipped_partial;
    double prep_ms_total, prep_ms_max;
} inspect_stats_t;

void inspect_configure(const inspect_cfg_t *cfg);
const inspect_cfg_t *inspect_config(void);
const inspect_stats_t *inspect_stats(void);

typedef struct inspect inspect_t;

/*
 * `host` is only used for logs. When `track_requests` is 0 the request side is
 * not parsed (plain-HTTP mode, where the proxy wrote the request itself) and
 * `first_method` (HS_METHOD_*) tells the response parser what was asked.
 * Returns NULL when inspection is disabled or memory is short.
 */
inspect_t *inspect_new(const char *host, int track_requests, int first_method);
void inspect_free(inspect_t *i);

/* Bytes going client -> server. Forwardable bytes are appended to `out`. */
int inspect_request(inspect_t *i, const char *p, size_t n, buf_t *out);
/* Bytes going server -> client. A held image produces no output until complete. */
int inspect_response(inspect_t *i, const char *p, size_t n, buf_t *out);
/* The server closed. Releases whatever is held. -1 if it cut a message short. */
int inspect_response_eof(inspect_t *i, buf_t *out);
/* 1 when the client->server direction must also pass through inspect_request(). */
int inspect_wants_requests(const inspect_t *i);

#endif
