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
 *    preprocessed on a worker thread, then delivered unchanged. In DES-3.3 the
 *    verdict of fpgate-accel decides between that and a substitute.
 *
 * While an image is being processed the pair is "busy": the owner must stop
 * reading from the server (bytes already read are kept behind the image, in
 * order) and must not close the pair. When the worker is done,
 * inspect_collect() hands the bytes to forward through a callback.
 *
 * Held images are bounded per image and globally, and zeroed when released.
 */

typedef enum { POLICY_ALLOW = 0, POLICY_BLOCK = 1 } inspect_policy_t;

typedef struct {
    int enabled;                /* 0: pure forwarding, no HTTP parsing at all */
    size_t max_bytes;           /* largest image analysed; bigger ones stream */
    size_t budget_bytes;        /* total held across all connections */
    prep_cfg_t prep;
    int workers;                /* decode threads; 0 decodes inline in the event loop */
    unsigned delay_ms;          /* test hook: extra delay per decode job */
    int verbose;                /* log the path and Referer of every response, not only images */
    int classify;               /* 1: ask fpgate-accel and act on its answer; 0: only analyse */
    float threshold;            /* score_nsfw >= threshold is replaced */
    inspect_policy_t on_failure;    /* accelerator down, slow, erroring, or everything queued */
    inspect_policy_t on_unexamined; /* decodes fine but cannot be judged: GIF, AVIF, too many pixels */
    inspect_policy_t on_corrupt;    /* damaged or truncated image */
} inspect_cfg_t;

typedef struct {
    unsigned long seen;         /* image responses held for analysis */
    unsigned long decoded;
    unsigned long corrupt, unsupported_format, too_many_pixels, not_an_image;
    unsigned long skipped_oversize, skipped_budget, skipped_encoded, skipped_partial, skipped_busy;
    unsigned long accept_rewritten;   /* requests whose Accept lost its image/avif entry */
    unsigned long allowed, blocked;
    unsigned long blocked_nsfw, blocked_failure, blocked_corrupt, blocked_unexamined;
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
/* Opaque pointer handed back to the inspect_collect() callback. */
void inspect_set_ctx(inspect_t *i, void *ctx);

/*
 * Plain-HTTP mode only: remembers what was asked (path and Referer, query strings
 * dropped) so the log can say which page an image belongs to.
 */
void inspect_note_request(inspect_t *i, const char *path, size_t path_len, const char *referer,
                          size_t referer_len);

/* Bytes going client -> server. Forwardable bytes are appended to `out`. */
int inspect_request(inspect_t *i, const char *p, size_t n, buf_t *out);
/* Bytes going server -> client. A held image produces no output until complete. */
int inspect_response(inspect_t *i, const char *p, size_t n, buf_t *out);
/* The server closed. Releases whatever is held. -1 if it cut a message short. */
int inspect_response_eof(inspect_t *i, buf_t *out);
/* 1 when the client->server direction must also pass through inspect_request(). */
int inspect_wants_requests(const inspect_t *i);
/* 1 while an image of this pair is being decoded on a worker. */
int inspect_busy(const inspect_t *i);

/*
 * Where the bytes of a pair that was waiting for a worker or the accelerator go.
 * Set once at start-up; it is also used when the accelerator answers.
 */
typedef void (*inspect_ready_fn)(void *ctx, buf_t *out, int error);
void inspect_set_ready(inspect_ready_fn ready);

/*
 * Collects finished decode jobs. For each one whose pair is still alive,
 * `ready` receives the bytes to forward to the client (the released image and
 * whatever queued behind it) and `error` is non-zero if the pair must be
 * closed. Call when the pool's eventfd is readable.
 */
void inspect_collect(inspect_ready_fn ready);

#endif
