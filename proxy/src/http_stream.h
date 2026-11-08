#ifndef FPGATE_HTTP_STREAM_H
#define FPGATE_HTTP_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "picohttpparser.h"

/*
 * Incremental HTTP/1.1 message framer. It does not change the bytes: it tells
 * the owner where messages start and end and hands over the decoded payload,
 * so that images can be recognised inside keep-alive connections, chunked
 * bodies and pipelined requests.
 *
 * One stream per direction. The response stream needs to know the method of
 * the matching request (HEAD responses have no body), which they share
 * through an hs_pair_t.
 *
 * Anything ambiguous or malformed (both Content-Length and Transfer-Encoding,
 * conflicting lengths, bad chunk sizes) is an error: the owner closes the
 * connection rather than guess, because a parser that disagrees with the
 * browser is a way to smuggle an image past the filter.
 */

typedef enum {
    HS_BODY_NONE,
    HS_BODY_LENGTH,
    HS_BODY_CHUNKED,
    HS_BODY_CLOSE,          /* until the connection closes */
} hs_body_t;

typedef struct {
    int is_request;
    const char *method;     /* requests */
    const char *path;
    size_t path_len;
    size_t method_len;
    int status;             /* responses */
    int minor;
    const struct phr_header *hdrs;
    size_t nhdrs;
    const char *raw;        /* the head exactly as received, including the final CRLF CRLF */
    size_t raw_len;
    hs_body_t body;
    uint64_t content_length;    /* valid when body == HS_BODY_LENGTH */
} hs_head_t;

typedef struct {
    void (*head)(void *user, const hs_head_t *h);
    /*
     * Body bytes in order. `raw` is what travelled on the wire (including chunk
     * framing and trailers); `data` is the decoded payload part of it and may
     * be empty. The raw spans of all calls rebuild the body exactly.
     */
    void (*body)(void *user, const char *raw, size_t raw_len, const char *data, size_t data_len);
    void (*end)(void *user);                            /* message complete */
    void (*opaque)(void *user, const char *p, size_t n); /* after an Upgrade or CONNECT: not HTTP */
} hs_callbacks_t;

#define HS_QUEUE 32

typedef struct {
    uint8_t q[HS_QUEUE];    /* pending request methods, oldest first */
    unsigned head, count;
    int overflow;
    int upgraded;           /* a 101 or a CONNECT 2xx turned both directions opaque */
} hs_pair_t;

enum { HS_METHOD_OTHER = 0, HS_METHOD_HEAD = 1, HS_METHOD_CONNECT = 2 };

void hs_pair_init(hs_pair_t *p);
/* Registers a request that is not going to be parsed (plain-HTTP mode). */
void hs_pair_push(hs_pair_t *p, int method_kind);

typedef struct hs_stream hs_stream_t;

hs_stream_t *hs_new(int is_request, hs_pair_t *pair, const hs_callbacks_t *cb, void *user);
void hs_free(hs_stream_t *s);
/* Returns 0, or -1 on a framing error (close the connection). */
int hs_feed(hs_stream_t *s, const char *p, size_t n);
/* The sender closed. Returns 0 when that is a clean end, -1 when it cut a message short. */
int hs_eof(hs_stream_t *s);

/*
 * Pause: called from the `end` callback, it makes hs_feed() return right after
 * that message without consuming the rest of its input. hs_rest() then gives
 * the unconsumed part (valid until the caller's buffer goes away); feed it
 * again after hs_resume().
 */
void hs_stop(hs_stream_t *s);
void hs_resume(hs_stream_t *s);
size_t hs_rest(const hs_stream_t *s, const char **p);

#endif
