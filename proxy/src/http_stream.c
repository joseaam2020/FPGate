#define _GNU_SOURCE
#include "http_stream.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define HEAD_MAX   32768
#define MAX_HDRS   100
#define TRAILER_MAX 8192

enum {
    S_HEAD,
    S_LEN,
    S_CHUNK_SIZE,
    S_CHUNK_EXT,
    S_CHUNK_DATA,
    S_CHUNK_CRLF,
    S_TRAILER,
    S_CLOSE,
    S_OPAQUE,
};

struct hs_stream {
    int is_request;
    hs_pair_t *pair;
    hs_callbacks_t cb;
    void *user;
    int state;
    char *head;                 /* only used while a head arrives in pieces */
    size_t head_len, head_cap;
    uint64_t remaining;         /* S_LEN, S_CHUNK_DATA */
    uint64_t chunk_size;
    int chunk_digits;
    int cr;                     /* a CR was just seen on a framing line */
    int crlf_pos;               /* S_CHUNK_CRLF: 0 expects CR, 1 expects LF */
    size_t line_len;            /* S_TRAILER */
    size_t trailer_total;
};

typedef struct {
    const char *method, *path, *msg;
    size_t method_len, path_len, msg_len;
    int minor, status;
    struct phr_header h[MAX_HDRS];
    size_t nh;
} parsed_t;

void hs_pair_init(hs_pair_t *p) { memset(p, 0, sizeof *p); }

void hs_pair_push(hs_pair_t *p, int kind)
{
    if (p->count >= HS_QUEUE) {
        p->overflow = 1;
        return;
    }
    p->q[(p->head + p->count) % HS_QUEUE] = (uint8_t)kind;
    p->count++;
}

static int pair_pop(hs_pair_t *p)
{
    if (!p->count)
        return HS_METHOD_OTHER;
    int k = p->q[p->head];
    p->head = (p->head + 1) % HS_QUEUE;
    p->count--;
    return k;
}

hs_stream_t *hs_new(int is_request, hs_pair_t *pair, const hs_callbacks_t *cb, void *user)
{
    hs_stream_t *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    s->is_request = is_request;
    s->pair = pair;
    s->cb = *cb;
    s->user = user;
    s->state = S_HEAD;
    return s;
}

void hs_free(hs_stream_t *s)
{
    if (!s)
        return;
    free(s->head);
    free(s);
}

/* ---- header interpretation -------------------------------------------- */

static int hname_is(const struct phr_header *h, const char *name)
{
    return h->name_len == strlen(name) && strncasecmp(h->name, name, h->name_len) == 0;
}

/* Parses a Content-Length value: digits only, surrounding blanks allowed. */
static int parse_clen(const struct phr_header *h, uint64_t *out)
{
    const char *v = h->value;
    size_t n = h->value_len;
    while (n && (*v == ' ' || *v == '\t')) {
        v++;
        n--;
    }
    while (n && (v[n - 1] == ' ' || v[n - 1] == '\t'))
        n--;
    if (n == 0 || n > 18)
        return -1;
    uint64_t x = 0;
    for (size_t i = 0; i < n; i++) {
        if (v[i] < '0' || v[i] > '9')
            return -1;
        x = x * 10 + (uint64_t)(v[i] - '0');
    }
    *out = x;
    return 0;
}

/* True when the last coding of a Transfer-Encoding value is "chunked". */
static int te_ends_chunked(const struct phr_header *h)
{
    const char *v = h->value;
    size_t n = h->value_len;
    while (n && (v[n - 1] == ' ' || v[n - 1] == '\t'))
        n--;
    size_t i = n;
    while (i && v[i - 1] != ',' && v[i - 1] != ' ' && v[i - 1] != '\t')
        i--;
    return n - i == 7 && strncasecmp(v + i, "chunked", 7) == 0;
}

/*
 * Decides how the body is delimited. `no_body` is set by the caller for
 * responses that cannot have one (1xx, 204, 304, answers to HEAD).
 */
static int frame_message(int is_request, const parsed_t *pt, int no_body, hs_body_t *kind,
                         uint64_t *clen)
{
    int ncl = 0, nte = 0, chunked = 0;
    uint64_t cl = 0;

    for (size_t i = 0; i < pt->nh; i++) {
        const struct phr_header *h = &pt->h[i];
        if (!h->name)                   /* obsolete line folding continuation */
            continue;
        if (hname_is(h, "content-length")) {
            uint64_t v;
            if (parse_clen(h, &v) != 0)
                return -1;
            if (ncl && v != cl)
                return -1;              /* conflicting lengths */
            cl = v;
            ncl++;
        } else if (hname_is(h, "transfer-encoding")) {
            nte++;
            chunked = te_ends_chunked(h);
        }
    }
    if (nte > 1 || (nte && ncl))
        return -1;
    *clen = cl;
    if (no_body) {
        *kind = HS_BODY_NONE;
        return 0;
    }
    if (nte) {
        if (chunked) {
            *kind = HS_BODY_CHUNKED;
            return 0;
        }
        if (is_request)
            return -1;                  /* a request body must be delimited */
        *kind = HS_BODY_CLOSE;
        return 0;
    }
    if (ncl) {
        *kind = cl ? HS_BODY_LENGTH : HS_BODY_NONE;
        return 0;
    }
    *kind = is_request ? HS_BODY_NONE : HS_BODY_CLOSE;
    return 0;
}

static void message_done(hs_stream_t *s)
{
    s->state = S_HEAD;
    s->cb.end(s->user);
}

/* A complete head is available. Returns 0, or -1 on a framing error. */
static int process_head(hs_stream_t *s, const parsed_t *pt, const char *raw, size_t raw_len)
{
    hs_head_t h;
    memset(&h, 0, sizeof h);
    h.is_request = s->is_request;
    h.hdrs = pt->h;
    h.nhdrs = pt->nh;
    h.raw = raw;
    h.raw_len = raw_len;
    h.minor = pt->minor;

    int no_body = 0, opaque_after = 0;
    if (s->is_request) {
        h.method = pt->method;
        h.method_len = pt->method_len;
        int kind = HS_METHOD_OTHER;
        if (pt->method_len == 4 && memcmp(pt->method, "HEAD", 4) == 0)
            kind = HS_METHOD_HEAD;
        else if (pt->method_len == 7 && memcmp(pt->method, "CONNECT", 7) == 0)
            kind = HS_METHOD_CONNECT;
        hs_pair_push(s->pair, kind);
    } else {
        h.status = pt->status;
        if (h.status >= 100 && h.status < 200) {
            no_body = 1;
            if (h.status == 101) {
                (void)pair_pop(s->pair);
                s->pair->upgraded = 1;
                opaque_after = 1;
            }
        } else {
            int kind = pair_pop(s->pair);
            if (kind == HS_METHOD_HEAD || h.status == 204 || h.status == 304)
                no_body = 1;
            if (kind == HS_METHOD_CONNECT && h.status >= 200 && h.status < 300) {
                s->pair->upgraded = 1;
                opaque_after = 1;
            }
        }
    }
    if (frame_message(s->is_request, pt, no_body || opaque_after, &h.body, &h.content_length) != 0)
        return -1;

    s->cb.head(s->user, &h);
    if (opaque_after) {
        s->state = S_OPAQUE;
        return 0;
    }
    switch (h.body) {
    case HS_BODY_NONE:
        message_done(s);
        break;
    case HS_BODY_LENGTH:
        s->remaining = h.content_length;
        s->state = S_LEN;
        break;
    case HS_BODY_CHUNKED:
        s->state = S_CHUNK_SIZE;
        s->chunk_size = 0;
        s->chunk_digits = 0;
        s->cr = 0;
        break;
    default:
        s->state = S_CLOSE;
        break;
    }
    return 0;
}

static int try_parse(const hs_stream_t *s, const char *b, size_t len, size_t last, parsed_t *pt)
{
    pt->nh = MAX_HDRS;
    if (s->is_request)
        return phr_parse_request(b, len, &pt->method, &pt->method_len, &pt->path, &pt->path_len,
                                 &pt->minor, pt->h, &pt->nh, last);
    return phr_parse_response(b, len, &pt->minor, &pt->status, &pt->msg, &pt->msg_len, pt->h,
                              &pt->nh, last);
}

/* ---- the feed loop ------------------------------------------------------ */

static int hexval(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

int hs_feed(hs_stream_t *s, const char *p, size_t n)
{
    while (n > 0) {
        if ((s->pair->upgraded || s->pair->overflow) && s->state != S_OPAQUE)
            s->state = S_OPAQUE;

        switch (s->state) {
        case S_OPAQUE:
            s->cb.opaque(s->user, p, n);
            return 0;

        case S_HEAD: {
            parsed_t *pt = malloc(sizeof *pt);
            if (!pt)
                return -1;
            int rc = 0;
            if (s->head_len == 0) {
                int r = try_parse(s, p, n, 0, pt);
                if (r > 0) {
                    rc = process_head(s, pt, p, (size_t)r);
                    p += r;
                    n -= (size_t)r;
                } else if (r == -2) {
                    if (n >= HEAD_MAX) {
                        rc = -1;
                    } else {
                        s->head_cap = n < 1024 ? 1024 : n * 2;
                        char *nb = realloc(s->head, s->head_cap);
                        if (!nb) {
                            rc = -1;
                        } else {
                            s->head = nb;
                            memcpy(s->head, p, n);
                            s->head_len = n;
                            p += n;
                            n = 0;
                        }
                    }
                } else {
                    rc = -1;
                }
            } else {
                size_t room = HEAD_MAX - s->head_len;
                size_t take = n < room ? n : room;
                if (s->head_len + take > s->head_cap) {
                    size_t nc = s->head_cap * 2;
                    while (nc < s->head_len + take)
                        nc *= 2;
                    if (nc > HEAD_MAX)
                        nc = HEAD_MAX;
                    char *nb = realloc(s->head, nc);
                    if (!nb) {
                        free(pt);
                        return -1;
                    }
                    s->head = nb;
                    s->head_cap = nc;
                }
                memcpy(s->head + s->head_len, p, take);
                size_t prev = s->head_len;
                int r = try_parse(s, s->head, prev + take, prev, pt);
                if (r > 0) {
                    rc = process_head(s, pt, s->head, (size_t)r);
                    p += (size_t)r - prev;
                    n -= (size_t)r - prev;
                    s->head_len = 0;
                } else if (r == -2) {
                    s->head_len = prev + take;
                    p += take;
                    n -= take;
                    if (s->head_len >= HEAD_MAX)
                        rc = -1;
                } else {
                    rc = -1;
                }
            }
            free(pt);
            if (rc != 0)
                return -1;
            break;
        }

        case S_LEN: {
            size_t take = n < s->remaining ? n : (size_t)s->remaining;
            s->cb.body(s->user, p, take, p, take);
            p += take;
            n -= take;
            s->remaining -= take;
            if (s->remaining == 0)
                message_done(s);
            break;
        }

        case S_CLOSE:
            s->cb.body(s->user, p, n, p, n);
            return 0;

        case S_CHUNK_DATA: {
            size_t take = n < s->remaining ? n : (size_t)s->remaining;
            s->cb.body(s->user, p, take, p, take);
            p += take;
            n -= take;
            s->remaining -= take;
            if (s->remaining == 0) {
                s->state = S_CHUNK_CRLF;
                s->crlf_pos = 0;
            }
            break;
        }

        default: {
            /* Framing bytes (chunk sizes, CRLFs, trailers): handled one by one and
             * handed over as a single raw-only span. */
            const char *start = p;
            while (n > 0 && s->state != S_CHUNK_DATA && s->state != S_HEAD) {
                char c = *p++;
                n--;
                switch (s->state) {
                case S_CHUNK_SIZE:
                case S_CHUNK_EXT:
                    if (s->cr) {
                        if (c != '\n')
                            return -1;
                        s->cr = 0;
                        if (s->chunk_size == 0) {
                            s->state = S_TRAILER;
                            s->line_len = 0;
                            s->trailer_total = 0;
                        } else {
                            s->remaining = s->chunk_size;
                            s->state = S_CHUNK_DATA;
                        }
                    } else if (c == '\r') {
                        if (s->chunk_digits == 0)
                            return -1;
                        s->cr = 1;
                    } else if (s->state == S_CHUNK_SIZE) {
                        int v = hexval(c);
                        if (v >= 0) {
                            if (++s->chunk_digits > 12)
                                return -1;
                            s->chunk_size = s->chunk_size * 16 + (uint64_t)v;
                        } else if (c == ';' && s->chunk_digits > 0) {
                            s->state = S_CHUNK_EXT;
                        } else {
                            return -1;
                        }
                    } else if (c == '\n') {
                        return -1;
                    }
                    break;
                case S_CHUNK_CRLF:
                    if (s->crlf_pos == 0 ? c != '\r' : c != '\n')
                        return -1;
                    if (++s->crlf_pos == 2) {
                        s->state = S_CHUNK_SIZE;
                        s->chunk_size = 0;
                        s->chunk_digits = 0;
                        s->cr = 0;
                    }
                    break;
                case S_TRAILER:
                    if (++s->trailer_total > TRAILER_MAX)
                        return -1;
                    if (c == '\n') {
                        if (s->line_len == 0) {
                            /* flush the raw span before announcing the end */
                            s->cb.body(s->user, start, (size_t)(p - start), NULL, 0);
                            start = p;
                            message_done(s);
                        } else {
                            s->line_len = 0;
                        }
                    } else if (c != '\r') {
                        s->line_len++;
                    }
                    break;
                default:
                    return -1;
                }
            }
            if (p > start)
                s->cb.body(s->user, start, (size_t)(p - start), NULL, 0);
            break;
        }
        }
    }
    return 0;
}

int hs_eof(hs_stream_t *s)
{
    switch (s->state) {
    case S_HEAD:
        return s->head_len == 0 ? 0 : -1;
    case S_CLOSE:
        message_done(s);
        return 0;
    case S_OPAQUE:
        return 0;
    default:
        return -1;
    }
}
