#define _GNU_SOURCE
#include "inspect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "http_stream.h"
#include "image_detect.h"

static inspect_cfg_t g_cfg = {
    .enabled = 1,
    .max_bytes = 8u * 1024 * 1024,
    .budget_bytes = 64u * 1024 * 1024,
    .prep = { .size = 448, .center_crop = 1, .dct_scaling = 0, .max_pixels = 16000000ul },
};
static inspect_stats_t g_stats;
static size_t g_held;                   /* bytes currently held, all connections */

void inspect_configure(const inspect_cfg_t *cfg) { g_cfg = *cfg; }
const inspect_cfg_t *inspect_config(void) { return &g_cfg; }
const inspect_stats_t *inspect_stats(void) { return &g_stats; }

enum { M_STREAM, M_SNIFF, M_CAPTURE };

struct inspect {
    hs_pair_t pair;
    hs_stream_t *req, *resp;
    char host[256];
    buf_t *out;                         /* where forwardable bytes go during a feed */
    int oom;
    int mode;
    int chunked;
    buf_t head;                         /* held response head */
    buf_t braw;                         /* held body, as on the wire */
    buf_t bdata;                        /* held body, chunk framing removed (chunked only) */
    size_t held;
};

/* ---- held-memory accounting ---------------------------------------------- */

static void account(inspect_t *i)
{
    size_t now = i->head.len + i->braw.len + i->bdata.len;
    g_held += now - i->held;
    i->held = now;
}

static void drop_held(inspect_t *i)
{
    buf_wipe_free(&i->head);
    buf_wipe_free(&i->braw);
    buf_wipe_free(&i->bdata);
    g_held -= i->held;
    i->held = 0;
    i->mode = M_STREAM;
}

static void emit(inspect_t *i, const void *p, size_t n)
{
    if (n && buf_append(i->out, p, n) != 0)
        i->oom = 1;
}

/* Delivers what was held, unchanged, and goes back to streaming. */
static void release_held(inspect_t *i)
{
    emit(i, i->head.p, i->head.len);
    emit(i, i->braw.p, i->braw.len);
    drop_held(i);
}

/* ---- headers ---------------------------------------------------------------- */

static const struct phr_header *find_header(const hs_head_t *h, const char *name)
{
    size_t nl = strlen(name);
    for (size_t k = 0; k < h->nhdrs; k++)
        if (h->hdrs[k].name && h->hdrs[k].name_len == nl &&
            strncasecmp(h->hdrs[k].name, name, nl) == 0)
            return &h->hdrs[k];
    return NULL;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- the image path -------------------------------------------------------- */

static void analyse(inspect_t *i)
{
    const uint8_t *img = (const uint8_t *)(i->chunked ? i->bdata.p : i->braw.p);
    size_t n = i->chunked ? i->bdata.len : i->braw.len;
    img_kind_t kind = image_sniff(img, n);

    g_stats.seen++;
    if (kind == IMG_JPEG || kind == IMG_PNG) {
        prep_result_t r;
        double t0 = now_ms();
        prep_status_t st = image_prep(kind, img, n, &g_cfg.prep, &r);
        double ms = now_ms() - t0;
        if (st == PREP_OK) {
            g_stats.decoded++;
            g_stats.prep_ms_total += ms;
            if (ms > g_stats.prep_ms_max)
                g_stats.prep_ms_max = ms;
            fprintf(stderr, "IMG host=%s kind=%s src=%dx%d bytes=%zu prep=%.1fms\n", i->host,
                    image_kind_name(kind), r.src_w, r.src_h, n, ms);
            /* DES-3.3: r.rgb goes to fpgate-accel here and its score decides. */
            prep_result_free(&r);
        } else {
            if (st == PREP_CORRUPT)
                g_stats.corrupt++;
            else if (st == PREP_TOO_LARGE)
                g_stats.too_many_pixels++;
            else
                g_stats.unsupported_format++;
            fprintf(stderr, "IMG host=%s kind=%s bytes=%zu error=%s\n", i->host,
                    image_kind_name(kind), n, prep_status_name(st));
        }
    } else if (kind == IMG_OTHER) {
        g_stats.unsupported_format++;
        fprintf(stderr, "IMG host=%s kind=other bytes=%zu error=unsupported_format\n", i->host, n);
    } else {
        g_stats.not_an_image++;
        fprintf(stderr, "IMG host=%s kind=unknown bytes=%zu error=not_an_image\n", i->host, n);
    }
    release_held(i);                    /* verdict for now: deliver the original */
}

/* ---- response callbacks ------------------------------------------------------ */

static void resp_head(void *u, const hs_head_t *h)
{
    inspect_t *i = u;
    i->mode = M_STREAM;

    if (h->status == 200 && h->body != HS_BODY_NONE) {
        const struct phr_header *ct = find_header(h, "content-type");
        ct_class_t cls = ct ? image_classify_content_type(ct->value, ct->value_len) : CT_AMBIGUOUS;
        const struct phr_header *ce = find_header(h, "content-encoding");
        int encoded = ce && !(ce->value_len == 8 && strncasecmp(ce->value, "identity", 8) == 0);

        if (cls != CT_NOT_IMAGE) {
            if (encoded) {
                g_stats.skipped_encoded++;
            } else if (h->body == HS_BODY_LENGTH && h->content_length > g_cfg.max_bytes) {
                g_stats.skipped_oversize++;
            } else if (h->body == HS_BODY_LENGTH && cls == CT_AMBIGUOUS &&
                       h->content_length < IMG_SNIFF_BYTES) {
                /* too short to be an image */
            } else if (g_held + (h->body == HS_BODY_LENGTH ? h->content_length : 0) >
                       g_cfg.budget_bytes) {
                g_stats.skipped_budget++;
            } else {
                i->mode = cls == CT_IMAGE ? M_CAPTURE : M_SNIFF;
                i->chunked = h->body == HS_BODY_CHUNKED;
                i->braw.secret = i->bdata.secret = 1;
                if (buf_append(&i->head, h->raw, h->raw_len) != 0) {
                    i->oom = 1;
                    i->mode = M_STREAM;
                } else {
                    account(i);
                    return;
                }
            }
        }
    } else if (h->status == 206 && !h->is_request) {
        const struct phr_header *ct = find_header(h, "content-type");
        if (ct && image_classify_content_type(ct->value, ct->value_len) == CT_IMAGE)
            g_stats.skipped_partial++;
    }
    emit(i, h->raw, h->raw_len);
}

static void resp_body(void *u, const char *raw, size_t raw_len, const char *data, size_t data_len)
{
    inspect_t *i = u;
    if (i->mode == M_STREAM) {
        emit(i, raw, raw_len);
        return;
    }
    if (buf_append(&i->braw, raw, raw_len) != 0 ||
        (i->chunked && buf_append(&i->bdata, data, data_len) != 0)) {
        i->oom = 1;
        release_held(i);
        return;
    }
    account(i);

    if (i->braw.len > g_cfg.max_bytes) {
        g_stats.skipped_oversize++;
        release_held(i);
        return;
    }
    if (g_held > g_cfg.budget_bytes) {
        g_stats.skipped_budget++;
        release_held(i);
        return;
    }
    if (i->mode == M_SNIFF) {
        size_t have = i->chunked ? i->bdata.len : i->braw.len;
        if (have >= IMG_SNIFF_BYTES) {
            const uint8_t *b = (const uint8_t *)(i->chunked ? i->bdata.p : i->braw.p);
            if (image_sniff(b, have) == IMG_UNKNOWN)
                release_held(i);        /* not an image after all: stop holding */
            else
                i->mode = M_CAPTURE;
        }
    }
}

static void resp_end(void *u)
{
    inspect_t *i = u;
    if (i->mode == M_CAPTURE)
        analyse(i);
    else if (i->mode == M_SNIFF)
        release_held(i);
    i->mode = M_STREAM;
}

static void resp_opaque(void *u, const char *p, size_t n)
{
    emit(u, p, n);
}

/* ---- request callbacks: forward everything, the parser only tracks framing ---- */

static void req_head(void *u, const hs_head_t *h)
{
    emit(u, h->raw, h->raw_len);
}

static void req_body(void *u, const char *raw, size_t raw_len, const char *data, size_t data_len)
{
    (void)data;
    (void)data_len;
    emit(u, raw, raw_len);
}

static void req_end(void *u) { (void)u; }

/* ---- public API -------------------------------------------------------------- */

inspect_t *inspect_new(const char *host, int track_requests, int first_method)
{
    if (!g_cfg.enabled)
        return NULL;
    inspect_t *i = calloc(1, sizeof *i);
    if (!i)
        return NULL;
    snprintf(i->host, sizeof i->host, "%s", host);
    hs_pair_init(&i->pair);
    hs_callbacks_t rcb = { resp_head, resp_body, resp_end, resp_opaque };
    i->resp = hs_new(0, &i->pair, &rcb, i);
    if (track_requests) {
        hs_callbacks_t qcb = { req_head, req_body, req_end, resp_opaque };
        i->req = hs_new(1, &i->pair, &qcb, i);
    } else {
        hs_pair_push(&i->pair, first_method);
    }
    if (!i->resp || (track_requests && !i->req)) {
        inspect_free(i);
        return NULL;
    }
    return i;
}

void inspect_free(inspect_t *i)
{
    if (!i)
        return;
    drop_held(i);
    hs_free(i->req);
    hs_free(i->resp);
    free(i);
}

int inspect_request(inspect_t *i, const char *p, size_t n, buf_t *out)
{
    i->out = out;
    if (hs_feed(i->req, p, n) != 0 || i->oom)
        return -1;
    return 0;
}

int inspect_response(inspect_t *i, const char *p, size_t n, buf_t *out)
{
    i->out = out;
    if (hs_feed(i->resp, p, n) != 0 || i->oom)
        return -1;
    return 0;
}

int inspect_response_eof(inspect_t *i, buf_t *out)
{
    i->out = out;
    int rc = hs_eof(i->resp);
    if (i->mode != M_STREAM)
        release_held(i);                /* the message was cut short: pass on what we have */
    return rc;
}

int inspect_wants_requests(const inspect_t *i) { return i->req != NULL; }
