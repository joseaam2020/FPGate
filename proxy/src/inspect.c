#define _GNU_SOURCE
#include "inspect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "decode_pool.h"
#include "http_stream.h"
#include "image_detect.h"

static inspect_cfg_t g_cfg = {
    .enabled = 1,
    .max_bytes = 8u * 1024 * 1024,
    .budget_bytes = 64u * 1024 * 1024,
    .prep = { .size = 448, .center_crop = 1, .dct_scaling = 0, .max_pixels = 16000000ul },
    .workers = 1,
};
static inspect_stats_t g_stats;
static size_t g_held;                   /* bytes currently held, all connections */

void inspect_configure(const inspect_cfg_t *cfg) { g_cfg = *cfg; }
const inspect_cfg_t *inspect_config(void) { return &g_cfg; }
const inspect_stats_t *inspect_stats(void) { return &g_stats; }

enum { M_STREAM, M_SNIFF, M_CAPTURE };

/* What a response answers, for the log. Query strings are dropped on purpose. */
struct rctx {
    char path[128];
    char ref[128];                      /* Referer without scheme, i.e. the page */
};

#define RCTX_QUEUE 32

/* Copies up to the first '?' or '#', truncated. */
static void cut_url(char *dst, size_t n, const char *p, size_t len)
{
    size_t k = 0;
    while (k < len && k + 1 < n && p[k] != '?' && p[k] != '#') {
        dst[k] = p[k];
        k++;
    }
    dst[k] = '\0';
}

/* Media type only, without parameters. */
static void cut_type(char *dst, size_t n, const char *p, size_t len)
{
    while (len && (*p == ' ' || *p == '\t')) {
        p++;
        len--;
    }
    size_t k = 0;
    while (k < len && k + 1 < n && p[k] != ';' && p[k] != ' ' && p[k] != '\t') {
        dst[k] = p[k];
        k++;
    }
    dst[k] = '\0';
}


static void cut_referer(char *dst, size_t n, const char *p, size_t len)
{
    if (len >= 8 && strncasecmp(p, "https://", 8) == 0) {
        p += 8;
        len -= 8;
    } else if (len >= 7 && strncasecmp(p, "http://", 7) == 0) {
        p += 7;
        len -= 7;
    }
    cut_url(dst, n, p, len);
}

/* " path=... ref=..." when verbose logging is on, else nothing. */
static const char *ctx_suffix(char *buf, size_t n, const struct rctx *c)
{
    buf[0] = '\0';
    if (g_cfg.verbose)
        snprintf(buf, n, " path=%s ref=%s", c->path[0] ? c->path : "-", c->ref[0] ? c->ref : "-");
    return buf;
}

/* A decode in flight. It owns the held bytes while a worker reads them. */
typedef struct ijob {
    dp_job_t dp;
    struct inspect *owner;              /* NULL once the connection is gone */
    char host[256];
    struct rctx ctx;
    buf_t head, braw, bdata;
    size_t held;
    int chunked;
} ijob_t;

struct inspect {
    hs_pair_t pair;
    hs_stream_t *req, *resp;
    char host[256];
    void *ctx;
    struct rctx rq[RCTX_QUEUE];         /* requests sent, oldest first, awaiting their response */
    unsigned rq_head, rq_count;
    struct rctx cur;                    /* the request the response being handled answers */
    char cur_type[48];                  /* its Content-Type, for the log */
    char cur_size[24];                  /* its size or framing, for the log */
    int cur_status;
    buf_t *out;                         /* where forwardable bytes go during a feed */
    int oom;
    int mode;
    int chunked;
    buf_t head;                         /* held response head */
    buf_t braw;                         /* held body, as on the wire */
    buf_t bdata;                        /* held body, chunk framing removed (chunked only) */
    size_t held;
    int waiting;                        /* a decode job is in flight */
    ijob_t *job;
    buf_t backlog;                      /* server bytes that arrived behind the held image */
    int eof_pending;                    /* the server closed while the job was in flight */
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

/* Counts and logs the outcome of a decode, wherever it ran. */
static void record_result(const char *host, const struct rctx *c, img_kind_t kind, size_t bytes,
                          prep_status_t st, const prep_result_t *r, double ms)
{
    char sfx[300];
    ctx_suffix(sfx, sizeof sfx, c);
    if (st == PREP_OK) {
        g_stats.decoded++;
        g_stats.prep_ms_total += ms;
        if (ms > g_stats.prep_ms_max)
            g_stats.prep_ms_max = ms;
        fprintf(stderr, "IMG host=%s kind=%s src=%dx%d bytes=%zu prep=%.1fms%s\n", host,
                image_kind_name(kind), r->src_w, r->src_h, bytes, ms, sfx);
        return;
    }
    if (st == PREP_CORRUPT)
        g_stats.corrupt++;
    else if (st == PREP_TOO_LARGE)
        g_stats.too_many_pixels++;
    else
        g_stats.unsupported_format++;
    fprintf(stderr, "IMG host=%s kind=%s bytes=%zu error=%s%s\n", host, image_kind_name(kind),
            bytes, prep_status_name(st), sfx);
}

/* An image-looking response that is not going to be analysed, and why. */
static void log_skip(const inspect_t *i, const char *type, const char *reason, const char *size)
{
    char sfx[300];
    fprintf(stderr, "IMG host=%s type=%s skipped=%s bytes=%s%s\n", i->host, type, reason, size,
            ctx_suffix(sfx, sizeof sfx, &i->cur));
}

/*
 * The message is complete. JPEG/PNG go to a worker and the pair waits; anything
 * else is settled on the spot. Either way the original is what gets delivered
 * until DES-3.3 adds the accelerator's verdict.
 */
static void analyse(inspect_t *i)
{
    const uint8_t *img = (const uint8_t *)(i->chunked ? i->bdata.p : i->braw.p);
    size_t n = i->chunked ? i->bdata.len : i->braw.len;
    img_kind_t kind = image_sniff(img, n);

    g_stats.seen++;
    if (kind == IMG_JPEG || kind == IMG_PNG || kind == IMG_WEBP) {
        if (dp_active()) {
            ijob_t *j = calloc(1, sizeof *j);
            if (j) {
                j->dp.kind = kind;
                j->dp.data = img;
                j->dp.len = n;
                j->dp.cfg = g_cfg.prep;
                j->dp.user = j;
                if (dp_submit(&j->dp) == 0) {
                    /* The heap blocks stay where they are; only their owner changes. */
                    j->owner = i;
                    j->chunked = i->chunked;
                    snprintf(j->host, sizeof j->host, "%s", i->host);
                    j->ctx = i->cur;
                    j->head = i->head;
                    j->braw = i->braw;
                    j->bdata = i->bdata;
                    j->held = i->held;
                    memset(&i->head, 0, sizeof i->head);
                    memset(&i->braw, 0, sizeof i->braw);
                    memset(&i->bdata, 0, sizeof i->bdata);
                    i->held = 0;                /* g_held still counts it, now through the job */
                    i->job = j;
                    i->waiting = 1;
                    i->mode = M_STREAM;
                    hs_stop(i->resp);
                    return;
                }
                free(j);
            }
            g_stats.skipped_busy++;             /* queue full: let it through unexamined */
            {
                char sfx[300];
                fprintf(stderr, "IMG host=%s kind=%s bytes=%zu skipped=busy%s\n", i->host,
                        image_kind_name(kind), n, ctx_suffix(sfx, sizeof sfx, &i->cur));
            }
        } else {
            prep_result_t r;
            if (g_cfg.delay_ms)
                usleep(g_cfg.delay_ms * 1000u);     /* test hook, same delay the workers get */
            double t0 = now_ms();
            prep_status_t st = image_prep(kind, img, n, &g_cfg.prep, &r);
            record_result(i->host, &i->cur, kind, n, st, &r, now_ms() - t0);
            if (st == PREP_OK)
                prep_result_free(&r);
        }
    } else {
        char sfx[300];
        ctx_suffix(sfx, sizeof sfx, &i->cur);
        if (kind == IMG_OTHER) {
            g_stats.unsupported_format++;
            fprintf(stderr, "IMG host=%s kind=other type=%s bytes=%zu error=unsupported_format%s\n",
                    i->host, i->cur_type, n, sfx);
        } else {
            g_stats.not_an_image++;
            fprintf(stderr, "IMG host=%s kind=unknown type=%s bytes=%zu error=not_an_image%s\n",
                    i->host, i->cur_type, n, sfx);
        }
    }
    release_held(i);                    /* verdict for now: deliver the original */
}

/* ---- response callbacks ------------------------------------------------------ */

static void log_resp(const inspect_t *i, const char *action)
{
    if (!g_cfg.verbose)
        return;
    char sfx[300];
    fprintf(stderr, "RESP host=%s status=%d type=%s bytes=%s action=%s%s\n", i->host, i->cur_status,
            i->cur_type[0] ? i->cur_type : "-", i->cur_size, action,
            ctx_suffix(sfx, sizeof sfx, &i->cur));
}

static void ctx_pop(inspect_t *i)
{
    memset(&i->cur, 0, sizeof i->cur);
    if (i->rq_count) {
        i->cur = i->rq[i->rq_head];
        i->rq_head = (i->rq_head + 1) % RCTX_QUEUE;
        i->rq_count--;
    }
}

static const char *framing_name(hs_body_t b)
{
    switch (b) {
    case HS_BODY_LENGTH:  return "content-length";
    case HS_BODY_CHUNKED: return "chunked";
    case HS_BODY_CLOSE:   return "until-close";
    default:              return "none";
    }
}

static void resp_head(void *u, const hs_head_t *h)
{
    inspect_t *i = u;
    i->mode = M_STREAM;

    if (h->status >= 200)
        ctx_pop(i);                     /* 1xx are interim: the real answer is still to come */

    const struct phr_header *ct = find_header(h, "content-type");
    i->cur_type[0] = '\0';
    if (ct)
        cut_type(i->cur_type, sizeof i->cur_type, ct->value, ct->value_len);
    const char *type = i->cur_type[0] ? i->cur_type : "-";
    char *size = i->cur_size;
    i->cur_status = h->status;
    if (h->body == HS_BODY_LENGTH)
        snprintf(i->cur_size, sizeof i->cur_size, "%llu", (unsigned long long)h->content_length);
    else
        snprintf(i->cur_size, sizeof i->cur_size, "%s", framing_name(h->body));

    ct_class_t cls = ct ? image_classify_content_type(ct->value, ct->value_len) : CT_AMBIGUOUS;
    const char *action = "stream";

    if (h->status == 200 && h->body != HS_BODY_NONE) {
        const struct phr_header *ce = find_header(h, "content-encoding");
        int encoded = ce && !(ce->value_len == 8 && strncasecmp(ce->value, "identity", 8) == 0);

        if (cls != CT_NOT_IMAGE) {
            if (encoded) {
                g_stats.skipped_encoded++;
                log_skip(i, type, "encoded", size);
                action = "skip-encoded";
            } else if (h->body == HS_BODY_LENGTH && h->content_length > g_cfg.max_bytes) {
                g_stats.skipped_oversize++;
                log_skip(i, type, "oversize", size);
                action = "skip-oversize";
            } else if (h->body == HS_BODY_LENGTH && cls == CT_AMBIGUOUS &&
                       h->content_length < IMG_SNIFF_BYTES) {
                action = "stream";      /* too short to be an image */
            } else if (g_held + (h->body == HS_BODY_LENGTH ? h->content_length : 0) >
                       g_cfg.budget_bytes) {
                g_stats.skipped_budget++;
                log_skip(i, type, "budget", size);
                action = "skip-budget";
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
    } else if (h->status == 206 && !h->is_request && cls == CT_IMAGE) {
        g_stats.skipped_partial++;
        log_skip(i, type, "partial", size);
        action = "skip-partial";
    }

    if (h->status >= 200)
        log_resp(i, action);
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
{
                log_resp(i, "sniffed-not-image");
                release_held(i);        /* not an image after all: stop holding */
            }
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
{
        log_resp(i, "sniffed-too-short");
        release_held(i);
    }
    if (!i->waiting)
        i->mode = M_STREAM;
}

static void resp_opaque(void *u, const char *p, size_t n)
{
    emit(u, p, n);
}

/* ---- request callbacks: forward everything, the parser only tracks framing ---- */

static void ctx_push(inspect_t *i, const char *path, size_t path_len, const char *ref,
                     size_t ref_len)
{
    if (i->rq_count >= RCTX_QUEUE)
        return;                         /* absurdly deep pipeline: those responses go unlabelled */
    struct rctx *c = &i->rq[(i->rq_head + i->rq_count) % RCTX_QUEUE];
    memset(c, 0, sizeof *c);
    if (path)
        cut_url(c->path, sizeof c->path, path, path_len);
    if (ref)
        cut_referer(c->ref, sizeof c->ref, ref, ref_len);
    i->rq_count++;
}

/*
 * Forwards a request head. The only edit: image/avif is removed from Accept so
 * that servers fall back to WebP (decodable here) instead of AVIF (not).
 */
static void emit_request_head(inspect_t *i, const hs_head_t *h)
{
    for (size_t k = 0; k < h->nhdrs; k++) {
        const struct phr_header *a = &h->hdrs[k];
        if (!a->name || a->name_len != 6 || strncasecmp(a->name, "accept", 6) != 0)
            continue;
        if (a->value < h->raw || a->value + a->value_len > h->raw + h->raw_len)
            continue;
        char tmp[1024];
        size_t tn;
        if (accept_drop_avif(a->value, a->value_len, tmp, sizeof tmp, &tn) != 1)
            continue;
        size_t off = (size_t)(a->value - h->raw);
        emit(i, h->raw, off);
        emit(i, tmp, tn);
        emit(i, h->raw + off + a->value_len, h->raw_len - off - a->value_len);
        g_stats.accept_rewritten++;
        return;
    }
    emit(i, h->raw, h->raw_len);
}

static void req_head(void *u, const hs_head_t *h)
{
    inspect_t *i = u;
    const struct phr_header *rf = find_header(h, "referer");
    ctx_push(i, h->path, h->path_len, rf ? rf->value : NULL, rf ? rf->value_len : 0);
    emit_request_head(i, h);
}

void inspect_note_request(inspect_t *i, const char *path, size_t path_len, const char *referer,
                          size_t referer_len)
{
    ctx_push(i, path, path_len, referer, referer_len);
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
    i->backlog.secret = 1;
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
    if (i->job)
        i->job->owner = NULL;           /* the worker's result will be discarded */
    drop_held(i);
    buf_wipe_free(&i->backlog);
    hs_free(i->req);
    hs_free(i->resp);
    free(i);
}

void inspect_set_ctx(inspect_t *i, void *ctx) { i->ctx = ctx; }
int inspect_wants_requests(const inspect_t *i) { return i->req != NULL; }
int inspect_busy(const inspect_t *i) { return i->waiting; }

int inspect_request(inspect_t *i, const char *p, size_t n, buf_t *out)
{
    i->out = out;
    if (hs_feed(i->req, p, n) != 0 || i->oom)
        return -1;
    return 0;
}

/* Keeps what the parser did not consume because the pair started waiting. */
static int stash_rest(inspect_t *i)
{
    const char *rp;
    size_t rn = hs_rest(i->resp, &rp);
    return rn && buf_append(&i->backlog, rp, rn) != 0 ? -1 : 0;
}

int inspect_response(inspect_t *i, const char *p, size_t n, buf_t *out)
{
    i->out = out;
    if (i->waiting)                     /* behind the held image: keep the order */
        return buf_append(&i->backlog, p, n) != 0 ? -1 : 0;
    if (hs_feed(i->resp, p, n) != 0 || i->oom)
        return -1;
    if (i->waiting && stash_rest(i) != 0)
        return -1;
    return 0;
}

static int flush_eof(inspect_t *i)
{
    int rc = hs_eof(i->resp);
    if (!i->waiting && i->mode != M_STREAM)
        release_held(i);                /* the message was cut short: pass on what we have */
    return rc;
}

int inspect_response_eof(inspect_t *i, buf_t *out)
{
    i->out = out;
    if (i->waiting) {
        i->eof_pending = 1;
        return 0;
    }
    return flush_eof(i);
}

/* ---- finished decodes ---------------------------------------------------------- */

static void wipe_job(ijob_t *j)
{
    buf_wipe_free(&j->head);
    buf_wipe_free(&j->braw);
    buf_wipe_free(&j->bdata);
    g_held -= j->held;
    j->held = 0;
}

/* Feeds the bytes that queued behind the image, which may start another job. */
static int pump(inspect_t *i)
{
    while (!i->waiting && buf_pending(&i->backlog)) {
        buf_t bl = i->backlog;
        memset(&i->backlog, 0, sizeof i->backlog);
        i->backlog.secret = 1;
        int rc = hs_feed(i->resp, bl.p + bl.off, buf_pending(&bl));
        if (rc == 0 && i->waiting && stash_rest(i) != 0)
            rc = -1;
        buf_wipe_free(&bl);
        if (rc != 0)
            return -1;
    }
    if (!i->waiting && i->eof_pending) {
        i->eof_pending = 0;
        (void)flush_eof(i);
    }
    return i->oom ? -1 : 0;
}

void inspect_collect(inspect_ready_fn ready)
{
    dp_job_t *dj;
    while ((dj = dp_take_done())) {
        ijob_t *j = dj->user;
        size_t n = j->chunked ? j->bdata.len : j->braw.len;

        record_result(j->host, &j->ctx, dj->kind, n, dj->status, &dj->result, dj->ms);
        if (dj->status == PREP_OK) {
            /* DES-3.3: dj->result.rgb goes to fpgate-accel here and its score decides. */
            prep_result_free(&dj->result);
        }

        inspect_t *i = j->owner;
        if (!i) {                       /* the connection died while the worker ran */
            wipe_job(j);
            free(j);
            continue;
        }
        buf_t out = {0};
        i->out = &out;
        i->job = NULL;
        i->waiting = 0;
        hs_resume(i->resp);
        emit(i, j->head.p, j->head.len);    /* the original, unchanged */
        emit(i, j->braw.p, j->braw.len);
        wipe_job(j);
        free(j);
        int err = pump(i);
        ready(i->ctx, &out, err);
        buf_wipe_free(&out);
    }
}
