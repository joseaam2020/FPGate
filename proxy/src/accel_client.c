#define _GNU_SOURCE
#include "accel_client.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "accel_proto.h"

#define PROBE_TIMEOUT_MS 2000u

typedef struct req {
    struct req *next;
    uint32_t id;
    int kind;                   /* ACCEL_MSG_CLASSIFY, or ACCEL_MSG_STATUS for a probe */
    uint8_t *rgb;
    size_t len;
    int w, h;
    accel_done_fn done;
    void *user;
    int cancelled;
    uint64_t t_queued, t_sent;
} req_t;

#define DONE_RING 64

static struct {
    accel_cfg_t cfg;
    char path[108];
    int on, epfd, fd, registered;
    int degraded;
    req_t *qh, *qt;             /* waiting */
    unsigned qn;
    req_t *cur;                 /* in flight */
    accel_req_hdr_t hdr;
    size_t sent;                /* bytes of header + payload already written */
    unsigned char rbuf[sizeof(accel_resp_hdr_t)];
    size_t rlen;
    uint32_t next_id;
    uint64_t last_probe;
    accel_stats_t st;
    uint64_t done_ms[DONE_RING];    /* when the last answers arrived, for the throughput */
    unsigned long done_n;
} A = { .fd = -1 };

static char g_tag;

void *accel_tag(void) { return &g_tag; }
int accel_enabled(void) { return A.on; }
int accel_degraded(void) { return A.degraded; }
const accel_stats_t *accel_stats(void) { return &A.st; }
const accel_cfg_t *accel_config(void) { return &A.cfg; }

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

/* ---- finishing requests ------------------------------------------------------ */

static void wipe_rgb(req_t *r)
{
    if (r->rgb) {
        explicit_bzero(r->rgb, r->len);
        free(r->rgb);
        r->rgb = NULL;
    }
}

/* Answers a request, exactly once, and releases it. */
static void complete(req_t *r, accel_outcome_t oc, const accel_resp_hdr_t *resp)
{
    wipe_rgb(r);
    if (r->kind == ACCEL_MSG_CLASSIFY) {
        accel_result_t res;
        memset(&res, 0, sizeof res);
        res.outcome = oc;
        res.wait_ms = (unsigned)(now_ms() - r->t_queued);
        if (resp) {
            res.score = resp->score_nsfw;
            res.hw_cycles = resp->hw_cycles;
            res.latency_us = resp->latency_us;
        }
        switch (oc) {
        case ACCEL_R_OK:
            A.st.ok++;
            if (A.st.ok == 1 || res.latency_us < A.st.lat_min_us)
                A.st.lat_min_us = res.latency_us;
            if (res.latency_us > A.st.lat_max_us)
                A.st.lat_max_us = res.latency_us;
            A.st.lat_sum_us += res.latency_us;
            A.done_ms[A.done_n++ % DONE_RING] = now_ms();
            break;
        case ACCEL_R_ERROR:      A.st.errors++; break;
        case ACCEL_R_TIMEOUT:    A.st.timeouts++; break;
        case ACCEL_R_DOWN:       A.st.down++; break;
        case ACCEL_R_QUEUE_FULL: A.st.queue_full++; break;
        case ACCEL_R_CANCELLED:  A.st.cancelled++; break;
        }
        if (r->done)
            r->done(r->user, &res);
    }
    free(r);
}

/* ---- the socket -------------------------------------------------------------- */

static size_t total_len(const req_t *r) { return sizeof(accel_req_hdr_t) + r->len; }

static void set_events(void)
{
    if (A.fd < 0)
        return;
    struct epoll_event e;
    memset(&e, 0, sizeof e);
    e.events = EPOLLIN | ((A.cur && A.sent < total_len(A.cur)) ? EPOLLOUT : 0);
    e.data.ptr = &g_tag;
    epoll_ctl(A.epfd, A.registered ? EPOLL_CTL_MOD : EPOLL_CTL_ADD, A.fd, &e);
    A.registered = 1;
}

static void sock_close(void)
{
    if (A.fd >= 0) {
        epoll_ctl(A.epfd, EPOLL_CTL_DEL, A.fd, NULL);
        close(A.fd);
    }
    A.fd = -1;
    A.registered = 0;
    A.rlen = 0;
}

static int sock_open(void)
{
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -1;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", A.path);
    if (connect(fd, (struct sockaddr *)&sa, sizeof sa) != 0) {
        close(fd);
        return -1;
    }
    A.fd = fd;
    A.rlen = 0;
    A.registered = 0;
    set_events();
    return 0;
}

/* Writes as much of the in-flight request as the socket takes. -1 on a broken link. */
static int try_send(void)
{
    req_t *r = A.cur;
    size_t hdrn = sizeof A.hdr, total = total_len(r);

    while (A.sent < total) {
        const char *p;
        size_t n;
        if (A.sent < hdrn) {
            p = (const char *)&A.hdr + A.sent;
            n = hdrn - A.sent;
        } else {
            p = (const char *)r->rgb + (A.sent - hdrn);
            n = total - A.sent;
        }
        ssize_t s = send(A.fd, p, n, MSG_NOSIGNAL);
        if (s < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                break;
            if (errno == EINTR)
                continue;
            return -1;
        }
        A.sent += (size_t)s;
    }
    if (A.sent >= total)
        wipe_rgb(r);                    /* the tensor has left: nothing to keep */
    set_events();
    return 0;
}

/* ---- degraded mode ------------------------------------------------------------ */

static void enter_degraded(const char *why)
{
    if (!A.degraded) {
        A.degraded = 1;
        A.st.degraded_entries++;
        fprintf(stderr, "accel: degraded mode (%s); images follow the failure policy until it "
                        "answers again\n", why);
    }
    A.last_probe = now_ms();
    sock_close();

    req_t *q = A.qh;                    /* nothing waits for a dead accelerator */
    A.qh = A.qt = NULL;
    A.qn = 0;
    while (q) {
        req_t *n = q->next;
        complete(q, ACCEL_R_DOWN, NULL);
        q = n;
    }
}

static void leave_degraded(void)
{
    if (A.degraded) {
        A.degraded = 0;
        fprintf(stderr, "accel: answering again, back to normal\n");
    }
}

/* The in-flight request failed. The link is dropped so a late answer cannot be misread. */
static void fail_cur(accel_outcome_t oc, const char *why)
{
    req_t *r = A.cur;
    A.cur = NULL;
    A.sent = 0;
    sock_close();
    if (r && r->kind == ACCEL_MSG_STATUS) {
        complete(r, oc, NULL);          /* a failed probe: stay degraded, try again later */
        return;
    }
    enter_degraded(why);
    if (r)
        complete(r, oc, NULL);
}

static void start_next(void)
{
    while (!A.cur && A.qh) {
        req_t *r = A.qh;
        A.qh = r->next;
        if (!A.qh)
            A.qt = NULL;
        A.qn--;
        r->next = NULL;
        if (r->cancelled) {
            complete(r, ACCEL_R_CANCELLED, NULL);
            continue;
        }
        if (A.fd < 0 && sock_open() != 0) {
            enter_degraded("cannot connect");
            complete(r, ACCEL_R_DOWN, NULL);
            return;
        }
        A.cur = r;
        r->id = ++A.next_id;
        r->t_sent = now_ms();
        memset(&A.hdr, 0, sizeof A.hdr);
        memcpy(A.hdr.magic, ACCEL_REQ_MAGIC, 4);
        A.hdr.version = ACCEL_PROTO_VERSION;
        A.hdr.type = (uint8_t)r->kind;
        A.hdr.channels = 3;
        A.hdr.pixfmt = ACCEL_PIXFMT_RGB8_HWC;
        A.hdr.request_id = r->id;
        A.hdr.width = (uint16_t)r->w;
        A.hdr.height = (uint16_t)r->h;
        A.hdr.payload_len = (uint32_t)r->len;
        A.sent = 0;
        if (try_send() != 0) {
            fail_cur(ACCEL_R_ERROR, "send failed");
            return;
        }
    }
}

/* ---- answers ------------------------------------------------------------------- */

static void handle_response(void)
{
    accel_resp_hdr_t h;
    memcpy(&h, A.rbuf, sizeof h);
    if (memcmp(h.magic, ACCEL_RESP_MAGIC, 4) != 0 || h.version != ACCEL_PROTO_VERSION) {
        fail_cur(ACCEL_R_ERROR, "malformed answer");
        return;
    }
    if (!A.cur || h.request_id != A.cur->id)
        return;                         /* answer to something we no longer wait for */

    req_t *r = A.cur;
    A.cur = NULL;
    A.sent = 0;

    if (r->kind == ACCEL_MSG_STATUS) {
        if (h.status == ACCEL_OK)
            leave_degraded();
        else
            sock_close();               /* alive but busy, stuck or broken: not ready yet */
        complete(r, ACCEL_R_OK, &h);
        start_next();
        return;
    }
    if (h.status == ACCEL_OK && h.score_nsfw >= 0.0f && h.score_nsfw <= 1.0f) {
        complete(r, ACCEL_R_OK, &h);
        start_next();
        return;
    }
    complete(r, ACCEL_R_ERROR, &h);
    if (h.status == ACCEL_HW_ERROR || h.status == ACCEL_DEGRADED)
        enter_degraded("the accelerator reports a hardware problem");
    else
        start_next();                   /* e.g. busy or unsupported: this image's problem only */
}

void accel_on_event(uint32_t ev)
{
    if (!A.on || A.fd < 0)
        return;
    if ((ev & EPOLLOUT) && A.cur && try_send() != 0) {
        fail_cur(ACCEL_R_ERROR, "send failed");
        return;
    }
    if (!(ev & (EPOLLIN | EPOLLHUP | EPOLLERR)))
        return;
    for (;;) {
        ssize_t n = recv(A.fd, A.rbuf + A.rlen, sizeof A.rbuf - A.rlen, 0);
        if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
            if (A.cur)
                fail_cur(ACCEL_R_ERROR, "connection lost");
            else
                sock_close();           /* idle link closed by the other side: reconnect later */
            return;
        }
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return;
        }
        A.rlen += (size_t)n;
        if (A.rlen < sizeof A.rbuf)
            continue;
        A.rlen = 0;
        handle_response();
        if (A.fd < 0)
            return;
    }
}

void accel_tick(void)
{
    if (!A.on)
        return;
    uint64_t now = now_ms();

    if (A.cur) {
        unsigned limit = A.cur->kind == ACCEL_MSG_STATUS && A.cfg.timeout_ms > PROBE_TIMEOUT_MS
                             ? PROBE_TIMEOUT_MS : A.cfg.timeout_ms;
        if (now - A.cur->t_sent > limit)
            fail_cur(ACCEL_R_TIMEOUT, "no answer in time");
        return;
    }
    if (A.degraded && now - A.last_probe >= A.cfg.probe_ms) {
        A.last_probe = now;
        if (A.fd < 0 && sock_open() != 0)
            return;                     /* still down */
        req_t *r = calloc(1, sizeof *r);
        if (!r)
            return;
        r->kind = ACCEL_MSG_STATUS;
        r->t_queued = r->t_sent = now;
        r->id = ++A.next_id;
        A.cur = r;
        memset(&A.hdr, 0, sizeof A.hdr);
        memcpy(A.hdr.magic, ACCEL_REQ_MAGIC, 4);
        A.hdr.version = ACCEL_PROTO_VERSION;
        A.hdr.type = ACCEL_MSG_STATUS;
        A.hdr.request_id = r->id;
        A.sent = 0;
        if (try_send() != 0)
            fail_cur(ACCEL_R_ERROR, "probe failed");
    }
}

/* ---- public API ------------------------------------------------------------------ */

int accel_start(const accel_cfg_t *cfg, int epfd)
{
    if (!cfg->path || strlen(cfg->path) >= sizeof A.path)
        return -1;
    A.cfg = *cfg;
    if (!A.cfg.timeout_ms)
        A.cfg.timeout_ms = 10000;
    if (!A.cfg.probe_ms)
        A.cfg.probe_ms = 2000;
    snprintf(A.path, sizeof A.path, "%s", cfg->path);
    A.epfd = epfd;
    A.fd = -1;
    A.on = 1;
    return 0;
}

void accel_classify(uint8_t *rgb, size_t len, int width, int height, accel_done_fn done,
                    void *user)
{
    A.st.requests++;
    req_t *r = calloc(1, sizeof *r);
    if (!r) {
        explicit_bzero(rgb, len);
        free(rgb);
        accel_result_t res;
        memset(&res, 0, sizeof res);
        res.outcome = ACCEL_R_ERROR;
        A.st.errors++;
        done(user, &res);
        return;
    }
    r->kind = ACCEL_MSG_CLASSIFY;
    r->rgb = rgb;
    r->len = len;
    r->w = width;
    r->h = height;
    r->done = done;
    r->user = user;
    r->t_queued = now_ms();

    if (!A.on || A.degraded) {
        complete(r, ACCEL_R_DOWN, NULL);
        return;
    }
    if (A.cur && A.qn >= A.cfg.queue_max) {
        complete(r, ACCEL_R_QUEUE_FULL, NULL);
        return;
    }
    if (A.qt)
        A.qt->next = r;
    else
        A.qh = r;
    A.qt = r;
    A.qn++;
    start_next();
}

void accel_cancel(void *user)
{
    req_t **pp = &A.qh, *prev = NULL;
    while (*pp) {
        req_t *q = *pp;
        if (q->user == user) {
            *pp = q->next;
            if (A.qt == q)
                A.qt = prev;
            A.qn--;
            complete(q, ACCEL_R_CANCELLED, NULL);
            pp = prev ? &prev->next : &A.qh;
            continue;
        }
        prev = q;
        pp = &q->next;
    }
    if (A.cur && A.cur->user == user)
        A.cur->cancelled = 1;           /* its answer is still read, the owner just ignores it */
}

void accel_stop(void)
{
    if (!A.on)
        return;
    A.on = 0;
    req_t *r = A.cur;
    A.cur = NULL;
    sock_close();
    if (r)
        complete(r, ACCEL_R_CANCELLED, NULL);
    req_t *q = A.qh;
    A.qh = A.qt = NULL;
    A.qn = 0;
    while (q) {
        req_t *n = q->next;
        complete(q, ACCEL_R_CANCELLED, NULL);
        q = n;
    }
}

double accel_throughput(void)
{
    uint64_t now = now_ms();
    unsigned long n = A.done_n < DONE_RING ? A.done_n : DONE_RING;
    unsigned recent = 0;
    uint64_t oldest = now;
    for (unsigned long i = 0; i < n; i++) {
        uint64_t t = A.done_ms[i];
        if (now - t <= 60000u) {
            recent++;
            if (t < oldest)
                oldest = t;
        }
    }
    if (recent < 2)
        return 0.0;
    double span = (double)(now - oldest) / 1000.0;
    return span > 0.0 ? (double)recent / span : 0.0;
}
