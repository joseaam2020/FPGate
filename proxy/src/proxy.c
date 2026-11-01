#define _GNU_SOURCE
#include "proxy.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "picohttpparser.h"

#define HDR_MAX       16384          /* max size of a request head */
#define MAX_HEADERS   64
#define OUT_HIGH      (256 * 1024)   /* backpressure threshold per direction */
#define IDLE_SECS     60
#define MAX_CONNS     1024
#define HOST_MAX      256

/* ---- growable output buffer ------------------------------------------- */

typedef struct {
    char *p;
    size_t len, off, cap;
} buf_t;

static size_t buf_pending(const buf_t *b) { return b->len - b->off; }

static int buf_append(buf_t *b, const void *d, size_t n)
{
    if (b->off == b->len)
        b->off = b->len = 0;
    if (b->len + n > b->cap) {
        size_t nc = b->cap ? b->cap : 4096;
        while (nc < b->len + n)
            nc *= 2;
        char *np = realloc(b->p, nc);
        if (!np)
            return -1;
        b->p = np;
        b->cap = nc;
    }
    memcpy(b->p + b->len, d, n);
    b->len += n;
    return 0;
}

static int buf_printf(buf_t *b, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof tmp)
        return -1;
    return buf_append(b, tmp, (size_t)n);
}

static void buf_free(buf_t *b)
{
    free(b->p);
    memset(b, 0, sizeof *b);
}

/* ---- connections ------------------------------------------------------ */

enum { ST_LISTEN, ST_HEADERS, ST_CONNECTING, ST_RELAY };

typedef struct conn {
    int fd;
    int state;
    int is_client;
    struct conn *peer;
    struct conn *prev, *next;       /* all-connections list */
    struct conn *dead_next;         /* deferred-free list */
    buf_t out;                      /* bytes waiting to be written to fd */
    char *hdr;                      /* client only: request head accumulator */
    size_t hdr_len;
    int eof;                        /* read side reached EOF */
    int wr_shut;                    /* shutdown(SHUT_WR) already done */
    int paused;                     /* EPOLLIN off because peer is backed up */
    int tunnel;                     /* CONNECT or Upgrade: no HTTP framing */
    int dead;
    uint32_t events;
    time_t last;
} conn_t;

static int g_ep = -1;
static conn_t *g_all;
static conn_t *g_dead;
static int g_nconns;

static void set_events(conn_t *c, uint32_t ev, int add)
{
    struct epoll_event e;
    memset(&e, 0, sizeof e);
    e.events = ev;
    e.data.ptr = c;
    epoll_ctl(g_ep, add ? EPOLL_CTL_ADD : EPOLL_CTL_MOD, c->fd, &e);
    c->events = ev;
}

static void update_events(conn_t *c)
{
    if (c->dead)
        return;
    uint32_t ev = 0;
    if (c->state == ST_CONNECTING && !c->is_client)
        ev = EPOLLOUT;
    else if (c->state == ST_HEADERS)
        ev = EPOLLIN;
    else if (c->state == ST_RELAY) {
        if (!c->paused && !c->eof)
            ev = EPOLLIN;
        if (buf_pending(&c->out))
            ev |= EPOLLOUT;
    }
    if (ev != c->events)
        set_events(c, ev, 0);
}

static conn_t *conn_new(int fd, int is_client, int state)
{
    conn_t *c = calloc(1, sizeof *c);
    if (!c)
        return NULL;
    c->fd = fd;
    c->is_client = is_client;
    c->state = state;
    c->last = time(NULL);
    c->next = g_all;
    if (g_all)
        g_all->prev = c;
    g_all = c;
    g_nconns++;
    return c;
}

static void conn_kill(conn_t *c)
{
    if (c->dead)
        return;
    c->dead = 1;
    close(c->fd);
    if (c->prev)
        c->prev->next = c->next;
    else
        g_all = c->next;
    if (c->next)
        c->next->prev = c->prev;
    c->dead_next = g_dead;
    g_dead = c;
    g_nconns--;
}

static void pair_kill(conn_t *c)
{
    conn_t *p = c->peer;
    conn_kill(c);
    if (p)
        conn_kill(p);
}

static void reap_dead(void)
{
    while (g_dead) {
        conn_t *c = g_dead;
        g_dead = c->dead_next;
        buf_free(&c->out);
        free(c->hdr);
        free(c);
    }
}

/* ---- helpers ---------------------------------------------------------- */

static void send_error(conn_t *c, const char *status, const char *msg)
{
    char resp[512];
    int n = snprintf(resp, sizeof resp,
                     "HTTP/1.1 %s\r\nContent-Type: text/plain; charset=utf-8\r\n"
                     "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
                     status, strlen(msg), msg);
    if (n > 0 && (size_t)n < sizeof resp)
        (void)send(c->fd, resp, (size_t)n, MSG_NOSIGNAL | MSG_DONTWAIT);
}

static int name_is(const struct phr_header *h, const char *name)
{
    return h->name_len == strlen(name) && strncasecmp(h->name, name, h->name_len) == 0;
}

static int is_hop_by_hop(const struct phr_header *h)
{
    static const char *const names[] = {
        "connection", "proxy-connection", "keep-alive", "te", "trailer",
        "proxy-authorization", "proxy-authenticate", "upgrade", NULL
    };
    for (int i = 0; names[i]; i++)
        if (name_is(h, names[i]))
            return 1;
    return 0;
}

/* Splits "host", "host:port" or "[v6]:port" in place. Returns 0 on success. */
static int parse_authority(const char *s, size_t n, char *host, char *port,
                           const char *default_port)
{
    if (n == 0 || n >= HOST_MAX)
        return -1;
    char tmp[HOST_MAX];
    memcpy(tmp, s, n);
    tmp[n] = '\0';

    const char *h = tmp, *p = NULL;
    char *end;
    if (tmp[0] == '[') {
        end = strchr(tmp, ']');
        if (!end)
            return -1;
        *end = '\0';
        h = tmp + 1;
        if (end[1] == ':')
            p = end + 2;
        else if (end[1] != '\0')
            return -1;
    } else {
        char *colon = strrchr(tmp, ':');
        if (colon) {
            *colon = '\0';
            p = colon + 1;
        }
    }
    if (*h == '\0')
        return -1;
    if (!p || *p == '\0')
        p = default_port;
    for (const char *q = p; *q; q++)
        if (*q < '0' || *q > '9')
            return -1;
    if (strlen(p) > 5)
        return -1;
    snprintf(host, HOST_MAX, "%s", h);
    snprintf(port, 8, "%s", p);
    return 0;
}

/*
 * Blocking DNS lookup + non-blocking connect.
 * TODO: resolver cache / async resolution; getaddrinfo stalls the reactor.
 */
static int upstream_connect(const char *host, const char *port)
{
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0 || !res)
        return -1;

    int fd = socket(res->ai_family, res->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd >= 0) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (connect(fd, res->ai_addr, res->ai_addrlen) != 0 && errno != EINPROGRESS) {
            close(fd);
            fd = -1;
        }
    }
    freeaddrinfo(res);
    return fd;
}

/* ---- request handling ------------------------------------------------- */

/*
 * Parses the request head in c->hdr. Returns 1 when the upstream connection
 * was started, 0 if more data is needed, -1 when the client was answered
 * with an error and must be closed.
 */
static int handle_request_head(conn_t *c)
{
    const char *method, *path;
    size_t method_len, path_len, nh = MAX_HEADERS;
    int minor;
    struct phr_header hs[MAX_HEADERS];

    int r = phr_parse_request(c->hdr, c->hdr_len, &method, &method_len, &path,
                              &path_len, &minor, hs, &nh, 0);
    if (r == -2) {
        if (c->hdr_len >= HDR_MAX) {
            send_error(c, "431 Request Header Fields Too Large", "FPGate: encabezados demasiado grandes\n");
            return -1;
        }
        return 0;
    }
    if (r < 0) {
        send_error(c, "400 Bad Request", "FPGate: solicitud invalida\n");
        return -1;
    }

    char host[HOST_MAX], port[8];
    buf_t head = {0};
    int is_connect = method_len == 7 && memcmp(method, "CONNECT", 7) == 0;
    const char *target_path = "/";
    size_t target_path_len = 1;

    if (is_connect) {
        if (parse_authority(path, path_len, host, port, "443") != 0) {
            send_error(c, "400 Bad Request", "FPGate: destino CONNECT invalido\n");
            return -1;
        }
    } else {
        if (path_len < 8 || strncasecmp(path, "http://", 7) != 0) {
            send_error(c, "400 Bad Request",
                       "FPGate: este es un proxy, use la forma absoluta http://host/ruta\n");
            return -1;
        }
        const char *auth = path + 7;
        size_t rest = path_len - 7, alen = 0;
        while (alen < rest && auth[alen] != '/' && auth[alen] != '?')
            alen++;
        if (parse_authority(auth, alen, host, port, "80") != 0) {
            send_error(c, "400 Bad Request", "FPGate: URL invalida\n");
            return -1;
        }
        if (alen < rest && auth[alen] == '/') {
            target_path = auth + alen;
            target_path_len = rest - alen;
        } else if (alen < rest) {          /* "http://host?query" */
            target_path = NULL;
        }
    }

    conn_t *u = NULL;
    int ufd = upstream_connect(host, port);
    if (ufd < 0) {
        send_error(c, "502 Bad Gateway", "FPGate: no se pudo contactar al servidor\n");
        goto fail;
    }
    u = conn_new(ufd, 0, ST_CONNECTING);
    if (!u) {
        close(ufd);
        send_error(c, "503 Service Unavailable", "FPGate: sin memoria\n");
        goto fail;
    }

    if (is_connect) {
        c->tunnel = u->tunnel = 1;
        if (buf_append(&c->out, "HTTP/1.1 200 Connection Established\r\n\r\n", 39) != 0)
            goto fail_u;
        /* Bytes after the CONNECT head (if any) are the client's first TLS bytes. */
        if ((size_t)r < c->hdr_len &&
            buf_append(&u->out, c->hdr + r, c->hdr_len - (size_t)r) != 0)
            goto fail_u;
    } else {
        int has_host = 0, upgrade = 0;
        for (size_t i = 0; i < nh; i++) {
            if (name_is(&hs[i], "host"))
                has_host = 1;
            if (name_is(&hs[i], "upgrade"))
                upgrade = 1;
        }
        if (buf_append(&head, method, method_len) != 0 || buf_append(&head, " ", 1) != 0)
            goto fail_u;
        if (target_path) {
            if (buf_append(&head, target_path, target_path_len) != 0)
                goto fail_u;
        } else {
            const char *a = path + 7;
            size_t alen = 0;
            while (a[alen] != '?')
                alen++;
            if (buf_append(&head, "/", 1) != 0 ||
                buf_append(&head, a + alen, path_len - 7 - alen) != 0)
                goto fail_u;
        }
        if (buf_append(&head, " HTTP/1.1\r\n", 11) != 0)
            goto fail_u;
        if (!has_host && buf_printf(&head, "Host: %s%s%s\r\n", host,
                                    strcmp(port, "80") ? ":" : "",
                                    strcmp(port, "80") ? port : "") != 0)
            goto fail_u;
        for (size_t i = 0; i < nh; i++) {
            if (is_hop_by_hop(&hs[i]))
                continue;
            if (buf_append(&head, hs[i].name, hs[i].name_len) != 0 ||
                buf_append(&head, ": ", 2) != 0 ||
                buf_append(&head, hs[i].value, hs[i].value_len) != 0 ||
                buf_append(&head, "\r\n", 2) != 0)
                goto fail_u;
        }
        if (upgrade) {                      /* e.g. WebSocket: pass through as a tunnel */
            c->tunnel = u->tunnel = 1;
            for (size_t i = 0; i < nh; i++)
                if (name_is(&hs[i], "upgrade")) {
                    buf_append(&head, "Upgrade: ", 9);
                    buf_append(&head, hs[i].value, hs[i].value_len);
                    buf_append(&head, "\r\n", 2);
                }
            buf_append(&head, "Connection: Upgrade\r\n\r\n", 23);
        } else {
            buf_append(&head, "Connection: close\r\n\r\n", 21);
        }
        if (buf_append(&u->out, head.p, head.len) != 0)
            goto fail_u;
        if ((size_t)r < c->hdr_len &&
            buf_append(&u->out, c->hdr + r, c->hdr_len - (size_t)r) != 0)
            goto fail_u;
    }

    buf_free(&head);
    free(c->hdr);
    c->hdr = NULL;
    c->hdr_len = 0;
    c->peer = u;
    u->peer = c;
    c->state = ST_CONNECTING;
    set_events(u, EPOLLOUT, 1);
    update_events(c);
    fprintf(stderr, "%s %s:%s\n", is_connect ? "CONNECT" : "HTTP", host, port);
    return 1;

fail_u:
    conn_kill(u);
    send_error(c, "503 Service Unavailable", "FPGate: sin memoria\n");
fail:
    buf_free(&head);
    return -1;
}

static void on_client_head(conn_t *c)
{
    ssize_t n = recv(c->fd, c->hdr + c->hdr_len, HDR_MAX - c->hdr_len, 0);
    if (n == 0) {
        conn_kill(c);
        return;
    }
    if (n < 0) {
        if (errno != EAGAIN && errno != EINTR)
            conn_kill(c);
        return;
    }
    c->hdr_len += (size_t)n;
    c->last = time(NULL);
    if (handle_request_head(c) < 0)
        conn_kill(c);
}

/* ---- relay ------------------------------------------------------------ */

static int flush_out(conn_t *c)
{
    while (buf_pending(&c->out)) {
        ssize_t n = send(c->fd, c->out.p + c->out.off, buf_pending(&c->out), MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return 0;
            if (errno == EINTR)
                continue;
            return -1;
        }
        c->out.off += (size_t)n;
    }
    return 0;
}

/* Propagates EOF to the peer and decides when the pair is finished. */
static void check_done(conn_t *c)
{
    conn_t *p = c->peer;
    if (c->dead || !p || p->dead)
        return;
    if (c->eof && !buf_pending(&p->out) && !p->wr_shut && !(!c->tunnel && !c->is_client)) {
        shutdown(p->fd, SHUT_WR);
        p->wr_shut = 1;
    }
    if (!c->tunnel && !c->is_client && c->eof && !buf_pending(&p->out)) {
        pair_kill(c);               /* plain HTTP: response complete (upstream closed) */
        return;
    }
    if (c->eof && p->eof && !buf_pending(&c->out) && !buf_pending(&p->out))
        pair_kill(c);
}

static void on_relay_readable(conn_t *c)
{
    char tmp[16384];
    ssize_t n = recv(c->fd, tmp, sizeof tmp, 0);
    if (n < 0) {
        if (errno != EAGAIN && errno != EINTR)
            pair_kill(c);
        return;
    }
    c->last = time(NULL);
    conn_t *p = c->peer;
    if (n == 0) {
        c->eof = 1;
        update_events(c);
        check_done(c);
        return;
    }
    size_t off = 0;
    if (!buf_pending(&p->out)) {
        ssize_t s = send(p->fd, tmp, (size_t)n, MSG_NOSIGNAL);
        if (s < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            pair_kill(c);
            return;
        }
        if (s > 0)
            off = (size_t)s;
    }
    if (off < (size_t)n && buf_append(&p->out, tmp + off, (size_t)n - off) != 0) {
        pair_kill(c);
        return;
    }
    if (buf_pending(&p->out) > OUT_HIGH)
        c->paused = 1;
    update_events(p);
    update_events(c);
}

static void on_relay_writable(conn_t *c)
{
    if (flush_out(c) < 0) {
        pair_kill(c);
        return;
    }
    c->last = time(NULL);
    conn_t *p = c->peer;
    if (p->paused && !buf_pending(&c->out))
        p->paused = 0;
    update_events(c);
    update_events(p);
    check_done(c);
    check_done(p);
}

static void on_upstream_connect(conn_t *u)
{
    int err = 0;
    socklen_t len = sizeof err;
    conn_t *c = u->peer;
    getsockopt(u->fd, SOL_SOCKET, SO_ERROR, &err, &len);
    if (err) {
        send_error(c, "502 Bad Gateway", "FPGate: no se pudo contactar al servidor\n");
        pair_kill(u);
        return;
    }
    u->state = c->state = ST_RELAY;
    u->last = c->last = time(NULL);
    update_events(u);
    update_events(c);
}

/* ---- main loop -------------------------------------------------------- */

static void on_event(conn_t *c, uint32_t ev)
{
    if (c->dead)
        return;
    switch (c->state) {
    case ST_HEADERS:
        if (ev & (EPOLLERR | EPOLLHUP))
            conn_kill(c);
        else if (ev & EPOLLIN)
            on_client_head(c);
        break;
    case ST_CONNECTING:
        if (c->is_client) {
            if (ev & (EPOLLERR | EPOLLHUP))
                pair_kill(c);
        } else if (ev & (EPOLLOUT | EPOLLERR | EPOLLHUP)) {
            on_upstream_connect(c);
        }
        break;
    case ST_RELAY:
        if (ev & EPOLLERR) {
            pair_kill(c);
            break;
        }
        if (ev & EPOLLOUT)
            on_relay_writable(c);
        if (!c->dead && (ev & (EPOLLIN | EPOLLHUP)))
            on_relay_readable(c);
        break;
    default:
        break;
    }
}

static void accept_clients(conn_t *l)
{
    for (;;) {
        int fd = accept4(l->fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0)
            return;
        if (g_nconns >= MAX_CONNS) {
            close(fd);
            continue;
        }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        conn_t *c = conn_new(fd, 1, ST_HEADERS);
        if (!c || !(c->hdr = malloc(HDR_MAX))) {
            if (c)
                conn_kill(c);
            else
                close(fd);
            continue;
        }
        set_events(c, EPOLLIN, 1);
    }
}

int proxy_run(int port, volatile sig_atomic_t *stop)
{
    int lfd = socket(AF_INET6, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    int v6 = lfd >= 0;
    if (!v6)
        lfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (lfd < 0) {
        perror("socket");
        return -1;
    }
    int one = 1, zero = 0;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    int rc;
    if (v6) {
        struct sockaddr_in6 a6;
        memset(&a6, 0, sizeof a6);
        a6.sin6_family = AF_INET6;
        a6.sin6_addr = in6addr_any;
        a6.sin6_port = htons((uint16_t)port);
        setsockopt(lfd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);
        rc = bind(lfd, (struct sockaddr *)&a6, sizeof a6);
    } else {
        struct sockaddr_in a4;
        memset(&a4, 0, sizeof a4);
        a4.sin_family = AF_INET;
        a4.sin_addr.s_addr = htonl(INADDR_ANY);
        a4.sin_port = htons((uint16_t)port);
        rc = bind(lfd, (struct sockaddr *)&a4, sizeof a4);
    }
    if (rc != 0 || listen(lfd, 128) != 0) {
        perror("bind/listen");
        close(lfd);
        return -1;
    }

    g_ep = epoll_create1(EPOLL_CLOEXEC);
    if (g_ep < 0) {
        perror("epoll_create1");
        close(lfd);
        return -1;
    }
    conn_t listener;
    memset(&listener, 0, sizeof listener);
    listener.fd = lfd;
    listener.state = ST_LISTEN;
    set_events(&listener, EPOLLIN, 1);
    fprintf(stderr, "fpgate-proxy listening on :%d\n", port);

    struct epoll_event evs[64];
    time_t last_sweep = time(NULL);
    while (!*stop) {
        int n = epoll_wait(g_ep, evs, 64, 1000);
        for (int i = 0; i < n; i++) {
            conn_t *c = evs[i].data.ptr;
            if (c == &listener)
                accept_clients(&listener);
            else
                on_event(c, evs[i].events);
        }
        time_t now = time(NULL);
        if (now != last_sweep) {
            last_sweep = now;
            for (conn_t *c = g_all; c; c = c->next)
                if (now - c->last > IDLE_SECS && !c->dead)
                    pair_kill(c);
        }
        reap_dead();
    }

    while (g_all)
        conn_kill(g_all);
    reap_dead();
    close(g_ep);
    close(lfd);
    return 0;
}
