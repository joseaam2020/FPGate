#define _GNU_SOURCE
#include "admin.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "accel_client.h"
#include "auth.h"
#include "blocklog.h"
#include "inspect.h"
#include "picohttpparser.h"
#include "settings.h"
#include "substitute.h"
#include "tls_mitm.h"
#include "web_assets.h"

#define MAX_CONNS      8
#define HEAD_MAX       8192
#define BODY_MAX       (2u * 1024 * 1024 + 4096)   /* the stand-in image allows 2 MiB */
#define IDLE_SECS      20
#define MAX_HEADERS    32
#define COOKIE_NAME    "fpgate_session"
#define LOG_PAGE_MAX   50

enum { A_FREE, A_HANDSHAKE, A_READ, A_WRITE };

typedef struct {
    int state;
    int fd;
    tls_session_t *tls;
    char *req;                  /* what was received so far, NUL-terminated */
    size_t len, cap;
    buf_t out;
    time_t last;
    uint32_t events;
    char local_ip[64];
} aconn_t;

static struct {
    int ep, lfd;
    admin_cfg_t cfg;
    char ca_path[300];
    aconn_t c[MAX_CONNS];
} G = {.lfd = -1, .ep = -1};

static char g_listen_tag;

/* ---- small helpers ------------------------------------------------------------------------- */

static const char *status_text(int s)
{
    switch (s) {
    case 200: return "OK";
    case 302: return "Found";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    default:  return "Internal Server Error";
    }
}

static void respond(buf_t *o, int status, const char *ctype, const void *body, size_t n,
                    const char *extra, int head_only)
{
    buf_printf(o,
               "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
               "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
               "X-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\n"
               "Content-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; "
               "script-src 'self' 'unsafe-inline'; img-src 'self' data:\r\n"
               "Connection: close\r\n%s\r\n",
               status, status_text(status), ctype, n, extra ? extra : "");
    if (!head_only && n)
        buf_append(o, body, n);
}

/* Appends `s` as the inside of a JSON string. */
static void json_escape(buf_t *b, const char *s)
{
    for (; *s; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == '"' || ch == '\\')
            buf_printf(b, "\\%c", ch);
        else if (ch < 0x20)
            buf_printf(b, "\\u%04x", ch);
        else
            buf_append(b, s, 1);
    }
}

static void respond_json(buf_t *o, int status, const buf_t *json, const char *extra)
{
    respond(o, status, "application/json; charset=utf-8", json->p, json->len, extra, 0);
}

/* Error body: a stable code for programs and a sentence for people (RU-06). */
static void respond_error(buf_t *o, int status, const char *code, const char *message)
{
    buf_t j = {0};
    buf_printf(&j, "{\"error\":\"%s\",\"message\":\"", code);
    json_escape(&j, message);
    buf_append(&j, "\"}", 2);
    respond_json(o, status, &j, NULL);
    buf_free(&j);
}

/*
 * Reads the value of "key" from a flat JSON object into out. Handles strings
 * (with \" and \\) and bare numbers; nothing more is needed here.
 */
static int json_get(const char *body, const char *key, char *out, size_t cap)
{
    char pat[48];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p)
        return -1;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
        p++;
    if (*p++ != ':')
        return -1;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
        p++;
    size_t n = 0;
    if (*p == '"') {
        for (p++; *p && *p != '"'; p++) {
            if (*p == '\\') {
                p++;
                if (*p != '"' && *p != '\\' && *p != '/')
                    return -1;
            }
            if (n + 1 >= cap)
                return -1;
            out[n++] = *p;
        }
        if (*p != '"')
            return -1;
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\r' && *p != '\n') {
            if (n + 1 >= cap)
                return -1;
            out[n++] = *p++;
        }
    }
    out[n] = 0;
    return 0;
}

static const struct phr_header *find_header(const struct phr_header *hs, size_t nh, const char *name)
{
    size_t l = strlen(name);
    for (size_t i = 0; i < nh; i++)
        if (hs[i].name_len == l && strncasecmp(hs[i].name, name, l) == 0)
            return &hs[i];
    return NULL;
}

/* The session token in the Cookie header, or "" . */
static void cookie_token(const struct phr_header *hs, size_t nh, char out[AUTH_TOKEN_LEN + 1])
{
    out[0] = 0;
    const struct phr_header *h = find_header(hs, nh, "cookie");
    if (!h)
        return;
    char val[512];
    snprintf(val, sizeof val, "%.*s", (int)h->value_len, h->value);
    const char *p = strstr(val, COOKIE_NAME "=");
    if (!p || (p != val && p[-1] != ' ' && p[-1] != ';'))
        return;
    p += strlen(COOKIE_NAME "=");
    size_t n = strcspn(p, "; ");
    if (n == AUTH_TOKEN_LEN) {
        memcpy(out, p, n);
        out[n] = 0;
    }
}

/* Appends `data` with every `key` replaced by `value`. */
static void append_subst(buf_t *b, const char *data, size_t n, const char *key, const char *value)
{
    size_t kl = strlen(key), i = 0, from = 0;
    while (i + kl <= n) {
        if (memcmp(data + i, key, kl) == 0) {
            buf_append(b, data + from, i - from);
            buf_append(b, value, strlen(value));
            i += kl;
            from = i;
        } else {
            i++;
        }
    }
    buf_append(b, data + from, n - from);
}

static void send_page(buf_t *o, const unsigned char *page, unsigned len, const char *local_ip,
                      int head_only)
{
    char addr[96];
    snprintf(addr, sizeof addr, "%s:%d", local_ip[0] ? local_ip : "IP-DE-LA-PLACA", G.cfg.proxy_port);
    buf_t body = {0};
    append_subst(&body, (const char *)page, len, "{{PROXY_ADDR}}", addr);
    respond(o, 200, "text/html; charset=utf-8", body.p, body.len, NULL, head_only);
    buf_free(&body);
}

static void send_ca(buf_t *o, int head_only)
{
    FILE *f = fopen(G.ca_path, "rb");
    char data[4096];
    size_t n = f ? fread(data, 1, sizeof data, f) : 0;
    if (f)
        fclose(f);
    if (n == 0) {
        respond_error(o, 404, "ca_missing", "El certificado todavía no está disponible.");
        return;
    }
    respond(o, 200, "application/x-x509-ca-cert", data, n,
            "Content-Disposition: attachment; filename=\"fpgate-ca.crt\"\r\n", head_only);
}

/* ---- API ------------------------------------------------------------------------------------- */

static const char *status_name(void)
{
    if (!accel_enabled())
        return "config_inicial";
    return accel_degraded() ? "degradado" : "normal";
}

static void api_state(buf_t *o)
{
    const inspect_cfg_t *ic = inspect_config();
    const inspect_stats_t *is = inspect_stats();
    const accel_stats_t *as = accel_stats();
    const char *ctype = substitute_custom_type();
    double avg_ms = as->ok ? (double)as->lat_sum_us / (double)as->ok / 1000.0 : 0.0;
    buf_t j = {0};
    buf_printf(&j,
               "{\"threshold\":%.4f,\"status\":\"%s\",\"https\":%s,\"persistent\":%s,"
               "\"substitute\":{\"custom\":%s,\"type\":\"%s\"},"
               "\"accel\":{\"latency_avg_ms\":%.2f,\"latency_min_ms\":%.2f,\"latency_max_ms\":%.2f,"
               "\"throughput\":%.2f,\"answered\":%lu,\"errors\":%lu,\"timeouts\":%lu},"
               "\"images\":{\"analysed\":%lu,\"allowed\":%lu,\"blocked\":%lu},"
               "\"blocklog\":{\"enabled\":%s,\"dropped\":%lu}}",
               (double)ic->threshold, status_name(), tls_enabled() ? "true" : "false",
               settings_persistent() ? "true" : "false", ctype ? "true" : "false", ctype ? ctype : "",
               avg_ms, as->ok ? (double)as->lat_min_us / 1000.0 : 0.0, (double)as->lat_max_us / 1000.0,
               accel_throughput(), as->ok, as->errors, as->timeouts, is->seen, is->allowed,
               is->blocked, blocklog_enabled() ? "true" : "false", blocklog_stats()->dropped);
    respond_json(o, 200, &j, NULL);
    buf_free(&j);
}

static void api_blocklog(buf_t *o, const char *query)
{
    int page = 1, size = 5;
    for (const char *p = query; p && *p;) {
        if (!strncmp(p, "page=", 5))
            page = atoi(p + 5);
        else if (!strncmp(p, "size=", 5))
            size = atoi(p + 5);
        p = strchr(p, '&');
        if (p)
            p++;
    }
    if (page < 1)
        page = 1;
    if (size < 1 || size > LOG_PAGE_MAX)
        size = 5;
    buf_t j = {0};
    if (!blocklog_enabled()) {
        buf_printf(&j, "{\"enabled\":false,\"page\":1,\"pages\":1,\"total\":0,\"rows\":[]}");
        respond_json(o, 200, &j, NULL);
        buf_free(&j);
        return;
    }
    blocklog_row_t *rows = malloc(sizeof *rows * LOG_PAGE_MAX);
    long long total = 0;
    int n = rows ? blocklog_page(page, size, rows, LOG_PAGE_MAX, &total) : -1;
    if (n < 0) {
        free(rows);
        respond_error(o, 500, "blocklog_unavailable", "No se pudo leer la bitácora.");
        return;
    }
    long long pages = total ? (total + size - 1) / size : 1;
    buf_printf(&j, "{\"enabled\":true,\"page\":%d,\"pages\":%lld,\"total\":%lld,\"rows\":[", page,
               pages, total);
    for (int i = 0; i < n; i++) {
        buf_printf(&j, "%s{\"id\":%lld,\"ts\":%lld,\"host\":\"", i ? "," : "", rows[i].id, rows[i].ts);
        json_escape(&j, rows[i].host);
        buf_append(&j, "\",\"reason\":\"", 12);
        json_escape(&j, rows[i].reason);
        buf_printf(&j, "\",\"score\":%s", rows[i].score < 0 ? "null" : "");
        if (rows[i].score >= 0)
            buf_printf(&j, "%.4f", rows[i].score);
        buf_append(&j, "}", 1);
    }
    buf_append(&j, "]}", 2);
    free(rows);
    respond_json(o, 200, &j, NULL);
    buf_free(&j);
}

static void api_threshold(buf_t *o, const char *body)
{
    char v[32], err[200];
    char *end;
    if (json_get(body, "value", v, sizeof v) != 0) {
        respond_error(o, 400, "bad_request", "Falta el valor del umbral.");
        return;
    }
    double d = strtod(v, &end);
    if (end == v || *end || settings_set_threshold(d, err, sizeof err) != 0) {
        respond_error(o, 400, "bad_threshold", "El umbral debe ser un número entre 0 y 1.");
        return;
    }
    buf_t j = {0};
    buf_printf(&j, "{\"threshold\":%.4f,\"persistent\":%s}", d, settings_persistent() ? "true" : "false");
    respond_json(o, 200, &j, NULL);
    buf_free(&j);
}

static void api_substitute_upload(buf_t *o, const char *body, size_t n)
{
    char up[400], err[200];
    snprintf(up, sizeof up, "%s/.upload", settings_dir());
    int fd = open(up, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (!f) {
        if (fd >= 0)
            close(fd);
        respond_error(o, 500, "storage", "No se pudo guardar la imagen en el dispositivo.");
        return;
    }
    int ok = fwrite(body, 1, n, f) == n;
    ok = fclose(f) == 0 && ok;
    if (!ok) {
        unlink(up);
        respond_error(o, 500, "storage", "No se pudo guardar la imagen en el dispositivo.");
        return;
    }
    if (settings_set_substitute(up, err, sizeof err) != 0) {
        fprintf(stderr, "admin: stand-in image refused: %s\n", err);
        respond_error(o, 400, "bad_image", "La imagen no es válida. Usa un JPG, PNG o WebP de hasta 2 MB.");
        return;
    }
    buf_t j = {0};
    buf_printf(&j, "{\"type\":\"%s\"}", substitute_custom_type());
    respond_json(o, 200, &j, NULL);
    buf_free(&j);
}

static void api_login(buf_t *o, const char *body)
{
    char user[64], pass[160], token[AUTH_TOKEN_LEN + 1];
    unsigned retry = 0;
    if (json_get(body, "user", user, sizeof user) != 0 || json_get(body, "password", pass, sizeof pass) != 0 ||
        !user[0] || !pass[0]) {
        respond_error(o, 400, "missing_fields", "Completa el usuario y la contraseña.");
        return;
    }
    auth_result_t r = auth_login(user, pass, token, &retry);
    explicit_bzero(pass, sizeof pass);
    if (r == AUTH_LOCKED) {
        char m[120];
        snprintf(m, sizeof m, "Demasiados intentos. Espera %u segundos e intenta de nuevo.", retry);
        respond_error(o, 429, "locked", m);
        return;
    }
    if (r != AUTH_OK) {
        respond_error(o, 401, "bad_credentials", "Usuario o contraseña incorrectos. Intenta de nuevo.");
        return;
    }
    char cookie[160];
    snprintf(cookie, sizeof cookie, "Set-Cookie: " COOKIE_NAME "=%s; Path=/; HttpOnly; Secure; SameSite=Strict\r\n", token);
    buf_t j = {0};
    buf_printf(&j, "{\"ok\":true}");
    respond_json(o, 200, &j, cookie);
    buf_free(&j);
}

/* ---- routing --------------------------------------------------------------------------------- */

static int same_origin(const struct phr_header *hs, size_t nh)
{
    const struct phr_header *org = find_header(hs, nh, "origin"), *host = find_header(hs, nh, "host");
    if (!org)
        return 1;                   /* not a cross-site browser request */
    if (!host)
        return 0;
    const char *p = org->value;
    size_t n = org->value_len;
    if (n >= 8 && strncasecmp(p, "https://", 8) == 0) {
        p += 8;
        n -= 8;
    } else {
        return 0;
    }
    return n == host->value_len && strncasecmp(p, host->value, n) == 0;
}

static void route(aconn_t *c, const char *method, size_t mlen, const char *path, size_t plen,
                  const struct phr_header *hs, size_t nh, const char *body, size_t blen)
{
    buf_t *o = &c->out;
    char target[512];
    snprintf(target, sizeof target, "%.*s", (int)plen, path);
    char *query = strchr(target, '?');
    if (query)
        *query++ = 0;
    int get = mlen == 3 && !memcmp(method, "GET", 3);
    int head = mlen == 4 && !memcmp(method, "HEAD", 4);
    int post = mlen == 4 && !memcmp(method, "POST", 4);

    char token[AUTH_TOKEN_LEN + 1];
    cookie_token(hs, nh, token);
    int logged = auth_session_valid(token);

    /* Pages and the public certificate. */
    if (get || head) {
        if (!strcmp(target, "/")) {
            if (logged)
                respond(o, 302, "text/plain", "", 0, "Location: /dashboard\r\n", head);
            else
                send_page(o, web_login, web_login_len, c->local_ip, head);
            return;
        }
        if (!strcmp(target, "/dashboard")) {
            if (!logged)
                respond(o, 302, "text/plain", "", 0, "Location: /\r\n", head);
            else
                send_page(o, web_dashboard, web_dashboard_len, c->local_ip, head);
            return;
        }
        if (!strcmp(target, "/certificado")) {
            send_page(o, web_certificate, web_certificate_len, c->local_ip, head);
            return;
        }
        if (!strcmp(target, "/ca.crt")) {
            send_ca(o, head);
            return;
        }
    }

    if (strncmp(target, "/api/", 5) != 0) {
        respond_error(o, 404, "not_found", "Esa página no existe.");
        return;
    }
    if (post && !same_origin(hs, nh)) {
        respond_error(o, 403, "cross_site", "Solicitud rechazada.");
        return;
    }
    if (post && !strcmp(target, "/api/login")) {
        api_login(o, body);
        return;
    }
    if (!logged) {
        respond_error(o, 401, "login_required", "Inicia sesión para continuar.");
        return;
    }
    if (get && !strcmp(target, "/api/state"))
        api_state(o);
    else if (get && !strcmp(target, "/api/blocklog"))
        api_blocklog(o, query);
    else if (get && !strcmp(target, "/api/substitute")) {
        buf_t img = {0};
        if (substitute_response(&img, 192, 144, 1, "preview") == 0 && img.len)
            buf_append(o, img.p, img.len);
        else
            respond_error(o, 500, "preview", "No se pudo preparar la vista previa.");
        buf_free(&img);
    } else if (post && !strcmp(target, "/api/threshold"))
        api_threshold(o, body);
    else if (post && !strcmp(target, "/api/substitute"))
        api_substitute_upload(o, body, blen);
    else if (post && !strcmp(target, "/api/logout")) {
        auth_logout(token);
        respond(o, 200, "application/json", "{\"ok\":true}", 11,
                "Set-Cookie: " COOKIE_NAME "=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict\r\n", 0);
    } else
        respond_error(o, 405, "method_not_allowed", "Operación no permitida.");
}

/* Returns 1 when the request is complete and `out` is filled, 0 when more bytes are needed. */
static int try_request(aconn_t *c)
{
    const char *method, *path;
    size_t mlen, plen, nh = MAX_HEADERS;
    int minor;
    struct phr_header hs[MAX_HEADERS];
    int r = phr_parse_request(c->req, c->len, &method, &mlen, &path, &plen, &minor, hs, &nh, 0);
    if (r == -2) {
        if (c->len >= HEAD_MAX) {
            respond_error(&c->out, 431, "head_too_large", "Solicitud demasiado grande.");
            return 1;
        }
        return 0;
    }
    if (r < 0) {
        respond_error(&c->out, 400, "bad_request", "Solicitud inválida.");
        return 1;
    }
    size_t blen = 0;
    const struct phr_header *cl = find_header(hs, nh, "content-length");
    if (find_header(hs, nh, "transfer-encoding")) {
        respond_error(&c->out, 400, "bad_request", "Solicitud inválida.");
        return 1;
    }
    if (cl) {
        char v[24];
        snprintf(v, sizeof v, "%.*s", (int)cl->value_len, cl->value);
        char *end;
        unsigned long long n = strtoull(v, &end, 10);
        if (end == v || *end || n > BODY_MAX) {
            respond_error(&c->out, 413, "too_large", "El archivo es demasiado grande.");
            return 1;
        }
        blen = (size_t)n;
    }
    if (c->len - (size_t)r < blen)
        return 0;
    route(c, method, mlen, path, plen, hs, nh, c->req + r, blen);
    return 1;
}

/* Names for the interface's certificate: host name, host name.local, localhost, and the address in use. */
static void cert_names(const char *local_ip, char *out, size_t cap)
{
    char me[64] = "";
    size_t n = 0;
    gethostname(me, sizeof me - 1);
    out[0] = 0;
    if (me[0]) {
        n += (size_t)snprintf(out + n, cap - n, "%s,", me);
        if (!strchr(me, '.'))
            n += (size_t)snprintf(out + n, cap - n, "%s.local,", me);
    }
    n += (size_t)snprintf(out + n, cap - n, "localhost,127.0.0.1");
    if (local_ip[0] && strcmp(local_ip, "127.0.0.1") != 0 && n < cap)
        snprintf(out + n, cap - n, ",%s", local_ip);
}

/* ---- connections ---------------------------------------------------------------------------- */

static void set_events(aconn_t *c, uint32_t ev, int add)
{
    struct epoll_event e;
    memset(&e, 0, sizeof e);
    e.events = ev;
    e.data.ptr = c;
    epoll_ctl(G.ep, add ? EPOLL_CTL_ADD : EPOLL_CTL_MOD, c->fd, &e);
    c->events = ev;
}

static void conn_close(aconn_t *c)
{
    if (c->state == A_FREE)
        return;
    if (c->tls) {
        tls_close_notify(c->tls);
        tls_free(c->tls);
    }
    close(c->fd);
    if (c->req) {
        explicit_bzero(c->req, c->cap);    /* may hold a password */
        free(c->req);
    }
    buf_wipe_free(&c->out);
    memset(c, 0, sizeof *c);
    c->fd = -1;
}

static void do_handshake(aconn_t *c)
{
    int r = tls_handshake(c->tls);
    if (r == 0) {
        c->state = A_READ;
        set_events(c, EPOLLIN, 0);
        return;
    }
    if (r == TLS_ERR) {
        conn_close(c);
        return;
    }
    uint32_t ev = r == TLS_WANT_WRITE ? EPOLLOUT : EPOLLIN;
    if (ev != c->events)
        set_events(c, ev, 0);
}

static void do_read(aconn_t *c)
{
    for (;;) {
        if (c->cap - c->len < 2048) {
            size_t ncap = c->cap ? c->cap * 2 : 4096;
            if (ncap > HEAD_MAX + BODY_MAX + 8)
                ncap = HEAD_MAX + BODY_MAX + 8;
            if (ncap <= c->len + 1) {
                respond_error(&c->out, 413, "too_large", "Solicitud demasiado grande.");
                c->state = A_WRITE;
                set_events(c, EPOLLOUT, 0);
                return;
            }
            char *p = malloc(ncap);
            if (!p) {
                conn_close(c);
                return;
            }
            if (c->req) {
                memcpy(p, c->req, c->len);
                explicit_bzero(c->req, c->cap);
                free(c->req);
            }
            c->req = p;
            c->cap = ncap;
        }
        ssize_t n = tls_read(c->tls, c->req + c->len, c->cap - c->len - 1);
        if (n == TLS_WANT_READ)
            return;
        if (n <= 0) {
            conn_close(c);
            return;
        }
        c->len += (size_t)n;
        c->req[c->len] = 0;
        c->last = time(NULL);
        if (try_request(c)) {
            c->state = A_WRITE;
            set_events(c, EPOLLOUT, 0);
            return;
        }
    }
}

static void do_write(aconn_t *c)
{
    while (buf_pending(&c->out)) {
        ssize_t n = tls_write(c->tls, c->out.p + c->out.off, buf_pending(&c->out));
        if (n == TLS_WANT_WRITE)
            return;
        if (n <= 0) {
            conn_close(c);
            return;
        }
        c->out.off += (size_t)n;
    }
    conn_close(c);
}

/*
 * Starts serving an already connected, non-blocking socket. `names` are the names
 * the certificate is issued for (NULL: the board's own names and the address used).
 * Takes ownership of `fd` only when it returns 0.
 */
static int start_conn(int fd, const char *names)
{
    aconn_t *c = NULL;
    for (int i = 0; i < MAX_CONNS && !c; i++)
        if (G.c[i].state == A_FREE)
            c = &G.c[i];
    if (!c)
        return -1;
    memset(c, 0, sizeof *c);
    c->fd = fd;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    struct sockaddr_storage ss;
    socklen_t sl = sizeof ss;
    if (getsockname(fd, (struct sockaddr *)&ss, &sl) == 0) {
        char ip[64] = "";
        if (ss.ss_family == AF_INET6)
            inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&ss)->sin6_addr, ip, sizeof ip);
        else
            inet_ntop(AF_INET, &((struct sockaddr_in *)&ss)->sin_addr, ip, sizeof ip);
        snprintf(c->local_ip, sizeof c->local_ip, "%s", strncmp(ip, "::ffff:", 7) == 0 ? ip + 7 : ip);
    }
    char own[200];
    if (!names) {
        /* The address changes (DHCP); the name does not. The certificate covers both. */
        cert_names(c->local_ip, own, sizeof own);
        names = own;
    }
    c->tls = tls_server_new(&c->fd, names);
    if (!c->tls) {
        memset(c, 0, sizeof *c);
        return -1;
    }
    c->state = A_HANDSHAKE;
    c->last = time(NULL);
    set_events(c, EPOLLIN, 1);
    do_handshake(c);
    return 0;
}

static void accept_all(void)
{
    for (;;) {
        int fd = accept4(G.lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0)
            return;
        if (start_conn(fd, NULL) != 0)
            close(fd);
    }
}

/* ---- public entry points --------------------------------------------------------------------- */

void admin_configure(const admin_cfg_t *cfg)
{
    G.cfg = *cfg;
    snprintf(G.ca_path, sizeof G.ca_path, "%s", cfg->ca_cert_path ? cfg->ca_cert_path : "");
    G.cfg.ca_cert_path = G.ca_path;
}

void admin_setup_response(const char *method, size_t method_len, const char *path, size_t path_len,
                          const char *local_ip, buf_t *out)
{
    int get = method_len == 3 && !memcmp(method, "GET", 3);
    int head = method_len == 4 && !memcmp(method, "HEAD", 4);
    if (!get && !head) {
        respond_error(out, 405, "method_not_allowed", "Operación no permitida.");
        return;
    }
    size_t n = 0;
    while (n < path_len && path[n] != '?')
        n++;
    if ((n == 1 && path[0] == '/') || (n == 12 && !memcmp(path, "/certificado", 12)))
        send_page(out, web_certificate, web_certificate_len, local_ip, head);
    else if (n == 7 && !memcmp(path, "/ca.crt", 7))
        send_ca(out, head);
    else
        respond_error(out, 404, "not_found", "Esa página no existe.");
}

int admin_start(const admin_cfg_t *cfg, int epfd)
{
    admin_configure(cfg);
    if (!tls_enabled()) {
        fprintf(stderr, "admin: the interface needs the CA (-C/-K); it is not started\n");
        return -1;
    }
    if (!auth_ready()) {
        fprintf(stderr, "admin: no administrator credentials; see --set-admin-password\n");
        return -1;
    }
    G.lfd = socket(AF_INET6, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    int v6 = G.lfd >= 0;
    if (!v6)
        G.lfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (G.lfd < 0)
        return -1;
    int one = 1, zero = 0, rc;
    setsockopt(G.lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (v6) {
        struct sockaddr_in6 a;
        memset(&a, 0, sizeof a);
        a.sin6_family = AF_INET6;
        a.sin6_addr = in6addr_any;
        a.sin6_port = htons((uint16_t)cfg->port);
        setsockopt(G.lfd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);
        rc = bind(G.lfd, (struct sockaddr *)&a, sizeof a);
    } else {
        struct sockaddr_in a;
        memset(&a, 0, sizeof a);
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons((uint16_t)cfg->port);
        rc = bind(G.lfd, (struct sockaddr *)&a, sizeof a);
    }
    if (rc != 0 || listen(G.lfd, 16) != 0) {
        fprintf(stderr, "admin: cannot listen on :%d: %s\n", cfg->port, strerror(errno));
        close(G.lfd);
        G.lfd = -1;
        return -1;
    }
    G.ep = epfd;
    struct epoll_event e;
    memset(&e, 0, sizeof e);
    e.events = EPOLLIN;
    e.data.ptr = &g_listen_tag;
    epoll_ctl(epfd, EPOLL_CTL_ADD, G.lfd, &e);
    return 0;
}

void admin_stop(void)
{
    for (int i = 0; i < MAX_CONNS; i++)
        conn_close(&G.c[i]);
    if (G.lfd >= 0)
        close(G.lfd);
    G.lfd = -1;
}

int admin_enabled(void) { return G.lfd >= 0; }

int admin_adopt(int fd)
{
    return G.lfd >= 0 ? start_conn(fd, ADMIN_VHOST) : -1;
}

int admin_owns(const void *tag)
{
    if (G.lfd < 0)
        return 0;
    if (tag == &g_listen_tag)
        return 1;
    for (int i = 0; i < MAX_CONNS; i++)
        if (tag == &G.c[i])
            return 1;
    return 0;
}

void admin_on_event(void *tag, uint32_t events)
{
    if (tag == &g_listen_tag) {
        accept_all();
        return;
    }
    aconn_t *c = tag;
    if (c->state == A_FREE)
        return;
    if (events & (EPOLLERR | EPOLLHUP)) {
        conn_close(c);
        return;
    }
    if (c->state == A_HANDSHAKE)
        do_handshake(c);
    if (c->state == A_READ)
        do_read(c);
    if (c->state == A_WRITE)
        do_write(c);
}

void admin_tick(void)
{
    time_t now = time(NULL);
    for (int i = 0; i < MAX_CONNS; i++)
        if (G.c[i].state != A_FREE && now - G.c[i].last > IDLE_SECS)
            conn_close(&G.c[i]);
}
