/*
 * The HTTP framer reads whatever a server or a client sends. Input layout:
 *   byte 0: bit 0 = response stream (else request), bit 1 = the request was HEAD,
 *           bit 2 = pause after every message (hs_stop / hs_rest / hs_resume)
 *   byte 1: size of the pieces the rest is fed in (1..256), to hit every split point
 */
#include <stddef.h>
#include <stdint.h>

#include "http_stream.h"

static unsigned long g_calls;
static int g_pause, g_stopped;
static hs_stream_t *g_s;

static void on_head(void *u, const hs_head_t *h)
{
    (void)u;
    g_calls += h->nhdrs + h->raw_len;
    for (size_t i = 0; i < h->nhdrs; i++)       /* touch every byte the parser points at */
        g_calls += (h->hdrs[i].name ? (unsigned char)h->hdrs[i].name[0] : 0) + h->hdrs[i].value_len;   /* name is NULL on a folded line */
}

static void on_body(void *u, const char *raw, size_t raw_len, const char *data, size_t data_len)
{
    (void)u;
    if (raw_len)
        g_calls += (unsigned char)raw[raw_len - 1];
    if (data_len)
        g_calls += (unsigned char)data[data_len - 1];
}

static void on_end(void *u)
{
    (void)u;
    if (g_pause) {
        hs_stop(g_s);
        g_stopped = 1;
    }
}

static void on_opaque(void *u, const char *p, size_t n)
{
    (void)u;
    if (n)
        g_calls += (unsigned char)p[n - 1];
}

/* Feeds like the proxy does: after a pause, the unconsumed part is fed again. */
static int feed(const char *p, size_t n)
{
    while (n) {
        g_stopped = 0;
        if (hs_feed(g_s, p, n) < 0)
            return -1;
        if (!g_stopped)
            return 0;
        hs_resume(g_s);
        n = hs_rest(g_s, &p);
    }
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 2)
        return 0;
    int resp = data[0] & 1;
    g_pause = (data[0] >> 2) & 1;
    size_t piece = (size_t)data[1] + 1;
    const char *p = (const char *)data + 2;
    size_t n = size - 2;

    hs_pair_t pair;
    hs_pair_init(&pair);
    if (resp)
        hs_pair_push(&pair, (data[0] & 2) ? HS_METHOD_HEAD : HS_METHOD_OTHER);
    hs_callbacks_t cb = {.head = on_head, .body = on_body, .end = on_end, .opaque = on_opaque};
    g_s = hs_new(!resp, &pair, &cb, NULL);
    if (!g_s)
        return 0;
    int bad = 0;
    for (size_t off = 0; off < n && !bad; off += piece)
        bad = feed(p + off, n - off < piece ? n - off : piece) < 0;
    if (!bad)
        (void)hs_eof(g_s);
    hs_free(g_s);
    return 0;
}
