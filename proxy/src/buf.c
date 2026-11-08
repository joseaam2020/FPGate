#define _GNU_SOURCE
#include "buf.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int buf_append(buf_t *b, const void *d, size_t n)
{
    if (b->off == b->len)
        b->off = b->len = 0;
    if (b->len + n > b->cap) {
        size_t nc = b->cap ? b->cap : 4096;
        while (nc < b->len + n)
            nc *= 2;
        char *np;
        if (b->secret) {
            /* realloc could leave the old block, with its contents, in the heap */
            np = malloc(nc);
            if (!np)
                return -1;
            if (b->p) {
                memcpy(np, b->p, b->len);
                explicit_bzero(b->p, b->cap);
                free(b->p);
            }
        } else {
            np = realloc(b->p, nc);
            if (!np)
                return -1;
        }
        b->p = np;
        b->cap = nc;
    }
    if (n)
        memcpy(b->p + b->len, d, n);
    b->len += n;
    return 0;
}

int buf_printf(buf_t *b, const char *fmt, ...)
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

void buf_free(buf_t *b)
{
    int secret = b->secret;
    free(b->p);
    memset(b, 0, sizeof *b);
    b->secret = secret;
}

void buf_wipe_free(buf_t *b)
{
    if (b->p)
        explicit_bzero(b->p, b->cap);
    buf_free(b);
}
