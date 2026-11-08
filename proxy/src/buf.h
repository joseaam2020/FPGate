#ifndef FPGATE_BUF_H
#define FPGATE_BUF_H

#include <stddef.h>

/* Growable byte buffer with a read offset (a simple FIFO). */
typedef struct {
    char *p;
    size_t len, off, cap;
    int secret;             /* growth copies then wipes the old block (RS-01) */
} buf_t;

static inline size_t buf_pending(const buf_t *b) { return b->len - b->off; }

int buf_append(buf_t *b, const void *d, size_t n);
int buf_printf(buf_t *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void buf_free(buf_t *b);
/* Overwrites the whole allocation with zeros before freeing (RS-01). */
void buf_wipe_free(buf_t *b);

#endif
