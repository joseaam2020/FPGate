#ifndef FPGATE_BLOCKLOG_H
#define FPGATE_BLOCKLOG_H

#include <stddef.h>

/*
 * Log of blocked images (RF-10, RF-11, RS-05), kept in a SQLite file.
 *
 * Only metadata is stored: when, the host (no path, no query), the score and
 * threshold that applied, and why. Never image bytes or URLs.
 *
 * Writes never block the event loop: blocklog_add() queues the row and a
 * writer thread commits them in batches, so the microSD sees a few writes per
 * second at most (WAL, synchronous=NORMAL). If the queue is full the row is
 * dropped and counted rather than slowing the proxy down.
 */

#define BLOCKLOG_HOST_MAX 254
#define BLOCKLOG_REASON_MAX 31

typedef struct {
    long long id;
    long long ts;               /* epoch seconds, UTC */
    char host[BLOCKLOG_HOST_MAX + 1];
    double score;               /* negative when there was no score (failures) */
    double threshold;
    char reason[BLOCKLOG_REASON_MAX + 1];
} blocklog_row_t;

typedef struct {
    unsigned long queued, written, dropped, failed;
} blocklog_stats_t;

/* Opens (creating if needed) the database and starts the writer. 0, or -1. */
int blocklog_open(const char *path);
/* Flushes what is queued and stops the writer. Safe without a prior open. */
void blocklog_close(void);
int blocklog_enabled(void);
const blocklog_stats_t *blocklog_stats(void);

/*
 * Queues one block. `host` may carry a port or brackets; only the name is kept.
 * `score` < 0 means there was none.
 */
void blocklog_add(const char *host, double score, double threshold, const char *reason);

/*
 * Newest first. Fills up to `max` rows of page number `page` (1-based) and
 * returns how many, or -1 on error. `total` receives the number of rows.
 */
int blocklog_page(int page, int size, blocklog_row_t *rows, int max, long long *total);

#endif
