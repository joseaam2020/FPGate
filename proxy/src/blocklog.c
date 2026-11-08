#include "blocklog.h"

#include <ctype.h>
#include <pthread.h>
#include <sqlite3.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define QUEUE_MAX 256
#define BATCH_DELAY_US 250000   /* lets a burst of blocks share one transaction */

static const char *SCHEMA =
    "CREATE TABLE IF NOT EXISTS blocks ("
    "  id        INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  ts        INTEGER NOT NULL,"
    "  host      TEXT    NOT NULL,"
    "  score     REAL,"
    "  threshold REAL    NOT NULL,"
    "  reason    TEXT    NOT NULL);"
    "CREATE INDEX IF NOT EXISTS idx_blocks_ts ON blocks(ts DESC);";

static struct {
    int open;
    sqlite3 *wdb, *rdb;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int stop;
    blocklog_row_t q[QUEUE_MAX];
    int head, count;
    pthread_mutex_t rmu;        /* serialises readers on rdb */
    blocklog_stats_t st;
} L = {.mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER,
       .rmu = PTHREAD_MUTEX_INITIALIZER};

/* Reduces "Host:443", "[::1]:443" or "WWW.Example.COM" to the bare lower-case name. */
static void host_only(char *dst, size_t cap, const char *host)
{
    const char *s = host ? host : "";
    size_t n = strlen(s);
    if (*s == '[') {
        const char *e = strchr(s, ']');
        s++;
        n = e ? (size_t)(e - s) : strlen(s);
    } else {
        const char *c = strchr(s, ':');
        if (c && !strchr(c + 1, ':'))   /* one colon: a port; several: bare IPv6 */
            n = (size_t)(c - s);
    }
    if (n >= cap)
        n = cap - 1;
    for (size_t k = 0; k < n; k++)
        dst[k] = (char)tolower((unsigned char)s[k]);
    dst[n] = 0;
}

static void write_batch(blocklog_row_t *rows, int n)
{
    sqlite3_stmt *ins = NULL;
    int ok = sqlite3_exec(L.wdb, "BEGIN", NULL, NULL, NULL) == SQLITE_OK &&
             sqlite3_prepare_v2(L.wdb,
                 "INSERT INTO blocks(ts,host,score,threshold,reason) VALUES(?,?,?,?,?)", -1,
                 &ins, NULL) == SQLITE_OK;
    int done = 0;
    for (int k = 0; ok && k < n; k++) {
        sqlite3_bind_int64(ins, 1, rows[k].ts);
        sqlite3_bind_text(ins, 2, rows[k].host, -1, SQLITE_STATIC);
        if (rows[k].score < 0)
            sqlite3_bind_null(ins, 3);
        else
            sqlite3_bind_double(ins, 3, rows[k].score);
        sqlite3_bind_double(ins, 4, rows[k].threshold);
        sqlite3_bind_text(ins, 5, rows[k].reason, -1, SQLITE_STATIC);
        if (sqlite3_step(ins) != SQLITE_DONE)
            ok = 0;
        else
            done++;
        sqlite3_reset(ins);
    }
    sqlite3_finalize(ins);
    if (ok && sqlite3_exec(L.wdb, "COMMIT", NULL, NULL, NULL) == SQLITE_OK) {
        pthread_mutex_lock(&L.mu);
        L.st.written += (unsigned long)n;
        pthread_mutex_unlock(&L.mu);
        return;
    }
    sqlite3_exec(L.wdb, "ROLLBACK", NULL, NULL, NULL);
    pthread_mutex_lock(&L.mu);
    L.st.failed += (unsigned long)n;
    pthread_mutex_unlock(&L.mu);
    fprintf(stderr, "blocklog: could not write %d rows (%d inserted before the error): %s\n", n,
            done, sqlite3_errmsg(L.wdb));
}

static void *writer(void *arg)
{
    (void)arg;
    static blocklog_row_t batch[QUEUE_MAX];
    pthread_mutex_lock(&L.mu);
    for (;;) {
        while (L.count == 0 && !L.stop)
            pthread_cond_wait(&L.cv, &L.mu);
        if (L.count == 0 && L.stop)
            break;
        if (!L.stop) {
            pthread_mutex_unlock(&L.mu);
            usleep(BATCH_DELAY_US);
            pthread_mutex_lock(&L.mu);
        }
        int n = L.count;
        for (int k = 0; k < n; k++)
            batch[k] = L.q[(L.head + k) % QUEUE_MAX];
        L.head = (L.head + n) % QUEUE_MAX;
        L.count = 0;
        pthread_mutex_unlock(&L.mu);
        write_batch(batch, n);
        pthread_mutex_lock(&L.mu);
    }
    pthread_mutex_unlock(&L.mu);
    return NULL;
}

int blocklog_open(const char *path)
{
    if (L.open)
        return 0;
    char *err = NULL;
    if (sqlite3_open_v2(path, &L.wdb, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                          SQLITE_OPEN_FULLMUTEX, NULL) != SQLITE_OK) {
        fprintf(stderr, "blocklog: cannot open %s: %s\n", path, L.wdb ? sqlite3_errmsg(L.wdb) : "?");
        goto fail;
    }
    sqlite3_busy_timeout(L.wdb, 2000);
    /* WAL plus NORMAL keeps microSD writes few; a crash can lose the last blocks only. */
    if (sqlite3_exec(L.wdb, "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;", NULL, NULL,
                     &err) != SQLITE_OK ||
        sqlite3_exec(L.wdb, SCHEMA, NULL, NULL, &err) != SQLITE_OK) {
        fprintf(stderr, "blocklog: %s: %s\n", path, err ? err : "?");
        sqlite3_free(err);
        goto fail;
    }
    if (sqlite3_open_v2(path, &L.rdb, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, NULL) !=
        SQLITE_OK) {
        fprintf(stderr, "blocklog: cannot reopen %s for reading\n", path);
        goto fail;
    }
    sqlite3_busy_timeout(L.rdb, 2000);
    L.stop = 0;
    L.head = L.count = 0;
    memset(&L.st, 0, sizeof L.st);
    if (pthread_create(&L.thread, NULL, writer, NULL) != 0)
        goto fail;
    L.open = 1;
    return 0;
fail:
    if (L.rdb)
        sqlite3_close(L.rdb);
    if (L.wdb)
        sqlite3_close(L.wdb);
    L.rdb = L.wdb = NULL;
    return -1;
}

void blocklog_close(void)
{
    if (!L.open)
        return;
    pthread_mutex_lock(&L.mu);
    L.stop = 1;
    pthread_cond_signal(&L.cv);
    pthread_mutex_unlock(&L.mu);
    pthread_join(L.thread, NULL);
    sqlite3_close(L.rdb);
    sqlite3_close(L.wdb);
    L.rdb = L.wdb = NULL;
    L.open = 0;
}

int blocklog_enabled(void) { return L.open; }

const blocklog_stats_t *blocklog_stats(void) { return &L.st; }

void blocklog_add(const char *host, double score, double threshold, const char *reason)
{
    if (!L.open)
        return;
    blocklog_row_t r = {0};
    r.ts = (long long)time(NULL);
    host_only(r.host, sizeof r.host, host);
    r.score = score;
    r.threshold = threshold;
    snprintf(r.reason, sizeof r.reason, "%s", reason ? reason : "unknown");

    pthread_mutex_lock(&L.mu);
    if (L.count == QUEUE_MAX) {
        L.st.dropped++;
    } else {
        L.q[(L.head + L.count) % QUEUE_MAX] = r;
        L.count++;
        L.st.queued++;
        pthread_cond_signal(&L.cv);
    }
    pthread_mutex_unlock(&L.mu);
}

int blocklog_page(int page, int size, blocklog_row_t *rows, int max, long long *total)
{
    if (!L.open || page < 1 || size < 1)
        return -1;
    if (size > max)
        size = max;
    int n = -1;
    sqlite3_stmt *q = NULL;
    pthread_mutex_lock(&L.rmu);
    if (total) {
        *total = 0;
        if (sqlite3_prepare_v2(L.rdb, "SELECT COUNT(*) FROM blocks", -1, &q, NULL) == SQLITE_OK &&
            sqlite3_step(q) == SQLITE_ROW)
            *total = sqlite3_column_int64(q, 0);
        sqlite3_finalize(q);
        q = NULL;
    }
    if (sqlite3_prepare_v2(L.rdb,
            "SELECT id,ts,host,score,threshold,reason FROM blocks ORDER BY id DESC LIMIT ? OFFSET ?",
            -1, &q, NULL) == SQLITE_OK) {
        sqlite3_bind_int(q, 1, size);
        sqlite3_bind_int64(q, 2, (long long)(page - 1) * size);
        n = 0;
        while (n < size && sqlite3_step(q) == SQLITE_ROW) {
            blocklog_row_t *r = &rows[n++];
            r->id = sqlite3_column_int64(q, 0);
            r->ts = sqlite3_column_int64(q, 1);
            snprintf(r->host, sizeof r->host, "%s", (const char *)sqlite3_column_text(q, 2));
            r->score = sqlite3_column_type(q, 3) == SQLITE_NULL ? -1.0 : sqlite3_column_double(q, 3);
            r->threshold = sqlite3_column_double(q, 4);
            snprintf(r->reason, sizeof r->reason, "%s", (const char *)sqlite3_column_text(q, 5));
        }
    }
    sqlite3_finalize(q);
    pthread_mutex_unlock(&L.rmu);
    return n;
}
