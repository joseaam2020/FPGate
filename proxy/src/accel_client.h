#ifndef FPGATE_ACCEL_CLIENT_H
#define FPGATE_ACCEL_CLIENT_H

#include <stddef.h>
#include <stdint.h>

/*
 * Non-blocking client of fpgate-accel (protocol in accel_proto.h).
 *
 *  - One request is in flight at a time, like the accelerator itself; the rest
 *    wait in a bounded FIFO. A full queue is answered at once, not delayed.
 *  - Every request is answered exactly once through its callback, whatever
 *    happens: result, error, timeout, accelerator down, queue full, cancelled.
 *  - Any failure puts the client in degraded mode: new requests are answered
 *    DOWN without touching the socket, and a periodic status probe brings the
 *    client back as soon as the accelerator reports OK again.
 *  - After a timeout the connection is dropped, so a late answer can never be
 *    mistaken for the answer to a later image (request ids guard that too).
 *
 * The socket is registered in the caller's epoll set under accel_tag().
 */

typedef enum {
    ACCEL_R_OK = 0,         /* score is valid */
    ACCEL_R_ERROR,          /* the accelerator reported an error, or the link broke */
    ACCEL_R_TIMEOUT,
    ACCEL_R_DOWN,           /* degraded mode, or could not connect */
    ACCEL_R_QUEUE_FULL,
    ACCEL_R_CANCELLED,
} accel_outcome_t;

typedef struct {
    accel_outcome_t outcome;
    float score;
    uint32_t hw_cycles;
    uint32_t latency_us;    /* time the accelerator reports for the inference */
    unsigned wait_ms;       /* time from accel_classify() to the answer, queue included */
} accel_result_t;

typedef void (*accel_done_fn)(void *user, const accel_result_t *r);

typedef struct {
    const char *path;       /* Unix socket of fpgate-accel */
    unsigned timeout_ms;    /* per request, counted from when it is sent */
    unsigned queue_max;     /* requests allowed to wait behind the one in flight */
    unsigned probe_ms;      /* status probe period while degraded */
} accel_cfg_t;

typedef struct {
    unsigned long requests, ok, errors, timeouts, down, queue_full, cancelled;
    unsigned long degraded_entries;
    uint32_t lat_min_us, lat_max_us;    /* inference time the accelerator reports */
    uint64_t lat_sum_us;
} accel_stats_t;

/* Returns 0, or -1 if the path is unusable. Does not connect yet. */
int accel_start(const accel_cfg_t *cfg, int epfd);
/* Answers everything still pending with ACCEL_R_CANCELLED and closes the socket. */
void accel_stop(void);
int accel_enabled(void);
int accel_degraded(void);
const accel_stats_t *accel_stats(void);
/* The settings in force, with defaults filled in. */
const accel_cfg_t *accel_config(void);
/* Answers per second over the last minute (0 when idle). */
double accel_throughput(void);

/* Compare with epoll_event.data.ptr to know an event belongs to the accelerator. */
void *accel_tag(void);
void accel_on_event(uint32_t events);
/* Timers (timeouts, status probes). Call at least once a second. */
void accel_tick(void);

/*
 * Queues a classification of a width x height RGB image. Takes ownership of
 * `rgb`, which is zeroed and freed once it has been sent or dropped. `done` may
 * run before this function returns.
 */
void accel_classify(uint8_t *rgb, size_t len, int width, int height, accel_done_fn done,
                    void *user);
/* Cancels the requests of `user`; each one's callback runs with ACCEL_R_CANCELLED. */
void accel_cancel(void *user);

#endif
