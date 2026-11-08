#ifndef FPGATE_DECODE_POOL_H
#define FPGATE_DECODE_POOL_H

#include "image_prep.h"

/*
 * Worker threads that decode and resize images off the event loop, so that a
 * large photo cannot stall unrelated connections. The reactor submits a job,
 * keeps serving other sockets, and is woken through an eventfd when a job is
 * done. Workers touch nothing but the job they were given.
 */

typedef struct dp_job dp_job_t;
struct dp_job {
    /* input: the data must stay untouched until the job comes back */
    img_kind_t kind;
    const uint8_t *data;
    size_t len;
    prep_cfg_t cfg;
    /* output */
    prep_status_t status;
    prep_result_t result;
    double ms;
    /* owned by the caller */
    void *user;
    dp_job_t *next;
};

#define DP_QUEUE_MAX 32

/*
 * Starts `nthreads` workers. `delay_ms` makes each job sleep first (tests use
 * it to emulate a slow board). Returns the eventfd to poll for readability,
 * or -1 on failure. With nthreads == 0 nothing is started and -1 is returned.
 */
int dp_start(int nthreads, unsigned delay_ms);
/* Stops the workers; jobs not yet run come back as PREP_NOMEM through dp_take_done(). */
void dp_stop(void);
int dp_active(void);
/* 0 queued, -1 if the queue is full or the pool is not running. */
int dp_submit(dp_job_t *job);
/* Next finished job, or NULL. Drains the eventfd once the list is empty. */
dp_job_t *dp_take_done(void);

#endif
