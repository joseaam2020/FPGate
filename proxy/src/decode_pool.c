#define _GNU_SOURCE
#include "decode_pool.h"

#include <pthread.h>
#include <stdint.h>
#include <sys/eventfd.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 4
#define STACK_BYTES (512 * 1024)

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static dp_job_t *g_qh, *g_qt;           /* waiting */
static unsigned g_qn;
static dp_job_t *g_dh, *g_dt;           /* finished */
static pthread_t g_thr[MAX_THREADS];
static int g_nthr, g_efd = -1, g_stop;
static unsigned g_delay_ms;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void push(dp_job_t **h, dp_job_t **t, dp_job_t *j)
{
    j->next = NULL;
    if (*t)
        (*t)->next = j;
    else
        *h = j;
    *t = j;
}

static dp_job_t *pop(dp_job_t **h, dp_job_t **t)
{
    dp_job_t *j = *h;
    if (j) {
        *h = j->next;
        if (!*h)
            *t = NULL;
        j->next = NULL;
    }
    return j;
}

static void *worker(void *arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_qh && !g_stop)
            pthread_cond_wait(&g_cv, &g_mu);
        if (!g_qh) {                    /* stopping and nothing left */
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        dp_job_t *j = pop(&g_qh, &g_qt);
        g_qn--;
        pthread_mutex_unlock(&g_mu);

        if (g_delay_ms)
            usleep(g_delay_ms * 1000u);
        double t0 = now_ms();
        j->status = image_prep(j->kind, j->data, j->len, &j->cfg, &j->result);
        j->ms = now_ms() - t0;

        pthread_mutex_lock(&g_mu);
        push(&g_dh, &g_dt, j);
        pthread_mutex_unlock(&g_mu);
        uint64_t one = 1;
        if (write(g_efd, &one, sizeof one) < 0) {
            /* the counter is saturated or the pool is closing: the reactor will
             * still find the job on its next wake-up */
        }
    }
}

int dp_start(int nthreads, unsigned delay_ms)
{
    if (nthreads <= 0 || g_nthr)
        return -1;
    if (nthreads > MAX_THREADS)
        nthreads = MAX_THREADS;
    g_efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (g_efd < 0)
        return -1;
    g_stop = 0;
    g_delay_ms = delay_ms;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACK_BYTES);
    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&g_thr[g_nthr], &attr, worker, NULL) != 0)
            break;
        g_nthr++;
    }
    pthread_attr_destroy(&attr);
    if (g_nthr == 0) {
        close(g_efd);
        g_efd = -1;
        return -1;
    }
    return g_efd;
}

void dp_stop(void)
{
    if (!g_nthr)
        return;
    pthread_mutex_lock(&g_mu);
    g_stop = 1;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    for (int i = 0; i < g_nthr; i++)
        pthread_join(g_thr[i], NULL);
    g_nthr = 0;

    pthread_mutex_lock(&g_mu);
    dp_job_t *j;
    while ((j = pop(&g_qh, &g_qt))) {   /* never ran */
        j->status = PREP_NOMEM;
        push(&g_dh, &g_dt, j);
    }
    g_qn = 0;
    pthread_mutex_unlock(&g_mu);
}

int dp_active(void) { return g_nthr > 0 && !g_stop; }

int dp_submit(dp_job_t *job)
{
    pthread_mutex_lock(&g_mu);
    if (!dp_active() || g_qn >= DP_QUEUE_MAX) {
        pthread_mutex_unlock(&g_mu);
        return -1;
    }
    push(&g_qh, &g_qt, job);
    g_qn++;
    pthread_cond_signal(&g_cv);
    pthread_mutex_unlock(&g_mu);
    return 0;
}

dp_job_t *dp_take_done(void)
{
    pthread_mutex_lock(&g_mu);
    dp_job_t *j = pop(&g_dh, &g_dt);
    pthread_mutex_unlock(&g_mu);
    if (!j && g_efd >= 0) {
        uint64_t v;
        if (read(g_efd, &v, sizeof v) < 0) {
            /* nothing to drain */
        }
        /* a job may have landed between the pop and the drain */
        pthread_mutex_lock(&g_mu);
        j = pop(&g_dh, &g_dt);
        pthread_mutex_unlock(&g_mu);
    }
    return j;
}
