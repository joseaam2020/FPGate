/*
 * mock_accel: a stand-in for fpgate-accel that runs on the board itself.
 *
 *   mock_accel SOCKET [--mean | --score X] [--latency-ms N] [--quiet]
 *
 * --mean       score = mean pixel value / 255, so bright images are "NSFW": a way
 *              to see the replacement happen in a real browser
 * --score X    the same score for every image
 * --latency-ms N   pretend the inference takes N milliseconds (the design expects 3000-7000)
 *
 * One client at a time, one image at a time, like the real accelerator.
 * The wire format is src/accel_proto.h; mock_accel.py is the configurable version.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "accel_proto.h"

static volatile sig_atomic_t g_stop;
static void on_signal(int s) { (void)s; g_stop = 1; }

static int read_full(int fd, void *buf, size_t n)
{
    char *p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r == 0)
            return -1;
        if (r < 0) {
            if (errno == EINTR && !g_stop)
                continue;
            return -1;
        }
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n)
{
    const char *p = buf;
    while (n) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

static void reply(int fd, const accel_req_hdr_t *rq, uint8_t status, float score, uint32_t us)
{
    accel_resp_hdr_t rs;
    memset(&rs, 0, sizeof rs);
    memcpy(rs.magic, ACCEL_RESP_MAGIC, 4);
    rs.version = ACCEL_PROTO_VERSION;
    rs.type = rq->type;
    rs.status = status;
    rs.request_id = rq->request_id;
    rs.score_nsfw = score;
    rs.hw_cycles = us * 150u;                 /* 150 MHz, like the design */
    rs.latency_us = us;
    (void)write_full(fd, &rs, sizeof rs);
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    int use_mean = 1, quiet = 0;
    float fixed = 0.0f;
    unsigned latency_ms = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mean"))
            use_mean = 1;
        else if (!strcmp(argv[i], "--score") && i + 1 < argc)
            use_mean = 0, fixed = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--latency-ms") && i + 1 < argc)
            latency_ms = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
        else if (argv[i][0] != '-' && !path)
            path = argv[i];
        else {
            fprintf(stderr, "usage: %s SOCKET [--mean | --score X] [--latency-ms N] [--quiet]\n", argv[0]);
            return 2;
        }
    }
    if (!path || strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        fprintf(stderr, "a socket path is required\n");
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    unlink(path);
    int ls = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a;
    memset(&a, 0, sizeof a);
    a.sun_family = AF_UNIX;
    snprintf(a.sun_path, sizeof a.sun_path, "%s", path);
    if (ls < 0 || bind(ls, (struct sockaddr *)&a, sizeof a) != 0 || listen(ls, 4) != 0) {
        perror("mock_accel");
        return 1;
    }
    fprintf(stderr, "mock_accel on %s: %s, latency %u ms\n", path,
            use_mean ? "score = mean pixel / 255" : "fixed score", latency_ms);

    while (!g_stop) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0)
            continue;
        for (;;) {
            accel_req_hdr_t rq;
            if (read_full(fd, &rq, sizeof rq) != 0)
                break;
            if (memcmp(rq.magic, ACCEL_REQ_MAGIC, 4) != 0)
                break;
            if (rq.version != ACCEL_PROTO_VERSION || rq.payload_len > ACCEL_MAX_PAYLOAD) {
                reply(fd, &rq, ACCEL_UNSUPPORTED, 0, 0);
                break;
            }
            unsigned char *px = rq.payload_len ? malloc(rq.payload_len) : NULL;
            if (rq.payload_len && (!px || read_full(fd, px, rq.payload_len) != 0)) {
                free(px);
                break;
            }
            if (rq.type == ACCEL_MSG_STATUS) {
                reply(fd, &rq, ACCEL_OK, 0, 0);
                free(px);
                continue;
            }
            if (rq.type != ACCEL_MSG_CLASSIFY || rq.channels != 3 || rq.pixfmt != 0 ||
                rq.payload_len != (uint32_t)rq.width * rq.height * 3u || !rq.payload_len) {
                reply(fd, &rq, ACCEL_UNSUPPORTED, 0, 0);
                free(px);
                continue;
            }
            float score = fixed;
            if (use_mean) {
                unsigned long long sum = 0;
                for (uint32_t k = 0; k < rq.payload_len; k++)
                    sum += px[k];
                score = (float)((double)sum / rq.payload_len / 255.0);
            }
            free(px);
            if (latency_ms)
                usleep(latency_ms * 1000u);
            if (!quiet)
                fprintf(stderr, "classify id=%u %ux%u -> %.3f\n", rq.request_id, rq.width,
                        rq.height, (double)score);
            reply(fd, &rq, ACCEL_OK, score, latency_ms * 1000u);
        }
        close(fd);
    }
    unlink(path);
    return 0;
}
