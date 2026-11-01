#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "proxy.h"

#define FPGATE_PROXY_VERSION "0.1.0"
#define DEFAULT_PORT 8080

static volatile sig_atomic_t g_stop;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

int main(int argc, char **argv)
{
    int port = DEFAULT_PORT;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            printf("fpgate-proxy %s\n", FPGATE_PROXY_VERSION);
            return 0;
        } else {
            fprintf(stderr, "usage: %s [-p port] [-v]\n", argv[0]);
            return 2;
        }
    }
    if (port < 1 || port > 65535) {
        fprintf(stderr, "invalid port\n");
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    return proxy_run(port, &g_stop) == 0 ? 0 : 1;
}
