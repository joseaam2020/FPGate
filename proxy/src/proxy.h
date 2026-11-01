#ifndef FPGATE_PROXY_H
#define FPGATE_PROXY_H

#include <signal.h>

/*
 * Runs the proxy event loop on the given TCP port until *stop becomes
 * non-zero. Returns 0 on clean shutdown, -1 on a fatal startup error.
 */
int proxy_run(int port, volatile sig_atomic_t *stop);

#endif
