#ifndef FPGATE_PROXY_H
#define FPGATE_PROXY_H

#include <signal.h>

#include "accel_client.h"

/*
 * Optional link to fpgate-accel. Without it (cfg == NULL, the default) images are
 * analysed and logged but always delivered as they came. Must be called before
 * proxy_run().
 */
void proxy_set_accel(const accel_cfg_t *cfg);

/*
 * Runs the proxy event loop on the given TCP port until *stop becomes
 * non-zero. Returns 0 on clean shutdown, -1 on a fatal startup error.
 */
int proxy_run(int port, volatile sig_atomic_t *stop);

#endif
