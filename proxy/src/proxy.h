#ifndef FPGATE_PROXY_H
#define FPGATE_PROXY_H

#include <signal.h>

#include "accel_client.h"
#include "admin.h"

/*
 * Optional link to fpgate-accel. Without it (cfg == NULL, the default) images are
 * analysed and logged but always delivered as they came. Must be called before
 * proxy_run().
 */
void proxy_set_accel(const accel_cfg_t *cfg);

/*
 * The administration interface (HTTPS, port cfg->port; 0 keeps it off) and the
 * public http://fpgate.setup page. Must be called before proxy_run().
 */
void proxy_set_admin(const admin_cfg_t *cfg);

/* `fn` runs on the event loop after *flag is set (SIGHUP); the flag is cleared first. */
void proxy_set_reload(volatile sig_atomic_t *flag, void (*fn)(void));

/*
 * Runs the proxy event loop on the given TCP port until *stop becomes
 * non-zero. Returns 0 on clean shutdown, -1 on a fatal startup error.
 */
int proxy_run(int port, volatile sig_atomic_t *stop);

#endif
