#ifndef FPGATE_ADMIN_H
#define FPGATE_ADMIN_H

#include <stdint.h>

#include "buf.h"

/*
 * The administration web interface (login, dashboard, certificate page) and
 * the JSON API behind it, served over HTTPS from the proxy's own event loop.
 * The certificate comes from the FPGate CA, issued for the address the browser
 * used to reach the board, so installing the CA once covers this too.
 *
 *   /               login            /dashboard      settings and block log (session)
 *   /certificado    install the CA   /ca.crt         the public CA certificate
 *   /api/...        JSON, session cookie required except login
 *
 * Each connection carries one request ("Connection: close").
 */

/*
 * https://fpgate.admin through the proxy: the browser asks the proxy for a
 * CONNECT to this name and gets the interface itself, with a certificate for the
 * name, so it works whatever the board's address or host name is.
 */
#define ADMIN_VHOST "fpgate.admin"

typedef struct {
    int port;                   /* HTTPS port of the interface */
    int proxy_port;             /* shown in the instructions of the certificate page */
    const char *ca_cert_path;   /* the public certificate offered for download */
} admin_cfg_t;

/* Returns 0, or -1 (reason on stderr). Needs tls_init() with a CA, and auth_load(). */
int admin_start(const admin_cfg_t *cfg, int epfd);
void admin_stop(void);
/* 1 when the interface is listening. */
int admin_enabled(void);
/*
 * Serves a socket the proxy already accepted (a CONNECT to ADMIN_VHOST) over TLS,
 * with a certificate for ADMIN_VHOST. Takes ownership of `fd` on 0; on -1 it is the
 * caller's to close.
 */
int admin_adopt(int fd);
/* 1 when an epoll event's data pointer belongs to the interface. */
int admin_owns(const void *tag);
void admin_on_event(void *tag, uint32_t events);
/* Drops idle connections. Call about once a second. */
void admin_tick(void);

/*
 * http://fpgate.setup through the proxy: the certificate page and ca.crt, no
 * login. Appends a complete response to `out`; `local_ip` is the proxy address
 * the client reached. Needs admin_configure() to have run.
 */
void admin_configure(const admin_cfg_t *cfg);
void admin_setup_response(const char *method, size_t method_len, const char *path,
                          size_t path_len, const char *local_ip, buf_t *out);

#endif
