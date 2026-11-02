#ifndef FPGATE_TLS_MITM_H
#define FPGATE_TLS_MITM_H

#include <stddef.h>
#include <sys/types.h>

/*
 * TLS interception layer (mbedTLS). Two roles per intercepted CONNECT:
 *   - server side towards the client, with a leaf certificate generated on
 *     the fly and signed by the FPGate CA;
 *   - client side towards the real server, with certificate verification.
 * All I/O is non-blocking; the fd is read through a pointer so that the
 * session always sees the owner's current descriptor.
 */

typedef struct tls_session tls_session_t;

enum {
    TLS_ERR        = -1,
    TLS_WANT_READ  = -2,
    TLS_WANT_WRITE = -3,
};

/*
 * Loads the CA (certificate + encrypted key) and the trust bundle used to
 * verify upstream servers. Returns 0 on success, -1 on failure.
 */
int tls_init(const char *ca_cert, const char *ca_key, const char *ca_pass,
             const char *trust_bundle);
void tls_cleanup(void);

/* 1 when interception is currently allowed (CA loaded and within validity). */
int tls_enabled(void);
/* Re-evaluates the CA validity (RS-07); returns the new tls_enabled(). */
int tls_ca_recheck(void);

tls_session_t *tls_server_new(int *fd, const char *host);
tls_session_t *tls_client_new(int *fd, const char *host);
void tls_free(tls_session_t *s);

/* Returns 0 when done, TLS_WANT_READ / TLS_WANT_WRITE, or TLS_ERR. */
int tls_handshake(tls_session_t *s);
/* >0 bytes, 0 on orderly close, TLS_WANT_READ, TLS_ERR. */
ssize_t tls_read(tls_session_t *s, void *buf, size_t len);
/*
 * >0 bytes accepted, TLS_WANT_WRITE, TLS_ERR. After TLS_WANT_WRITE the data
 * is retained inside the session: call again with at least the same bytes.
 */
ssize_t tls_write(tls_session_t *s, const void *buf, size_t len);
void tls_close_notify(tls_session_t *s);
const char *tls_error(const tls_session_t *s);

#endif
