#ifndef FPGATE_AUTH_H
#define FPGATE_AUTH_H

#include <stddef.h>

/*
 * Administrator login (RS-03). The credentials file holds one line,
 * `user:iterations:salt-hex:hash-hex`, with PBKDF2-HMAC-SHA256; there is no
 * default password. A login yields a random session token kept in memory only.
 */

#define AUTH_TOKEN_LEN 32       /* hex characters, without the terminator */

/* Writes the credentials file (mode 600). Returns 0, or -1 with a message in err. */
int auth_write_credentials(const char *path, const char *user, const char *password, char *err,
                           size_t err_len);
/* Loads the file; logins are refused until this succeeds. */
int auth_load(const char *path, char *err, size_t err_len);
int auth_ready(void);

typedef enum { AUTH_OK, AUTH_BAD_CREDENTIALS, AUTH_LOCKED } auth_result_t;

/*
 * Checks a login and, when it is right, opens a session and fills `token`.
 * After AUTH_MAX_FAILS wrong attempts in a row every login is refused for a
 * while, whoever sends it (`retry_s` says how long, for the message).
 */
auth_result_t auth_login(const char *user, const char *password, char token[AUTH_TOKEN_LEN + 1],
                         unsigned *retry_s);
/* 1 when the token belongs to a live session; refreshes its idle timer. */
int auth_session_valid(const char *token);
void auth_logout(const char *token);

#endif
