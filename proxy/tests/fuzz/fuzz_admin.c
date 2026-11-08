/*
 * The administration interface's request handling: HTTP parsing, routing, JSON
 * fields, query strings, the image upload. The session check is replaced by "always
 * logged in" so that the authenticated routes are reachable too. TLS is not part of
 * this target (mbedTLS has its own fuzzing); the code under test starts at the bytes
 * the TLS layer hands over.
 */
#define _GNU_SOURCE
#include <stdio.h>
#define fprintf(...) ((void)0)      /* keep the run quiet; the messages are not under test */
#define auth_session_valid fuzz_session_valid
#include "../../src/admin.c"
#undef auth_session_valid
#undef fprintf

int fuzz_session_valid(const char *token) { (void)token; return 1; }

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    static int init;
    if (!init) {
        admin_cfg_t cfg = {.port = 0, .proxy_port = 8080, .ca_cert_path = "/nonexistent"};
        admin_configure(&cfg);
        init = 1;
    }
    if (size > HEAD_MAX + BODY_MAX)
        return 0;
    aconn_t c;
    memset(&c, 0, sizeof c);
    c.req = malloc(size + 1);
    if (!c.req)
        return 0;
    memcpy(c.req, data, size);
    c.req[size] = 0;
    c.len = c.cap = size;
    snprintf(c.local_ip, sizeof c.local_ip, "192.168.0.7");
    (void)try_request(&c);
    buf_free(&c.out);
    free(c.req);
    return 0;
}
