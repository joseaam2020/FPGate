#define _GNU_SOURCE
#include "tls_mitm.h"

#include <arpa/inet.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>

#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/error.h>
#include <mbedtls/net_sockets.h>
#include <mbedtls/oid.h>
#include <mbedtls/pk.h>
#include <mbedtls/ssl.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>

#define LEAF_CACHE_SIZE   128
#define LEAF_DER_MAX      1536
#define LEAF_VALID_SECS   (7 * 24 * 3600)
#define LEAF_REUSE_MARGIN (24 * 3600)
#define HOST_MAX          256
#define WRITE_CHUNK       16384
#define MAX_SAN           8

#ifndef BUILD_EPOCH
#define BUILD_EPOCH 0
#endif

/* ---- global state ----------------------------------------------------- */

typedef struct {
    char host[HOST_MAX];
    unsigned char der[LEAF_DER_MAX];
    size_t der_len;
    time_t not_after;
    time_t used;
} leaf_entry_t;

static int g_ready;
static int g_enabled;
static mbedtls_entropy_context g_entropy;
static mbedtls_ctr_drbg_context g_drbg;
static mbedtls_x509_crt g_ca_crt;
static mbedtls_pk_context g_ca_key;
static mbedtls_pk_context g_leaf_key;      /* one key shared by every leaf */
static mbedtls_x509_crt g_bundle;
static mbedtls_ssl_config g_client_conf;
static leaf_entry_t g_cache[LEAF_CACHE_SIZE];

static const char *k_alpn[] = { "http/1.1", NULL };   /* HTTP/2 is out of scope */

struct tls_session {
    mbedtls_ssl_context ssl;
    mbedtls_ssl_config *own_conf;          /* server role only */
    mbedtls_x509_crt own_crt;              /* server role only */
    int *fd;
    size_t retry_len;
    int close_sent;
    char err[160];
};

static void set_err(tls_session_t *s, const char *what, int code)
{
    char tmp[96];
    mbedtls_strerror(code, tmp, sizeof tmp);
    snprintf(s->err, sizeof s->err, "%s: %s (-0x%04x)", what, tmp, (unsigned)-code);
}

/* ---- BIO over a non-blocking fd --------------------------------------- */

static int bio_send(void *ctx, const unsigned char *buf, size_t len)
{
    ssize_t n = send(*(int *)ctx, buf, len, MSG_NOSIGNAL);
    if (n >= 0)
        return (int)n;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        return MBEDTLS_ERR_SSL_WANT_WRITE;
    if (errno == EPIPE || errno == ECONNRESET)
        return MBEDTLS_ERR_NET_CONN_RESET;
    return MBEDTLS_ERR_NET_SEND_FAILED;
}

static int bio_recv(void *ctx, unsigned char *buf, size_t len)
{
    ssize_t n = recv(*(int *)ctx, buf, len, 0);
    if (n >= 0)
        return (int)n;
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        return MBEDTLS_ERR_SSL_WANT_READ;
    if (errno == ECONNRESET)
        return MBEDTLS_ERR_NET_CONN_RESET;
    return MBEDTLS_ERR_NET_RECV_FAILED;
}

/* ---- CA and leaf certificates ----------------------------------------- */

static int ca_is_valid_now(void)
{
    return !mbedtls_x509_time_is_past(&g_ca_crt.valid_to) &&
           !mbedtls_x509_time_is_future(&g_ca_crt.valid_from);
}

static void fmt_time(time_t t, char *out, size_t n)
{
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(out, n, "%Y%m%d%H%M%S", &tm);
}

static int make_leaf(const char *host, unsigned char *der_out, size_t *der_len,
                     time_t *not_after)
{
    mbedtls_x509write_cert crt;
    mbedtls_x509_san_list san[MAX_SAN];
    mbedtls_asn1_sequence eku;
    unsigned char serial[16], ip[MAX_SAN][16], buf[LEAF_DER_MAX + 512];
    char subject[HOST_MAX + 16], issuer[256], nb[16], na[16], names[HOST_MAX], *name[MAX_SAN];
    int nnames = 0;
    time_t now = time(NULL);
    int ret;

    mbedtls_x509write_crt_init(&crt);
    memset(san, 0, sizeof san);
    memset(&eku, 0, sizeof eku);

    if (mbedtls_ctr_drbg_random(&g_drbg, serial, sizeof serial) != 0)
        goto fail;
    serial[0] = (unsigned char)((serial[0] & 0x7F) | 0x01);

    /* `host` is one name, or several separated by commas: all of them go in the SAN. */
    snprintf(names, sizeof names, "%s", host);
    for (char *save = NULL, *t = strtok_r(names, ",", &save); t && nnames < MAX_SAN;
         t = strtok_r(NULL, ",", &save))
        name[nnames++] = t;
    if (nnames == 0)
        goto fail;
    snprintf(subject, sizeof subject, "CN=%s", name[0]);
    if (mbedtls_x509_dn_gets(issuer, sizeof issuer, &g_ca_crt.subject) < 0)
        goto fail;

    fmt_time(now - 24 * 3600, nb, sizeof nb);
    fmt_time(now + LEAF_VALID_SECS, na, sizeof na);

    mbedtls_x509write_crt_set_version(&crt, MBEDTLS_X509_CRT_VERSION_3);
    mbedtls_x509write_crt_set_md_alg(&crt, MBEDTLS_MD_SHA256);
    mbedtls_x509write_crt_set_subject_key(&crt, &g_leaf_key);
    mbedtls_x509write_crt_set_issuer_key(&crt, &g_ca_key);
    if (mbedtls_x509write_crt_set_subject_name(&crt, subject) != 0 ||
        mbedtls_x509write_crt_set_issuer_name(&crt, issuer) != 0 ||
        mbedtls_x509write_crt_set_serial_raw(&crt, serial, sizeof serial) != 0 ||
        mbedtls_x509write_crt_set_validity(&crt, nb, na) != 0 ||
        mbedtls_x509write_crt_set_basic_constraints(&crt, 0, -1) != 0 ||
        mbedtls_x509write_crt_set_key_usage(&crt, MBEDTLS_X509_KU_DIGITAL_SIGNATURE) != 0 ||
        mbedtls_x509write_crt_set_subject_key_identifier(&crt) != 0 ||
        mbedtls_x509write_crt_set_authority_key_identifier(&crt) != 0)
        goto fail;

    eku.buf.tag = MBEDTLS_ASN1_OID;
    eku.buf.p = (unsigned char *)MBEDTLS_OID_SERVER_AUTH;
    eku.buf.len = MBEDTLS_OID_SIZE(MBEDTLS_OID_SERVER_AUTH);
    if (mbedtls_x509write_crt_set_ext_key_usage(&crt, &eku) != 0)
        goto fail;

    /* The names go in the SAN: browsers ignore the CN. */
    for (int k = 0; k < nnames; k++) {
        san[k].next = k + 1 < nnames ? &san[k + 1] : NULL;
        if (inet_pton(AF_INET, name[k], ip[k]) == 1) {
            san[k].node.type = MBEDTLS_X509_SAN_IP_ADDRESS;
            san[k].node.san.unstructured_name.p = ip[k];
            san[k].node.san.unstructured_name.len = 4;
        } else if (inet_pton(AF_INET6, name[k], ip[k]) == 1) {
            san[k].node.type = MBEDTLS_X509_SAN_IP_ADDRESS;
            san[k].node.san.unstructured_name.p = ip[k];
            san[k].node.san.unstructured_name.len = 16;
        } else {
            san[k].node.type = MBEDTLS_X509_SAN_DNS_NAME;
            san[k].node.san.unstructured_name.tag = MBEDTLS_ASN1_IA5_STRING;
            san[k].node.san.unstructured_name.p = (unsigned char *)name[k];
            san[k].node.san.unstructured_name.len = strlen(name[k]);
        }
    }
    if (mbedtls_x509write_crt_set_subject_alternative_name(&crt, san) != 0)
        goto fail;

    ret = mbedtls_x509write_crt_der(&crt, buf, sizeof buf, mbedtls_ctr_drbg_random, &g_drbg);
    if (ret <= 0 || ret > LEAF_DER_MAX)
        goto fail;
    memcpy(der_out, buf + sizeof buf - (size_t)ret, (size_t)ret);
    *der_len = (size_t)ret;
    *not_after = now + LEAF_VALID_SECS;
    mbedtls_x509write_crt_free(&crt);
    return 0;

fail:
    mbedtls_x509write_crt_free(&crt);
    return -1;
}

/* Returns the cached DER for host, generating (and caching) it when needed. */
static leaf_entry_t *leaf_get(const char *host)
{
    time_t now = time(NULL);
    leaf_entry_t *victim = &g_cache[0];

    if (strlen(host) >= HOST_MAX)
        return NULL;
    for (int i = 0; i < LEAF_CACHE_SIZE; i++) {
        leaf_entry_t *e = &g_cache[i];
        if (e->der_len && strcmp(e->host, host) == 0 &&
            e->not_after - now > LEAF_REUSE_MARGIN) {
            e->used = now;
            return e;
        }
        if (!e->der_len || e->used < victim->used)
            victim = e;
        if (!e->der_len)
            break;
    }
    if (make_leaf(host, victim->der, &victim->der_len, &victim->not_after) != 0) {
        victim->der_len = 0;
        return NULL;
    }
    snprintf(victim->host, sizeof victim->host, "%s", host);
    victim->used = now;
    return victim;
}

/* ---- init / cleanup --------------------------------------------------- */

int tls_init(const char *ca_cert, const char *ca_key, const char *ca_pass,
             const char *trust_bundle)
{
    struct stat st;
    int ret;
    char msg[96];

    if ((time_t)BUILD_EPOCH > time(NULL))
        fprintf(stderr, "warning: system clock is earlier than this build; "
                        "certificate validation will fail until the date is set\n");

    if (psa_crypto_init() != PSA_SUCCESS) {
        fprintf(stderr, "tls: psa_crypto_init failed\n");
        return -1;
    }
    mbedtls_entropy_init(&g_entropy);
    mbedtls_ctr_drbg_init(&g_drbg);
    mbedtls_x509_crt_init(&g_ca_crt);
    mbedtls_pk_init(&g_ca_key);
    mbedtls_pk_init(&g_leaf_key);
    mbedtls_x509_crt_init(&g_bundle);
    mbedtls_ssl_config_init(&g_client_conf);
    g_ready = 1;

    if (mbedtls_ctr_drbg_seed(&g_drbg, mbedtls_entropy_func, &g_entropy,
                              (const unsigned char *)"fpgate", 6) != 0) {
        fprintf(stderr, "tls: RNG seeding failed\n");
        return -1;
    }

    if (stat(ca_key, &st) == 0 && (st.st_mode & 077))
        fprintf(stderr, "warning: %s is accessible by group/others (expected mode 600)\n", ca_key);

    if ((ret = mbedtls_x509_crt_parse_file(&g_ca_crt, ca_cert)) != 0) {
        mbedtls_strerror(ret, msg, sizeof msg);
        fprintf(stderr, "tls: cannot load CA certificate %s: %s\n", ca_cert, msg);
        return -1;
    }
    if ((ret = mbedtls_pk_parse_keyfile(&g_ca_key, ca_key, ca_pass,
                                        mbedtls_ctr_drbg_random, &g_drbg)) != 0) {
        mbedtls_strerror(ret, msg, sizeof msg);
        fprintf(stderr, "tls: cannot load CA key %s (wrong password?): %s\n", ca_key, msg);
        return -1;
    }
    if (mbedtls_pk_check_pair(&g_ca_crt.pk, &g_ca_key, mbedtls_ctr_drbg_random, &g_drbg) != 0) {
        fprintf(stderr, "tls: CA key does not match the CA certificate\n");
        return -1;
    }

    if ((ret = mbedtls_x509_crt_parse_file(&g_bundle, trust_bundle)) < 0) {
        mbedtls_strerror(ret, msg, sizeof msg);
        fprintf(stderr, "tls: cannot load trust bundle %s: %s\n", trust_bundle, msg);
        return -1;
    }

    if (mbedtls_pk_setup(&g_leaf_key, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)) != 0 ||
        mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(g_leaf_key),
                            mbedtls_ctr_drbg_random, &g_drbg) != 0) {
        fprintf(stderr, "tls: leaf key generation failed\n");
        return -1;
    }

    if (mbedtls_ssl_config_defaults(&g_client_conf, MBEDTLS_SSL_IS_CLIENT,
                                    MBEDTLS_SSL_TRANSPORT_STREAM,
                                    MBEDTLS_SSL_PRESET_DEFAULT) != 0)
        return -1;
    mbedtls_ssl_conf_min_tls_version(&g_client_conf, MBEDTLS_SSL_VERSION_TLS1_2);
    mbedtls_ssl_conf_authmode(&g_client_conf, MBEDTLS_SSL_VERIFY_REQUIRED);
    mbedtls_ssl_conf_ca_chain(&g_client_conf, &g_bundle, NULL);
    mbedtls_ssl_conf_rng(&g_client_conf, mbedtls_ctr_drbg_random, &g_drbg);
    mbedtls_ssl_conf_alpn_protocols(&g_client_conf, k_alpn);

    tls_ca_recheck();
    return 0;
}

int tls_ca_recheck(void)
{
    int ok = g_ready && ca_is_valid_now();
    if (ok != g_enabled)
        fprintf(stderr, ok ? "tls: CA valid, HTTPS interception enabled\n"
                           : "tls: CA expired or not yet valid, HTTPS interception disabled "
                             "(falling back to blind tunnels)\n");
    g_enabled = ok;
    return ok;
}

int tls_enabled(void) { return g_enabled; }

void tls_cleanup(void)
{
    if (!g_ready)
        return;
    mbedtls_ssl_config_free(&g_client_conf);
    mbedtls_x509_crt_free(&g_bundle);
    mbedtls_pk_free(&g_leaf_key);
    mbedtls_pk_free(&g_ca_key);
    mbedtls_x509_crt_free(&g_ca_crt);
    mbedtls_ctr_drbg_free(&g_drbg);
    mbedtls_entropy_free(&g_entropy);
    memset(g_cache, 0, sizeof g_cache);
    g_ready = g_enabled = 0;
}

/* ---- sessions --------------------------------------------------------- */

static tls_session_t *session_new(int *fd)
{
    tls_session_t *s = calloc(1, sizeof *s);
    if (!s)
        return NULL;
    s->fd = fd;
    mbedtls_ssl_init(&s->ssl);
    mbedtls_x509_crt_init(&s->own_crt);
    return s;
}

tls_session_t *tls_server_new(int *fd, const char *host)
{
    if (!g_enabled)
        return NULL;
    leaf_entry_t *e = leaf_get(host);
    tls_session_t *s = e ? session_new(fd) : NULL;
    if (!s)
        return NULL;

    s->own_conf = calloc(1, sizeof *s->own_conf);
    if (!s->own_conf)
        goto fail;
    mbedtls_ssl_config_init(s->own_conf);
    if (mbedtls_ssl_config_defaults(s->own_conf, MBEDTLS_SSL_IS_SERVER,
                                    MBEDTLS_SSL_TRANSPORT_STREAM,
                                    MBEDTLS_SSL_PRESET_DEFAULT) != 0 ||
        mbedtls_x509_crt_parse_der(&s->own_crt, e->der, e->der_len) != 0 ||
        mbedtls_ssl_conf_own_cert(s->own_conf, &s->own_crt, &g_leaf_key) != 0 ||
        mbedtls_ssl_conf_alpn_protocols(s->own_conf, k_alpn) != 0)
        goto fail;
    mbedtls_ssl_conf_min_tls_version(s->own_conf, MBEDTLS_SSL_VERSION_TLS1_2);
    mbedtls_ssl_conf_rng(s->own_conf, mbedtls_ctr_drbg_random, &g_drbg);

    if (mbedtls_ssl_setup(&s->ssl, s->own_conf) != 0)
        goto fail;
    mbedtls_ssl_set_bio(&s->ssl, s->fd, bio_send, bio_recv, NULL);
    return s;

fail:
    tls_free(s);
    return NULL;
}

tls_session_t *tls_client_new(int *fd, const char *host)
{
    if (!g_ready)
        return NULL;
    tls_session_t *s = session_new(fd);
    if (!s)
        return NULL;
    if (mbedtls_ssl_setup(&s->ssl, &g_client_conf) != 0 ||
        mbedtls_ssl_set_hostname(&s->ssl, host) != 0) {
        tls_free(s);
        return NULL;
    }
    mbedtls_ssl_set_bio(&s->ssl, s->fd, bio_send, bio_recv, NULL);
    return s;
}

void tls_free(tls_session_t *s)
{
    if (!s)
        return;
    mbedtls_ssl_free(&s->ssl);
    if (s->own_conf) {
        mbedtls_ssl_config_free(s->own_conf);
        free(s->own_conf);
    }
    mbedtls_x509_crt_free(&s->own_crt);
    free(s);
}

int tls_handshake(tls_session_t *s)
{
    int ret = mbedtls_ssl_handshake(&s->ssl);
    if (ret == 0)
        return 0;
    if (ret == MBEDTLS_ERR_SSL_WANT_READ)
        return TLS_WANT_READ;
    if (ret == MBEDTLS_ERR_SSL_WANT_WRITE)
        return TLS_WANT_WRITE;
    set_err(s, "handshake", ret);
    if (ret == MBEDTLS_ERR_X509_CERT_VERIFY_FAILED) {
        char info[96];
        uint32_t flags = mbedtls_ssl_get_verify_result(&s->ssl);
        mbedtls_x509_crt_verify_info(info, sizeof info, "", flags);
        size_t n = strlen(s->err);
        snprintf(s->err + n, sizeof s->err - n, " %s", info);
        n = strlen(s->err);
        while (n && (s->err[n - 1] == '\n' || s->err[n - 1] == ' '))
            s->err[--n] = '\0';
    }
    return TLS_ERR;
}

ssize_t tls_read(tls_session_t *s, void *buf, size_t len)
{
    for (;;) {
        int ret = mbedtls_ssl_read(&s->ssl, buf, len);
        if (ret > 0)
            return ret;
        if (ret == 0 || ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY)
            return 0;
        if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE)
            return TLS_WANT_READ;
        if (ret == MBEDTLS_ERR_SSL_RECEIVED_NEW_SESSION_TICKET)
            continue;
        set_err(s, "read", ret);
        return TLS_ERR;
    }
}

ssize_t tls_write(tls_session_t *s, const void *buf, size_t len)
{
    /* mbedTLS wants the retry after WANT_WRITE to use the same length. */
    if (s->retry_len)
        len = s->retry_len;
    else if (len > WRITE_CHUNK)
        len = WRITE_CHUNK;

    int ret = mbedtls_ssl_write(&s->ssl, buf, len);
    if (ret > 0) {
        s->retry_len = 0;
        return ret;
    }
    if (ret == MBEDTLS_ERR_SSL_WANT_WRITE || ret == MBEDTLS_ERR_SSL_WANT_READ) {
        s->retry_len = len;
        return TLS_WANT_WRITE;
    }
    set_err(s, "write", ret);
    return TLS_ERR;
}

void tls_close_notify(tls_session_t *s)
{
    if (s->close_sent)
        return;
    s->close_sent = 1;
    (void)mbedtls_ssl_close_notify(&s->ssl);
}

const char *tls_error(const tls_session_t *s) { return s->err[0] ? s->err : "unknown"; }
