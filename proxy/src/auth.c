#define _GNU_SOURCE
#include "auth.h"

#include <mbedtls/md.h>
#include <mbedtls/pkcs5.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ITERATIONS      20000u
#define SALT_LEN        16
#define HASH_LEN        32
#define USER_MAX        32
#define PASSWORD_MAX    128
#define MAX_SESSIONS    8
#define IDLE_SECS       (30 * 60)
#define LIFETIME_SECS   (8 * 60 * 60)
#define AUTH_MAX_FAILS  5
#define LOCK_SECS       30

static struct {
    int ready;
    char user[USER_MAX + 1];
    unsigned iterations;
    uint8_t salt[SALT_LEN], hash[HASH_LEN];
    int fails;
    time_t locked_until;
} A;

static struct {
    char token[AUTH_TOKEN_LEN + 1];
    time_t created, last;
} S[MAX_SESSIONS];

static void to_hex(char *dst, const uint8_t *src, size_t n)
{
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        dst[2 * i] = d[src[i] >> 4];
        dst[2 * i + 1] = d[src[i] & 15];
    }
    dst[2 * n] = 0;
}

static int from_hex(uint8_t *dst, size_t n, const char *s)
{
    if (strlen(s) != 2 * n)
        return -1;
    for (size_t i = 0; i < n; i++) {
        unsigned v = 0;
        for (int k = 0; k < 2; k++) {
            char ch = s[2 * i + k];
            unsigned x = ch >= '0' && ch <= '9' ? (unsigned)(ch - '0')
                         : ch >= 'a' && ch <= 'f' ? (unsigned)(ch - 'a' + 10) : 99u;
            if (x > 15)
                return -1;
            v = v * 16 + x;
        }
        dst[i] = (uint8_t)v;
    }
    return 0;
}

static int random_bytes(void *p, size_t n)
{
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f)
        return -1;
    size_t got = fread(p, 1, n, f);
    fclose(f);
    return got == n ? 0 : -1;
}

static int derive(const char *password, const uint8_t *salt, unsigned iterations, uint8_t *out)
{
    return mbedtls_pkcs5_pbkdf2_hmac_ext(MBEDTLS_MD_SHA256, (const unsigned char *)password,
                                         strlen(password), salt, SALT_LEN, iterations, HASH_LEN,
                                         out);
}

/* Compares without stopping at the first difference. */
static int same(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint8_t d = 0;
    for (size_t i = 0; i < n; i++)
        d |= (uint8_t)(a[i] ^ b[i]);
    return d == 0;
}

int auth_write_credentials(const char *path, const char *user, const char *password, char *err,
                           size_t err_len)
{
    size_t ul = strlen(user);
    if (ul == 0 || ul > USER_MAX || strpbrk(user, ":\r\n")) {
        snprintf(err, err_len, "the user name must be 1 to %d characters and have no ':'", USER_MAX);
        return -1;
    }
    if (strlen(password) < 8 || strlen(password) > PASSWORD_MAX) {
        snprintf(err, err_len, "the password must be 8 to %d characters", PASSWORD_MAX);
        return -1;
    }
    uint8_t salt[SALT_LEN], hash[HASH_LEN];
    if (random_bytes(salt, sizeof salt) != 0 || derive(password, salt, ITERATIONS, hash) != 0) {
        snprintf(err, err_len, "could not derive the password hash");
        return -1;
    }
    char sh[2 * SALT_LEN + 1], hh[2 * HASH_LEN + 1], tmp[512];
    to_hex(sh, salt, sizeof salt);
    to_hex(hh, hash, sizeof hash);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!f) {
        snprintf(err, err_len, "cannot write %s", tmp);
        if (fd >= 0)
            close(fd);
        return -1;
    }
    int ok = fprintf(f, "%s:%u:%s:%s\n", user, ITERATIONS, sh, hh) > 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp, path) != 0) {
        unlink(tmp);
        snprintf(err, err_len, "cannot write %s", path);
        return -1;
    }
    return 0;
}

int auth_load(const char *path, char *err, size_t err_len)
{
    char line[256];
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(err, err_len, "cannot open %s", path);
        return -1;
    }
    struct stat st;
    if (fstat(fileno(f), &st) == 0 && (st.st_mode & 077))
        fprintf(stderr, "warning: %s is readable by other users; chmod 600 it\n", path);
    char *got = fgets(line, sizeof line, f);
    fclose(f);
    if (!got) {
        snprintf(err, err_len, "%s is empty", path);
        return -1;
    }
    line[strcspn(line, "\r\n")] = 0;
    char *user = line, *it = strchr(user, ':');
    char *salt = it ? strchr(it + 1, ':') : NULL, *hash = salt ? strchr(salt + 1, ':') : NULL;
    if (!hash) {
        snprintf(err, err_len, "%s is not a credentials file", path);
        return -1;
    }
    *it++ = *salt++ = *hash++ = 0;
    unsigned long n = strtoul(it, NULL, 10);
    if (strlen(user) == 0 || strlen(user) > USER_MAX || n < 1000 || n > 10000000ul ||
        from_hex(A.salt, SALT_LEN, salt) != 0 || from_hex(A.hash, HASH_LEN, hash) != 0) {
        snprintf(err, err_len, "%s is damaged", path);
        return -1;
    }
    memcpy(A.user, user, strlen(user) + 1);     /* length checked above */
    A.iterations = (unsigned)n;
    A.ready = 1;
    return 0;
}

int auth_ready(void) { return A.ready; }

auth_result_t auth_login(const char *user, const char *password, char token[AUTH_TOKEN_LEN + 1],
                         unsigned *retry_s)
{
    time_t now = time(NULL);
    if (!A.ready)
        return AUTH_BAD_CREDENTIALS;
    if (now < A.locked_until) {
        if (retry_s)
            *retry_s = (unsigned)(A.locked_until - now);
        return AUTH_LOCKED;
    }
    /* The hash is always computed, so a wrong user name costs the same time. */
    uint8_t h[HASH_LEN];
    int ok = strlen(password) <= PASSWORD_MAX && derive(password, A.salt, A.iterations, h) == 0 &&
             same(h, A.hash, HASH_LEN) && strcmp(user, A.user) == 0;
    explicit_bzero(h, sizeof h);
    if (!ok) {
        if (++A.fails >= AUTH_MAX_FAILS) {
            A.fails = 0;
            A.locked_until = now + LOCK_SECS;
        }
        return AUTH_BAD_CREDENTIALS;
    }
    A.fails = 0;

    int slot = 0;
    for (int i = 0; i < MAX_SESSIONS; i++) {
        if (!S[i].token[0]) {
            slot = i;
            break;
        }
        if (S[i].last < S[slot].last)
            slot = i;
    }
    uint8_t raw[AUTH_TOKEN_LEN / 2];
    if (random_bytes(raw, sizeof raw) != 0)
        return AUTH_BAD_CREDENTIALS;
    to_hex(S[slot].token, raw, sizeof raw);
    S[slot].created = S[slot].last = now;
    memcpy(token, S[slot].token, AUTH_TOKEN_LEN + 1);
    return AUTH_OK;
}

int auth_session_valid(const char *token)
{
    if (!token || strlen(token) != AUTH_TOKEN_LEN)
        return 0;
    time_t now = time(NULL);
    for (int i = 0; i < MAX_SESSIONS; i++) {
        if (!S[i].token[0] || !same((const uint8_t *)S[i].token, (const uint8_t *)token, AUTH_TOKEN_LEN))
            continue;
        if (now - S[i].last > IDLE_SECS || now - S[i].created > LIFETIME_SECS) {
            memset(&S[i], 0, sizeof S[i]);
            return 0;
        }
        S[i].last = now;
        return 1;
    }
    return 0;
}

void auth_logout(const char *token)
{
    if (!token || strlen(token) != AUTH_TOKEN_LEN)
        return;
    for (int i = 0; i < MAX_SESSIONS; i++)
        if (S[i].token[0] && same((const uint8_t *)S[i].token, (const uint8_t *)token, AUTH_TOKEN_LEN))
            memset(&S[i], 0, sizeof S[i]);
}
