#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "proxy.h"
#include "tls_mitm.h"

#define FPGATE_PROXY_VERSION "0.2.0"
#define DEFAULT_PORT 8080

static volatile sig_atomic_t g_stop;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [-p port] [-C ca.crt -K ca.key [-P passfile] [-B bundle.pem]] [-v]\n"
            "  -C/-K  CA certificate and (encrypted) private key; enables HTTPS interception\n"
            "  -P     file holding the CA key password (default: $FPGATE_CA_PASS)\n"
            "  -B     trusted CA bundle used to verify the real servers\n",
            prog);
}

/* Reads the first line of a file into buf, stripping the newline. */
static int read_secret(const char *path, char *buf, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    if (!fgets(buf, (int)n, f)) {
        fclose(f);
        return -1;
    }
    fclose(f);
    buf[strcspn(buf, "\r\n")] = '\0';
    return 0;
}

int main(int argc, char **argv)
{
    int port = DEFAULT_PORT;
    const char *ca_cert = NULL, *ca_key = NULL, *pass_file = NULL;
    const char *bundle = "/etc/fpgate/ca-bundle.pem";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-C") == 0 && i + 1 < argc) {
            ca_cert = argv[++i];
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            ca_key = argv[++i];
        } else if (strcmp(argv[i], "-P") == 0 && i + 1 < argc) {
            pass_file = argv[++i];
        } else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) {
            bundle = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            printf("fpgate-proxy %s\n", FPGATE_PROXY_VERSION);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (port < 1 || port > 65535) {
        fprintf(stderr, "invalid port\n");
        return 2;
    }
    if (!ca_cert != !ca_key) {
        fprintf(stderr, "-C and -K must be given together\n");
        return 2;
    }

    if (ca_cert) {
        char pass[256] = "";
        if (pass_file) {
            if (read_secret(pass_file, pass, sizeof pass) != 0) {
                fprintf(stderr, "cannot read password file %s\n", pass_file);
                return 2;
            }
        } else if (getenv("FPGATE_CA_PASS")) {
            snprintf(pass, sizeof pass, "%s", getenv("FPGATE_CA_PASS"));
        }
        int rc = tls_init(ca_cert, ca_key, pass, bundle);
        explicit_bzero(pass, sizeof pass);
        if (rc != 0)
            return 1;
    } else {
        fprintf(stderr, "no CA configured: HTTPS goes through blind tunnels\n");
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    int rc = proxy_run(port, &g_stop) == 0 ? 0 : 1;
    tls_cleanup();
    return rc;
}
