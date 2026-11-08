#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "auth.h"
#include "proxy.h"
#include "inspect.h"
#include "blocklog.h"
#include "settings.h"
#include "substitute.h"
#include "tls_mitm.h"

#define FPGATE_PROXY_VERSION "0.5.0"
#define DEFAULT_PORT 8080

static volatile sig_atomic_t g_stop, g_reload;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

static void on_hup(int sig)
{
    (void)sig;
    g_reload = 1;
}

/* Runs on the event loop after SIGHUP: re-reads the configuration file. */
static void reload_settings(void)
{
    char err[200];
    if (settings_reload(err, sizeof err) == 0)
        fprintf(stderr, "configuration reloaded\n");
    else
        fprintf(stderr, "configuration NOT reloaded, keeping the current one: %s\n", err);
}

/* Reads a line from stdin; with a terminal, nothing is echoed. */
static int read_password(char *buf, size_t n)
{
    struct termios t, quiet;
    int tty = isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &t) == 0;
    if (tty) {
        fprintf(stderr, "Administrator password: ");
        quiet = t;
        quiet.c_lflag &= ~(tcflag_t)ECHO;
        tcsetattr(STDIN_FILENO, TCSANOW, &quiet);
    }
    char *got = fgets(buf, (int)n, stdin);
    if (tty) {
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
        fputc('\n', stderr);
    }
    if (!got)
        return -1;
    buf[strcspn(buf, "\r\n")] = 0;
    return 0;
}

static int parse_policy(const char *s, inspect_policy_t *out)
{
    if (strcmp(s, "block") == 0)
        *out = POLICY_BLOCK;
    else if (strcmp(s, "allow") == 0)
        *out = POLICY_ALLOW;
    else
        return -1;
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [-p port] [-C ca.crt -K ca.key [-P passfile] [-B bundle.pem]]\n"
            "          [-S size] [-Q] [-D] [-L max_bytes] [-W workers] [-V] [-N] [-v]\n"
            "  -C/-K  CA certificate and (encrypted) private key; enables HTTPS interception\n"
            "  -P     file holding the CA key password (default: $FPGATE_CA_PASS)\n"
            "  -B     trusted CA bundle used to verify the real servers\n"
            "  -S     side of the square image sent to the accelerator (default 448)\n"
            "  -Q     squash to size x size instead of resize shorter side + centre crop\n"
            "  -D     let the JPEG decoder scale by 1/2, 1/4, 1/8 (faster, not bit-exact with PIL)\n"
            "  -L     largest image analysed, in bytes (default 8388608)\n"
            "  -W     image decode worker threads (default 1, 0 = decode inside the event loop)\n"
            "  -V     verbose: path and Referer on every image line, plus a line per response\n"
            "  -N     do not inspect HTTP at all: pure forwarding\n"
            "\n"
            "Judging images (needs fpgate-accel; without --accel images are only analysed):\n"
            "  --accel PATH          Unix socket of fpgate-accel; images it rates NSFW are replaced\n"
            "  --threshold X         score >= X is replaced (default 0.50)\n"
            "  --accel-timeout-ms N  limit per image once sent to the accelerator (default 10000)\n"
            "  --queue-max N         images allowed to wait for the accelerator (default 8)\n"
            "  --on-failure P        accelerator down, slow or erroring: block (default) or allow\n"
            "  --on-unexamined P     GIF, AVIF, too many pixels...: allow (default) or block\n"
            "  --on-corrupt P        damaged image: block (default) or allow\n"
            "  --substitute FILE     JPEG/PNG/WebP shown instead of a blocked image (default: grey PNG)\n"
            "  --block-log FILE      SQLite log of blocks: host, score, threshold, reason (no URLs, no pixels)\n"
            "Administration (the HTTPS interface needs -C/-K):\n"
            "  --config FILE         TOML with the threshold and stand-in image; the admin page writes it, SIGHUP re-reads it\n"
            "  --admin-credentials F administrator login file; turns the interface on\n"
            "  --admin-port N        port of the interface (default 8443)\n"
            "  --set-admin-password F  reads a password, writes F and exits (--admin-user NAME, default admin)\n",
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
    inspect_cfg_t icfg = *inspect_config();
    accel_cfg_t acfg;
    memset(&acfg, 0, sizeof acfg);
    acfg.queue_max = 8;
    const char *substitute_path = NULL, *blocklog_path = NULL;
    const char *config_path = NULL, *admin_creds = NULL, *set_password_file = NULL;
    const char *admin_user = "admin";
    int admin_port = 8443;

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
        } else if (strcmp(argv[i], "-S") == 0 && i + 1 < argc) {
            icfg.prep.size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-Q") == 0) {
            icfg.prep.center_crop = 0;
        } else if (strcmp(argv[i], "-D") == 0) {
            icfg.prep.dct_scaling = 1;
        } else if (strcmp(argv[i], "-L") == 0 && i + 1 < argc) {
            icfg.max_bytes = strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-V") == 0) {
            icfg.verbose = 1;
        } else if (strcmp(argv[i], "-N") == 0) {
            icfg.enabled = 0;
        } else if (strcmp(argv[i], "-W") == 0 && i + 1 < argc) {
            icfg.workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            icfg.delay_ms = (unsigned)atoi(argv[++i]);     /* test hook: slow decodes */
        } else if (strcmp(argv[i], "--accel") == 0 && i + 1 < argc) {
            acfg.path = argv[++i];
            icfg.classify = 1;
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            icfg.threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--accel-timeout-ms") == 0 && i + 1 < argc) {
            acfg.timeout_ms = (unsigned)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--queue-max") == 0 && i + 1 < argc) {
            acfg.queue_max = (unsigned)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--on-failure") == 0 && i + 1 < argc) {
            if (parse_policy(argv[++i], &icfg.on_failure) != 0)
                return usage(argv[0]), 2;
        } else if (strcmp(argv[i], "--on-unexamined") == 0 && i + 1 < argc) {
            if (parse_policy(argv[++i], &icfg.on_unexamined) != 0)
                return usage(argv[0]), 2;
        } else if (strcmp(argv[i], "--on-corrupt") == 0 && i + 1 < argc) {
            if (parse_policy(argv[++i], &icfg.on_corrupt) != 0)
                return usage(argv[0]), 2;
        } else if (strcmp(argv[i], "--substitute") == 0 && i + 1 < argc) {
            substitute_path = argv[++i];
        } else if (strcmp(argv[i], "--block-log") == 0 && i + 1 < argc) {
            blocklog_path = argv[++i];
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--admin-credentials") == 0 && i + 1 < argc) {
            admin_creds = argv[++i];
        } else if (strcmp(argv[i], "--admin-port") == 0 && i + 1 < argc) {
            admin_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--admin-user") == 0 && i + 1 < argc) {
            admin_user = argv[++i];
        } else if (strcmp(argv[i], "--set-admin-password") == 0 && i + 1 < argc) {
            set_password_file = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            printf("fpgate-proxy %s\n", FPGATE_PROXY_VERSION);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (set_password_file) {
        char pw[200], err[200] = "no password given";
        if (read_password(pw, sizeof pw) != 0 ||
            auth_write_credentials(set_password_file, admin_user, pw, err, sizeof err) != 0) {
            explicit_bzero(pw, sizeof pw);
            fprintf(stderr, "%s\n", err);
            return 1;
        }
        explicit_bzero(pw, sizeof pw);
        fprintf(stderr, "credentials for '%s' written to %s\n", admin_user, set_password_file);
        return 0;
    }
    if (port < 1 || port > 65535 || admin_port < 1 || admin_port > 65535) {
        fprintf(stderr, "invalid port\n");
        return 2;
    }
    if (icfg.prep.size < 16 || icfg.prep.size > 1024 || icfg.max_bytes < 1024 ||
        icfg.workers < 0 || icfg.workers > 4) {
        fprintf(stderr, "invalid image settings\n");
        return 2;
    }
    if (icfg.threshold < 0.0f || icfg.threshold > 1.0f) {
        fprintf(stderr, "--threshold must be between 0 and 1\n");
        return 2;
    }
    if (icfg.classify && (icfg.workers < 1 || !icfg.enabled)) {
        fprintf(stderr, "--accel needs inspection and at least one decode worker (-W 1)\n");
        return 2;
    }
    if (substitute_path) {
        char err[160];
        if (substitute_load(substitute_path, err, sizeof err) != 0) {
            fprintf(stderr, "--substitute: %s\n", err);
            return 2;
        }
    }
    if (blocklog_path && blocklog_open(blocklog_path) != 0)
        return 1;
    if (acfg.path)
        proxy_set_accel(&acfg);
    inspect_configure(&icfg);
    if (!ca_cert != !ca_key) {
        fprintf(stderr, "-C and -K must be given together\n");
        return 2;
    }
    {
        char err[200];
        if (settings_init(config_path, err, sizeof err) != 0) {
            fprintf(stderr, "--config: %s\n", err);
            return 2;
        }
        if (admin_creds && auth_load(admin_creds, err, sizeof err) != 0) {
            fprintf(stderr, "--admin-credentials: %s\n", err);
            return 2;
        }
        if (admin_creds && !ca_cert) {
            fprintf(stderr, "--admin-credentials needs the CA (-C/-K): the interface is served over HTTPS\n");
            return 2;
        }
    }
    if (ca_cert) {
        admin_cfg_t acf = {.port = admin_creds ? admin_port : 0, .proxy_port = port,
                           .ca_cert_path = ca_cert};
        proxy_set_admin(&acf);
    }
    proxy_set_reload(&g_reload, reload_settings);

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
    sa.sa_handler = on_hup;
    sigaction(SIGHUP, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    int rc = proxy_run(port, &g_stop) == 0 ? 0 : 1;
    blocklog_close();
    substitute_free();
    tls_cleanup();
    return rc;
}
