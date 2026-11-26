/*
 * Checks the C packers (src/fpga_pack.c) against the files written by the Python packers that the
 * board already validated:
 *   weights: package layer -> bytes must equal accel/vectors/full_<layer>/mem_w.bin
 *   activations: x.hex of a real slice -> bytes must equal accel/vectors/<slice>/mem_a.bin
 *
 *   test_pack PKG_DIR INDEX VECTORS_DIR
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/fpga_pack.h"
#include "../src/pkg.h"

static int failures;

static uint8_t *read_all(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n)
        exit(2);
    fclose(f);
    *len = (size_t)n;
    return b;
}

static void check(const char *what, const uint8_t *got, const uint8_t *ref, size_t n)
{
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            if (!bad) first = i;
            bad++;
        }
    printf("%s %-46s %zu bytes, %zu differ%s\n", bad ? "FAIL" : "PASS", what, n, bad,
           bad ? " (first at byte)" : "");
    if (bad) {
        printf("     first difference at byte %zu\n", first);
        failures++;
    }
}

static const layer_t *layer(const pkg_t *p, const char *name)
{
    for (int i = 0; i < p->n_layers; i++)
        if (!strcmp(p->layers[i].name, name))
            return &p->layers[i];
    fprintf(stderr, "no layer %s\n", name);
    exit(2);
}

static void test_weights(const pkg_t *p, const char *vdir, const char *lname, const char *dirname)
{
    char path[600];
    size_t ref_len;
    snprintf(path, sizeof path, "%s/%s/mem_w.bin", vdir, dirname);
    uint8_t *ref = read_all(path, &ref_len);
    const layer_t *L = layer(p, lname);
    size_t n = fpga_weights_bytes(L);
    uint8_t *got = malloc(n);
    fpga_pack_weights(L, got);
    char what[80];
    snprintf(what, sizeof what, "weights %s (%d x %d)", lname, L->out, L->in);
    if (n != ref_len) {
        printf("FAIL %s: size %zu, reference %zu\n", what, n, ref_len);
        failures++;
    } else {
        check(what, got, ref, n);
    }
    free(ref);
    free(got);
}

/* x.hex: one value per line, 2 or 4 hex digits (two's complement) */
static void test_acts(const char *vdir, const char *slice)
{
    char path[600];
    int R, C, K, ib, ob;
    snprintf(path, sizeof path, "%s/%s/params.txt", vdir, slice);
    FILE *f = fopen(path, "r");
    if (!f || fscanf(f, "%d %d %d %d %d", &R, &C, &K, &ib, &ob) != 5) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    snprintf(path, sizeof path, "%s/%s/x.hex", vdir, slice);
    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    int16_t *x16 = malloc((size_t)R * K * sizeof *x16);
    int8_t *x8 = malloc((size_t)R * K);
    char line[64];
    for (int i = 0; i < R * K; i++) {
        if (!fgets(line, sizeof line, f))
            exit(2);
        unsigned v = (unsigned)strtoul(line, NULL, 16);
        if (ib > 8) {
            x16[i] = (int16_t)(v & 0xFFFF);
        } else {
            x8[i] = (int8_t)(v & 0xFF);
        }
    }
    fclose(f);
    size_t ref_len;
    snprintf(path, sizeof path, "%s/%s/mem_a.bin", vdir, slice);
    uint8_t *ref = read_all(path, &ref_len);
    const int planes = ib > 8 ? 2 : 1;
    size_t n = fpga_acts_bytes(R, K, planes);
    uint8_t *got = malloc(n);
    if (ib > 8)
        fpga_pack_acts16(x16, R, K, got);
    else
        fpga_pack_acts8(x8, R, K, got);
    char what[80];
    snprintf(what, sizeof what, "activations %s (%d x %d, %d bit)", slice, R, K, ib);
    if (n != ref_len) {
        printf("FAIL %s: size %zu, reference %zu\n", what, n, ref_len);
        failures++;
    } else {
        check(what, got, ref, n);
    }
    free(ref); free(got); free(x16); free(x8);
}

/* the testbench images (tools/pack_attn_full.py): hex, one 256-bit beat per line, byte 31 first */
static uint8_t *read_hex_beats(const char *path, size_t *len)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    size_t cap = 1 << 16, n = 0;
    uint8_t *b = malloc(cap);
    char line[200];
    while (fgets(line, sizeof line, f)) {
        if (strlen(line) < 64)
            continue;
        if (n + 32 > cap) {
            cap *= 2;
            b = realloc(b, cap);
        }
        for (int i = 0; i < 32; i++) {
            char h[3] = {line[2 * (31 - i)], line[2 * (31 - i) + 1], 0};
            b[n + i] = (uint8_t)strtoul(h, NULL, 16);
        }
        n += 32;
    }
    fclose(f);
    *len = n;
    return b;
}

static void test_attention(const char *vdir, const char *head)
{
    char path[700];
    const int T = 1025, HD = 64, KP = 1026;
    int64_t dummy;
    (void)dummy;
    snprintf(path, sizeof path, "%s/%s/q8.bin", vdir, head);
    size_t n;
    int8_t *q = (int8_t *)read_all(path, &n);
    snprintf(path, sizeof path, "%s/%s/k8.bin", vdir, head);
    int8_t *k = (int8_t *)read_all(path, &n);
    snprintf(path, sizeof path, "%s/%s/v8.bin", vdir, head);
    int8_t *v = (int8_t *)read_all(path, &n);
    snprintf(path, sizeof path, "%s/%s/params.txt", vdir, head);
    FILE *f = fopen(path, "r");
    long long a, b, av_m0, av_sh;
    int t, h;
    if (!f || fscanf(f, "%d %d %lld %lld %lld %lld", &t, &h, &a, &b, &av_m0, &av_sh) != 6)
        exit(2);
    fclose(f);

    size_t rl;
    snprintf(path, sizeof path, "%s/%s/full_a.hex", vdir, head);
    uint8_t *ref = read_hex_beats(path, &rl);
    uint8_t *got = malloc(fpga_acts_bytes(32, HD, 1));
    fpga_pack_acts8_strided(q, 32, HD, HD, got);
    char what[100];
    snprintf(what, sizeof what, "attention Q %s (32 rows)", head);
    if (rl != fpga_acts_bytes(32, HD, 1)) { printf("FAIL %s: size\n", what); failures++; } else check(what, got, ref, rl);
    free(ref); free(got);

    snprintf(path, sizeof path, "%s/%s/full_wk.hex", vdir, head);
    ref = read_hex_beats(path, &rl);
    got = malloc(fpga_kt_bytes(T, HD));
    fpga_pack_kt(k, T, HD, HD, got);
    snprintf(what, sizeof what, "attention K^T %s (65 tiles)", head);
    if (rl != fpga_kt_bytes(T, HD)) { printf("FAIL %s: size %zu vs %zu\n", what, rl, fpga_kt_bytes(T, HD)); failures++; } else check(what, got, ref, rl);
    free(ref); free(got);

    snprintf(path, sizeof path, "%s/%s/full_wv.hex", vdir, head);
    ref = read_hex_beats(path, &rl);
    got = malloc(fpga_v_bytes(HD, KP));
    fpga_pack_v(v, T, HD, HD, KP, (int32_t)av_m0, (int)av_sh, got);
    snprintf(what, sizeof what, "attention V %s (4 tiles, K = 1026)", head);
    if (rl != fpga_v_bytes(HD, KP)) { printf("FAIL %s: size %zu vs %zu\n", what, rl, fpga_v_bytes(HD, KP)); failures++; } else check(what, got, ref, rl);
    free(ref); free(got); free(q); free(k); free(v);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s PKG_DIR INDEX VECTORS_DIR\n", argv[0]);
        return 2;
    }
    pkg_t *p = malloc(sizeof *p);
    if (pkg_load(p, argv[1], argv[2]) != 0)
        return 2;
    test_weights(p, argv[3], "b0.q_w", "full_b0_q_w");
    test_weights(p, argv[3], "b0.fc2_w", "full_b0_fc2_w");
    test_acts(argv[3], "b0_q_w");
    test_acts(argv[3], "b0_v_w");
    test_acts(argv[3], "b0_fc2_w");
    test_acts(argv[3], "b1_fc2_w");
    test_attention(argv[3], "attn_b0_h0");
    test_attention(argv[3], "attn_b11_h5");
    printf(failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
