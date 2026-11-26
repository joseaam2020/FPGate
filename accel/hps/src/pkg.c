#include "pkg.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sha256.h"

static const char *const FILE_NAMES[PKG_F_COUNT] = {"weights_i8.bin", "epilogue_i32.bin", "epilogue_f64.bin",
                                                    "tables.bin", "hps_f32.bin"};

static int map_file(pkg_t *p, int idx)
{
    char path[600];
    snprintf(path, sizeof path, "%s/%s", p->dir, FILE_NAMES[idx]);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "pkg: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        fprintf(stderr, "pkg: cannot map %s: %s\n", path, strerror(errno));
        return -1;
    }
    p->file[idx] = m;
    p->file_len[idx] = (size_t)st.st_size;
    return 0;
}

static int file_index(const char *name)
{
    for (int i = 0; i < PKG_F_COUNT; i++)
        if (!strcmp(name, FILE_NAMES[i]))
            return i;
    return -1;
}

static const layer_t *find_layer(const pkg_t *p, const char *name)
{
    for (int i = 0; i < p->n_layers; i++)
        if (!strcmp(p->layers[i].name, name))
            return &p->layers[i];
    return NULL;
}

#define AT(idx, off) (p->file[idx] + (off))

static int bad(FILE *f, const char *what, const char *line)
{
    fprintf(stderr, "pkg: %s: %s", what, line);
    fclose(f);
    return -1;
}

int pkg_load(pkg_t *p, const char *pkg_dir, const char *index_path)
{
    memset(p, 0, sizeof *p);
    snprintf(p->dir, sizeof p->dir, "%s", pkg_dir);
    for (int i = 0; i < PKG_F_COUNT; i++)
        if (map_file(p, i) != 0)
            return -1;

    FILE *f = fopen(index_path, "r");
    if (!f) {
        fprintf(stderr, "pkg: cannot open index %s: %s\n", index_path, strerror(errno));
        return -1;
    }
    char line[1024];
    long t_exp = 0, t_rcp = 0, t_cos = 0, t_sin = 0, h_cls = 0, h_pos = 0, h_fg = 0, h_fb = 0;
    while (fgets(line, sizeof line, f)) {
        char key[32];
        if (sscanf(line, "%31s", key) != 1)
            continue;
        const char *r = line + strlen(key);
        if (!strcmp(key, "version")) {
            int v = 0;
            sscanf(r, "%d", &v);
            if (v != 1)
                return bad(f, "unsupported index version", line);
        } else if (!strcmp(key, "ln_eps")) {
            p->ln_eps = strtod(r, NULL);
        } else if (!strcmp(key, "image")) {
            sscanf(r, "%d %d %d %d", &p->size, &p->patch, &p->tokens, &p->channels);
        } else if (!strcmp(key, "rope")) {
            sscanf(r, "%d %d", &p->rope_frac, &p->rope_pre);
        } else if (!strcmp(key, "softmax_bits")) {
            sscanf(r, "%d", &p->softmax_bits);
        } else if (!strcmp(key, "silu_in_bits")) {
            sscanf(r, "%d", &p->silu_in_bits);
        } else if (!strcmp(key, "file")) {
            char name[64], sha[65];
            if (sscanf(r, "%63s %64s", name, sha) == 2) {
                int i = file_index(name);
                if (i >= 0)
                    snprintf(p->file_sha[i], sizeof p->file_sha[i], "%s", sha);
            }
        } else if (!strcmp(key, "tables")) {
            sscanf(r, "%ld %ld %ld %ld", &t_exp, &t_rcp, &t_cos, &t_sin);
        } else if (!strcmp(key, "hps")) {
            sscanf(r, "%ld %ld %ld %ld", &h_cls, &h_pos, &h_fg, &h_fb);
        } else if (!strcmp(key, "layer")) {
            if (p->n_layers >= PKG_MAX_LAYERS)
                return bad(f, "too many layers", line);
            layer_t *L = &p->layers[p->n_layers];
            char name[24], kind[16], sx[64];
            long w, bias, a, b;
            int out, in, ib, ob;
            if (sscanf(r, "%23s %15s %d %d %d %d %63s %ld %ld %ld %ld", name, kind, &out, &in, &ib, &ob, sx, &w,
                       &bias, &a, &b) != 11)
                return bad(f, "bad layer line", line);
            snprintf(L->name, sizeof L->name, "%s", name);
            L->is_int = !strcmp(kind, "int");
            L->out = out;
            L->in = in;
            L->in_bits = ib;
            L->out_bits = ob;
            L->s_x = strtod(sx, NULL);
            L->w = (const int8_t *)AT(PKG_F_W8, w);
            L->bias = bias >= 0 ? (const int32_t *)AT(PKG_F_EPI32, bias) : NULL;
            if (L->is_int) {
                L->m0 = (const int32_t *)AT(PKG_F_EPI32, a);
                L->shift = (const int32_t *)AT(PKG_F_EPI32, b);
            } else {
                L->deq = (const double *)AT(PKG_F_EPI64, a);
            }
            p->n_layers++;
        } else if (!strcmp(key, "block")) {
            if (p->n_blocks >= PKG_MAX_BLOCKS)
                return bad(f, "too many blocks", line);
            block_t *B = &p->blocks[p->n_blocks];
            int idx, heads, neurons;
            char sctx[64], sh[64], sproj[64], sfc2[64];
            long long smm, sms, avm, avs, swm, sws;
            long silu, n1g, n1b, n2g, n2b, mg, mb;
            if (sscanf(r, "%d %d %d %63s %lld %lld %lld %lld %ld %lld %lld %63s %63s %63s %ld %ld %ld %ld %ld %ld",
                       &idx, &heads, &neurons, sctx, &smm, &sms, &avm, &avs, &silu, &swm, &sws, sh, sproj, sfc2,
                       &n1g, &n1b, &n2g, &n2b, &mg, &mb) != 20)
                return bad(f, "bad block line", line);
            B->heads = heads;
            B->neurons = neurons;
            B->s_ctx = strtod(sctx, NULL);
            B->s_h = strtod(sh, NULL);
            B->s_proj = strtod(sproj, NULL);
            B->s_fc2 = strtod(sfc2, NULL);
            B->sm_m0 = smm; B->sm_sh = sms; B->av_m0 = avm; B->av_sh = avs; B->sw_m0 = swm; B->sw_sh = sws;
            B->silu = (const int16_t *)AT(PKG_F_TAB, silu);
            B->n1_g = (const float *)AT(PKG_F_HPS, n1g);
            B->n1_b = (const float *)AT(PKG_F_HPS, n1b);
            B->n2_g = (const float *)AT(PKG_F_HPS, n2g);
            B->n2_b = (const float *)AT(PKG_F_HPS, n2b);
            B->mlpn_g = (const float *)AT(PKG_F_HPS, mg);
            B->mlpn_b = (const float *)AT(PKG_F_HPS, mb);
            static const char *const sfx[7] = {"q_w", "k_w", "v_w", "proj_w", "fc1g_w", "fc1x_w", "fc2_w"};
            const layer_t **dst[7] = {&B->q, &B->k, &B->v, &B->proj, &B->fc1g, &B->fc1x, &B->fc2};
            for (int i = 0; i < 7; i++) {
                char nm[40];
                snprintf(nm, sizeof nm, "b%d.%s", idx, sfx[i]);
                *dst[i] = find_layer(p, nm);
                if (!*dst[i]) {
                    fprintf(stderr, "pkg: missing layer %s\n", nm);
                    fclose(f);
                    return -1;
                }
            }
            p->n_blocks++;
        } else if (!strcmp(key, "vector")) {
            if (p->n_vectors >= PKG_MAX_VECTORS)
                continue;
            vector_ref_t *V = &p->vectors[p->n_vectors];
            int idx;
            unsigned a, b, c, d;
            if (sscanf(r, "%d %255s %15s %u %u %u %u %64s", &idx, V->path, V->label, &a, &b, &c, &d, V->patch_sha) >= 7) {
                V->logits_hex[0] = a;
                V->logits_hex[1] = b;
                V->logits_hex[2] = c;
                V->logits_hex[3] = d;
                p->n_vectors++;
            }
        }
    }
    fclose(f);

    p->exp_tab = (const uint16_t *)AT(PKG_F_TAB, t_exp);
    p->recip_tab = (const uint16_t *)AT(PKG_F_TAB, t_rcp);
    p->rope_cos = (const int16_t *)AT(PKG_F_TAB, t_cos);
    p->rope_sin = (const int16_t *)AT(PKG_F_TAB, t_sin);
    p->cls = (const float *)AT(PKG_F_HPS, h_cls);
    p->pos = (const float *)AT(PKG_F_HPS, h_pos);
    p->fcn_g = (const float *)AT(PKG_F_HPS, h_fg);
    p->fcn_b = (const float *)AT(PKG_F_HPS, h_fb);
    p->patch_embed = find_layer(p, "patch_embed");
    p->head = find_layer(p, "head");
    if (!p->patch_embed || !p->head || p->n_blocks == 0 || p->size == 0) {
        fprintf(stderr, "pkg: incomplete index\n");
        return -1;
    }
    return 0;
}

int pkg_verify(const pkg_t *p)
{
    for (int i = 0; i < PKG_F_COUNT; i++) {
        sha256_t s;
        char hex[65];
        sha256_init(&s);
        sha256_update(&s, p->file[i], p->file_len[i]);
        sha256_hex(&s, hex);
        if (strcmp(hex, p->file_sha[i]) != 0) {
            fprintf(stderr, "pkg: SHA-256 mismatch in %s\n", FILE_NAMES[i]);
            return -1;
        }
    }
    return 0;
}

void pkg_unload(pkg_t *p)
{
    for (int i = 0; i < PKG_F_COUNT; i++)
        if (p->file[i])
            munmap((void *)p->file[i], p->file_len[i]);
    memset(p, 0, sizeof *p);
}
