#ifndef FPGATE_PKG_H
#define FPGATE_PKG_H
/*
 * Model package loader. The package (weights_i8.bin, epilogue_i32.bin, epilogue_f64.bin,
 * tables.bin, hps_f32.bin) is mapped read-only; the layout comes from the flat index that
 * tools/gen_pkg_index.py writes from manifest.json, so no JSON parser is needed here.
 */
#include <stddef.h>
#include <stdint.h>

#define PKG_MAX_LAYERS 128
#define PKG_MAX_BLOCKS 16
#define PKG_MAX_VECTORS 32

enum { PKG_F_W8, PKG_F_EPI32, PKG_F_EPI64, PKG_F_TAB, PKG_F_HPS, PKG_F_COUNT };

typedef struct {
    char name[24];
    int is_int;                 /* 1: INT epilogue (M0, shift); 0: float output (deq) */
    int out, in, in_bits, out_bits;
    double s_x;
    const int8_t *w;            /* [out][in] */
    const int32_t *bias;        /* [out] or NULL */
    const int32_t *m0, *shift;  /* [out], int layers */
    const double *deq;          /* [out], float layers */
} layer_t;

typedef struct {
    int heads, neurons;
    double s_ctx, s_h, s_proj, s_fc2;
    int64_t sm_m0, sm_sh, av_m0, av_sh, sw_m0, sw_sh;
    const int16_t *silu;        /* 255 entries, index = g + 127 */
    const float *n1_g, *n1_b, *n2_g, *n2_b, *mlpn_g, *mlpn_b;
    const layer_t *q, *k, *v, *proj, *fc1g, *fc1x, *fc2;
} block_t;

typedef struct {
    char path[256];
    char label[16];
    uint32_t logits_hex[4];
    char patch_sha[65];         /* SHA-256 of the float32 patch-embedding output (exact stage) */
} vector_ref_t;

typedef struct {
    char dir[256];
    int size, patch, tokens, channels;
    int rope_frac, rope_pre;
    int softmax_bits, silu_in_bits;
    double ln_eps;
    const uint8_t *file[PKG_F_COUNT];
    size_t file_len[PKG_F_COUNT];
    char file_sha[PKG_F_COUNT][65];
    const uint16_t *exp_tab, *recip_tab;
    const int16_t *rope_cos, *rope_sin;           /* [1024][64] Q1.14 */
    const float *cls, *pos, *fcn_g, *fcn_b;
    int n_layers;
    layer_t layers[PKG_MAX_LAYERS];
    int n_blocks;
    block_t blocks[PKG_MAX_BLOCKS];
    int n_vectors;
    vector_ref_t vectors[PKG_MAX_VECTORS];
    const layer_t *patch_embed, *head;
} pkg_t;

/* returns 0 on success; error text goes to stderr */
int pkg_load(pkg_t *p, const char *pkg_dir, const char *index_path);
/* SHA-256 of every package file against the index (reads ~60 MB) */
int pkg_verify(const pkg_t *p);
void pkg_unload(pkg_t *p);

#endif
