#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include "accel_drv.h"
#include "fpga_pack.h"
#include "ops.h"

/* layout of the reserved DDR3 region used by the FPGA backend (offsets from ACCEL_REGION_PHYS) */
#define FPGA_A_OFF 0x000000u        /* activations, tile-major */
#define FPGA_O_OFF 0x800000u        /* outputs, row-major */
#define FPGA_W_OFF 0x1000000u       /* all packed weights */
#define FPGA_REGION_LEN 0x8000000u  /* 128 MiB: 16 MiB of buffers + 56 MiB of weights + attention buffers */
/* attention buffers of one head (offsets from the region base); S and P never leave the DDR3 */
#define ATT_OFF 0x5000000u
#define ATT_Q (ATT_OFF + 0x000000u)         /* 128 KiB slot: Q as activations */
#define ATT_K (ATT_OFF + 0x020000u)         /* 128 KiB slot: K^T tiles */
#define ATT_V (ATT_OFF + 0x040000u)         /* 128 KiB slot: V tiles */
#define ATT_S (ATT_OFF + 0x060000u)         /* 5 MiB slot: INT32 scores, 1040 columns per row */
#define ATT_P (ATT_OFF + 0x560000u)         /* 3 MiB slot: probabilities, activation format */
#define ATT_C (ATT_OFF + 0x860000u)         /* 1 MiB slot: context [tokens][heads * hd] INT8 */
#define ATT_END (ATT_OFF + 0x960000u)
/* overlapped attention: two sets of per-head buffers, set p = head % 2 (Q, K^T, V, S and P as above) */
#define ATT2_OFF 0x6000000u
#define ATT2_SET 0xA00000u
#define ATT2_Q(p) (ATT2_OFF + (p) * ATT2_SET + 0x000000u)
#define ATT2_K(p) (ATT2_OFF + (p) * ATT2_SET + 0x020000u)
#define ATT2_V(p) (ATT2_OFF + (p) * ATT2_SET + 0x040000u)
#define ATT2_S(p) (ATT2_OFF + (p) * ATT2_SET + 0x060000u)
#define ATT2_P(p) (ATT2_OFF + (p) * ATT2_SET + 0x560000u)
#define ATT2_END (ATT2_OFF + 2 * ATT2_SET)

#define PL_MAX 8                    /* chunks (and slots) of the pipeline */
#define PL_AREA 0x800000u           /* size of the activation area and of the output area of the region */

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* run a statement and add its wall time to a counter of model_stats_t */
#define TIMED(field, stmt) do { double _t0 = now_s(); stmt; m->st.field += now_s() - _t0; } while (0)

/* ---- per-image event trace (--trace FILE); with tracing off every hook is one pointer test ---- */
#define TR_MAX 65536
typedef struct { double t0, t1; char kind[8]; char name[24]; int block, idx; uint64_t cycles; } tr_ev_t;
static tr_ev_t *g_tr;                   /* NULL = tracing off */
static const char *g_tr_path;
static int g_tr_n, g_tr_full, g_tr_img, g_tr_block;
static double g_tr_base;

static void tr_add(const char *kind, const char *name, int idx, double t0, double t1, uint64_t cycles)
{
    if (g_tr_n >= TR_MAX) {
        if (!g_tr_full)
            fprintf(stderr, "trace: event buffer full (%d), the rest of the image is not recorded\n", TR_MAX);
        g_tr_full = 1;
        return;
    }
    tr_ev_t *e = &g_tr[g_tr_n++];
    e->t0 = t0 - g_tr_base;
    e->t1 = t1 - g_tr_base;
    snprintf(e->kind, sizeof e->kind, "%s", kind);
    snprintf(e->name, sizeof e->name, "%s", name);
    e->block = g_tr_block;
    e->idx = idx;
    e->cycles = cycles;
}

int model_trace_open(const char *path)
{
    g_tr = (tr_ev_t *)malloc(sizeof(tr_ev_t) * TR_MAX);
    if (!g_tr)
        return -1;
    g_tr_path = path;
    g_tr_img = 0;
    return 0;
}

static void tr_begin(void)
{
    g_tr_n = g_tr_full = g_tr_block = 0;
    g_tr_base = now_s();
}

/* writes FILE for the first image, FILE.1, FILE.2, ... for the next ones */
static void tr_end(void)
{
    char path[1024];
    if (g_tr_img == 0)
        snprintf(path, sizeof path, "%s", g_tr_path);
    else
        snprintf(path, sizeof path, "%s.%d", g_tr_path, g_tr_img);
    g_tr_img++;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "trace: cannot write %s\n", path);
        return;
    }
    fprintf(f, "t0,t1,kind,name,block,idx,cycles\n");
    for (int i = 0; i < g_tr_n; i++)
        fprintf(f, "%.6f,%.6f,%s,%s,%d,%d,%llu\n", g_tr[i].t0, g_tr[i].t1, g_tr[i].kind, g_tr[i].name, g_tr[i].block,
                g_tr[i].idx, (unsigned long long)g_tr[i].cycles);
    fclose(f);
}

struct model {
    const pkg_t *p;
    int tokens, dim, max_neurons;
    float *x;           /* residual stream [tokens][dim] */
    float *n;           /* LayerNorm output [tokens][max(dim, neurons)] */
    float *hh;          /* SwiGLU output [tokens][neurons] */
    float *ctxf;        /* context as float [tokens][512] */
    float *cols;        /* im2col [patches][588] */
    float *patch_out;   /* patch embedding output [patches][dim] */
    int8_t *a8;         /* quantised activations */
    int16_t *a16;
    int32_t *acc;       /* GEMM accumulators [tokens][max_out] */
    int16_t *q16, *k16, *v16, *y16, *g16, *u16, *h16;
    int8_t *q8, *k8, *v8, *ctx8;

    /* FPGA backend (NULL hw = everything on the CPU) */
    accel_hw_t *hw;
    int check, verbose, max_blocks, fpga_attn;
    int att_overlap, att_warned;        /* ATT_OVL_* mode and whether the pack-only warning was printed */
    uint8_t *att_stage;                 /* staging for the Q / K^T / V images of one head */
    size_t att_stage_len;
    int8_t *att_chk;                    /* CPU context used by --check */
    uint32_t w_off[PKG_MAX_LAYERS];     /* byte offset of each layer's packed weights in the region, or 0 */
    uint8_t *stage;                     /* cached staging for packing */
    size_t stage_len;
    uint8_t *rdbuf;                     /* copy of the output matrix read back from the region */
    int16_t *chk16;                     /* CPU result used by --check */
    int a_ok, a_rows, a_k, a_bits;      /* activations currently packed in the region */
    int chunk_rows;                     /* rows per engine operation when overlapping ARM and FPGA; 0 = sequential */
    /* row-chunk pipeline (see model_set_pipeline); buffers are per chunk, sized for chunk_rows rows */
    int cpu_patch;                      /* --cpu-patch: patch embedding on the CPU even with the engine */
    int pipeline, pl_cap, pl_ok;
    int rlp_fuse;                       /* PS_RLP instead of READ, RES, LN, PREP after proj and fc2 */
    int mlp_fuse;                       /* PS_MLP instead of READ(fc1x), SWIGLU, LN(mlpn), PREP(fc2) */
    int want_hw_rope, want_hw_swiglu;   /* model_set_hw_fusion, read by model_attach_fpga */
    int hw_rope;                        /* q and k come out of the engine already rotated (INT8), no PS_ROPE */
    uint32_t sw_off[PKG_MAX_BLOCKS];    /* region offset of the block's interleaved gate / up weights, or 0 */
    struct pl_sched *pl_sched;          /* engine scheduler of the running pipeline, for pl_engine_tick inside a step */
    size_t pl_slot;                     /* bytes of each chunk's slot in the region's activation and output areas */
    float *pl_n[PL_MAX];                /* LayerNorm output / float input of the next layer */
    int8_t *pl_a8[PL_MAX];              /* quantised input (kept for the next layer that reuses it, and --check) */
    int16_t *pl_a16[PL_MAX];
    uint8_t *pl_stage[PL_MAX];          /* packing staging, also the temporary of the 8-bit output read */
    int32_t *pl_acc[PL_MAX];            /* --check only */
    int16_t *pl_chk[PL_MAX];
    float *pl_row[PL_MAX];              /* MLP: per-row SwiGLU and LayerNorm outputs of one 16-row group */
    model_stats_t st;
};

static int attention_fpga(model_t *m, const block_t *B, int T, int A, int H, int hd);
static int attention_fpga_select(model_t *m, const block_t *B, int T, int A, int H, int hd);
static void pl_free(model_t *m);

static void *xalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "model: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

model_t *model_create(const pkg_t *p)
{
    model_t *m = calloc(1, sizeof *m);
    m->p = p;
    m->tokens = p->tokens;
    m->dim = p->patch_embed->out;
    int attn = 0, neu = 0;
    for (int i = 0; i < p->n_blocks; i++) {
        if (p->blocks[i].q->out > attn) attn = p->blocks[i].q->out;
        if (p->blocks[i].neurons > neu) neu = p->blocks[i].neurons;
    }
    m->max_neurons = neu;
    const size_t T = (size_t)m->tokens;
    const size_t widest = (size_t)(neu > m->dim ? neu : m->dim);
    m->x = xalloc(T * m->dim * sizeof(float));
    m->n = xalloc(T * widest * sizeof(float));
    m->hh = xalloc(T * neu * sizeof(float));
    m->ctxf = xalloc(T * attn * sizeof(float));
    m->cols = xalloc((size_t)(T - 1) * p->patch_embed->in * sizeof(float));
    m->patch_out = xalloc((size_t)(T - 1) * m->dim * sizeof(float));
    m->a8 = xalloc(T * widest);
    m->a16 = xalloc(T * widest * 2);
    m->acc = xalloc(T * widest * sizeof(int32_t));
    m->q16 = xalloc(T * attn * 2); m->k16 = xalloc(T * attn * 2); m->v16 = xalloc(T * attn * 2);
    m->y16 = xalloc(T * widest * 2);
    m->g16 = xalloc(T * neu * 2); m->u16 = xalloc(T * neu * 2); m->h16 = xalloc(T * neu * 2);
    m->q8 = xalloc(T * attn); m->k8 = xalloc(T * attn); m->v8 = xalloc(T * attn); m->ctx8 = xalloc(T * attn);
    return m;
}

void model_destroy(model_t *m)
{
    if (!m)
        return;
    free(m->x); free(m->n); free(m->hh); free(m->ctxf); free(m->cols); free(m->patch_out);
    free(m->a8); free(m->a16); free(m->acc);
    free(m->q16); free(m->k16); free(m->v16); free(m->y16); free(m->g16); free(m->u16); free(m->h16);
    free(m->q8); free(m->k8); free(m->v8); free(m->ctx8);
    free(m->stage); free(m->rdbuf); free(m->chk16); free(m->att_stage); free(m->att_chk);
    pl_free(m);
    free(m);
}

/* the whole GEMM + epilogue on the CPU, from the quantised input already in a8 / a16 */
static void linear_cpu(model_t *m, const layer_t *L, int rows, int16_t *out)
{
    if (L->in_bits == 8)
        gemm_s8(m->a8, L->w, m->acc, rows, L->out, L->in);
    else
        gemm_s16(m->a16, L->w, m->acc, rows, L->out, L->in);
    epilogue_int(m->acc, rows, L->out, L->bias, L->m0, L->shift, L->out_bits, out);
}

/* the same operation on the FPGA: pack the input, run, read the output matrix back
 * (sequential: the ARM waits for the engine; this is the --no-overlap behaviour) */
static int linear_fpga_seq(model_t *m, const layer_t *L, int rows, int16_t *out)
{
    const int K = L->in, N = L->out, planes = L->in_bits > 8 ? 2 : 1;
    const int out16 = L->out_bits > 8;
    double t0 = now_s();
    if (!(m->a_ok && m->a_rows == rows && m->a_k == K && m->a_bits == L->in_bits)) {
        const size_t bytes = fpga_acts_bytes(rows, K, planes);
        if (planes == 1)
            fpga_pack_acts8(m->a8, rows, K, m->stage);
        else
            fpga_pack_acts16(m->a16, rows, K, m->stage);
        memcpy(m->hw->region + FPGA_A_OFF, m->stage, bytes);
        accel_hw_clean(m->hw, FPGA_A_OFF, bytes);
        m->a_ok = 1; m->a_rows = rows; m->a_k = K; m->a_bits = L->in_bits;
    }
    double t1 = now_s();
    accel_op_t op = {.M = (uint32_t)rows, .N = (uint32_t)N, .K = (uint32_t)K, .in_bits = L->in_bits,
                     .out_bits = L->out_bits, .spp = planes == 2 ? 4 : 16,
                     .a_phys = ACCEL_REGION_PHYS + FPGA_A_OFF, .w_phys = ACCEL_REGION_PHYS + m->w_off[L - m->p->layers],
                     .out_phys = ACCEL_REGION_PHYS + FPGA_O_OFF};
    accel_stat_t st;
    if (accel_hw_run(m->hw, &op, &st, 60.0) != 0) {
        m->a_ok = 0;
        return -1;
    }
    double t2 = now_s();
    const size_t n = (size_t)rows * N;
    accel_hw_inval(m->hw, FPGA_O_OFF, n * (out16 ? 2 : 1));
    memcpy(m->rdbuf, m->hw->region + FPGA_O_OFF, n * (out16 ? 2 : 1));
    if (out16)
        memcpy(out, m->rdbuf, n * 2);
    else
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
            out[i] = ((const int8_t *)m->rdbuf)[i];
    double t3 = now_s();
    m->st.ops++;
    m->st.cycles += st.cycles;
    m->st.t_pack += t1 - t0;
    m->st.t_run += t2 - t1;
    m->st.t_read += t3 - t2;
    size_t bad = 0;
    if (m->check) {
        linear_cpu(m, L, rows, m->chk16);
        for (size_t i = 0; i < n; i++)
            bad += m->chk16[i] != out[i];
        m->st.mismatches += bad;
    }
    if (m->verbose)
        fprintf(stderr, "  fpga %-10s M=%d N=%-4d K=%-4d: %9llu cycles (%.1f ms at %.0f MHz), pack %.1f ms, run %.1f ms, "
                        "read %.1f ms%s\n", L->name, rows, N, K, (unsigned long long)st.cycles, st.cycles / (m->hw->mhz * 1e3), m->hw->mhz,
                (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, m->check ? (bad ? "  MISMATCH" : "  identical to CPU") : "");
    if (m->check && bad)
        fprintf(stderr, "  *** %s: %zu of %zu outputs differ from the CPU result\n", L->name, bad, n);
    return 0;
}

/* ---- overlapped version: the layer is cut into chunks of rows (whole engine panels) ----
 * While the engine works on chunk c, the ARM packs the input of chunk c+1 and reads back the
 * output of chunk c-1. Chunks touch disjoint parts of the activation and output buffers, so there
 * is no hazard; only one engine operation is in flight at a time. */

/* pack rows [row0, row0 + nrows) of the quantised input into the activation buffer of the region */
static void pack_chunk(model_t *m, const layer_t *L, int row0, int nrows)
{
    const int K = L->in, planes = L->in_bits > 8 ? 2 : 1;
    const size_t off = (size_t)(row0 / 16) * planes * (size_t)K * 16;      /* row0 is a multiple of 16 */
    const size_t bytes = fpga_acts_bytes(nrows, K, planes);
    if (planes == 1)
        fpga_pack_acts8(m->a8 + (size_t)row0 * K, nrows, K, m->stage + off);
    else
        fpga_pack_acts16(m->a16 + (size_t)row0 * K, nrows, K, m->stage + off);
    memcpy(m->hw->region + FPGA_A_OFF + off, m->stage + off, bytes);
    accel_hw_clean(m->hw, FPGA_A_OFF + off, bytes);
}

/* read rows [row0, row0 + nrows) of the output matrix into out (int16 container) */
static void read_chunk(model_t *m, const layer_t *L, int row0, int nrows, int16_t *out)
{
    const int N = L->out, out16 = L->out_bits > 8;
    const size_t esz = out16 ? 2 : 1, n = (size_t)nrows * N;
    const uint8_t *src = m->hw->region + FPGA_O_OFF + (size_t)row0 * N * esz;
    uint8_t *tmp = m->rdbuf + (size_t)row0 * N * esz;
    accel_hw_inval(m->hw, FPGA_O_OFF + (size_t)row0 * N * esz, n * esz);
    memcpy(tmp, src, n * esz);
    if (out16)
        memcpy(out + (size_t)row0 * N, tmp, n * 2);
    else {
        int16_t *o = out + (size_t)row0 * N;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
            o[i] = ((const int8_t *)tmp)[i];
    }
}

static int linear_fpga_chunked(model_t *m, const layer_t *L, int rows, int16_t *out)
{
    const int K = L->in, N = L->out, planes = L->in_bits > 8 ? 2 : 1;
    const int out16 = L->out_bits > 8;
    const size_t esz = out16 ? 2 : 1;
    int chunk = m->chunk_rows;
    if (chunk > rows)
        chunk = rows;
    const int nch = (rows + chunk - 1) / chunk;
    const int need_pack = !(m->a_ok && m->a_rows == rows && m->a_k == K && m->a_bits == L->in_bits);
    double t_pack = 0, t_read = 0, t_wait = 0;
    unsigned long long cycles = 0;
    const double w0 = now_s();

    if (need_pack) {
        const double t = now_s();
        pack_chunk(m, L, 0, chunk < rows ? chunk : rows);
        t_pack += now_s() - t;
    }
    for (int c = 0; c < nch; c++) {
        const int row0 = c * chunk, nr = rows - row0 < chunk ? rows - row0 : chunk;
        accel_op_t op = {.M = (uint32_t)nr, .N = (uint32_t)N, .K = (uint32_t)K, .in_bits = L->in_bits,
                         .out_bits = L->out_bits, .spp = planes == 2 ? 4 : 16,
                         .a_phys = ACCEL_REGION_PHYS + FPGA_A_OFF +
                                   (uint32_t)((size_t)(row0 / 16) * planes * (size_t)K * 16),
                         .w_phys = ACCEL_REGION_PHYS + m->w_off[L - m->p->layers],
                         .out_phys = ACCEL_REGION_PHYS + FPGA_O_OFF + (uint32_t)((size_t)row0 * N * esz)};
        double t = now_s();
        if (accel_hw_start(m->hw, &op) != 0) {
            m->a_ok = 0;
            return -1;
        }
        t_wait += now_s() - t;
        /* the engine is busy: do the ARM work that does not depend on this chunk's result */
        if (need_pack && c + 1 < nch) {
            t = now_s();
            const int n0 = (c + 1) * chunk;
            pack_chunk(m, L, n0, rows - n0 < chunk ? rows - n0 : chunk);
            t_pack += now_s() - t;
        }
        if (c > 0) {
            t = now_s();
            read_chunk(m, L, (c - 1) * chunk, chunk, out);
            t_read += now_s() - t;
        }
        accel_stat_t st;
        t = now_s();
        if (accel_hw_wait(m->hw, &op, &st, 60.0) != 0) {
            m->a_ok = 0;
            return -1;
        }
        t_wait += now_s() - t;
        cycles += st.cycles;
    }
    {
        const double t = now_s();
        const int last = (nch - 1) * chunk;
        read_chunk(m, L, last, rows - last, out);
        t_read += now_s() - t;
    }
    m->a_ok = 1; m->a_rows = rows; m->a_k = K; m->a_bits = L->in_bits;

    const size_t n = (size_t)rows * N;
    m->st.ops += nch;
    m->st.cycles += cycles;
    m->st.t_pack += t_pack;
    m->st.t_run += t_wait;
    m->st.t_read += t_read;
    m->st.t_lin_wall += now_s() - w0;
    size_t bad = 0;
    if (m->check) {
        linear_cpu(m, L, rows, m->chk16);
        for (size_t i = 0; i < n; i++)
            bad += m->chk16[i] != out[i];
        m->st.mismatches += bad;
    }
    if (m->verbose)
        fprintf(stderr, "  fpga %-10s M=%d N=%-4d K=%-4d: %d chunks, %9llu cycles, pack %.1f ms, wait %.1f ms, "
                        "read %.1f ms, wall %.1f ms%s\n", L->name, rows, N, K, nch, cycles, t_pack * 1e3, t_wait * 1e3,
                t_read * 1e3, (now_s() - w0) * 1e3, m->check ? (bad ? "  MISMATCH" : "  identical to CPU") : "");
    if (m->check && bad)
        fprintf(stderr, "  *** %s: %zu of %zu outputs differ from the CPU result\n", L->name, bad, n);
    return 0;
}

static int linear_fpga(model_t *m, const layer_t *L, int rows, int16_t *out)
{
    /* the chunk size is a whole number of engine panels (16 subtiles of 16 rows for 8-bit inputs) */
    if (m->chunk_rows <= 0 || rows <= m->chunk_rows)
        return linear_fpga_seq(m, L, rows, out);
    return linear_fpga_chunked(m, L, rows, out);
}

/* quantise the float input, run the INT GEMM and the epilogue; result in int16 container.
 * Returns 0, or -1 if the FPGA failed. reuse_quant: the input is the one quantised by the previous call. */
static int linear_int(model_t *m, const layer_t *L, const float *in, int rows, int16_t *out, int reuse_quant)
{
    if (!reuse_quant) {
        m->a_ok = 0;                    /* a new input: whatever is packed in the region is stale */
        if (L->in_bits == 8)
            TIMED(t_quant, quantize_f32(in, (size_t)rows * L->in, L->s_x, 8, m->a8));
        else
            TIMED(t_quant, quantize_f32(in, (size_t)rows * L->in, L->s_x, 16, m->a16));
    }
    if (m->hw && fpga_supports(L) && m->w_off[L - m->p->layers])
        return linear_fpga(m, L, rows, out);
    linear_cpu(m, L, rows, out);
    return 0;
}

static void dbg16(model_dbg_fn f, void *c, int b, const char *what, const int16_t *d, size_t n)
{
    char nm[48];
    if (!f)
        return;
    snprintf(nm, sizeof nm, "b%d.%s", b, what);
    f(c, nm, d, n, 2);
}

/* ======================= row-chunk pipeline between two attentions =======================
 *
 * Between the attention of block b and the attention of block b+1 everything works row by row:
 * proj, residual sum, LayerNorm, fc1 (gate, up), SwiGLU, LayerNorm, fc2, residual sum, then the next
 * block's LayerNorm, q, k, v and RoPE. The rows are cut into chunks; each chunk follows the same list of
 * steps. A step is either ARM work or one engine operation. One engine operation runs at a time (the
 * engine does one thing at a time anyway); while it runs the ARM executes ready steps of other chunks.
 * Chunks touch disjoint rows of the shared matrices and have their own scratch buffers and their own
 * slot in the region's activation and output areas, so no two steps conflict. Every step does exactly what
 * the sequential code does to the same rows, so the result is identical. */

typedef enum { PS_LN, PS_CTXF, PS_PREP, PS_RUN, PS_READ, PS_RES, PS_SWIGLU, PS_ROPE, PS_MLP, PS_RLP } pstep_kind;

typedef struct {
    pstep_kind kind;
    const layer_t *L;           /* PREP, RUN, READ; RLP: the layer whose output is added to x */
    const layer_t *L2;          /* RLP: the layer whose INT8 input is prepared from the LayerNorm of x */
    const block_t *B;           /* CTXF, SWIGLU, ROPE, MLP */
    int from_hh;                /* LN: the input is hh, else the residual stream x */
    const float *g, *b;         /* LN */
    int dim;                    /* LN */
    int direct;                 /* PREP: copy ctx8 (already INT8 in the layer's scale) instead of quantising pl_n */
    int16_t *dst;               /* READ: matrix [tokens][L->out] that receives the rows */
    int8_t *dst8;               /* READ: instead of dst, an INT8 matrix [tokens][L->out] (the region holds INT8 rows) */
    int ncols;                  /* READ: columns of INT16 rows when they are not L->out (fused SwiGLU output) */
    int rope_hd;                /* RUN: head dimension when the epilogue rotates the output (RoPE), else 0 */
    int sw;                     /* RUN: the fused SwiGLU of block B (interleaved weights); MLP, SWIGLU: h comes from fc1 */
    int v_only;                 /* ROPE: only the v conversion (q and k are already rotated) */
    const int16_t *src;         /* RES: x += src * scale */
    double scale;
} pstep_t;

typedef struct { int row0, nr, pc; } pjob_t;

static void pl_free(model_t *m)
{
    for (int j = 0; j < PL_MAX; j++) {
        free(m->pl_n[j]); free(m->pl_a8[j]); free(m->pl_a16[j]); free(m->pl_stage[j]); free(m->pl_acc[j]);
        free(m->pl_chk[j]); free(m->pl_row[j]);
        m->pl_row[j] = NULL;
        m->pl_n[j] = NULL; m->pl_a8[j] = NULL; m->pl_a16[j] = NULL; m->pl_stage[j] = NULL; m->pl_acc[j] = NULL;
        m->pl_chk[j] = NULL;
    }
    m->pl_ok = 0;
}

/* (re)allocates the per-chunk buffers for the current chunk size; returns 1 if the pipeline can be used */
static int pl_prepare(model_t *m)
{
    const int cap = m->chunk_rows;
    if (!m->hw || !m->pipeline || cap <= 0 || cap % 16)
        return 0;
    if (m->pl_cap == cap)
        return m->pl_ok;
    pl_free(m);
    m->pl_cap = cap;
    const pkg_t *p = m->p;
    const int nj = (m->tokens + cap - 1) / cap;
    int maxk = m->dim > m->max_neurons ? m->dim : m->max_neurons;
    for (int bi = 0; bi < p->n_blocks; bi++) {
        const block_t *B = &p->blocks[bi];
        const layer_t *ls[7] = {B->q, B->k, B->v, B->proj, B->fc1g, B->fc1x, B->fc2};
        for (int i = 0; i < 7; i++) {
            if (!fpga_supports(ls[i]) || !(m->w_off[ls[i] - p->layers] || (m->sw_off[bi] && (i == 4 || i == 5)))) {
                fprintf(stderr, "model: pipeline off: layer %s is not on the FPGA\n", ls[i]->name);
                return 0;                   /* some layer stays on the CPU: no pipeline */
            }
            if (ls[i]->in > maxk) maxk = ls[i]->in;
            if (ls[i]->out > maxk) maxk = ls[i]->out;
        }
    }
    const size_t slot = ((size_t)cap * maxk * 2 + 63) & ~(size_t)63;
    if (nj > PL_MAX || slot * (size_t)nj > PL_AREA) {
        fprintf(stderr, "model: pipeline off: %d chunks of %d rows do not fit %d slots (use a larger --chunk)\n", nj,
                cap, PL_MAX);
        return 0;
    }
    m->pl_slot = slot;
    for (int j = 0; j < nj; j++) {
        m->pl_n[j] = malloc((size_t)cap * maxk * sizeof(float));
        m->pl_a8[j] = malloc((size_t)cap * maxk);
        m->pl_a16[j] = malloc((size_t)cap * maxk * 2);
        m->pl_stage[j] = malloc(slot);
        m->pl_row[j] = malloc((size_t)32 * maxk * sizeof(float));
        if (m->check) {
            m->pl_acc[j] = malloc((size_t)cap * maxk * 4);
            m->pl_chk[j] = malloc((size_t)cap * maxk * 2);
        }
        if (!m->pl_n[j] || !m->pl_a8[j] || !m->pl_a16[j] || !m->pl_stage[j] || !m->pl_row[j] || (m->check && (!m->pl_acc[j] || !m->pl_chk[j]))) {
            pl_free(m);
            return 0;
        }
    }
    m->pl_ok = 1;
    return 1;
}

/* adds the time since the last lap to a counter of model_stats_t and starts a new lap */
#define PL_LAP(field) do { const double _n = now_s(); m->st.field += _n - tt; tt = _n; } while (0)

static void pl_tick_inside(model_t *m);

/* one ARM step on one chunk */
static void pl_arm(model_t *m, const pstep_t *s, const pjob_t *j, int slot)
{
    const pkg_t *p = m->p;
    const int D = m->dim, r0 = j->row0, nr = j->nr;
    double tt = now_s();
    const double t_step0 = tt;          /* PL_LAP moves tt, so the trace needs the step's own start */
    switch (s->kind) {
    case PS_LN: {
        const float *in = (s->from_hh ? m->hh : m->x) + (size_t)r0 * s->dim;
        layer_norm(in, nr, s->dim, s->g, s->b, p->ln_eps, m->pl_n[slot]);
        PL_LAP(pl_ln);
        break;
    }
    case PS_CTXF: {
        const int A = s->B->q->out;
        const int8_t *c = m->ctx8 + (size_t)r0 * A;
        float *o = m->pl_n[slot];
        const double sc = s->B->s_ctx;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)nr * A; i++)
            o[i] = (float)((double)c[i] * sc);
        PL_LAP(pl_quant);
        break;
    }
    case PS_PREP: {
        const layer_t *L = s->L;
        const int K = L->in, planes = L->in_bits > 8 ? 2 : 1;
        if (s->direct)
            memcpy(m->pl_a8[slot], m->ctx8 + (size_t)r0 * K, (size_t)nr * K);
        else if (planes == 1)
            quantize_f32(m->pl_n[slot], (size_t)nr * K, L->s_x, 8, m->pl_a8[slot]);
        else
            quantize_f32(m->pl_n[slot], (size_t)nr * K, L->s_x, 16, m->pl_a16[slot]);
        PL_LAP(pl_quant);
        if (planes == 1)
            fpga_pack_acts8(m->pl_a8[slot], nr, K, m->pl_stage[slot]);
        else
            fpga_pack_acts16(m->pl_a16[slot], nr, K, m->pl_stage[slot]);
        memcpy(m->hw->region + FPGA_A_OFF + (size_t)slot * m->pl_slot, m->pl_stage[slot],
               fpga_acts_bytes(nr, K, planes));
        accel_hw_clean(m->hw, FPGA_A_OFF + (size_t)slot * m->pl_slot, fpga_acts_bytes(nr, K, planes));
        PL_LAP(pl_pack);
        break;
    }
    case PS_READ: {
        const layer_t *L = s->L;
        const int N = s->ncols ? s->ncols : L->out, out16 = s->dst8 ? 0 : s->ncols ? 1 : L->out_bits > 8;
        const size_t n = (size_t)nr * N;
        const uint8_t *src = m->hw->region + FPGA_O_OFF + (size_t)slot * m->pl_slot;
        int16_t *dst = s->dst8 ? NULL : s->dst + (size_t)r0 * N;
        accel_hw_inval(m->hw, FPGA_O_OFF + (size_t)slot * m->pl_slot, n * (out16 ? 2 : 1));
        if (s->dst8)
            memcpy(s->dst8 + (size_t)r0 * N, src, n);
        else if (out16)
            memcpy(dst, src, n * 2);
        else {
            uint8_t *tmp = m->pl_stage[slot];
            memcpy(tmp, src, n);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
                dst[i] = ((const int8_t *)tmp)[i];
        }
        PL_LAP(pl_read);
        if (m->check && !s->dst8 && !s->ncols) {
            if (L->in_bits == 8)
                gemm_s8(m->pl_a8[slot], L->w, m->pl_acc[slot], nr, N, L->in);
            else
                gemm_s16(m->pl_a16[slot], L->w, m->pl_acc[slot], nr, N, L->in);
            epilogue_int(m->pl_acc[slot], nr, N, L->bias, L->m0, L->shift, L->out_bits, m->pl_chk[slot]);
            size_t bad = 0;
            for (size_t i = 0; i < n; i++)
                bad += m->pl_chk[slot][i] != dst[i];
            m->st.mismatches += bad;
            if (bad)
                fprintf(stderr, "  *** %s rows %d..%d: %zu of %zu outputs differ from the CPU result\n", L->name, r0,
                        r0 + nr - 1, bad, n);
        }
        break;
    }
    case PS_RES: {
        float *x = m->x + (size_t)r0 * D;
        const int16_t *y = s->src + (size_t)r0 * D;
        const double sc = s->scale;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)nr * D; i++)
            x[i] += (float)((double)y[i] * sc);
        PL_LAP(pl_other);
        break;
    }
    case PS_SWIGLU: {
        const block_t *B = s->B;
        const int N = B->neurons;
        const int64_t sw_rnd = (int64_t)1 << (B->sw_sh - 1);
        const int qm_in = qmax_of(p->silu_in_bits);
        const size_t off = (size_t)r0 * N;
        if (s->sw) {
            /* the engine did the SwiGLU: h16 was read from the region, only the float scaling is left */
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < (ptrdiff_t)nr * N; i++)
                m->hh[off + i] = (float)((double)m->h16[off + i] * B->s_h);
            PL_LAP(pl_swiglu);
            break;
        }
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)nr * N; i++) {
            int64_t y = (int64_t)B->silu[m->g16[off + i] + qm_in] * m->u16[off + i];
            y = (y * B->sw_m0 + sw_rnd) >> B->sw_sh;
            m->h16[off + i] = (int16_t)(y > 32767 ? 32767 : y < -32767 ? -32767 : y);
            m->hh[off + i] = (float)((double)m->h16[off + i] * B->s_h);
        }
        PL_LAP(pl_swiglu);
        break;
    }
    case PS_MLP: {
        /* READ(fc1x), SWIGLU, LN(mlpn) and PREP(fc2) fused by 16-row groups: g from g16, u straight from the
         * region's output slot, the same per-element SwiGLU, layer_norm_row (the code of layer_norm) and
         * quantize_f32, then the group's subtile is packed straight into the slot's activation area.
         * h16 and hh are not written: inside the pipeline only PS_LN (from_hh) reads hh, and it is replaced. */
        const block_t *B = s->B;
        const layer_t *L = B->fc2;
        const int N = B->neurons, K = L->in;
        const int64_t sw_rnd = (int64_t)1 << (B->sw_sh - 1);
        const int qm_in = qmax_of(p->silu_in_bits);
        const float epsf = (float)p->ln_eps;
        const size_t o_off = FPGA_O_OFF + (size_t)slot * m->pl_slot;
        const size_t a_off = FPGA_A_OFF + (size_t)slot * m->pl_slot;
        const int ub = s->sw ? 2 : B->fc1x->out_bits > 8 ? 2 : 1;    /* the region's output: h (INT16) or fc1x's */
        const uint8_t *ureg = m->hw->region + o_off;
        int16_t *a16 = m->pl_a16[slot];
        for (int g0 = 0; g0 < nr; g0 += 16) {
            const int gn = nr - g0 < 16 ? nr - g0 : 16;
            accel_hw_inval(m->hw, o_off + (size_t)g0 * N * ub, (size_t)gn * N * ub);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < gn; i++) {
                const int16_t *gr = m->g16 + (size_t)(r0 + g0 + i) * N;
                const uint8_t *ur = ureg + (size_t)(g0 + i) * N * ub;
                float *hr = m->pl_row[slot] + (size_t)i * N;
                float *nrow = m->pl_row[slot] + (size_t)(16 + i) * N;
                const int16_t *u16 = (const int16_t *)ur;
                const int8_t *u8 = (const int8_t *)ur;
                if (s->sw)
                    for (int k = 0; k < N; k++)
                        hr[k] = (float)((double)u16[k] * B->s_h);   /* h from the engine's fused SwiGLU */
                else
                    for (int k = 0; k < N; k++) {
                        const int u = ub == 2 ? u16[k] : u8[k];
                        int64_t y = (int64_t)B->silu[gr[k] + qm_in] * u;
                        y = (y * B->sw_m0 + sw_rnd) >> B->sw_sh;
                        const int16_t h = (int16_t)(y > 32767 ? 32767 : y < -32767 ? -32767 : y);
                        hr[k] = (float)((double)h * B->s_h);
                    }
                layer_norm_row(hr, N, B->mlpn_g, B->mlpn_b, epsf, nrow);
                quantize_f32(nrow, (size_t)K, L->s_x, 16, a16 + (size_t)i * K);
            }
            uint8_t *hi = m->hw->region + a_off + (size_t)(g0 / 16) * 2 * K * 16;
            fpga_pack_acts16_sub(a16, gn, K, hi, hi + (size_t)K * 16);
            pl_tick_inside(m);          /* ~2 ms per group: the engine does not wait for the whole step */
        }
        accel_hw_clean(m->hw, a_off, fpga_acts_bytes(nr, K, 2));
        PL_LAP(pl_swiglu);
        break;
    }
    case PS_RLP: {
        /* READ, RES, LN and PREP (8 bits) fused by 16-row groups: the layer's output is read straight from the
         * region's output slot, x += y * scale row by row, layer_norm_row of the new x, quantise to the next layer's
         * INT8 input and pack the group's subtile straight into the slot's activation area. */
        const layer_t *L = s->L, *L2 = s->L2;
        const int K = L2->in, ob = L->out_bits > 8 ? 2 : 1;
        const double sc = s->scale;
        const float epsf = (float)p->ln_eps;
        const size_t o_off = FPGA_O_OFF + (size_t)slot * m->pl_slot;
        const size_t a_off = FPGA_A_OFF + (size_t)slot * m->pl_slot;
        int8_t *a8 = m->pl_a8[slot];
        for (int g0 = 0; g0 < nr; g0 += 16) {
            const int gn = nr - g0 < 16 ? nr - g0 : 16;
            accel_hw_inval(m->hw, o_off + (size_t)g0 * D * ob, (size_t)gn * D * ob);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < gn; i++) {
                float *x = m->x + (size_t)(r0 + g0 + i) * D;
                const uint8_t *yr = m->hw->region + o_off + (size_t)(g0 + i) * D * ob;
                float *nrow = m->pl_row[slot] + (size_t)i * D;
                if (ob == 2)
                    for (int k = 0; k < D; k++)
                        x[k] += (float)((double)((const int16_t *)yr)[k] * sc);
                else
                    for (int k = 0; k < D; k++)
                        x[k] += (float)((double)((const int8_t *)yr)[k] * sc);
                layer_norm_row(x, D, s->g, s->b, epsf, nrow);
                quantize_f32(nrow, (size_t)K, L2->s_x, 8, a8 + (size_t)i * K);
            }
            fpga_pack_acts8_sub(a8, gn, K, m->hw->region + a_off + (size_t)(g0 / 16) * K * 16);
            pl_tick_inside(m);          /* ~1 ms per group: the engine does not wait for the whole step */
        }
        accel_hw_clean(m->hw, a_off, fpga_acts_bytes(nr, K, 1));
        PL_LAP(pl_ln);
        break;
    }
    case PS_ROPE: {
        const block_t *B = s->B;
        const int A = B->q->out, H = B->heads, hd = A / H;
        const size_t off = (size_t)r0 * A;
        if (!s->v_only) {
            rope_int_range(m->q16 + off, r0, nr, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, m->q8 + off);
            rope_int_range(m->k16 + off, r0, nr, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, m->k8 + off);
        }
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)nr * A; i++)
            m->v8[off + i] = (int8_t)m->v16[off + i];
        PL_LAP(pl_rope);
        break;
    }
    default:
        break;
    }
    if (g_tr) {
        static const char *const nm[] = {"ln", "ctxf", "prep", "run", "read", "res", "swiglu", "rope", "mlp", "rlp"};
        tr_add("arm", nm[s->kind], slot, t_step0, now_s(), 0);
    }
}

/* runs the step list on all chunks; 0 on success */
/* engine side of the pipeline loop */
typedef struct pl_sched {
    pjob_t *jobs;
    int nj;
    const pstep_t *steps;
    int ns;
    int remaining, inflight;    /* chunks not finished; chunk on the engine, or -1 */
    accel_op_t op;              /* the operation in flight */
    double t_op, t_end;         /* start of the operation in flight; when it was first seen finished (trace) */
    int in_j;
    const char *in_name;
    int err;                    /* set by a tick that failed inside an ARM step */
} pl_sched_t;

/* Collects the engine operation that has finished and starts the first chunk waiting for the engine. Called by
 * the pipeline loop and, through m->pl_sched, from inside long ARM steps between row groups, so the engine gets
 * its next operation within a millisecond instead of waiting for the whole step. The chunk whose ARM step is in
 * progress is never started: its program counter still points at that ARM step, not at a PS_RUN. 0 on success. */
static int pl_engine_tick(model_t *m, pl_sched_t *S)
{
    if (S->inflight >= 0 && !accel_hw_busy(m->hw)) {
        accel_stat_t st;
        if (accel_hw_wait(m->hw, &S->op, &st, 60.0) != 0)
            return -1;
        m->st.cycles += st.cycles;
        m->st.ops++;
        if (g_tr)
            tr_add("eng", S->in_name, S->in_j, S->t_op, S->t_end > 0 ? S->t_end : now_s(), st.cycles);
        if (++S->jobs[S->inflight].pc == S->ns)
            S->remaining--;
        S->inflight = -1;
    }
    if (S->inflight < 0) {
        for (int j = 0; j < S->nj; j++) {
            if (S->jobs[j].pc >= S->ns || S->steps[S->jobs[j].pc].kind != PS_RUN)
                continue;
            const pstep_t *st = &S->steps[S->jobs[j].pc];
            const layer_t *L = st->L;
            const int K = L->in, N = L->out, planes = L->in_bits > 8 ? 2 : 1;
            accel_op_t *op = &S->op;
            memset(op, 0, sizeof *op);
            op->M = (uint32_t)S->jobs[j].nr; op->N = (uint32_t)N; op->K = (uint32_t)K;
            op->in_bits = L->in_bits; op->out_bits = L->out_bits; op->spp = planes == 2 ? 4 : 16;
            op->a_phys = ACCEL_REGION_PHYS + FPGA_A_OFF + (uint32_t)((size_t)j * m->pl_slot);
            op->w_phys = ACCEL_REGION_PHYS + m->w_off[L - m->p->layers];
            op->out_phys = ACCEL_REGION_PHYS + FPGA_O_OFF + (uint32_t)((size_t)j * m->pl_slot);
            const char *name = L->name;
            if (st->rope_hd) {
                /* RoPE in the epilogue: INT8 rows, already rotated; the token of row 0 is the chunk's first row */
                op->out_bits = 8;
                op->rope_hd = (uint32_t)st->rope_hd;
                op->row0 = (uint32_t)S->jobs[j].row0;
            } else if (st->sw) {
                /* fc1 with the fused SwiGLU: gate and up tiles interleaved, INT16 h of B->neurons columns */
                const block_t *B = st->B;
                op->N = 2u * (uint32_t)B->neurons;
                op->out_bits = 8;
                op->w_phys = ACCEL_REGION_PHYS + m->sw_off[B - m->p->blocks];
                op->stride_out = 2u * (uint32_t)B->neurons;
                op->swiglu = 1;
                op->sw_tab = B->silu;
                op->sw_m0 = (uint32_t)B->sw_m0;
                op->sw_shift = (uint32_t)B->sw_sh;
                name = "fc1_swiglu";
            }
            if (accel_hw_start(m->hw, op) != 0)
                return -1;
            S->inflight = j;
            S->t_op = now_s();
            S->t_end = 0;
            S->in_j = j;
            S->in_name = name;
            break;
        }
    }
    return 0;
}

/* inside an ARM step: give the engine its next operation if it is idle; errors are reported through S->err */
static void pl_tick_inside(model_t *m)
{
    if (m->pl_sched && !m->pl_sched->err && pl_engine_tick(m, m->pl_sched) != 0)
        m->pl_sched->err = 1;
}

static int pl_run(model_t *m, const pstep_t *steps, int ns)
{
    const int T = m->tokens, cap = m->pl_cap;
    const int nj = (T + cap - 1) / cap;
    pjob_t jobs[PL_MAX];
    for (int j = 0; j < nj; j++) {
        jobs[j].row0 = j * cap;
        jobs[j].nr = T - j * cap < cap ? T - j * cap : cap;
        jobs[j].pc = 0;
    }
    pl_sched_t S = {.jobs = jobs, .nj = nj, .steps = steps, .ns = ns, .remaining = nj, .inflight = -1,
                    .in_j = -1, .in_name = ""};
    const double w0 = now_s();
    m->a_ok = 0;                        /* the region's activation area is about to hold chunk data (also if we fail) */
    m->pl_sched = &S;
    while (S.remaining > 0) {
        /* 1 + 2. collect the finished operation and start the next one */
        if (pl_engine_tick(m, &S) != 0) {
            m->pl_sched = NULL;
            return -1;
        }
        /* 3. one ARM step of a chunk that is not on the engine (long steps call pl_engine_tick between row groups) */
        int did = 0;
        for (int j = 0; j < nj && !did; j++) {
            if (j == S.inflight || jobs[j].pc >= ns || steps[jobs[j].pc].kind == PS_RUN)
                continue;
            pl_arm(m, &steps[jobs[j].pc], &jobs[j], j);
            if (S.err) {
                m->pl_sched = NULL;
                return -1;
            }
            if (++jobs[j].pc == ns)
                S.remaining--;
            did = 1;
            if (g_tr && S.inflight >= 0 && S.t_end == 0 && !accel_hw_busy(m->hw))
                S.t_end = now_s();
        }
        /* 4. nothing for the ARM to do: wait for the engine (spin on its status) */
        if (!did && S.inflight >= 0) {
            const double t = now_s();
            while (accel_hw_busy(m->hw) && now_s() - S.t_op < 60.0)
                ;
            m->st.pl_idle += now_s() - t;
            if (g_tr) {
                const double te = now_s();
                tr_add("idle", "wait", S.in_j, t, te, 0);
                if (S.t_end == 0 && !accel_hw_busy(m->hw))
                    S.t_end = te;
            }
            if (now_s() - S.t_op >= 60.0) {
                fprintf(stderr, "pipeline: the engine did not finish in 60 s\n");
                accel_hw_wait(m->hw, &S.op, NULL, 0.1);
                m->pl_sched = NULL;
                return -1;
            }
        }
    }
    m->pl_sched = NULL;
    m->st.pl_wall += now_s() - w0;
    m->st.t_lin_wall += now_s() - w0;
    m->a_ok = 0;                        /* the region's activation area now holds chunk data */
    return 0;
}

static pstep_t *pl_add(pstep_t *s, int *n, pstep_kind k)
{
    pstep_t *r = &s[(*n)++];
    memset(r, 0, sizeof *r);
    r->kind = k;
    return r;
}

/* PREP (unless the previous layer's quantised input is reused), RUN, READ. With rope_hd the epilogue rotates the output
 * (RoPE) and the INT8 rows go to dst8; dst8 alone reads an INT8 layer straight into an INT8 matrix. */
static void pl_linear_ex(pstep_t *s, int *n, const layer_t *L, const layer_t *prev, int16_t *dst, int8_t *dst8,
                         int rope_hd, int direct)
{
    const int reuse = prev && prev->s_x == L->s_x && prev->in_bits == L->in_bits && prev->in == L->in && !direct;
    if (!reuse) {
        pstep_t *p = pl_add(s, n, PS_PREP);
        p->L = L; p->direct = direct;
    }
    pstep_t *u = pl_add(s, n, PS_RUN);
    u->L = L; u->rope_hd = rope_hd;
    pstep_t *r = pl_add(s, n, PS_READ);
    r->L = L; r->dst = dst; r->dst8 = dst8;
}

static void pl_linear(pstep_t *s, int *n, const layer_t *L, const layer_t *prev, int16_t *dst, int direct)
{
    pl_linear_ex(s, n, L, prev, dst, NULL, 0, direct);
}

/* LayerNorm 1 of block B, q, k, v and RoPE: what a block starts with */
static void pl_block_head(model_t *m, pstep_t *s, int *n, const block_t *B, int fused)
{
    if (m->hw_rope) {
        /* q and k are rotated by the engine's epilogue and read as INT8 straight into q8 / k8; v (INT8) too */
        const int hd = B->q->out / B->heads, v8 = B->v->out_bits == 8;
        if (fused) {
            pstep_t *u = pl_add(s, n, PS_RUN);
            u->L = B->q; u->rope_hd = hd;
            pstep_t *r = pl_add(s, n, PS_READ);
            r->L = B->q; r->dst8 = m->q8;
        } else {
            pstep_t *ln = pl_add(s, n, PS_LN);
            ln->dim = m->dim; ln->g = B->n1_g; ln->b = B->n1_b;
            pl_linear_ex(s, n, B->q, NULL, NULL, m->q8, hd, 0);
        }
        pl_linear_ex(s, n, B->k, B->q, NULL, m->k8, hd, 0);
        pl_linear_ex(s, n, B->v, B->k, v8 ? NULL : m->v16, v8 ? m->v8 : NULL, 0, 0);
        if (!v8) {
            pstep_t *r = pl_add(s, n, PS_ROPE);
            r->B = B; r->v_only = 1;
        }
        return;
    }
    if (fused) {
        /* a PS_RLP before this already did LN 1 and the packing of q's input */
        pl_add(s, n, PS_RUN)->L = B->q;
        pstep_t *r = pl_add(s, n, PS_READ);
        r->L = B->q; r->dst = m->q16;
    } else {
        pstep_t *ln = pl_add(s, n, PS_LN);
        ln->dim = m->dim; ln->g = B->n1_g; ln->b = B->n1_b;
        pl_linear(s, n, B->q, NULL, m->q16, 0);
    }
    pl_linear(s, n, B->k, B->q, m->k16, 0);
    pl_linear(s, n, B->v, B->k, m->v16, 0);
    pl_add(s, n, PS_ROPE)->B = B;
}

/* everything after the attention of block bi, up to and including the next block's RoPE (if any) */
static int pl_after_attention(model_t *m, int bi, int nb)
{
    const pkg_t *p = m->p;
    const block_t *B = &p->blocks[bi];
    const int A = B->q->out;
    pstep_t s[48];
    int n = 0;
    const int ctx_direct = B->proj->in_bits == 8 && B->proj->in == A &&
                           fabs(B->s_ctx - B->proj->s_x) <= 1e-9 * fabs(B->proj->s_x);
    /* PS_RLP (not with --check: the per-layer verification needs the separate steps) fuses READ, RES, LN and the
     * INT8 PREP of the next layer; the next layer must take the LayerNorm of x as an INT8 input */
    const int rlp = !m->check && m->rlp_fuse;
    const int rlp1 = rlp && B->fc1g->in_bits == 8 && B->fc1g->in == m->dim;
    const int rlp2 = rlp && bi + 1 < nb && p->blocks[bi + 1].q->in_bits == 8 && p->blocks[bi + 1].q->in == m->dim;
    if (!ctx_direct)
        pl_add(s, &n, PS_CTXF)->B = B;
    pstep_t *ln;
    /* fused SwiGLU on fc1: one engine operation over the interleaved gate / up weights gives h (INT16), instead of
     * fc1g, fc1x and the SwiGLU on the ARM (the weights of such a block exist only interleaved, see model_attach_fpga) */
    const int swf = m->sw_off[bi] != 0;
    pstep_t *swrun = NULL;
    if (rlp1) {
        pstep_t *pp = pl_add(s, &n, PS_PREP);
        pp->L = B->proj; pp->direct = ctx_direct;
        pl_add(s, &n, PS_RUN)->L = B->proj;
        pstep_t *f = pl_add(s, &n, PS_RLP);
        f->L = B->proj; f->L2 = B->fc1g; f->scale = B->s_proj;
        f->dim = m->dim; f->g = B->n2_g; f->b = B->n2_b;
        if (swf) {
            swrun = pl_add(s, &n, PS_RUN);
        } else {
            pl_add(s, &n, PS_RUN)->L = B->fc1g;
            pstep_t *rd = pl_add(s, &n, PS_READ);
            rd->L = B->fc1g; rd->dst = m->g16;
        }
    } else {
        pl_linear(s, &n, B->proj, NULL, m->y16, ctx_direct);
        pstep_t *r = pl_add(s, &n, PS_RES);
        r->src = m->y16; r->scale = B->s_proj;
        ln = pl_add(s, &n, PS_LN);
        ln->dim = m->dim; ln->g = B->n2_g; ln->b = B->n2_b;
        if (swf) {
            pstep_t *pp = pl_add(s, &n, PS_PREP);
            pp->L = B->fc1g;
            swrun = pl_add(s, &n, PS_RUN);
        } else
            pl_linear(s, &n, B->fc1g, NULL, m->g16, 0);
    }
    if (swrun) {
        swrun->L = B->fc1g; swrun->B = B; swrun->sw = 1;
    }
    /* fused MLP tail (not with --check: the per-layer verification needs the separate steps). fc1g and fc1x
     * share the output slot, so READ(fc1g) -> g16 stays: a second output slot per chunk does not fit the 8 MiB
     * output area at the usual chunk sizes (6 chunks x 1.18 MB x 2 > 8 MiB). */
    const int fuse = !m->check && m->mlp_fuse && B->fc2->in_bits > 8 && B->fc2->in == B->neurons &&
                     B->fc1x->out == B->neurons && B->fc1x->s_x == B->fc1g->s_x &&
                     B->fc1x->in_bits == B->fc1g->in_bits && B->fc1x->in == B->fc1g->in;
    const int fuse_sw = !m->check && m->mlp_fuse && B->fc2->in_bits > 8 && B->fc2->in == B->neurons;
    if (swf && fuse_sw) {
        pstep_t *f = pl_add(s, &n, PS_MLP);
        f->B = B; f->sw = 1;
        pl_add(s, &n, PS_RUN)->L = B->fc2;
    } else if (swf) {
        pstep_t *rd = pl_add(s, &n, PS_READ);
        rd->L = B->fc1g; rd->dst = m->h16; rd->ncols = B->neurons;
        pstep_t *f = pl_add(s, &n, PS_SWIGLU);
        f->B = B; f->sw = 1;
        ln = pl_add(s, &n, PS_LN);
        ln->from_hh = 1; ln->dim = B->neurons; ln->g = B->mlpn_g; ln->b = B->mlpn_b;
        pstep_t *pp = pl_add(s, &n, PS_PREP);
        pp->L = B->fc2;
        pl_add(s, &n, PS_RUN)->L = B->fc2;
    } else if (fuse) {
        pl_add(s, &n, PS_RUN)->L = B->fc1x;   /* reuses fc1g's quantised input */
        pl_add(s, &n, PS_MLP)->B = B;
        pl_add(s, &n, PS_RUN)->L = B->fc2;
    } else {
        pl_linear(s, &n, B->fc1x, B->fc1g, m->u16, 0);
        pl_add(s, &n, PS_SWIGLU)->B = B;
        ln = pl_add(s, &n, PS_LN);
        ln->from_hh = 1; ln->dim = B->neurons; ln->g = B->mlpn_g; ln->b = B->mlpn_b;
        pstep_t *pp = pl_add(s, &n, PS_PREP);
        pp->L = B->fc2;
        pl_add(s, &n, PS_RUN)->L = B->fc2;
    }
    if (rlp2) {
        const block_t *N = &p->blocks[bi + 1];
        pstep_t *f = pl_add(s, &n, PS_RLP);
        f->L = B->fc2; f->L2 = N->q; f->scale = B->s_fc2;
        f->dim = m->dim; f->g = N->n1_g; f->b = N->n1_b;
        pl_block_head(m, s, &n, N, 1);
        return pl_run(m, s, n);
    }
    pstep_t *rd = pl_add(s, &n, PS_READ);
    rd->L = B->fc2; rd->dst = m->y16;
    pstep_t *r = pl_add(s, &n, PS_RES);
    r->src = m->y16; r->scale = B->s_fc2;
    if (bi + 1 < nb)
        pl_block_head(m, s, &n, &p->blocks[bi + 1], 0);
    return pl_run(m, s, n);
}

/* Patch embedding GEMM on the engine: INT8 input from m->a8, raw INT32 accumulators (no bias, no epilogue) into
 * m->acc. Returns 0 when it ran on the engine; -1 when it did not (no FPGA, --cpu-patch, not packed, or an engine
 * error) and the caller must run gemm_s8. With --check it also computes the CPU accumulators and counts differences. */
static int patch_fpga(model_t *m, const layer_t *L, int rows)
{
    const size_t wi = (size_t)(L - m->p->layers);
    if (!m->hw || m->cpu_patch || !m->w_off[wi])
        return -1;
    const int K = L->in, N = L->out;
    const size_t n = (size_t)rows * N;
    if (fpga_acts_bytes(rows, K, 1) > m->stage_len || n * 4 > FPGA_W_OFF - FPGA_O_OFF)
        return -1;
    fpga_pack_acts8(m->a8, rows, K, m->stage);
    memcpy(m->hw->region + FPGA_A_OFF, m->stage, fpga_acts_bytes(rows, K, 1));
    accel_hw_clean(m->hw, FPGA_A_OFF, fpga_acts_bytes(rows, K, 1));
    m->a_ok = 0;
    accel_op_t op = {.M = (uint32_t)rows, .N = (uint32_t)N, .K = (uint32_t)K, .in_bits = 8, .out_bits = 32,
                     .spp = 16, .a_phys = ACCEL_REGION_PHYS + FPGA_A_OFF, .w_phys = ACCEL_REGION_PHYS + m->w_off[wi],
                     .out_phys = ACCEL_REGION_PHYS + FPGA_O_OFF, .raw32 = 1, .stride_out = (uint32_t)N * 4};
    accel_stat_t st;
    if (accel_hw_run(m->hw, &op, &st, 60.0) != 0)
        return -1;
    accel_hw_inval(m->hw, FPGA_O_OFF, n * 4);
    memcpy(m->acc, m->hw->region + FPGA_O_OFF, n * 4);
    m->st.ops++;
    m->st.cycles += st.cycles;
    if (m->check) {
        int32_t *ref = xalloc(n * sizeof *ref);
        gemm_s8(m->a8, L->w, ref, rows, N, K);
        size_t bad = 0;
        for (size_t i = 0; i < n; i++)
            bad += ref[i] != m->acc[i];
        free(ref);
        m->st.mismatches += bad;
        if (bad)
            fprintf(stderr, "  *** patch_embed: %zu of %zu accumulators differ from the CPU result\n", bad, n);
        else if (m->verbose)
            fprintf(stderr, "  fpga patch_embed M=%d N=%d K=%d: identical to CPU\n", rows, N, K);
    }
    return 0;
}

void model_set_att_overlap(model_t *m, int mode)
{
    m->att_overlap = mode;
}

void model_set_cpu_patch(model_t *m, int on)
{
    m->cpu_patch = on;
}

int model_forward(model_t *m, const float *img, float logits[4], model_dbg_fn dbg, void *dbg_ctx)
{
    const pkg_t *p = m->p;
    const int T = m->tokens, D = m->dim, NP = T - 1;
    const int side = p->size / p->patch;           /* 32 */
    const int pk = p->patch, ps = p->patch_embed->in;   /* 14, 588 */

    const double t_patch0 = now_s();
    if (g_tr)
        tr_begin();
    /* ---- patch embedding: unfold (channel, ky, kx), INT8 GEMM, float de-quantisation ---- */
#pragma omp parallel for schedule(static) collapse(2)
    for (int py = 0; py < side; py++)
        for (int px = 0; px < side; px++) {
            float *c = m->cols + (size_t)(py * side + px) * ps;
            for (int ch = 0; ch < p->channels; ch++)
                for (int ky = 0; ky < pk; ky++)
                    for (int kx = 0; kx < pk; kx++)
                        c[(ch * pk + ky) * pk + kx] =
                            img[((size_t)ch * p->size + (size_t)(py * pk + ky)) * p->size + (size_t)(px * pk + kx)];
        }
    {
        const layer_t *L = p->patch_embed;
        quantize_f32(m->cols, (size_t)NP * ps, L->s_x, 8, m->a8);
        if (patch_fpga(m, L, NP) != 0)
            gemm_s8(m->a8, L->w, m->acc, NP, L->out, L->in);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < NP; t++)
            for (int c = 0; c < D; c++) {
                int64_t a = m->acc[(size_t)t * D + c];
                if (L->bias)
                    a += L->bias[c];
                m->patch_out[(size_t)t * D + c] = (float)((double)a * L->deq[c]);
            }
        if (dbg)
            dbg(dbg_ctx, "patch_embed", m->patch_out, (size_t)NP * D, -4);
        for (int c = 0; c < D; c++)
            m->x[c] = p->cls[c] + p->pos[c];
        for (int t = 0; t < NP; t++)
            for (int c = 0; c < D; c++)
                m->x[(size_t)(t + 1) * D + c] = m->patch_out[(size_t)t * D + c] + p->pos[(size_t)(t + 1) * D + c];
    }

    m->st.t_patch += now_s() - t_patch0;
    if (g_tr)
        tr_add("cpu", "patch_embed", 0, t_patch0, now_s(), 0);

    /* ---- transformer blocks ---- */
    const int nb = (m->max_blocks > 0 && m->max_blocks < p->n_blocks) ? m->max_blocks : p->n_blocks;
    /* with the pipeline, the stretch between two attentions runs in row chunks (see pl_run). The stage
     * observer (dbg) then only receives patch_embed and the contexts: the per-layer matrices do not exist
     * as a whole at one moment. Callers that need them call model_set_pipeline(m, 0) first. */
    const int pipe = nb > 0 && pl_prepare(m);
    if (pipe) {
        pstep_t s[16];
        int ns = 0;
        pl_block_head(m, s, &ns, &p->blocks[0], 0);
        if (pl_run(m, s, ns))
            return -1;
    }
    for (int bi = 0; bi < nb; bi++) {
        const block_t *B = &p->blocks[bi];
        const int A = B->q->out, H = B->heads, hd = A / H, N = B->neurons;
        g_tr_block = bi;

        if (!pipe) {
        TIMED(t_ln, layer_norm(m->x, T, D, B->n1_g, B->n1_b, p->ln_eps, m->n));
        if (linear_int(m, B->q, m->n, T, m->q16, 0)) return -1;
        if (linear_int(m, B->k, m->n, T, m->k16, B->k->s_x == B->q->s_x)) return -1;
        if (linear_int(m, B->v, m->n, T, m->v16, B->v->s_x == B->k->s_x)) return -1;
        dbg16(dbg, dbg_ctx, bi, "q_w", m->q16, (size_t)T * A);
        dbg16(dbg, dbg_ctx, bi, "k_w", m->k16, (size_t)T * A);
        dbg16(dbg, dbg_ctx, bi, "v_w", m->v16, (size_t)T * A);

        TIMED(t_rope, rope_int(m->q16, T, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, m->q8));
        TIMED(t_rope, rope_int(m->k16, T, A, H, hd, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre, m->k8));
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * A; i++)
            m->v8[i] = (int8_t)m->v16[i];
        }

        if (m->fpga_attn) {
            TIMED(t_attn, if (attention_fpga_select(m, B, T, A, H, hd)) return -1);
        } else {
            TIMED(t_attn, attention_int(p, B, m->q8, m->k8, m->v8, T, A, H, hd, m->ctx8));
        }
        if (dbg) {
            char nm[48];
            snprintf(nm, sizeof nm, "b%d.ctx_i", bi);
            dbg(dbg_ctx, nm, m->ctx8, (size_t)T * A, 1);
        }
        if (pipe) {
            if (pl_after_attention(m, bi, nb))
                return -1;
            continue;
        }
        /* proj quantises ctx * s_ctx with its own input scale. When both scales are the same, the
         * float round trip returns the INT8 value unchanged (|c| <= 127, relative error ~1e-5 of one
         * step, far from a rounding boundary), so ctx8 is used directly. Otherwise do it the long way. */
        const int ctx_direct = B->proj->in_bits == 8 && B->proj->in == A &&
                               fabs(B->s_ctx - B->proj->s_x) <= 1e-9 * fabs(B->proj->s_x);
        if (ctx_direct) {
            TIMED(t_quant, memcpy(m->a8, m->ctx8, (size_t)T * A));
            m->a_ok = 0;
            if (linear_int(m, B->proj, NULL, T, m->y16, 1)) return -1;
        } else {
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < (ptrdiff_t)T * A; i++)
                m->ctxf[i] = (float)((double)m->ctx8[i] * B->s_ctx);
            if (linear_int(m, B->proj, m->ctxf, T, m->y16, 0)) return -1;
        }
        dbg16(dbg, dbg_ctx, bi, "proj_w", m->y16, (size_t)T * D);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * D; i++)
            m->x[i] += (float)((double)m->y16[i] * B->s_proj);

        TIMED(t_ln, layer_norm(m->x, T, D, B->n2_g, B->n2_b, p->ln_eps, m->n));
        if (linear_int(m, B->fc1g, m->n, T, m->g16, 0)) return -1;
        if (linear_int(m, B->fc1x, m->n, T, m->u16, B->fc1x->s_x == B->fc1g->s_x)) return -1;
        dbg16(dbg, dbg_ctx, bi, "fc1g_w", m->g16, (size_t)T * N);
        dbg16(dbg, dbg_ctx, bi, "fc1x_w", m->u16, (size_t)T * N);

        const double t_sw0 = now_s();
        const int64_t sw_rnd = (int64_t)1 << (B->sw_sh - 1);
        const int qm_in = qmax_of(p->silu_in_bits);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * N; i++) {
            int64_t y = (int64_t)B->silu[m->g16[i] + qm_in] * m->u16[i];
            y = (y * B->sw_m0 + sw_rnd) >> B->sw_sh;
            m->h16[i] = (int16_t)(y > 32767 ? 32767 : y < -32767 ? -32767 : y);
            m->hh[i] = (float)((double)m->h16[i] * B->s_h);
        }
        m->st.t_swiglu += now_s() - t_sw0;
        dbg16(dbg, dbg_ctx, bi, "swiglu_i", m->h16, (size_t)T * N);

        TIMED(t_ln, layer_norm(m->hh, T, N, B->mlpn_g, B->mlpn_b, p->ln_eps, m->n));
        if (linear_int(m, B->fc2, m->n, T, m->y16, 0)) return -1;
        dbg16(dbg, dbg_ctx, bi, "fc2_w", m->y16, (size_t)T * D);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)T * D; i++)
            m->x[i] += (float)((double)m->y16[i] * B->s_fc2);
    }

    /* ---- pooling, final LayerNorm, head ---- */
    const double t_head0 = now_s();
    g_tr_block = nb;
    float pooled[1024], pn[1024];
    for (int c = 0; c < D; c++) {
        double s = 0.0;
        for (int t = 1; t < T; t++)
            s += m->x[(size_t)t * D + c];
        pooled[c] = (float)(s / (T - 1));
    }
    TIMED(t_ln, layer_norm(pooled, 1, D, p->fcn_g, p->fcn_b, p->ln_eps, pn));
    {
        const layer_t *L = p->head;
        int8_t q[1024];
        int32_t acc[16];
        quantize_f32(pn, (size_t)D, L->s_x, 8, q);
        gemm_s8(q, L->w, acc, 1, L->out, L->in);
        for (int c = 0; c < L->out; c++) {
            int64_t a = acc[c];
            if (L->bias)
                a += L->bias[c];
            logits[c] = (float)((double)a * L->deq[c]);
        }
    }
    if (dbg)
        dbg(dbg_ctx, "logits", logits, 4, -4);
    if (g_tr) {
        tr_add("cpu", "head", 0, t_head0, now_s(), 0);
        tr_end();
    }
    return 0;
}

/* ---- attention on the FPGA: per head Q.K^T (RAW32) -> softmax unit -> P.V, S and P stay in the DDR3 ---- */

/* --check, head 0 only: the scores and probabilities of rows 0..15 and of the last row against the CPU */
static size_t att_check_stages(model_t *m, const block_t *B, int T, int A, int NK, int KP, int h, int hd, uint32_t s_off,
                               uint32_t p_off)
{
    const uint8_t *reg = m->hw->region;
    size_t bad_s = 0, bad_p = 0;
    int32_t *S = malloc((size_t)T * sizeof *S);
    uint16_t *P = malloc((size_t)T * sizeof *P);
    for (int n = 0; n < 17; n++) {
        const int r = n < 16 ? n : T - 1;
        const int8_t *qi = m->q8 + (size_t)r * A + (size_t)h * hd;
        for (int j = 0; j < T; j++) {
            const int8_t *kj = m->k8 + (size_t)j * A + (size_t)h * hd;
            int32_t s = 0;
            for (int d = 0; d < hd; d++)
                s += (int32_t)qi[d] * (int32_t)kj[d];
            S[j] = s;
        }
        const int32_t *hw_s = (const int32_t *)(reg + s_off + (size_t)r * NK * 4);
        for (int j = 0; j < T; j++)
            bad_s += hw_s[j] != S[j];
        softmax_row_int(m->p, B, S, T, P);
        const int sub = r / 16, i = r % 16;
        for (int k = 0; k < KP; k++) {
            const uint8_t hi = reg[p_off + (size_t)sub * 2 * KP * 16 + (size_t)k * 16 + i];
            const uint8_t lo = reg[p_off + (size_t)sub * 2 * KP * 16 + (size_t)KP * 16 + (size_t)k * 16 + i];
            const uint16_t got = (uint16_t)((hi << 8) | lo);
            bad_p += got != (k < T ? P[k] : 0);
        }
    }
    free(S);
    free(P);
    if (m->verbose || bad_s || bad_p)
        fprintf(stderr, "  fpga attention head %d: scores of 17 rows %s (%zu differ), probabilities %s (%zu differ)\n", h,
                bad_s ? "MISMATCH" : "identical to the CPU", bad_s, bad_p ? "MISMATCH" : "identical to the CPU", bad_p);
    return bad_s + bad_p;
}

static int attention_fpga(model_t *m, const block_t *B, int T, int A, int H, int hd)
{
    accel_hw_t *hw = m->hw;
    const int NK = ((T + 15) / 16) * 16;            /* keys padded to whole 16-key tiles */
    const int KP = (T + 1) & ~1;                    /* k extent of P and V: even */
    const int subs = (T + 15) / 16;
    const int spp_s = subs < 255 ? subs : 255;
    const int spp_pv = 16384 / (2 * KP) < 255 ? 16384 / (2 * KP) : 255;
    accel_sm_sim_ctx_t sctx = {.pkg = m->p, .block = B};     /* used by the software model only */
    for (int h = 0; h < H; h++) {
        double t0 = now_s();
        size_t n;
        n = fpga_acts_bytes(T, hd, 1);
        fpga_pack_acts8_strided(m->q8 + (size_t)h * hd, T, hd, A, m->att_stage);
        memcpy(hw->region + ATT_Q, m->att_stage, n);
        accel_hw_clean(hw, ATT_Q, n);
        n = fpga_kt_bytes(T, hd);
        fpga_pack_kt(m->k8 + (size_t)h * hd, T, A, hd, m->att_stage);
        memcpy(hw->region + ATT_K, m->att_stage, n);
        accel_hw_clean(hw, ATT_K, n);
        n = fpga_v_bytes(hd, KP);
        fpga_pack_v(m->v8 + (size_t)h * hd, T, A, hd, KP, (int32_t)B->av_m0, (int)B->av_sh, m->att_stage);
        memcpy(hw->region + ATT_V, m->att_stage, n);
        accel_hw_clean(hw, ATT_V, n);
        double t1 = now_s();
        if (g_tr)
            tr_add("cpu", "att_pack", h, t0, t1, 0);

        accel_stat_t st;
        accel_op_t s_op = {.M = (uint32_t)T, .N = (uint32_t)NK, .K = (uint32_t)hd, .in_bits = 8, .out_bits = 32,
                           .spp = spp_s, .a_phys = ACCEL_REGION_PHYS + ATT_Q, .w_phys = ACCEL_REGION_PHYS + ATT_K,
                           .out_phys = ACCEL_REGION_PHYS + ATT_S, .raw32 = 1, .stride_out = (uint32_t)NK * 4};
        if (accel_hw_run(hw, &s_op, &st, 60.0) != 0)
            return -1;
        m->st.att_cyc_s += st.cycles;
        double t2 = now_s();
        if (g_tr)
            tr_add("eng", "att_s", h, t1, t2, st.cycles);

        accel_sm_op_t sm_op = {.rows = (uint32_t)T, .cols = (uint32_t)T, .kp = (uint32_t)KP,
                               .s_phys = ACCEL_REGION_PHYS + ATT_S, .s_stride = (uint32_t)NK * 4,
                               .p_phys = ACCEL_REGION_PHYS + ATT_P, .m0 = (uint32_t)B->sm_m0, .sh = (int)B->sm_sh,
                               .sim_ctx = &sctx};
        if (accel_hw_run_sm(hw, &sm_op, &st, 60.0) != 0)
            return -1;
        m->st.att_cyc_sm += st.cycles;
        double t3 = now_s();
        if (g_tr)
            tr_add("sm", "softmax", h, t2, t3, st.cycles);

        if (m->check && h == 0)
            m->st.att_mismatches += att_check_stages(m, B, T, A, NK, KP, h, hd, ATT_S, ATT_P);

        accel_op_t pv_op = {.M = (uint32_t)T, .N = (uint32_t)hd, .K = (uint32_t)KP, .in_bits = 16, .out_bits = 8,
                            .spp = spp_pv, .a_phys = ACCEL_REGION_PHYS + ATT_P, .w_phys = ACCEL_REGION_PHYS + ATT_V,
                            .out_phys = ACCEL_REGION_PHYS + ATT_C + (uint32_t)h * hd, .in_unsigned = 1,
                            .stride_out = (uint32_t)A};
        if (accel_hw_run(hw, &pv_op, &st, 60.0) != 0)
            return -1;
        m->st.att_cyc_pv += st.cycles;
        double t4 = now_s();
        if (g_tr)
            tr_add("eng", "att_pv", h, t3, t4, st.cycles);
        m->st.t_att_pack += t1 - t0;
        m->st.t_att_s += t2 - t1;
        m->st.t_att_sm += t3 - t2;
        m->st.t_att_pv += t4 - t3;
    }
    double t5 = now_s();
    accel_hw_inval(hw, ATT_C, (size_t)T * A);
    memcpy(m->ctx8, hw->region + ATT_C, (size_t)T * A);
    m->st.t_att_read += now_s() - t5;

    if (m->check) {
        const pkg_t *p = m->p;
        attention_int(p, B, m->q8, m->k8, m->v8, T, A, H, hd, m->att_chk);
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)T * A; i++)
            bad += m->att_chk[i] != m->ctx8[i];
        m->st.att_mismatches += bad;
        if (m->verbose || bad)
            fprintf(stderr, "  fpga attention, whole block: context %s (%zu of %zu differ)\n",
                    bad ? "MISMATCH" : "identical to the CPU", bad, (size_t)T * A);
    }
    return 0;
}


/* ---- attention with the heads overlapped: engine (Q.K^T and P.V), softmax unit and ARM packing at the same time ----
 * Two sets of buffers (parity h % 2) so the packing and the scores of head h+1 can proceed while head h is still
 * in the softmax or in P.V. Engine queue: S0, S1, PV0, S2, PV1, ..., S(H-1), PV(H-2), PV(H-1); the softmax runs in
 * head order. Dependencies (the set of parity p is reused every two heads):
 *   pack(h) needs PV(h-2) done;  S(h) needs pack(h) and SM(h-2) done;
 *   SM(h) needs S(h) and PV(h-2) done;  PV(h) needs SM(h) done.
 * full = 0 (pack-only): the softmax and the engine are never in flight together. */

#define ATT_MAX_HEADS 32

typedef struct { int pv, h; } att_job_t;      /* pv: 0 = scores Q.K^T, 1 = context P.V */

static int attention_fpga_ovl(model_t *m, const block_t *B, int T, int A, int H, int hd, int full)
{
    accel_hw_t *hw = m->hw;
    const int NK = ((T + 15) / 16) * 16;
    const int KP = (T + 1) & ~1;
    const int subs = (T + 15) / 16;
    const int spp_s = subs < 255 ? subs : 255;
    const int spp_pv = 16384 / (2 * KP) < 255 ? 16384 / (2 * KP) : 255;
    if (H > ATT_MAX_HEADS) {
        fprintf(stderr, "model: %d heads, the overlapped attention supports %d\n", H, ATT_MAX_HEADS);
        return -1;
    }
    accel_sm_sim_ctx_t sctx = {.pkg = m->p, .block = B};
    accel_op_t s_op[ATT_MAX_HEADS], pv_op[ATT_MAX_HEADS];
    accel_sm_op_t sm_op[ATT_MAX_HEADS];
    for (int h = 0; h < H; h++) {
        const int p = h & 1;
        s_op[h] = (accel_op_t){.M = (uint32_t)T, .N = (uint32_t)NK, .K = (uint32_t)hd, .in_bits = 8, .out_bits = 32,
                               .spp = spp_s, .a_phys = ACCEL_REGION_PHYS + ATT2_Q(p),
                               .w_phys = ACCEL_REGION_PHYS + ATT2_K(p), .out_phys = ACCEL_REGION_PHYS + ATT2_S(p),
                               .raw32 = 1, .stride_out = (uint32_t)NK * 4};
        sm_op[h] = (accel_sm_op_t){.rows = (uint32_t)T, .cols = (uint32_t)T, .kp = (uint32_t)KP,
                                   .s_phys = ACCEL_REGION_PHYS + ATT2_S(p), .s_stride = (uint32_t)NK * 4,
                                   .p_phys = ACCEL_REGION_PHYS + ATT2_P(p), .m0 = (uint32_t)B->sm_m0,
                                   .sh = (int)B->sm_sh, .sim_ctx = &sctx};
        pv_op[h] = (accel_op_t){.M = (uint32_t)T, .N = (uint32_t)hd, .K = (uint32_t)KP, .in_bits = 16, .out_bits = 8,
                                .spp = spp_pv, .a_phys = ACCEL_REGION_PHYS + ATT2_P(p),
                                .w_phys = ACCEL_REGION_PHYS + ATT2_V(p),
                                .out_phys = ACCEL_REGION_PHYS + ATT_C + (uint32_t)h * hd, .in_unsigned = 1,
                                .stride_out = (uint32_t)A};
    }
    att_job_t jobs[2 * ATT_MAX_HEADS];
    int nj = 0;
    jobs[nj++] = (att_job_t){0, 0};
    for (int h = 1; h < H; h++) {
        jobs[nj++] = (att_job_t){0, h};
        jobs[nj++] = (att_job_t){1, h - 1};
    }
    jobs[nj++] = (att_job_t){1, H - 1};

    int packed[ATT_MAX_HEADS] = {0}, s_done[ATT_MAX_HEADS] = {0}, sm_done[ATT_MAX_HEADS] = {0}, pv_done[ATT_MAX_HEADS] = {0};
    int eng_job = -1, sm_h = -1;        /* in flight: index into jobs / head of the softmax */
    int eng_next = 0, sm_next = 0, pack_next = 0, n_pv = 0, rc = -1;
    double eng_t0 = 0, sm_t0 = 0;
    accel_stat_t st;

    while (n_pv < H) {
        int progress = 0;

        /* collect what finished */
        if (eng_job >= 0) {
            const att_job_t *j = &jobs[eng_job];
            const accel_op_t *op = j->pv ? &pv_op[j->h] : &s_op[j->h];
            const int busy = accel_hw_busy(hw);
            const double now = now_s();
            if (!busy || now - eng_t0 > 60.0) {
                eng_job = -1;
                if (accel_hw_wait(hw, op, &st, busy ? 0.0 : 60.0) != 0)
                    goto fail;
                const double t1 = now_s();
                if (j->pv) {
                    pv_done[j->h] = 1;
                    n_pv++;
                    m->st.att_cyc_pv += st.cycles;
                    m->st.t_att_pv += t1 - eng_t0;
                } else {
                    s_done[j->h] = 1;
                    m->st.att_cyc_s += st.cycles;
                    m->st.t_att_s += t1 - eng_t0;
                }
                if (g_tr)
                    tr_add("eng", j->pv ? "att_pv" : "att_s", j->h, eng_t0, t1, st.cycles);
                progress = 1;
            }
        }
        if (sm_h >= 0) {
            const int busy = accel_hw_sm_busy(hw);
            if (!busy || now_s() - sm_t0 > 60.0) {
                const int h = sm_h;
                sm_h = -1;
                if (accel_hw_sm_wait(hw, &sm_op[h], &st, busy ? 0.0 : 60.0) != 0)
                    goto fail;
                const double t1 = now_s();
                sm_done[h] = 1;
                m->st.att_cyc_sm += st.cycles;
                m->st.t_att_sm += t1 - sm_t0;
                if (g_tr)
                    tr_add("sm", "softmax", h, sm_t0, t1, st.cycles);
                /* head 0: S and P of set 0 are intact now (S(2) and SM(2) cannot have started yet) */
                if (m->check && h == 0)
                    m->st.att_mismatches += att_check_stages(m, B, T, A, NK, KP, 0, hd, ATT2_S(0), ATT2_P(0));
                progress = 1;
            }
        }

        /* launch the next engine job */
        if (eng_job < 0 && eng_next < nj && (full || sm_h < 0)) {
            const int pv = jobs[eng_next].pv, h = jobs[eng_next].h;
            int ok;
            if (pv)
                ok = sm_done[h];
            else
                ok = packed[h] && (h < 2 || sm_done[h - 2]);
            if (ok) {
                if (accel_hw_start(hw, pv ? &pv_op[h] : &s_op[h]) != 0)
                    goto fail;
                eng_t0 = now_s();
                eng_job = eng_next++;
                progress = 1;
            }
        }
        /* launch the next softmax */
        if (sm_h < 0 && sm_next < H && (full || eng_job < 0)) {
            const int h = sm_next;
            if (s_done[h] && (h < 2 || pv_done[h - 2])) {
                if (accel_hw_sm_start(hw, &sm_op[h]) != 0)
                    goto fail;
                sm_t0 = now_s();
                sm_h = h;
                sm_next++;
                progress = 1;
            }
        }

        /* nothing to collect or launch: pack the next head if its buffers are free */
        if (!progress && pack_next < H && (pack_next < 2 || pv_done[pack_next - 2])) {
            const int h = pack_next, p = h & 1;
            const double t0 = now_s();
            size_t n;
            n = fpga_acts_bytes(T, hd, 1);
            fpga_pack_acts8_strided(m->q8 + (size_t)h * hd, T, hd, A, m->att_stage);
            memcpy(hw->region + ATT2_Q(p), m->att_stage, n);
            accel_hw_clean(hw, ATT2_Q(p), n);
            n = fpga_kt_bytes(T, hd);
            fpga_pack_kt(m->k8 + (size_t)h * hd, T, A, hd, m->att_stage);
            memcpy(hw->region + ATT2_K(p), m->att_stage, n);
            accel_hw_clean(hw, ATT2_K(p), n);
            n = fpga_v_bytes(hd, KP);
            fpga_pack_v(m->v8 + (size_t)h * hd, T, A, hd, KP, (int32_t)B->av_m0, (int)B->av_sh, m->att_stage);
            memcpy(hw->region + ATT2_V(p), m->att_stage, n);
            accel_hw_clean(hw, ATT2_V(p), n);
            const double t1 = now_s();
            m->st.t_att_pack += t1 - t0;
            if (g_tr)
                tr_add("cpu", "att_pack", h, t0, t1, 0);
            packed[h] = 1;
            pack_next++;
        }
    }
    rc = 0;
fail:
    if (rc != 0) {
        /* leave both units idle (a wait on a busy unit resets it) */
        if (eng_job >= 0) {
            const att_job_t *j = &jobs[eng_job];
            accel_hw_wait(hw, j->pv ? &pv_op[j->h] : &s_op[j->h], NULL, 0.0);
        }
        if (sm_h >= 0)
            accel_hw_sm_wait(hw, &sm_op[sm_h], NULL, 0.0);
        return -1;
    }

    double t5 = now_s();
    accel_hw_inval(hw, ATT_C, (size_t)T * A);
    memcpy(m->ctx8, hw->region + ATT_C, (size_t)T * A);
    m->st.t_att_read += now_s() - t5;

    if (m->check) {
        const pkg_t *p = m->p;
        attention_int(p, B, m->q8, m->k8, m->v8, T, A, H, hd, m->att_chk);
        size_t bad = 0;
        for (size_t i = 0; i < (size_t)T * A; i++)
            bad += m->att_chk[i] != m->ctx8[i];
        m->st.att_mismatches += bad;
        if (m->verbose || bad)
            fprintf(stderr, "  fpga attention (overlapped), whole block: context %s (%zu of %zu differ)\n",
                    bad ? "MISMATCH" : "identical to the CPU", bad, (size_t)T * A);
    }
    return 0;
}

/* choose the attention variant for this bitstream and run it */
static int attention_fpga_select(model_t *m, const block_t *B, int T, int A, int H, int hd)
{
    int mode = m->att_overlap;
    if (mode == ATT_OVL_OFF)
        return attention_fpga(m, B, T, A, H, hd);
    if (mode == ATT_OVL_AUTO) {
        if (m->hw->hw_version >= ACCEL_HW_VERSION_H5B)
            mode = ATT_OVL_FORCE;
        else {
            if (!m->att_warned)
                fprintf(stderr, "model: engine VERSION %d has no output FIFO credit: softmax and engine are not "
                                "overlapped (packing only)\n", m->hw->hw_version);
            m->att_warned = 1;
            mode = ATT_OVL_PACK_ONLY;
        }
    }
    return attention_fpga_ovl(m, B, T, A, H, hd, mode == ATT_OVL_FORCE);
}

int model_attach_fpga(model_t *m, accel_hw_t *hw, int check, int verbose, int fpga_attn)
{
    const pkg_t *p = m->p;
    if (hw->region_len < FPGA_REGION_LEN) {
        fprintf(stderr, "model: the mapped region (%zu bytes) is smaller than the %u needed\n", hw->region_len,
                FPGA_REGION_LEN);
        return -1;
    }
    size_t max_stage = 0, max_out = 0;
    uint32_t off = FPGA_W_OFF;
    double t0 = now_s();
    int n_fpga = 0, n_sw = 0;
    uint8_t *tile = NULL;
    size_t tile_len = 0;
    /* fused SwiGLU: the fc1 gate and up weights of an eligible block are packed interleaved, and only that way (the
     * region has no room for a second copy), so these two layers cannot run on the engine on their own: the paths that
     * do not use the pipeline (and --check, which turns the fusion off here) never get this */
    uint8_t skip[PKG_MAX_LAYERS] = {0};
    memset(m->sw_off, 0, sizeof m->sw_off);
    if (m->want_hw_swiglu && (hw->features & ACCEL_FEAT_SWIGLU) && !check && p->silu_in_bits == 8)
        for (int bi = 0; bi < p->n_blocks; bi++) {
            const block_t *B = &p->blocks[bi];
            const layer_t *G = B->fc1g, *U = B->fc1x;
            if (fpga_supports(G) && fpga_supports(U) && G->out == U->out && G->in == U->in && G->s_x == U->s_x &&
                G->in_bits == U->in_bits && G->out_bits == 8 && U->out_bits == 8 && G->out == B->neurons &&
                B->sw_m0 > 0 && B->sw_m0 < ((int64_t)1 << 31) && B->sw_sh >= 1 && B->sw_sh <= 63)
                skip[G - p->layers] = skip[U - p->layers] = 1;
        }
    for (int i = 0; i < p->n_layers; i++) {
        const layer_t *L = &p->layers[i];
        if (!fpga_supports(L))
            continue;
        const size_t a = fpga_acts_bytes(m->tokens, L->in, L->in_bits > 8 ? 2 : 1);
        const size_t o = (size_t)m->tokens * L->out * (L->out_bits > 8 ? 2 : 1);
        if (a > max_stage) max_stage = a;
        if (o > max_out) max_out = o;
        if (skip[i])
            continue;
        const size_t bytes = fpga_weights_bytes(L);
        if (off + bytes > ATT_OFF) {
            fprintf(stderr, "model: the packed weights do not fit the reserved region\n");
            free(tile);
            return -1;
        }
        if (bytes > tile_len) {
            free(tile);
            tile = malloc(bytes);
            tile_len = bytes;
        }
        fpga_pack_weights(L, tile);
        memcpy(hw->region + off, tile, bytes);
        accel_hw_clean(hw, off, bytes);
        m->w_off[i] = off;
        off = (off + (uint32_t)bytes + 63u) & ~63u;
        n_fpga++;
    }
    for (int bi = 0; bi < p->n_blocks; bi++) {
        const block_t *B = &p->blocks[bi];
        if (!skip[B->fc1g - p->layers])
            continue;
        const size_t bytes = fpga_weights_pair_bytes(B->fc1g, B->fc1x);
        if (off + bytes > ATT_OFF) {
            fprintf(stderr, "model: the packed weights do not fit the reserved region\n");
            free(tile);
            return -1;
        }
        if (bytes > tile_len) {
            free(tile);
            tile = malloc(bytes);
            tile_len = bytes;
        }
        fpga_pack_weights_pair(B->fc1g, B->fc1x, tile);
        memcpy(hw->region + off, tile, bytes);
        accel_hw_clean(hw, off, bytes);
        m->sw_off[bi] = off;
        off = (off + (uint32_t)bytes + 63u) & ~63u;
        n_sw++;
    }
    free(tile);
    if (max_stage > FPGA_O_OFF - FPGA_A_OFF || max_out > FPGA_W_OFF - FPGA_O_OFF) {
        fprintf(stderr, "model: activation or output buffers do not fit their region slots\n");
        return -1;
    }
    m->stage = malloc(max_stage);
    m->stage_len = max_stage;
    m->rdbuf = malloc(max_out);
    m->chk16 = malloc((size_t)m->tokens * (m->max_neurons > m->dim ? m->max_neurons : m->dim) * sizeof(int16_t));
    m->hw = hw;
    m->check = check;
    m->verbose = verbose;
    m->a_ok = 0;
    m->chunk_rows = 192;
    m->pipeline = 1;
    m->mlp_fuse = 1;
    m->rlp_fuse = 1;
    /* RoPE in the engine's epilogue (T41): q and k leave the engine rotated. Not with --check, which compares q / k
     * before the rotation with the CPU. The bitstream must announce it and the tables must be separable. */
    m->hw_rope = 0;
    if (m->want_hw_rope && (hw->features & ACCEL_FEAT_ROPE) && !check && m->tokens <= 1025) {
        int ok = 1;
        for (int bi = 0; bi < p->n_blocks; bi++) {
            const block_t *B = &p->blocks[bi];
            ok &= B->heads > 0 && B->q->out == B->k->out && B->q->out / B->heads == 64 && B->q->out_bits == 16 &&
                  B->k->out_bits == 16 && B->q->s_x == B->k->s_x;
        }
        if (ok && accel_hw_load_rope(hw, p->rope_cos, p->rope_sin, p->rope_frac, p->rope_pre) == 0)
            m->hw_rope = 1;
    }
    if (fpga_attn) {
        if (!hw->has_softmax) {
            fprintf(stderr, "model: --fpga-attn needs a bitstream with the softmax unit (registers at 0x100)\n");
            return -1;
        }
        if (ATT2_END > FPGA_REGION_LEN || ATT_END > ATT2_OFF || m->tokens > 1040) {
            fprintf(stderr, "model: attention buffers do not fit the region\n");
            return -1;
        }
        const int hd = p->blocks[0].q->out / p->blocks[0].heads;
        size_t st = fpga_acts_bytes(m->tokens, hd, 1), k = fpga_kt_bytes(m->tokens, hd), v = fpga_v_bytes(hd, (m->tokens + 1) & ~1);
        m->att_stage_len = st > k ? (st > v ? st : v) : (k > v ? k : v);
        m->att_stage = malloc(m->att_stage_len);
        m->att_chk = malloc((size_t)m->tokens * p->blocks[0].q->out);
        m->fpga_attn = 1;
    }
    fprintf(stderr, "model: %d layers packed into the FPGA region (%.1f MiB of weights) in %.1f s\n", n_fpga,
            (off - FPGA_W_OFF) / 1048576.0, now_s() - t0);
    if (n_sw || m->hw_rope)
        fprintf(stderr, "model: engine features 0x%x: %s%s%s\n", hw->features, m->hw_rope ? "RoPE in the epilogue" : "",
                m->hw_rope && n_sw ? ", " : "", n_sw ? "fused SwiGLU on fc1 (weights interleaved)" : "");
    else if (m->want_hw_rope || m->want_hw_swiglu)
        fprintf(stderr, "model: engine features 0x%x: RoPE and SwiGLU stay on the ARM\n", hw->features);
    return 0;
}

void model_limit_blocks(model_t *m, int n) { m->max_blocks = n; }
void model_set_chunk(model_t *m, int rows) { m->chunk_rows = rows <= 0 ? 0 : (rows + 15) & ~15; }
void model_set_pipeline(model_t *m, int on) { m->pipeline = on; }
void model_set_mlp_fuse(model_t *m, int on) { m->mlp_fuse = on; }
void model_set_hw_fusion(model_t *m, int rope, int swiglu)
{
    m->want_hw_rope = rope;
    m->want_hw_swiglu = swiglu;
}
void model_set_rlp_fuse(model_t *m, int on) { m->rlp_fuse = on; }
const model_stats_t *model_stats(const model_t *m) { return &m->st; }
void model_stats_reset(model_t *m) { memset(&m->st, 0, sizeof m->st); }

float model_score(const float logits[4])
{
    double mx = logits[0];
    for (int i = 1; i < 4; i++)
        if (logits[i] > mx)
            mx = logits[i];
    double e[4], s = 0.0;
    for (int i = 0; i < 4; i++) {
        e[i] = exp((double)logits[i] - mx);
        s += e[i];
    }
    return (float)((e[1] + e[2] + e[3]) / s);
}

void model_prepare_rgb8(const uint8_t *rgb, int size, float *out)
{
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    const size_t plane = (size_t)size * size;
    for (size_t i = 0; i < plane; i++)
        for (int c = 0; c < 3; c++) {
            float v = (float)rgb[i * 3 + c] / 255.0f;
            out[c * plane + i] = (v - mean[c]) / stdv[c];
        }
}
