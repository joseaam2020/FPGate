#ifndef FPGATE_MODEL_H
#define FPGATE_MODEL_H
/*
 * Full model forward pass (CPU backend): image -> logits, following export_check.run().
 * Everything is computed on the HPS; the FPGA backend will replace the GEMM / attention
 * kernels in ops.c and keep this control flow.
 */
#include <stddef.h>

#include "pkg.h"

/* optional observer for intermediate results (tests); elem_bytes is 1, 2 or 4 (signed ints) or -4 for float32 */
typedef void (*model_dbg_fn)(void *ctx, const char *name, const void *data, size_t count, int elem_bytes);

typedef struct model model_t;

/* counters of the FPGA backend since the last reset */
typedef struct {
    int ops;                    /* operations run on the FPGA */
    unsigned long long cycles;  /* sum of the engine's CYCLES counters */
    double t_pack, t_run, t_read;   /* HPS time packing inputs, waiting for the engine, reading outputs */
    double t_lin_wall;              /* wall time of the overlapped linear layers (pack + wait + read overlap) */
    /* row-chunk pipeline between attentions: wall time, ARM busy time by kind of work (they add up to more
     * than the wall time when the ARM and the engine overlap), ARM time spent waiting for the engine */
    double pl_wall, pl_ln, pl_quant, pl_pack, pl_read, pl_swiglu, pl_rope, pl_other, pl_idle;
    unsigned long long mismatches;  /* outputs that differ from the CPU result (check mode) */
    /* HPS time by stage, seconds (CPU work that stays on the ARM) */
    double t_patch, t_ln, t_rope, t_attn, t_swiglu, t_quant;
    /* attention on the FPGA: engine cycles of the three steps, HPS time packing Q/K/V, waiting, reading the context */
    unsigned long long att_cyc_s, att_cyc_sm, att_cyc_pv, att_mismatches;
    double t_att_pack, t_att_s, t_att_sm, t_att_pv, t_att_read;
} model_stats_t;

model_t *model_create(const pkg_t *p);
void model_destroy(model_t *m);

struct accel_hw;
/* fpga_attn: also run the attention (Q.K^T, softmax, P.V) on the FPGA; needs the softmax unit in the bitstream.
 * Run the linear layers on the FPGA: packs all weights into the reserved DDR3 region (once) and sends
 * every INT GEMM that the engine supports to it. check: also compute each layer on the CPU and count
 * differences. Returns 0 on success. patch_embed runs on the engine too (see model_set_cpu_patch); the head
 * stays on the CPU. */
int model_attach_fpga(model_t *m, struct accel_hw *hw, int check, int verbose, int fpga_attn);
/* rows per engine operation for the overlapped linear layers (rounded up to a multiple of 16);
 * 0 = no overlap: the ARM waits for every operation (the behaviour before overlapping). Default 256. */
void model_set_chunk(model_t *m, int rows);
/* 1 (default when the FPGA is attached): between two attentions the rows go through LayerNorm, the linear
 * layers, RoPE, SwiGLU and the residual sums in chunks, so the ARM works on one chunk while the engine
 * works on another. 0: every layer runs on all rows before the next starts (the behaviour before the
 * pipeline). Needs model_set_chunk > 0; falls back by itself when something does not fit. */
void model_set_pipeline(model_t *m, int on);
/* inside the pipeline, fuse the MLP tail (SwiGLU, LayerNorm, fc2 packing) by 16-row groups; on by default */
void model_set_mlp_fuse(model_t *m, int on);
void model_set_rlp_fuse(model_t *m, int on);
/* Use the engine's RoPE epilogue (q and k come out rotated, INT8) and the SwiGLU fused on fc1 when the bitstream announces
 * them in FEATURES (ACCEL_FEAT_*); otherwise (an h6 bitstream reads 0xDEADBEEF there) nothing changes. Call it BEFORE
 * model_attach_fpga: the fused fc1 needs its gate and up weights interleaved at packing time, in place of the separate
 * ones, so the fused blocks run fc1 on the CPU if the pipeline is not used. Both are off by default and always off with
 * check. */
void model_set_hw_fusion(model_t *m, int rope, int swiglu);
/* How the FPGA attention overlaps the heads (set before the first forward). ATT_OVL_OFF (--no-att-overlap): the
 * sequential attention_fpga. ATT_OVL_AUTO (default): full overlap (softmax, engine and packing at the same time)
 * only when the engine VERSION is the h5b one (ACCEL_HW_VERSION_H5B), else pack-only with a warning.
 * ATT_OVL_PACK_ONLY (--att-overlap-pack-only): packing overlaps, but the softmax and the engine never run at the
 * same time. ATT_OVL_FORCE (--att-overlap-force): full overlap whatever the bitstream (tests only). */
enum { ATT_OVL_AUTO = 0, ATT_OVL_OFF, ATT_OVL_PACK_ONLY, ATT_OVL_FORCE };
void model_set_att_overlap(model_t *m, int mode);
/* 1 (--cpu-patch): keep the patch embedding GEMM on the CPU; default 0 runs it on the engine (RAW32 output). */
void model_set_cpu_patch(model_t *m, int on);
/* --trace: record per-image events (engine, ARM steps, idle, softmax, packing) and write them as CSV
 * t0,t1,kind,name,block,idx,cycles after each model_forward (path, path.1, path.2, ...). 0 on success. */
int model_trace_open(const char *path);
/* run only the first n blocks (tests); 0 = all */
void model_limit_blocks(model_t *m, int n);
const model_stats_t *model_stats(const model_t *m);
void model_stats_reset(model_t *m);

/* img: float32 CHW, normalised, size*size*3. logits: 4 floats. Returns 0 on success. */
int model_forward(model_t *m, const float *img, float logits[4], model_dbg_fn dbg, void *dbg_ctx);

/* score_nsfw = softmax(logits)[1:].sum() */
float model_score(const float logits[4]);

/* RGB uint8 HWC -> normalised float32 CHW (torchvision ToTensor + Normalize, float32 arithmetic) */
void model_prepare_rgb8(const uint8_t *rgb, int size, float *out);

#endif
