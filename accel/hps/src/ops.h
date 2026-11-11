#ifndef FPGATE_OPS_H
#define FPGATE_OPS_H
/*
 * Integer and float kernels of the model, written to match model/quant/export_check.py
 * (the executable specification of the package). The CPU backend uses them directly; the
 * FPGA backend replaces the GEMM and attention kernels and keeps the rest.
 *
 * Rounding rules: float -> int quantisation is round-half-to-even (like torch.round),
 * the fixed-point epilogue is (acc * M0 + 2^(shift-1)) >> shift with an arithmetic shift,
 * saturated to +-(2^(bits-1) - 1).
 */
#include <stddef.h>
#include <stdint.h>

#include "pkg.h"

static inline int qmax_of(int bits) { return (1 << (bits - 1)) - 1; }

/* q = clamp(round(x / (float)scale)); writes int8 when bits == 8, int16 otherwise */
void quantize_f32(const float *x, size_t n, double scale, int bits, void *q);

/* C[M][N] int32 = A[M][K] * W[N][K]^T; the accumulator is INT32 like the hardware */
void gemm_s8(const int8_t *A, const int8_t *W, int32_t *C, int M, int N, int K);
void gemm_s16(const int16_t *A, const int8_t *W, int32_t *C, int M, int N, int K);

/* out = sat((acc + bias) * M0 + 2^(shift-1)) >> shift), per output channel; int16 container */
void epilogue_int(const int32_t *acc, int M, int N, const int32_t *bias, const int32_t *m0, const int32_t *shift,
                  int bits, int16_t *out);

/* LayerNorm exactly as float_ref.layer_norm: (x - mu) / sqrt(var + eps) * g + b in float32 */
void layer_norm(const float *x, int rows, int d, const float *g, const float *b, double eps, float *y);

/* RoPE on INT16 pre-RoPE values (INT8 result); x is [tokens][stride] and each head has hd columns.
 * Token 0 (the class token) is not rotated. cos/sin are [tokens-1][hd] in Q1.(frac). */
void rope_int(const int16_t *x, int tokens, int stride, int heads, int hd, const int16_t *cos_tab,
              const int16_t *sin_tab, int frac, int pre_shift, int8_t *out);

/* Integer softmax of one row of n scores (UQ0.softmax_bits output, bit-exact with export_check.softmax) */
void softmax_row_int(const pkg_t *p, const block_t *b, const int32_t *S, int n, uint16_t *P);

/* One head, with all intermediates (reference dumps for the RTL tests): q, k, v are INT8 [tokens][stride] starting
 * at the head's first column; S = [tokens][tokens] INT32, P = [tokens][tokens] UQ0.16, ctx = [tokens][hd] INT8. */
void attention_head_ref(const pkg_t *p, const block_t *b, const int8_t *q, const int8_t *k, const int8_t *v,
                        int tokens, int stride, int hd, int32_t *S, uint16_t *P, int8_t *ctx);

/* Integer attention for one block: q, k, v are INT8 [tokens][stride]; writes the INT8 context [tokens][stride]. */
void attention_int(const pkg_t *p, const block_t *b, const int8_t *q, const int8_t *k, const int8_t *v, int tokens,
                   int stride, int heads, int hd, int8_t *ctx);

#endif
