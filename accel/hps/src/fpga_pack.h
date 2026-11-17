#ifndef FPGATE_FPGA_PACK_H
#define FPGATE_FPGA_PACK_H
/*
 * DDR3 formats of the accelerator (accel/DECISIONES.md, section "H2"; reference implementation
 * in accel/tools/pack_layer.py and dump_full_layer.py):
 *
 *   activations, tile-major:  [subtile of 16 rows][plane][k][16 bytes]
 *       byte i of a word = value of row (16*subtile + i) at input k; rows past the end are zero;
 *       16-bit inputs use two planes: plane 0 = high byte, plane 1 = low byte
 *   weights, per tile of 16 output channels: 16 entries x 16 bytes of epilogue constants
 *       ({bias i32, M0 i32, shift u8, 7 pad bytes} little endian), then K words of 16 bytes
 *       (byte j = weight of channel 16*tile + j at input k)
 *   output: plain row-major [M][N], int16 or int8 little endian (stride N * element size)
 */
#include <stddef.h>
#include <stdint.h>

#include "pkg.h"

size_t fpga_weights_bytes(const layer_t *L);
void fpga_pack_weights(const layer_t *L, uint8_t *dst);

/* Attention operands of one head (INT8 head slices [tokens][stride], the head's first column at the pointer):
 *   Q as activations (K = hd, one plane), rows [0, tokens);
 *   K^T as weight tiles of 16 keys (constants unused: the scores are raw INT32): word d of a tile has the 16
 *     keys' value at channel d; keys past `tokens` are zero; `ntiles` = ceil(tokens / 16);
 *   V as weight tiles of 16 output channels, K = kp words each (word k = V[k][16t .. 16t+15], zero past `tokens`),
 *     with the constants {bias 0, M0 = av_m0, shift = av_sh} of the context requantisation. */
void fpga_pack_acts8_strided(const int8_t *a, int rows, int K, int stride, uint8_t *dst);
size_t fpga_kt_bytes(int tokens, int hd);
void fpga_pack_kt(const int8_t *k, int tokens, int stride, int hd, uint8_t *dst);
size_t fpga_v_bytes(int hd, int kp);
void fpga_pack_v(const int8_t *v, int tokens, int stride, int hd, int kp, int32_t av_m0, int av_sh, uint8_t *dst);

size_t fpga_acts_bytes(int rows, int K, int planes);
void fpga_pack_acts8(const int8_t *a, int rows, int K, uint8_t *dst);
void fpga_pack_acts16(const int16_t *a, int rows, int K, uint8_t *dst);
/* one subtile of fpga_pack_acts16: rows (<= 16) rows of a, zeros below; hi and lo planes of K * 16 bytes each */
/* one subtile of fpga_pack_acts8: rows (<= 16) rows of a, zeros below; K * 16 bytes */
void fpga_pack_acts8_sub(const int8_t *a, int rows, int K, uint8_t *d);
void fpga_pack_acts16_sub(const int16_t *a, int rows, int K, uint8_t *hi, uint8_t *lo);

/* the engine needs N a multiple of 16 and K even */
static inline int fpga_supports(const layer_t *L) { return L->is_int && (L->out % 16) == 0 && (L->in % 2) == 0; }

#endif
