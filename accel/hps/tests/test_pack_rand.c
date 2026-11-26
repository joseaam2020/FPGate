#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "fpga_pack.h"

int main(void)
{
    const int cfg[][2] = {{37, 100}, {16, 16}, {33, 1232}, {1025, 768}, {5, 7}, {48, 31}, {1025, 64}};
    srand(3);
    long bad = 0;
    for (int c = 0; c < 7; c++) {
        int rows = cfg[c][0], K = cfg[c][1], subs = (rows + 15) / 16;
        int8_t *a8 = malloc((size_t)rows * K);
        int16_t *a16 = malloc((size_t)rows * K * 2);
        for (int i = 0; i < rows * K; i++) { a8[i] = (int8_t)rand(); a16[i] = (int16_t)(rand() * 7); }
        uint8_t *d8 = calloc(1, fpga_acts_bytes(rows, K, 1)), *d16 = calloc(1, fpga_acts_bytes(rows, K, 2));
        uint8_t *ds = calloc(1, fpga_acts_bytes(rows, K, 1));
        fpga_pack_acts8(a8, rows, K, d8);
        fpga_pack_acts16(a16, rows, K, d16);
        fpga_pack_acts8_strided(a8, rows, K, K, ds);
        for (int s = 0; s < subs; s++)
            for (int k = 0; k < K; k++)
                for (int i = 0; i < 16; i++) {
                    int r = s * 16 + i;
                    uint8_t e8 = r < rows ? (uint8_t)a8[(size_t)r * K + k] : 0;
                    uint16_t e16 = r < rows ? (uint16_t)a16[(size_t)r * K + k] : 0;
                    bad += d8[(size_t)s * K * 16 + (size_t)k * 16 + i] != e8;
                    bad += ds[(size_t)s * K * 16 + (size_t)k * 16 + i] != e8;
                    bad += d16[(size_t)s * 2 * K * 16 + (size_t)k * 16 + i] != (e16 >> 8);
                    bad += d16[(size_t)s * 2 * K * 16 + (size_t)K * 16 + (size_t)k * 16 + i] != (e16 & 255);
                }
    }
    printf("pack mismatches: %ld\n", bad);
    return bad != 0;
}
