#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ops.h"

int main(void)
{
    const int shapes[][3] = {{50, 768, 588}, {7, 13, 21}, {16, 64, 8}, {3, 5, 1}, {20, 9, 130}};
    srand(7);
    long bad = 0;
    for (int s = 0; s < 5; s++) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        int8_t *A = malloc((size_t)M * K), *W = malloc((size_t)N * K);
        int32_t *C = malloc((size_t)M * N * 4);
        for (int i = 0; i < M * K; i++) A[i] = (int8_t)(rand() % 255 - 127);
        for (int i = 0; i < N * K; i++) W[i] = (int8_t)(rand() % 255 - 127);
        if (s == 0) { for (int i = 0; i < M * K; i += 3) A[i] = -128; for (int i = 0; i < N * K; i += 5) W[i] = -128; }
        gemm_s8(A, W, C, M, N, K);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                int32_t r = 0;
                for (int k = 0; k < K; k++) r += (int32_t)A[(size_t)i * K + k] * W[(size_t)j * K + k];
                bad += r != C[(size_t)i * N + j];
            }
    }
    printf("gemm_s8 mismatches: %ld\n", bad);
    return bad != 0;
}
