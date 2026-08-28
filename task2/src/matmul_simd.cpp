// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"
#include <iostream>

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    for (int i = 0; i < M; i+=4) {
        const float* a[4] = {A + static_cast<long>(i) * lda,
                             A + static_cast<long>(i+1) * lda,
                             A + static_cast<long>(i+2) * lda,
                             A + static_cast<long>(i+3) * lda};
        for (int j = 0; j < N; j+=8) {
            const float* b0 = B + static_cast<long>(j) * ldb;
            const float* b1 = B + static_cast<long>(j+1) * ldb;
            const float* b2 = B + static_cast<long>(j+2) * ldb;
            const float* b3 = B + static_cast<long>(j+3) * ldb;
            const float* b4 = B + static_cast<long>(j+4) * ldb;
            const float* b5 = B + static_cast<long>(j+5) * ldb;
            const float* b6 = B + static_cast<long>(j+6) * ldb;
            const float* b7 = B + static_cast<long>(j+7) * ldb;
            
            float temp[8];
            for (int t = 0; t < 4; ++t) {
                __m256 acc = _mm256_setzero_ps();
                for (int p = 0; p < K; ++p) {
                    float Aa = a[t][p];
                    acc = _mm256_add_ps(acc, 
                            _mm256_mul_ps(
                                _mm256_set_ps(Aa, Aa, Aa, Aa, Aa, Aa, Aa, Aa),
                                _mm256_set_ps((float)b0[p], (float)b1[p], (float)b2[p], (float)b3[p], (float)b4[p], (float)b5[p], (float)b6[p], (float)b7[p])
                            ));
                }
                _mm256_storeu_ps(temp, acc);
                C[static_cast<long>(i+t) * ldc + j] = (float)temp[7];
                C[static_cast<long>(i+t) * ldc + j+1] = (float)temp[6];
                C[static_cast<long>(i+t) * ldc + j+2] = (float)temp[5];
                C[static_cast<long>(i+t) * ldc + j+3] = (float)temp[4];
                C[static_cast<long>(i+t) * ldc + j+4] = (float)temp[3];
                C[static_cast<long>(i+t) * ldc + j+5] = (float)temp[2];
                C[static_cast<long>(i+t) * ldc + j+6] = (float)temp[1];
                C[static_cast<long>(i+t) * ldc + j+7] = (float)temp[0];
            }
        }
    }
}
