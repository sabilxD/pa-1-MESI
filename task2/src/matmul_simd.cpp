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
        for (int j = 0; j < N; j+=4) {
            const float* b0 = B + static_cast<long>(j) * ldb;
            const float* b1 = B + static_cast<long>(j+1) * ldb;
            const float* b2 = B + static_cast<long>(j+2) * ldb;
            const float* b3 = B + static_cast<long>(j+3) * ldb;
            
            double temp[4];
            for (int t = 0; t < 4; ++t) {
                // std::cout << i << " " << j << " " << t << std::endl;
                __m256d acc = _mm256_setzero_pd();
                for (int p = 0; p < K; ++p) {
                    double Aa = (double)a[t][p];
                    acc = _mm256_add_pd(acc, 
                            _mm256_mul_pd(
                                _mm256_set_pd(Aa, Aa, Aa, Aa),
                                _mm256_set_pd((double)b0[p], (double)b1[p], (double)b2[p], (double)b3[p])
                            ));
                }
                _mm256_storeu_pd(temp, acc);
                // first number is at highest position (3), weird
                C[static_cast<long>(i+t) * ldc + j] = (float)temp[3];
                C[static_cast<long>(i+t) * ldc + j+1] = (float)temp[2];
                C[static_cast<long>(i+t) * ldc + j+2] = (float)temp[1];
                C[static_cast<long>(i+t) * ldc + j+3] = (float)temp[0];
            }
        }
    }
}
