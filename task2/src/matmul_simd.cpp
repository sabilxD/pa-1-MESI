// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"
#include <iostream>

static inline float horizontal_sum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    
    __m128 shuf = _mm_movehdup_ps(sum128); 
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p=0;
            for (; p + 7 < K; p += 8) {
                __m256 b0 = _mm256_loadu_ps(b+p);
                __m256 a0 = _mm256_loadu_ps(a+p);
                acc = _mm256_fmadd_ps(a0, b0, acc);
            }
            float c = horizontal_sum(acc);
            for (; p < K; p++) c += a[p]*b[p];
            C[static_cast<long>(i) * ldc + j] = c;
        }
    }
}


// static inline float horizontal_sum128(__m128 v) {
//     __m128 shuf = _mm_movehdup_ps(v);
//     __m128 sums = _mm_add_ps(v, shuf);
//     shuf = _mm_movehl_ps(shuf, sums);
//     sums = _mm_add_ss(sums, shuf);
//     return _mm_cvtss_f32(sums);
// }

// void matmul_simd(const float* A, const float* B, float* C,
//                  int M, int N, int K, int lda, int ldb, int ldc) {

//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             const float* a = A + static_cast<long>(i) * lda;
//             const float* b = B + static_cast<long>(j) * ldb;
//             __m128 acc = _mm_setzero_ps();
//             int p = 0;
//             for (; p + 3 < K; p += 4) {
//                 __m128 a0 = _mm_loadu_ps(a + p);
//                 __m128 b0 = _mm_loadu_ps(b + p);
//                 acc = _mm_fmadd_ps(a0, b0, acc);
//             }
//             float c = horizontal_sum128(acc);
//             for (; p < K; ++p) c += a[p] * b[p];
//             C[static_cast<long>(i) * ldc + j] = c;
//         }
//     }
// }