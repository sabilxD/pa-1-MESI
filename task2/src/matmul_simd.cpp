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
    for (int i=0; i+3<M; i+=4) {
        const float *a[4] = {A + static_cast<long>(i)*lda, A + static_cast<long>(i+1)*lda, A + static_cast<long>(i+2)*lda, A + static_cast<long>(i+3)*lda};
        for (int j=0; j+2<N; j+=3) {
            const float *b[3] = {B + static_cast<long>(j) * ldb, B + static_cast<long>(j+1) * ldb, B + static_cast<long>(j+2) * ldb};
            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc21 = _mm256_setzero_ps();
            __m256 acc31 = _mm256_setzero_ps();
            __m256 acc02 = _mm256_setzero_ps();
            __m256 acc12 = _mm256_setzero_ps();
            __m256 acc22 = _mm256_setzero_ps();
            __m256 acc32 = _mm256_setzero_ps();
            
            int p=0;
            for (; p+31<K; p+=32) {
                
                __m256 b0 = _mm256_loadu_ps(b[0] + p);
                __m256 b1 = _mm256_loadu_ps(b[1] + p);
                __m256 b2 = _mm256_loadu_ps(b[2] + p);

                __m256 a_val = _mm256_loadu_ps(a[0] + p);
                acc00 = _mm256_fmadd_ps(a_val, b0, acc00);
                acc01 = _mm256_fmadd_ps(a_val, b1, acc01);
                acc02 = _mm256_fmadd_ps(a_val, b2, acc02);

                a_val = _mm256_loadu_ps(a[1] + p);
                acc10 = _mm256_fmadd_ps(a_val, b0, acc10);
                acc11 = _mm256_fmadd_ps(a_val, b1, acc11);
                acc12 = _mm256_fmadd_ps(a_val, b2, acc12);

                a_val = _mm256_loadu_ps(a[2] + p);
                acc20 = _mm256_fmadd_ps(a_val, b0, acc20);
                acc21 = _mm256_fmadd_ps(a_val, b1, acc21);
                acc22 = _mm256_fmadd_ps(a_val, b2, acc22);

                a_val = _mm256_loadu_ps(a[3] + p);
                acc30 = _mm256_fmadd_ps(a_val, b0, acc30);
                acc31 = _mm256_fmadd_ps(a_val, b1, acc31);
                acc32 = _mm256_fmadd_ps(a_val, b2, acc32);

                b0 = _mm256_loadu_ps(b[0] + p + 8);
                b1 = _mm256_loadu_ps(b[1] + p + 8);
                b2 = _mm256_loadu_ps(b[2] + p + 8);

                a_val = _mm256_loadu_ps(a[0] + p + 8);
                acc00 = _mm256_fmadd_ps(a_val, b0, acc00);
                acc01 = _mm256_fmadd_ps(a_val, b1, acc01);
                acc02 = _mm256_fmadd_ps(a_val, b2, acc02);

                a_val = _mm256_loadu_ps(a[1] + p + 8);
                acc10 = _mm256_fmadd_ps(a_val, b0, acc10);
                acc11 = _mm256_fmadd_ps(a_val, b1, acc11);
                acc12 = _mm256_fmadd_ps(a_val, b2, acc12);

                a_val = _mm256_loadu_ps(a[2] + p + 8);
                acc20 = _mm256_fmadd_ps(a_val, b0, acc20);
                acc21 = _mm256_fmadd_ps(a_val, b1, acc21);
                acc22 = _mm256_fmadd_ps(a_val, b2, acc22);

                a_val = _mm256_loadu_ps(a[3] + p + 8);
                acc30 = _mm256_fmadd_ps(a_val, b0, acc30);
                acc31 = _mm256_fmadd_ps(a_val, b1, acc31);
                acc32 = _mm256_fmadd_ps(a_val, b2, acc32);

                b0 = _mm256_loadu_ps(b[0] + p + 16);
                b1 = _mm256_loadu_ps(b[1] + p + 16);
                b2 = _mm256_loadu_ps(b[2] + p + 16);

                a_val = _mm256_loadu_ps(a[0] + p + 16);
                acc00 = _mm256_fmadd_ps(a_val, b0, acc00);
                acc01 = _mm256_fmadd_ps(a_val, b1, acc01);
                acc02 = _mm256_fmadd_ps(a_val, b2, acc02);

                a_val = _mm256_loadu_ps(a[1] + p + 16);
                acc10 = _mm256_fmadd_ps(a_val, b0, acc10);
                acc11 = _mm256_fmadd_ps(a_val, b1, acc11);
                acc12 = _mm256_fmadd_ps(a_val, b2, acc12);

                a_val = _mm256_loadu_ps(a[2] + p + 16);
                acc20 = _mm256_fmadd_ps(a_val, b0, acc20);
                acc21 = _mm256_fmadd_ps(a_val, b1, acc21);
                acc22 = _mm256_fmadd_ps(a_val, b2, acc22);

                a_val = _mm256_loadu_ps(a[3] + p + 16);
                acc30 = _mm256_fmadd_ps(a_val, b0, acc30);
                acc31 = _mm256_fmadd_ps(a_val, b1, acc31);
                acc32 = _mm256_fmadd_ps(a_val, b2, acc32);

                b0 = _mm256_loadu_ps(b[0] + p + 24);
                b1 = _mm256_loadu_ps(b[1] + p + 24);
                b2 = _mm256_loadu_ps(b[2] + p + 24);

                a_val = _mm256_loadu_ps(a[0] + p + 24);
                acc00 = _mm256_fmadd_ps(a_val, b0, acc00);
                acc01 = _mm256_fmadd_ps(a_val, b1, acc01);
                acc02 = _mm256_fmadd_ps(a_val, b2, acc02);

                a_val = _mm256_loadu_ps(a[1] + p + 24);
                acc10 = _mm256_fmadd_ps(a_val, b0, acc10);
                acc11 = _mm256_fmadd_ps(a_val, b1, acc11);
                acc12 = _mm256_fmadd_ps(a_val, b2, acc12);

                a_val = _mm256_loadu_ps(a[2] + p + 24);
                acc20 = _mm256_fmadd_ps(a_val, b0, acc20);
                acc21 = _mm256_fmadd_ps(a_val, b1, acc21);
                acc22 = _mm256_fmadd_ps(a_val, b2, acc22);

                a_val = _mm256_loadu_ps(a[3] + p + 24);
                acc30 = _mm256_fmadd_ps(a_val, b0, acc30);
                acc31 = _mm256_fmadd_ps(a_val, b1, acc31);
                acc32 = _mm256_fmadd_ps(a_val, b2, acc32);
            }

            for (; p + 7 < K; p += 8) {
                __m256 b0 = _mm256_loadu_ps(b[0] + p);
                __m256 b1 = _mm256_loadu_ps(b[1] + p);
                __m256 b2 = _mm256_loadu_ps(b[2] + p);

                __m256 a_val = _mm256_loadu_ps(a[0] + p);
                acc00 = _mm256_fmadd_ps(a_val, b0, acc00);
                acc01 = _mm256_fmadd_ps(a_val, b1, acc01);
                acc02 = _mm256_fmadd_ps(a_val, b2, acc02);

                a_val = _mm256_loadu_ps(a[1] + p);
                acc10 = _mm256_fmadd_ps(a_val, b0, acc10);
                acc11 = _mm256_fmadd_ps(a_val, b1, acc11);
                acc12 = _mm256_fmadd_ps(a_val, b2, acc12);

                a_val = _mm256_loadu_ps(a[2] + p);
                acc20 = _mm256_fmadd_ps(a_val, b0, acc20);
                acc21 = _mm256_fmadd_ps(a_val, b1, acc21);
                acc22 = _mm256_fmadd_ps(a_val, b2, acc22);

                a_val = _mm256_loadu_ps(a[3] + p);
                acc30 = _mm256_fmadd_ps(a_val, b0, acc30);
                acc31 = _mm256_fmadd_ps(a_val, b1, acc31);
                acc32 = _mm256_fmadd_ps(a_val, b2, acc32);
            }


            float c00 = horizontal_sum(acc00);
            float c01 = horizontal_sum(acc01);
            float c02 = horizontal_sum(acc02);
            float c10 = horizontal_sum(acc10);
            float c11 = horizontal_sum(acc11);
            float c12 = horizontal_sum(acc12);
            float c20 = horizontal_sum(acc20);
            float c21 = horizontal_sum(acc21);
            float c22 = horizontal_sum(acc22);
            float c30 = horizontal_sum(acc30);
            float c31 = horizontal_sum(acc31);
            float c32 = horizontal_sum(acc32);

            for (; p < K; ++p) {
                c00 += a[0][p] * b[0][p];
                c01 += a[0][p] * b[1][p];
                c02 += a[0][p] * b[2][p];
                c10 += a[1][p] * b[0][p];
                c11 += a[1][p] * b[1][p];
                c12 += a[1][p] * b[2][p];
                c20 += a[2][p] * b[0][p];
                c21 += a[2][p] * b[1][p];
                c22 += a[2][p] * b[2][p];
                c30 += a[3][p] * b[0][p];
                c31 += a[3][p] * b[1][p];
                c32 += a[3][p] * b[2][p];
            }

            C[static_cast<long>(i + 0) * ldc + (j + 0)] = c00;
            C[static_cast<long>(i + 0) * ldc + (j + 1)] = c01;
            C[static_cast<long>(i + 0) * ldc + (j + 2)] = c02;

            C[static_cast<long>(i + 1) * ldc + (j + 0)] = c10;
            C[static_cast<long>(i + 1) * ldc + (j + 1)] = c11;
            C[static_cast<long>(i + 1) * ldc + (j + 2)] = c12;

            C[static_cast<long>(i + 2) * ldc + (j + 0)] = c20;
            C[static_cast<long>(i + 2) * ldc + (j + 1)] = c21;
            C[static_cast<long>(i + 2) * ldc + (j + 2)] = c22;

            C[static_cast<long>(i + 3) * ldc + (j + 0)] = c30;
            C[static_cast<long>(i + 3) * ldc + (j + 1)] = c31;
            C[static_cast<long>(i + 3) * ldc + (j + 2)] = c32;
        }

        for (int j = (N / 3) * 3; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            for (int row = 0; row < 4; ++row) {
                const float* a = A + static_cast<long>(i + row) * lda;
                __m256 acc = _mm256_setzero_ps();
                int p=0;
                for (; p+7 < K; p+=8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p), acc);
                }
                float _acc = horizontal_sum(acc);
                for (; p<K; p++) _acc += a[p]*b[p];
                C[static_cast<long>(i + row) * ldc + j] = _acc;
            }
        }
    }

    for (int i = (M / 4) * 4; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p=0;
            for (; p+7 < K; p+=8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p), acc);
            }
            float _acc = horizontal_sum(acc);
            for (; p<K; p++) _acc += a[p]*b[p];
            C[static_cast<long>(i) * ldc + j] = _acc;
        }
    }
}