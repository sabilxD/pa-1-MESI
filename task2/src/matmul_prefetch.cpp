#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

static inline float horizontal_sum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int MC = 120;      
    constexpr int NC = 120;       
    constexpr int PF_DIST = 128;
    // static_assert(MC % 4 == 0, "MC must be a multiple of the register block height (4)");
    // static_assert(NC % 3 == 0, "NC must be a multiple of the register block width (3)");

    const int M4 = M - (M % 4);    
    const int N3 = N - (N % 3);   // Microkernel size is 4*3 and M and N maynot be multiples of it, the tail section is handled manually

    for (int j0 = 0; j0 < N3; j0 += NC) {
        int jBlockEnd = std::min(j0 + NC, N3);
        if (jBlockEnd + NC < N3) {  // Trying to load the entire next CC * K block of B into lower level cache
            for (int j=jBlockEnd; j<jBlockEnd+NC; j++) {
                const float *b = B + static_cast<long>(j) * ldb;
                _mm_prefetch(reinterpret_cast<const char *>(&b[0]), _MM_HINT_T1);
                // for (int k=0; k<K; k++) {
                //     _mm_prefetch(reinterpret_cast<const char*>(&b[k]), _MM_HINT_T1); // T2 ??
                // }
            }
        }

        for (int i0 = 0; i0 < M4; i0 += MC) {
            int iBlockEnd = std::min(i0 + MC, M4);
            if (iBlockEnd + MC < M4) { // Trying to load the entire next MC * K block of A into lower level cache
                for (int i=iBlockEnd; i<iBlockEnd+MC; i++) {
                    const float *a = A + static_cast<long>(i) * lda;
                    _mm_prefetch(reinterpret_cast<const char *>(&a[0]), _MM_HINT_T1);
                    // for (int k=0; k<K; k++) {
                    //     _mm_prefetch(reinterpret_cast<const char*>(&a[k]), _MM_HINT_T1);
                    // }
                }
            }

            for (int i = i0; i < iBlockEnd; i += 4) {
                const float* a[4] = {A + static_cast<long>(i) * lda, A + static_cast<long>(i + 1) * lda, A + static_cast<long>(i + 2) * lda, A + static_cast<long>(i + 3) * lda};
                for (int j = j0; j < jBlockEnd; j += 3) {
                    const float* b[3] = {B + static_cast<long>(j) * ldb, B + static_cast<long>(j + 1) * ldb, B + static_cast<long>(j + 2) * ldb};

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

                    int p = 0;
                    for (; p+31<K; p+=32) {
                        _mm_prefetch(reinterpret_cast<const char*>(&a[0][p + PF_DIST]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&a[1][p + PF_DIST]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&a[2][p + PF_DIST]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&a[3][p + PF_DIST]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&b[0][p + PF_DIST]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&b[1][p + PF_DIST]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&b[2][p + PF_DIST]), _MM_HINT_T0);
                        
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
                        _mm_prefetch(reinterpret_cast<const char*>(&a[0][p + PF_DIST/2]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&a[1][p + PF_DIST/2]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&a[2][p + PF_DIST/2]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&a[3][p + PF_DIST/2]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&b[0][p + PF_DIST/2]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&b[1][p + PF_DIST/2]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&b[2][p + PF_DIST/2]), _MM_HINT_T0);
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
            }
        }
    }

    for (int i = 0; i < M4; i += 4) {
        for (int j = N3; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            for (int row = 0; row < 4; ++row) {
                const float* a = A + static_cast<long>(i + row) * lda;
                __m256 acc = _mm256_setzero_ps();
                int p = 0;
                for (; p + 7 < K; p += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
                float _acc = horizontal_sum(acc);
                for (; p < K; ++p) _acc += a[p] * b[p];
                C[static_cast<long>(i + row) * ldc + j] = _acc;
            }
        }
    }

    for (int i = M4; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 7 < K; p += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
            float _acc = horizontal_sum(acc);
            for (; p < K; ++p) _acc += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = _acc;
        }
    }
}