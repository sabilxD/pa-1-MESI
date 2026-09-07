#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    constexpr int PF_DIST = 256;
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; p += 16) {

                if (j * ldb + p + PF_DIST < N * K) {
                    _mm_prefetch(
                        reinterpret_cast<const char*>(b + p + PF_DIST),
                        _MM_HINT_T0
                    );
                }

                for (int q = 0; q < 16 && p + q < K; ++q) {
                    acc += a[p + q] * b[p + q];
                }
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}