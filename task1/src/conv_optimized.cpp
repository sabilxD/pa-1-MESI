// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"
#include <algorithm>

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int TY = (H >= 2048) ? 64 : H;

    for (int by = 0; by < H; by += TY) {
        const int y_end = std::min(by + TY, H);

        for (int oy = by; oy < y_end; ++oy) {
            float* out_row = &out[oy * W];
            int ox = 0;

            for (; ox + 63 < W; ox += 64) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                __m256 acc4 = _mm256_setzero_ps();
                __m256 acc5 = _mm256_setzero_ps();
                __m256 acc6 = _mm256_setzero_ps();
                __m256 acc7 = _mm256_setzero_ps();

                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row = &in[(oy + ky) * in_stride + ox];
                    const float* ker_row = &ker[ky * K];

                    for (int kx = 0; kx < K; ++kx) {
                        const __m256 k_vec = _mm256_set1_ps(ker_row[kx]);

                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 0),  k_vec, acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 8),  k_vec, acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 16), k_vec, acc2);
                        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 24), k_vec, acc3);
                        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 32), k_vec, acc4);
                        acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 40), k_vec, acc5);
                        acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 48), k_vec, acc6);
                        acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx + 56), k_vec, acc7);
                    }
                }

                _mm256_storeu_ps(out_row + ox + 0,  acc0);
                _mm256_storeu_ps(out_row + ox + 8,  acc1);
                _mm256_storeu_ps(out_row + ox + 16, acc2);
                _mm256_storeu_ps(out_row + ox + 24, acc3);
                _mm256_storeu_ps(out_row + ox + 32, acc4);
                _mm256_storeu_ps(out_row + ox + 40, acc5);
                _mm256_storeu_ps(out_row + ox + 48, acc6);
                _mm256_storeu_ps(out_row + ox + 56, acc7);
            }

            for (; ox < W; ox += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row = &in[(oy + ky) * in_stride + ox];
                    const float* ker_row = &ker[ky * K];
                    for (int kx = 0; kx < K; ++kx) {
                        const __m256 k_vec = _mm256_set1_ps(ker_row[kx]);
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx), k_vec, acc);
                    }
                }
                _mm256_storeu_ps(out_row + ox, acc);
            }
        }
    }
}
