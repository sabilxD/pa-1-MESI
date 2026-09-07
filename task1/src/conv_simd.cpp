// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd128(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p; 

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 k_vec = _mm_set1_ps(ker[ky * K + kx]);
                    __m128 in_vec = _mm_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    acc = _mm_fmadd_ps(in_vec, k_vec, acc);
                }
            }
            _mm_storeu_ps(&out[oy * W + ox], acc);
        }
    }
}

void conv_simd256(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 k_vec = _mm256_set1_ps(ker[ky * K + kx]);
                    __m256 in_vec = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    acc = _mm256_fmadd_ps(in_vec, k_vec, acc);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], acc);
        }
    }
}

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    // conv_simd128(in, out, ker, H, W, K);
    conv_simd256(in, out, ker, H, W, K);
    
}
