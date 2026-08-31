// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm>

void conv_tiled(const float* in, float* out, const float* ker,
                int H, int W, int K, int TY, int TX) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy_start = 0; oy_start < H; oy_start += TY) {
        for (int ox_start = 0; ox_start < W; ox_start += TX) {
            const int oy_end = std::min(oy_start + TY, H);
            const int ox_end = std::min(ox_start + TX, W);
            for (int oy = oy_start; oy < oy_end; oy++) {
                for (int ox = ox_start; ox < ox_end; ox++) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);
    conv_tiled(in,out,ker,H,W,K,512, 512);
    
}