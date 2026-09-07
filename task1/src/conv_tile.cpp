// conv_tile.cpp  STAGE 3: CACHE TILING  (Task 1B)


#include <algorithm>

#include "convolution.h"

void conv_tiled(const float* in, float* out, const float* ker,
                int H, int W, int K, int TY, int TX) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy_start = 0; oy_start < H; oy_start += TY) {
        const int oy_end = std::min(oy_start + TY, H);
        for (int ox_start = 0; ox_start < W; ox_start += TX) {
            const int ox_end = std::min(ox_start + TX, W);
            for (int oy = oy_start; oy < oy_end; oy++) {
                float* out_row = out + oy * W;
                for (int ox = ox_start; ox < ox_end; ox++) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        const float* in_row = in + (oy + ky) * in_stride + ox;
                        const float* ker_row = ker + ky * K;
                        for (int kx = 0; kx < K; kx++) {
                            acc += in_row[kx] * ker_row[kx];
                        }
                    }
                    out_row[ox] = acc;
                }
            }
        }
    }
}

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    int TY = 16;
    int TX = 256;
    conv_tiled(in, out, ker, H, W, K, TY, TX);
}