// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // for (int oy = 0; oy < H; ++oy) {
    //     for (int ox = 0; ox < W; ox+=4) {
    //         float acc1 = 0.0f;
    //         float acc2 = 0.0f;
    //         float acc3 = 0.0f;
    //         float acc4 = 0.0f;

    //         for (int ky = 0; ky < K; ++ky) {
    //             for (int kx = 0; kx < K; ++kx) {
    //                 acc1 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
    //                 acc2 += in[(oy + ky) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
    //                 acc3 += in[(oy + ky) * in_stride + (ox + 2 + kx)] * ker[ky * K + kx];
    //                 acc4 += in[(oy + ky) * in_stride + (ox + 3 + kx)] * ker[ky * K + kx];
    //             }
    //         }
    //         out[oy * W + ox] = acc1;
    //         out[oy * W + ox + 1] = acc2;
    //         out[oy * W + ox + 2] = acc3;
    //         out[oy * W + ox + 3] = acc4;
    //     }
    // }
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox+=8) {
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            float acc8 = 0.0f;


            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc1 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox + 2 + kx)] * ker[ky * K + kx];
                    acc4 += in[(oy + ky) * in_stride + (ox + 3 + kx)] * ker[ky * K + kx];
                    acc5 += in[(oy + ky) * in_stride + (ox + 4 + kx)] * ker[ky * K + kx];
                    acc6 += in[(oy + ky) * in_stride + (ox + 5 + kx)] * ker[ky * K + kx];
                    acc7 += in[(oy + ky) * in_stride + (ox + 6 + kx)] * ker[ky * K + kx];
                    acc8 += in[(oy + ky) * in_stride + (ox + 7 + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc1;
            out[oy * W + ox + 1] = acc2;
            out[oy * W + ox + 2] = acc3;
            out[oy * W + ox + 3] = acc4;
            out[oy * W + ox + 4] = acc5;
            out[oy * W + ox + 5] = acc6;
            out[oy * W + ox + 6] = acc7;
            out[oy * W + ox + 7] = acc8;
        }
    }
}
