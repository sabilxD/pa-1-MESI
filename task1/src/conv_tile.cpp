// conv_tile.cpp  STAGE 3: CACHE TILING  (Task 1B)
//
// IMPORTANT: the loop ORDER here is deliberately identical to conv_naive
// (oy, ox, ky, kx). Tiling is the ONLY change: two extra outer loops that walk the
// output in TY x TX blocks. This is required, not a stylistic choice.
//
// Task 1B asks for "the speedup achieved by the tiled version compared to the
// baseline" and for "the change in L1-D MPKI when moving from naive to tiled". Both
// of those attribute a result to tiling alone, so tiling must be the only difference
// from the baseline. Folding in the loop reordering from Task 1A would make the
// measured MPKI delta a property of reordering+tiling and impossible to attribute --
// and Task 1D plots "Loop reordering" and "Tiling" as separate techniques.
//
// Consequence, worth stating in the report rather than hiding: the baseline already
// keeps its accumulator in a register and touches each output element exactly once,
// so there is no long-distance reuse for blocking to shorten. Tiling therefore has
// little to work with here, and the speedup stays near 1.0x across every tile size.
// That is a real finding about when tiling helps, not a failed implementation.

#include <algorithm>

#include "convolution.h"

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
    // Tile size swept by benchmark_tiles.sh; see tile_results.csv for the MPKI and
    // execution-time data behind this choice.
    conv_tiled(in, out, ker, H, W, K, 512, 512);
}