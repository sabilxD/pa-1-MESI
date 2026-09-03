#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

// Forward declarations of the 128-bit and 256-bit SIMD functions
void conv_simd128(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_simd256(const float* in, float* out, const float* ker, int H, int W, int K);

struct SimdVariant {
    std::string name;
    int simd_bits;
    int vector_width;
    ConvFn fn;
};

int main(int argc, char** argv) {
    // Correctness tolerance
    const float kTol = 1e-3f;
    const int kWarmup = 3;
    const int kReps = 9;
    const unsigned seed = 1234u;

    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> kernels = {3, 5};

    if (argc >= 2 && std::strcmp(argv[1], "--csv-header") == 0) {
        std::cout << "matrix_size,H,W,K,variant,simd_bits,vector_width,time_ms,gflops,speedup,correct\n";
        return 0;
    }

    std::vector<SimdVariant> variants = {
        {"Scalar (Naive)", 32, 1, conv_naive},
        {"SIMD-128 (SSE)", 128, 4, conv_simd128},
        {"SIMD-256 (AVX2)", 256, 8, conv_simd256}
    };

    // Print CSV header
    std::cout << "matrix_size,H,W,K,variant,simd_bits,vector_width,time_ms,gflops,speedup,correct\n";

    for (int K : kernels) {
        for (int N : sizes) {
            int H = N;
            int W = N;

            float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
            float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
            float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
            float* ref = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

            pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
            pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
            float* in = pa1::make_padded(img, H, W, K);

            // Compute reference
            conv_naive(in, ref, ker, H, W, K);

            const double flops = pa1::conv_flops(H, W, K);
            double naive_ms = 0.0;

            for (const auto& var : variants) {
                auto run = [&]() { var.fn(in, out, ker, H, W, K); };
                run(); // Warmup / check
                const bool ok = (pa1::max_abs_diff(out, ref, H, W) <= kTol);

                // For very large sizes, reduce repetitions to keep benchmark fast
                int reps = (N >= 4096) ? 3 : kReps;
                int warmup = (N >= 4096) ? 1 : kWarmup;

                const double ms = pa1::time_median_ms(run, warmup, reps);
                const double gflops = flops / (ms * 1e6);

                if (var.simd_bits == 32) {
                    naive_ms = ms;
                }
                const double speedup = (ms > 0.0) ? (naive_ms / ms) : 1.0;

                std::cout << N << "," << H << "," << W << "," << K << ","
                          << var.name << "," << var.simd_bits << "," << var.vector_width << ","
                          << ms << "," << gflops << "," << speedup << ","
                          << (ok ? "yes" : "NO") << "\n";
            }

            pa1::free_floats(in);
            pa1::free_floats(img);
            pa1::free_floats(ker);
            pa1::free_floats(out);
            pa1::free_floats(ref);
        }
    }

    return 0;
}
