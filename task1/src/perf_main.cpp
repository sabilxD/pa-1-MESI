// perf_main.cpp — Isolated runner for hardware counter profiling via perf stat.
//
// Unlike main.cpp, this binary:
// 1. Never runs conv_naive unless explicitly requested as the target stage.
// 2. Allocates and initializes memory before any measurement.
// 3. Executes ONLY the chosen stage for the requested number of repetitions.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

struct Stage {
    const char* name;
    ConvFn fn;
};

static const Stage kStages[] = {
    {"naive", conv_naive},
    {"reorder", conv_reorder},
    {"unroll", conv_unroll},
    {"tile", conv_tile},
    {"simd", conv_simd},
    {"optimized", conv_optimized},
};
static constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

static const Stage* find_stage(const char* name) {
    for (int i = 0; i < kNumStages; ++i) {
        if (std::strcmp(kStages[i].name, name) == 0) return &kStages[i];
    }
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: %s <stage> [H] [W] [K] [reps] [seed]\n", argv[0]);
        std::printf("Stages: naive | reorder | unroll | tile | simd | optimized\n");
        return 1;
    }

    const Stage* stage = find_stage(argv[1]);
    if (!stage) {
        std::fprintf(stderr, "Error: Unknown stage '%s'\n", argv[1]);
        return 1;
    }

    int H = 2048, W = 2048, K = 3;
    int reps = 1;
    unsigned seed = 1234u;

    if (argc >= 5) {
        H = std::atoi(argv[2]);
        W = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        reps = std::atoi(argv[5]);
    }
    if (argc >= 7) {
        seed = static_cast<unsigned>(std::strtoul(argv[6], nullptr, 10));
    }

    if (H <= 0 || W <= 0 || K <= 0 || reps <= 0) {
        std::fprintf(stderr, "Error: H, W, K, and reps must be positive.\n");
        return 1;
    }
    if (W % 8 != 0) {
        std::fprintf(stderr, "Error: W (%d) must be a multiple of 8.\n", W);
        return 1;
    }
    if (K % 2 == 0) {
        std::fprintf(stderr, "Error: K (%d) must be odd.\n", K);
        return 1;
    }

    // Allocate buffers
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);

    // Warm up page tables by touching output buffer
    std::memset(out, 0, static_cast<std::size_t>(H) * W * sizeof(float));

    // Execute ONLY the requested stage for 'reps' iterations
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        stage->fn(in, out, ker, H, W, K);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_run_ms = total_ms / reps;
    double flops_per_run = pa1::conv_flops(H, W, K);
    double gflops = flops_per_run / (per_run_ms * 1e6);

    std::printf("stage: %s | H: %d | W: %d | K: %d | reps: %d | total_time: %.3f ms | per_run: %.3f ms | GFLOP/s: %.2f\n",
                stage->name, H, W, K, reps, total_ms, per_run_ms, gflops);

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);

    return 0;
}
