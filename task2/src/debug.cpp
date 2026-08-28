#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

static constexpr float kRelTol = 1e-4f;

static constexpr int kDefN = 1024;
static constexpr unsigned kDefSeed = 1234u;

static constexpr int kWarmup = 1;
static constexpr int kReps = 3;

struct Stage {
    const char* name;
    const char* key;
    MatMulFn fn;
    bool scored;  // false only for the naive reference
};

static const Stage kStages[] = {
    {"naive (ref)", "naive", matmul_naive, false},
    {"simd", "simd", matmul_simd, true},
    {"prefetch", "prefetch", matmul_prefetch, true},
    {"optimized", "optimized", matmul_optimized, true},
};
static constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

static double run_config(int M, int N, int K, unsigned seed, const char* only,
                         bool show, bool scored) {
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);

    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);


    // Contiguous microbenchmark strides.
    const int lda = K, ldb = K, ldc = N;

    // Reference output from the naive kernel.
    matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
    const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
    const float tol = kRelTol * (ref_mag + 1e-30f);

    const double flops = pa1::matmul_flops(M, N, K);
    int correct_scored = 0;
    int incorrect_scored = 0;

    if (show) {
        std::printf("\n=== Workload  M=%d N=%d K=%d seed=%u %s ===\n", M, N, K, seed,
                    scored ? "(GRADED)" : "(not graded)");
        std::printf("%-14s  %-7s  %10s  %10s\n", "stage", "correct", "time(ms)",
                    "GFLOP/s");
        std::printf("--------------------------------------------------------------\n");
    }

    int stage_idx = -1;
    for (int s = 0; s < kNumStages; ++s) {
        if (std::strcmp(kStages[s].key, only) == 0) {
            stage_idx = s;
            break;
        }
    }
    if (stage_idx == -1) {
        std::printf("error, invalid stage %s", only);
    }

    auto run = [&]() { kStages[stage_idx].fn(A, B, C, M, N, K, lda, ldb, ldc); };
    run();  // one untimed run to produce the result we check
    const bool ok =
        (pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N) <= tol);
    if (kStages[stage_idx].scored) {
        if (ok) ++correct_scored; else ++incorrect_scored;
    }

    const double ms = pa1::time_median_ms(run, kWarmup, kReps);
    const double gflops = flops / (ms * 1e6);  // ms->s and FLOP->GFLOP

    if (show) {
        std::printf("%-14s  %-7s  %10.3f  %10.2f\n", kStages[stage_idx].name,
                    ok ? "yes" : "NO", ms, gflops);
    }

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    pa1::free_floats(ref);

    return static_cast<double>(incorrect_scored);

}


static const char* match_stage(const char* s) {
    if (std::strcmp(s, "all") == 0) return "all";
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(s, kStages[i].key) == 0) return kStages[i].key;
    return nullptr;
}

static void usage(const char* prog) {
    std::printf("Usage:\n");
    std::printf("  %s <stage> [M N K] [seed]     check one stage on a workload\n", prog);
    std::printf("\nstage: naive | simd | prefetch | optimized | all\n");
    std::printf("M, N, K default to %d; seed defaults to %u. All sizes must be positive.\n",
                kDefN, kDefSeed);
    std::printf("\nExamples:\n");
    std::printf("  %s simd            %s prefetch 1024 1024 1024        %s all 512 512 512\n",
                prog, prog, prog);
}

int main (int argc, char** argv) {
    const char* stage = match_stage(argv[1]);
    if (!stage) {
        std::printf("error: unknown stage '%s'\n\n", argv[1]);
        return 1;
    }

    int M = kDefN, N = kDefN, K = kDefN;
    unsigned seed = kDefSeed;
    if (argc >= 5) {
        M = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    } else if (argc != 2) {
        std::printf("error: give either no dimensions, or all three (M N K).\n\n");
        usage(argv[0]);
        return 1;
    }
    if (argc >= 6) seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));

    if (M <= 0 || N <= 0 || K <= 0) {
        std::printf("error: M, N, K must be positive.\n");
        return 1;
    }

    run_config(M, N, K, seed, stage, /*show=*/true, /*scored=*/false);
    return 0;

}
