# Task 1B — Tiling and L1-D Cache Behaviour

**Workload:** 2D convolution, `conv_naive` vs `conv_tile`, single-threaded, pinned to CPU 0.
**Machine:** 13th Gen Intel Core i7-13700HX. **L1-D = 48 KB** per P-core (12-way, 64 B lines),
L2 = 1.25 MB per core, L3 = 30 MB shared. No AVX-512 (fused off on this hybrid part).
**Build:** `-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma` (pinned by the Makefile).

Reproduce everything here with `./bench_1d.sh` (see *Reproducing* at the end).

---

## 1. Baseline profile

`conv_naive`, 2048×2048, K=3:

| metric | value |
|---|---|
| execution time | 21.067 ms |
| L1-D hit rate | **99.4736 %** |
| L1-D miss rate | 0.5264 % |
| **L1-D MPKI** | **1.0353** |

The baseline already hits L1 **99.47 %** of the time. This single number governs everything
that follows: there is almost no miss traffic available to remove.

### Why the baseline is already cache-friendly

Per output pixel `conv_naive` touches three things:

| data | reuse | can tiling shorten it? |
|---|---|---|
| `out[oy*W+ox]` | written **once**, never re-read | No — zero reuse |
| `ker[]` | K² floats (36 B at K=3), permanently resident | No |
| `in[]` | each element feeds up to K² outputs | **Only candidate** |

So only `in[]` matters, and it has two reuse distances:

- **Within a row:** input element `x` serves outputs `ox = x-K+1 … x` — K consecutive
  iterations, a few cycles apart. Always L1.
- **Across rows:** input row `r` serves output rows `r-K+1 … r`. Between uses the kernel
  touches K input rows, i.e. a **reuse window of `K·(W+K-1)·4` bytes**.

At K=3, W=2048 that window is **24 KB — already inside the 48 KB L1**. Tiling exists to
shrink a working set into cache; here it is in cache before tiling starts.

---

## 2. Tile-size sweep (2048×2048, K=3)

| tile | time (ms) | speedup | L1-D hit | MPKI |
|---|---|---|---|---|
| naive | 21.067 | 1.000x | 99.4736 % | 1.0353 |
| 4×4 | 25.404 | 0.79x | 99.5153 % | 1.1002 |
| 8×8 | 26.574 | 0.75x | 99.5268 % | 1.0283 |
| 16×16 | 22.248 | 0.91x | 99.2429 % | 1.6072 |
| 32×32 | 20.688 | 0.98x | 99.3887 % | 1.2825 |
| 64×64 | 22.784 | 0.88x | 99.5080 % | 1.0257 |
| 128×128 | 20.015 | 1.01x | 99.5885 % | **0.8556** |
| 256×256 | 18.325 | 1.08x | 99.5744 % | 0.8835 |
| 512×512 | 17.987 | 1.10x | 99.5763 % | 0.8787 |
| 1024×1024 | 18.572 | 1.10x | 99.5820 % | 0.8667 |
| 2048×2048 | 17.747 | **1.16x** | 99.5374 % | 0.9589 |

**Best MPKI:** 128×128 (0.8556). **Best time:** 2048×2048 — which is a *single tile*, i.e.
no blocking at all. The two optima disagree, which is the first sign that MPKI is not what
is setting the runtime.

**Small tiles are actively harmful.** 16×16 *raises* MPKI to 1.6072, 55 % worse than naive.
Cause: every tile must load a (TY+K-1)×(TX+K-1) input halo to produce TY×TX outputs. At
16×16 that is 18×18 = 324 loads for 256 outputs — **27 % pure overhead**, and the halo rows
and columns are re-fetched by each neighbouring tile. Below roughly 128×128 the halo
overhead exceeds any locality benefit.

---

## 3. MPKI and speedup vs matrix size (K=3)

| size | tile | time (ms) | speedup | L1-D hit | MPKI |
|---|---|---|---|---|---|
| 512 | naive | 1.257 | 1.000x | 99.5279 % | 0.9336 |
| 512 | 32×32 | 1.246 | 0.99x | 99.4362 % | 1.1854 |
| 512 | 128×128 | 1.124 | 1.06x | 99.5457 % | 0.9443 |
| 512 | 512×512 | 1.105 | 1.12x | 99.5740 % | 0.8876 |
| 1024 | naive | 5.244 | 1.000x | 99.5482 % | 0.8897 |
| 1024 | 32×32 | 5.002 | 1.01x | 99.3795 % | 1.3029 |
| 1024 | 128×128 | 4.652 | 1.06x | 99.5374 % | 0.9619 |
| 1024 | 512×512 | 4.447 | 1.14x | 99.5745 % | 0.8821 |
| 2048 | naive | 20.219 | 1.000x | 99.4692 % | 1.0441 |
| 2048 | 32×32 | 20.931 | 0.96x | 99.3917 % | 1.2759 |
| 2048 | 128×128 | 19.950 | 1.02x | 99.5457 % | 0.9445 |
| 2048 | 512×512 | 18.390 | 1.11x | 99.5806 % | 0.8699 |
| **4096** | **naive** | 79.761 | 1.000x | **98.9584 %** | **2.0485** |
| 4096 | 32×32 | 83.272 | 0.96x | 99.3849 % | 1.2903 |
| 4096 | 128×128 | 81.656 | 0.98x | 99.5415 % | 0.9530 |
| 4096 | 512×512 | 74.011 | 1.08x | 99.5768 % | **0.8778** |

**N = 4096 is the interesting row.** The naive reuse window is `3·4098·4 = 48.0 KB`, i.e.
*exactly* L1 capacity, so it starts thrashing: hit rate falls to 98.9584 % and **MPKI doubles
to 2.0485**. Tiling repairs it completely — 512×512 restores 99.5768 % and **MPKI 0.8778, a
57 % reduction**.

And the speedup for that 57 % miss reduction is **1.08x**.

Note also that **tiled MPKI is flat across every matrix size** (0.8699–0.9619 for the 512
tile from N=512 to N=4096) while naive MPKI is size-dependent. That is exactly the intended
behaviour of tiling: the working set becomes `K·(TX+K-1)·4` (6 KB for TX=512, K=3),
independent of N. Tiling *does* what it is supposed to do.

---

## 4. Kernel-size sweep — forcing the naive baseline to miss

Growing K grows the reuse window `K·(W+K-1)·4`, so it crosses 48 KB around K=7. This is the
strongest test: make the baseline genuinely cache-limited and see whether tiling then pays.

2048×2048, tile 128×128:

| K | window | naive MPKI | tile MPKI | **MPKI cut** | naive ms | tile ms | speedup |
|---|---|---|---|---|---|---|---|
| 3 | 24.0 KB | 1.0646 | 0.9332 | 12 % | 20.269 | 20.423 | 1.02x |
| 5 | 40.1 KB | 1.0329 | 0.4962 | 52 % | 55.782 | 57.724 | 0.98x |
| 7 | 56.2 KB | 1.2219 | **0.3296** | **73 %** | 89.057 | 92.868 | 1.00x |
| 9 | 72.3 KB | 1.0180 | 0.2875 | 72 % | 156.280 | 164.969 | 0.96x |
| 11 | 88.4 KB | 0.9060 | 0.3645 | 60 % | 248.961 | 238.986 | 1.02x |
| 15 | 120.8 KB | 0.6694 | **0.2328** | 65 % | 495.753 | 503.976 | 0.98x |

Corresponding hit rates rise from 99.4588 %→99.5510 % (K=3) to **99.7771 %→99.9234 %** (K=15).

**This is the decisive result.** Tiling removes up to **73 % of all L1-D misses**, and the
speedup is **0.96–1.02x — indistinguishable from 1.0x at every single K.**

---

## Answers to the three questions

### Q1. Report the change in L1-D MPKI from naive to tiled. Justify.

MPKI falls in every configuration where the baseline actually misses:

| configuration | naive MPKI | tiled MPKI | reduction |
|---|---|---|---|
| 2048², K=3, tile 128 | 1.0353 | 0.8556 | 17 % |
| 4096², K=3, tile 512 | 2.0485 | 0.8778 | **57 %** |
| 2048², K=7, tile 128 | 1.2219 | 0.3296 | **73 %** |
| 2048², K=15, tile 128 | 0.6694 | 0.2328 | 65 % |

**Justification.** Tiling replaces the reuse window `K·(W+K-1)·4` with `K·(TX+K-1)·4`. At
K=7, W=2048 that is 56.2 KB → 3.7 KB, a 15× reduction that moves the sliding-window rows
from "spilling out of L1" to "comfortably resident", so the vertical reuse across output
rows now hits instead of missing.

The exception is **small tiles, where MPKI gets worse** (16×16 → 1.6072 vs naive 1.0353),
because the (TY+K-1)×(TX+K-1) halo is re-fetched by every neighbouring tile: at 16×16 that
is 27 % redundant loads, which outweighs the locality gained.

### Q2. How did MPKI vary across matrix sizes and tile sizes? Explain via cache hierarchy and working-set sizes.

**vs matrix size.** Naive MPKI is flat (0.89–1.04) for N = 512…2048, then **doubles to 2.0485
at N=4096**. The threshold is predicted exactly by the working-set formula: the reuse window
`3·(N+2)·4` reaches 48 KB — L1 capacity — at N≈4094. Below it the window is L1-resident;
above it, consecutive output rows evict each other's input rows and every row must be
re-fetched from L2.

Tiled MPKI is **independent of matrix size** (0.87–0.96 for tile 512 across all N), because
the tile, not the image, sets the working set: `K·(TX+K-1)·4` = 6 KB regardless of N.

**vs tile size.** MPKI is U-shaped. Very small tiles (16×16: 1.6072, 32×32: 1.2825) are worse
than naive because halo re-fetch dominates. It improves to a minimum around **128×128
(0.8556)**, where the tile is large enough to amortise the halo and small enough to be
L1-resident. Beyond that it flattens and creeps back up (2048×2048: 0.9589) as the working
set grows back toward the untiled case.

**Cache-hierarchy note:** the misses being traded here are L1→L2, not L2→DRAM. At 2048²
the input is 16 MB, which exceeds L2 (1.25 MB) but sits inside L3 (30 MB). So a "miss"
costs an L2/L3 hit of ~14–40 cycles, not a ~200-cycle DRAM trip — another reason the
runtime barely responds.

### Q3. Did you achieve a speedup? If not, analyse the limiting factors and propose solutions.

**No meaningful speedup: 0.96–1.16x, and the best time comes from the configuration that
does no tiling at all.** Three independent lines of evidence show why.

**(a) There was never any miss traffic to recover.** The baseline hit rate is 99.4736 %.
Across the whole tile sweep the hit rate moves within a band of **0.35 percentage points**
(99.2429 % → 99.5885 %) while speedup swings **55 %** (0.75x → 1.16x). Correlation of hit
rate with speedup is only **+0.387**; correlation of *instruction count* with speedup is
**−0.819**. Runtime is tracking instructions retired, not cache behaviour.

**(b) The kernel is latency-bound, not memory-bound.** The inner loop is a serial dependency
chain — `acc += in[...] * ker[...]`, K² times, each FMA depending on the previous. At K=3
that is 9 dependent FMAs × ~4-cycle latency ≈ **36 cycles per output pixel**, while the 9
loads it needs issue in parallel and hide completely underneath. Optimising memory when the
bottleneck is arithmetic latency is pure Amdahl. The K sweep proves it directly: removing
73 % of misses at K=7 changed the runtime by 0 %.

**(c) The residual ~1.1x is not a cache effect.** Two controls:

- Making `conv_tile` **byte-identical to `conv_naive`** still measures **1.01x** (0.98–1.03
  over 5 runs), so harness/ordering bias is ~1–3 %.
- Disassembly shows both compile to the *same* inner loop (`vmovss`/`vmulss`/`vaddss`/
  `add`/`cmp`/`jne`). But `conv_naive` **spills registers in its outer loops** — it reloads
  three values from the stack on every `oy` iteration (`mov edi,[rsp-0xc]`, `mov ecx,[rsp-0x8]`,
  `mov r13d,[rsp-0x4]`). `conv_tiled` hoists its bounds into `oy_end`/`ox_end` locals, so GCC
  allocates cleanly and skips the reloads: **373 M vs 413 M instructions per invocation, ~10 %
  fewer**, which buys the ~10 % time difference.

So the ~1.1x is a **register-allocation artifact of the restructured loop, not a
memory-hierarchy effect.**

**Proposed solutions — attack the real bottleneck.** The limiter is the serial FMA chain and
instruction count, so the techniques that pay are the ones that break that chain:

| technique | measured speedup vs naive |
|---|---|
| tiling | ~1.1x |
| loop unrolling (independent accumulator chains) | **2.98x** |
| SIMD (AVX2, 8 lanes/FMA) | **6.03x** |

Tiling would become worthwhile only once the kernel is fast enough to be memory-bound —
i.e. *after* SIMD — or at working sets that exceed **L2**, not L1. At 2048², K=3 neither
condition holds.

---

## Reproducing

```bash
cd task1
sudo sysctl -w kernel.perf_event_paranoid=1   # one-time; or run the script with sudo
./bench_1d.sh                                 # all four experiments, ~5 min
./bench_1d.sh tiles | sizes | kernels         # one experiment only
REPS=5 ./bench_1d.sh                          # more repetitions (median reported)
```

Outputs `1d_tiles.csv`, `1d_sizes.csv`, `1d_kernels.csv`. The first two are the two plots
the task asks for: **MPKI vs matrix size** and **speedup vs matrix size**, per tile size.
`src/` is never modified — the sweep builds in a throwaway copy under `/tmp`.

The script checks the environment before measuring and warns on anything that distorts
results. These checks are not decoration; each one produced wrong numbers during this work:

| check | why it matters |
|---|---|
| `perf` accessible | without it there are no cache numbers at all |
| governor = `performance` | on `powersave` the naive baseline swung 920 → 479 ms, silently doubling every speedup |
| AC connected | on battery the core sat at 800 MHz against a 4.8 GHz maximum |
| `platform_profile` | `balanced` caps sustained clock at ~3.4 GHz vs ~4.2 GHz |
| no background load | a VPN at 58 % CPU stole memory bandwidth and dragged one kernel from 10.71x to 6.98x |
| pinned to CPU 0 | unpinned, the scheduler migrates onto an E-core mid-measurement |
| HW prefetcher state reported | MSR `0x1A4`; changes miss counts, so it must be recorded alongside them |

Timings vary ±5 % run to run even when clean, so all figures above are **medians of 3**.
A single run is not evidence.
