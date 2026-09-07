#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Load benchmark results
df = pd.read_csv("1c_comprehensive_results.csv")

os.makedirs("figures", exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

kernel_sizes = sorted(df['K'].unique())

# ==============================================================================
# Plot 1: Speedup vs Matrix Size (128-bit vs 256-bit SIMD)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

variants = ["simd128", "simd256"]
labels = {"simd128": "SIMD-128 (SSE 4-wide)", "simd256": "SIMD-256 (AVX2 8-wide)"}
colors = {"simd128": "#1f77b4", "simd256": "#d62728"}
markers = {"simd128": "o", "simd256": "s"}

for idx, K in enumerate(kernel_sizes):
    ax = axes[idx]
    df_k = df[df['K'] == K]

    # Reference lines for theoretical vector width speedups
    # ax.axhline(y=4.0, color='#1f77b4', linestyle='--', alpha=0.5, label='Theoretical 4x (128-bit)')
    # ax.axhline(y=8.0, color='#d62728', linestyle='--', alpha=0.5, label='Theoretical 8x (256-bit)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Naive Baseline (1x)')

    for var in variants:
        df_var = df_k[df_k['stage'] == var]
        ax.plot(df_var['N'], df_var['speedup'],
                marker=markers[var], markersize=8, linewidth=2.2,
                color=colors[var], label=labels[var])

    ax.set_xscale('log', base=2)
    ax.set_xticks(df['N'].unique())
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Matrix Dimension $N$ ($N \\times N$)")
    if idx == 0:
        ax.set_ylabel("Speedup over Scalar Naive ($\\times$)")
    ax.set_title(f"Kernel Size $K = {K}\\times {K}$")
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.legend(loc='lower right' if idx == 1 else 'upper right', frameon=True)
    ax.set_ylim(0, 10.5)

plt.suptitle("Task 1C: SIMD Speedup vs. Matrix Size Across Register Widths", y=1.02)
plt.tight_layout()
plt.savefig("figures/1c_simd_speedup.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/1c_simd_speedup.png")

# ==============================================================================
# Plot 2: Instructions (Millions) vs Matrix Size (Log Scale)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

all_stages = ["naive", "simd128", "simd256"]
stage_labels = {"naive": "Scalar Naive", "simd128": "SIMD-128 (SSE)", "simd256": "SIMD-256 (AVX2)"}
stage_colors = {"naive": "#7f7f7f", "simd128": "#1f77b4", "simd256": "#d62728"}
stage_markers = {"naive": "^", "simd128": "o", "simd256": "s"}

for idx, K in enumerate(kernel_sizes):
    ax = axes[idx]
    df_k = df[df['K'] == K]

    for stage in all_stages:
        df_s = df_k[df_k['stage'] == stage]
        ax.plot(df_s['N'], df_s['instructions'] / 1e6,
                marker=stage_markers[stage], markersize=8, linewidth=2.2,
                color=stage_colors[stage], label=stage_labels[stage])

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(df['N'].unique())
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Matrix Dimension $N$ ($N \\times N$)")
    ax.set_ylabel("Instructions ($\\times 10^6$, log scale)")
    ax.set_title(f"Kernel Size $K = {K}\\times {K}$")
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.legend(loc='upper left', frameon=True)

plt.suptitle("Task 1C: Instruction Count Reduction Across SIMD Register Widths", y=1.02)
plt.tight_layout()
plt.savefig("figures/1c_simd_instructions.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/1c_simd_instructions.png")

# ==============================================================================
# Plot 3: Throughput (GFLOP/s) vs Matrix Size
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for idx, K in enumerate(kernel_sizes):
    ax = axes[idx]
    df_k = df[df['K'] == K]

    for stage in all_stages:
        df_s = df_k[df_k['stage'] == stage]
        ax.plot(df_s['N'], df_s['gflops'],
                marker=stage_markers[stage], markersize=8, linewidth=2.2,
                color=stage_colors[stage], label=stage_labels[stage])

    ax.set_xscale('log', base=2)
    ax.set_xticks(df['N'].unique())
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Matrix Dimension $N$ ($N \\times N$)")
    if idx == 0:
        ax.set_ylabel("Compute Throughput (GFLOP/s)")
    ax.set_title(f"Kernel Size $K = {K}\\times {K}$")
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.legend(loc='upper right', frameon=True)

plt.suptitle("Task 1C: Compute Throughput (GFLOP/s) Scaling", y=1.02)
plt.tight_layout()
plt.savefig("figures/1c_simd_gflops.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/1c_simd_gflops.png")
