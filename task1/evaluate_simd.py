#!/usr/bin/env python3
import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def build_benchmark():
    print("[1/4] Compiling benchmark executable...")
    cmd = [
        "g++", "-std=c++17", "-O2", "-fno-tree-vectorize", "-mavx2", "-mfma",
        "-Iinclude", "-Wall",
        "src/conv_naive.cpp", "src/conv_simd.cpp", "src/benchmark_simd.cpp",
        "-o", "bin/benchmark_simd"
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("Error: Compilation failed!")
        sys.exit(1)
    print("      Compiled successfully to bin/benchmark_simd.")

def run_benchmark(csv_file="simd_benchmark_results.csv"):
    print("[2/4] Running multi-size SIMD benchmarks...")
    cmd = ["./bin/benchmark_simd"]
    with open(csv_file, "w") as f:
        res = subprocess.run(cmd, stdout=f)
    if res.returncode != 0:
        print("Error: Benchmark execution failed!")
        sys.exit(1)
    print(f"      Benchmark results saved to {csv_file}.")

def plot_results(csv_file="simd_benchmark_results.csv"):
    print("[3/4] Generating performance and speedup plots...")
    df = pd.read_csv(csv_file)
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })

    # Plot 1: Speedup vs Matrix Size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    kernel_sizes = sorted(df['K'].unique())
    simd_variants = ["SIMD-128 (SSE)", "SIMD-256 (AVX2)"]
    colors = {"SIMD-128 (SSE)": "#1f77b4", "SIMD-256 (AVX2)": "#d62728"}
    markers = {"SIMD-128 (SSE)": "o", "SIMD-256 (AVX2)": "s"}

    for idx, K in enumerate(kernel_sizes):
        ax = axes[idx]
        df_k = df[df['K'] == K]
        
        # Horizontal reference lines for theoretical max speedups
        ax.axhline(y=4.0, color='#1f77b4', linestyle='--', alpha=0.5, label='Theoretical 4x (128-bit)')
        ax.axhline(y=8.0, color='#d62728', linestyle='--', alpha=0.5, label='Theoretical 8x (256-bit)')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Scalar Baseline (1x)')

        for var in simd_variants:
            df_var = df_k[df_k['variant'] == var]
            ax.plot(df_var['matrix_size'], df_var['speedup'],
                    marker=markers[var], markersize=8, linewidth=2.5,
                    color=colors[var], label=var)

        ax.set_xscale('log', base=2)
        ax.set_xticks(df['matrix_size'].unique())
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Matrix Size (N x N)")
        ax.set_ylabel("Speedup over Scalar Naive")
        ax.set_title(f"Speedup vs Matrix Size (Kernel {K}x{K})")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend(loc='lower right' if idx==1 else 'upper left')
        ax.set_ylim(0, 10)

    plt.suptitle("SIMD Speedup across Matrix Sizes (128-bit vs 256-bit SIMD)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("simd_speedup_vs_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("      Saved: simd_speedup_vs_size.png")

    # Plot 2: Throughput (GFLOP/s) vs Matrix Size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    all_variants = ["Scalar (Naive)", "SIMD-128 (SSE)", "SIMD-256 (AVX2)"]
    all_colors = {"Scalar (Naive)": "#7f7f7f", "SIMD-128 (SSE)": "#1f77b4", "SIMD-256 (AVX2)": "#d62728"}
    all_markers = {"Scalar (Naive)": "^", "SIMD-128 (SSE)": "o", "SIMD-256 (AVX2)": "s"}

    for idx, K in enumerate(kernel_sizes):
        ax = axes[idx]
        df_k = df[df['K'] == K]

        for var in all_variants:
            df_var = df_k[df_k['variant'] == var]
            ax.plot(df_var['matrix_size'], df_var['gflops'],
                    marker=all_markers[var], markersize=8, linewidth=2.2,
                    color=all_colors[var], label=var)

        ax.set_xscale('log', base=2)
        ax.set_xticks(df['matrix_size'].unique())
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Matrix Size (N x N)")
        ax.set_ylabel("Throughput (GFLOP/s)")
        ax.set_title(f"Compute Throughput (Kernel {K}x{K})")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend(loc='best')

    plt.suptitle("Compute Throughput (GFLOP/s) vs Matrix Size", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("simd_gflops_vs_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("      Saved: simd_gflops_vs_size.png")

    # Plot 3: Execution Time vs Matrix Size (Log-Log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, K in enumerate(kernel_sizes):
        ax = axes[idx]
        df_k = df[df['K'] == K]

        for var in all_variants:
            df_var = df_k[df_k['variant'] == var]
            ax.plot(df_var['matrix_size'], df_var['time_ms'],
                    marker=all_markers[var], markersize=8, linewidth=2.2,
                    color=all_colors[var], label=var)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        ax.set_xticks(df['matrix_size'].unique())
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Matrix Size (N x N)")
        ax.set_ylabel("Execution Time (ms, log scale)")
        ax.set_title(f"Execution Time (Kernel {K}x{K})")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend(loc='upper left')

    plt.suptitle("Execution Time Scaling with Matrix Size (Log-Log)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("simd_execution_time_vs_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("      Saved: simd_execution_time_vs_size.png")

def print_summary(csv_file="simd_benchmark_results.csv"):
    print("\n[4/4] Consolidated Benchmark Summary:")
    df = pd.read_csv(csv_file)
    for K in sorted(df['K'].unique()):
        print(f"\n==================== Kernel K = {K}x{K} ====================")
        sub = df[df['K'] == K][['matrix_size', 'variant', 'time_ms', 'gflops', 'speedup', 'correct']]
        pvt = sub.pivot(index='matrix_size', columns='variant', values=['time_ms', 'gflops', 'speedup'])
        print(sub.to_string(index=False))

if __name__ == "__main__":
    build_benchmark()
    run_benchmark()
    plot_results()
    print_summary()
