#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

df = pd.read_csv("1d_comprehensive_results.csv")
os.makedirs("figures", exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14
})

stages = ["reorder", "unroll", "tile", "simd", "optimized"]
stage_labels = {
    "reorder": "conv_reorder",
    "unroll": "conv_unroll",
    "tile": "conv_tile",
    "simd": "conv_simd",
    "optimized": "conv_optimized"
}
colors = {
    "reorder": "#1f77b4",     # blue
    "unroll": "#ff7f0e",      # orange
    "tile": "#2ca02c",        # green
    "simd": "#9467bd",        # purple
    "optimized": "#d62728"    # bold red
}
markers = {
    "reorder": "o",
    "unroll": "s",
    "tile": "^",
    "simd": "D",
    "optimized": "*"
}

# ==============================================================================
# Graph 1: Speedup vs Matrix Size at K=3
# ==============================================================================
plt.figure(figsize=(7, 5.2))
df_k3 = df[df['K'] == 3]

plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Naive Baseline (1x)')

for st in stages:
    df_st = df_k3[df_k3['stage'] == st]
    msize = 11 if st == "optimized" else 7
    lw = 2.8 if st == "optimized" else 2.0
    plt.plot(df_st['N'], df_st['speedup'],
             marker=markers[st], markersize=msize, linewidth=lw,
             color=colors[st], label=stage_labels[st])

plt.xscale('log', base=2)
plt.xticks(df['N'].unique(), labels=[str(n) for n in df['N'].unique()])
plt.xlabel("Matrix Dimension $N$ ($N \\times N$)")
plt.ylabel("Speedup over Naive Baseline ($\\times$)")
plt.title("Speedup vs. Matrix Size ($K = 3$)")
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(loc='upper right', frameon=True)
plt.ylim(0, 16.5)
plt.tight_layout()
plt.savefig("figures/1d_speedup_matrix_k3.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/1d_speedup_matrix_k3.png")

# ==============================================================================
# Graph 2: Speedup vs Kernel Size at N=2048
# ==============================================================================
plt.figure(figsize=(7, 5.2))
df_n2048 = df[df['N'] == 2048]

plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Naive Baseline (1x)')

for st in stages:
    df_st = df_n2048[df_n2048['stage'] == st]
    msize = 11 if st == "optimized" else 7
    lw = 2.8 if st == "optimized" else 2.0
    plt.plot(df_st['K'], df_st['speedup'],
             marker=markers[st], markersize=msize, linewidth=lw,
             color=colors[st], label=stage_labels[st])

plt.xticks(df['K'].unique(), labels=[f"{k}x{k}" for k in df['K'].unique()])
plt.xlabel("Kernel Size $K$ ($K \\times K$)")
plt.ylabel("Speedup over Naive Baseline ($\\times$)")
plt.title("Speedup vs. Kernel Size ($N = 2048$)")
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(loc='upper left', frameon=True)
plt.ylim(0, 16.5)
plt.tight_layout()
plt.savefig("figures/1d_speedup_kernel_n2048.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/1d_speedup_kernel_n2048.png")

# Also generate the combined side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

# Subplot (a): Speedup vs Matrix Size (K=3)
ax = axes[0]
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Naive Baseline (1x)')
for st in stages:
    df_st = df_k3[df_k3['stage'] == st]
    msize = 11 if st == "optimized" else 7
    lw = 2.8 if st == "optimized" else 2.0
    ax.plot(df_st['N'], df_st['speedup'],
            marker=markers[st], markersize=msize, linewidth=lw,
            color=colors[st], label=stage_labels[st])
ax.set_xscale('log', base=2)
ax.set_xticks(df['N'].unique())
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("Matrix Dimension $N$ ($N \\times N$)")
ax.set_ylabel("Speedup over  Naive ($\\times$)")
ax.set_title("(a) Speedup vs. Matrix Size ($K = 3$)")
ax.grid(True, which="both", ls="-", alpha=0.4)
ax.legend(loc='upper right', frameon=True)
ax.set_ylim(0, 16.5)

# Subplot (b): Speedup vs Kernel Size (N=2048)
ax = axes[1]
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Naive Baseline (1x)')
for st in stages:
    df_st = df_n2048[df_n2048['stage'] == st]
    msize = 11 if st == "optimized" else 7
    lw = 2.8 if st == "optimized" else 2.0
    ax.plot(df_st['K'], df_st['speedup'],
            marker=markers[st], markersize=msize, linewidth=lw,
            color=colors[st], label=stage_labels[st])
ax.set_xticks(df['K'].unique())
ax.set_xlabel("Kernel Dimension $K$ ($K \\times K$)")
ax.set_title("(b) Speedup vs. Kernel Size ($N = 2048$)")
ax.grid(True, which="both", ls="-", alpha=0.4)
ax.legend(loc='upper left', frameon=True)

plt.suptitle("Task 1D: Holistic Performance Evaluation of Convolution Optimization Stages", y=1.02)
plt.tight_layout()
plt.savefig("figures/1d_all_speedups.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/1d_all_speedups.png")
