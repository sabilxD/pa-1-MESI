import csv
import matplotlib.pyplot as plt
import numpy as np

csv_file = "1a_comprehensive_results.csv"

data = {}
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        K = int(row["K"])
        N = int(row["N"])
        stage = row["stage"]
        if K not in data:
            data[K] = {}
        if stage not in data[K]:
            data[K][stage] = {}
        data[K][stage][N] = {
            "time_ms": float(row["time_ms"]),
            "gflops": float(row["gflops"]),
            "speedup": float(row["speedup"]),
            "instructions": int(row["instructions"]),
            "cycles": int(row["cycles"]),
            "ipc": float(row["ipc"]),
            "branches": int(row["branches"]),
            "branch_misses": int(row["branch_misses"]),
            "branch_mpki": float(row["branch_mpki"]),
            "l1_mpki": float(row["l1_mpki"]),
        }

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# Infer kernels and sizes dynamically from data
kernels = sorted(list(data.keys()))
sample_stage = next(iter(data[kernels[0]].keys()))
sizes = sorted(list(data[kernels[0]][sample_stage].keys()))

# Dynamically generate color palettes and markers based on length of lists
marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
colors_k = {k: plt.cm.tab10(i % 10) for i, k in enumerate(kernels)}
markers_k = {k: marker_list[i % len(marker_list)] for i, k in enumerate(kernels)}

colors_n = {s: plt.cm.viridis(val) for s, val in zip(sizes, np.linspace(0.1, 0.85, len(sizes)))}
markers_n = {s: marker_list[i % len(marker_list)] for i, s in enumerate(sizes)}

# ==============================================================================
# REORDER PLOTS
# ==============================================================================

# Plot 1: Reorder Speedup vs Matrix Size
plt.figure(figsize=(7, 4.5))
for K in kernels:
    sp = [data[K]['reorder'][N]["speedup"] for N in sizes]
    plt.plot(sizes, sp, marker=markers_k[K], linewidth=2, markersize=6, color=colors_k[K], label=f"K={K}x{K}")

plt.axhline(1.0, color='gray', linestyle=':', label='Naive Baseline (1.0x)')
plt.xscale('log', base=2)
plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("Matrix Dimension N (N x N)", fontweight='bold')
plt.ylabel("Speedup vs Naive", fontweight='bold')
plt.title("Loop Reordering: Speedup vs. Matrix Size", fontweight='bold')
plt.ylim(0.8, 2.2)
plt.legend(ncol=3, frameon=True)
plt.savefig("figures/1a_reorder_speedup_matrix.png", dpi=300)
plt.close()

# Plot 2: Reorder Speedup vs Kernel Size
plt.figure(figsize=(7, 4.5))
for N in sizes:
    sp = [data[K]['reorder'][N]["speedup"] for K in kernels]
    plt.plot(kernels, sp, marker=markers_n[N], linewidth=2, markersize=6, color=colors_n[N], label=f"N={N}")

plt.axhline(1.0, color='gray', linestyle=':', label='Naive Baseline (1.0x)')
plt.xticks(kernels, [f"{K}x{K}" for K in kernels])
plt.xlabel("Kernel Dimension K (K x K)", fontweight='bold')
plt.ylabel("Speedup vs Naive", fontweight='bold')
plt.title("Loop Reordering: Speedup vs. Kernel Size K", fontweight='bold')
plt.ylim(0.8, 2.2)
plt.legend(ncol=3, frameon=True)
plt.savefig("figures/1a_reorder_speedup_kernel.png", dpi=300)
plt.close()

# ==============================================================================
# UNROLL PLOTS
# ==============================================================================

# Plot 3: Unroll Speedup vs Matrix Size
plt.figure(figsize=(7, 4.5))
for K in kernels:
    sp = [data[K]['unroll'][N]["speedup"] for N in sizes]
    plt.plot(sizes, sp, marker=markers_k[K], linewidth=2, markersize=6, color=colors_k[K], label=f"K={K}x{K}")

plt.axhline(1.0, color='gray', linestyle=':', label='Naive Baseline (1.0x)')
plt.xscale('log', base=2)
plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("Matrix Dimension N (N x N)", fontweight='bold')
plt.ylabel("Speedup vs Naive", fontweight='bold')
plt.title("Loop Unrolling (8x): Speedup vs. Matrix Size", fontweight='bold')
plt.ylim(1.0, 4.2)
plt.legend(ncol=3, frameon=True)
plt.savefig("figures/1a_unroll_speedup_matrix.png", dpi=300)
plt.close()

# Plot 4: Unroll Speedup vs Kernel Size
plt.figure(figsize=(7, 4.5))
for N in sizes:
    sp = [data[K]['unroll'][N]["speedup"] for K in kernels]
    plt.plot(kernels, sp, marker=markers_n[N], linewidth=2, markersize=6, color=colors_n[N], label=f"N={N}")

plt.axhline(1.0, color='gray', linestyle=':', label='Naive Baseline (1.0x)')
plt.xticks(kernels, [f"{K}x{K}" for K in kernels])
plt.xlabel("Kernel Dimension K (K x K)", fontweight='bold')
plt.ylabel("Speedup vs Naive", fontweight='bold')
plt.title("Loop Unrolling (8x): Speedup vs. Kernel Size K", fontweight='bold')
plt.ylim(1.0, 4.2)
plt.legend(ncol=3, frameon=True)
plt.savefig("figures/1a_unroll_speedup_kernel.png", dpi=300)
plt.close()

# Plot 5: Instructions (Millions) Comparison at K=3 (Naive vs Unroll)
plt.figure(figsize=(7, 4.5))
eval_sizes = sizes
x = np.arange(len(eval_sizes))
width = 0.35

inst_naive = [data[3]['naive'][N]["instructions"] / 1e6 for N in eval_sizes]
inst_unroll = [data[3]['unroll'][N]["instructions"] / 1e6 for N in eval_sizes]

plt.bar(x - width/2, inst_naive, width, label='Naive', color='#d62728')
plt.bar(x + width/2, inst_unroll, width, label='Unroll (8x)', color='#2ca02c')

for i in range(len(eval_sizes)):
    pct = (1.0 - inst_unroll[i] / inst_naive[i]) * 100
    plt.text(x[i], inst_naive[i]*1.05, f"-{pct:.1f}%", ha='center', fontweight='bold', fontsize=10)

plt.xticks(x, [str(s) for s in eval_sizes])
plt.yscale('log')
plt.xlabel("Matrix Dimension N (K=3)", fontweight='bold')
plt.ylabel("Instructions (Millions, Log Scale)", fontweight='bold')
plt.title("Loop Unrolling: 50% Reduction in Instructions", fontweight='bold')
plt.legend(frameon=True)
plt.savefig("figures/1a_unroll_instructions.png", dpi=300)
plt.close()

# Plot 6: Clock Cycles Comparison (K=3) (Naive vs Unroll)
plt.figure(figsize=(7, 4.5))
cyc_naive = [data[3]['naive'][N]["cycles"] / 1e6 for N in eval_sizes]
cyc_unroll = [data[3]['unroll'][N]["cycles"] / 1e6 for N in eval_sizes]

plt.plot(eval_sizes, cyc_naive, 'o-', label='Naive Cycles', color='#d62728', linewidth=2)
plt.plot(eval_sizes, cyc_unroll, 's-', label='Unroll (8x) Cycles', color='#2ca02c', linewidth=2)

plt.xscale('log', base=2)
plt.yscale('log')
plt.xticks(eval_sizes, [str(s) for s in eval_sizes])
plt.xlabel("Matrix Dimension N (K=3)", fontweight='bold')
plt.ylabel("Clock Cycles (Millions, Log Scale)", fontweight='bold')
plt.title("Loop Unrolling: Execution Clock Cycles Reduction", fontweight='bold')
plt.legend(frameon=True)
plt.savefig("figures/1a_unroll_cycles.png", dpi=300)
plt.close()

print("All separated plots generated successfully in figures/")
