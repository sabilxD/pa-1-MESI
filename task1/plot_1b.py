import csv
import matplotlib.pyplot as plt
import numpy as np

csv_file = "1b_comprehensive_results.csv"

# Load data
data = {}
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        K = int(row["K"])
        N = int(row["N"])
        cfg = row["tile_config"]
        if K not in data:
            data[K] = {}
        if cfg not in data[K]:
            data[K][cfg] = {}
        data[K][cfg][N] = {
            "time_ms": float(row["time_ms"]),
            "gflops": float(row["gflops"]),
            "speedup": float(row["speedup"]),
            "instructions": int(row["instructions"]),
            "cycles": int(row["cycles"]),
            "ipc": float(row["ipc"]),
            "l1_loads": int(row["l1_loads"]),
            "l1_misses": int(row["l1_misses"]),
            "l1_miss_rate_pct": float(row["l1_miss_rate_pct"]),
            "l1_mpki": float(row["l1_mpki"]),
        }

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

kernels = sorted(list(data.keys()))
all_cfgs = list(data[kernels[0]].keys())
# Select representative tile configurations for clean, readable plots
selected_cfgs = [
    "tile_16x16", "tile_32x32", "tile_64x64", "tile_128x128",
    "tile_32x128", "tile_16x256" ,"tile_64x256"
]
sizes = sorted(list(data[kernels[0]]["naive"].keys()))

# Dynamic colors and markers based on length of selected_cfgs
marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
colors = [plt.cm.tab10(i % 10) for i in range(len(selected_cfgs))]
markers = [marker_list[i % len(marker_list)] for i in range(len(selected_cfgs))]
cfg_color = {c: colors[i] for i, c in enumerate(selected_cfgs)}
cfg_marker = {c: markers[i] for i, c in enumerate(selected_cfgs)}

# ==============================================================================
# Plot 1: Speedup vs Matrix Size (K=3 and K=9)
# ==============================================================================
for K in kernels:
    plt.figure(figsize=(7.5, 4.8))
    for cfg in selected_cfgs:
        sp = [data[K][cfg][N]["speedup"] for N in sizes]
        label_name = cfg.replace("tile_", "Tile ").replace("x", r"$\times$")
        plt.plot(sizes, sp, marker=cfg_marker[cfg], linewidth=2, markersize=6,
                 color=cfg_color[cfg], label=label_name)

    plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Naive Baseline (1.0x)')
    plt.xscale('log', base=2)
    plt.xticks(sizes, [str(s) for s in sizes])
    plt.xlabel("Matrix Dimension N (N x N)", fontweight='bold')
    plt.ylabel("Speedup vs. Naive", fontweight='bold')
    plt.title(f"Task 1B: Tiling Speedup vs. Matrix Size (K={K}x{K})", fontweight='bold')
    plt.ylim(0.75, 1.25)
    plt.legend(ncol=3, frameon=True, fontsize=9)
    plt.savefig(f"figures/1b_speedup_matrix_k{K}.png", dpi=300)
    plt.close()

# ==============================================================================
# Plot 2: L1-D MPKI vs Matrix Size (K=3 and K=9)
# ==============================================================================
for K in kernels:
    plt.figure(figsize=(7.5, 4.8))
    mpki_naive = [data[K]["naive"][N]["l1_mpki"] for N in sizes]
    plt.plot(sizes, mpki_naive, 'k--', marker='o', linewidth=2.5, markersize=7, label='Naive Baseline')

    for cfg in selected_cfgs:
        mpki = [data[K][cfg][N]["l1_mpki"] for N in sizes]
        label_name = cfg.replace("tile_", "Tile ").replace("x", r"$\times$")
        plt.plot(sizes, mpki, marker=cfg_marker[cfg], linewidth=1.8, markersize=6,
                 color=cfg_color[cfg], label=label_name)

    plt.xscale('log', base=2)
    plt.xticks(sizes, [str(s) for s in sizes])
    plt.xlabel("Matrix Dimension N (N x N)", fontweight='bold')
    plt.ylabel("L1-D Cache MPKI (Misses / 1000 Inst)", fontweight='bold')
    plt.title(f"Task 1B: L1-D MPKI vs. Matrix Size (K={K}x{K})", fontweight='bold')
    plt.legend(ncol=3, frameon=True, fontsize=9)
    plt.savefig(f"figures/1b_l1_mpki_matrix_k{K}.png", dpi=300)
    plt.close()

# ==============================================================================
# Plot 3: L1-D Miss Rate (%) Comparison across Tile Sizes (N=2048, K=3 and K=9)
# ==============================================================================
plt.figure(figsize=(8, 4.8))
eval_cfgs = ["naive"] + selected_cfgs
x = np.arange(len(eval_cfgs))
width = 0.35

mr_k3 = [data[3][c][2048]["l1_mpki"] for c in eval_cfgs]
mr_k9 = [data[9][c][2048]["l1_mpki"] for c in eval_cfgs]

labels = [c.replace("tile_", "").replace("x", r"$\times$") for c in eval_cfgs]

plt.bar(x - width/2, mr_k3, width, label='K=3 (N=2048)', color='#1f77b4')
plt.bar(x + width/2, mr_k9, width, label='K=9 (N=2048)', color='#ff7f0e')

plt.xticks(x, labels, rotation=25)
plt.xlabel("Tile Configuration (TY x TX)", fontweight='bold')
plt.ylabel("L1-D MPKI", fontweight='bold')
plt.title("Task 1B: L1-D Load MPKI Across Tile Geometries (N=2048)", fontweight='bold')
plt.legend(frameon=True)
plt.savefig("figures/1b_l1_miss_rate_comparison.png", dpi=300)
plt.close()

print("All Task 1B plots generated successfully in figures/")
