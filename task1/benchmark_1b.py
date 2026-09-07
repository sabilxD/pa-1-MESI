import subprocess
import re
import csv
import os
import sys

# Configurations
sizes = [256, 512, 1024, 2048, 4096]
kernels = [3, 9]

tile_configs = [
    {"name": "tile_16x16",   "TY": 16,  "TX": 16},
    {"name": "tile_16x64",   "TY": 16,  "TX": 64},
    {"name": "tile_16x256",  "TY": 16,  "TX": 256},
    {"name": "tile_32x32",   "TY": 32,  "TX": 32},
    {"name": "tile_32x64",   "TY": 32,  "TX": 64},
    {"name": "tile_32x128",  "TY": 32,  "TX": 128},
    {"name": "tile_32x256",  "TY": 32,  "TX": 256},
    {"name": "tile_64x64",   "TY": 64,  "TX": 64},
    {"name": "tile_64x128",  "TY": 64,  "TX": 128},
    {"name": "tile_64x256",  "TY": 64,  "TX": 256},
    {"name": "tile_128x128", "TY": 128, "TX": 128},
]

events = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/branches/",
    "cpu_core/branch-misses/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/"
]
event_str = ",".join(events)

csv_file = "1b_comprehensive_results.csv"
fields = [
    "K", "N", "tile_config", "TY", "TX", "time_ms", "gflops", "speedup",
    "instructions", "cycles", "ipc",
    "branches", "branch_misses", "branch_miss_rate_pct", "branch_mpki",
    "l1_loads", "l1_misses", "l1_miss_rate_pct", "l1_mpki"
]

results = []

def parse_perf_counters(stderr):
    def parse_counter(pattern):
        m = re.search(rf'([0-9,]+)\s+{pattern}', stderr)
        if m:
            return int(m.group(1).replace(",", ""))
        return 0

    instructions = parse_counter(r'cpu_core/instructions/')
    cycles = parse_counter(r'cpu_core/cycles/')
    branches = parse_counter(r'cpu_core/branches/')
    branch_misses = parse_counter(r'cpu_core/branch-misses/')
    l1_loads = parse_counter(r'cpu_core/L1-dcache-loads/')
    l1_misses = parse_counter(r'cpu_core/L1-dcache-load-misses/')

    ipc = instructions / cycles if cycles > 0 else 0.0
    branch_miss_rate = (branch_misses / branches * 100.0) if branches > 0 else 0.0
    branch_mpki = (branch_misses / instructions * 1000.0) if instructions > 0 else 0.0
    l1_miss_rate = (l1_misses / l1_loads * 100.0) if l1_loads > 0 else 0.0
    l1_mpki = (l1_misses / instructions * 1000.0) if instructions > 0 else 0.0

    return {
        "instructions": instructions, "cycles": cycles, "ipc": ipc,
        "branches": branches, "branch_misses": branch_misses,
        "branch_miss_rate_pct": branch_miss_rate, "branch_mpki": branch_mpki,
        "l1_loads": l1_loads, "l1_misses": l1_misses,
        "l1_miss_rate_pct": l1_miss_rate, "l1_mpki": l1_mpki
    }

print("Starting Task 1B Tiling Benchmark...")
print(f"Matrix sizes: {sizes}")
print(f"Kernel sizes: {kernels}")
print(f"Tile configs: {[c['name'] for c in tile_configs]}")
print("-" * 80)

for K in kernels:
    for N in sizes:
        print(f"\n--- Running Workload K={K}, N={N} ---")

        # 1. Measure naive baseline hardware counters via perf_run
        cmd_perf_naive = [
            "taskset", "-c", "0",
            "perf", "stat", "-e", event_str,
            "./bin/perf_run", "naive", str(N), str(N), str(K), "1"
        ]
        proc_pn = subprocess.run(cmd_perf_naive, capture_output=True, text=True, check=True)
        perf_naive = parse_perf_counters(proc_pn.stderr)

        # 2. Measure official naive timing from ./bin/conv
        cmd_conv_naive = [
            "taskset", "-c", "0",
            "./bin/conv", "naive", str(N), str(N), str(K)
        ]
        proc_cn = subprocess.run(cmd_conv_naive, capture_output=True, text=True, check=True)
        naive_match = re.search(r'naive\s+\(ref\)\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_cn.stdout)
        n_time = float(naive_match.group(1)) if naive_match else 0.0
        n_gflops = float(naive_match.group(2)) if naive_match else 0.0

        n_row = {
            "K": K, "N": N, "tile_config": "naive", "TY": 0, "TX": 0,
            "time_ms": n_time, "gflops": n_gflops, "speedup": 1.0,
            **perf_naive
        }
        results.append(n_row)
        print(f"  [naive]        time={n_time:8.3f} ms | speedup= 1.00x | L1MPKI={perf_naive['l1_mpki']:5.2f} | L1MissRate={perf_naive['l1_miss_rate_pct']:5.2f}%")

        # 3. Sweep tile configurations
        for cfg in tile_configs:
            name = cfg["name"]
            ty, tx = cfg["TY"], cfg["TX"]

            env = os.environ.copy()
            env["PA1_TY"] = str(ty)
            env["PA1_TX"] = str(tx)

            # Official timing & speedup from ./bin/conv
            cmd_conv_tile = [
                "taskset", "-c", "0",
                "./bin/conv", "tile", str(N), str(N), str(K)
            ]
            proc_ct = subprocess.run(cmd_conv_tile, env=env, capture_output=True, text=True, check=True)
            tile_match = re.search(r'tile\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_ct.stdout)
            t_time = float(tile_match.group(1)) if tile_match else 0.0
            t_gflops = float(tile_match.group(2)) if tile_match else 0.0
            t_speedup = float(tile_match.group(3)) if tile_match else 1.0

            # Hardware counters from perf_run
            cmd_perf_tile = [
                "taskset", "-c", "0",
                "perf", "stat", "-e", event_str,
                "./bin/perf_run", "tile", str(N), str(N), str(K), "1"
            ]
            proc_pt = subprocess.run(cmd_perf_tile, env=env, capture_output=True, text=True, check=True)
            perf_tile = parse_perf_counters(proc_pt.stderr)

            t_row = {
                "K": K, "N": N, "tile_config": name, "TY": ty, "TX": tx,
                "time_ms": t_time, "gflops": t_gflops, "speedup": t_speedup,
                **perf_tile
            }
            results.append(t_row)
            print(f"  [{name:12s}] time={t_time:8.3f} ms | speedup={t_speedup:5.2f}x | L1MPKI={perf_tile['l1_mpki']:5.2f} | L1MissRate={perf_tile['l1_miss_rate_pct']:5.2f}%")

# Save to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\nTask 1B Benchmark complete! Data saved to {csv_file}")
