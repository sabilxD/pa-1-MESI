#!/usr/bin/env python3
import subprocess
import re
import csv
import os
import sys

# Configurations
sizes = [256, 512, 1024, 2048, 4096]
kernels = [3, 9, 15, 21]
stages = ["reorder", "unroll", "tile", "simd", "optimized"]

events = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/branches/",
    "cpu_core/branch-misses/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/"
]
event_str = ",".join(events)

csv_file = "1d_comprehensive_results.csv"
fields = [
    "K", "N", "stage",
    "time_ms", "gflops", "speedup",
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

print("================================================================================")
print("Starting Task 1D Benchmark: Comparing All Stages vs Naive Baseline")
print(f"Matrix sizes: {sizes}")
print(f"Kernel sizes: {kernels}")
print(f"Stages: naive, {', '.join(stages)}")
print("================================================================================")

for K in kernels:
    for N in sizes:
        print(f"\n>>> Running K={K}, N={N} <<<")

        # 1. Naive baseline
        cmd_perf_naive = [
            "taskset", "-c", "0",
            "perf", "stat", "-e", event_str,
            "./bin/perf_run", "naive", str(N), str(N), str(K), "1"
        ]
        proc_pn = subprocess.run(cmd_perf_naive, capture_output=True, text=True, check=True)
        perf_naive = parse_perf_counters(proc_pn.stderr)

        cmd_conv_naive = ["taskset", "-c", "0", "./bin/conv", "naive", str(N), str(N), str(K)]
        proc_cn = subprocess.run(cmd_conv_naive, capture_output=True, text=True, check=True)
        m_naive = re.search(r'naive\s+\(ref\)\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_cn.stdout)
        naive_time = float(m_naive.group(1)) if m_naive else 0.0
        naive_gflops = float(m_naive.group(2)) if m_naive else 0.0

        naive_row = {
            "K": K, "N": N, "stage": "naive",
            "time_ms": naive_time, "gflops": naive_gflops, "speedup": 1.0,
            **perf_naive
        }
        results.append(naive_row)
        print(f"  [naive]     time={naive_time:8.3f} ms | speedup= 1.00x | inst={perf_naive['instructions']/1e6:8.2f}M | IPC={perf_naive['ipc']:4.2f}")

        # 2. Optimized stages
        for stage in stages:
            cmd_conv_stage = ["taskset", "-c", "0", "./bin/conv", stage, str(N), str(N), str(K)]
            proc_cs = subprocess.run(cmd_conv_stage, capture_output=True, text=True, check=True)
            m_stage = re.search(rf'{stage}\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_cs.stdout)
            s_time = float(m_stage.group(1)) if m_stage else 0.0
            s_gflops = float(m_stage.group(2)) if m_stage else 0.0
            s_speedup = (naive_time / s_time) if s_time > 0 else 1.0

            cmd_perf_stage = [
                "taskset", "-c", "0",
                "perf", "stat", "-e", event_str,
                "./bin/perf_run", stage, str(N), str(N), str(K), "1"
            ]
            proc_ps = subprocess.run(cmd_perf_stage, capture_output=True, text=True, check=True)
            perf_stage = parse_perf_counters(proc_ps.stderr)

            s_row = {
                "K": K, "N": N, "stage": stage,
                "time_ms": s_time, "gflops": s_gflops, "speedup": s_speedup,
                **perf_stage
            }
            results.append(s_row)
            print(f"  [{stage:9s}] time={s_time:8.3f} ms | speedup={s_speedup:5.2f}x | inst={perf_stage['instructions']/1e6:8.2f}M | IPC={perf_stage['ipc']:4.2f}")

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\n================================================================================")
print(f"Task 1D Benchmark Completed! Data successfully saved to {csv_file}")
print("================================================================================")
