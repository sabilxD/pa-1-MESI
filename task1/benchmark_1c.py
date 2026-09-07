#!/usr/bin/env python3
import subprocess
import re
import csv
import os
import sys

# Configurations
sizes = [256, 512, 1024, 2048, 4096]
kernels = [3, 9]

events = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/branches/",
    "cpu_core/branch-misses/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/"
]
event_str = ",".join(events)

csv_file = "1c_comprehensive_results.csv"
fields = [
    "K", "N", "stage", "simd_bits", "vector_width",
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
print("Starting Task 1C Benchmark: SIMD-128 (SSE) & SIMD-256 (AVX2) vs Naive")
print(f"Matrix sizes: {sizes}")
print(f"Kernel sizes: {kernels}")
print("================================================================================")

for K in kernels:
    for N in sizes:
        print(f"\n>>> Running K={K}, N={N} <<<")

        # -------------------------------------------------------------
        # 1. Measure Naive Baseline
        # -------------------------------------------------------------
        cmd_conv_naive = ["taskset", "-c", "0", "./bin/conv", "naive", str(N), str(N), str(K)]
        proc_c = subprocess.run(cmd_conv_naive, capture_output=True, text=True, check=True)
        m_naive = re.search(r'naive\s+\(ref\)\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_c.stdout)
        naive_time = float(m_naive.group(1)) if m_naive else 0.0
        naive_gflops = float(m_naive.group(2)) if m_naive else 0.0

        cmd_perf_naive = [
            "taskset", "-c", "0",
            "perf", "stat", "-e", event_str,
            "./bin/perf_run", "naive", str(N), str(N), str(K), "1"
        ]
        proc_pn = subprocess.run(cmd_perf_naive, capture_output=True, text=True, check=True)
        perf_naive = parse_perf_counters(proc_pn.stderr)

        naive_row = {
            "K": K, "N": N, "stage": "naive", "simd_bits": 32, "vector_width": 1,
            "time_ms": naive_time, "gflops": naive_gflops, "speedup": 1.0,
            **perf_naive
        }
        results.append(naive_row)
        print(f"  [naive]   time={naive_time:8.3f} ms | speedup= 1.00x | inst={perf_naive['instructions']/1e6:8.2f}M | IPC={perf_naive['ipc']:4.2f}")

        # -------------------------------------------------------------
        # 2. Measure SIMD-128 (SSE)
        # -------------------------------------------------------------
        env_128 = os.environ.copy()
        env_128["PA1_SIMD_WIDTH"] = "128"

        cmd_conv_128 = ["taskset", "-c", "0", "./bin/conv", "simd", str(N), str(N), str(K)]
        proc_128 = subprocess.run(cmd_conv_128, capture_output=True, text=True, check=True, env=env_128)
        m_128 = re.search(r'simd\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_128.stdout)
        t_128 = float(m_128.group(1)) if m_128 else 0.0
        gf_128 = float(m_128.group(2)) if m_128 else 0.0
        sp_128 = (naive_time / t_128) if t_128 > 0 else 1.0

        cmd_perf_128 = [
            "taskset", "-c", "0",
            "perf", "stat", "-e", event_str,
            "./bin/perf_run", "simd", str(N), str(N), str(K), "1"
        ]
        proc_p128 = subprocess.run(cmd_perf_128, capture_output=True, text=True, check=True, env=env_128)
        perf_128 = parse_perf_counters(proc_p128.stderr)

        row_128 = {
            "K": K, "N": N, "stage": "simd128", "simd_bits": 128, "vector_width": 4,
            "time_ms": t_128, "gflops": gf_128, "speedup": sp_128,
            **perf_128
        }
        results.append(row_128)
        inst_drop_128 = (1.0 - perf_128['instructions'] / perf_naive['instructions']) * 100.0 if perf_naive['instructions'] > 0 else 0.0
        print(f"  [simd128] time={t_128:8.3f} ms | speedup={sp_128:5.2f}x | inst={perf_128['instructions']/1e6:8.2f}M ({inst_drop_128:+.1f}%) | IPC={perf_128['ipc']:4.2f}")

        # -------------------------------------------------------------
        # 3. Measure SIMD-256 (AVX2)
        # -------------------------------------------------------------
        env_256 = os.environ.copy()
        env_256["PA1_SIMD_WIDTH"] = "256"

        cmd_conv_256 = ["taskset", "-c", "0", "./bin/conv", "simd", str(N), str(N), str(K)]
        proc_256 = subprocess.run(cmd_conv_256, capture_output=True, text=True, check=True, env=env_256)
        m_256 = re.search(r'simd\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', proc_256.stdout)
        t_256 = float(m_256.group(1)) if m_256 else 0.0
        gf_256 = float(m_256.group(2)) if m_256 else 0.0
        sp_256 = (naive_time / t_256) if t_256 > 0 else 1.0

        cmd_perf_256 = [
            "taskset", "-c", "0",
            "perf", "stat", "-e", event_str,
            "./bin/perf_run", "simd", str(N), str(N), str(K), "1"
        ]
        proc_p256 = subprocess.run(cmd_perf_256, capture_output=True, text=True, check=True, env=env_256)
        perf_256 = parse_perf_counters(proc_p256.stderr)

        row_256 = {
            "K": K, "N": N, "stage": "simd256", "simd_bits": 256, "vector_width": 8,
            "time_ms": t_256, "gflops": gf_256, "speedup": sp_256,
            **perf_256
        }
        results.append(row_256)
        inst_drop_256 = (1.0 - perf_256['instructions'] / perf_naive['instructions']) * 100.0 if perf_naive['instructions'] > 0 else 0.0
        print(f"  [simd256] time={t_256:8.3f} ms | speedup={sp_256:5.2f}x | inst={perf_256['instructions']/1e6:8.2f}M ({inst_drop_256:+.1f}%) | IPC={perf_256['ipc']:4.2f}")

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\n================================================================================")
print(f"Task 1C Benchmarking Completed! Data successfully saved to {csv_file}")
print("================================================================================")
