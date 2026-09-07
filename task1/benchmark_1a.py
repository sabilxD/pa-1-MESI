import subprocess
import re
import csv
import sys

# Configurations
sizes = [256, 512, 1024, 2048, 4096]
kernels = [3, 9, 15, 21]
stages = ['reorder', 'unroll']

events = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/branches/",
    "cpu_core/branch-misses/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/"
]
event_str = ",".join(events)

csv_file = "1a_comprehensive_results.csv"
fields = [
    "K", "N", "stage", "time_ms", "gflops", "speedup",
    "instructions", "cycles", "ipc",
    "branches", "branch_misses", "branch_miss_rate_pct", "branch_mpki",
    "l1_loads", "l1_misses", "l1_mpki"
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
    l1_mpki = (l1_misses / instructions * 1000.0) if instructions > 0 else 0.0

    return {
        "instructions": instructions, "cycles": cycles, "ipc": ipc,
        "branches": branches, "branch_misses": branch_misses,
        "branch_miss_rate_pct": branch_miss_rate, "branch_mpki": branch_mpki,
        "l1_loads": l1_loads, "l1_misses": l1_misses, "l1_mpki": l1_mpki
    }

print("Starting Task 1A Benchmark (Timing from ./bin/conv, Counters from ./bin/perf_run)...")
print(f"Matrix sizes: {sizes}")
print(f"Kernel sizes: {kernels}")
print("-" * 80)

for K in kernels:
    for N in sizes:
        print(f"\n--- Running K={K}, N={N} ---")
        
        # 1. Measure naive hardware counters via perf_run
        cmd_perf_naive = [
            "taskset", "-c", "0",
            "perf", "stat", "-e", event_str,
            "./bin/perf_run", "naive", str(N), str(N), str(K), "1"
        ]
        proc_pn = subprocess.run(cmd_perf_naive, capture_output=True, text=True, check=True)
        perf_naive = parse_perf_counters(proc_pn.stderr)

        naive_recorded = False

        for stage in stages:
            # 2. Measure official timing & speedup via ./bin/conv
            cmd_conv = [
                "taskset", "-c", "0",
                "./bin/conv", stage, str(N), str(N), str(K)
            ]
            proc_c = subprocess.run(cmd_conv, capture_output=True, text=True, check=True)
            conv_out = proc_c.stdout

            # Parse naive line and stage line
            # naive (ref)     yes          32.085        2.35      1.00x
            # reorder         yes          21.157        3.57      1.52x
            naive_match = re.search(r'naive\s+\(ref\)\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', conv_out)
            stage_match = re.search(rf'{stage}\s+\w+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)x', conv_out)

            if not naive_recorded and naive_match:
                n_time = float(naive_match.group(1))
                n_gflops = float(naive_match.group(2))
                n_row = {
                    "K": K, "N": N, "stage": "naive",
                    "time_ms": n_time, "gflops": n_gflops, "speedup": 1.0,
                    **perf_naive
                }
                results.append(n_row)
                naive_recorded = True
                print(f"  [naive]   time={n_time:8.3f} ms | speedup= 1.00x | IPC={perf_naive['ipc']:4.2f}")

            s_time = float(stage_match.group(1)) if stage_match else 0.0
            s_gflops = float(stage_match.group(2)) if stage_match else 0.0
            s_speedup = float(stage_match.group(3)) if stage_match else 1.0

            # 3. Measure stage hardware counters via perf_run
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
            print(f"  [{stage:7s}] time={s_time:8.3f} ms | speedup={s_speedup:5.2f}x | IPC={perf_stage['ipc']:4.2f} | BrMPKI={perf_stage['branch_mpki']:4.2f} | L1MPKI={perf_stage['l1_mpki']:4.2f}")

# Write to CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"\nAll benchmark runs finished! Saved to {csv_file}")
