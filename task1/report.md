# Task 1B

## 1. Profiling baseline 2D convolution code:
- L1D cache size : 48 KB for Performace Core
- Baseline Naive Performance report:
    - Command: `sudo taskset -c 0 perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses ./bin/conv naive`
    - Instructions: 4,541,823,417 
    - L1-dcache-loads: 890,208,677 
    - L1-dcache-load-misses: 4,695,032 
    - L1-D cache misses per kilo instructions (MPKI): 1.033

