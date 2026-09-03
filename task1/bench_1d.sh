#!/usr/bin/env bash
# bench_1d.sh - CS683 PA-1, Task 1B: L1-D cache behaviour of tiled 2D convolution.
#
# Produces every number the task asks for:
#   1. L1-D cache size, and the naive baseline's L1-D MPKI
#   2. MPKI + speedup for a sweep of tile sizes
#   3. MPKI + speedup vs MATRIX SIZE, for several tile sizes
#   4. MPKI + speedup vs KERNEL SIZE K (where the reuse window outgrows L1)
#
# Usage:
#   ./bench_1d.sh              # everything (~5 min)
#   ./bench_1d.sh tiles        # just the tile-size sweep
#   ./bench_1d.sh sizes        # just the matrix-size sweep
#   ./bench_1d.sh kernels      # just the K sweep
#   REPS=5 ./bench_1d.sh       # more repetitions (median is reported)
#
# Requires perf. If it refuses, run ONE of:
#   sudo sysctl -w kernel.perf_event_paranoid=1     # then re-run this normally
#   sudo ./bench_1d.sh                              # or just run it as root
#
# Your src/ is never modified: everything builds in a throwaway copy under /tmp.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CPU="${CPU:-0}"          # pin to one core; core 0 is a P-core on hybrid Intel parts
REPS="${REPS:-3}"
WHAT="${1:-all}"
OUT="${OUT:-$HERE}"

# ---------------------------------------------------------------- environment
# Every one of these checks exists because ignoring it produced wrong numbers.
env_report() {
    local BAD=0
    echo "============================================================"
    echo " ENVIRONMENT"
    echo "============================================================"

    local MODEL L1D L1WAY
    MODEL=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
    L1D=$(lscpu -C 2>/dev/null | awk '$1=="L1d"{print $2; exit}')
    echo "  CPU                : ${MODEL:-unknown}"
    echo "  L1-D cache (1 core): ${L1D:-unknown}"
    echo "  L2 / L3            : $(lscpu -C 2>/dev/null | awk '$1=="L2"{printf "%s  ", $2} $1=="L3"{print $2}')"
    echo "  pinned to CPU      : $CPU"

    # perf availability -- without it there are no cache numbers at all.
    if ! command -v perf >/dev/null; then
        echo "  perf               : NOT INSTALLED   <-- install linux-tools; cannot continue"; BAD=1
    elif ! perf stat -e instructions true >/dev/null 2>&1; then
        echo "  perf               : BLOCKED (perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null))"
        echo "                       fix: sudo sysctl -w kernel.perf_event_paranoid=1   (or run this script with sudo)"; BAD=1
    else
        echo "  perf               : OK"
    fi

    # Frequency scaling. A ramping clock silently doubled our speedup ratios once.
    local GOV
    GOV=$(cat /sys/devices/system/cpu/cpu$CPU/cpufreq/scaling_governor 2>/dev/null)
    if [ "${GOV:-}" = "performance" ]; then
        echo "  cpufreq governor   : performance  OK"
    else
        echo "  cpufreq governor   : ${GOV:-unknown}   <-- WARNING, timings will drift"
        echo "                       fix: sudo cpupower frequency-set -g performance"; BAD=1
    fi

    # Laptops: on battery the core sat at 800 MHz against a 4.8 GHz max.
    local AC; AC=$(cat /sys/class/power_supply/A[CD]*/online 2>/dev/null | head -1)
    [ "${AC:-1}" = "0" ] && { echo "  power              : ON BATTERY  <-- WARNING, plug in"; BAD=1; } \
                         || echo "  power              : AC connected  OK"

    local PP; PP=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)
    if [ -n "${PP:-}" ] && [ "$PP" != "performance" ]; then
        echo "  platform_profile   : $PP   <-- limits turbo headroom"
        echo "                       fix: echo performance | sudo tee /sys/firmware/acpi/platform_profile"
    fi

    # A busy background process steals memory bandwidth and skews memory-bound stages.
    local BUSY
    BUSY=$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $1>20 {print $2"("$1"%)"}' | head -3 | tr '\n' ' ')
    [ -n "$BUSY" ] && { echo "  background load    : $BUSY  <-- WARNING, stop these"; BAD=1; } \
                   || echo "  background load    : quiet  OK"

    # Hardware prefetchers change cache-miss counts; just report what they are.
    if command -v rdmsr >/dev/null && [ "$(id -u)" -eq 0 ]; then
        local M; M=$(rdmsr -p "$CPU" 0x1a4 2>/dev/null)
        [ -n "${M:-}" ] && echo "  HW prefetchers     : MSR 0x1a4 = 0x$M  ($([ "$M" = "0" ] && echo 'all ENABLED' || echo 'some DISABLED'))"
    else
        echo "  HW prefetchers     : not checked (needs root + msr-tools); assumed enabled"
    fi

    echo
    [ "$BAD" -ne 0 ] && echo "  >>> Some checks failed. Numbers will be noisy; fix them for report-quality data." && echo
    return 0
}

# ---------------------------------------------------------------- perf helpers
# Hybrid Intel CPUs split counters into cpu_core/... and cpu_atom/...; prefer cpu_core.
ctr() {
    local EV="$1" F="$2" V
    V=$(grep -E "cpu_core/$EV/" "$F" 2>/dev/null | awk '{gsub(/,/,"",$1); if($1+0>0){print $1; exit}}')
    [ -z "$V" ] && V=$(grep -E "[[:space:]]$EV([[:space:]]|$)" "$F" 2>/dev/null \
                       | grep -vE 'cpu_atom|not counted|not supported' \
                       | awk '{gsub(/,/,"",$1); if($1+0>0){print $1; exit}}')
    echo "${V:-0}"
}
med() { sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.3f", v[int((NR+1)/2)]; else printf "NA"}'; }

# measure <stage> <H> <W> <K>  ->  sets MS SPD HIT MISS MPKI (stage-attributed)
# main.cpp always runs naive as the baseline, so "./conv tile" executes naive AND tile.
# We profile naive alone too, and subtract, to attribute counters to the stage itself.
measure() {
    local STAGE="$1" H="$2" W="$3" K="$4" D
    perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses \
        taskset -c "$CPU" ./bin/conv naive "$H" "$W" "$K" > "$TMP/n.txt" 2>&1
    local NI NL NM; NI=$(ctr instructions "$TMP/n.txt"); NL=$(ctr L1-dcache-loads "$TMP/n.txt"); NM=$(ctr L1-dcache-load-misses "$TMP/n.txt")

    if [ "$STAGE" = "naive" ]; then
        read HIT MISS MPKI <<<"$(awk -v l="$NL" -v m="$NM" -v i="$NI" 'BEGIN{
            if(l>0)printf "%.4f %.4f ",(l-m)/l*100,m/l*100; else printf "NA NA ";
            if(i>0)printf "%.4f",m/i*1000; else printf "NA"}')"
    else
        perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses \
            taskset -c "$CPU" ./bin/conv "$STAGE" "$H" "$W" "$K" > "$TMP/t.txt" 2>&1
        local TI TL TM; TI=$(ctr instructions "$TMP/t.txt"); TL=$(ctr L1-dcache-loads "$TMP/t.txt"); TM=$(ctr L1-dcache-load-misses "$TMP/t.txt")
        read HIT MISS MPKI <<<"$(awk -v tl="$TL" -v nl="$NL" -v tm="$TM" -v nm="$NM" -v ti="$TI" -v ni="$NI" 'BEGIN{
            dl=(tl-nl)/10; dm=(tm-nm)/10; di=(ti-ni)/10;
            if(dl>0)printf "%.4f %.4f ",(dl-dm)/dl*100,dm/dl*100; else printf "NA NA ";
            if(di>0)printf "%.4f",dm/di*1000; else printf "NA"}')"
    fi

    D=$(mktemp -d)
    for r in $(seq 1 "$REPS"); do taskset -c "$CPU" ./bin/conv "$STAGE" "$H" "$W" "$K" > "$D/$r" 2>&1; done
    MS=$(cat "$D"/* | awk -v s="^$STAGE" '$0~s && $NF ~ /^[0-9.]+x$/ {print $(NF-2)}' | med)
    SPD=$(cat "$D"/* | awk -v s="^$STAGE" '$0~s && $NF ~ /^[0-9.]+x$/ {sub(/x$/,"",$NF); print $NF}' | med)
    rm -rf "$D"
}

set_tile() { sed -i "s/conv_tiled(in, out, ker, H, W, K, [0-9]*, [0-9]*)/conv_tiled(in, out, ker, H, W, K, $1, $1)/" src/conv_tile.cpp; make >/dev/null 2>&1; }

# ---------------------------------------------------------------- setup
env_report
TMP=$(mktemp -d); trap 'rm -rf "$TMP" "$WORK"' EXIT
WORK=$(mktemp -d); cp -r "$HERE" "$WORK/t1"; cd "$WORK/t1"; make >/dev/null 2>&1 || { echo "build failed"; exit 1; }

# ---------------------------------------------------------------- 1. baseline
if [ "$WHAT" = all ]; then
    echo "============================================================"
    echo " 1. BASELINE  (naive, 2048x2048, K=3)"
    echo "============================================================"
    measure naive 2048 2048 3
    printf "  time %s ms   L1-D hit %s%%   miss %s%%   MPKI %s\n\n" "$MS" "$HIT" "$MISS" "$MPKI"
fi

# ---------------------------------------------------------------- 2. tile size
if [ "$WHAT" = all ] || [ "$WHAT" = tiles ]; then
    echo "============================================================"
    echo " 2. TILE SIZE SWEEP  (2048x2048, K=3)"
    echo "============================================================"
    C="$OUT/1d_tiles.csv"; echo "tile,time_ms,speedup,hit_rate_pct,miss_rate_pct,mpki" > "$C"
    printf "  %-10s %10s %9s %11s %9s\n" tile time_ms speedup hit_rate mpki
    measure naive 2048 2048 3
    printf "  %-10s %10s %9s %10s%% %9s\n" "naive" "$MS" "1.000x" "$HIT" "$MPKI"
    echo "naive,$MS,1.000,$HIT,$MISS,$MPKI" >> "$C"
    for T in 4 8 16 32 64 128 256 512 1024 2048; do
        set_tile "$T" || continue
        measure tile 2048 2048 3
        printf "  %-10s %10s %8sx %10s%% %9s\n" "${T}x${T}" "$MS" "$SPD" "$HIT" "$MPKI"
        echo "${T}x${T},$MS,$SPD,$HIT,$MISS,$MPKI" >> "$C"
    done
    echo "  -> $C"; echo
fi

# ---------------------------------------------------------------- 3. matrix size
if [ "$WHAT" = all ] || [ "$WHAT" = sizes ]; then
    echo "============================================================"
    echo " 3. MATRIX SIZE SWEEP  (K=3)"
    echo "============================================================"
    C="$OUT/1d_sizes.csv"; echo "size,tile,time_ms,speedup,hit_rate_pct,miss_rate_pct,mpki" > "$C"
    printf "  %-7s %-10s %10s %9s %11s %9s\n" size tile time_ms speedup hit_rate mpki
    for N in 512 1024 2048 4096; do
        measure naive "$N" "$N" 3
        printf "  %-7s %-10s %10s %9s %10s%% %9s\n" "$N" "naive" "$MS" "1.000x" "$HIT" "$MPKI"
        echo "$N,naive,$MS,1.000,$HIT,$MISS,$MPKI" >> "$C"
        for T in 32 128 512; do
            set_tile "$T" || continue
            measure tile "$N" "$N" 3
            printf "  %-7s %-10s %10s %8sx %10s%% %9s\n" "$N" "${T}x${T}" "$MS" "$SPD" "$HIT" "$MPKI"
            echo "$N,${T}x${T},$MS,$SPD,$HIT,$MISS,$MPKI" >> "$C"
        done
    done
    echo "  -> $C"; echo
fi

# ---------------------------------------------------------------- 4. kernel size
if [ "$WHAT" = all ] || [ "$WHAT" = kernels ]; then
    echo "============================================================"
    echo " 4. KERNEL SIZE SWEEP  (2048x2048, tile 128x128)"
    echo "    naive reuse window = K*(W+K-1)*4 bytes; it outgrows L1 around K=7"
    echo "============================================================"
    C="$OUT/1d_kernels.csv"; echo "K,stage,time_ms,speedup,hit_rate_pct,miss_rate_pct,mpki,window_KB" > "$C"
    printf "  %-4s %-10s %10s %9s %11s %9s %10s\n" K stage time_ms speedup hit_rate mpki window
    for K in 3 5 7 9 11 15; do
        WKB=$(awk -v k="$K" 'BEGIN{printf "%.1f", k*(2048+k-1)*4/1024}')
        measure naive 2048 2048 "$K"
        printf "  %-4s %-10s %10s %9s %10s%% %9s %8sKB\n" "$K" "naive" "$MS" "1.000x" "$HIT" "$MPKI" "$WKB"
        echo "$K,naive,$MS,1.000,$HIT,$MISS,$MPKI,$WKB" >> "$C"
        set_tile 128
        measure tile 2048 2048 "$K"
        printf "  %-4s %-10s %10s %8sx %10s%% %9s %8sKB\n" "$K" "tile128" "$MS" "$SPD" "$HIT" "$MPKI" "$WKB"
        echo "$K,tile128,$MS,$SPD,$HIT,$MISS,$MPKI,$WKB" >> "$C"
    done
    echo "  -> $C"; echo
fi

echo "Done."
