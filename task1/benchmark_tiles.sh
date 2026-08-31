#!/usr/bin/env bash

set -euo pipefail

SRC="./src/conv_tile.cpp"
BINARY="./bin/conv"
CPU=0

TILE_SIZES=(4 8 16 32 64 128 256 512)

CSV="tile_results.csv"
RAW="tile_raw.log"

# ============================================================
# Check prerequisites
# ============================================================

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: Cannot find $SRC"
    exit 1
fi

if [[ ! -f "Makefile" && ! -f "makefile" && ! -f "GNUmakefile" ]]; then
    echo "ERROR: Cannot find Makefile"
    exit 1
fi

# ============================================================
# Prepare result files
# ============================================================

echo "tile,time_ms,gflops,speedup,instructions,l1_loads,l1_misses,mpki,miss_rate" > "$CSV"
: > "$RAW"

# ============================================================
# Change tile size in conv_tile.cpp
# ============================================================

set_tile_size() {
    local T="$1"

    python3 - "$SRC" "$T" <<'PY'
import sys
import re

filename = sys.argv[1]
tile = sys.argv[2]

with open(filename, "r") as f:
    text = f.read()

pattern = (
    r'(conv_tiled\s*\(\s*'
    r'in\s*,\s*out\s*,\s*ker\s*,\s*'
    r'H\s*,\s*W\s*,\s*K\s*,\s*)'
    r'\d+\s*,\s*\d+'
    r'(\s*\)\s*;)'
)

replacement = rf'\g<1>{tile}, {tile}\g<2>'

new_text, count = re.subn(pattern, replacement, text)

if count == 0:
    print(f"ERROR: Could not find conv_tiled(...) call in {filename}")
    sys.exit(1)

with open(filename, "w") as f:
    f.write(new_text)

print(f"Set tile size to {tile}x{tile}")
PY
}

# ============================================================
# Run experiments
# ============================================================

for T in "${TILE_SIZES[@]}"; do

    echo
    echo "============================================================"
    echo "                 TILE SIZE: ${T}x${T}"
    echo "============================================================"

    # --------------------------------------------------------
    # Modify source
    # --------------------------------------------------------

    set_tile_size "$T"

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    echo
    echo "[1/3] Compiling..."

    make

    # --------------------------------------------------------
    # Run perf
    # --------------------------------------------------------

    echo
    echo "[2/3] Running benchmark..."

    PERF_OUTPUT=$(mktemp)

    sudo taskset -c "$CPU" perf stat \
        -e instructions,L1-dcache-loads,L1-dcache-load-misses \
        "$BINARY" tile \
        > "$PERF_OUTPUT" 2>&1

    # Save complete raw output
    {
        echo
        echo "============================================================"
        echo "TILE ${T}x${T}"
        echo "============================================================"
        cat "$PERF_OUTPUT"
    } >> "$RAW"

    # --------------------------------------------------------
    # Extract harness result
    # --------------------------------------------------------

    STAGE_LINE=$(grep -E '^[[:space:]]*tile[[:space:]]+yes' "$PERF_OUTPUT" || true)

    if [[ -z "$STAGE_LINE" ]]; then
        echo
        echo "ERROR: Could not find tile stage in output."
        echo
        cat "$PERF_OUTPUT"
        rm -f "$PERF_OUTPUT"
        exit 1
    fi

    # Example:
    #
    # tile            yes          21.412        3.53      0.84x
    #
    # Fields:
    # $1 = tile
    # $2 = yes
    # $3 = time
    # $4 = GFLOP/s
    # $5 = speedup

    TIME_MS=$(echo "$STAGE_LINE" | awk '{print $3}')
    GFLOPS=$(echo "$STAGE_LINE" | awk '{print $4}')
    SPEEDUP=$(echo "$STAGE_LINE" | awk '{print $5}' | sed 's/x//')

    # --------------------------------------------------------
    # Extract perf counters
    # --------------------------------------------------------

    INSTRUCTIONS=$(grep -E 'cpu_core/instructions/' "$PERF_OUTPUT" \
        | awk '{print $1}' \
        | tr -d ',' \
        | head -1)

    L1_LOADS=$(grep -E 'cpu_core/L1-dcache-loads/' "$PERF_OUTPUT" \
        | awk '{print $1}' \
        | tr -d ',' \
        | head -1)

    L1_MISSES=$(grep -E 'cpu_core/L1-dcache-load-misses/' "$PERF_OUTPUT" \
        | awk '{print $1}' \
        | tr -d ',' \
        | head -1)

    # --------------------------------------------------------
    # Validate counters
    # --------------------------------------------------------

    if [[ -z "$INSTRUCTIONS" || -z "$L1_LOADS" || -z "$L1_MISSES" ]]; then
        echo
        echo "ERROR: Could not parse perf counters."
        echo
        cat "$PERF_OUTPUT"
        rm -f "$PERF_OUTPUT"
        exit 1
    fi

    # --------------------------------------------------------
    # Calculate MPKI
    # --------------------------------------------------------

    MPKI=$(awk \
        -v misses="$L1_MISSES" \
        -v instructions="$INSTRUCTIONS" \
        'BEGIN {
            if (instructions > 0)
                printf "%.4f", misses / instructions * 1000;
            else
                print "NA";
        }')

    # --------------------------------------------------------
    # Calculate L1-D miss rate
    # --------------------------------------------------------

    MISS_RATE=$(awk \
        -v misses="$L1_MISSES" \
        -v loads="$L1_LOADS" \
        'BEGIN {
            if (loads > 0)
                printf "%.4f", misses / loads * 100;
            else
                print "NA";
        }')

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    echo
    echo "[3/3] Result:"
    echo

    printf "%-10s %12s %10s %10s %15s %15s %15s %10s %14s\n" \
        "Tile" \
        "Time(ms)" \
        "GFLOP/s" \
        "Speedup" \
        "Instructions" \
        "L1 Loads" \
        "L1 Misses" \
        "MPKI" \
        "Miss Rate"

    printf "%-10s %12s %10s %10s %15s %15s %15s %10s %13s%%\n" \
        "${T}x${T}" \
        "$TIME_MS" \
        "$GFLOPS" \
        "$SPEEDUP" \
        "$INSTRUCTIONS" \
        "$L1_LOADS" \
        "$L1_MISSES" \
        "$MPKI" \
        "$MISS_RATE"

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    echo "${T}x${T},${TIME_MS},${GFLOPS},${SPEEDUP},${INSTRUCTIONS},${L1_LOADS},${L1_MISSES},${MPKI},${MISS_RATE}" \
        >> "$CSV"

    rm -f "$PERF_OUTPUT"
done

# ============================================================
# Consolidated results
# ============================================================

echo
echo
echo "============================================================"
echo "                  CONSOLIDATED RESULTS"
echo "============================================================"
echo

column -s, -t "$CSV"

echo
echo "CSV saved to: $CSV"
echo "Raw perf output saved to: $RAW"