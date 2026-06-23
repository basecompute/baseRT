#!/usr/bin/env bash
# Throughput benchmark for the baseRT engine over .base models.
#
#   ./run_bench.sh [model.base ...]
#
# With no args it benchmarks every *.base under $MODELS_DIR. Results are written
# as CSV to $RESULTS. Requires the engine binary (baseRT_bench) — download a
# release and unpack into $BUILD_DIR (see ../README.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
BENCH="${BENCH:-$BUILD_DIR/baseRT_bench}"
RESULTS="${RESULTS:-$REPO_ROOT/benchmarks/results/$(uname -m)_baseRT.csv}"
PP_VALS="${PP_VALS:-128 256 512}"
TG_VAL="${TG_VAL:-128}"
REPS="${REPS:-5}"

[ -x "$BENCH" ] || { echo "baseRT_bench not found at $BENCH — download an engine release into $BUILD_DIR"; exit 1; }

if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=("$MODELS_DIR"/*.base); fi

mkdir -p "$(dirname "$RESULTS")"
echo "model,size_mb,engine,test,tok_per_sec,stddev" > "$RESULTS"

parse() { # bench-table-output  model-name
  while IFS= read -r line; do
    echo "$line" | grep -qE '\| .* MiB \|' || continue
    local test tps std
    test=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$4); print $4}')
    tps=$(echo "$line"  | awk -F'|' '{gsub(/^ +| +$/,"",$5); split($5,a," ± "); print a[1]}')
    std=$(echo "$line"  | awk -F'|' '{gsub(/^ +| +$/,"",$5); split($5,a," ± "); print a[2]}')
    [ -n "$test" ] && [ -n "$tps" ] && echo "$2,0,baseRT,$test,$tps,$std" >> "$RESULTS"
  done <<< "$1"
}

for m in "${MODELS[@]}"; do
  [ -f "$m" ] || { echo "skip (missing): $m"; continue; }
  name="$(basename "$m")"
  echo "Benchmarking: $name"
  for pp in $PP_VALS; do
    out="$("$BENCH" "$m" -p "$pp" -n "$TG_VAL" -r "$REPS" 2>&1)"
    parse "$out" "$name"
  done
done

echo "=== done -> $RESULTS ==="
cat "$RESULTS"
