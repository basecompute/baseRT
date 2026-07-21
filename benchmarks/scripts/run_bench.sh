#!/usr/bin/env bash
# Run basert-bench for one or more .base models and write normalized CSV results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
BASERT="${BASERT:-$REPO_ROOT/build/basert-bench}"
PYTHON="${PYTHON:-python3}"
PP_VALS="${PP_VALS:-128 256 512}"
TG_VAL="${TG_VAL:-128}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-1}"
ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
RESULTS="${RESULTS:-$REPO_ROOT/benchmarks/results/${ARCH}_baseRT.csv}"
PARSER="$SCRIPT_DIR/parse_bench.py"

usage() {
    echo "Usage: $0 [model.base ...]" >&2
    echo "With no model arguments, benchmarks every models/*.base file." >&2
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ ! -x "$BASERT" ]]; then
    echo "error: basert-bench is missing or not executable: $BASERT" >&2
    exit 1
fi
if [[ ! -r "$PARSER" ]]; then
    echo "error: benchmark output parser is missing or unreadable: $PARSER" >&2
    exit 1
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: Python interpreter not found: $PYTHON" >&2
    exit 1
fi
if ! is_positive_integer "$TG_VAL"; then
    echo "error: TG_VAL must be a positive integer: $TG_VAL" >&2
    exit 2
fi
if ! is_positive_integer "$REPS"; then
    echo "error: REPS must be a positive integer: $REPS" >&2
    exit 2
fi
if ! is_nonnegative_integer "$WARMUP"; then
    echo "error: WARMUP must be a non-negative integer: $WARMUP" >&2
    exit 2
fi

IFS=' ' read -r -a prompt_values <<< "$PP_VALS"
if [[ ${#prompt_values[@]} -eq 0 ]]; then
    echo "error: PP_VALS must contain at least one prompt length" >&2
    exit 2
fi
for prompt in "${prompt_values[@]}"; do
    if ! is_positive_integer "$prompt"; then
        echo "error: PP_VALS must contain positive integers: $PP_VALS" >&2
        exit 2
    fi
done

if [[ $# -gt 0 ]]; then
    models=("$@")
else
    shopt -s nullglob
    models=("$MODELS_DIR"/*.base)
    shopt -u nullglob
fi
if [[ ${#models[@]} -eq 0 ]]; then
    usage
    echo "error: no .base models found in $MODELS_DIR" >&2
    exit 1
fi
for model in "${models[@]}"; do
    if [[ ! -f "$model" ]]; then
        echo "error: model not found: $model" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$RESULTS")"
temporary_results="$(mktemp "${RESULTS}.tmp.XXXXXX")"
trap 'rm -f "$temporary_results"' EXIT
printf 'model,size_mb,engine,test,tok_per_sec,stddev\n' > "$temporary_results"

total=${#models[@]}
index=0
for model in "${models[@]}"; do
    index=$((index + 1))
    echo "[$index/$total] Benchmarking: $model"
    for prompt in "${prompt_values[@]}"; do
        echo "  pp${prompt}/tg${TG_VAL}, reps=${REPS}, warmup=${WARMUP}"
        if output=$("$BASERT" "$model" \
            -p "$prompt" -n "$TG_VAL" -r "$REPS" -w "$WARMUP" 2>&1); then
            printf '%s\n' "$output"
        else
            status=$?
            printf '%s\n' "$output" >&2
            echo "error: basert-bench failed for $model (pp${prompt}/tg${TG_VAL})" >&2
            exit "$status"
        fi
        printf '%s\n' "$output" | "$PYTHON" "$PARSER" \
            --expect "pp${prompt}" --expect "tg${TG_VAL}" \
            >> "$temporary_results"
    done
done

mv "$temporary_results" "$RESULTS"
trap - EXIT
echo "Wrote results to $RESULTS"
