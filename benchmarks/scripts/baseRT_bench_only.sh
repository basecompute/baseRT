#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
BASERT="${BASERT:-$REPO_ROOT/build/baseRT_bench}"
RESULTS="${RESULTS:-/tmp/bench_results_baseRT.csv}"

echo "model,size_mb,engine,test,tok_per_sec,stddev" > "$RESULTS"

MODELS=(
  "Qwen3-0.6B-Q2_K.gguf"
  "Qwen3-0.6B-Q3_K_S.gguf"
  "Qwen3-0.6B-Q3_K_M.gguf"
  "Qwen3-0.6B-Q4_0.gguf"
  "Qwen3-0.6B-Q5_K_S.gguf"
  "Qwen3-0.6B-Q5_K_M.gguf"
  "Qwen3-0.6B-Q8_0.gguf"
  "Qwen3-4B-Q4_K_M.gguf"
  "Qwen3-4B-Q8_0.gguf"
  "Qwen3-8B-Q8_0.gguf"
  "Qwen3.5-4B-Q4_K_M.gguf"
  "gemma-3-1b-it-Q4_K_M.gguf"
  "gemma-3-1b-it-Q8_0.gguf"
  "Llama-3.2-1B-Instruct-Q4_0.gguf"
  "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  "llama-3.2-1b-instruct-q8_0.gguf"
  "Llama-3.2-3B-Instruct-Q8_0.gguf"
  "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
  "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
)

PP_VALS="128 256 512"
TG_VAL=128
REPS=5

parse_baseRT() {
  local output="$1"
  local model="$2"
  while IFS= read -r line; do
    if echo "$line" | grep -qE '\| .* MiB \|'; then
      local test=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$4); print $4}')
      local tps=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$5); split($5,a," ± "); print a[1]}')
      local std=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$5); split($5,a," ± "); print a[2]}')
      if [ -n "$test" ] && [ -n "$tps" ]; then
        echo "${model},0,baseRT,${test},${tps},${std}" >> "$RESULTS"
      fi
    fi
  done <<< "$output"
}

total=${#MODELS[@]}
count=0

for m in "${MODELS[@]}"; do
  count=$((count + 1))
  model_path="${MODELS_DIR}/${m}"

  if [ ! -f "$model_path" ]; then
    echo "[$count/$total] SKIP (missing): $m"
    continue
  fi

  echo "[$count/$total] Benchmarking baseRT: $m"

  for pp in $PP_VALS; do
    echo "  pp${pp}/tg${TG_VAL}, ${REPS} reps..."
    baseRT_out=$("$BASERT" "$model_path" -p $pp -n $TG_VAL -r $REPS 2>&1)
    parse_baseRT "$baseRT_out" "$m"
  done

  echo "  Done: $m"
done

echo "=== BASERT BENCHMARKS COMPLETE ==="
cat "$RESULTS"
