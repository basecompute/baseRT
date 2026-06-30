#!/bin/bash
# Clean benchmark — single process, no overlap
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
BASERT="${BASERT:-$REPO_ROOT/build/basert-bench}"
RESULTS="${RESULTS:-/tmp/bench_clean.csv}"

echo "model,engine,test,tok_per_sec,stddev" > "$RESULTS"

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
        echo "${model},baseRT,${test},${tps},${std}" >> "$RESULTS"
      fi
    fi
  done <<< "$output"
}

parse_llama() {
  local output="$1"
  local model="$2"
  while IFS= read -r line; do
    if echo "$line" | grep -qE '^\| .* \| .*(pp|tg)[0-9]'; then
      local test=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$7); print $7}')
      local tps=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$8); split($8,a," ± "); print a[1]}')
      local std=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/,"",$8); split($8,a," ± "); print a[2]}')
      if [ -n "$test" ] && [ -n "$tps" ]; then
        echo "${model},llama.cpp,${test},${tps},${std}" >> "$RESULTS"
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
    echo "[$count/$total] SKIP: $m"
    continue
  fi

  echo "[$count/$total] $m"

  for pp in $PP_VALS; do
    echo "  pp${pp}/tg${TG_VAL} r=${REPS}"

    # BaseRT first
    echo "    baseRT..."
    baseRT_out=$("$BASERT" "$model_path" -p $pp -n $TG_VAL -r $REPS 2>&1)
    parse_baseRT "$baseRT_out" "$m"

    # Then llama.cpp
    echo "    llama.cpp..."
    llama_out=$(llama-bench -m "$model_path" -p $pp -n $TG_VAL -r $REPS 2>/dev/null)
    parse_llama "$llama_out" "$m"
  done

  echo "  done"
done

echo ""
echo "=== COMPLETE ==="
echo "Lines: $(wc -l < "$RESULTS")"
