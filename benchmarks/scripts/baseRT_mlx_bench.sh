#!/bin/bash
# BaseRT on MLX-format models — with 30s cooldown between models to avoid thermal throttling
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASERT="${BASERT:-$REPO_ROOT/build/basert-bench}"
CACHE="${CACHE:-$HOME/.cache/huggingface/hub}"
RESULTS="${RESULTS:-/tmp/bench_baseRT_mlx_cool.csv}"

echo "model,engine,test,tok_per_sec,stddev" > "$RESULTS"

MODELS=(
  "Qwen3-0.6B-3bit|$CACHE/models--mlx-community--Qwen3-0.6B-3bit/snapshots/05aff87f9d2642185b057d0d1ca1068a1482081e/"
  "Qwen3-0.6B-4bit|$CACHE/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8/"
  "Qwen3-0.6B-4bit-AWQ|$CACHE/models--mlx-community--Qwen3-0.6B-4bit-AWQ/snapshots/3c064b3401d4a7d355262a1d518faa823a4d8f11/"
  "Qwen3-0.6B-4bit-DWQ|$CACHE/models--mlx-community--Qwen3-0.6B-4bit-DWQ/snapshots/e630d870397d5a2d95fe0c9075c6f499fc0fc5c8/"
  "Qwen3-0.6B-6bit|$CACHE/models--mlx-community--Qwen3-0.6B-6bit/snapshots/45d962b21b1e813c3e9a7f3505391e72e8daba1e/"
  "Qwen3-0.6B-8bit|$CACHE/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068/"
  "Qwen3-0.6B-bf16|$CACHE/models--mlx-community--Qwen3-0.6B-bf16/snapshots/42096995f6402fde107068cf530136fe64b604f8/"
  "Qwen3-0.6B-fp16|$CACHE/models--mlx-community--Qwen3-0.6B/snapshots/0eaa7e5f5c5956433ed65cf71c7048d3deaae1f8/"
  "Gemma-3-1B-4bit|$CACHE/models--mlx-community--gemma-3-1b-it-4bit/snapshots/2d44e83dc9e80843d22fb941d3d699a0b1351aa6/"
  "Gemma-3-4B-4bit|$CACHE/models--mlx-community--gemma-3-text-4b-it-4bit/snapshots/4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e/"
  "Llama-3.2-1B-4bit|$CACHE/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912/"
  "Llama-3.2-1B-8bit|$CACHE/models--mlx-community--Llama-3.2-1B-Instruct-8bit/snapshots/d48cdf0a4ea22d893b7c63a99d6a693e24822795/"
  "Llama-3.2-3B-4bit|$CACHE/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e/"
  "Llama-3.2-3B-8bit|$CACHE/models--mlx-community--Llama-3.2-3B-Instruct-8bit/snapshots/ff054899609078569493def2823f9acd2780c0c9/"
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
        echo "  ${model},baseRT,${test},${tps},${std}"
      fi
    fi
  done <<< "$output"
}

total=${#MODELS[@]}
count=0

cd "$(dirname "$BASERT")" || exit 1

for entry in "${MODELS[@]}"; do
  IFS='|' read -r label model_path <<< "$entry"
  count=$((count + 1))

  if [ ! -d "$model_path" ]; then
    echo "[$count/$total] SKIP: $label (path missing)"
    continue
  fi

  echo "[$count/$total] $label"

  for pp in $PP_VALS; do
    echo "  pp${pp}/tg${TG_VAL} r=${REPS}"
    baseRT_out=$("$BASERT" "$model_path" -p $pp -n $TG_VAL -r $REPS 2>&1)
    parse_baseRT "$baseRT_out" "$label"
  done

  echo "  done — cooling 30s..."
  sleep 30
done

echo ""
echo "=== BASERT MLX BENCHMARKS COMPLETE ==="
cat "$RESULTS"
