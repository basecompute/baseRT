#!/bin/bash
# MLX-lm benchmark — with 30s cooldown between models to avoid thermal throttling.
# `mlx_lm.benchmark` ships with `mlx-lm` (pip install mlx-lm). Set MLX to the
# absolute path of the binary if it's not on PATH (e.g. installed in a venv).
MLX="${MLX:-mlx_lm.benchmark}"
RESULTS="${RESULTS:-/tmp/bench_mlx_cool.csv}"

echo "model,engine,test,tok_per_sec,stddev" > "$RESULTS"

MODELS=(
  "mlx-community/Qwen3-0.6B-3bit|Qwen3-0.6B-3bit"
  "mlx-community/Qwen3-0.6B-4bit|Qwen3-0.6B-4bit"
  "mlx-community/Qwen3-0.6B-4bit-AWQ|Qwen3-0.6B-4bit-AWQ"
  "mlx-community/Qwen3-0.6B-4bit-DWQ|Qwen3-0.6B-4bit-DWQ"
  "mlx-community/Qwen3-0.6B-6bit|Qwen3-0.6B-6bit"
  "mlx-community/Qwen3-0.6B-8bit|Qwen3-0.6B-8bit"
  "mlx-community/Qwen3-0.6B-bf16|Qwen3-0.6B-bf16"
  "mlx-community/Qwen3-0.6B|Qwen3-0.6B-fp16"
  "mlx-community/gemma-3-1b-it-4bit|Gemma-3-1B-4bit"
  "mlx-community/gemma-3-text-4b-it-4bit|Gemma-3-4B-4bit"
  "mlx-community/Llama-3.2-1B-Instruct-4bit|Llama-3.2-1B-4bit"
  "mlx-community/Llama-3.2-1B-Instruct-8bit|Llama-3.2-1B-8bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit|Llama-3.2-3B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-8bit|Llama-3.2-3B-8bit"
)

PP_VALS="128 256 512"
GEN=128
REPS=5

parse_mlx() {
  local output="$1"
  local model="$2"
  local pp="$3"

  local avg_line=$(echo "$output" | grep "^Averages:")
  if [ -n "$avg_line" ]; then
    local pp_tps=$(echo "$avg_line" | sed -n 's/.*prompt_tps=\([0-9.]*\).*/\1/p')
    local tg_tps=$(echo "$avg_line" | sed -n 's/.*generation_tps=\([0-9.]*\).*/\1/p')

    local pp_vals=()
    local tg_vals=()
    while IFS= read -r line; do
      local pv=$(echo "$line" | sed -n 's/.*prompt_tps=\([0-9.]*\).*/\1/p')
      local tv=$(echo "$line" | sed -n 's/.*generation_tps=\([0-9.]*\).*/\1/p')
      if [ -n "$pv" ]; then pp_vals+=("$pv"); fi
      if [ -n "$tv" ]; then tg_vals+=("$tv"); fi
    done <<< "$(echo "$output" | grep "^Trial")"

    local pp_std=$(python3 -c "
import statistics
vals = [$(IFS=,; echo "${pp_vals[*]}")]
print(f'{statistics.pstdev(vals):.2f}' if len(vals) > 1 else '0.00')
" 2>/dev/null || echo "0.00")

    local tg_std=$(python3 -c "
import statistics
vals = [$(IFS=,; echo "${tg_vals[*]}")]
print(f'{statistics.pstdev(vals):.2f}' if len(vals) > 1 else '0.00')
" 2>/dev/null || echo "0.00")

    if [ -n "$pp_tps" ]; then
      echo "${model},mlx,pp${pp},${pp_tps},${pp_std}" >> "$RESULTS"
      echo "  ${model},mlx,pp${pp},${pp_tps},${pp_std}"
    fi
    if [ -n "$tg_tps" ]; then
      echo "${model},mlx,tg${GEN},${tg_tps},${tg_std}" >> "$RESULTS"
      echo "  ${model},mlx,tg${GEN},${tg_tps},${tg_std}"
    fi
  fi
}

total=${#MODELS[@]}
count=0

for entry in "${MODELS[@]}"; do
  IFS='|' read -r mlx_repo label <<< "$entry"
  count=$((count + 1))

  echo "[$count/$total] $label ($mlx_repo)"

  for pp in $PP_VALS; do
    echo "  pp${pp}/tg${GEN} r=${REPS}"
    mlx_out=$("$MLX" --model "$mlx_repo" -p $pp -g $GEN -n $REPS 2>&1)
    parse_mlx "$mlx_out" "$label" "$pp"
  done

  echo "  done — cooling 30s..."
  sleep 30
done

echo ""
echo "=== MLX BENCHMARKS COMPLETE ==="
cat "$RESULTS"
