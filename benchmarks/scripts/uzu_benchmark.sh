#!/bin/bash
# Uzu benchmark — apples-to-apples comparison using serve endpoint
# Starts uzu serve, calibrates prompt length to hit exact token targets
# (accounting for chat template overhead), then benchmarks at pp128/pp256/pp512.
# Matches baseRT_bench methodology: controlled token counts, separate pp and tg.
#
# Uzu is an external project (https://github.com/...). UZU_DIR must
# point to a checkout of it; we don't pick a default because the layout
# of a sibling checkout differs from machine to machine.
UZU_DIR="${UZU_DIR:?set UZU_DIR to your uzu checkout (e.g. ~/Projects/uzu)}"
UZU_CLI="${UZU_DIR}/target/release/cli"
UZU_MODELS="${UZU_DIR}/models/0.1.8"
HELPER_DIR="${UZU_DIR}/tools/helpers"
RESULTS="${RESULTS:-/tmp/bench_uzu.csv}"
UZU_PORT="${UZU_PORT:-8000}"

echo "model,engine,test,tok_per_sec,stddev" > "$RESULTS"

# Models to benchmark (repo_id|label)
MODELS=(
  "Qwen/Qwen3-0.6B-MLX-4bit|Qwen3-0.6B-4bit"
  "mlx-community/Llama-3.2-1B-Instruct-4bit|Llama-3.2-1B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit|Llama-3.2-3B-4bit"
  "mlx-community/gemma-3-1b-it-4bit|Gemma-3-1B-4bit"
  "mlx-community/gemma-3-4b-it-4bit|Gemma-3-4B-4bit"
  "mlx-community/Llama-3.2-1B-Instruct-8bit|Llama-3.2-1B-8bit"
  "mlx-community/Llama-3.2-3B-Instruct-8bit|Llama-3.2-3B-8bit"
  "mlx-community/gemma-3-1b-it-8bit|Gemma-3-1B-8bit"
  "mlx-community/gemma-3-4b-it-8bit|Gemma-3-4B-8bit"
  "Qwen/Qwen3-0.6B-MLX-8bit|Qwen3-0.6B-8bit"
  "Qwen/Qwen3-4B-MLX-4bit|Qwen3-4B-4bit"
  "Qwen/Qwen3-4B-MLX-8bit|Qwen3-4B-8bit"
  "Qwen/Qwen3-8B-MLX-4bit|Qwen3-8B-4bit"
  "Qwen/Qwen3-8B-MLX-8bit|Qwen3-8B-8bit"
)

PP_VALS="128 256 512"
TG_VAL=128
REPS=5

get_model_dir() {
  local repo_id="$1"
  for dir in "${UZU_MODELS}"/*/; do
    if [ -f "${dir}config.json" ]; then
      local repo=$(python3 -c "import json; print(json.load(open('${dir}config.json')).get('repo',''))" 2>/dev/null)
      if [ "$repo" = "$repo_id" ]; then
        echo "$dir"
        return 0
      fi
    fi
  done
  return 1
}

download_model() {
  local repo_id="$1"
  echo "  Downloading ${repo_id}..."
  (cd "$HELPER_DIR" && uv run main.py download-model "$repo_id" 2>&1)
}

wait_for_server() {
  local max_wait=120
  local waited=0
  while [ $waited -lt $max_wait ]; do
    if curl -s http://localhost:${UZU_PORT}/chat/completions -X POST \
         -H "Content-Type: application/json" \
         -d '{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":1}' \
         > /dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

kill_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
  fi
}

# Send one request via file-based JSON, return response to stdout
send_request() {
  local prompt_file="$1"
  local max_tokens="$2"
  local request_file="/tmp/uzu_request.json"

  python3 -c "
import json
with open('${prompt_file}') as f:
    prompt = f.read()
print(json.dumps({
    'messages': [{'role': 'user', 'content': prompt}],
    'max_completion_tokens': ${max_tokens},
    'stream': False
}))
" > "$request_file"

  curl -s http://localhost:${UZU_PORT}/chat/completions \
    -X POST -H "Content-Type: application/json" \
    -d @"$request_file"
}

# Calibrate: find the number of "hello" repetitions that produces exactly target_tokens
# after chat template expansion. Binary search on word count.
calibrate_prompt() {
  local target="$1"
  local prompt_file="/tmp/uzu_calibrate.txt"

  # Binary search: find word count where total tokens == target
  local lo=1
  local hi=$((target * 2))
  local best_count=$((target - 35))
  local resp_file="/tmp/uzu_calibrate_resp.json"

  for iter in $(seq 1 10); do
    local mid=$(( (lo + hi) / 2 ))
    python3 -c "print(' '.join(['hello'] * ${mid}))" > "$prompt_file"
    send_request "$prompt_file" 1 > "$resp_file"
    local actual=$(python3 -c "
import json
with open('${resp_file}') as f:
    r = json.load(f)
print(r['stats']['total_stats']['tokens_count_input'])
" 2>/dev/null)

    if [ -z "$actual" ] || [ "$actual" = "" ]; then
      break
    fi

    if [ "$actual" -eq "$target" ]; then
      best_count=$mid
      break
    elif [ "$actual" -lt "$target" ]; then
      lo=$mid
      best_count=$mid
    else
      hi=$mid
      best_count=$mid
    fi
  done

  echo "$best_count"
}

total=${#MODELS[@]}
count=0

for entry in "${MODELS[@]}"; do
  IFS='|' read -r repo_id label <<< "$entry"
  count=$((count + 1))

  echo "[$count/$total] $label ($repo_id)"

  # Find or download model
  model_dir=$(get_model_dir "$repo_id")
  if [ -z "$model_dir" ]; then
    download_model "$repo_id"
    model_dir=$(get_model_dir "$repo_id")
    if [ -z "$model_dir" ]; then
      echo "  SKIP (download failed): $repo_id"
      continue
    fi
  fi

  # Start uzu serve
  echo "  Starting uzu serve..."
  "$UZU_CLI" serve "$model_dir" > /dev/null 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    echo "  SKIP (server failed to start): $repo_id"
    kill_server
    continue
  fi
  echo "  Server ready."

  # Warmup
  echo "  Warmup..."
  echo "Hello, how are you?" > /tmp/uzu_prompt.txt
  send_request /tmp/uzu_prompt.txt 8 > /dev/null 2>&1

  # Calibrate prompt lengths for this model's tokenizer + chat template
  echo "  Calibrating prompt lengths..."
  for pp in $PP_VALS; do
    word_count=$(calibrate_prompt "$pp")
    eval "WORDS_PP${pp}=${word_count}"

    # Verify
    python3 -c "print(' '.join(['hello'] * ${word_count}))" > "/tmp/uzu_prompt_pp${pp}.txt"
    send_request "/tmp/uzu_prompt_pp${pp}.txt" 1 > /tmp/uzu_verify.json
    actual_tokens=$(python3 -c "
import json
with open('/tmp/uzu_verify.json') as f:
    r = json.load(f)
print(r['stats']['total_stats']['tokens_count_input'])
" 2>/dev/null)
    echo "    pp${pp}: ${word_count} words -> ${actual_tokens} tokens"
  done

  for pp in $PP_VALS; do
    echo "  pp${pp}/tg${TG_VAL} x${REPS}..."
    prompt_file="/tmp/uzu_prompt_pp${pp}.txt"

    pp_samples=""
    tg_samples=""
    actual_pp=""

    for r in $(seq 1 $REPS); do
      response_file="/tmp/uzu_response.json"
      send_request "$prompt_file" "$TG_VAL" > "$response_file"

      metrics=$(python3 -c "
import json, sys
try:
    with open('${response_file}') as f:
        r = json.load(f)
    s = r['stats']
    pp_tps = s['prefill_stats']['processed_tokens_per_second']
    tg_tps = s['generate_stats']['processed_tokens_per_second'] if s.get('generate_stats') else 0.0
    prompt_n = s['total_stats']['tokens_count_input']
    print(f'{prompt_n} {pp_tps:.2f} {tg_tps:.2f}')
except Exception as e:
    print(f'ERROR {e}', file=sys.stderr)
    print('')
" 2>/dev/null)

      if [ -n "$metrics" ]; then
        actual_pp=$(echo "$metrics" | awk '{print $1}')
        pp_tps=$(echo "$metrics" | awk '{print $2}')
        tg_tps=$(echo "$metrics" | awk '{print $3}')
        pp_samples="${pp_samples} ${pp_tps}"
        tg_samples="${tg_samples} ${tg_tps}"
      fi
    done

    if [ -n "$pp_samples" ]; then
      python3 -c "
import statistics
pp_vals = [float(x) for x in '${pp_samples}'.split()]
tg_vals = [float(x) for x in '${tg_samples}'.split()]

pp_mean = statistics.mean(pp_vals)
pp_std = statistics.pstdev(pp_vals) if len(pp_vals) > 1 else 0.0
tg_mean = statistics.mean(tg_vals)
tg_std = statistics.pstdev(tg_vals) if len(tg_vals) > 1 else 0.0

print(f'${label},uzu,pp${pp},{pp_mean:.2f},{pp_std:.2f}')
print(f'${label},uzu,tg${TG_VAL},{tg_mean:.2f},{tg_std:.2f}')
" | tee -a "$RESULTS" | while IFS= read -r line; do echo "    $line"; done
    fi
  done

  kill_server
  echo "  Done — cooling 30s..."
  sleep 30
done

echo ""
echo "=== UZU BENCHMARKS COMPLETE ==="
echo "Results in: $RESULTS"
echo ""
cat "$RESULTS"
