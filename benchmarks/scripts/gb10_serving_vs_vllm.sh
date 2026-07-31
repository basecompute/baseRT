#!/usr/bin/env bash
# Reproducible baseRT-vs-vLLM serving sweep on GB10 (DGX Spark).
#
# For each model it launches vLLM (docker) and baseRT serve in turn (never
# together — they'd contend for the GPU), drives BOTH through the same client
# (bench/bench.sh -> bench_serving.py) at c1/8/16/32, and emits a matched
# out-tok/s comparison table. This is the committed replacement for the old
# session-scratchpad run32b.sh / vllm_sweep.sh (which produced the July numbers
# but were never checked in).
#
# Usage:  benchmarks/scripts/gb10_serving_vs_vllm.sh [model_key ...]
#   model_key: one of the keys in the MODELS table below (default: all).
# Env:
#   CONC="1 8 16 32"     concurrency points
#   OUT=benchmarks/gb10  output dir (writes serving-vs-vllm.csv; the .md summary
#                        is regenerated from the CSV separately, not by this script)
#   VLLM_IMAGE=hellohal2064/vllm-dgx-spark-gb10:latest
#   BASERT_SERVE=build-rel/basert-serve
#   HF_HUB=/mnt/nas-models/hf-hub
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

CONC="${CONC:-1 8 16 32}"
OUT="${OUT:-benchmarks/gb10}"
VLLM_IMAGE="${VLLM_IMAGE:-hellohal2064/vllm-dgx-spark-gb10:latest}"
BASERT_SERVE="${BASERT_SERVE:-build-rel/basert-serve}"
HF_HUB="${HF_HUB:-/mnt/nas-models/hf-hub}"
PORT=8000
RESDIR="$OUT/serving_raw"
mkdir -p "$RESDIR"

# key | baseRT .base | vLLM HF repo dir (the models--ORG--NAME cache dir; the
# whole repo is mounted so snapshot symlinks into blobs/ resolve in-container).
# FAIR 8-bit lane: baseRT serves its int8 (cuda-q8) bundle vs vLLM fp8 — both
# ~8-bit, comparable footprint. (q4mix is baseRT's lighter shipping format; the
# 4-bit lane is a separate baseRT-vs-llama.cpp table, no fair vLLM 4-bit here.)
# Canonical 11-model set, matching the Metal 3-way sweep (bench_3way).
declare -A BASE=(
  [gemma-3-1b]="$HOME/models/gemma-3-1b-it-cuda-q8.base"
  [gemma-4-e2b]="$HOME/models/gemma-4-E2B-it-cuda-q8.base"
  [gemma-4-26b]="$HOME/models/gemma-4-26B-A4B-it-cuda-q8.base"
  [llama-3.2-1b]="$HOME/models/Llama-3.2-1B-Instruct-cuda-q8.base"
  [llama-3.2-3b]="$HOME/models/Llama-3.2-3B-Instruct-cuda-q8.base"
  [qwen3-0.6b]="$HOME/models/Qwen3-0.6B-cuda-q8.base"
  [qwen3-30b-a3b]="$HOME/models/Qwen3-30B-A3B-Instruct-2507-cuda-q8.base"
  [qwen3.5-2b]="$HOME/models/Qwen3.5-2B-cuda-q8.base"
  [qwen3.5-35b-a3b]="$HOME/models/Qwen3.5-35B-A3B-cuda-q8.base"
  [qwen3.6-27b]="$HOME/models/Qwen3.6-27B-cuda-q8.base"
  [qwen3.6-35b]="$HOME/models/Qwen3.6-35B-A3B-cuda-q8.base"
)
declare -A HFREPO=(
  [gemma-3-1b]="$HF_HUB/models--unsloth--gemma-3-1b-it"
  [gemma-4-e2b]="$HF_HUB/models--google--gemma-4-E2B-it"
  [gemma-4-26b]="$HF_HUB/models--google--gemma-4-26B-A4B-it"
  [llama-3.2-1b]="$HF_HUB/models--meta-llama--Llama-3.2-1B-Instruct"
  [llama-3.2-3b]="$HF_HUB/models--meta-llama--Llama-3.2-3B-Instruct"
  [qwen3-0.6b]="$HF_HUB/models--Qwen--Qwen3-0.6B"
  [qwen3-30b-a3b]="$HF_HUB/models--Qwen--Qwen3-30B-A3B-Instruct-2507"
  [qwen3.5-2b]="$HF_HUB/models--Qwen--Qwen3.5-2B"
  [qwen3.5-35b-a3b]="$HF_HUB/models--Qwen--Qwen3.5-35B-A3B"
  [qwen3.6-27b]="$HF_HUB/models--Qwen--Qwen3.6-27B"
  [qwen3.6-35b]="$HF_HUB/models--Qwen--Qwen3.6-35B-A3B"
)
# vLLM image per model. The GB10-optimized image (default) is Qwen-SPECIALISED:
# its entrypoint bakes in `--reasoning-parser qwen3 --tool-call-parser qwen3_coder
# --enable-auto-tool-choice`, which CRASH on non-Qwen archs at startup (observed:
# llama/mistral containers exit 1 before serving). It also has older Transformers
# that don't recognise the qwen3_5/qwen3_6 hybrid arch. So:
#   - Dense Qwen3 + Qwen3-MoE (30B): GB10 image (the fairest vLLM-on-GB10 baseline;
#     qwen parsers apply cleanly).
#   - Qwen3.5/3.6 hybrids: stock v0.19.1 (resolves Qwen3_5ForConditionalGeneration).
#   - Llama / Mistral / Gemma-4: stock v0.20.0 (no qwen parsers; gemma-4 arch is
#     0.20-only). Gemma-4 is multimodal, so the stock branch raises max-num-batched-
#     tokens past its per-image MM token count (2496) or vLLM refuses to start.
VLLM_IMAGE_HYBRID="${VLLM_IMAGE_HYBRID:-vllm/vllm-openai:v0.19.1-cu130}"
VLLM_IMAGE_STOCK="${VLLM_IMAGE_STOCK:-vllm/vllm-openai:v0.20.0-cu130}"
declare -A VIMG=(
  [qwen3.5-2b]="$VLLM_IMAGE_HYBRID"
  [qwen3.5-35b-a3b]="$VLLM_IMAGE_HYBRID"
  [qwen3.6-27b]="$VLLM_IMAGE_HYBRID"
  [qwen3.6-35b]="$VLLM_IMAGE_HYBRID"
  [gemma-3-1b]="$VLLM_IMAGE_STOCK"
  [gemma-4-e2b]="$VLLM_IMAGE_STOCK"
  [gemma-4-26b]="$VLLM_IMAGE_STOCK"
  [llama-3.2-1b]="$VLLM_IMAGE_STOCK"
  [llama-3.2-3b]="$VLLM_IMAGE_STOCK"
)
# qwen3-0.6b + qwen3-30b-a3b (dense/MoE Qwen) use the default GB10 image.

KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(gemma-3-1b gemma-4-e2b gemma-4-26b \
    llama-3.2-1b llama-3.2-3b qwen3-0.6b qwen3-30b-a3b \
    qwen3.5-2b qwen3.5-35b-a3b qwen3.6-27b qwen3.6-35b)

VLLM_CID=""; BASERT_PID=""
cleanup() {
  [ -n "$VLLM_CID" ] && docker rm -f "$VLLM_CID" >/dev/null 2>&1
  if [ -n "$BASERT_PID" ]; then
    kill "$BASERT_PID" >/dev/null 2>&1
    # WAIT for the process to actually exit — a bare kill returns immediately,
    # but baseRT's CUDA context (model weights + KV, up to ~100GB on GB10's
    # unified memory) is only released when the process fully tears down. If the
    # NEXT model's vLLM launches before that, it sees a near-full GPU and
    # OOM-crashes on startup (observed: models 2..N of a sweep silently skipped
    # with 'vLLM never became ready', the container gone by the time we log it).
    for _ in $(seq 1 30); do kill -0 "$BASERT_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$BASERT_PID" >/dev/null 2>&1; wait "$BASERT_PID" 2>/dev/null
  fi
  VLLM_CID=""; BASERT_PID=""
  # A few seconds for the CUDA driver to reclaim the freed allocations before
  # the next engine probes memory. Cheap insurance vs a whole wasted model run.
  sleep 8
}
trap cleanup EXIT

wait_ready() { # timeout_s
  local t=0
  while [ "$t" -lt "$1" ]; do
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && return 0
    sleep 3; t=$((t+3))
  done
  return 1
}

# extract out_tok_per_s for a given tput/cN label from a bench JSONL
ots() { # jsonl label
  python3 - "$1" "$2" <<'PY'
import json,sys
path,label=sys.argv[1],sys.argv[2]
try:
    for line in open(path):
        line=line.strip()
        if not line or line[0] != '{': continue
        d=json.loads(line)
        if d.get("label")==label:
            # Drop a data point with ANY failed request — bench_serving.py still
            # reports out_tok_per_s over the wall window even when requests
            # errored (common at high concurrency), which would understate/
            # corrupt the throughput row. Only a clean burst counts.
            print("" if d.get("failed",0) else d.get("out_tok_per_s","")); break
    else: print("")
except FileNotFoundError: print("")
PY
}

CSV="$OUT/serving-vs-vllm.csv"
# Append-safe: only write the header if the CSV doesn't already exist, so a
# subsequent run over additional model keys extends the table instead of
# wiping the earlier rows.
[ -f "$CSV" ] || echo "model,concurrency,baseRT_out_tok_s,vllm_out_tok_s,baseRT_over_vllm_pct" > "$CSV"

served_model_id() { curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; }

for k in "${KEYS[@]}"; do
  base="${BASE[$k]:-}"; repo="${HFREPO[$k]:-}"
  [ -f "$base" ] || { echo "::skip $k — baseRT bundle missing: $base"; continue; }
  [ -d "$repo/snapshots" ] || { echo "::skip $k — HF repo missing: $repo"; continue; }
  # Deterministic snapshot: the revision refs/main points at, not the
  # lexicographically-first hash `ls | head -1` would pick.
  snaprel="snapshots/$( { cat "$repo/refs/main" 2>/dev/null || ls -t "$repo/snapshots" | head -1; } )"
  echo "############ $k (baseRT=$base  vLLM=$repo/$snaprel) ############"

  # ---- vLLM. Both images mount the whole repo (snapshot symlinks into blobs/
  #      resolve) and serve under the mounted path (query /v1/models for the id).
  #      GB10 image = env-driven entrypoint; stock v0.19.1 = standard `vllm serve`
  #      arg style. ----
  img="${VIMG[$k]:-$VLLM_IMAGE}"
  echo ">>> launching vLLM ($img)"
  cleanup
  if [ "$img" = "$VLLM_IMAGE" ]; then
    VLLM_CID=$(docker run -d --rm --gpus all --network host \
        -e MODEL_PATH="/models/repo/$snaprel" -e HOST=0.0.0.0 -e PORT="$PORT" \
        -e MAX_MODEL_LEN=8192 -e GPU_MEMORY_UTIL=0.85 -e ATTENTION_BACKEND=FLASH_ATTN \
        -v "$repo":/models/repo:ro \
        "$img" --quantization fp8 --max-num-seqs 32 --enable-prefix-caching 2>/dev/null)
  else
    # --max-num-batched-tokens 8192: required for multimodal Gemma-4 (its per-image
    # MM token count 2496 > the default 2048 budget → vLLM refuses to start);
    # harmless (a larger prefill batch budget) for the text-only Llama/Mistral.
    VLLM_CID=$(docker run -d --rm --gpus all --network host \
        -v "$repo":/models/repo:ro \
        "$img" --model "/models/repo/$snaprel" --port "$PORT" --quantization fp8 \
        --max-model-len 8192 --max-num-seqs 32 --enable-prefix-caching \
        --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 2>/dev/null)
  fi
  if wait_ready 900; then
    mid=$(served_model_id)
    bench/bench.sh --url "http://127.0.0.1:$PORT" --model "$mid" --tag "vllm-$k" \
      --concurrency "$CONC" --out "$RESDIR" >"$RESDIR/vllm-$k.log" 2>&1
  else echo "  vLLM never became ready — skipping $k"; docker logs "$VLLM_CID" 2>&1 | tail -20; cleanup; continue; fi
  cleanup

  # ---- baseRT ----
  # Port must be free — a stale server on :PORT would make wait_ready pass
  # against the WRONG process and mislabel its results.
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "  :$PORT already serving before baseRT launch — skipping $k"; cleanup; continue; fi
  echo ">>> launching baseRT serve"
  "$BASERT_SERVE" "$base" --port "$PORT" --continuous-batching 32 --prefix-cache \
      --max-context 8192 >"$RESDIR/baseRT-$k.serve.log" 2>&1 &
  BASERT_PID=$!
  if wait_ready 300; then
    # Confirm CB actually engaged — for models where it isn't supported
    # (e.g. hybrid-MoE routes to the serial path) the row is baseRT's real
    # serving throughput but NOT continuous batching; label it so the CB-vs-
    # serial distinction is auditable rather than silently conflated.
    if grep -qiE 'continuous batching .*(disabled|unsupported|serial)|sequence_create.*UNSUPPORTED' \
         "$RESDIR/baseRT-$k.serve.log" 2>/dev/null; then
      echo "  note: $k baseRT is on the SERIAL path (CB not engaged) — throughput won't scale with concurrency"
    fi
    bench/bench.sh --url "http://127.0.0.1:$PORT" --model default --tag "baseRT-$k" \
      --concurrency "$CONC" --out "$RESDIR" >"$RESDIR/baseRT-$k.log" 2>&1
  else echo "  baseRT never became ready — skipping $k"; cleanup; continue; fi
  cleanup

  # ---- compare ----
  # Idempotent per model: drop any prior rows for this key so a rerun REPLACES
  # them instead of appending duplicates. Field-based (awk) so the '.' in keys
  # like qwen3.5-2b isn't treated as a regex wildcard; the header ($1=model) and
  # other models are preserved.
  awk -F, -v k="$k" 'NR==1 || $1!=k' "$CSV" > "$CSV.tmp" && mv "$CSV.tmp" "$CSV"
  for c in $CONC; do
    b=$(ots "$RESDIR/baseRT-$k.jsonl" "tput/c$c")
    v=$(ots "$RESDIR/vllm-$k.jsonl" "tput/c$c")
    pct=""; [ -n "$b" ] && [ -n "$v" ] && [ "$v" != "0.0" ] && pct=$(python3 -c "print(round(100*$b/$v))")
    echo "$k,$c,$b,$v,$pct" | tee -a "$CSV"
  done
done

echo "=== wrote $CSV (regenerate serving-vs-vllm.md from it; this script writes the CSV only) ==="
