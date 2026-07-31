#!/usr/bin/env bash
# vLLM single-stream prefill/decode via `vllm bench latency` (engine-layer, no
# server) — the apples-to-apples counterpart to bench_2way's baseRT/llama.cpp
# pp512 + tg128. batch-size 1. Three latency points isolate each phase by
# CANCELLING the fixed per-call overhead (framework/sampling/detokenize):
#   la = latency(in=512,  out=1)    ~= prefill(512)  + 1 decode + overhead
#   lb = latency(in=1024, out=1)    ~= prefill(1024) + 1 decode + overhead
#   lc = latency(in=512,  out=128)  ~= prefill(512)  + 128 decode + overhead
#   pp512 = 512 / (lb - la)      # marginal cost of the 513..1024 prompt tokens (overhead + 1 decode
#                                # cancel). This EQUALS pp512-from-scratch only if prefill is ~linear
#                                # in sequence length; treat it as the near-512 marginal prefill rate.
#   tg128 = 127 / (lc - la)      # marginal cost of 127 decode tokens (prefill + overhead cancel)
#   e2e   = 128 / lc             # single-request end-to-end gen rate (prefill amortized)
# Dense uses the GB10-optimized image; qwen3_5/qwen3_6 hybrids use stock v0.19.1
# (GB10 image's Transformers can't load them). vLLM = fp8 (engine-vs-native-
# format vs baseRT/llama.cpp q4/q8, same caveat as the throughput table).
#
# Usage: gb10_vllm_prefill_decode.sh [model_key ...]   (GPU must be free)
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

OUT="${OUT:-benchmarks/gb10}"
HF_HUB="${HF_HUB:-/mnt/nas-models/hf-hub}"
VLLM_IMAGE="${VLLM_IMAGE:-hellohal2064/vllm-dgx-spark-gb10:latest}"
VLLM_IMAGE_HYBRID="${VLLM_IMAGE_HYBRID:-vllm/vllm-openai:v0.19.1-cu130}"
VLLM_IMAGE_STOCK="${VLLM_IMAGE_STOCK:-vllm/vllm-openai:v0.20.0-cu130}"
INLEN="${INLEN:-512}"; OUTLEN="${OUTLEN:-128}"
# Absolute path — docker -v rejects a relative host path (reads it as a named volume).
mkdir -p "$OUT/serving_raw/vllm_lat"
JDIR="$(cd "$OUT/serving_raw/vllm_lat" && pwd)"

declare -A HFREPO=(
  [gemma-3-1b]="$HF_HUB/models--unsloth--gemma-3-1b-it"
  [gemma-4-e2b]="$HF_HUB/models--google--gemma-4-E2B-it"
  [gemma-4-26b]="$HF_HUB/models--google--gemma-4-26B-A4B-it"
  [llama-3.2-1b]="$HF_HUB/models--meta-llama--Llama-3.2-1B-Instruct"
  [llama-3.2-3b]="$HF_HUB/models--meta-llama--Llama-3.2-3B-Instruct"
  [qwen3-0.6b]="$HF_HUB/models--Qwen--Qwen3-0.6B"
  [qwen3-30b-a3b]="$HF_HUB/models--Qwen--Qwen3-30B-A3B-Instruct-2507"
  [qwen3.5-2b]="$HF_HUB/models--Qwen--Qwen3.5-2B"   [qwen3.5-35b-a3b]="$HF_HUB/models--Qwen--Qwen3.5-35B-A3B"
  [qwen3.6-27b]="$HF_HUB/models--Qwen--Qwen3.6-27B" [qwen3.6-35b]="$HF_HUB/models--Qwen--Qwen3.6-35B-A3B"
)
# bench latency OVERRIDES the entrypoint, so the GB10 image's baked qwen parsers
# don't apply here (unlike the serving script) — but gemma-4 still needs the 0.20
# image for the arch. Route Gemma/Llama to stock 0.20 for consistency with the
# serving run's vLLM side; dense/MoE Qwen use the default GB10 image.
declare -A VIMG=(
  [qwen3.5-2b]="$VLLM_IMAGE_HYBRID" [qwen3.5-35b-a3b]="$VLLM_IMAGE_HYBRID"
  [qwen3.6-27b]="$VLLM_IMAGE_HYBRID" [qwen3.6-35b]="$VLLM_IMAGE_HYBRID"
  [gemma-3-1b]="$VLLM_IMAGE_STOCK" [gemma-4-e2b]="$VLLM_IMAGE_STOCK"
  [gemma-4-26b]="$VLLM_IMAGE_STOCK"
  [llama-3.2-1b]="$VLLM_IMAGE_STOCK" [llama-3.2-3b]="$VLLM_IMAGE_STOCK"
)
KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(gemma-3-1b gemma-4-e2b gemma-4-26b \
    llama-3.2-1b llama-3.2-3b qwen3-0.6b qwen3-30b-a3b \
    qwen3.5-2b qwen3.5-35b-a3b qwen3.6-27b qwen3.6-35b)

# run vllm bench latency once; echo avg_latency seconds (empty on failure)
lat() { # key repo snaprel inlen outlen  -> avg_latency seconds
  local img="${VIMG[$1]:-$VLLM_IMAGE}"
  local tag="in$4.out$5"
  local jf="$JDIR/$1.$tag.json"
  rm -f "$jf"   # discard any stale JSON so a failed docker run (OOM/bad image) reads empty, not last run's value
  docker run --rm --gpus all --network host -v "$2":/models/repo:ro -v "$JDIR":/out \
    --entrypoint vllm "$img" bench latency \
    --model "/models/repo/$3" --quantization fp8 --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --input-len "$4" --output-len "$5" --batch-size 1 \
    --num-iters-warmup 3 --num-iters 5 --output-json "/out/$1.$tag.json" >"$JDIR/$1.$tag.log" 2>&1
  python3 -c "import json;print(json.load(open('$jf')).get('avg_latency',''))" 2>/dev/null
}

CSV="$OUT/vllm-prefill-decode.csv"
HDR="model,vllm_prefill_pp${INLEN}_tok_s,vllm_decode_tg${OUTLEN}_tok_s,vllm_e2e_tg${OUTLEN}_tok_s"
# Recreate the file when absent OR when its header doesn't match these INLEN/
# OUTLEN — an override must not silently mix, e.g., tg64 values under a tg128
# header (the column names encode the lengths). Same lengths → append-safe reuse.
[ -f "$CSV" ] && [ "$(head -1 "$CSV")" = "$HDR" ] || echo "$HDR" > "$CSV"
for k in "${KEYS[@]}"; do
  repo="${HFREPO[$k]:-}"; [ -d "$repo/snapshots" ] || { echo "::skip $k (no HF repo)"; continue; }
  snap="snapshots/$( { cat "$repo/refs/main" 2>/dev/null || ls -t "$repo/snapshots" | head -1; } )"
  echo ">>> $k: vllm bench latency  (in$INLEN/out1, in$((INLEN*2))/out1, in$INLEN/out$OUTLEN)"
  la=$(lat "$k" "$repo" "$snap" "$INLEN" 1)          # prefill INLEN + 1 decode + overhead
  lb=$(lat "$k" "$repo" "$snap" $((INLEN*2)) 1)      # prefill 2*INLEN + 1 decode + overhead
  lc=$(lat "$k" "$repo" "$snap" "$INLEN" "$OUTLEN")  # prefill INLEN + OUTLEN decode + overhead
  pp=""; tg=""; e2e=""
  # CLEAN prefill: (lb-la) is the marginal cost of INLEN extra prompt tokens, so
  # the fixed per-call overhead AND the shared 1 decode cancel -> pure prefill.
  [ -n "$la" ] && [ -n "$lb" ] && awk "BEGIN{exit !($lb>$la)}" && \
    pp=$(python3 -c "print(round($INLEN/($lb-$la),1))" 2>/dev/null)
  [ -n "$la" ] && [ -n "$lc" ] && awk "BEGIN{exit !($lc>$la)}" && \
    tg=$(python3 -c "print(round(($OUTLEN-1)/($lc-$la),1))" 2>/dev/null)
  [ -n "$lc" ] && e2e=$(python3 -c "print(round($OUTLEN/$lc,1))" 2>/dev/null)
  # Replace this model's row ONLY after a real measurement — if all three latency
  # runs failed (docker OOM, missing image, absent JSON) la/lb/lc are empty and we
  # keep the previously-collected row rather than overwrite it with an all-blank one.
  if [ -z "$la" ] && [ -z "$lb" ] && [ -z "$lc" ]; then echo "::skip $k (all latency runs failed — keeping prior row)"; continue; fi
  awk -F, -v k="$k" 'NR==1 || $1!=k' "$CSV" > "$CSV.tmp" && mv "$CSV.tmp" "$CSV"
  echo "$k,$pp,$tg,$e2e" | tee -a "$CSV"
done
echo "=== wrote $CSV ==="
