#!/usr/bin/env bash
# baseRT single-stream prefill/decode on GB10 via baseRT_cuda_bench, for the
# canonical model set at BOTH quants (cuda-q4mix = 4-bit, cuda-q8 = 8-bit). The
# 8-bit rows are the fair counterpart to vLLM fp8 (gb10_vllm_prefill_decode.sh);
# the 4-bit rows pair with llama.cpp Q4 (bench_2way). pp512 + tg128, r=5,
# page-cache reclaimed first. Writes benchmarks/gb10/basert-prefill-decode.csv.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BIN="${BASERT_BIN:-build-cuda/baseRT_cuda_bench}"
OUT="${OUT:-benchmarks/gb10}"
M="$HOME/models"
RECLAIM="$(dirname "$0")/reclaim_page_cache.sh"

# key -> "q4mix_bundle q8_bundle"
declare -A BUN=(
  [gemma-3-1b]="gemma-3-1b-it-cuda-q4mix gemma-3-1b-it-cuda-q8"
  [gemma-4-e2b]="gemma-4-E2B-it-cuda-q4mix gemma-4-E2B-it-cuda-q8"
  [gemma-4-26b]="gemma-4-26B-A4B-it-cuda-q4mix gemma-4-26B-A4B-it-cuda-q8"
  [llama-3.2-1b]="Llama-3.2-1B-Instruct-cuda-q4mix Llama-3.2-1B-Instruct-cuda-q8"
  [llama-3.2-3b]="Llama-3.2-3B-Instruct-cuda-q4mix Llama-3.2-3B-Instruct-cuda-q8"
  [qwen3-0.6b]="Qwen3-0.6B-cuda-q4mix Qwen3-0.6B-cuda-q8"
  [qwen3-30b-a3b]="Qwen3-30B-A3B-Instruct-2507-cuda-q4mix Qwen3-30B-A3B-Instruct-2507-cuda-q8"
  [qwen3.5-2b]="Qwen3.5-2B-cuda-q4mix Qwen3.5-2B-cuda-q8"
  [qwen3.5-35b-a3b]="Qwen3.5-35B-A3B-cuda-q4mix Qwen3.5-35B-A3B-cuda-q8"
  [qwen3.6-27b]="Qwen3.6-27B-cuda-q4mix Qwen3.6-27B-cuda-q8"
  [qwen3.6-35b]="Qwen3.6-35B-A3B-cuda-q4mix Qwen3.6-35B-A3B-cuda-q8"
)
KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(gemma-3-1b gemma-4-e2b gemma-4-26b \
  llama-3.2-1b llama-3.2-3b qwen3-0.6b qwen3-30b-a3b \
  qwen3.5-2b qwen3.5-35b-a3b qwen3.6-27b qwen3.6-35b)

mkdir -p "$OUT"  # the header redirect + later mv/tee fail silently (no set -e) if OUT is a fresh dir
CSV="$OUT/basert-prefill-decode.csv"
[ -f "$CSV" ] || echo "model,quant,pp512_tok_s,tg128_tok_s" > "$CSV"
run1() { # bundle -> "pp tg"
  local out pp tg
  # Reclaim page cache before EACH bundle: GB10 unified memory decode drops
  # 20-25% under high cache, and earlier bundles in this 22-run loop leave the
  # cache hot — so a once-at-start reclaim would bias later rows. Keeps rows
  # comparable (matches the "reclaim before every engine invocation" claim).
  # Redirect BOTH streams: the helper prints "(reclaiming ...)" to STDOUT, which
  # this captured function must not leak into its `pp tg` result.
  bash "$RECLAIM" 40 >/dev/null 2>&1 || true
  out=$("$BIN" "$M/$1.base" -p 512 -n 128 -r 5 2>&1)
  pp=$(echo "$out" | awk -F'|' '$4 ~ /pp512/ {gsub(/[^0-9.]/,"",$5);print substr($5,1,index($5,".")+2)}' | tail -1)
  tg=$(echo "$out" | awk -F'|' '$4 ~ /tg128/ {gsub(/[^0-9.]/,"",$5);print substr($5,1,index($5,".")+2)}' | tail -1)
  echo "$pp $tg"
}
for k in "${KEYS[@]}"; do
  read -r q4b q8b <<< "${BUN[$k]}"
  for pair in "q4mix:$q4b" "q8:$q8b"; do
    quant="${pair%%:*}"; bun="${pair#*:}"
    [ -f "$M/$bun.base" ] || { echo "::skip $k/$quant ($bun missing)"; continue; }
    read -r pp tg <<< "$(run1 "$bun")"
    # Only replace the existing (model,quant) row once BOTH rates parsed — an OOM
    # or a bench that emitted no pp512/tg128 row leaves them empty, and blowing
    # away a prior valid measurement with a blank is worse than keeping the old.
    if [ -z "$pp" ] || [ -z "$tg" ]; then echo "::skip $k/$quant (no measurement: pp='$pp' tg='$tg')"; continue; fi
    awk -F, -v k="$k" -v q="$quant" 'NR==1 || !($1==k && $2==q)' "$CSV" > "$CSV.tmp" && mv "$CSV.tmp" "$CSV"
    echo "$k,$quant,$pp,$tg" | tee -a "$CSV"
  done
done
echo "=== wrote $CSV ==="
