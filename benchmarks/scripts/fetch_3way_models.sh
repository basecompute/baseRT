#!/bin/bash
# Fetch the model artifacts the three-way benchmark needs, into MODELS_DIR,
# from pinned public sources. Idempotent: files already present are skipped.
#
# After this, run:  MODELS="<labels>" benchmarks/scripts/three_way_benchmark.sh
#
# Per engine:
#   baseRT   : .base pulled from the PUBLISHED basecompute HF catalog
#              (basecompute/<repo>) — no local conversions. Every baseRT model
#              in three_way_benchmark.sh MODEL_TABLE has a published repo,
#              including the MoE (Qwen3-30B, Gemma-4-26B) and hybrid-GDN
#              (Qwen3.5 / Qwen3.6) rows.
#   llama.cpp: Q4/Q8 GGUF from bartowski / ggml-org / unsloth
#   mlx-lm   : nothing to fetch — mlx_lm.benchmark pulls the HF repo on first use
#
# Env: MODELS_DIR (default ../../models), SET=q4|q8|all (default all).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/models}"
SET="${SET:-all}"
mkdir -p "$MODELS_DIR"

# rows: local_filename | hf_repo | remote_filename
# (local_filename matches the name used in three_way_benchmark.sh MODEL_TABLE;
#  a local_filename may contain a subpath, e.g. gguf/Qwen3.5-2B/…)
#
# baseRT throughput is independent of the exact quant scheme / instruct-vs-base
# variant (same architecture and size), so the .base saved under the table's
# label may pair with a Q4_K_M / Q8_0 GGUF and a 4/8-bit MLX repo.

# ── baseRT .base — all from the published basecompute catalog ──────────────
Q4_BASE=(
  # dense
  "Llama-3.2-1B-Q4.base|basecompute/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-Q4.base"
  "Llama-3.2-3B-Q4.base|basecompute/Llama-3.2-3B-Instruct|Llama-3.2-3B-Instruct-Q4.base"
  "Qwen3-0.6B-Q4_0.base|basecompute/Qwen3-0.6B|Qwen3-0.6B-Q4.base"
  "gemma-4-E2B-it-Q4_0.base|basecompute/gemma-4-E2B-it|gemma-4-E2B-it-Q4.base"
  # MoE
  "Qwen3-30B-A3B-Q4.base|basecompute/Qwen3-30B-A3B-Instruct-2507|Qwen3-30B-A3B-Instruct-2507-Q4.base"
  "Gemma-4-26B-A4B-Q4.base|basecompute/gemma-4-26B-A4B-it|gemma-4-26B-A4B-it-Q4.base"
  # hybrid-GDN (Qwen3.5 / Qwen3.6)
  "Qwen3.5-2B-Base-Q4.base|basecompute/Qwen3.5-2B-Base|Qwen3.5-2B-Base-Q4.base"
  "Qwen3.5-35B-A3B-Q4.base|basecompute/Qwen3.5-35B-A3B|Qwen3.5-35B-A3B-Q4.base"
  "Qwen3.6-27B-Q4.base|basecompute/Qwen3.6-27B|Qwen3.6-27B-Q4.base"
  "Qwen3.6-35B-A3B-Q4.base|basecompute/Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-Q4.base"
)
Q8_BASE=(
  # dense
  "Llama-3.2-1B-Q8.base|basecompute/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct-Q8.base"
  "Llama-3.2-3B-Q8.base|basecompute/Llama-3.2-3B-Instruct|Llama-3.2-3B-Instruct-Q8.base"
  "Qwen3-0.6B-Q8.base|basecompute/Qwen3-0.6B|Qwen3-0.6B-Q8.base"
  "gemma-4-E2B-it-Q8.base|basecompute/gemma-4-E2B-it|gemma-4-E2B-it-Q8.base"
  # MoE. The Instruct-2507 30B has no published Q8; the Thinking-2507 variant
  # does and is the same architecture/size, so throughput is identical.
  "Qwen3-30B-A3B-Q8.base|basecompute/Qwen3-30B-A3B-Thinking-2507|Qwen3-30B-A3B-Thinking-2507-Q8.base"
  "Gemma-4-26B-A4B-Q8.base|basecompute/gemma-4-26B-A4B-it|gemma-4-26B-A4B-it-Q8.base"
  # hybrid-GDN
  "Qwen3.5-2B-Base-Q8.base|basecompute/Qwen3.5-2B-Base|Qwen3.5-2B-Base-Q8.base"
  "Qwen3.6-27B-Q8.base|basecompute/Qwen3.6-27B|Qwen3.6-27B-Q8.base"
)

# ── llama.cpp GGUF ─────────────────────────────────────────────────────────
Q4_GGUF=(
  # dense
  "Llama-3.2-1B-Instruct-Q4_0.gguf|bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q4_0.gguf"
  "Llama-3.2-3B-Instruct-Q4_0.gguf|bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q4_0.gguf"
  "Qwen3-0.6B-Q4_0.gguf|ggml-org/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q4_0.gguf"
  "gemma-4-E2B-it-Q4_0.gguf|unsloth/gemma-4-E2B-it-GGUF|gemma-4-E2B-it-Q4_0.gguf"
  # MoE
  "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf|bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF|Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
  "google_gemma-4-26B-A4B-it-Q4_0.gguf|bartowski/google_gemma-4-26B-A4B-it-GGUF|google_gemma-4-26B-A4B-it-Q4_0.gguf"
  # hybrid-GDN
  "gguf/Qwen3.5-2B/Qwen_Qwen3.5-2B-Q4_K_M.gguf|bartowski/Qwen_Qwen3.5-2B-GGUF|Qwen_Qwen3.5-2B-Q4_K_M.gguf"
  "gguf/Qwen3.5-35B/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf|bartowski/Qwen_Qwen3.5-35B-A3B-GGUF|Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf"
  "gguf/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf|unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-Q4_K_M.gguf"
  "gguf/Qwen3.6-35B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)
Q8_GGUF=(
  # dense
  "Llama-3.2-1B-Instruct-Q8_0.gguf|bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q8_0.gguf"
  "Llama-3.2-3B-Instruct-Q8_0.gguf|bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q8_0.gguf"
  "Qwen3-0.6B-Q8_0.gguf|ggml-org/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q8_0.gguf"
  "unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf|unsloth/gemma-4-E2B-it-GGUF|gemma-4-E2B-it-Q8_0.gguf"
  # MoE (see baseRT note: 30B Q8 uses the Thinking-2507 variant)
  "Qwen_Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf|bartowski/Qwen_Qwen3-30B-A3B-Thinking-2507-GGUF|Qwen_Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf"
  "google_gemma-4-26B-A4B-it-Q8_0.gguf|bartowski/google_gemma-4-26B-A4B-it-GGUF|google_gemma-4-26B-A4B-it-Q8_0.gguf"
  # hybrid-GDN
  "gguf/Qwen3.5-2B/Qwen_Qwen3.5-2B-Q8_0.gguf|bartowski/Qwen_Qwen3.5-2B-GGUF|Qwen_Qwen3.5-2B-Q8_0.gguf"
  "gguf/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf|unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-Q8_0.gguf"
)

fetch() {  # local_name|repo|remote_name
  local row="$1"; IFS='|' read -r local repo remote <<< "$row"
  local dest="$MODELS_DIR/$local"
  if [ -f "$dest" ]; then echo "  have   $local"; return; fi
  echo "  fetch  $local  <-  $repo/$remote"
  mkdir -p "$(dirname "$dest")"
  local tmp; tmp="$(mktemp -d)"
  if hf download "$repo" "$remote" --local-dir "$tmp" >/dev/null 2>&1; then
    cp "$tmp/$remote" "$dest"
  else
    echo "    WARN: failed to fetch $repo/$remote (skipping)"
  fi
  rm -rf "$tmp"
}

echo "=== fetch 3-way models (set=$SET) into $MODELS_DIR ==="
# Inline expansion (no array indirection — works on macOS bash 3.2).
case "$SET" in
  q4)  ROWS=( "${Q4_BASE[@]}" "${Q4_GGUF[@]}" ) ;;
  q8)  ROWS=( "${Q8_BASE[@]}" "${Q8_GGUF[@]}" ) ;;
  all) ROWS=( "${Q4_BASE[@]}" "${Q4_GGUF[@]}" "${Q8_BASE[@]}" "${Q8_GGUF[@]}" ) ;;
  *)   echo "unknown SET=$SET (use q4|q8|all)"; exit 1 ;;
esac
for r in "${ROWS[@]}"; do fetch "$r"; done
echo "mlx-lm repos are pulled on first run by mlx_lm.benchmark (no local fetch)."
echo "Done. Labels (see MODEL_TABLE in three_way_benchmark.sh for the full set):"
echo "  Q4: Llama-3.2-1B-Q4 Llama-3.2-3B-Q4 Qwen3-0.6B-Q4 Gemma-4-E2B-Q4 \\"
echo "      Qwen3-30B-A3B-Q4 Gemma-4-26B-A4B-Q4 Qwen3.5-2B-Q4 Qwen3.5-35B-A3B-Q4 \\"
echo "      Qwen3.6-27B-Q4 Qwen3.6-35B-Q4"
echo "  Q8: Llama-3.2-1B-Q8 Llama-3.2-3B-Q8 Qwen3-0.6B-Q8 Gemma-4-E2B-Q8 \\"
echo "      Qwen3-30B-A3B-Q8 Gemma-4-26B-A4B-Q8 Qwen3.5-2B-Q8 Qwen3.6-27B-Q8"
echo "Run: MODELS=\"<labels>\" $SCRIPT_DIR/three_way_benchmark.sh"
