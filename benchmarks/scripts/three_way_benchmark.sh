#!/bin/bash
# Three-way benchmark: baseRT vs llama.cpp vs mlx-lm.
#
# - Sweeps pp ∈ {128, 256, 512, 1024, 2048} and tg=128 by default.
# - Pairs each model across the three formats (baseRT .base, GGUF, MLX HF repo).
# - Writes a flat CSV and a markdown summary.
#
# Reproduce (e.g. on an M3): fetch the artifacts from pinned public sources,
# then run a subset. baseRT .base come from the basecompute HF catalog, GGUFs
# from bartowski/ggml-org, MLX repos auto-pull on first use:
#   SET=q8 benchmarks/scripts/fetch_3way_models.sh   # q8 only; SET=all (default) is huge
#   MODELS="Llama-3.2-1B-Q8 Llama-3.2-3B-Q8 Qwen3-0.6B-Q8 Gemma-4-E2B-Q8" \
#     RESULTS=/tmp/q8.csv SUMMARY=/tmp/q8.md benchmarks/scripts/three_way_benchmark.sh
# MoE Q8 (Qwen3-30B/Gemma-4-26B) needs >24GB and OOMs on M3/M4 Pro — dense only.
#
# Env knobs:
#   BASERT_BIN   : path to basert-bench    (default: ../../build/basert-bench)
#   MODELS_DIR   : local model directory    (default: ../../models)
#   LLAMA_BENCH  : path to llama-bench      (default: llama-bench on PATH)
#   MLX_BENCH    : path to mlx_lm.benchmark (default: uv-run wrapper below)
#   PP_VALS      : prefill sweep            (default: "128 256 512 1024 2048")
#   TG_VAL       : decode length            (default: 128)
#   REPS         : trials per data point    (default: 5)
#   RESULTS      : CSV output               (default: /tmp/bench_3way.csv)
#   SUMMARY      : markdown output          (default: /tmp/bench_3way.md)
#   MODELS       : space-separated subset of MODEL_TABLE keys (default: all)
#   COOLDOWN     : seconds between models   (default: 20)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASERT_BIN="${BASERT_BIN:-$REPO_ROOT/build/basert-bench}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
# Default MLX wrapper: assumes uv is available and pulls mlx-lm in a throwaway env.
# Set MLX_BENCH=mlx_lm.benchmark if you have it on PATH.
MLX_BENCH="${MLX_BENCH:-uv run --python 3.12 --with mlx-lm --with mlx --no-project mlx_lm.benchmark}"

PP_VALS="${PP_VALS:-128 256 512 1024 2048}"
TG_VAL="${TG_VAL:-128}"
REPS="${REPS:-5}"
RESULTS="${RESULTS:-/tmp/bench_3way.csv}"
SUMMARY="${SUMMARY:-/tmp/bench_3way.md}"
COOLDOWN="${COOLDOWN:-20}"

# KV-cache precision, pinned identically on baseRT and llama.cpp so the decode
# comparison is like-for-like. baseRT's Auto resolves to Q8_0 and llama-bench
# defaults to f16, which is not a matched comparison; standardize on f16 (16).
# Verified on M1 Max that baseRT decode is KV-precision-insensitive (Q8 vs f16
# tg128 within noise even at pp2048), so this changes fairness, not baseRT's
# numbers. (16=f16, 8=Q8_0, 4=Q4_0.) mlx-lm uses its own default (f16).
KV_BITS="${KV_BITS:-16}"
case "$KV_BITS" in
  16) LLAMA_CACHE_TYPE="f16" ;;
  8)  LLAMA_CACHE_TYPE="q8_0" ;;
  4)  LLAMA_CACHE_TYPE="q4_0" ;;
  *)  echo "unknown KV_BITS=$KV_BITS (use 16|8|4)"; exit 1 ;;
esac

# MODEL_TABLE entries: label|baseRT_filename|gguf_filename|mlx_hf_repo
# Each component may be "-" to skip that engine for the model.
MODEL_TABLE=(
  "Llama-3.2-1B-Q4|Llama-3.2-1B-Q4.base|Llama-3.2-1B-Instruct-Q4_0.gguf|mlx-community/Llama-3.2-1B-Instruct-4bit"
  "Llama-3.2-3B-Q4|Llama-3.2-3B-Q4.base|Llama-3.2-3B-Instruct-Q4_0.gguf|mlx-community/Llama-3.2-3B-Instruct-4bit"
  "Qwen3-0.6B-Q4|Qwen3-0.6B-Q4_0.base|Qwen3-0.6B-Q4_0.gguf|mlx-community/Qwen3-0.6B-4bit"
#   "Gemma-3-1B-Q4|gemma-3-1b-it-Q4_K_M.base|gemma-3-1b-it-Q4_K_M.gguf|mlx-community/gemma-3-1b-it-4bit"
  # gemma-4 (NOT gemma-3n — a different model class) benchmarks fine under
  # mlx_lm.benchmark, so the gemma-4-E2B rows use the real gemma-4 mlx repos
  # (mlx-community/gemma-4-e2b-it-{4,8}bit) and the unsloth gemma-4 GGUFs.
  "Gemma-4-E2B-Q4|gemma-4-E2B-it-Q4_0.base|gemma-4-E2B-it-Q4_0.gguf|mlx-community/gemma-4-e2b-it-4bit"
  # MoE rows. llama.cpp side: Qwen has Q4_0 from bartowski; Gemma Q4_0 from
  # bartowski/google_gemma-4-26B-A4B-it-GGUF (was IQ4_XS before that existed).
  "Qwen3-30B-A3B-Q4|Qwen3-30B-A3B-Q4.base|Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf|mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
  # ── Qwen 3.5/3.6 (hybrid GDN) rows. baseRT .base from the published
  # basecompute catalog; GGUFs from bartowski (2B, 3.5-35B) / unsloth
  # (27B, 3.6-35B); mlx from the official mlx-community repos.
  # 27B/35B need >24GB (m5 / m1-max-64G class); 2B runs everywhere.
  "Qwen3.5-2B-Q4|Qwen3.5-2B-Base-Q4.base|gguf/Qwen3.5-2B/Qwen_Qwen3.5-2B-Q4_K_M.gguf|mlx-community/Qwen3.5-2B-4bit"
  "Qwen3.5-2B-Q8|Qwen3.5-2B-Base-Q8.base|gguf/Qwen3.5-2B/Qwen_Qwen3.5-2B-Q8_0.gguf|mlx-community/Qwen3.5-2B-8bit"
  "Qwen3.5-35B-A3B-Q4|Qwen3.5-35B-A3B-Q4.base|gguf/Qwen3.5-35B/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf|mlx-community/Qwen3.5-35B-A3B-4bit"
  "Qwen3.6-27B-Q4|Qwen3.6-27B-Q4.base|gguf/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf|mlx-community/Qwen3.6-27B-4bit"
  "Qwen3.6-35B-Q4|Qwen3.6-35B-A3B-Q4.base|gguf/Qwen3.6-35B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|mlx-community/Qwen3.6-35B-A3B-4bit"
  "Qwen3.6-27B-Q8|Qwen3.6-27B-Q8.base|gguf/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf|mlx-community/Qwen3.6-27B-8bit"
  "Gemma-4-26B-A4B-Q4|Gemma-4-26B-A4B-Q4.base|google_gemma-4-26B-A4B-it-Q4_0.gguf|mlx-community/gemma-4-26b-a4b-it-4bit"

  # ── Q8 (8-bit) rows ────────────────────────────────────────────────────
  # The MoE Q8 weights (Qwen3-30B-A3B ~32GB, Gemma-4-26B-A4B ~28GB) exceed the
  # 24GB working set on an M4 Pro and OOM at inference — those two rows only
  # run on big-memory boxes (M1 Max 64GB, GB10 121GB); elsewhere the missing
  # files skip them.
  # baseRT .base from the basecompute catalog (Q8); llama.cpp Q8_0 GGUF from
  # bartowski / ggml-org; mlx-community 8-bit repos.
  "Llama-3.2-1B-Q8|Llama-3.2-1B-Q8.base|Llama-3.2-1B-Instruct-Q8_0.gguf|mlx-community/Llama-3.2-1B-Instruct-8bit"
  "Llama-3.2-3B-Q8|Llama-3.2-3B-Q8.base|Llama-3.2-3B-Instruct-Q8_0.gguf|mlx-community/Llama-3.2-3B-Instruct-8bit"
  "Qwen3-0.6B-Q8|Qwen3-0.6B-Q8.base|Qwen3-0.6B-Q8_0.gguf|mlx-community/Qwen3-0.6B-8bit"
#   "Gemma-3-1B-Q8|gemma-3-1b-it-Q8.base|gemma-3-1b-it-Q8_0.gguf|mlx-community/gemma-3-1b-it-8bit"
  "Gemma-4-E2B-Q8|gemma-4-E2B-it-Q8.base|unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf|mlx-community/gemma-4-e2b-it-8bit"
  # MoE Q8 (big-memory boxes only — see note above). The Qwen 30B Q8 row uses
  # the Thinking-2507 variant across baseRT + GGUF (only published Q8 for this
  # architecture); mlx pairs the same-arch Instruct-2507 8-bit repo.
  "Qwen3-30B-A3B-Q8|Qwen3-30B-A3B-Q8.base|Qwen_Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf|mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit"
  "Gemma-4-26B-A4B-Q8|Gemma-4-26B-A4B-Q8.base|google_gemma-4-26B-A4B-it-Q8_0.gguf|mlx-community/gemma-4-26b-a4b-it-8bit"
)

# Optional subset: e.g. MODELS="Qwen3-0.6B-Q4 Llama-3.2-1B-Q4"
SELECTED="${MODELS:-}"

mkdir -p "$(dirname "$RESULTS")" "$(dirname "$SUMMARY")"
echo "model,engine,test,tok_per_sec,stddev" > "$RESULTS"

# ── parsers (Python — robust to BSD/GNU awk + pipe-table edge cases) ───────
# Engine outputs are written to a temp file; Python reads via env path so the
# heredoc (which IS python3's stdin source) doesn't fight with the data.
# NB: the XXXXXX placeholder must be the trailing component — macOS/BSD mktemp
# does not substitute X's when a suffix (.py) follows, so it would create a
# literal, non-unique file that a killed run leaves behind and breaks the next
# run. python3 runs a file regardless of extension, so no suffix is needed.
PARSER_PY="$(mktemp "${TMPDIR:-/tmp}/three_way_parser.XXXXXX")"
cat > "$PARSER_PY" <<'PY'
import os, re, sys, statistics
engine, label, tg, results_path, output_path = sys.argv[1:6]
with open(output_path) as f:
    out = f.read()

rows = []  # (test, tps, std)

if engine in ("baseRT", "llama.cpp"):
    for line in out.splitlines():
        if "|" not in line: continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2: continue
        test, tv = cells[-2], cells[-1]
        if not re.fullmatch(r"(pp|tg)\d+", test): continue
        m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", tv)
        if not m: continue
        rows.append((test, m.group(1), m.group(2)))
elif engine == "mlx-lm":
    pp_vals, tg_vals = [], []
    for line in out.splitlines():
        if not line.startswith("Trial "): continue
        for k, dest in (("prompt_tps=", pp_vals), ("generation_tps=", tg_vals)):
            i = line.find(k)
            if i < 0: continue
            j = line.find(",", i)
            try: dest.append(float(line[i+len(k):j if j>0 else None]))
            except ValueError: pass
    def stats(v):
        if not v: return None
        return statistics.fmean(v), statistics.pstdev(v) if len(v) > 1 else 0.0
    pp_stats = stats(pp_vals); tg_stats = stats(tg_vals)
    pp_tag = os.environ.get("MLX_PP_TAG", "")
    if pp_stats and pp_tag:
        rows.append((f"pp{pp_tag}", f"{pp_stats[0]:.2f}", f"{pp_stats[1]:.2f}"))
    if tg_stats:
        rows.append((f"tg{tg}", f"{tg_stats[0]:.2f}", f"{tg_stats[1]:.2f}"))

with open(results_path, "a") as f:
    for test, tps, std in rows:
        f.write(f"{label},{engine},{test},{tps},{std}\n")
PY
trap 'rm -f "$PARSER_PY"' EXIT

emit_rows() {
  # Args: engine label tg_val; stdin = engine output
  local engine="$1" label="$2" tg="$3"
  local tmpfile
  tmpfile="$(mktemp /tmp/three_way_out.XXXXXX)"
  cat > "$tmpfile"
  python3 "$PARSER_PY" "$engine" "$label" "$tg" "$RESULTS" "$tmpfile"
  rm -f "$tmpfile"
}

parse_baseRT() { emit_rows "baseRT"    "$2" "$TG_VAL" <<< "$1"; }
parse_llama()  { emit_rows "llama.cpp" "$2" "$TG_VAL" <<< "$1"; }
parse_mlx()    { MLX_PP_TAG="$3" emit_rows "mlx-lm" "$2" "$TG_VAL" <<< "$1"; }

# ── runners ────────────────────────────────────────────────────────────────
run_baseRT() {
  local model_path="$1" label="$2" pp="$3" tg="$4"
  echo "    baseRT pp${pp}/tg${tg}..."
  ${PRE_RUN_CMD:-true}
  local out
  out=$(BASERT_KV_BITS="$KV_BITS" "$BASERT_BIN" "$model_path" -p "$pp" -n "$tg" -r "$REPS" 2>&1) || {
    echo "      ERROR: basert-bench failed"; return
  }
  parse_baseRT "$out" "$label"
}

run_llama() {
  local model_path="$1" label="$2" pp="$3" tg="$4"
  echo "    llama.cpp pp${pp}/tg${tg}..."
  ${PRE_RUN_CMD:-true}
  local out
  out=$("$LLAMA_BENCH" -m "$model_path" -p "$pp" -n "$tg" -r "$REPS" \
        -ctk "$LLAMA_CACHE_TYPE" -ctv "$LLAMA_CACHE_TYPE" 2>&1) || {
    echo "      ERROR: llama-bench failed"; return
  }
  parse_llama "$out" "$label"
}

run_mlx() {
  local repo="$1" label="$2" pp="$3" tg="$4"
  echo "    mlx-lm pp${pp}/tg${tg}..."
  local out
  out=$($MLX_BENCH --model "$repo" --prompt-tokens "$pp" --generation-tokens "$tg" --num-trials "$REPS" 2>&1) || {
    echo "      ERROR: mlx_lm.benchmark failed"; return
  }
  parse_mlx "$out" "$label" "$pp"
}

# ── main loop ──────────────────────────────────────────────────────────────
total=${#MODEL_TABLE[@]}
idx=0
for row in "${MODEL_TABLE[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r label baseRT_file gguf_file mlx_repo <<< "$row"

  if [ -n "$SELECTED" ] && ! grep -qw "$label" <<< "$SELECTED"; then
    continue
  fi

  echo ""
  echo "[$idx/$total] === $label ==="
  baseRT_path="$MODELS_DIR/$baseRT_file"
  gguf_path="$MODELS_DIR/$gguf_file"

  have_baseRT=0; [ "$baseRT_file" != "-" ] && [ -f "$baseRT_path" ] && have_baseRT=1
  have_gguf=0;   [ "$gguf_file"   != "-" ] && [ -f "$gguf_path"   ] && have_gguf=1
  have_mlx=0;    [ "$mlx_repo"    != "-" ] && have_mlx=1

  if [ "$have_baseRT" = 0 ] && [ "$have_gguf" = 0 ] && [ "$have_mlx" = 0 ]; then
    echo "  SKIP — no engines have this model"
    continue
  fi
  [ "$have_baseRT" = 0 ] && echo "  (no baseRT: $baseRT_file)"
  [ "$have_gguf"   = 0 ] && echo "  (no GGUF: $gguf_file)"

  for pp in $PP_VALS; do
    echo "  pp=$pp tg=$TG_VAL r=$REPS"
    [ "$have_baseRT" = 1 ] && run_baseRT "$baseRT_path" "$label" "$pp" "$TG_VAL"
    [ "$have_gguf"   = 1 ] && run_llama  "$gguf_path"   "$label" "$pp" "$TG_VAL"
    [ "$have_mlx"    = 1 ] && run_mlx    "$mlx_repo"    "$label" "$pp" "$TG_VAL"
  done

  if [ "$idx" -lt "$total" ] && [ "$COOLDOWN" -gt 0 ]; then
    echo "  cooling ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

# ── markdown summary ───────────────────────────────────────────────────────
SUMMARY_PY="$(mktemp "${TMPDIR:-/tmp}/three_way_summary.XXXXXX")"
cat > "$SUMMARY_PY" <<'PY'
import csv, sys, collections, statistics, platform, subprocess
results_path, summary_path = sys.argv[1], sys.argv[2]

rows = list(csv.DictReader(open(results_path)))
if not rows:
    open(summary_path, "w").write("# Benchmark\n\nNo results.\n")
    print(f"\n(no rows in {results_path})"); sys.exit(0)

# {(model, test): {engine: [tps, tps, ...]}}   — list so tg128 across pp runs averages
by_cell = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try:
        by_cell[(r["model"], r["test"])][r["engine"]].append(float(r["tok_per_sec"]))
    except ValueError:
        pass

def mean_of(d, engine):
    vs = d.get(engine)
    return statistics.fmean(vs) if vs else None

models = sorted({m for (m, _) in by_cell})
tests = sorted({t for (_, t) in by_cell},
               key=lambda x: (0 if x.startswith("pp") else 1, int(x[2:])))
engines = ["baseRT", "llama.cpp", "mlx-lm"]

# Machine fingerprint
try:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
except Exception:
    chip = platform.processor() or "unknown"

lines = [
    "# Three-way benchmark — baseRT vs llama.cpp vs mlx-lm",
    "",
    f"Machine: {chip} ({platform.machine()}, {platform.system()} {platform.release()})",
    f"Data points: {len(rows)} (reps=5 per data point; tg128 cells average across pp sweeps)",
    "",
]

for m in models:
    lines.append(f"## {m}")
    lines.append("| Test | " + " | ".join(engines) + " | baseRT vs llama.cpp | baseRT vs mlx-lm |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for t in tests:
        cell = by_cell.get((m, t), {})
        if not cell:
            continue
        vals = [mean_of(cell, e) for e in engines]
        row = [t]
        for v in vals:
            row.append(f"{v:.1f}" if v is not None else "—")
        base = mean_of(cell, "baseRT")
        for ref in (mean_of(cell, "llama.cpp"), mean_of(cell, "mlx-lm")):
            if base is None or ref is None or ref == 0:
                row.append("—")
            else:
                row.append(f"{100.0 * (base / ref - 1.0):+.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

# Decode-only headline
lines.append("## Decode (tg128) headline")
lines.append("| Model | baseRT | llama.cpp | mlx-lm | vs llama.cpp | vs mlx-lm |")
lines.append("|---|---:|---:|---:|---:|---:|")
for m in models:
    cell = by_cell.get((m, "tg128"), {})
    base = mean_of(cell, "baseRT")
    l = mean_of(cell, "llama.cpp")
    x = mean_of(cell, "mlx-lm")
    def fmt(v): return f"{v:.1f}" if v else "—"
    def pct(a, b):
        if a is None or b is None or b == 0: return "—"
        return f"{100.0 * (a / b - 1.0):+.1f}%"
    lines.append(f"| {m} | {fmt(base)} | {fmt(l)} | {fmt(x)} | {pct(base, l)} | {pct(base, x)} |")

open(summary_path, "w").write("\n".join(lines) + "\n")
print(f"\nCSV : {results_path}")
print(f"MD  : {summary_path}")
PY
python3 "$SUMMARY_PY" "$RESULTS" "$SUMMARY"
rm -f "$SUMMARY_PY"

echo ""
echo "=== DONE ==="
