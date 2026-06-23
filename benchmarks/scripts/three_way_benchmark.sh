#!/bin/bash
# Three-way benchmark: baseRT vs llama.cpp vs mlx-lm.
#
# - Sweeps pp ∈ {128, 256, 512, 1024, 2048} and tg=128 by default.
# - Pairs each model across the three formats (baseRT .base, GGUF, MLX HF repo).
# - Writes a flat CSV and a markdown summary.
#
# Env knobs:
#   BASERT_BIN   : path to baseRT_bench    (default: ../../build/baseRT_bench)
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

BASERT_BIN="${BASERT_BIN:-$REPO_ROOT/build/baseRT_bench}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
# Default MLX wrapper: assumes uv is available and pulls mlx-lm in a throwaway env.
# Set MLX_BENCH=mlx_lm.benchmark if you have it on PATH.
MLX_BENCH="${MLX_BENCH:-uv run --with mlx-lm --with mlx --no-project mlx_lm.benchmark}"

PP_VALS="${PP_VALS:-128 256 512 1024 2048}"
TG_VAL="${TG_VAL:-128}"
REPS="${REPS:-5}"
RESULTS="${RESULTS:-/tmp/bench_3way.csv}"
SUMMARY="${SUMMARY:-/tmp/bench_3way.md}"
COOLDOWN="${COOLDOWN:-20}"

# MODEL_TABLE entries: label|baseRT_filename|gguf_filename|mlx_hf_repo
# Each component may be "-" to skip that engine for the model.
MODEL_TABLE=(
  "Llama-3.2-1B-Q4|Llama-3.2-1B-Q4.base|Llama-3.2-1B-Instruct-Q4_0.gguf|mlx-community/Llama-3.2-1B-Instruct-4bit"
  "Llama-3.2-3B-Q4|Llama-3.2-3B-Q4.base|Llama-3.2-3B-Instruct-Q4_0.gguf|mlx-community/Llama-3.2-3B-Instruct-4bit"
  "Qwen3-0.6B-Q4|Qwen3-0.6B-Q4_0.base|Qwen3-0.6B-Q4_0.gguf|mlx-community/Qwen3-0.6B-4bit"
  "Gemma-3-1B-Q4|gemma-3-1b-it-Q4_K_M.base|gemma-3-1b-it-Q4_K_M.gguf|mlx-community/gemma-3-1b-it-4bit"
  "Gemma-4-E2B-Q4|gemma-4-E2B-it-Q4_0.base|gemma-4-E2B-it-Q4_0.gguf|mlx-community/gemma-3n-E2B-it-4bit"
  # MoE rows. llama.cpp side: Qwen has Q4_0 from bartowski; Gemma's only
  # widely-available <Q5 GGUF is IQ4_XS from CelesteImperia (closest 4-bit).
  "Qwen3-30B-A3B-Q4|Qwen3-30B-A3B-Q4.base|Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf|mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
  "Gemma-4-26B-A4B-Q4|Gemma-4-26B-A4B-Q4.base|Gemma-4-26B-MoE-IQ4_XS.gguf|mlx-community/gemma-4-26b-a4b-it-4bit"
)

# Optional subset: e.g. MODELS="Qwen3-0.6B-Q4 Llama-3.2-1B-Q4"
SELECTED="${MODELS:-}"

mkdir -p "$(dirname "$RESULTS")" "$(dirname "$SUMMARY")"
echo "model,engine,test,tok_per_sec,stddev" > "$RESULTS"

# ── parsers (Python — robust to BSD/GNU awk + pipe-table edge cases) ───────
# Engine outputs are written to a temp file; Python reads via env path so the
# heredoc (which IS python3's stdin source) doesn't fight with the data.
PARSER_PY="$(mktemp /tmp/three_way_parser.XXXXXX.py)"
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
  local out
  out=$("$BASERT_BIN" "$model_path" -p "$pp" -n "$tg" -r "$REPS" 2>&1) || {
    echo "      ERROR: baseRT_bench failed"; return
  }
  parse_baseRT "$out" "$label"
}

run_llama() {
  local model_path="$1" label="$2" pp="$3" tg="$4"
  echo "    llama.cpp pp${pp}/tg${tg}..."
  local out
  out=$("$LLAMA_BENCH" -m "$model_path" -p "$pp" -n "$tg" -r "$REPS" 2>&1) || {
    echo "      ERROR: llama-bench failed"; return
  }
  parse_llama "$out" "$label"
}

run_mlx() {
  local repo="$1" label="$2" pp="$3" tg="$4"
  echo "    mlx-lm pp${pp}/tg${tg}..."
  local out
  out=$($MLX_BENCH --model "$repo" -p "$pp" -g "$tg" -n "$REPS" 2>&1) || {
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
SUMMARY_PY="$(mktemp /tmp/three_way_summary.XXXXXX.py)"
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
