#!/bin/bash
# GB10 (CUDA) two-way wrapper around three_way_benchmark.sh:
# baseRT (build-cuda/baseRT_cuda_bench) vs llama.cpp CUDA; mlx-lm is not
# available on Linux/CUDA so its leg is disabled via MLX_BENCH=false
# (fails instantly per row and is skipped in results).
#
# Same MODEL_TABLE / protocol as the M4 Pro run: pp 128..2048, tg128, 5 reps.
# Fetch artifacts first: SET=all benchmarks/scripts/fetch_3way_models.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export BASERT_BIN="${BASERT_BIN:-$REPO_ROOT/build-cuda/baseRT_cuda_bench}"
export LLAMA_BENCH="${LLAMA_BENCH:-$HOME/llama.cpp/build/bin/llama-bench}"
export MLX_BENCH=false
export RESULTS="${RESULTS:-$REPO_ROOT/benchmarks/gb10/bench_2way.csv}"
export SUMMARY="${SUMMARY:-$REPO_ROOT/benchmarks/gb10/bench_2way.md}"
# GB10 unified memory: heavy page cache depresses large-model decode 20-25%.
# Reclaim before every engine invocation when cache exceeds 40GB.
export PRE_RUN_CMD="${PRE_RUN_CMD:-bash $SCRIPT_DIR/reclaim_page_cache.sh 40}"

exec bash "$SCRIPT_DIR/three_way_benchmark.sh"
