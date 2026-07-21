# Benchmarks

Reproduce baseRT throughput numbers on your own hardware.

## Prerequisites

1. Download an engine release and unpack it into `build/` at the repo root
   (see the top-level README's "Getting the engine"). You need `build/basert-bench`.
2. Convert a model to `.base` (see `base-convert/`) into `models/`.

## Run

```sh
# benchmark every models/*.base, 5 reps, pp ∈ {128,256,512}, tg 128
benchmarks/scripts/run_bench.sh

# or a specific model / custom sweep
PP_VALS="128 512" REPS=3 benchmarks/scripts/run_bench.sh models/your-model.base
```

Results are written to `benchmarks/results/<arch>_baseRT.csv` with columns
`model,size_mb,engine,test,tok_per_sec,stddev`. `ppN` rows are prefill
throughput at prompt length N; `tgN` rows are decode (token-generation)
throughput.

## Tool-state regression smoke

`tool_state_smoke.py` checks whether repeated tool schemas preserve isolated
arguments between requests. It sends distinct `lookup_key` sentinels in 10
sequential requests, then concurrent batches of 2 and 4 by default. Failures are
consistent with cross-request state leakage, but this diagnostic does not prove
an engine root cause by itself. It is opt-in for tool-capable models; plain chat
models are expected to fail it:

```sh
python3 benchmarks/scripts/tool_state_smoke.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model model.base \
  --sequential 10 --concurrency 2 4 \
  --timeout 300 --max-tokens 2048
```

Set `BASERT_API_KEY` in the environment when the server requires a bearer
token. The harness uses only the Python standard library. Each mismatch is
printed as a `FAIL` JSON record, followed by a `SUMMARY` JSON record, and any
mismatch makes the process exit nonzero.

The manual **Server smoke** workflow can run this check after its basic chat
completion by enabling `run_tool_state_smoke`. Pure unit tests use a local fake
HTTP server and do not require an engine or model:

```sh
python3 -m unittest benchmarks.tests.test_tool_state_smoke -v
```

## Example results

`results/m4-pro_baseRT.csv` and `results/m3-base_baseRT.csv` hold reference
numbers from earlier runs (dated by commit). They are not continuously
republished — regenerate locally for an apples-to-apples comparison with your
build and models.

## Legacy / comparison scripts

The other scripts in `scripts/` (`mlx_benchmark.sh`, `gguf_benchmark.sh`,
`three_way_benchmark.sh`, `uzu_benchmark.sh`, …) compare against external engines
(mlx-lm, llama.cpp, uzu). They require those tools installed and predate the
`.base`-only loader, so they reference GGUF inputs — keep them as a reference for
methodology rather than a turnkey run.
