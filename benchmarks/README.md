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
