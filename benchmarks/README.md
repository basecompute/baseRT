# Benchmarks

Reproduce baseRT throughput numbers on your own hardware.

## Prerequisites

1. Download an engine release and unpack it into `build/` at the repo root
   (see the top-level README's "Getting the engine"). You need `build/basert-bench`.
2. Convert a model to `.base` (see `base-convert/`) into `models/`.
3. Python 3 (the output parser uses only the standard library).

## Run

```sh
# benchmark every models/*.base, 5 reps, pp ∈ {128,256,512}, tg 128
benchmarks/scripts/run_bench.sh

# or a specific model / custom sweep
PP_VALS="128 512" REPS=3 benchmarks/scripts/run_bench.sh models/your-model.base
```

The runner accepts one or more explicit `.base` paths. With no arguments it
benchmarks every `models/*.base` file. These environment variables control a
run:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PP_VALS` | `128 256 512` | Space-separated prefill lengths. |
| `TG_VAL` | `128` | Decode length. |
| `REPS` | `5` | Measured repetitions. |
| `WARMUP` | `1` | Warm-up iterations. |
| `BASERT` | `build/basert-bench` | Benchmark executable. |
| `RESULTS` | `benchmarks/results/<arch>_baseRT.csv` | Output CSV path. |

The output parser requires exactly one expected `ppN` and `tgN` row from every
invocation. A benchmark or output-format failure therefore stops the run instead
of publishing a partial CSV.

Results are written to `benchmarks/results/<arch>_baseRT.csv` with columns
`model,size_mb,engine,test,tok_per_sec,stddev`. `ppN` rows are prefill
throughput at prompt length N; `tgN` rows are independent decode
(token-generation) microbenchmarks. A `tgN` result is not decode-after-prefill
for the adjacent `ppN` row, even though both rows come from the same invocation.
These single-stream microbenchmarks are not a production server workload; use
concurrent clients against `basert serve --continuous-batching` to measure
server capacity and scheduling behavior.

## Example results

`results/m1-max-64gb_baseRT.csv` contains BaseRT 0.1.7 results for
Qwen3.6-35B-A3B `default-q4` on an M1 Max with 64 GB unified memory; its adjacent
[metadata file](results/m1-max-64gb_baseRT.md) records the host, model artifact,
engine, and methodology. `results/m4-pro_baseRT.csv` and
`results/m3-base_baseRT.csv` hold reference numbers from earlier runs (dated by
commit). They are not continuously republished — regenerate locally for an
apples-to-apples comparison with your build and models.

## Tests

The parser and runner tests use only the Python standard library and a fake
benchmark executable; they do not require an engine release or model:

```sh
python3 -m unittest discover -s benchmarks/tests -v
```

## Legacy / comparison scripts

The other scripts in `scripts/` (`mlx_benchmark.sh`, `gguf_benchmark.sh`,
`three_way_benchmark.sh`, `uzu_benchmark.sh`, …) compare against external engines
(mlx-lm, llama.cpp, uzu). They require those tools installed and predate the
`.base`-only loader, so they reference GGUF inputs — keep them as a reference for
methodology rather than a turnkey run.
