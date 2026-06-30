# Benchmarking

`basert bench` measures prefill and decode throughput for a model.

```sh
basert bench <model> [-p N] [-n N] [-r N] [-w N] [--paged-kv]
```

| Flag | Purpose |
| --- | --- |
| `-p N` | Prefill length (prompt tokens). |
| `-n N` | Number of tokens to generate (decode). |
| `-r N` | Repetitions (averaged). |
| `-w N` | Warm-up iterations. |
| `--paged-kv` | Benchmark the paged KV cache path. |

```sh
basert bench Qwen/Qwen3-4B -p 512 -n 128 -r 3
```

The output reports prefill tokens/sec (prompt processing) and decode tokens/sec
(generation), which are the two numbers that matter for latency and throughput.

## Reproducible benchmark scripts

The [`benchmarks/`](https://github.com/basecompute/baseRT/tree/main/benchmarks)
directory has scripts that drive `basert bench` across models and context
lengths, plus reference results under `benchmarks/results/`. They expect the
engine bundle unpacked into `build/` (so `build/basert-bench` exists) — see
[Installation](../getting-started/installation.md). Start with
`benchmarks/scripts/` and the `benchmarks/README.md`.

## Methodology notes

- Use controlled token counts (`-p`/`-n`) and several repetitions (`-r`) with a
  warm-up (`-w`) so the first-call allocation cost doesn't skew results.
- Prefill and decode are reported separately; they stress different parts of the
  engine (compute-bound matmuls vs. memory-bound single-token steps).
- For server-style throughput, benchmark under `basert serve
  --continuous-batching` with concurrent clients rather than the single-stream
  `bench` tool.
