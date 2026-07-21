# M1 Max 64 GB — Qwen3.6-35B-A3B

Metadata for [`m1-max-64gb_baseRT.csv`](m1-max-64gb_baseRT.csv). The complete
second-run console output is preserved in
[`m1-max-64gb_baseRT.raw.md`](m1-max-64gb_baseRT.raw.md).

## Environment

- **Run date:** 2026-07-21
- **Host:** Apple M1 Max, 64 GB unified memory
- **OS:** macOS 26.4.1
- **Engine:** BaseRT 0.1.7, official macOS arm64 release
- **Engine binary SHA-256:** `f81d0ddd3c3eae1ff43f734aa0c76786b8b2e9b9fe3e1c5ec28f8d581a972721`
- **Base repository revision:** `1e923026`
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Source revision:** `995ad96eacd98c81ed38be0c5b274b04031597b0`
- **Conversion profile:** `default-q4`
- **Conversion command:** `basert pull Qwen/Qwen3.6-35B-A3B`
- **Converter version:** `base-convert v0.1.6` (invoked by `basert pull v0.1.6`)
- **`.base` file size:** 20,699,643,904 bytes
- **`.base` SHA-256:** `74c4a9256e64971e919c41b78b03ff8e0ec74b4330058113f5785097706ed1d0`

## Invocation

```sh
PP_VALS="128 256 512 2048 6000 43000" \
TG_VAL=128 REPS=5 WARMUP=1 \
benchmarks/scripts/run_bench.sh /path/to/qwen3.6-35b-a3b-default-q4.base
```

Each `tok_per_sec` value is the mean throughput over five measured repetitions
after one warm-up iteration. The complete sweep was repeated independently
through `run_bench.sh`; all throughput values reproduced within 1.2%, and the
CSV below records that second end-to-end run. `stddev` is copied from the
engine-reported standard deviation. `size_mb` is copied verbatim from the
corresponding `basert-bench` output row's `size` column; it is not the `.base`
file size. The reported value grows from 20,889 MiB
to 21,651 MiB for the longest prompt sweep, so it is intentionally retained per
row rather than replaced with one file-level value.

## Interpretation

`basert-bench` reports an independent prefill microbenchmark (`ppN`) and decode
microbenchmark (`tgN`) in each invocation. A `tg128` row does **not** measure 128
decode tokens after the prompt length named by the adjacent `ppN` row. The
repeated `tg128` rows are retained as separate observations from each invocation.

These are single-stream engine microbenchmarks. They do not measure concurrent
requests, continuous batching, scheduling, HTTP overhead, time to first token,
or end-to-end latency. Use a concurrent client workload against `basert serve
--continuous-batching` for production server capacity claims.
