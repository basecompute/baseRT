# Benchmarks

Reproduce baseRT throughput numbers on your own hardware.

## Run

Install baseRT (see [Installation](../docs/getting-started/installation.md)),
then benchmark a model with `basert bench`:

```sh
basert bench basecompute/Qwen3-4B              # pp512 / tg128, 5 reps
basert bench ./models/your-model.base -p 128 -n 128 -r 3
```

`ppN` is prefill throughput at prompt length N; `tgN` is decode
(token-generation) throughput. See
[Benchmarking](../docs/guides/benchmarking.md) for every flag and for
ComputeArena, which saves a signed report.

## Example results

`results/m4-pro_baseRT.csv` and `results/m3-base_baseRT.csv` hold reference
numbers from earlier runs (dated by commit). They are not continuously
republished — regenerate locally for an apples-to-apples comparison with your
build and models.

## Scripts

The scripts in `scripts/` sweep `basert-bench` over a model list (override the
binary with `BASERT`) and compare against external engines (mlx-lm). They need
those tools installed and name their own model files; keep them as a reference
for methodology rather than a turnkey run.
