# Benchmarking

`basert bench` measures prefill and decode throughput for a model. `basert
computearena` runs the same kind of measurement as a signed report you can
submit to [ComputeArena](https://computearena.ai).

## `basert bench`

```sh
basert bench <model> [-p N] [-n N] [-r N] [-w N] [options]
```

`<model>` is a hub id (with `:variant` or `--variant` if several are
installed) or a path to a `.base` file.

| Flag | Default | Purpose |
| --- | --- | --- |
| `-p, --pp N` | `512` | Prefill length in prompt tokens (`0` skips prefill). |
| `-n, --tg N` | `128` | Decode length in generated tokens (`0` skips decode). |
| `--tg-at LIST` | — | Also decode at each of these live context lengths (e.g. `1k,4k,8k`), best of the repetitions. |
| `-r, --reps N` | `5` | Timed repetitions per test. |
| `-w, --warmup N` | `3` | Minimum warm-up repetitions. Extended until two in a row are within 2% of each other, up to 12. |
| `--fill MODE` | `bos` | What the prefill prompt is made of (see below). |
| `--paged-kv` | off | Benchmark the paged KV cache path. |
| `--no-ane` | off | Prefill on the GPU alone (no Neural Engine split). |

```sh
basert bench basecompute/Qwen3-8B -p 512 -n 128 -r 3
basert bench basecompute/Qwen3-8B -p 0 -n 128 --tg-at 1k,4k,8k
```

The output is a table with one row per test: `pp<N>` rows are prefill
tokens/sec (prompt processing) and `tg<N>` rows are decode tokens/sec
(generation), each as mean ± standard deviation.

### Prefill input

`--fill` sets the tokens the prefill test reads:

| Mode | Prompt |
| --- | --- |
| `bos` | Every position is the BOS token. |
| `random[:SEED]` | Uniform random tokens from the vocabulary (seed 42 by default). |
| `corpus:PATH` | BOS plus the tokenized text of `PATH`, cycled to length. |

Mixture-of-experts routing depends on the input: a BOS fill sends every
position to the same experts and overstates MoE prefill. Use `corpus` with real
text for MoE models. Rows from a non-default fill are tagged `pp<N>:<mode>`, so
they never mix with `bos` numbers. Decode is unaffected by `--fill`.

## ComputeArena

[ComputeArena](https://computearena.ai) runs prefill/decode benchmarks, saves
each run as a signed report, and accepts submitted reports. Its command-line
tool is installed separately (see the
[ComputeArena quickstart](https://computearena.ai/quickstart)); `basert
computearena` runs its BaseRT side with the benchmark harness that ships with
BaseRT.

```sh
basert computearena run ./models/my-model.base   # run and save a signed report
basert computearena list                         # reports on this machine
basert computearena inspect <report>             # print one report (path or run id)
basert computearena verify <report>              # check its Ed25519 signature
basert computearena login                        # connect this installation
basert computearena submit                       # pick reports to submit
```

`run` takes a local model file rather than a hub id. `basert list --json`
shows the path of each installed model. Its options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--pp LIST` | `128,256,512,1024,2048,4096,8192,16384` | Prefill token counts. |
| `--tg N` | `128` | Decode tokens per repetition. |
| `-r, --reps N` | `3` | Recorded repetitions. |
| `-w, --warmup N` | `3` | Warm-up repetitions. |
| `--cooldown` | off | Wait for the machine to cool before measured phases (can take much longer). |
| `-y, --yes` | off | Run without prompts. |
| `-o, --output PATH` | — | Write the report here instead of the local report directory. |

`basert computearena identify <model> <huggingface-url>` ties a model file you
downloaded yourself to the exact Hugging Face file it came from, by SHA-256.
Run `basert computearena <command> --help` for the rest.

## Reproducible benchmark scripts

The [`benchmarks/`](https://github.com/basecompute/baseRT/tree/main/benchmarks)
directory has scripts that drive `basert-bench` over `.base` files and write
CSV results, plus reference results under `benchmarks/results/`. The scripts
look for `build/basert-bench` at the repo root; set `BASERT` to the path of
another `basert-bench` binary. Start with `benchmarks/README.md`.

## Methodology notes

- Use controlled token counts (`-p`/`-n`) and several repetitions (`-r`). The
  adaptive warm-up keeps first-call allocation and clock ramp-up out of the
  timed runs; large models may need more warm-up to settle the page cache.
- Prefill and decode are reported separately; they stress different parts of
  the engine (compute-bound matmuls vs. memory-bound single-token steps).
- Decode slows as the context fills. `--tg-at` measures it at the depths you
  care about rather than only at the start of a conversation.
- For server-style throughput, benchmark `basert serve --continuous-batching N`
  with concurrent clients rather than the single-stream `bench` tool.
  `basert serve` speculates by default, so pass `--speculate off` to measure
  plain decoding.
