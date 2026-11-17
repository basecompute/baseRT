# Command reference

Every `basert` command and its flags, as of BaseRT 0.5.1. Run
`basert <command> --help` for the authoritative, version-specific list.

---

## `basert pull`

Download a model into the local hub cache — a pre-converted `.base` from the
catalog, or a raw Hugging Face repo converted on the fly.

```sh
basert pull <id>[:<variant>] [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `<id>` | — | `basecompute/<name>` (catalog) or `org/model` (raw HF repo). |
| `:<variant>` | — | A variant as `basert list` names it (`default-q8`), a quant (`q8`), or a size (`small`) of a release that comes in sizes. |
| `--profile <path>` | — | Override the auto-selected quant profile (path to a profile JSON). |
| `--target <scheme>` | `base-q4` | Quant scheme for convert-on-pull when no profile applies. Given, it also picks a pre-converted repo's file by quant. |
| `--file <name>` | — | Take this `.base` file from the repo (e.g. `Foo-Medium.base`) rather than picking one by quant or size. |
| `--revision <ref>` | `main` | HF revision / branch / tag. |
| `--force` | off | Re-download / re-convert even if cached. |
| `--dry-run` | off | Resolve and print the plan without downloading. |

`--target` takes `base-q2`, `base-q3`, `base-q4`, `base-q5`, `base-q6`,
`base-q8`, `bf16`, `mxfp4` or `nvfp4`.

A release that comes in sizes (files named `-XS`, `-Small`, `-Medium`,
`-Large`, `-XL`) takes the Medium when nothing names a file, and refuses a
`--target`. A variant already installed is used as it is, without contacting
Hugging Face.

## `basert list`

List models in the local cache.

```sh
basert list [--remote] [--json]
```

| Flag | Purpose |
| --- | --- |
| `--remote` | Also list catalog models that aren't installed yet. |
| `--json` | Emit JSON instead of a table, with each model's path and source file. |

## `basert rm`

Remove all locally installed variants of an exact model id shown by `basert list`.

```sh
basert rm <model>
basert rm Qwen/Qwen3-4B
```

Deletes each installed variant's `model.base` and `hub.json`, then removes empty
variant/model directories. Other files and `.src` HF staging are retained.
Uses `$BASERT_MODELS_DIR`, like `pull` and `list`. No confirmation is required;
a missing model or unsafe id returns a non-zero exit status. Paths and
`:variant` selectors are not accepted.

## `basert convert`

Convert a source model (GGUF / HF safetensors / MLX) to `.base`.

```sh
basert convert <input> [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `<input>` | — | Source model path (or a bundle name with `--synthetic`). |
| `-o, --output <path>` | `<input>.base` | Output `.base` file. |
| `--target <scheme>` | `base-q4` | Quant scheme (the fallback when `--profile` is set). Same values as `pull --target`. |
| `--profile <path>` | — | Per-tensor canonical-quant profile JSON, e.g. `base-convert/profiles/default-q4.json`. |
| `--awq-profile <path>` | — | AWQ activation-stats sidecar JSON; needs `--profile`. See [AWQ calibration](../guides/converting.md#awq-calibration). |
| `--imatrix` | off | Importance-weighted rounding from the `--awq-profile` sidecar instead of AWQ's rotation. |
| `--allow-quant-from-quant` | off | Allow `--profile` on an already-quantized source (GGUF, MLX 4/8-bit). |
| `--kquant-passthrough` | off | GGUF only: copy k-quant and IQ super-blocks into the bundle as they are. |
| `--mlx-passthrough` | off | Copy an MLX 4-bit (group 64) checkpoint's packed weights as they are. Needs `--target base-q4` and no `--profile`. |
| `--validate` | off | After `--mlx-passthrough`, compare every written tensor with the source. |
| `--mmproj <path>` | — | GGUF only: fold a companion `mmproj-*.gguf` vision tower into the bundle. |
| `--mmproj-config <path>` | — | HF `config.json` for the multimodal settings an mmproj GGUF cannot carry. |
| `--mtp-head <head.base>` | — | Embed a converted Gemma 4 MTP head in the bundle, as `add-speculator` does. |
| `--quantize-drafter` | off | Drafter sidecars only: quantize the drafter's layers with `--target`. |
| `--direct-write` | off | Write straight into the output file (peak disk use is the bundle size, not 2×). |
| `--synthetic` | off | Generate a dummy bundle (CI/testing). |

Less common flags (`--no-mlx-passthrough`, `--no-synth-tied-head`,
`--speculator-lm-head`, `--mtp-head-version`, `--mtp-draft-ranking`,
`--mtp-draft-vocab`, `--keep-drafter-fc-f16`, `--awq-search-rows`) are
described in `basert convert --help`. See
[Converting models](../guides/converting.md).

## `basert inspect`

Summarize a `.base` file's header, tensors, and slots.

```sh
basert inspect <input> [--verify-checksums]
```

| Flag | Purpose |
| --- | --- |
| `--verify-checksums` | Also verify per-tensor xxhash64 (slow for large models). |

## `basert add-speculator`

Embed a converted MTP head in a bundle as its built-in speculator head (Gemma
4's separate `-assistant` checkpoint).

```sh
basert add-speculator <target.base> --head <head.base> --output <out.base> [--head-version <v>]
```

| Flag | Purpose |
| --- | --- |
| `<target>` | The bundle the head drafts for. |
| `--head <path>` | The head, converted on its own (`basert convert google/gemma-4-E4B-it-assistant`), f16 or quantized (`--quantize-drafter --target base-q8`). |
| `-o, --output <path>` | Where to write the bundle (refused if it exists). |
| `--head-version <v>` | A version for the head, recorded in the bundle. |

## `basert keygen`

Generate an ed25519 keypair for signing.

```sh
basert keygen --out <dir> [--name <name>]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `-o, --out <dir>` | — | Output directory. Writes `<name>.secret` and `<name>.pub`. |
| `--name <name>` | `baseRT-key` | Base name for the key files. |

## `basert sign`

Sign an unsigned `.base` file.

```sh
basert sign <input> --output <path> --key <secret> [--key-id <id>]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `<input>` | — | Unsigned input `.base` file. |
| `-o, --output <path>` | — | Signed output `.base` file. |
| `--key <path>` | — | Path to a 32-byte ed25519 secret key. |
| `--key-id <id>` | `baseRT-default` | Human-readable key identifier stored in the file. |

## `basert verify`

Verify a signed `.base` file (exits non-zero on tampering).

```sh
basert verify <input> --pubkey <path>
```

| Flag | Purpose |
| --- | --- |
| `<input>` | Input `.base` file. |
| `--pubkey <path>` | Path to a 32-byte ed25519 public key. |

See [Signing & verification](../guides/signing.md).

---

## `basert self`

Update basert, or switch between the versions installed here. See
[Updating](../getting-started/installation.md#updating).

| Command | Purpose |
| --- | --- |
| `basert self update` | Install the newest release for this machine and make it the one in use, with the Superfluid it needs. |
| `basert self check` | Say whether a newer release exists, without installing it. |
| `basert self list` | The versions installed here, and which is in use. |
| `basert self use <version>` | Make an installed version the one in use (the one it replaces stays). |
| `basert self install --from <dir>` | Place an unpacked release bundle and make it the one in use (what the install script runs). |

`basert self update` flags:

| Flag | Purpose |
| --- | --- |
| `--version <v>` | Go to this release instead of the newest (an older one too), e.g. `0.5.0`. |
| `--dry-run` | Say what would be done, and change nothing. |
| `-y, --yes` | For a copy Homebrew, npm, pipx, uv or cargo installed: run that manager's update without asking (needed without a terminal). |

`basert self install` takes `--root <dir>`, the install directory (default
`$BASERT_INSTALL_DIR`, else `~/.basert`).

## `basert superfluid`

Install or check [Superfluid](https://github.com/basecompute/superfluid), the
server `basert serve` runs.

| Command | Purpose |
| --- | --- |
| `basert superfluid install [--force]` | Install the newest Superfluid release that serves this engine where `basert serve` looks for it. `--force` installs again even when that release, or a newer one, is there. |
| `basert superfluid status` | Show the release this basert was tested with, what is installed where, and whether it can serve the engine. Exits 0 when an installed Superfluid can, 1 when none is installed, 2 when none installed can. |

---

## Runtime commands

`chat`, `complete`, `bench` and `transcribe` exec `basert-<cmd>`; `serve`
runs Superfluid. Full flags in their guides.

**Selecting a quant variant.** For the model-taking commands, choose which
build to load with an inline `:variant` on the hub id, or the `--variant`
flag:

```sh
basert serve basecompute/gemma-4-E2B-it:default-q8
basert serve basecompute/gemma-4-E2B-it --variant default-q8   # equivalent
```

Run `basert list` to see installed variants (`default-q4`, `default-q8`, …). An
uninstalled variant is fetched on demand. The inline `:variant` wins if both are
given.

### `basert serve`

OpenAI-compatible HTTP server, run by Superfluid. See
[Serving an API](../guides/serving.md).

```sh
basert serve <model> [--model <model2> ...] [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `<model>` | — | A `.base` path or a hub id, pulled on first use. |
| `--model <model>` | — | Another model to serve. Repeatable. |
| `--host <addr>` | `127.0.0.1` | Bind address. |
| `--port <N>` | `8453` | Bind port. |
| `--api-key <key>` | — | Require Bearer token authentication. |
| `--max-context <N>` | auto | Context window. Sized for this machine's GPU memory, capped at the model's trained window. |
| `--max-tokens <N>` | rest of window | Cap on generated tokens for a request that names none. |
| `--kv-bits 4\|8\|16\|84` | per model | KV cache element width (`84` = K at Q8, V at Q4). |
| `--max-batch-size <N>` | `1` | Max concurrent sequences (`--max-batch` is an alias). |
| `--continuous-batching [N]` | off (`8` without N) | Serve up to N requests at once through one shared forward pass. |
| `--prefill-budget <N>` | — | Max prompt tokens per engine tick (64–4096). |
| `--speculate <strategy>` | `auto` | Speculative decoding: `auto` (the drafter the catalog pairs with the model, else the head its file carries, else none), `off`, `mtp-head`, `prompt-lookup`, or `dflash:<path or hub id>`. |
| `--offline` | off | Download nothing (no model, drafter or catalog); serve what is installed. |
| `--model-dir <path>` | — | Register every `*.base` under `<path>`, loaded on first request. |
| `--idle-timeout <N>` | `0` | Unload a model idle for N seconds (`0` = never). |
| `--rate-limit <N>` | `0` | Requests per minute per client (`0` = unlimited). |
| `--request-timeout <ms>` | — | Bounds fill-in-the-middle completions only. |
| `--drain-timeout <N>` | `60` | Seconds to wait for in-flight requests on `SIGUSR1`. |
| `--sessions <path>` | per port, in the state directory | Request-log directory. |
| `--files-max-bytes <N>` | — | Reject uploads that would push stored bytes above N. |
| `--files-expiry <N>` | — | Remove uploaded files older than N seconds. |
| `--files-sweep <N>` | `300` | Sweep cadence in seconds, with `--files-expiry`. |
| `--log-file <path>` | stderr | Write the access log and diagnostics here. |
| `--basert-lib <dir>` | beside `basert` | The engine library directory to serve with. |
| `-v, --verbose` | off | Engine diagnostics (load, GPU). |
| `--print-command` | — | Print the Superfluid command line this start would run, and exit. |
| `--version` | — | Print the version, and the Superfluid a start would run. |

Any other `--flag` is handed to `superfluid serve` as it is (for example
`--spec-bitexact`); everything after `--` is handed over untouched. See
[Superfluid's server docs](https://docs.basecompute.co/latest/serving/openai_compatible_server/).

Accepted so that existing command lines keep working, and no longer used:
`--paged-kv` and `--prefix-cache` (always on), `--prefix-cache-file`,
`--prefix-cache-save-interval`, `--files-dir`, `--media-dir`, `--metallib`,
`--prefill-chunk`, `--paged-weights`, `--no-paged-weights-retry`,
`--gpu-wait-timeout-ms`, `--decode-replay`, `--no-baked-decode`.

### `basert chat`

Interactive chat. See [Chat & completion](../guides/chat-complete.md).

```sh
basert chat <model> [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--temp F` | `0.7` | Temperature. |
| `--top-k N` | `40` | Top-K candidates. |
| `--top-p F` | `0.9` | Nucleus sampling threshold. |
| `--min-p F` | `0.0` | Min-P filter. |
| `--repeat-penalty F` | `1.1` | Repetition penalty. |
| `--max-tokens N` | `2048` | Max tokens per response. |
| `--max-context N` | auto | Context window, sized for this machine. |
| `--think` / `--no-think` | `--no-think` | Show or hide the model's reasoning. |
| `--kv-bits 4\|8\|16\|84` | per model | KV cache element width. |
| `--paged-kv` | off | Paged KV cache. |
| `--max-batch-size N` | `1` | Max concurrent sequences for batched decode. |
| `--no-ane` | off | Prefill on the GPU alone (no Neural Engine split on Apple Silicon). |
| `-v, --verbose` | off | Engine diagnostics. |

In the prompt, `/clear` resets the conversation and `quit` exits.

### `basert complete`

One-shot completion (text / image / audio).

```sh
basert complete <model> --prompt <text> [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--prompt <text>` | — | Prompt text (`-` reads stdin). Required unless `--prompt-file`. |
| `--prompt-file <path>` | — | Read the prompt from a file. |
| `--chat` | off | Wrap the prompt in the model's chat template (recommended for instruct models). |
| `--system <text>` | — | System prompt (with `--chat`). |
| `--image <path>` | — | Image input, for a vision model. |
| `--audio <path>` | — | Audio input, for an audio model. |
| `-n, --max-tokens N` | `2048` | Max tokens to generate. |
| `--temp F` | `0.0` | Temperature (`0` = greedy). |
| `--top-k N` | `40` | Top-K. |
| `--top-p F` | `0.9` | Top-P. |
| `--repeat-penalty F` | `1.0` | Repetition penalty. |
| `--spec` | off | Speculative decoding with the head the model's file carries. |
| `--kv-bits 4\|8\|16\|84` | per model | KV cache element width. |
| `--paged-kv` | off | Paged KV cache. |
| `--max-batch-size N` | `1` | Max concurrent sequences. |
| `--max-context N` | auto | Context budget, sized for this machine. |
| `--no-ane` | off | Prefill on the GPU alone. |
| `-v, --verbose` | off | Engine diagnostics. |

### `basert bench`

Throughput benchmark. See [Benchmarking](../guides/benchmarking.md).

```sh
basert bench <model> [-p N] [-n N] [-r N] [-w N] [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `-p, --pp N` | `512` | Prefill length (`0` skips it). |
| `-n, --tg N` | `128` | Decode length (`0` skips it). |
| `--tg-at LIST` | — | Decode at each listed context (e.g. `1k,4k,8k`). |
| `-r, --reps N` | `5` | Timed repetitions per test. |
| `-w, --warmup N` | `3` | Minimum warmup reps, extended until two reps agree within 2% (at most 12). |
| `--fill MODE` | `bos` | Prefill tokens: `bos`, `random[:SEED]` or `corpus[:PATH]`. |
| `--paged-kv` | off | Paged KV cache. |
| `--no-ane` | off | Prefill on the GPU alone. |

### `basert transcribe`

Audio transcription with a Whisper model.

```sh
basert transcribe <model> <audio.wav> [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--lang <code>` | `en` | Language code, or `auto` to detect (`auto` with `--task translate`). |
| `--task <task>` | `transcribe` | `transcribe`, or `translate` (any language to English). |
| `--prompt <text>` | — | Initial prompt to bias the decode (names, style, spellings). |
| `--no-condition-on-prev` | off | Don't condition each window on the previous text. |
| `--no-timestamps` | off | Disable timestamps (faster, no seeking). |

### `basert computearena`

Runs the [ComputeArena](https://computearena.ai/quickstart) CLI with BaseRT
selected, and hands it the benchmark harness that ships in the BaseRT bundle.
ComputeArena is installed separately; without it, the command says where to
get it.

```sh
basert computearena <command> [options]
```

Its commands include `run` (a prefill/decode benchmark saved as a signed
report), `list`, `inspect`, `verify`, `submit`, `login` and `logout`. Run
`basert computearena --help` for the commands your ComputeArena has.
