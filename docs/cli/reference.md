# Command reference

Every `basert` command and its flags. Run `basert <command> --help` for the
authoritative, version-specific list.

---

## `basert pull`

Download a model into the local hub cache — a pre-converted `.base` from the
catalog, or a raw HuggingFace repo converted on the fly.

```sh
basert pull <id> [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `<id>` | — | `basecompute/<name>` (catalog) or `org/model` (raw HF repo). |
| `--profile <path>` | — | Override the auto-selected quant profile. |
| `--target <scheme>` | `base-q4` | Quant scheme for convert-on-pull when no profile applies. |
| `--revision <ref>` | `main` | HF revision / branch / tag. |
| `--force` | off | Re-download / re-convert even if cached. |
| `--dry-run` | off | Resolve and print the plan without downloading. |

## `basert list`

List models in the local cache.

```sh
basert list [--remote] [--json]
```

| Flag | Purpose |
| --- | --- |
| `--remote` | Also list catalog models that aren't installed yet. |
| `--json` | Emit JSON instead of a table. |

## `basert convert`

Convert a source model (GGUF / HF / MLX) to `.base`.

```sh
basert convert <input> [options]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `<input>` | — | Source model path (or a bundle name with `--synthetic`). |
| `-o, --output <path>` | `<input>.base` | Output `.base` file. |
| `--target <scheme>` | — | Quant scheme (or fallback when `--profile` is set). |
| `--profile <path>` | — | Per-tensor canonical-quant profile JSON. |
| `--calibration <file>` | — | UTF-8 calibration text (required for AWQ). |
| `--calibration-tokens <N>` | — | Number of calibration tokens. |
| `--awq-mode <mode>` | — | AWQ calibration mode. |
| `--awq-profile <path>` | — | Precomputed AWQ activation-stats sidecar. |
| `--synthetic` | off | Generate a dummy bundle (CI/testing). |

See [Converting models](../guides/converting.md).

## `basert inspect`

Summarize a `.base` file's header, tensors, and slots.

```sh
basert inspect <input> [--verify-checksums]
```

| Flag | Purpose |
| --- | --- |
| `--verify-checksums` | Also verify per-tensor xxhash64 (slow for large models). |

## `basert keygen`

Generate an ed25519 keypair for signing.

```sh
basert keygen --output <dir> [--name <name>]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `-o, --output <dir>` | — | Output directory. Writes `<name>.secret` and `<name>.pub`. |
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

## Runtime commands (forwarded to the engine)

These exec `basert-<cmd>`. Full flags in their guides.

**Selecting a quant variant.** For the model-taking commands (`serve`, `chat`,
`complete`, `bench`, `profile`), choose which quant build to load with an inline
`:variant` on the hub id, or the `--variant` flag:

```sh
basert serve basecompute/gemma-4-E2B-it:default-q8
basert serve basecompute/gemma-4-E2B-it --variant default-q8   # equivalent
```

Run `basert list` to see installed variants (`default-q4`, `default-q8`, …). An
uninstalled variant is fetched on demand. The inline `:variant` wins if both are
given.

### `basert serve`

OpenAI-compatible HTTP server. See [Serving an API](../guides/serving.md).

```sh
basert serve --model <path> [--model <path2> ...] [options]
```

Key flags: `--host`, `--port`, `--api-key`, `--max-context`, `--max-tokens`,
`--kv-bits`, `--paged-kv`, `--continuous-batching [N]`, `--prefix-cache`,
`--rate-limit`, `--idle-timeout`, `--request-timeout`, `--drain-timeout`,
`--files-dir`, `--log-file`.

### `basert chat`

Interactive chat. See [Chat & completion](../guides/chat-complete.md).

```sh
basert chat <model> [options]
```

Key flags: `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`,
`--max-tokens`, `--max-context`, `--think`/`--no-think`, `--kv-bits`,
`--paged-kv`, `--max-batch-size`.

### `basert complete`

One-shot completion (text / image / audio).

```sh
basert complete <model> --prompt <text> [options]
```

Key flags: `--prompt`, `--system`, `--chat`, `--image`, `--audio`,
`--max-tokens`, `--temp`, `--top-k`, `--top-p`, `--repeat-penalty`, `--kv-bits`,
`--paged-kv`.

### `basert bench`

Throughput benchmark. See [Benchmarking](../guides/benchmarking.md).

```sh
basert bench <model> [-p N] [-n N] [-r N] [-w N] [--paged-kv]
```

### `basert transcribe`

Audio transcription (Whisper-class models).

```sh
basert transcribe <model> <audio> [options]
```

### `basert profile`

Profile prefill/decode timing for performance investigation.
