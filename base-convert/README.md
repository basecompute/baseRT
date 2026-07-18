# BaseRT

The `basert` CLI: a model hub, an offline converter from GGUF /
MLX-safetensors / HF-safetensors to the `.base` cache format, and a
launcher for the runtime tools (`basert serve`, `basert chat`, …).

See [FORMAT.md](FORMAT.md) for the on-disk layout.

## Workspace crates

| Crate           | Role                                                   |
|-----------------|--------------------------------------------------------|
| `base-format`   | `.base` file reader/writer, header schema              |
| `base-quant`    | Packing routines: `base_q4`, `base_q8`, `mxfp4`, `nvfp4` |
| `base-awq`      | Full AWQ calibration (clip + per-layer alpha search)   |
| `base-readers`  | Input readers: GGUF, MLX-safetensors, HF-safetensors   |
| `base-arch`     | Per-architecture tensor-name normalization             |
| `base-sign`     | ed25519 manifest signing + verification                |
| `base-hub`      | Model hub: HF download, cache layout, registry/catalog |
| `base-convert`  | CLI entry point (the `basert` binary)                   |

## Build

```
cargo build --release -p base-convert   # produces the `basert` binary
```

## Model hub: `pull` / `list`

`basert pull` fetches a model into a per-user cache
(`$BASERT_MODELS_DIR`, default `~/.cache/baseRT/models`) laid out as
`<org>/<model>/<variant>/model.base`. The runtime reads the same
directory.

```
# Pre-converted model from the BaseRT catalog — downloaded directly, no
# local conversion:
basert pull basecompute/<name>

# Any HF repo — source safetensors are downloaded and converted locally
# (generic default-q4 profile; pass --profile / --target to override):
basert pull meta-llama/Llama-3.2-1B
basert pull Qwen/Qwen3-0.6B --target base-q8

basert pull <id> --dry-run     # resolve + print the plan, download nothing
basert list                    # installed models (table; --json for JSON)
basert list --remote           # also show catalog models not yet installed
```

Gated/private repos use the standard HuggingFace token chain
(`$HF_TOKEN` / `~/.cache/huggingface/token`). Convert-on-pull in the
public build uses only generic profiles; tuned quantization is delivered
through pre-converted catalog artifacts.

## Runtime tools

`basert <cmd>` forwards to the matching runtime binary (`basert-<cmd>`):

```
basert serve    <model> [--model <model2> …]         # OpenAI-compatible server (--model repeats for multi-model)
basert chat     <model>                              # interactive chat
basert complete <model> --prompt <text>              # one-shot completion
basert bench    <model>                              # throughput benchmark
```

## Converter subcommands

```
basert convert  --source <path> --target base-q4 --output <path>
basert inspect  <path>                 # dump header + tensor inventory
basert keygen   <key-prefix>           # ed25519 keypair (writes .key, .pub)
basert sign     <bundle> <secret-key>  # signs an unsigned .base in-place
basert verify   <bundle> <public-key>  # exits non-zero on tampered file
```

Run `basert --help` for the full flag matrix.

## Model signing workflow

BaseRT ships an ed25519 signing facility for `.base` bundles. The
runtime does **not** currently verify signatures at load time
(planned for v1.1; tracked as P1 item S4). Until then, signing is an
operator-side workflow you can use *out-of-band* to detect tampering
or corruption before a `.base` file reaches a host:

```sh
# 1. Generate a keypair once. Keep `<prefix>.key` secret; distribute
#    `<prefix>.pub` to anyone who needs to verify your bundles.
basert keygen ./signing

# 2. Sign a converted bundle.
basert sign ./model.base ./signing.key

# 3. On the deployment host, verify before loading.
basert verify ./model.base ./signing.pub \
    || { echo "tamper detected — refusing to deploy"; exit 1; }

# 4. Hand the verified file to the runtime.
basert chat --model ./model.base
```

Notes:

- The signature covers the file header plus a SHA-256 of the weight
  blob. A bit-flip anywhere in the bundle invalidates the signature.
- An unsigned `.base` file is loadable by the runtime — this is by
  design for development workflows. Production deployments should
  gate on `basert verify` in their release pipeline.
- v1.1 will plumb verification into the runtime behind an opt-in
  switch so the load path itself can refuse tampered bundles. Until
  then, treat `basert verify` as the canonical check.
