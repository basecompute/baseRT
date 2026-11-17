# BaseRT

The `basert` CLI: a model hub, an offline converter from GGUF /
MLX-safetensors / HF-safetensors to the `.base` cache format, and a
launcher for the runtime tools (`basert serve`, `basert chat`, …).

See [FORMAT.md](FORMAT.md) for the on-disk layout.

## Workspace crates

| Crate           | Role                                                   |
|-----------------|--------------------------------------------------------|
| `base-format`   | `.base` file reader/writer, header schema              |
| `base-quant`    | Packing routines: `base_q2` … `base_q8`, `mxfp4`, `nvfp4`; quant profiles |
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
(`$BASERT_MODELS_DIR`, default `~/Library/Caches/baseRT/models` on macOS and
`~/.cache/baseRT/models` on Linux) laid out as
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

# A variant by the name `basert list` gives it (`default-q8`, `cuda-q4mix`,
# a `file-…` variant), or a quant; the form `basert chat` and `serve` take:
basert pull basecompute/<name>:default-q8
basert pull Qwen/Qwen3-0.6B:q8

# A release that comes in sizes (the BaseQuant repos' -XS/-Small/-Medium/
# -Large/-XL files): its Medium by default, another by size or file name:
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1:small
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1 --file Qwen3.8-27B-BaseQuant-V1-XL.base

basert pull <id> --dry-run     # resolve + print the plan, download nothing
basert list                    # installed models (table; --json for JSON)
basert list --remote           # also show catalog models not yet installed
```

Downloads stage under `<models dir>/.src/hf` — not the global HuggingFace
cache — and are moved into place on success, so a pulled model exists
exactly once on disk. Interrupted pulls leave their partial download in
staging and resume on retry. Convert-on-pull deletes the downloaded
safetensors sources after a successful conversion; set
`BASERT_KEEP_HF_SOURCES=1` to keep them (faster re-pulls of another quant
of the same repo, at the cost of a second multi-GB copy).

Gated/private repos use the standard HuggingFace token chain
(`$HF_TOKEN` / `~/.cache/huggingface/token`). Convert-on-pull in the
public build uses only generic profiles; tuned quantization is delivered
through pre-converted catalog artifacts.

## Runtime commands

Run a model with the BaseRT runtime:

```
basert serve    <model> [--model <model2> …]         # HTTP server: runs Superfluid, installed on first use
basert chat     <model>                              # interactive chat
basert complete <model> --prompt <text>              # one-shot completion
basert bench    <model>                              # throughput benchmark
```

## Converter subcommands

```
basert convert  <path> --target base-q4 -o <out.base>
basert inspect  <path> [--verify-checksums]       # dump header + tensor inventory
basert keygen   --out <dir> [--name <name>]       # ed25519 keypair (<name>.secret, <name>.pub)
basert sign     <bundle> --key <secret> -o <out>  # writes a signed copy
basert verify   <bundle> --pubkey <pub>           # exits non-zero on a tampered or wrongly signed file; 0 on an unsigned one
```

Probe tooling for HF checkpoints (per-tensor quant search; everything is
mapped to the same canonical names `convert` uses — see `basert inventory`):

```
basert inventory  <hf-dir>                       # canonical tensors + AWQ fold groups (JSON)
basert awq-search <hf-dir> --awq-profile <sidecar.json> --tensors <a,b;c> --cell 4:64
basert overlay    <hf-dir> --spec <spec.json> -o <overlay.base> [--awq-profile <sidecar.json>]
basert merge      <base.base> <overlay.base>... -o <out.base>
basert fakequant  <hf-dir> --tensor <canonical> --dtype base_q4 --group-size 64 --out <w.f32>
```

* `overlay` quantizes ONLY the tensors named in the spec — byte-identical
  to what `convert --profile` produces for them (same header, value hooks,
  AWQ fold, dtype routing) — plus whatever the AWQ fold rescales (norm
  gamma, fold consumers, carried f16). No other tensor's bytes are read,
  so a probe costs well under a second. Spec schema:

  ```json
  {"tensors": {
    "layers.3.mlp.down_proj.weight":   {"dtype": "base_q4", "group_size": 64, "scale_dtype": "bf16", "algo": "rtn"},
    "layers.3.self_attn.q_proj.weight": {"dtype": "base_q4", "group_size": 64, "algo": "awq", "awq_alpha": 0.5},
    "layers.5.mlp.down_proj.weight":   {"dtype": "base_q3", "group_size": 64, "algo": "gptq", "codes": "/path/codes.safetensors"},
    "layers.9.input_norm.weight":      {"dtype": "f16"}
  }}
  ```

  `dtype` as in profile rules; `group_size` defaults per dtype,
  `scale_dtype` to `bf16`, `symmetric` to `false`, `algo` to `rtn`.
  `algo: "awq"` needs `--awq-profile`. `algo: "gptq"` packs externally
  searched codes verbatim: the safetensors file holds `<T>.codes` (U8,
  `[N, K]`, values `0..2^bits-1`), `<T>.scales` and `<T>.biases` (F32,
  `[N, K/group_size]`), laid out exactly like the RTN packer's output
  (`base_quant::pack_codes`). The header records
  `provenance.overlay = {spec, fold_modified}`; `quant_profile` is
  `"overlay"`.
* `merge` rewrites the base bundle with same-named tensors taken from the
  overlays (later overlays win), entries and bytes copied verbatim
  (checksums kept). Header, slots and the mmproj sub-bundle come from the
  base; an overlay tensor the base lacks is an error. Records
  `provenance.merge = {base, overlays, replaced}`; a signature on the base
  is dropped (re-sign the result).
* `fakequant` writes `unpack(pack(w))` of one canonical tensor as raw
  little-endian f32 (row-major, no header) after the converter's value
  hooks, and prints `{"shape": [N, K], "elements": n}` on stdout — the
  reference for parity-testing an external quantizer replica.

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
