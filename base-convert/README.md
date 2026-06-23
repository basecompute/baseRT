# base-convert

Offline converter from GGUF / MLX-safetensors / HF-safetensors to the
`.base` cache format used by baseRT at runtime.

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
| `base-convert`  | CLI entry point                                        |

## Build

```
cargo build --release -p base-convert
```

## Subcommands

```
base-convert convert  --source <path> --target base-q4 --output <path>
base-convert inspect  <path>                 # dump header + tensor inventory
base-convert keygen   <key-prefix>           # ed25519 keypair (writes .key, .pub)
base-convert sign     <bundle> <secret-key>  # signs an unsigned .base in-place
base-convert verify   <bundle> <public-key>  # exits non-zero on tampered file
```

Run `base-convert --help` for the full flag matrix.

## Model signing workflow

baseRT ships an ed25519 signing facility for `.base` bundles. The
runtime does **not** currently verify signatures at load time
(planned for v1.1; tracked as P1 item S4). Until then, signing is an
operator-side workflow you can use *out-of-band* to detect tampering
or corruption before a `.base` file reaches a host:

```sh
# 1. Generate a keypair once. Keep `<prefix>.key` secret; distribute
#    `<prefix>.pub` to anyone who needs to verify your bundles.
base-convert keygen ./signing

# 2. Sign a converted bundle.
base-convert sign ./model.base ./signing.key

# 3. On the deployment host, verify before loading.
base-convert verify ./model.base ./signing.pub \
    || { echo "tamper detected — refusing to deploy"; exit 1; }

# 4. Hand the verified file to baseRT.
./build/baseRT_chat --model ./model.base
```

Notes:

- The signature covers the file header plus a SHA-256 of the weight
  blob. A bit-flip anywhere in the bundle invalidates the signature.
- An unsigned `.base` file is loadable by the runtime — this is by
  design for development workflows. Production deployments should
  gate on `base-convert verify` in their release pipeline.
- v1.1 will plumb verification into the runtime behind an opt-in
  switch so the load path itself can refuse tampered bundles. Until
  then, treat `base-convert verify` as the canonical check.
