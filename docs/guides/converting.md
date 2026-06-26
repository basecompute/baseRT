# Converting models

`basert convert` turns a source checkpoint into a `.base` bundle. Sources are
**GGUF**, **HuggingFace safetensors**, and **MLX safetensors**. Every source is
dequantized to f32 and re-packed via the canonical quantization path.

## Basic conversion

```sh
basert convert ./path/to/checkpoint \
    --target base-q4 \
    --output models/my-model.base
```

`--target` selects a quant scheme. Available schemes follow the
[canonical quant spec](../reference/quantization.md): `base_q2` … `base_q8`,
`bf16`, `f16`, `f32`.

## Profile-driven conversion

For per-tensor control, pass a **profile** instead of a single target. The
generic profiles ship in `base-convert/profiles/`:

```sh
basert convert ./checkpoint \
    --profile base-convert/profiles/default-q4.json \
    --output models/my-model.base
```

With `--profile`, per-tensor quant decisions come from the profile's rules;
`--target` becomes the fallback for tensors the profile's catch-all rule doesn't
cover. The profile name is recorded in the bundle header (`quant_profile`) for
audit. Write your own — see [Quant profiles](../reference/profiles.md).

## AWQ calibration

Activation-aware weight quantization improves low-bit quality. Provide
calibration text and an AWQ mode:

```sh
basert convert ./checkpoint \
    --profile base-convert/profiles/default-q4.json \
    --calibration calib.txt \
    --calibration-tokens 1024 \
    --awq-mode <mode> \
    --output models/my-model.base
```

Alternatively, pass a precomputed activation-stats sidecar with
`--awq-profile <path>` (produced by the engine's calibration mode). Tensors whose
profile rule is a canonical `base_qN` dtype run AWQ search + rotation before the
RTN pack; `bf16`/`f16` tensors are unaffected.

## Common flags

| Flag | Meaning |
| --- | --- |
| `-o, --output <path>` | Output `.base` file (defaults to `<input>.base`). |
| `--target <scheme>` | Quant scheme (or fallback when `--profile` is set). |
| `--profile <path>` | Per-tensor canonical-quant profile JSON. |
| `--calibration <file>` | UTF-8 calibration text (required for AWQ). |
| `--calibration-tokens <N>` | Number of calibration tokens. |
| `--awq-mode <mode>` | AWQ calibration mode. |
| `--awq-profile <path>` | Precomputed AWQ activation-stats sidecar. |
| `--synthetic` | Generate a dummy bundle (CI/testing; no file read). |

!!! warning "Quantizing from already-quantized inputs"
    The spec expects fp16/bf16/fp32 sources. Converting from an already-quantized
    GGUF (Q4_K_M, Q8_0, …) or MLX 4-bit/8-bit compounds error. An explicit
    override flag exists for users who don't have the fp16 checkpoint locally —
    see `basert convert --help`.

## Inspecting the result

```sh
basert inspect models/my-model.base          # header + tensor inventory + slots
basert inspect models/my-model.base --verify-checksums # also verify per-tensor xxhash64 (slow)
```
