# Quantization

BaseRT stores quantized weights in its own affine (MLX-style) formats,
`base_q2` … `base_q8`, chosen per tensor. A bundle can also carry weights that
were already quantized at the source — GGUF k-quants and i-quants, MLX 4-bit,
OCP MXFP4 — copied in byte for byte instead of re-quantized. AWQ calibration
is optional at convert time. This page is an overview; the dtypes and their
packing are defined by the canonical spec.

## Canonical spec

The authoritative quantization specification lives in the repo:

- **[`base-convert/CANONICAL_QUANT_SPEC.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/CANONICAL_QUANT_SPEC.md)**
  — dtypes, group sizes, scale dtypes, symmetric/asymmetric rules, and the
  per-tensor header fields.

## Dtypes

| Dtype | Bits | Default group size | Typical use |
| --- | --- | --- | --- |
| `base_q2`, `base_q3` | 2, 3 | 32 | Aggressive compression of bulk weights. |
| `base_q4`, `base_q5`, `base_q6` | 4, 5, 6 | 64 | Quantized weights (affine, grouped). |
| `base_q8` | 8 | 128 | High-quality quantized weights. |
| `mxfp4` | 4 | 32 | OCP MX FP4 (E2M1 codes, E8M0 scale per group); gpt-oss experts. |
| `nvfp4` | 4 | 16 | NVIDIA FP4 (E2M1 codes, E4M3 scale per group). |
| `bf16` / `f16` | 16 | — | Sensitive tensors (norms, routers, embeddings). |
| `f32` | 32 | — | Full precision where needed. |

Each `base_qN` tensor has:

- **`group_size`** — elements sharing one scale: 32, 64 or 128. Defaults to the
  dtype's value above and can be overridden per tensor.
- **`scale_dtype`** — `bf16` (default), `f16`, `e8m0`, or `e4m3` (q8 only).
- **`symmetric`** — default `false`: asymmetric, storing a scale and a zero
  point per group, as MLX's affine format does. `true` stores the scale only.

`basert convert --target` takes `base-q2` … `base-q8`, `bf16`, `mxfp4` and
`nvfp4`.

## GGUF k-quants and i-quants

`basert convert --kquant-passthrough` copies these GGUF super-block types into
the bundle verbatim:

- k-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- i-quants: `IQ4_XS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`

The weights stay bit-identical to the source file, so the copy adds no error
and needs no override flag. Each such tensor is recorded with
`layout: "gguf_super"`, `group_size: 256` and its GGUF type in
`source_ggml_type`; its `dtype` is the `base_qN` that sizes it from below
(`Q4_K` and `IQ4_XS` are filed as `base_q4`). They run with native kernels on
Apple Silicon, for prefill, decode and embedding lookup. In mixture-of-experts
expert stacks, `Q4_K`, `Q5_K` and `IQ4_XS` pass through; other types there take
the normal convert path.

Without `--kquant-passthrough`, GGUF weights are dequantized and re-packed.
Re-quantizing an already-quantized source compounds its error, so doing that
with a `--profile` needs `--allow-quant-from-quant`.

`--mmproj <mmproj-*.gguf>` folds a GGUF model's separate vision tower into the
same bundle; its weights honour `--kquant-passthrough` too.

## Other sources

- **MLX 4-bit, group 64.** The scheme is the same as `base_q4`, so
  `--target base-q4` copies the packed codes and reproduces the source weights
  bit for bit. `--no-mlx-passthrough` re-quantizes instead.
- **gpt-oss.** The checkpoint's MXFP4 experts are carried across verbatim.

## Choosing precision

- **`default-q4`** — a good general default: ~4-bit weights, strong
  size/quality balance.
- **`default-q6`**, **`default-q8`** — higher quality, larger files; good for
  sensitive models or when memory allows.
- **`default-q4-embq6`** — `default-q4` with the token embeddings at 6 bits
  instead of f16.
- The `*-f16scale` variants store f16 scales instead of bf16.

`basert pull <id>:default-q8` (or `--variant default-q8`) picks a quant build.
Per-tensor control comes from a [quant profile](profiles.md). Tuned,
model-specific quality is delivered through the catalog as pre-converted
bundles. Some catalog releases come in sizes rather than quants (`-XS`,
`-Small`, `-Medium`, `-Large`, `-XL`): `basert pull` takes the Medium unless
you name a size (`:small`) or a file (`--file`).

## AWQ calibration

Activation-aware weight quantization reduces low-bit error by scaling salient
input channels before rounding. In BaseRT the converter searches the scale
exponent (alpha) for each group of projections that share an input, and folds
the scales into the neighbouring weights during `basert convert`. The written
bundle needs no special kernel and nothing is applied at run time.

AWQ needs two things:

- A [profile](profiles.md) whose rules say `"algo": "awq"` for the tensors to
  calibrate (optionally with a fixed `awq_alpha` instead of the search).
- An activation-statistics sidecar, passed with `--awq-profile <sidecar.json>`.
  It is recorded by running calibration text through the model with the C
  API's `baseRT_calibrate_begin` / `baseRT_calibrate_prefill` /
  `baseRT_calibrate_save`, and carries a fingerprint of the checkpoint it was
  measured on; the converter refuses a sidecar measured on different weights.

Each AWQ tensor records `algo: "awq"` and the alpha chosen, and the header's
`calibration` block records the run. `--imatrix` uses the same sidecar for
importance-weighted rounding instead of AWQ scaling. See also
[Converting models](../guides/converting.md).

Commands for quantization work, all reading the Hugging Face checkpoint you
would convert:

| Command | What it does |
| --- | --- |
| `basert inventory <ckpt>` | Print the canonical tensor names and the AWQ fold groups the converter would form. |
| `basert fingerprint <ckpt>` | Print the fingerprint a sidecar for this checkpoint must carry. |
| `basert awq-search <ckpt>` | Run the converter's alpha search for chosen fold groups and report the result. |
| `basert fakequant <ckpt>` | Write one tensor's dequantized values at a given format. |
| `basert overlay <ckpt>` | Quantize only the tensors a spec names into a small `.base`. |
| `basert merge <base> <overlays>…` | Splice overlay bundles into a bundle, replacing those tensors. |

Run `basert <command> --help` for each one's options.
