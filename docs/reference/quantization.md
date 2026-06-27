# Quantization

BaseRT quantizes weights with affine (MLX-style) quantization and an optional
AWQ calibration pass. The on-disk dtypes and their defaults are defined by the
canonical spec; this page is an overview.

## Canonical spec

The authoritative quantization specification lives in the repo:

- **[`base-convert/CANONICAL_QUANT_SPEC.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/CANONICAL_QUANT_SPEC.md)**
  — dtypes, group sizes, scale dtypes, symmetric/asymmetric rules, and the
  per-tensor header fields.

## Dtypes

| Dtype | Bits | Typical use |
| --- | --- | --- |
| `base_q2` … `base_q8` | 2–8 | Quantized weights (affine, grouped). |
| `bf16` / `f16` | 16 | Sensitive tensors (norms, routers, embeddings). |
| `f32` | 32 | Full precision where needed. |

Each quantized tensor has:

- **`group_size`** — elements sharing one scale (per-bit-width default, e.g. 64
  for q4; overridable).
- **`scale_dtype`** — `bf16` | `f16` | `e8m0` | `e4m3` (e4m3 is q8-only).
- **`symmetric`** — default `false` (asymmetric, MLX-affine style).

## Choosing precision

- **`default-q4`** — a good general default: ~4-bit weights, strong size/quality
  balance.
- **`default-q8`** — higher quality, larger files; good for sensitive models or
  when memory allows.
- The `*-f16scale` / `*-bf16` variants control the scale dtype.

Per-tensor control comes from a [quant profile](profiles.md).

## AWQ calibration

Activation-aware weight quantization reduces low-bit error by scaling salient
channels before packing. Provide calibration text (or a precomputed
activation-stats sidecar) at convert time — see
[Converting models](../guides/converting.md#awq-calibration). Tuned,
model-specific quality is delivered through the catalog as pre-converted
artifacts.
