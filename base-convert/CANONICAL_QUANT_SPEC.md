# Canonical Quantization Spec — `base_q2` … `base_q8`

Companion to `FORMAT.md`. Pins the on-disk layout, scale dtypes, and
per-tensor flexibility for the canonical bit-widths.

## Goals

- One canonical kernel family per bit-width per backend.
- Source format restricted to **fp16 / bf16 / fp32**. Already-quantized
  sources are rejected with a clear error — no silent quant-to-quant
  re-pack (it's lossy and contradicts the bit budget). Users with a
  GGUF Q4_K_M only must re-fetch the bf16 original.
- Quant happens at convert time. Multiple methods supported (AWQ,
  GPTQ, SmoothQuant, RTN); converter picks per use case.
- Mixed precision per-tensor (e.g. `mlp.experts: q4`, `mlp.shared_ffn:
  q8`, `lm_head: bf16`).
- Multi-platform: Apple Silicon (Metal), CUDA, ROCm. **One bundle per
  target backend**; the converter takes `--target=metal|cuda|rocm`.

## Bit-width defaults

MLX `affine` is the baseline; group sizes are tuned where their
uniform `gs=64` underperforms.

| Bits | group_size | Default scale dtype | Allowed scale dtypes        | Symmetric? | Note                                                       |
|------|------------|---------------------|------------------------------|-----------|------------------------------------------------------------|
| q2   | **32**     | bf16                | bf16 / f16 / e8m0            | flexible  | gs=32 because q2 needs more scale resolution than gs=64    |
| q3   | **32**     | bf16                | bf16 / f16 / e8m0            | flexible  | same reasoning as q2                                       |
| q4   | 64         | bf16                | bf16 / f16 / e8m0            | flexible  | matches MLX-affine default                                  |
| q5   | 64         | bf16                | bf16 / f16 / e8m0            | flexible  | matches MLX                                                 |
| q6   | 64         | bf16                | bf16 / f16 / e8m0            | flexible  | matches MLX                                                 |
| q8   | **128**    | bf16                | bf16 / f16 / e4m3 / e8m0     | flexible  | gs=128 — 8-bit is accurate enough that scale density costs more than it returns |

Group sizes 32 / 64 / 128 are the only permitted values, matching MLX's
allowed set. Per-tensor overrides are valid (a `q4` tensor with
`group_size=32` is legal).

### Why per-bit-width tuning

- **q2 / q3 at gs=32**: at 2 bits there are only 4 representable
  values per group. Doubling group size from 32 to 64 doubles the
  range a single scale must cover, halving effective resolution.
  The scale-storage cost (1 byte per group at e8m0) at gs=32 is
  ~3.1% of the weight bytes — negligible.
- **q4 / q5 / q6 at gs=64**: MLX-affine default. No accuracy reason
  to diverge.
- **q8 at gs=128**: an 8-bit weight already has 256 levels per group;
  increasing gs from 64 to 128 doubles the range but resolution loss
  is sub-1%. Scale storage drops from 1.6% to 0.8% of weight bytes.

## Scale dtypes

Three scale-dtype families are supported. The choice is per-tensor,
declared in the tensor entry as `scale_dtype`.

### `bf16` — full-precision scale

- 2 bytes per group.
- Default for q4 / q5 / q6 / q8.
- Rationale: matches the model's compute dtype on Apple M-series and
  modern CUDA hardware. Zero conversion cost on dequant.

### `f16` — legacy

- 2 bytes per group.
- Allowed for backwards-compat with MLX bundles that store F16 scales.
- Avoid for new bundles unless source weights are F16 only.

### `e8m0` — power-of-2 exponent ("bit-based scaling")

- 1 byte per group. Stores `floor(log2(|max| / qmax))` as a uint8 with
  bias 127, like the OCP MX format.
- Opt-in for any bit-width. Use case: memory-constrained deployments
  where halving scale storage matters more than the precision loss.
- Dequant: `dequant = q * (2^(e - 127))`. Shift-only on integer paths;
  fast-path for power-of-2 hardware.
- Loses non-power-of-2 scale ratios. Calibration must round-half-up
  to the nearest log2. Empirically AWQ-calibrated weights tolerate
  this with <0.5 ppl regression at q4+.
- Not a default at any bit-width: bf16 is always safe; e8m0 is a
  size/accuracy trade the user must opt into per-tensor in the
  profile.

### `e4m3` — fp8 with mantissa, q8 only

- 1 byte per group. 4-bit exponent + 3-bit mantissa per OCP fp8.
- Permitted only on q8 because the scale precision must keep up with
  the weight precision. At q4 a 3-bit mantissa scale dominates error
  budget; at q8 it doesn't.

## Symmetric vs asymmetric

Per-tensor choice via the `symmetric` flag (default: `false` =
asymmetric, matching MLX-affine).

- **Asymmetric (default)**: stores both `scale` and `bias`. Bias is
  the zero-point in real units, `dequant = q * scale + bias`. 2× scale
  storage. Best fit for distributions with non-zero mean (most weight
  matrices).
- **Symmetric**: stores only `scale`. `dequant = (q - 2^(bits-1)) * scale`
  — q is treated as signed. Halves scale storage. Required for some
  CUDA tensor-core paths that don't accept a per-group bias.

The runtime kernel set must implement both for every bit-width. The
choice is independent of bit-width.

## Packing

**Per-target-backend.** The converter emits packing native to the
target backend. The bundle's header carries `target_backend ∈ {metal,
cuda_sm89, cuda_sm90, rocm_cdna3, cpu_avx2, …}`. A bundle built for
one backend is not portable to another — re-convert from source.

### `metal` (default for Apple Silicon)

- Inherit MLX's affine packing exactly, so byte-identical bundles can
  be produced from MLX checkpoints (when source is fp16/bf16, not from
  pre-quantized MLX weights).
- Power-of-2 widths (q2/q4/q8): concatenated little-endian, lane `i`
  at bit `i*bits` in a uint32. Pack factor = `32/bits`.
- q3 / q5: bit-spread across 3 / 5 bytes per 8 lanes.
- q6: bit-spread across 3 bytes per 4 lanes.
- See MLX `mlx/backend/common/quantized.h::get_pack_factor`.

### `cuda_sm89` / `cuda_sm90`

- Tensor-core friendly: ldmatrix-compatible 8×8 tile interleave for
  q4/q8. q2/q3/q5/q6 emulated via shared-memory unpack.
- Symmetric scales required for tensor-core paths (no per-group bias
  on the int matmul). Asymmetric tensors fall back to dp4a path.

### `rocm_cdna3`

- MFMA-tile interleave. Details deferred to the AMD kernel PR.

### `cpu_avx2` / `cpu_neon`

- Row-major contiguous, no tile interleave. Fast SIMD-unpack via AVX2
  shuffle / NEON tbl.

The `layout` field on each tensor entry names the exact packing (e.g.
`layout: "metal_lane_strided_q4"`, `layout: "cuda_8x8_q4_sym"`).
Runtime dispatch keys on `(target_backend, layout)`.

## Header schema additions

Three new top-level fields:

```jsonc
{
  "target_backend": "metal",          // metal | cuda_sm89 | cuda_sm90 | rocm_cdna3 | cpu_avx2 | cpu_neon
  "quant_profile": "gemma4-moe-q4mix-v1",  // identifier of the profile used at convert time
  "calibration": { "method": "awq", "tokens": 1024, "dataset": "wikitext-103" }
}
```

Per-tensor entries gain three optional fields (defaults preserve
backwards-compat with current `base_q4` bundles):

```jsonc
{
  "name": "model.layers.0.mlp.experts.gate_proj.weight",
  "dtype": "base_q4",                 // base_q2 | base_q3 | base_q4 | base_q5 | base_q6 | base_q8 | bf16 | f16 | f32
  "group_size": 64,                   // per-bit-width default; overridable
  "scale_dtype": "bf16",              // bf16 | f16 | e8m0 | e4m3
  "symmetric": false,                 // default: false (asymmetric)
  "layout": "metal_lane_strided_q4",
  // ... offsets/lengths as in FORMAT.md
}
```

## Quant profile JSON

Profiles live in `tools/base-convert/profiles/`. Reusable + diffable.
The converter consumes one via `--profile <path>`; the resulting
bundle records the profile name in `quant_profile` for audit.

```jsonc
// profiles/gemma4-moe-q4mix-v1.json
{
  "name": "gemma4-moe-q4mix-v1",
  "arch": "gemma4",
  "calibration": { "method": "awq", "tokens": 1024, "dataset": "wikitext-103" },
  "rules": [
    { "pattern": "model.embed_tokens.weight",                       "dtype": "bf16" },
    { "pattern": "model.layers.*.input_layernorm.weight",           "dtype": "bf16" },
    { "pattern": "model.layers.*.post_attention_layernorm.weight",  "dtype": "bf16" },
    { "pattern": "model.layers.*.self_attn.{q,k,v,o}_proj.weight",  "dtype": "base_q4", "scale_dtype": "bf16", "group_size": 64 },
    { "pattern": "model.layers.*.mlp.gate_proj.weight",             "dtype": "base_q8", "scale_dtype": "e4m3", "group_size": 128 },
    { "pattern": "model.layers.*.mlp.up_proj.weight",               "dtype": "base_q8", "scale_dtype": "e4m3", "group_size": 128 },
    { "pattern": "model.layers.*.mlp.down_proj.weight",             "dtype": "base_q8", "scale_dtype": "e4m3", "group_size": 128 },
    { "pattern": "model.layers.*.mlp.experts.*.{gate,up,down}_proj.weight", "dtype": "base_q4", "scale_dtype": "bf16", "group_size": 64 },
    { "pattern": "model.layers.*.mlp.router.weight",                "dtype": "bf16" },
    { "pattern": "model.norm.weight",                               "dtype": "bf16" },
    { "pattern": "lm_head.weight",                                  "dtype": "base_q8", "scale_dtype": "bf16", "group_size": 128 }
  ]
}
```

Pattern matching is glob-style (`*` matches anything except `.`,
`**` matches anything; `{a,b}` is alternation). First-match-wins.
A profile must cover every tensor — converter fails loud on
unmatched. The `default-q4` profile is the catch-all for plain
dense models.

## Migration

- Existing `base_q4` (group_size=64, asymmetric, F16 scale, MLX
  packing) bundles remain valid. Backwards-compat: new converter reads
  them as `dtype=base_q4, group_size=64, scale_dtype=f16, symmetric=false,
  layout=metal_lane_strided_q4, target_backend=metal`.
- Existing `base_q8` bundles upgrade similarly.
- `scale_dtype=bf16` is the default for `base_q4`/`q5`/`q6`/`q8`: scales and
  biases are stored as bf16. Norms are stored at f16. The `*-f16scale.json`
  profiles (`default-q4-f16scale.json`, `default-q8-f16scale.json`) select
  f16 scales instead.
- The source checkpoint must be fp16/bf16/fp32. Re-quantizing an
  already-quantized checkpoint (e.g. a GGUF Q4_K) is not a supported target.

## Validation criterion

A bundle passes validation when:
- Every tensor's `(target_backend, dtype, group_size, scale_dtype,
  symmetric, layout)` tuple has a registered kernel in the target
  backend.
- Perplexity on a held-out WikiText-103 split is within 1% of the
  fp16 baseline (q4+) or within 5% (q3) or within 15% (q2).
- A 16-token TraceReference slot dequant matches the source-fp16
  forward pass within RMS error < 1e-3 (q4+) / 5e-3 (q2/q3).

## Open items deferred to phase 3

- Whether to also support a uniform `gs=64` "compatibility" mode for
  q2/q3 (for cross-method validation against literature).
- Whether to expose the calibration α schedule (per-layer or per-
  tensor) in the profile, or hide it behind the `--method` flag.
- Whether bf16 / f16 / fp32 source detection should be auto or
  user-declared via `--source-dtype`.
