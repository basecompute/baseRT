# The `.base` format

`.base` is BaseRT's on-disk model container: one file holding a JSON header
that describes the architecture, tokenizer and tensor inventory, followed by
the packed weights. A bundle can carry a vision or audio tower and a
speculative-decoding head in the same file, can be signed (ed25519), and
records where every tensor came from.

You produce `.base` files with [`basert convert`](../guides/converting.md) or
[`basert pull`](../guides/models.md), and inspect them with `basert inspect`.

## Canonical spec

The authoritative specifications live in the repo:

- **[`base-convert/FORMAT.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/FORMAT.md)**
  — file layout, header schema, tensor table, extension slots, and integrity.
- **[`base-convert/CANONICAL_QUANT_SPEC.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/CANONICAL_QUANT_SPEC.md)**
  — the quantized dtypes and their packing.

The reader and writer are the `base-format` crate in
[`base-convert/crates/`](https://github.com/basecompute/baseRT/tree/main/base-convert/crates).

## Inspecting a bundle

```sh
basert inspect models/your-model.base
basert inspect models/your-model.base --verify-checksums   # verify per-tensor xxhash64
```

This prints the header (architecture, config), the tensor inventory (names,
dtypes, shapes, group sizes), and the multimodal and speculator sections when
present.

## File layout

| Offset | Size | Field |
| --- | --- | --- |
| `0x00` | 4 | Magic, `BASE` |
| `0x04` | 4 | `format_version`, u32 little-endian. Currently **1**. |
| `0x08` | 8 | `header_len`, u64 little-endian |
| `0x10` | `header_len` | Header: canonical JSON (UTF-8, keys sorted, no insignificant whitespace) |
| — | — | Zero padding to the next 64 KiB boundary |
| — | — | Weights blob |
| — | — | Optional extension slots |

All integers are little-endian. The weights blob starts on a 64 KiB boundary
so tensors can be mapped without copying on every supported GPU. Inside the
blob each tensor is aligned for its compute region (see `alignment` below).

## Header fields

The header records, among other fields:

- `schema` — header schema number (currently 1).
- `arch` — the model architecture; it drives the runtime's forward path.
- `quant_scheme` — the bundle's default dtype. Individual tensors may differ.
- `created`, `baserT_version`, `min_hw` — when and by which converter the
  file was written, and the minimum hardware it targets.
- `source` — the source checkpoint's format (`gguf`, `mlx_safetensors` or
  `hf_safetensors`), file name and SHA-256.
- `tokenizer` — the Hugging Face tokenizer, embedded verbatim.
- `config` — the model configuration (hidden size, layers, heads, RoPE, …).
- `metadata` — converter-derived keys the runtime reads directly (for example
  Whisper token ids).
- `target_backend` — the backend the packing targets. Most bundles say
  `metal` and load on every platform.
- `quant_profile` — the name of the [quant profile](profiles.md) used at
  convert time, for audit.
- `alignment` — per-region alignment, in log2 bytes (`accel_align_log2`,
  `gpu_page_log2`, `cpu_align_log2`).
- `flags` — file-level capability bits (quantized, MoE, SSM, hybrid, LoRA,
  speculator, tied embeddings, sliding window, signed, …).
- `layers` — an optional per-layer map (attention, SSM, MoE expert counts).
- `tensors` — one entry per tensor; see below.
- `mmproj` — an optional vision or audio tower: its own `arch`, `config` and
  `tensors`, stored in the same weights blob.
- `speculator` — an optional drafter for speculative decoding: the
  checkpoint's own multi-token-prediction head, or a converted block drafter.
- `calibration` — the AWQ audit trail when calibration was applied: `mode`,
  `calib_tokens`, `dataset`, `source_fingerprint` and the per-layer alphas.
- `provenance` — which source tensors each bundle tensor was built from and
  what was done to them (copied verbatim, sliced, stacked, permuted). The
  runtime ignores it; verification tools use it.
- `sig` — the ed25519 signature, when the file is signed.

## Tensor entries

Each entry in `tensors` carries:

- `name`, `dtype`, `shape` — the shape is outermost axis first: a linear
  weight is `[out, in]` and an expert stack `[E, out, in]`.
- `offset`, `length` — the tensor's bytes within the weights blob, plus
  `scale_offset` / `scale_length` and `bias_offset` / `bias_length` for the
  per-group scales and zero points of a quantized tensor.
- `group_size`, `scale_dtype`, `symmetric` — the quantization grid.
- `layout` — the packing (`rowmajor` when absent). `gguf_super` marks GGUF
  super-blocks copied in verbatim, with `source_ggml_type` naming the GGUF
  type.
- `residency` (`hot`, `warm`, `cold`) and `compute_region` (`accelerator`,
  `gpu`, `cpu`) — placement hints for the loader.
- `flags` — per-tensor bits (expert weight, shared, tied, …).
- `checksum_xxh64` — the hash of the tensor's bytes.
- `algo` and `awq_alpha` — the rounding method (`rtn` or `awq`) and the AWQ
  alpha chosen, recorded for audit.

See [Quantization](quantization.md) for what the dtypes and grid fields mean.

## Integrity and signing

- Every tensor has an xxhash64 checksum, checked on demand
  (`basert inspect --verify-checksums`).
- A signed bundle carries an ed25519 signature over the canonical header
  (without `sig`) followed by the SHA-256 of the weights blob. See
  [Signing bundles](../guides/signing.md).

## Versions

`format_version` changes only on a breaking change to the layout or header
shape, and a loader refuses a version it does not know. New header fields are
optional, so an older reader ignores fields it does not recognise. Bundles that
spell their dtype `base4` / `base8` (the first format revision) are read as
`base_q4` / `base_q8`.
