# `.base` File Format — v1

Single-file cache format for baseRT. Replaces runtime GGUF/MLX loading.
Design goal: one format, one loader, kernel-native packing, mmap-friendly.

## File layout

```
offset    bytes      field
──────    ────────   ────────────────────────────────────────────────
0x0000    4          magic = b"BASE"
0x0004    4          format_version (u32 LE) = 1
0x0008    8          header_len (u64 LE) — length of header_json in bytes
0x0010    N          header_json (UTF-8, exactly header_len bytes)
──────    ────────   (padding: zero-bytes to next 64 KiB boundary)
0x???     M          weights_blob (raw tensor data, per-tensor alignment)
EOF
```

- All integers little-endian. All known target hardware is LE; we do not
  support BE hosts.
- `header_json` is canonical JSON (no trailing whitespace, keys sorted)
  so signatures are reproducible.
- Weights blob starts at a 64 KiB boundary (max page size across target
  platforms) so tensors aligned to 16 KiB (Apple) or 64 KiB (NVIDIA) land
  at page-aligned absolute file offsets — a precondition for zero-copy
  buffer creation via `MTLBuffer.makeBufferWithBytesNoCopy` and
  `cudaHostRegister`.
- Individual tensors inside the blob are aligned per their
  `compute_region` (64 B accelerator, page-size GPU, 64 B CPU). See
  §"Compute regions" below.
- Extension slots (LoRA deltas, compiled graphs, correctness traces…)
  live after the weights blob. See §"Extension slots".

## Compute regions

Every tensor declares a `compute_region`: `accelerator`, `gpu`, or `cpu`.
The region drives alignment and determines whether zero-copy buffer
creation is possible on the target hardware.

| Region       | Tenants                                        | Default align | Rationale                                                                 |
|--------------|------------------------------------------------|---------------|---------------------------------------------------------------------------|
| accelerator  | MLP/attn projections, MoE expert stacks        | 64–128 B      | ANE DMA (64 B), Tensor Core / Matrix Core tile (128 B), Hexagon NPU tile  |
| gpu          | Embeddings, norms, anything on shader path     | 16 KiB / 64 KiB | Metal page size (Apple) / cudaHostRegister granularity (NVIDIA/AMD)      |
| cpu          | SSM A-matrices, precomputed RoPE, tokenizer    | 64 B          | Cache line                                                                |

Alignment is read from the header's `alignment` field (log2 bytes per
region), not hardcoded by the loader. Converters can override the
GPU page size: `14` (16 KiB) for Apple-first shipping, `16` (64 KiB)
for NVIDIA/AMD targets. A universal bundle uses 16, costing a few MB
of padding for cross-platform portability.

**Zero-copy invariant**: a GPU-region tensor's absolute file offset
must be a multiple of the GPU page size. `BaseReader::tensor_is_zero_-
copy_eligible` verifies this at load; failure means the file was
produced by a buggy writer or has been tampered with.

## Design invariants

### No per-forward transformation

`.base` stores weights in the layout kernels consume directly. The runtime
must not transpose, repack, fuse, or fold scales on a per-forward basis.

All of the following happen at **convert-time**, baked into the bytes on
disk:

- Weight transpose to the order the target backend's GEMM expects
- AWQ per-channel scales folded into `W` (with α vector stored in
  `calibration` for audit / transcode)
- Q / K / V fusion (`qkv` stored under the fused name)
- Gate + up fusion for SwiGLU MLPs
- Asymmetric INT4 zero-point separated into explicit `bias` region
- SIMD / tensor-core tile interleave (declared via `layout`)

### Cross-backend repack is cached, not per-forward

A Metal-tile-interleaved `base_q4` tensor is not a CUDA-tensor-core layout.
Two options at convert-time:

1. Converter targets a specific layout (`--layout tile_8x8_mlx`) — best
   for appliance shipping.
2. Converter emits `rowmajor` and the first-run load repacks per-backend.
   Repacked bundle is cached at `~/.cache/baseRT/repacked/<sha>.<target>.base`
   so the cost is paid once per (model, backend) pair, not per process.

Either way, the invariant holds: **zero transformation in the forward
pass**. The `layout` field on each tensor tells the kernel whether it can
mmap-direct or needs the cached repack.

### Packing conventions

| Region              | Layout                                           |
|---------------------|--------------------------------------------------|
| Quantized weights   | Per-scheme (see "Quant scheme layouts" below)   |
| Scales              | Contiguous fp16 block, pointed by `scale_offset` |
| Biases (zero-points)| Contiguous fp16 block, pointed by `bias_offset`  |
| AWQ scales          | Contiguous fp16 block, per-in-channel            |
| f16 / bf16 / f32    | Row-major, byte-packed                           |

Scales and biases live in separate regions (not interleaved with weight
bytes) so GEMM kernels can prefetch them into registers once per group.

## Architecture coverage

The header schema is arch-open. Supported categories at v1:

- **Dense transformer** (Llama, Qwen, Gemma 3) — baseline.
- **MoE** (Qwen3-30B-A3B, Mixtral, Gemma 4 26B-A4B, DeepSeek-v2):
  expert stacks stored as `[num_experts, out, in]` 3D tensors. Router /
  gate tensors are separate 2D matrices. `config.moe_layout` declares
  `grouped` (all experts' up-projs contiguous, then gate-projs, then
  down-projs) vs `interleaved` (per-expert tuple contiguous). The choice
  drives page-adjacency for the table-decode kernel; `grouped` is the
  default because it matches the current Metal MoE kernel's access pattern.
- **SSM / Mamba / RWKV**: `A`, `B`, `C`, `D`, `delta_proj`, `dt_bias`,
  1D conv kernels all fit the tensor-name-and-shape schema. Per-tensor
  `dtype` allows f32-sensitive parameters (`dt_bias`, log-scaled `A`) to
  stay unquantized while the rest quantize. Runtime state shape goes in
  `config.ssm_state_shape`.
- **Hybrid** (Jamba, Mamba-2 with attention): `config.per_layer_override`
  is a list of `{layer_idx, sub_arch}` records that let specific layers
  declare a different sub-architecture. Tensor names include the layer
  index so the schema already disambiguates.
- **Multimodal** (Gemma 4 audio/vision, Llava): the separate tower's
  weights go under the `mmproj` sub-bundle. Tower config is
  `mmproj.config`. Projection head weights live in the main bundle under
  canonical names (`multi_modal_projector.*`).

Adding a new architecture = adding a `base-arch/<name>.rs` mapping from
source tensor names to canonical names. **No `format_version` bump**
required — the schema is arch-open.

## Residency

baseRT is deployed on unified-memory devices (Apple Silicon, mobile)
where budgeting GPU-resident memory against RAM matters. `.base` carries
metadata to help the runtime residency planner:

### Tensor ordering convention

Tensors are written to `weights_blob` in this order (and listed in the
`tensors` array in the same order):

```
embed_tokens
layer_0:  input_norm, qkv, o_proj, post_attn_norm, mlp_gate_up, mlp_down
layer_1:  ...
...
final_norm
lm_head
```

Rationale: OS page-cache readahead naturally prefetches layer N+1 while
layer N runs. MoE expert tensors cluster at the end of each layer's
block.

### Per-tensor residency hints

Each `TensorEntry` may carry an optional `residency` field:

- `hot` — always resident (embeddings, norms, routers, lm_head)
- `warm` — resident while owning layer is active (most weights)
- `cold` — on-demand (MoE inactive experts, pipeline-parallel stages)

Hints are priors, not constraints. Runtime measures access and adjusts.

### Budget metadata in header

```jsonc
"residency": {
  "total_weights_bytes": 16234567890,
  "hot_weights_bytes":    215000000,
  "max_expert_bytes_per_token": 130000000,   // MoE only
  "recommended_min_device_mb": 6000
}
```

Runtime compares against device RAM + projected KV cache and either:
- budgets `MTLResidencySet` accordingly,
- warns at load if `recommended_min_device_mb` exceeds available RAM,
- or falls back to CPU inference for cold tensors.

`.base` does **not** store KV cache or activation buffers — those are
runtime allocations.

## Tensor flags

Each tensor carries a `TensorFlags` bitfield (serialized as 0x-hex for
human-inspectable JSON). Flags are additive and survive transcoding.

| Bit  | Flag            | Meaning                                                         |
|------|-----------------|-----------------------------------------------------------------|
| 0    | TRANSPOSED      | Bytes are stored transposed; `shape` reflects logical dims      |
| 1    | EXPERT_WEIGHT   | Part of an MoE expert stack                                     |
| 2    | SHARED          | Aliases another tensor (loader resolves)                        |
| 3    | SSM_A_MATRIX    | SSM state-transition matrix; loader enforces f32 + cpu          |
| 4    | LORA_DELTA      | Lives in an extension slot, not the main blob                   |
| 5    | TIED            | Same bytes as another canonical tensor (e.g. lm_head ↔ embed)  |

The `SSM_A_MATRIX` constraint is load-time enforced: a file with such a
tensor in any non-CPU region or any dtype other than f32 is rejected
with `InvalidSsmAMatrix`. This catches the "I quantized the recurrent
state-transition matrix and now generation NaNs after 100 tokens"
silent correctness bug at load time rather than at inference.

## Header flags

Top-level `HeaderFlags` bitfield (also 0x-hex in JSON). Lets the loader
answer structural questions with a single bit-test.

| Bit  | Flag              |
|------|-------------------|
| 0    | QUANTIZED         |
| 1    | HAS_MOE           |
| 2    | HAS_SSM           |
| 3    | HAS_HYBRID        |
| 4    | HAS_LORA          |
| 5    | HAS_SPECULATOR    |
| 6    | HAS_COMPUTE_GRAPH |
| 7    | HAS_KV_WARMUP     |
| 8    | HAS_TRACE_REF     |
| 9    | ROPE_PRECOMPUTED  |
| 10   | TIED_EMBEDDINGS   |
| 11   | SLIDING_WINDOW    |
| 12   | SIGNED            |

## Layer map

A typed `layers: [LayerDescriptor]` array, one entry per layer, describes
the model's forward dispatch structure without requiring arch-string
pattern matching.

```jsonc
"layers": [
  { "kind": "attention_gqa",   "compute_hint": "accelerator" },
  { "kind": "attention_moe",   "moe_n_experts": 128, "moe_n_active": 8 },
  { "kind": "ssm",
    "compute_hint": "cpu",
    "precision": { "force_fp32_ssm": true } },
  { "kind": "attention_gqa",   "shared_attn_layer": 0 }  // Zamba-style
]
```

Supported kinds:

- `attention_dense`, `attention_gqa`, `attention_sliding` — attention
  variants
- `ssm`, `ssm_moe` — SSM (Mamba, RWKV) with optional MoE FFN
- `attention_moe`, `dense_moe` — attention + MoE FFN
- `dense_mlp` — classic transformer block

Per-layer precision overrides (`force_fp32_attn`, `force_fp32_ssm`,
`no_quantize`) let the converter mark specific layers as untouchable
while the rest quantize. `shared_attn_layer: N` enables Zamba-style
attention weight sharing by referencing an earlier layer's attention
weights.

## LoRA adapter bundles

A LoRA adapter is shipped as a stand-alone `.base` file, NOT as an
extension slot on a base bundle. The runtime opens it with the same
`WeightProvider` it uses for the base model, then installs it via
`baseRT_lora_load(model, path)`. `tools/lora_convert.py` produces the
file from a PEFT (HuggingFace) source.

### Header shape

The header is a regular `.base` header with two additions:

```jsonc
{
  "schema": 1,
  "arch": "llama",                 // base model's arch — informational
  "quant_scheme": "f16",           // LoRA halves are always f16
  "config": {                      // stub — extract_config requires non-zero
    "hidden_size": 1,              // values for these five keys; the
    "num_hidden_layers": <N>,      // adapter loader never reads them
    "num_attention_heads": 1,
    "head_dim": 1,
    "vocab_size": 1
  },
  "metadata": {
    "lora.rank":    16,            // adapter logical rank — used for scale
                                   // = α / r when pre-scaling B halves
    "lora.alpha":   64.0,
    "lora.max_seq": 4096           // optional; sizes the per-target
                                   // intermediate / delta scratch buffers
  },
  "tensors": [
    {
      "name":   "lora.layers.0.attention.q.weight.A",
      "dtype":  "f16",
      "shape":  [r_eff, K],        // see "Per-target effective rank"
      "offset": …, "length": …
    },
    {
      "name":   "lora.layers.0.attention.q.weight.B",
      "dtype":  "f16",
      "shape":  [N, r_eff],
      "offset": …, "length": …
    }
    // … one (A, B) pair per LoRA-patched linear
  ]
}
```

### Tensor naming

Each linear layer the adapter patches has a pair of f16 tensors:

```
lora.<canonical>.A   shape [r_eff, K]   — input-side low-rank projection
lora.<canonical>.B   shape [N,     r_eff] — output-side low-rank projection
```

`<canonical>` is the per-layer linear name the runtime's `dispatch_gemm`
passes as `tensor_name` — i.e. `layers.<i>.attention.q.weight`,
`layers.<i>.attention.output.weight`, `layers.<i>.ffn.gate.weight`,
`layers.<i>.ffn.down.weight`, etc. The runtime applies the delta
inline via two extra GEMMs after the main GEMM:

```
intermediate = x @ Aᵀ        // (M, K) @ (K, r_eff) → (M, r_eff)
delta        = intermediate @ Bᵀ   // (M, r_eff) @ (r_eff, N) → (M, N)
out         += delta         // residual_add (in-place)
```

`B` is pre-scaled by `α / lora.rank` at conversion time so the kernel
side needs no scale parameter.

### Per-target effective rank

`r_eff` need not equal `lora.rank`. When a runtime arch fuses multiple
projections at dispatch (e.g. Llama's QKV at `attention.q.weight` with
N = q_dim + 2·kv_dim), the converter emits a block-diagonal stack of
the per-member adapters:

```
fused_A = stack[A_q ; A_k ; A_v]              shape (3·r, K)
fused_B = block_diag(B_q, B_k, B_v)           shape (N_q + N_k + N_v, 3·r)
```

Then `fused_B @ fused_A @ x` produces the three independent low-rank
corrections vertically stacked into the fused output — exactly what
the main GEMM's caller expects to receive. Off-diagonal blocks in
`fused_B` are zero; that's wasted compute but correct, and the
overhead is negligible at typical adapter ranks (4–64).

`lora.rank` stays the per-member logical rank for `α / r` scaling;
the apply-delta code reads `r_eff` from each tensor's shape in the
header, not from the metadata. The loader rejects A/B pairs whose
ranks disagree.

### Per-arch fusion topology

`tools/lora_convert.py` derives the fusion choice from the PEFT
`adapter_config.json`'s `base_model_name_or_path`:

| Base model family                            | QKV          | gate+up      |
|----------------------------------------------|--------------|--------------|
| Llama / Gemma 3 / Qwen3 / Mistral / BERT     | fused (3·r)  | fused (2·r)  |
| Gemma 4                                      | separate     | separate     |

Adding a new arch is a one-line entry in `ARCH_TOPOLOGY`. The
runtime side needs no changes — `dispatch_gemm`'s tensor_name +
shape together disambiguate.

## Extension slots

A typed, length-prefixed, forward-compat section after the weights blob.
Each slot:

```
  u16  slot_kind         (see below; unknown values preserved)
  u16  slot_flags        (bit 0 REQUIRED, bit 1 COMPRESSED_ZSTD)
  u64  payload_length
  u64  payload_xxh64     (0 = unchecked)
  u8[payload_length]     payload
  <pad to 8-byte boundary>
```

Known kinds (all optional):

- `0x0001 LoraWeights` — rank-decomposed LoRA deltas that fuse onto the
  base at load
- `0x0002 ComputeGraph` — precompiled MPSGraph archive (Apple), CUDA
  graph (NVIDIA), or hipGraph (AMD). Saves seconds of cold start
- `0x0003 KvWarmup` — pre-filled KV cache tokens for common system
  prompts
- `0x0004 TraceReference` — 16-token fixed input + expected fp32
  logits. `baserT --validate` runs the forward pass at load and fails
  loud on RMS-error regression — catches silent quantization bugs per
  file, forever
- `0x0005 RopeTables` — precomputed cos/sin. Runtime skips RoPE compute
- `0x0006 CalibrationData` — per-tensor AWQ activation stats, JSON
- `0x00FF Custom` — vendor-defined

**Forward compatibility**: unknown `slot_kind` values are not an error.
The reader preserves the bytes and skips dispatch. New slot kinds ship
as minor-version additions. Slots marked `REQUIRED` that the loader
doesn't know how to process are a hard failure.

## Integrity

- **Per-tensor xxh64** in each `TensorEntry.checksum_xxh64`. Computed at
  write, lazy-verified at read. Strict mode (`baserT --validate`)
  eagerly verifies all tensors before dispatch.
- **Per-slot xxh64** in each extension slot header. Verified when the
  slot is parsed.
- **ed25519 signature** over canonical-JSON header + sha256 of weights
  blob. Enterprise builds fail loud on missing/invalid signature;
  default builds warn but proceed with `--allow-unsigned`.

Corruption detection (xxh64) and provenance (ed25519) are independent:
a tampered file may pass xxh64 if the attacker recomputed checksums;
ed25519 catches that because the attacker lacks the signing key.

## Header schema

```jsonc
{
  "schema": 1,
  "arch": "qwen3",                 // model arch; drives tensor-name convention
  "quant_scheme": "base_q4",       // one of: base_q4, base_q8, bf16, mxfp4, nvfp4
  "min_hw": "apple_m1",            // minimum hardware generation
  "created": "2026-04-24T12:00:00Z",
  "baserT_version": "0.9.0",
  "source": {                      // provenance of the conversion
    "format": "gguf",              // gguf | mlx_safetensors | hf_safetensors
    "sha256": "...",               // hash of source file(s)
    "filename": "qwen3-30b-a3b.gguf"
  },
  "tokenizer": {                   // HF tokenizer, embedded verbatim
    "model": { ... },
    "added_tokens": [...],
    "normalizer": {...},
    "pre_tokenizer": {...},
    "post_processor": {...},
    "decoder": {...}
  },
  "config": {                      // model config (hidden_dim, n_layers, etc.)
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "vocab_size": 151936,
    "rope_theta": 1000000,
    "rms_norm_eps": 1e-6,
    // ... arch-specific fields
  },
  "tensors": [
    {
      "name": "model.layers.0.self_attn.qkv_proj.weight",  // q/k/v fused at convert-time
      "dtype": "base_q4",          // base_q4 | base_q8 | bf16 | f16 | f32 | mxfp4 | nvfp4
      "shape": [2560, 2048],       // [fused_out, in]
      "offset": 0,                 // byte offset within weights_blob
      "length": 655360,            // byte length of packed data
      "scale_offset": 655360,      // optional: offset of scales (fp16)
      "scale_length": 5120,
      "bias_offset": 660480,       // optional: zero-point biases (asymmetric)
      "bias_length": 5120,
      "awq_scale_offset": 665600,  // optional: per-in-channel AWQ scales
      "awq_scale_length": 4096,
      "group_size": 64,            // quant group size; absent for bf16/f16
      "layout": "tile_8x8_mlx",    // kernel-native layout; rowmajor if absent
      "residency": "warm"          // hot | warm | cold; absent means "warm"
    }
    // ... one entry per tensor
  ],
  "mmproj": {                      // optional: multimodal projector
    "arch": "gemma4_vision",
    "tensors": [ ... ]             // same shape as top-level tensors
  },
  "calibration": {                 // optional: AWQ audit trail
    "mode": "full",                // full | lite | none
    "calib_tokens": 512,
    "per_layer_alpha": {           // retained for transcoding
      "model.layers.0.self_attn.q_proj": [...]
    }
  },
  "sig": {                         // optional: ed25519 signature
    "alg": "ed25519",
    "key_id": "baserT-official-2026",
    "signature": "base64(sig_over_header_bytes_without_sig_field || weights_blob_sha256)"
  }
}
```

### Canonical JSON for signing

The `sig` field is computed as:

```
payload = canonical_json(header with "sig" field removed)
         || sha256(weights_blob)
signature = ed25519_sign(private_key, payload)
```

Canonical JSON:
- UTF-8
- Keys sorted lexicographically at every level
- No insignificant whitespace
- Numbers in shortest round-trip form
- Strings escape only `\`, `"`, and control chars

### Quant scheme layouts in weights_blob

Each scheme has a fixed per-group layout. The header's `group_size` field
is authoritative for the tensor; kernels dispatch on `(scheme, group_size)`.

- **`base_q4`** (INT4 asymmetric, group_size=64): `[packed_i4s: group_size/2 bytes][scale: f16 in scales region][bias: f16 in biases region]`. Scales and biases are stored in separate contiguous regions within weights_blob, pointed to by `scale_offset` / `bias_offset`.
- **`base_q8`** (INT8 symmetric, group_size=128): `[packed_i8s: group_size bytes][scale: f16]`.
- **`bf16` / `f16` / `f32`**: raw, no scale region. `scale_offset` absent.
- **`mxfp4`** (OCP MX FP4 E2M1 + E8M0 shared scale, group_size=32): `[packed_fp4s: group_size/2 bytes][scale: u8 (E8M0)]`.
- **`nvfp4`** (NVIDIA FP4 E2M1 + E4M3 shared scale, group_size=16): `[packed_fp4s: group_size/2 bytes][scale: u8 (E4M3)]`.
- **`passthrough_gguf`**: the source GGUF tensor bytes wrapped unchanged.
  `layout` MUST be `gguf_super`. Escape hatch; requires the matching
  GGUF-scheme kernels to remain available at runtime. Not a default.

## Version policy

- `format_version` bumps on **any** breaking change to layout or header shape.
- Loader rejects unknown versions with a clear error. No silent migration.
- Forward-compat is via new optional header fields; old loaders ignore
  fields they don't recognize (JSON's natural extensibility).

## Minimum tensor inventory

Every valid `.base` bundle must contain, at minimum:
- `model.embed_tokens.weight` (or arch-equivalent)
- All per-layer attention + MLP weights as named by the arch
- `model.norm.weight` (final norm)
- `lm_head.weight` (or share with `embed_tokens`)
- A valid `tokenizer` field in the header

Conversion fails loud if any arch-required tensor is missing.

## Not in scope (v1)

- Sharding across multiple files (may add `.base.00001` naming in v2 if
  single-file >100GB becomes common)
- Streaming convert (whole-model must fit in RAM during convert; runtime
  is mmap-only so it's streaming-friendly post-convert)
- Delta/LoRA packaging (separate `.base-lora` format, future)

## Runtime auto-convert

When baseRT is pointed at a `.gguf` / `.safetensors` / safetensors-dir, it:

1. Computes `input_hash` = sha256 of file bytes (or sorted-listing+sizes
   hash for a directory).
2. Checks `~/.cache/baseRT/converted/<input_hash>.base`.
3. If present: mmap and load.
4. If absent: shell out to `base-convert <input> -o <cache-path>`, show
   progress, then load.

The cache path can be overridden with `--cache-dir`. `--force-reconvert`
bypasses the cache. `--no-auto-convert` fails loud on non-`.base` inputs.
