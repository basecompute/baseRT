use bitflags::bitflags;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

bitflags! {
    /// File-level capability flags. Loader can use these as a fast
    /// dispatch: "does this model have MoE?" is one bit-test rather than
    /// scanning the layer map.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct HeaderFlags: u32 {
        /// At least one tensor uses a sub-fp16 dtype.
        const QUANTIZED         = 1 << 0;
        /// Model has mixture-of-experts layers.
        const HAS_MOE           = 1 << 1;
        /// Model has SSM (Mamba / S4 / RWKV) layers.
        const HAS_SSM           = 1 << 2;
        /// Model interleaves attention and SSM (Jamba, Zamba, Bamba).
        const HAS_HYBRID        = 1 << 3;
        /// Extension slots contain LoRA delta weights.
        const HAS_LORA          = 1 << 4;
        /// Extension slots contain a paired speculator model.
        const HAS_SPECULATOR    = 1 << 5;
        /// Extension slots contain precompiled compute graphs.
        const HAS_COMPUTE_GRAPH = 1 << 6;
        /// Extension slots contain a KV-cache warmup sequence.
        const HAS_KV_WARMUP     = 1 << 7;
        /// Extension slots contain a correctness trace reference.
        const HAS_TRACE_REF     = 1 << 8;
        /// Precomputed RoPE cos/sin tables available in extension slots —
        /// runtime can skip computing them.
        const ROPE_PRECOMPUTED  = 1 << 9;
        /// Input embeddings and lm_head share the same tensor bytes.
        const TIED_EMBEDDINGS   = 1 << 10;
        /// At least one attention layer uses sliding-window attention.
        const SLIDING_WINDOW    = 1 << 11;
        /// Manifest is ed25519-signed (sig field present and trusted).
        const SIGNED            = 1 << 12;
    }
}

impl Serialize for HeaderFlags {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(&format_args!("0x{:08x}", self.bits()))
    }
}

impl<'de> Deserialize<'de> for HeaderFlags {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s: String = Deserialize::deserialize(d)?;
        let bits = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u32::from_str_radix(hex, 16).map_err(serde::de::Error::custom)?
        } else {
            s.parse::<u32>().map_err(serde::de::Error::custom)?
        };
        Ok(HeaderFlags::from_bits_truncate(bits))
    }
}

bitflags! {
    /// Per-tensor flags. Stored in the JSON as a lowercase-hex u32 string.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct TensorFlags: u32 {
        /// Weights are stored transposed for the GEMM kernel — i.e. the
        /// declared `shape` is `[out, in]` but the bytes are in `[in, out]`
        /// order. Rarely set with the `layout` system; present for audit.
        const TRANSPOSED     = 1 << 0;
        /// Part of an MoE expert stack. `expert_id` in the name (or the
        /// shape's outer dimension) identifies the expert.
        const EXPERT_WEIGHT  = 1 << 1;
        /// Shared with another tensor (typical: lm_head ↔ embed_tokens).
        /// The reader resolves aliases when building its lookup.
        const SHARED         = 1 << 2;
        /// SSM state-transition matrix (Mamba `A`, RWKV decay). Must be
        /// f32 in the cpu region — loader enforces this.
        const SSM_A_MATRIX   = 1 << 3;
        /// LoRA delta weight carried in an extension slot, not the main
        /// blob. Should not appear in the primary `tensors` array.
        const LORA_DELTA     = 1 << 4;
        /// Tied embeddings: this tensor is the same bytes as another
        /// canonical tensor (typically embed_tokens).
        const TIED           = 1 << 5;
    }
}

impl Serialize for TensorFlags {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(&format_args!("0x{:08x}", self.bits()))
    }
}

impl<'de> Deserialize<'de> for TensorFlags {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s: String = Deserialize::deserialize(d)?;
        let bits = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u32::from_str_radix(hex, 16).map_err(serde::de::Error::custom)?
        } else {
            s.parse::<u32>().map_err(serde::de::Error::custom)?
        };
        Ok(TensorFlags::from_bits_truncate(bits))
    }
}

/// Bundle-default quant scheme. Per-tensor `TensorDtype` overrides this
/// default for individual tensors (mixed precision is the norm under the
/// canonical-quant migration).
///
/// New bundles use the canonical `BaseQN` variants (serialize as
/// `"base_qN"`). Reading is back-compat with v1.0 bundles via serde
/// `alias`: `"base4"` deserializes to `BaseQ4`, `"base8"` to `BaseQ8`.
/// Re-serializing changes `"base4"` → `"base_q4"`, so old bundles aren't
/// byte-exact roundtrippable through the writer; reading + dispatch
/// remain identical.
///
/// `PassthroughGguf` is the v1 escape hatch — deprecated under the
/// canonical-quant migration; runtime kernels backing it (`gemv_q8_0`,
/// `moe_simd_gemm_q4_k`, etc.) are scheduled for removal in Phase 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantScheme {
    #[serde(rename = "base_q2")]
    BaseQ2,
    #[serde(rename = "base_q3")]
    BaseQ3,
    /// Canonical 4-bit. Reads "base_q4" or legacy "base4".
    #[serde(rename = "base_q4", alias = "base4")]
    BaseQ4,
    #[serde(rename = "base_q5")]
    BaseQ5,
    #[serde(rename = "base_q6")]
    BaseQ6,
    /// Canonical 8-bit. Reads "base_q8" or legacy "base8".
    #[serde(rename = "base_q8", alias = "base8")]
    BaseQ8,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "mxfp4")]
    Mxfp4,
    #[serde(rename = "nvfp4")]
    Nvfp4,
    #[serde(rename = "passthrough_gguf")]
    PassthroughGguf,
}

/// Per-group scale storage dtype. Independent of the weight bit-width.
/// See `CANONICAL_QUANT_SPEC.md` for the per-bit-width default + allowed
/// table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScaleDtype {
    /// 2 B/group. Default for q4/q5/q6/q8 and the safe choice everywhere.
    #[default]
    Bf16,
    /// 2 B/group. Legacy MLX-affine compatibility.
    F16,
    /// 1 B/group. Power-of-2 exponent (OCP MX-style). Opt-in for
    /// memory-constrained deployments; loses non-power-of-2 ratios.
    E8m0,
    /// 1 B/group. fp8 with 4-bit exponent + 3-bit mantissa. Permitted
    /// only on q8 (mantissa precision must keep up with the weight).
    E4m3,
}

impl ScaleDtype {
    /// Bytes per group occupied by a single scale value.
    pub fn bytes_per_group(self) -> u64 {
        match self {
            ScaleDtype::Bf16 | ScaleDtype::F16 => 2,
            ScaleDtype::E8m0 | ScaleDtype::E4m3 => 1,
        }
    }
}

/// Target hardware backend a bundle is packed for. Bundles are not
/// portable across backends — the converter writes one backend's
/// kernel-native packing per `--target=` invocation. The runtime
/// rejects bundles whose `target_backend` doesn't match its build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TargetBackend {
    /// Apple Silicon GPU. Inherits MLX-affine packing for compatibility
    /// with byte-equivalent MLX checkpoints.
    #[default]
    Metal,
    /// NVIDIA Ada / Hopper (sm_89 / sm_90). Tensor-core friendly tile
    /// layouts.
    CudaSm89,
    CudaSm90,
    /// AMD CDNA3 (MI300). MFMA tile layout.
    RocmCdna3,
    /// CPU AVX2 / NEON fallback paths. Row-major contiguous packing.
    CpuAvx2,
    CpuNeon,
}

/// Tensor on-disk layout. The kernel that consumes a tensor asserts which
/// layout it expects; mismatch triggers a one-time repack cached on disk.
///
/// Power-of-2 widths (q2/q4/q8) under `MetalLaneStrided`: lanes
/// concatenated little-endian into uint32, lane `i` at bit `i*bits`,
/// matching MLX-affine. `MetalBitSpread*` covers q3/q5/q6 where lanes
/// straddle byte boundaries (3/5/3 bytes per 8/8/4 lanes respectively).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// Plain row-major [rows, cols] with no tile interleave. Default for
    /// bf16/f16/f32 tensors and for anything the kernel will repack.
    Rowmajor,
    /// Metal SIMD-group 8×8 tile interleave (matches simd_gemm_q4 family).
    /// Legacy alias for v1.0 base_q4 bundles.
    Tile8x8Mlx,
    /// Metal lane-strided (MLX-affine). For q2/q4/q8.
    MetalLaneStrided,
    /// Metal bit-spread layout for q3/q5/q6. Sub-byte widths that
    /// straddle byte boundaries. Disambiguator (which width) lives on
    /// the tensor's `dtype`.
    MetalBitSpread,
    /// CUDA tensor-core 16×8×16 tile layout (Blackwell).
    CudaTcM16n8k16,
    /// CUDA dp4a path (non-tensor-core fallback). Row-major contiguous
    /// for sub-byte unpack via shared memory.
    CudaDp4a,
    /// AMD MFMA tile layout (CDNA3).
    RocmMfma,
    /// CPU AVX2 / NEON row-major contiguous, fast SIMD-unpack via
    /// shuffle / tbl.
    CpuRowmajor,
    /// Raw GGUF super-block layout (only when quant_scheme = passthrough_gguf).
    /// Deprecated under canonical-quant migration.
    GgufSuper,
}

/// Residency hint for the runtime's MTLResidencySet / GPU paging planner.
///
/// Hints are a prior, not a constraint. The runtime may override based on
/// measured access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyHint {
    /// Always keep resident (embeddings, norms, routers, lm_head).
    Hot,
    /// Resident while the owning layer is active (most weights).
    Warm,
    /// On-demand (MoE inactive experts, pipeline-parallel other stages).
    Cold,
}

/// Which sub-region of the weights blob a tensor lives in. The region
/// drives the tensor's alignment, which in turn drives whether zero-copy
/// buffer creation is possible on the target hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeRegion {
    /// Highest-throughput matmul unit: ANE, Tensor Cores, Matrix Cores,
    /// Hexagon NPU. Tight alignment (64-128 B). Typical tenants: MLP
    /// weights, attention projections, MoE expert stacks.
    Accelerator,
    /// General-purpose GPU shader path. Must be page-aligned so
    /// MTLBuffer.makeBufferWithBytesNoCopy (Apple) or cudaHostRegister
    /// (NVIDIA/AMD) succeeds without silent copy fallback. 16 KiB on
    /// Apple, 64 KiB on NVIDIA/AMD.
    #[default]
    Gpu,
    /// Host CPU path. Cache-line aligned (64 B). Tenants: SSM A-matrices
    /// (numerically sensitive, must stay f32), precomputed RoPE tables,
    /// tokenizer-adjacent tensors.
    Cpu,
}


/// Per-region alignment in log2 bytes. Stored in the header so the runtime
/// knows the packed layout without hardcoding assumptions, and so the
/// converter can target different hardware page sizes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Accelerator-region tensor alignment, log2 bytes.
    /// Default: 6 (64 B). CUDA Tensor Core / AMD Matrix Core builds
    /// override to 7 (128 B).
    pub accel_align_log2: u8,
    /// GPU-region tensor alignment, log2 bytes.
    /// Default: 14 (16 KiB) — Apple Metal page. NVIDIA/AMD override to
    /// 16 (64 KiB) to match cudaHostRegister / hipHostRegister.
    pub gpu_page_log2: u8,
    /// CPU-region tensor alignment, log2 bytes. Default: 6 (64 B).
    pub cpu_align_log2: u8,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            accel_align_log2: 6,
            gpu_page_log2: 14,
            cpu_align_log2: 6,
        }
    }
}

impl AlignmentConfig {
    pub fn align_for(&self, region: ComputeRegion) -> u64 {
        let log2 = match region {
            ComputeRegion::Accelerator => self.accel_align_log2,
            ComputeRegion::Gpu => self.gpu_page_log2,
            ComputeRegion::Cpu => self.cpu_align_log2,
        };
        1u64 << log2
    }
}

/// Per-tensor dtype. A tensor's dtype may differ from the bundle's default
/// `quant_scheme` — for example a `base_q4` bundle typically stores its
/// `lm_head` as `bf16`. Mixed precision (per-tensor bit-width selection)
/// is the norm under the canonical-quant migration.
///
/// New variants `BaseQ2`/`BaseQ3`/`BaseQ5`/`BaseQ6` join `BaseQ4`/`BaseQ8`
/// for full coverage of the bit-widths in the canonical spec. Legacy
/// JSON `"base4"` / `"base8"` still deserialize via aliases on the
/// canonical variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    #[serde(rename = "base_q2")]
    BaseQ2,
    #[serde(rename = "base_q3")]
    BaseQ3,
    /// Canonical 4-bit. Reads "base_q4" or legacy "base4".
    #[serde(rename = "base_q4", alias = "base4")]
    BaseQ4,
    #[serde(rename = "base_q5")]
    BaseQ5,
    #[serde(rename = "base_q6")]
    BaseQ6,
    /// Canonical 8-bit. Reads "base_q8" or legacy "base8".
    #[serde(rename = "base_q8", alias = "base8")]
    BaseQ8,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "mxfp4")]
    Mxfp4,
    #[serde(rename = "nvfp4")]
    Nvfp4,
}

impl TensorDtype {
    /// Bit-width per weight element. None for non-quantized dtypes.
    pub fn bits_per_weight(self) -> Option<u32> {
        match self {
            TensorDtype::BaseQ2 => Some(2),
            TensorDtype::BaseQ3 => Some(3),
            TensorDtype::BaseQ4 => Some(4),
            TensorDtype::BaseQ5 => Some(5),
            TensorDtype::BaseQ6 => Some(6),
            TensorDtype::BaseQ8 => Some(8),
            TensorDtype::Mxfp4 | TensorDtype::Nvfp4 => Some(4),
            TensorDtype::Bf16 | TensorDtype::F16 => Some(16),
            TensorDtype::F32 => Some(32),
        }
    }

    /// True if this dtype carries per-group scales (and possibly biases).
    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            TensorDtype::BaseQ2
                | TensorDtype::BaseQ3
                | TensorDtype::BaseQ4
                | TensorDtype::BaseQ5
                | TensorDtype::BaseQ6
                | TensorDtype::BaseQ8
                | TensorDtype::Mxfp4
                | TensorDtype::Nvfp4
        )
    }

    /// Canonical default group_size per bit-width, per the canonical-quant
    /// spec (`CANONICAL_QUANT_SPEC.md`).
    pub fn default_group_size(self) -> Option<u32> {
        match self {
            TensorDtype::BaseQ2 | TensorDtype::BaseQ3 => Some(32),
            TensorDtype::BaseQ4 | TensorDtype::BaseQ5 | TensorDtype::BaseQ6 => Some(64),
            TensorDtype::BaseQ8 => Some(128),
            TensorDtype::Mxfp4 => Some(32),
            TensorDtype::Nvfp4 => Some(16),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub length: u64,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub scale_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub scale_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bias_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bias_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub awq_scale_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub awq_scale_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub group_size: Option<u32>,

    /// Per-group scale dtype. None = inherit the bit-width's canonical
    /// default per `CANONICAL_QUANT_SPEC.md` (bf16 for all bit-widths
    /// 2026-04-29). Only meaningful for quantized dtypes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub scale_dtype: Option<ScaleDtype>,

    /// True = symmetric quant (only `scale`, no `bias`; `dequant = (q -
    /// 2^(bits-1)) * scale`). False/absent = asymmetric (default,
    /// matches MLX-affine: stores both `scale` and `bias`,
    /// `dequant = q * scale + bias`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub symmetric: bool,

    /// On-disk layout. If absent, rowmajor is assumed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub layout: Option<Layout>,

    /// Residency hint for the runtime paging planner.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub residency: Option<ResidencyHint>,

    /// Which sub-region this tensor lives in. Drives alignment and
    /// zero-copy buffer creation at load. Defaults to `Gpu` if absent.
    #[serde(default, skip_serializing_if = "is_default_region")]
    pub compute_region: ComputeRegion,

    /// Bitflags. See `TensorFlags`. Absent = empty.
    #[serde(default, skip_serializing_if = "TensorFlags::is_empty")]
    pub flags: TensorFlags,

    /// xxhash64 of the tensor's raw bytes (not including scale/bias
    /// regions). Computed at write, lazily verified on first access. None
    /// when the writer skipped hashing (fast-path test fixtures).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum_xxh64: Option<u64>,

    /// Source GGUF ggml_type code, when this tensor is a bit-exact
    /// passthrough of a GGUF block layout (Layout::GgufSuper). Lets the
    /// runtime dispatch to the matching native kernel
    /// (`moe_simd_gemm_q4_k`, `moe_simd_gemm_q5_0`, `moe_simd_gemm_q6_k`,
    /// etc.) without us widening the .base `TensorDtype` enum with one
    /// variant per GGUF block scheme. Values are GGUF dtype codes
    /// (Q4_K=12, Q5_0=6, Q6_K=14, Q8_0=8, …); absent = re-quanted to
    /// the bundle's `quant_scheme`, no GGUF-native dispatch needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ggml_type: Option<u32>,
}

fn is_default_region(r: &ComputeRegion) -> bool {
    matches!(r, ComputeRegion::Gpu)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub format: String,
    pub sha256: String,
    pub filename: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(flatten)]
    pub fields: BTreeMap<String, serde_json::Value>,
}

/// Opaque HF tokenizer spec, embedded verbatim.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerBlob {
    #[serde(flatten)]
    pub fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    pub mode: String,
    pub calib_tokens: u32,
    #[serde(default)]
    pub per_layer_alpha: BTreeMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub alg: String,
    pub key_id: String,
    /// Base64-encoded signature bytes.
    pub signature: String,
}

/// Top-level header. Serialized as JSON with sorted keys for
/// reproducible signing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub schema: u32,
    pub arch: String,
    pub quant_scheme: QuantScheme,
    pub min_hw: String,
    pub created: String,
    #[serde(rename = "baserT_version")]
    pub base_rt_version: String,

    pub source: SourceInfo,
    pub tokenizer: TokenizerBlob,
    pub config: ModelConfig,

    /// Hardware backend this bundle is packed for. Bundles aren't
    /// portable across backends — the converter writes one backend's
    /// kernel-native packing per `--target=` invocation. Defaults to
    /// `metal` for backwards-compat with v1.0 bundles, which were all
    /// implicitly Metal-targeted.
    #[serde(default)]
    pub target_backend: TargetBackend,

    /// Identifier of the quant profile used at convert time (see
    /// `tools/base-convert/profiles/*.json`). Empty for v1.0 bundles
    /// produced before profiles existed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub quant_profile: String,

    /// Per-region alignment. Drives how the writer places tensors in the
    /// blob and how the loader validates zero-copy eligibility.
    #[serde(default)]
    pub alignment: AlignmentConfig,

    /// File-level capability flags. See `HeaderFlags`.
    #[serde(default, skip_serializing_if = "HeaderFlags::is_empty")]
    pub flags: HeaderFlags,

    /// Ordered layer descriptors. When present, len must equal
    /// `config["num_hidden_layers"]` (not currently enforced at parse
    /// time — loader responsibility).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerDescriptor>,

    pub tensors: Vec<TensorEntry>,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub mmproj: Option<MmprojBundle>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub calibration: Option<CalibrationInfo>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sig: Option<Signature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmprojBundle {
    pub arch: String,
    /// Multimodal config block. Mirrors the open-namespace shape of
    /// top-level `config`: holds `vision_config`/`audio_config` sub-objects
    /// and the multimodal token IDs (`image_token_id`, `boi_token_id`,
    /// etc.) that the runtime needs to wire image/audio prefill. Empty
    /// when the source had no multimodal config (e.g. text-only with a
    /// stray tower checkpoint — unusual).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub config: BTreeMap<String, serde_json::Value>,
    pub tensors: Vec<TensorEntry>,
}

/// Per-layer kind. Drives runtime dispatch: which forward path to run
/// (attention vs SSM), which KV / SSM buffers to allocate, and whether
/// the layer feeds into an MoE FFN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Standard dense multi-head attention (no GQA).
    AttentionDense,
    /// Grouped-query attention.
    AttentionGqa,
    /// Sliding-window attention (local, bounded context).
    AttentionSliding,
    /// Pure SSM (Mamba / S4 / RWKV) layer, no attention.
    Ssm,
    /// SSM with MoE FFN (e.g. Bamba).
    SsmMoe,
    /// Attention with MoE FFN (Mixtral, Qwen3-30B-A3B, Gemma4 26B-A4B).
    AttentionMoe,
    /// Dense attention + MoE FFN (when attention_variant is neither GQA
    /// nor sliding; rare).
    DenseMoe,
    /// Standard transformer block (attention + dense MLP).
    DenseMlp,
}

/// Per-layer precision overrides. Absent = inherit bundle default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct LayerPrecision {
    /// Force attention compute to f32 (numerical sensitivity).
    #[serde(default, skip_serializing_if = "is_false")]
    pub force_fp32_attn: bool,
    /// Force SSM state update to f32 — strongly recommended.
    #[serde(default, skip_serializing_if = "is_false")]
    pub force_fp32_ssm: bool,
    /// Skip quantization for this layer entirely.
    #[serde(default, skip_serializing_if = "is_false")]
    pub no_quantize: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Typed layer descriptor. One entry per layer; the i-th entry describes
/// layer i. Bounded size keeps dispatch table construction cheap and
/// lets the runtime plan KV / SSM buffer sizes without scanning tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDescriptor {
    pub kind: LayerKind,

    /// For MoE layers: total experts in this layer. 0 for dense layers.
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub moe_n_experts: u16,
    /// For MoE layers: experts active per token (top-K).
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub moe_n_active: u16,

    /// For Zamba-style weight-shared attention: index of the layer whose
    /// attention weights this layer reuses. None if not shared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_attn_layer: Option<u16>,

    /// Preferred compute unit for the forward pass of this layer.
    /// Hint only; runtime measures and adjusts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_hint: Option<ComputeRegion>,

    /// Per-layer precision overrides. Default = empty.
    #[serde(default, skip_serializing_if = "is_default_precision")]
    pub precision: LayerPrecision,
}

fn is_zero_u16(n: &u16) -> bool {
    *n == 0
}

fn is_default_precision(p: &LayerPrecision) -> bool {
    !p.force_fp32_attn && !p.force_fp32_ssm && !p.no_quantize
}

impl Header {
    /// Canonical JSON encoding used for signing and for on-disk storage.
    /// BTreeMap + serde_json preserves sorted keys at every level.
    pub fn to_canonical_json(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(self)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    /// Produce a copy of `self` with the `sig` field cleared — used as the
    /// input to signature computation so the signature doesn't need to
    /// sign itself.
    pub fn without_sig(&self) -> Self {
        let mut clone = self.clone();
        clone.sig = None;
        clone
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each canonical TensorDtype variant must serialize to its
    /// `base_qN` JSON name, not the snake_case enum-variant name.
    #[test]
    fn tensor_dtype_canonical_serialization() {
        assert_eq!(
            serde_json::to_string(&TensorDtype::BaseQ2).unwrap(),
            "\"base_q2\""
        );
        assert_eq!(
            serde_json::to_string(&TensorDtype::BaseQ3).unwrap(),
            "\"base_q3\""
        );
        assert_eq!(
            serde_json::to_string(&TensorDtype::BaseQ4).unwrap(),
            "\"base_q4\""
        );
        assert_eq!(
            serde_json::to_string(&TensorDtype::BaseQ5).unwrap(),
            "\"base_q5\""
        );
        assert_eq!(
            serde_json::to_string(&TensorDtype::BaseQ6).unwrap(),
            "\"base_q6\""
        );
        assert_eq!(
            serde_json::to_string(&TensorDtype::BaseQ8).unwrap(),
            "\"base_q8\""
        );
    }

    /// v1.0 bundles wrote `"base4"` / `"base8"`. Canonical-quant migration
    /// renames to `"base_q4"` / `"base_q8"` for consistency with the
    /// q2/q3/q5/q6 names. Old bundles must still deserialize via aliases.
    #[test]
    fn tensor_dtype_v1_back_compat() {
        let q4: TensorDtype = serde_json::from_str("\"base4\"").unwrap();
        assert_eq!(q4, TensorDtype::BaseQ4);
        let q8: TensorDtype = serde_json::from_str("\"base8\"").unwrap();
        assert_eq!(q8, TensorDtype::BaseQ8);

        // QuantScheme has the same back-compat contract.
        let q4: QuantScheme = serde_json::from_str("\"base4\"").unwrap();
        assert_eq!(q4, QuantScheme::BaseQ4);
        let q8: QuantScheme = serde_json::from_str("\"base8\"").unwrap();
        assert_eq!(q8, QuantScheme::BaseQ8);
    }

    /// Per the canonical-quant spec: q2/q3 default gs=32, q4/q5/q6 = 64,
    /// q8 = 128.
    #[test]
    fn tensor_dtype_default_group_size_per_spec() {
        assert_eq!(TensorDtype::BaseQ2.default_group_size(), Some(32));
        assert_eq!(TensorDtype::BaseQ3.default_group_size(), Some(32));
        assert_eq!(TensorDtype::BaseQ4.default_group_size(), Some(64));
        assert_eq!(TensorDtype::BaseQ5.default_group_size(), Some(64));
        assert_eq!(TensorDtype::BaseQ6.default_group_size(), Some(64));
        assert_eq!(TensorDtype::BaseQ8.default_group_size(), Some(128));
        assert_eq!(TensorDtype::Bf16.default_group_size(), None);
    }

    #[test]
    fn tensor_dtype_bits_per_weight() {
        assert_eq!(TensorDtype::BaseQ2.bits_per_weight(), Some(2));
        assert_eq!(TensorDtype::BaseQ3.bits_per_weight(), Some(3));
        assert_eq!(TensorDtype::BaseQ4.bits_per_weight(), Some(4));
        assert_eq!(TensorDtype::BaseQ5.bits_per_weight(), Some(5));
        assert_eq!(TensorDtype::BaseQ6.bits_per_weight(), Some(6));
        assert_eq!(TensorDtype::BaseQ8.bits_per_weight(), Some(8));
        assert_eq!(TensorDtype::Bf16.bits_per_weight(), Some(16));
        assert_eq!(TensorDtype::F32.bits_per_weight(), Some(32));
    }

    /// New scale-dtype enum: bf16/f16=2 bytes, e8m0/e4m3=1 byte.
    #[test]
    fn scale_dtype_serialization_and_size() {
        assert_eq!(
            serde_json::to_string(&ScaleDtype::Bf16).unwrap(),
            "\"bf16\""
        );
        assert_eq!(serde_json::to_string(&ScaleDtype::F16).unwrap(), "\"f16\"");
        assert_eq!(
            serde_json::to_string(&ScaleDtype::E8m0).unwrap(),
            "\"e8m0\""
        );
        assert_eq!(
            serde_json::to_string(&ScaleDtype::E4m3).unwrap(),
            "\"e4m3\""
        );

        assert_eq!(ScaleDtype::Bf16.bytes_per_group(), 2);
        assert_eq!(ScaleDtype::F16.bytes_per_group(), 2);
        assert_eq!(ScaleDtype::E8m0.bytes_per_group(), 1);
        assert_eq!(ScaleDtype::E4m3.bytes_per_group(), 1);
    }

    #[test]
    fn target_backend_serialization() {
        assert_eq!(
            serde_json::to_string(&TargetBackend::Metal).unwrap(),
            "\"metal\""
        );
        assert_eq!(
            serde_json::to_string(&TargetBackend::CudaSm89).unwrap(),
            "\"cuda_sm89\""
        );
        assert_eq!(
            serde_json::to_string(&TargetBackend::CudaSm90).unwrap(),
            "\"cuda_sm90\""
        );
        assert_eq!(
            serde_json::to_string(&TargetBackend::RocmCdna3).unwrap(),
            "\"rocm_cdna3\""
        );
    }

    /// `target_backend` defaults to Metal for v1.0 back-compat.
    #[test]
    fn header_target_backend_defaults_to_metal_when_absent() {
        // Build a minimal header JSON without target_backend (v1.0 shape).
        let json = serde_json::json!({
            "schema": 1,
            "arch": "test",
            "quant_scheme": "base_q4",
            "min_hw": "apple_m1",
            "created": "2026-04-29T00:00:00Z",
            "baserT_version": "0.1.0",
            "source": {
                "format": "test",
                "sha256": "0".repeat(64),
                "filename": "x"
            },
            "tokenizer": {},
            "config": {},
            "tensors": []
        });
        let header: Header = serde_json::from_value(json).unwrap();
        assert_eq!(header.target_backend, TargetBackend::Metal);
        assert_eq!(header.quant_profile, "");
    }

    /// Per-tensor scale_dtype/symmetric default to None/false. New
    /// fields don't appear in the JSON when unset (back-compat).
    #[test]
    fn tensor_entry_canonical_fields_optional() {
        let entry = TensorEntry {
            name: "x".into(),
            dtype: TensorDtype::BaseQ4,
            shape: vec![128, 64],
            offset: 0,
            length: 64,
            scale_offset: None,
            scale_length: None,
            bias_offset: None,
            bias_length: None,
            awq_scale_offset: None,
            awq_scale_length: None,
            group_size: Some(64),
            scale_dtype: None,
            symmetric: false,
            layout: None,
            residency: None,
            compute_region: ComputeRegion::Gpu,
            flags: TensorFlags::empty(),
            checksum_xxh64: None,
            source_ggml_type: None,
        };
        let s = serde_json::to_string(&entry).unwrap();
        assert!(!s.contains("scale_dtype"), "scale_dtype: None should skip");
        assert!(!s.contains("symmetric"), "symmetric: false should skip");
    }

    /// scale_dtype + symmetric round-trip when set.
    #[test]
    fn tensor_entry_canonical_fields_roundtrip() {
        let entry = TensorEntry {
            name: "x".into(),
            dtype: TensorDtype::BaseQ4,
            shape: vec![128, 64],
            offset: 0,
            length: 64,
            scale_offset: None,
            scale_length: None,
            bias_offset: None,
            bias_length: None,
            awq_scale_offset: None,
            awq_scale_length: None,
            group_size: Some(64),
            scale_dtype: Some(ScaleDtype::E8m0),
            symmetric: true,
            layout: None,
            residency: None,
            compute_region: ComputeRegion::Gpu,
            flags: TensorFlags::empty(),
            checksum_xxh64: None,
            source_ggml_type: None,
        };
        let s = serde_json::to_string(&entry).unwrap();
        let back: TensorEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back.scale_dtype, Some(ScaleDtype::E8m0));
        assert!(back.symmetric);
        assert_eq!(back.dtype, TensorDtype::BaseQ4);
    }
}
