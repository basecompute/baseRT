//! `basert overlay` / `basert merge` / `basert fakequant`: the autoquant probe
//! primitives.
//!
//! An overlay is a `.base` holding a SUBSET of a checkpoint's tensors — the
//! ones a spec names (plus whatever an AWQ fold modifies) — produced by the
//! very same per-tensor path `basert convert --profile` runs: `cmd_overlay`
//! builds the profile the spec implies and calls `convert_hf` with an
//! `OverlayPlan` in the `QuantContext`; `convert_generic` then skips every
//! tensor the plan does not keep and packs gptq entries from their codes
//! file. The bytes of an overlaid tensor are therefore byte-identical to the
//! full conversion's, and `basert merge` splices them into a base bundle
//! verbatim. `basert fakequant` writes the dequantized image of one tensor
//! (an overlay of one entry, read back) for parity tests.
//!
//! A k-quant spec entry (`dtype: "q4_k"` …) takes ggml super-blocks packed
//! elsewhere — llama.cpp's own row-major blocks, from a `blocks` safetensors
//! — and files them exactly as `convert --kquant-passthrough` files the same
//! ggml type (`kquant_passthrough_entry`: base_qN under `layout: gguf_super`,
//! `source_ggml_type`, the bytes verbatim), so the quant stage can put the
//! k-quant formats on the same menu as the affine ones — on a fused 3-D MoE
//! expert stack too, in the types the engine's expert GEMM dispatches (Q4_K,
//! Q5_K, IQ4_XS: `expert_stack_kquant`).
//!
//! A dense-values spec entry (`dtype: "f16" | "bf16"`, `values: <safetensors>`)
//! takes the tensor's halves as GIVEN — `<canonical>.values` — and files them
//! as a `convert --profile` f16 / bf16 rule files the same tensor
//! (`emit_values`: the bytes verbatim, no scales), so a format the engine has
//! no kernel for can be quantized and dequantized elsewhere (autoquant's
//! simulated formats — k-quants on MoE expert stacks) and probed, or built as
//! a stand-in, on the carry's own half kernels.

use super::{
    convert_hf, expert_stack_kquant, kquant_passthrough_dtype, kquant_passthrough_entry,
    QuantContext, TargetScheme,
};
use anyhow::{bail, Context, Result};
use base_format::{Layout, TensorDtype, TensorEntry, BLOB_ALIGNMENT, PREFIX_LEN};
use base_readers::gguf::{GgmlType, TensorInfo};
use base_readers::safetensors::SafetensorsFile;
use clap::Parser;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct OverlayArgs {
    /// FP HF checkpoint directory (the convert source).
    pub input: PathBuf,
    /// Overlay spec JSON: `{"tensors": {"<canonical>": {"dtype": "base_q4",
    /// "group_size": 64, "scale_dtype": "bf16", "symmetric": false,
    /// "algo": "rtn" | "awq" | "gptq", "awq_alpha": 0.5,
    /// "codes": "<codes.safetensors>"}}}`. Every field but `dtype` is
    /// optional; `codes` is required for (and only for) `gptq`. The codes
    /// safetensors holds `<canonical>.codes` (U8, [N, K]) plus
    /// `<canonical>.scales` / `.biases` (F32, [N, K/group_size]); for a
    /// fused 3-D MoE expert tensor [E, N, K] either the 3-D shapes
    /// ([E, N, K] / [E, N, K/gs]) or the flat [E·N, K] / [E·N, K/gs]
    /// (expert e = rows [e·N, (e+1)·N), the layout convert packs).
    ///
    /// The k-quant form: `{"dtype": "q4_k", "blocks": "<blocks.safetensors>",
    /// "algo": "rtn" | "gptq"}` with `dtype` one of q2_k, q3_k, q4_k, q5_k,
    /// q6_k, iq4_xs. The blocks safetensors holds `<canonical>.blocks` (U8,
    /// [N, (K/256)·block_bytes]; for a fused 3-D MoE expert tensor [E, N, K]
    /// — q4_k, q5_k and iq4_xs only, the types the expert GEMM dispatches — either
    /// [E, N, (K/256)·block_bytes] or the flat [E·N, (K/256)·block_bytes]):
    /// ggml's row-major super-blocks exactly as llama.cpp stores them (K a
    /// multiple of 256; 84 / 110 / 144 / 176 / 210 / 136 bytes per block). `algo` is only recorded (default rtn);
    /// `group_size`, `scale_dtype`, `symmetric`, `codes` and `awq_alpha`
    /// are errors with this form. The entry is filed as `convert
    /// --kquant-passthrough` files the same ggml type.
    ///
    /// The dense-values form: `{"dtype": "f16" | "bf16", "values":
    /// "<values.safetensors>", "algo": "rtn" | "gptq"}`. The values
    /// safetensors holds `<canonical>.values` (F16 / BF16 matching `dtype`)
    /// in the tensor's source shape — [N, K], or a fused 3-D MoE expert
    /// tensor as [E, N, K] or flat [E·N, K] — e.g. the dequantized image of
    /// a format the engine has no kernel for. `algo` is only recorded
    /// (default rtn); `group_size`, `scale_dtype`, `symmetric`, `codes`,
    /// `awq_alpha` and `blocks` are errors with this form. The entry is
    /// filed as a `--profile` f16 / bf16 rule files the same tensor, the
    /// bytes verbatim.
    #[arg(long)]
    pub spec: PathBuf,
    /// Output overlay `.base`.
    #[arg(short = 'o', long)]
    pub output: PathBuf,
    /// AWQ activation sidecar (what `convert --awq-profile` takes).
    /// Required when any spec entry says `algo: "awq"`.
    #[arg(long)]
    pub awq_profile: Option<PathBuf>,
    /// AWQ alpha search: score an evenly strided subset of N rows per
    /// member (0 = all rows). Same value as the eventual `convert`.
    #[arg(long, default_value_t = 0)]
    pub awq_search_rows: usize,
    /// Do not refuse a sidecar whose fingerprint mismatches the checkpoint.
    #[arg(long)]
    pub no_fingerprint_check: bool,
}

#[derive(Parser, Debug)]
pub struct MergeArgs {
    /// Base `.base` bundle: header, slots, mmproj, speculator and every
    /// tensor no overlay replaces come from here.
    pub base: PathBuf,
    /// Overlay `.base` bundles (from `basert overlay`). Later overlays win
    /// when several carry the same tensor.
    #[arg(required = true)]
    pub overlays: Vec<PathBuf>,
    /// Output `.base`.
    #[arg(short = 'o', long)]
    pub output: PathBuf,
    /// Write the language model only: leave out the base's mmproj
    /// (vision / audio towers) and speculator (MTP) sections. The
    /// provenance keeps the base's record and notes the drop.
    #[arg(long)]
    pub lm_only: bool,
}

#[derive(Parser, Debug)]
pub struct FakequantArgs {
    /// FP HF checkpoint directory (the convert source). Not needed with
    /// `--bundle`.
    #[arg(required_unless_present = "bundle")]
    pub input: Option<PathBuf>,
    /// Canonical tensor name (see `basert inventory`).
    #[arg(long)]
    pub tensor: String,
    /// Quant dtype: `base_q2` .. `base_q8` (or `f16` / `bf16` for a plain
    /// half round-trip). Not needed with `--bundle`.
    #[arg(long, required_unless_present = "bundle")]
    pub dtype: Option<String>,
    /// Dequantize `--tensor` straight out of this existing `.base` (an
    /// overlay, a merged bundle, a `--kquant-passthrough` conversion …)
    /// instead of quantizing it from the checkpoint: the image the runtime
    /// would see, for any dtype `basert` can decode (base_qN, the GGUF
    /// super-block formats, f16 / bf16 / f32).
    #[arg(long, conflicts_with_all = ["group_size", "scale_dtype", "symmetric"])]
    pub bundle: Option<PathBuf>,
    /// Group size (default: the dtype's canonical group size).
    #[arg(long)]
    pub group_size: Option<u32>,
    /// Scale dtype: `bf16` | `f16` | `e8m0` | `e4m3`.
    #[arg(long, default_value = "bf16")]
    pub scale_dtype: String,
    /// Symmetric (no-bias) grid instead of affine.
    #[arg(long)]
    pub symmetric: bool,
    /// Output file: little-endian f32, row-major, no header.
    #[arg(long)]
    pub out: PathBuf,
}

pub(crate) fn is_canonical_quant(dtype: TensorDtype) -> bool {
    matches!(
        dtype,
        TensorDtype::BaseQ2
            | TensorDtype::BaseQ3
            | TensorDtype::BaseQ4
            | TensorDtype::BaseQ5
            | TensorDtype::BaseQ6
            | TensorDtype::BaseQ8
    )
}

fn target_for(dtype: TensorDtype) -> TargetScheme {
    match dtype {
        TensorDtype::BaseQ2 => TargetScheme::BaseQ2,
        TensorDtype::BaseQ3 => TargetScheme::BaseQ3,
        TensorDtype::BaseQ4 => TargetScheme::BaseQ4,
        TensorDtype::BaseQ5 => TargetScheme::BaseQ5,
        TensorDtype::BaseQ6 => TargetScheme::BaseQ6,
        TensorDtype::BaseQ8 => TargetScheme::BaseQ8,
        TensorDtype::Mxfp4 => TargetScheme::Mxfp4,
        TensorDtype::Nvfp4 => TargetScheme::Nvfp4,
        // carry dtypes: the bf16 carry scheme
        _ => TargetScheme::Bf16,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OverlayAlgo {
    Rtn,
    Awq,
    /// Codes searched elsewhere (GPTQ or any other off-line grid), packed
    /// verbatim via `base_quant::pack_codes`.
    Gptq,
}

/// The k-quant form of a spec entry: the ggml super-block type and the
/// safetensors holding its packed blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KquantBlocks {
    pub ggml_type: GgmlType,
    pub blocks: PathBuf,
}

/// The ggml super-block types an overlay spec can name, by their spec name.
/// The GEMV / GEMM / embedding kernels of exactly these exist in the Metal
/// backend (`dtype_descriptor.h`); the grid i-quants need a search the
/// quant stage does not run, so they are not on this menu.
const KQUANT_TYPES: [(&str, GgmlType); 6] = [
    ("q2_k", GgmlType::Q2K),
    ("q3_k", GgmlType::Q3K),
    ("q4_k", GgmlType::Q4K),
    ("q5_k", GgmlType::Q5K),
    ("q6_k", GgmlType::Q6K),
    ("iq4_xs", GgmlType::Iq4Xs),
];

pub(crate) fn kquant_type_from_name(name: &str) -> Option<GgmlType> {
    KQUANT_TYPES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| *t)
}

pub(crate) fn kquant_type_name(ty: GgmlType) -> &'static str {
    KQUANT_TYPES
        .iter()
        .find(|(_, t)| *t == ty)
        .map(|(n, _)| *n)
        .expect("a type from KQUANT_TYPES")
}

/// One `tensors` entry of an overlay spec (schema in `OverlayArgs::spec`).
#[derive(Clone, Debug)]
pub(crate) struct OverlaySpecEntry {
    pub name: String,
    /// The bundle dtype. For the k-quant form, the base_qN the passthrough
    /// files the ggml type under (`kquant_passthrough_dtype`).
    pub dtype: TensorDtype,
    pub group_size: Option<u32>,
    pub scale_dtype: Option<base_format::ScaleDtype>,
    pub symmetric: Option<bool>,
    pub algo: OverlayAlgo,
    pub awq_alpha: Option<f32>,
    pub codes: Option<PathBuf>,
    /// Set for the k-quant form; the affine fields above are then None.
    pub kquant: Option<KquantBlocks>,
    /// The values safetensors of the dense-values form (`dtype` is then f16
    /// or bf16; the affine fields above are None).
    pub values: Option<PathBuf>,
}

impl OverlaySpecEntry {
    /// Whether the entry's bytes come from a producer file (gptq codes,
    /// k-quant blocks or given values) rather than from `pack_tensor`.
    pub fn is_prepacked(&self) -> bool {
        self.algo == OverlayAlgo::Gptq || self.kquant.is_some() || self.values.is_some()
    }
}

pub(crate) fn parse_overlay_spec(spec: &serde_json::Value) -> Result<Vec<OverlaySpecEntry>> {
    const KEYS: [&str; 9] = [
        "dtype",
        "group_size",
        "scale_dtype",
        "symmetric",
        "algo",
        "awq_alpha",
        "codes",
        "blocks",
        "values",
    ];
    // The affine keys the k-quant form has no use for: a super-block carries
    // its scales and mins inside, and its bytes are never searched here.
    const AFFINE_ONLY: [&str; 5] = [
        "group_size",
        "scale_dtype",
        "symmetric",
        "codes",
        "awq_alpha",
    ];
    let tensors = spec
        .get("tensors")
        .and_then(|t| t.as_object())
        .ok_or_else(|| {
            anyhow::anyhow!("spec: expected {{\"tensors\": {{<canonical>: {{\"dtype\": ...}}}}}}")
        })?;
    if tensors.is_empty() {
        bail!("spec names no tensor");
    }
    let mut out = Vec::with_capacity(tensors.len());
    for (name, v) in tensors {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("spec: {name}: expected an object"))?;
        for k in obj.keys() {
            if !KEYS.contains(&k.as_str()) {
                bail!("spec: {name}: unknown key {k:?} (expected one of {KEYS:?})");
            }
        }
        let dtype_json = obj
            .get("dtype")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("spec: {name}: missing \"dtype\""))?;
        if let Some(ggml_type) = dtype_json.as_str().and_then(kquant_type_from_name) {
            let dtype_name = kquant_type_name(ggml_type);
            if let Some(k) = AFFINE_ONLY.iter().find(|k| obj.contains_key(**k)) {
                bail!(
                    "spec: {name}: {k:?} is not valid with the k-quant dtype {dtype_name:?} \
                     (its super-blocks carry their scales inside; give \"blocks\" and optionally \"algo\")"
                );
            }
            if obj.contains_key("values") {
                bail!(
                    "spec: {name}: \"values\" is not valid with the k-quant dtype {dtype_name:?} \
                     (give \"blocks\"; dense values go with dtype \"f16\" / \"bf16\")"
                );
            }
            let blocks = obj
                .get("blocks")
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "spec: {name}: the k-quant dtype {dtype_name:?} needs \"blocks\": <safetensors path>"
                    )
                })?
                .as_str()
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("spec: {name}: blocks must be a path string"))?;
            let algo = match obj.get("algo").map(|a| a.as_str()) {
                None | Some(Some("rtn")) => OverlayAlgo::Rtn,
                Some(Some("gptq")) => OverlayAlgo::Gptq,
                Some(other) => bail!(
                    "spec: {name}: algo {other:?} with the k-quant dtype {dtype_name:?} \
                     (expected \"rtn\" or \"gptq\"; the blocks are taken as given)"
                ),
            };
            let (dtype, _) =
                kquant_passthrough_dtype(ggml_type).expect("KQUANT_TYPES pass through");
            out.push(OverlaySpecEntry {
                name: name.clone(),
                dtype,
                group_size: None,
                scale_dtype: None,
                symmetric: None,
                algo,
                awq_alpha: None,
                codes: None,
                kquant: Some(KquantBlocks { ggml_type, blocks }),
                values: None,
            });
            continue;
        }
        if obj.contains_key("blocks") {
            bail!(
                "spec: {name}: \"blocks\" is only valid with a k-quant dtype ({})",
                KQUANT_TYPES
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let dtype: TensorDtype =
            serde_json::from_value(dtype_json).with_context(|| format!("spec: {name}: dtype"))?;
        if let Some(values) = obj.get("values") {
            if !matches!(dtype, TensorDtype::F16 | TensorDtype::Bf16) {
                bail!(
                    "spec: {name}: \"values\" is only valid with dtype \"f16\" or \"bf16\" \
                     (the dense image of a format quantized elsewhere), got {dtype:?}"
                );
            }
            if let Some(k) = AFFINE_ONLY.iter().find(|k| obj.contains_key(**k)) {
                bail!(
                    "spec: {name}: {k:?} is not valid with \"values\" \
                     (the halves are filed as given; give \"dtype\" f16 / bf16 and optionally \"algo\")"
                );
            }
            let values = values
                .as_str()
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("spec: {name}: values must be a path string"))?;
            let algo = match obj.get("algo").map(|a| a.as_str()) {
                None | Some(Some("rtn")) => OverlayAlgo::Rtn,
                Some(Some("gptq")) => OverlayAlgo::Gptq,
                Some(other) => bail!(
                    "spec: {name}: algo {other:?} with \"values\" \
                     (expected \"rtn\" or \"gptq\"; the values are taken as given)"
                ),
            };
            out.push(OverlaySpecEntry {
                name: name.clone(),
                dtype,
                group_size: None,
                scale_dtype: None,
                symmetric: None,
                algo,
                awq_alpha: None,
                codes: None,
                kquant: None,
                values: Some(values),
            });
            continue;
        }
        let group_size = match obj.get("group_size") {
            None => None,
            Some(g) => Some(
                g.as_u64()
                    .filter(|&g| g > 0 && g <= u32::MAX as u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!("spec: {name}: group_size must be a positive integer")
                    })? as u32,
            ),
        };
        let scale_dtype = obj
            .get("scale_dtype")
            .map(|s| serde_json::from_value::<base_format::ScaleDtype>(s.clone()))
            .transpose()
            .with_context(|| format!("spec: {name}: scale_dtype"))?;
        let symmetric = obj
            .get("symmetric")
            .map(|b| {
                b.as_bool()
                    .ok_or_else(|| anyhow::anyhow!("spec: {name}: symmetric must be a bool"))
            })
            .transpose()?;
        let algo = match obj.get("algo").map(|a| a.as_str()) {
            None | Some(Some("rtn")) => OverlayAlgo::Rtn,
            Some(Some("awq")) => OverlayAlgo::Awq,
            Some(Some("gptq")) => OverlayAlgo::Gptq,
            Some(other) => {
                bail!("spec: {name}: algo {other:?} (expected \"rtn\", \"awq\" or \"gptq\")")
            }
        };
        let awq_alpha = obj
            .get("awq_alpha")
            .map(|a| {
                a.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| anyhow::anyhow!("spec: {name}: awq_alpha must be a number"))
            })
            .transpose()?;
        let codes = obj
            .get("codes")
            .map(|c| {
                c.as_str()
                    .map(PathBuf::from)
                    .ok_or_else(|| anyhow::anyhow!("spec: {name}: codes must be a path string"))
            })
            .transpose()?;
        if algo != OverlayAlgo::Rtn && !is_canonical_quant(dtype) {
            bail!("spec: {name}: algo {algo:?} needs a base_qN dtype, got {dtype:?}");
        }
        if awq_alpha.is_some() && algo != OverlayAlgo::Awq {
            bail!("spec: {name}: awq_alpha given but algo is not \"awq\"");
        }
        match (algo, &codes) {
            (OverlayAlgo::Gptq, None) => {
                bail!("spec: {name}: algo \"gptq\" needs \"codes\": <safetensors path>")
            }
            (OverlayAlgo::Gptq, Some(_)) if symmetric == Some(true) => {
                bail!("spec: {name}: gptq codes are affine (scales + biases); symmetric is not supported")
            }
            (a, Some(_)) if a != OverlayAlgo::Gptq => {
                bail!("spec: {name}: \"codes\" is only valid with algo \"gptq\"")
            }
            _ => {}
        }
        out.push(OverlaySpecEntry {
            name: name.clone(),
            dtype,
            group_size,
            scale_dtype,
            symmetric,
            algo,
            awq_alpha,
            codes,
            kquant: None,
            values: None,
        });
    }
    Ok(out)
}

/// File a k-quant spec entry from its blocks file: the super-blocks verbatim,
/// the entry exactly what `convert --kquant-passthrough` writes for the same
/// ggml type (`kquant_passthrough_entry`), plus `algo` for the record.
pub(crate) fn emit_kquant_blocks(
    e: &OverlaySpecEntry,
    canonical: &str,
    shape: Vec<u64>,
    st: &SafetensorsFile,
) -> Result<(TensorEntry, Vec<u8>)> {
    use base_readers::safetensors::StDtype;
    let kq = e.kquant.as_ref().expect("validated: a k-quant entry");
    let dtype_name = kquant_type_name(kq.ggml_type);
    // 2-D [N, K], or a fused 3-D MoE expert stack [E, N, K] in the types the
    // expert GEMM dispatches (`expert_stack_kquant`: Q4_K, Q5_K, IQ4_XS — the
    // engine's `dt_is_moe_kquant`; `kquant_passes_through` admits expert
    // stacks on the passthrough by the same list). The blocks of a stack are
    // the experts' row-major super-blocks laid end to end, expert by expert
    // — what autoquant packs — given as [E, N, cols] or flat [E·N, cols].
    if shape.len() != 2 && shape.len() != 3 {
        bail!("{canonical}: {dtype_name} blocks need a 2-D or 3-D (fused expert) tensor, source shape is {shape:?}");
    }
    if shape.len() == 3 && !expert_stack_kquant(kq.ggml_type) {
        bail!(
            "{canonical}: {dtype_name} blocks on a 3-D expert stack {shape:?}: the expert GEMM dispatches \
             Q4_K, Q5_K and IQ4_XS super-blocks only"
        );
    }
    let k = shape[shape.len() - 1];
    let rows: u64 = shape[..shape.len() - 1].iter().product();
    if k % 256 != 0 {
        bail!("{canonical}: in_features={k} is not a multiple of the 256-element super-block");
    }
    let (_, block_bytes) = kq.ggml_type.block_geometry();
    let cols = (k / 256) * block_bytes as u64;
    let want_flat = vec![rows, cols];
    let mut want_nd: Vec<u64> = shape[..shape.len() - 1].to_vec();
    want_nd.push(cols);
    let want_name = format!("{canonical}.blocks");
    let info = st
        .tensors
        .iter()
        .find(|t| t.name == want_name)
        .ok_or_else(|| anyhow::anyhow!("blocks file has no tensor {want_name:?}"))?;
    if info.dtype != StDtype::U8 {
        bail!("{want_name} must be U8, got {:?}", info.dtype);
    }
    if info.shape != want_flat && info.shape != want_nd {
        bail!(
            "{want_name} is shaped {:?}; {dtype_name} blocks for the source shape {shape:?} are [rows, (K/256)*{block_bytes}] = {want_flat:?} (or {want_nd:?})",
            info.shape
        );
    }
    let bytes = st.tensor_bytes(info);
    if bytes.len() as u64 != rows * cols {
        bail!(
            "{want_name} holds {} bytes, its shape {:?} says {}",
            bytes.len(),
            info.shape,
            rows * cols
        );
    }
    // TensorInfo is in GGUF ne order (fastest axis first): the declared
    // [N, K] / [E, N, K] reversed, which kquant_passthrough_entry reverses
    // back (an expert stack stays [E, N, K]).
    let src = TensorInfo {
        name: canonical.to_string(),
        shape: shape.iter().rev().copied().collect(),
        ggml_type: kq.ggml_type,
        data_offset: 0,
    };
    let (mut entry, data) = kquant_passthrough_entry(&src, bytes, canonical.to_string());
    entry.algo = Some(
        match e.algo {
            OverlayAlgo::Gptq => "gptq",
            _ => "rtn",
        }
        .to_string(),
    );
    Ok((entry, data))
}

/// File a dense-values spec entry from its values file: the halves verbatim,
/// the entry what `convert --profile` writes for an f16 / bf16 rule on the
/// same tensor (`pack_tensor`'s half branch: no scales, no group size, warm
/// on the accelerator — hot on the GPU for an embedding), plus `algo` for
/// the record. The source shape is kept as the entry's shape (a fused 3-D
/// expert stack stays [E, N, K], its halves flat row-major).
pub(crate) fn emit_values(
    e: &OverlaySpecEntry,
    canonical: &str,
    shape: Vec<u64>,
    st: &SafetensorsFile,
) -> Result<(TensorEntry, Vec<u8>)> {
    use base_format::{ComputeRegion, ResidencyHint, TensorFlags};
    use base_readers::safetensors::StDtype;
    debug_assert!(e.values.is_some(), "validated: a values entry");
    if shape.len() != 2 && shape.len() != 3 {
        bail!("{canonical}: dense values need a 2-D or 3-D (fused expert) tensor, source shape is {shape:?}");
    }
    let want_name = format!("{canonical}.values");
    let info = st
        .tensors
        .iter()
        .find(|t| t.name == want_name)
        .ok_or_else(|| anyhow::anyhow!("values file has no tensor {want_name:?}"))?;
    let want_dtype = match e.dtype {
        TensorDtype::F16 => StDtype::F16,
        TensorDtype::Bf16 => StDtype::Bf16,
        other => unreachable!("validated: values dtype is f16 / bf16, got {other:?}"),
    };
    if info.dtype != want_dtype {
        bail!(
            "{want_name} must be {want_dtype:?} (the spec says dtype {:?}), got {:?}",
            e.dtype,
            info.dtype
        );
    }
    let flat_rows: u64 = shape[..shape.len() - 1].iter().product();
    let flat = vec![flat_rows, *shape.last().unwrap()];
    if info.shape != shape && info.shape != flat {
        bail!(
            "{want_name} is shaped {:?}; the source tensor is {shape:?} (or flat {flat:?})",
            info.shape
        );
    }
    let bytes = st.tensor_bytes(info);
    let n_elems: u64 = shape.iter().product();
    if bytes.len() as u64 != n_elems * 2 {
        bail!(
            "{want_name} holds {} bytes, its shape {shape:?} says {}",
            bytes.len(),
            n_elems * 2
        );
    }
    let is_embedding = canonical == "embed_tokens.weight" || canonical == "lm_head.weight";
    let entry = TensorEntry {
        name: canonical.to_string(),
        dtype: e.dtype,
        shape,
        offset: 0,
        length: bytes.len() as u64,
        scale_offset: None,
        scale_length: None,
        bias_offset: None,
        bias_length: None,
        awq_scale_offset: None,
        awq_scale_length: None,
        group_size: None,
        layout: None,
        residency: Some(if is_embedding {
            ResidencyHint::Hot
        } else {
            ResidencyHint::Warm
        }),
        compute_region: if is_embedding {
            ComputeRegion::Gpu
        } else {
            ComputeRegion::Accelerator
        },
        scale_dtype: None,
        symmetric: false,
        flags: TensorFlags::empty(),
        checksum_xxh64: None,
        source_ggml_type: None,
        algo: Some(
            match e.algo {
                OverlayAlgo::Gptq => "gptq",
                _ => "rtn",
            }
            .to_string(),
        ),
        awq_alpha: None,
    };
    Ok((entry, bytes.to_vec()))
}

/// Pack a gptq spec entry from its codes file. Produces the SAME on-disk
/// layout as `convert_generic`'s quantized branch ([weights | scales |
/// biases], per-group scale dtype), so the runtime cannot tell the tensor
/// from an RTN one; only `entry.algo = "gptq"` records where the codes
/// came from.
pub(crate) fn emit_gptq_codes(
    e: &OverlaySpecEntry,
    canonical: &str,
    shape: Vec<u64>,
    st: &SafetensorsFile,
) -> Result<(TensorEntry, Vec<u8>)> {
    use base_format::{ComputeRegion, ResidencyHint, ScaleDtype, TensorFlags};
    use base_readers::safetensors::StDtype;
    let find = |suffix: &str| {
        let want = format!("{canonical}.{suffix}");
        st.tensors
            .iter()
            .find(|t| t.name == want)
            .ok_or_else(|| anyhow::anyhow!("codes file has no tensor {want:?}"))
    };
    let codes_info = find("codes")?;
    let scales_info = find("scales")?;
    let biases_info = find("biases")?;
    if codes_info.dtype != StDtype::U8 {
        bail!("{canonical}.codes must be U8, got {:?}", codes_info.dtype);
    }
    if scales_info.dtype != StDtype::F32 || biases_info.dtype != StDtype::F32 {
        bail!(
            "{canonical}.scales/.biases must be F32, got {:?}/{:?}",
            scales_info.dtype,
            biases_info.dtype
        );
    }
    // 2-D [N, K], or a fused 3-D MoE expert tensor [E, N, K] packed as
    // its flat [E·N, K] image (expert e = rows [e·N, (e+1)·N)) — the same
    // layout convert's RTN packer produces. Codes may come as [E, N, K]
    // or [E·N, K]; scales/biases as [E, N, K/gs] or [E·N, K/gs].
    if shape.len() != 2 && shape.len() != 3 {
        bail!("{canonical}: gptq codes need a 2-D or 3-D (fused expert) tensor, source shape is {shape:?}");
    }
    let flat_rows: u64 = shape[..shape.len() - 1].iter().product();
    let k_dim = *shape.last().unwrap();
    let flat = vec![flat_rows, k_dim];
    if codes_info.shape != shape && codes_info.shape != flat {
        bail!(
            "{canonical}.codes shape {:?} does not match the source tensor shape {:?} (or its flat {:?})",
            codes_info.shape,
            shape,
            flat
        );
    }
    let bits = e
        .dtype
        .bits_per_weight()
        .expect("validated: canonical quant dtype");
    let gs = e
        .group_size
        .or_else(|| e.dtype.default_group_size())
        .expect("canonical quant dtypes have a default group size") as usize;
    let scale_dtype = e.scale_dtype.unwrap_or(ScaleDtype::Bf16);
    let (n, k) = (flat_rows as usize, k_dim as usize);
    if k % gs != 0 {
        bail!("{canonical}: in_features={k} is not a multiple of group_size={gs}");
    }
    let want = vec![n as u64, (k / gs) as u64];
    let mut want_nd: Vec<u64> = shape[..shape.len() - 1].to_vec();
    want_nd.push((k / gs) as u64);
    let ok = |s: &Vec<u64>| *s == want || *s == want_nd;
    if !ok(&scales_info.shape) || !ok(&biases_info.shape) {
        bail!(
            "{canonical}.scales/.biases must be shaped {want_nd:?} or {want:?} ([.., K/group_size]), got {:?}/{:?}",
            scales_info.shape,
            biases_info.shape
        );
    }
    let codes = st.tensor_bytes(codes_info);
    let q_max = (1u32 << bits) - 1;
    if let Some(bad) = codes.iter().find(|&&c| c as u32 > q_max) {
        bail!(
            "{canonical}.codes holds value {bad}, out of range for {bits}-bit codes (max {q_max})"
        );
    }
    let to_f32 = |info: &base_readers::safetensors::StTensorInfo| -> Vec<f32> {
        st.tensor_bytes(info)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let scales = to_f32(scales_info);
    let biases = to_f32(biases_info);
    let cfg = base_quant::RtnConfig {
        bits,
        group_size: gs as u32,
        symmetric: false,
        scale_dtype,
    };
    let packed = base_quant::pack_codes(codes, &scales, &biases, cfg);

    let mut data =
        Vec::with_capacity(packed.packed_weights.len() + packed.scales.len() + packed.biases.len());
    data.extend_from_slice(&packed.packed_weights);
    let scale_off = data.len() as u64;
    data.extend_from_slice(&packed.scales);
    let bias_off = data.len() as u64;
    data.extend_from_slice(&packed.biases);
    let mut entry = TensorEntry {
        name: canonical.to_string(),
        dtype: e.dtype,
        shape,
        offset: 0,
        length: 0,
        scale_offset: Some(scale_off),
        scale_length: Some(packed.scales.len() as u64),
        bias_offset: Some(bias_off),
        bias_length: Some(packed.biases.len() as u64),
        awq_scale_offset: None,
        awq_scale_length: None,
        group_size: Some(cfg.group_size),
        layout: None,
        residency: Some(ResidencyHint::Warm),
        compute_region: ComputeRegion::Accelerator,
        scale_dtype: Some(scale_dtype),
        symmetric: false,
        flags: TensorFlags::empty(),
        checksum_xxh64: None,
        source_ggml_type: None,
        algo: Some("gptq".to_string()),
        awq_alpha: None,
    };
    entry.length = data.len() as u64;
    Ok((entry, data))
}

/// What `convert_generic` consults while it runs for `basert overlay`.
pub(crate) struct OverlayPlan {
    entries: HashMap<String, OverlaySpecEntry>,
    spec_json: serde_json::Value,
    /// The producer files (gptq codes, k-quant blocks, given values), each
    /// opened once.
    codes_files: RefCell<HashMap<PathBuf, SafetensorsFile>>,
}

impl OverlayPlan {
    pub fn entry(&self, canonical: &str) -> Option<&OverlaySpecEntry> {
        self.entries.get(canonical)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }
    pub fn spec_json(&self) -> &serde_json::Value {
        &self.spec_json
    }
    /// A prepacked entry's bytes from its producer file: gptq codes packed
    /// through `pack_codes`, k-quant blocks or given values verbatim (the
    /// file is opened once).
    pub fn prepacked_payload(
        &self,
        e: &OverlaySpecEntry,
        canonical: &str,
        shape: Vec<u64>,
    ) -> Result<(TensorEntry, Vec<u8>)> {
        let (path, what) = match (&e.kquant, &e.values) {
            (Some(kq), _) => (&kq.blocks, "blocks"),
            (None, Some(v)) => (v, "values"),
            (None, None) => (
                e.codes.as_ref().expect("validated: gptq has codes"),
                "codes",
            ),
        };
        let mut files = self.codes_files.borrow_mut();
        if !files.contains_key(path) {
            let st = SafetensorsFile::open(path)
                .with_context(|| format!("opening {what} file {}", path.display()))?;
            files.insert(path.clone(), st);
        }
        if e.kquant.is_some() {
            emit_kquant_blocks(e, canonical, shape, &files[path])
                .with_context(|| format!("k-quant blocks for {canonical}"))
        } else if e.values.is_some() {
            emit_values(e, canonical, shape, &files[path])
                .with_context(|| format!("dense values for {canonical}"))
        } else {
            emit_gptq_codes(e, canonical, shape, &files[path])
                .with_context(|| format!("gptq codes for {canonical}"))
        }
    }
}

/// The `QuantContext` an overlay conversion runs with: the profile the
/// spec implies (one exact-pattern rule per tensor, catch-all f16 for the
/// tensors a fold drags in) and the plan `convert_generic` filters by.
fn overlay_context(
    entries: Vec<OverlaySpecEntry>,
    spec_json: serde_json::Value,
    awq_profile: Option<PathBuf>,
    awq_search_rows: usize,
    skip_fingerprint: bool,
) -> Result<QuantContext> {
    // gptq and k-quant entries get an rtn rule under their bundle dtype (the
    // profile validator knows neither) — their bytes come from the codes /
    // blocks file, never from `pack_tensor`.
    let mut rules: Vec<serde_json::Value> = Vec::with_capacity(entries.len() + 1);
    for e in &entries {
        let mut r = serde_json::json!({"pattern": e.name, "dtype": e.dtype});
        if let Some(g) = e.group_size {
            r["group_size"] = serde_json::json!(g);
        }
        if let Some(sd) = e.scale_dtype {
            r["scale_dtype"] = serde_json::json!(sd);
        }
        if let Some(sym) = e.symmetric {
            r["symmetric"] = serde_json::json!(sym);
        }
        r["algo"] = serde_json::json!(if e.algo == OverlayAlgo::Awq {
            "awq"
        } else {
            "rtn"
        });
        if let Some(a) = e.awq_alpha {
            r["awq_alpha"] = serde_json::json!(a);
        }
        rules.push(r);
    }
    rules.push(serde_json::json!({"pattern": "**", "dtype": "f16"}));
    let profile_json = serde_json::json!({"name": "overlay", "arch": "*", "rules": rules});
    let profile = base_quant::QuantProfile::from_json(profile_json.to_string().as_bytes())
        .context("building the overlay profile from the spec")?;

    let wants_awq = entries.iter().any(|e| e.algo == OverlayAlgo::Awq);
    let awq_profile = match &awq_profile {
        Some(p) => Some(
            base_awq::AwqProfile::load(p)
                .with_context(|| format!("loading AWQ sidecar {}", p.display()))?,
        ),
        None if wants_awq => {
            bail!("the spec asks for algo \"awq\" but no --awq-profile sidecar was given")
        }
        None => None,
    };
    // The header's quant_scheme: the first quantized entry's (an all-carry
    // spec is a bf16-scheme bundle, like a carry).
    let target = entries
        .iter()
        .find(|e| is_canonical_quant(e.dtype))
        .map(|e| target_for(e.dtype))
        .unwrap_or(TargetScheme::Bf16);
    let plan = OverlayPlan {
        entries: entries.into_iter().map(|e| (e.name.clone(), e)).collect(),
        spec_json,
        codes_files: RefCell::new(HashMap::new()),
    };
    Ok(QuantContext {
        profile: Some(profile),
        awq_profile,
        awq_applied: std::sync::atomic::AtomicUsize::new(0),
        awq_shape_skipped: std::sync::atomic::AtomicUsize::new(0),
        awq_config: base_awq::AwqConfig {
            search_rows: awq_search_rows,
            ..Default::default()
        },
        awq_skip_fingerprint: skip_fingerprint,
        target,
        allow_quant_from_quant: false,
        // An overlay packs from the FP checkpoint exactly like
        // `convert --profile`: no passthrough, no validation pass, no
        // direct write, no imatrix, no sidecar / tied-head extras.
        q4_passthrough: false,
        quantize_drafter: false,
        keep_drafter_fc_f16: false,
        speculator_lm_head: false,
        mtp_draft_ranking: None,
        mtp_draft_vocab: None,
        kquant_passthrough: false,
        synth_tied_head: false,
        mlx_passthrough: false,
        validate: false,
        direct_write: false,
        imatrix: false,
        overlay: Some(plan),
    })
}

/// The tensors `convert_generic` keeps under an overlay plan: the spec's,
/// plus every tensor the AWQ fold modified (carried through the catch-all
/// so the bundle stays exact), minus nothing else. Also the plan's
/// consistency checks against the fold plan and the checkpoint's names.
pub(crate) fn overlay_keep_set(
    plan: &OverlayPlan,
    mapped: &[(String, String)],
    fold: &super::FoldPlan,
) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    let known: std::collections::HashSet<&str> = mapped.iter().map(|(_, c)| c.as_str()).collect();
    let unknown: Vec<&str> = plan.names().filter(|n| !known.contains(n)).collect();
    if !unknown.is_empty() {
        bail!(
            "spec names tensors this checkpoint does not have (see `basert inventory`): {}",
            unknown.join(", ")
        );
    }
    for n in plan.names() {
        let e = plan.entry(n).unwrap();
        if e.is_prepacked() && (fold.col_scale.contains_key(n) || fold.row_scale.contains_key(n)) {
            bail!(
                "{n}: {} cannot be combined with an AWQ fold that rescales this tensor \
                 (they were packed from the unscaled weights)",
                if e.kquant.is_some() {
                    "k-quant blocks"
                } else if e.values.is_some() {
                    "given values"
                } else {
                    "gptq codes"
                }
            );
        }
    }
    let fold_modified: BTreeSet<String> = fold
        .gamma_scale
        .keys()
        .chain(fold.row_scale.keys())
        .chain(fold.col_scale.keys())
        .filter(|k| plan.entry(k.as_str()).is_none())
        .cloned()
        .collect();
    let mut keep = fold_modified.clone();
    keep.extend(plan.names().map(|s| s.to_string()));
    Ok((keep, fold_modified))
}

pub fn cmd_overlay(args: OverlayArgs) -> Result<()> {
    let t0 = std::time::Instant::now();
    let spec_text = std::fs::read_to_string(&args.spec)
        .with_context(|| format!("reading spec {}", args.spec.display()))?;
    let spec_json: serde_json::Value = serde_json::from_str(&spec_text)
        .with_context(|| format!("parsing spec {}", args.spec.display()))?;
    let entries = parse_overlay_spec(&spec_json)?;
    let n_entries = entries.len();
    let ctx = overlay_context(
        entries,
        spec_json,
        args.awq_profile.clone(),
        args.awq_search_rows,
        args.no_fingerprint_check,
    )?;
    convert_hf(&args.input, &args.output, &ctx)?;
    eprintln!(
        "overlay: {n_entries} spec tensors -> {} ({:.1}s)",
        args.output.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Whether `a` and `b` name the same file (symlinks followed; hard links
/// compare equal). A path that does not exist names no file.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    let (Ok(ma), Ok(mb)) = (std::fs::metadata(a), std::fs::metadata(b)) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ma.dev() == mb.dev() && ma.ino() == mb.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = (ma, mb);
        match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
            (Ok(x), Ok(y)) => x == y,
            _ => false,
        }
    }
}

/// Refuse an output that is one of the inputs. The inputs are mmapped for
/// the whole run, and creating the output truncates it: `merge base.base
/// ov.base -o base.base` would destroy the base before its tensors are
/// read (a SIGBUS on the mapping at best, a silently empty source at
/// worst). Checked by file identity, so a symlink or hard link to an input
/// is refused too.
fn refuse_output_aliasing_inputs(
    what: &str,
    output: &std::path::Path,
    inputs: &[&std::path::Path],
) -> Result<()> {
    for input in inputs {
        if same_file(output, input) {
            bail!(
                "{what}: the output {} is the same file as the input {} — write to another path",
                output.display(),
                input.display()
            );
        }
    }
    Ok(())
}

/// A temporary sibling of `output` (same directory, so the final rename
/// stays on one filesystem). The output is only ever replaced whole, by the
/// rename after a successful write; a failure leaves whatever was there.
fn temp_sibling(output: &std::path::Path) -> std::path::PathBuf {
    let mut s = output.as_os_str().to_os_string();
    s.push(format!(".tmp-{}", std::process::id()));
    std::path::PathBuf::from(s)
}

pub fn cmd_merge(args: MergeArgs) -> Result<()> {
    use base_format::{BaseReader, BaseWriter, TensorPayload};
    let t0 = std::time::Instant::now();

    let inputs: Vec<&std::path::Path> = std::iter::once(args.base.as_path())
        .chain(args.overlays.iter().map(|p| p.as_path()))
        .collect();
    refuse_output_aliasing_inputs("merge", &args.output, &inputs)?;

    let base = BaseReader::open(&args.base)
        .with_context(|| format!("opening base {}", args.base.display()))?;
    let mut overlays: Vec<BaseReader> = Vec::with_capacity(args.overlays.len());
    for p in &args.overlays {
        overlays
            .push(BaseReader::open(p).with_context(|| format!("opening overlay {}", p.display()))?);
    }

    let base_entries: HashMap<&str, &TensorEntry> = base
        .header()
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    // name → index of the LAST overlay carrying it (later overlays win).
    let mut winner: HashMap<&str, usize> = HashMap::new();
    let mut unknown: Vec<String> = Vec::new();
    for (i, ov) in overlays.iter().enumerate() {
        if ov.header().arch != base.header().arch {
            bail!(
                "overlay {} is arch {:?} but the base is {:?}",
                args.overlays[i].display(),
                ov.header().arch,
                base.header().arch
            );
        }
        for t in &ov.header().tensors {
            match base_entries.get(t.name.as_str()) {
                None => unknown.push(format!("{} (from {})", t.name, args.overlays[i].display())),
                Some(b) if b.shape != t.shape => bail!(
                    "{}: overlay {} has shape {:?}, base has {:?}",
                    t.name,
                    args.overlays[i].display(),
                    t.shape,
                    b.shape
                ),
                Some(_) => {
                    winner.insert(t.name.as_str(), i);
                }
            }
        }
    }
    if !unknown.is_empty() {
        bail!(
            "overlay tensors absent from the base bundle:\n  {}",
            unknown.join("\n  ")
        );
    }

    // Header from the base; the tensor tables are rebuilt and the
    // signature no longer covers these bytes.
    let mut header = base.header().clone();
    if header.sig.take().is_some() {
        eprintln!(
            "  note: base was signed; the merged bundle is unsigned (re-sign with `basert sign`)"
        );
    }
    let mut replaced: Vec<String> = Vec::new();
    for t in &base.header().tensors {
        if winner.contains_key(t.name.as_str()) {
            replaced.push(t.name.clone());
        }
    }
    let merge_prov = |method: &str| {
        serde_json::json!({
            "base": args.base.display().to_string(),
            "overlays": args.overlays.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "replaced": replaced,
            "lm_only": args.lm_only,
            "method": method,
        })
    };
    let provenance = |merge: serde_json::Value| match base.header().provenance.clone() {
        Some(serde_json::Value::Object(mut m)) => {
            m.insert("merge".into(), merge);
            serde_json::Value::Object(m)
        }
        Some(other) => serde_json::json!({"base_provenance": other, "merge": merge}),
        None => serde_json::json!({"merge": merge}),
    };
    // Either path writes a temp sibling and renames it over the output only
    // once it is complete: a failure removes the temp and leaves whatever
    // was at the output untouched.
    let tmp_out = temp_sibling(&args.output);

    // ── fast path: clone the base and patch the replaced tensors in place ──
    // A dense overlay of the same dtype keeps every tensor's length, so the
    // base's blob layout (offsets, slots, mmproj, speculator) is reused as
    // is: the output is a copy-on-write clone of the base (APFS `clonefile`,
    // instant and free until written), its header region is rewritten with
    // the overlay's entries at the base's offsets, and only the replaced
    // bytes are written. The full rewrite below needs a whole second bundle
    // on disk (a 360 GB carry's probe = 360 GB more); this needs the overlay.
    let sections_kept =
        !args.lm_only || (base.header().mmproj.is_none() && base.header().speculator.is_none());
    let mut patched_header = header.clone();
    let mut in_place_ok = sections_kept;
    let mut patches: Vec<(usize, u64, u64)> = Vec::new(); // (overlay index, file offset, length)
    if in_place_ok {
        for t in patched_header.tensors.iter_mut() {
            let Some(&i) = winner.get(t.name.as_str()) else {
                continue;
            };
            let e = overlays[i]
                .header()
                .tensors
                .iter()
                .find(|e| e.name == t.name)
                .expect("winner index came from this overlay's table");
            let dense = |x: &TensorEntry| {
                x.scale_offset.is_none() && x.bias_offset.is_none() && x.awq_scale_offset.is_none()
            };
            if e.dtype != t.dtype || e.length != t.length || !dense(e) || !dense(t) {
                in_place_ok = false;
                break;
            }
            let mut entry = e.clone();
            entry.offset = t.offset;
            patches.push((i, base.blob_offset() + t.offset, t.length));
            *t = entry;
        }
    }
    if in_place_ok {
        patched_header.provenance = Some(provenance(merge_prov("clone_patch")));
        let header_json = patched_header
            .to_canonical_json()
            .context("serializing the merged header")?;
        // The clone keeps the base's header_len when the JSON fits in it
        // (a `convert --direct-write` base reserves 64 MiB, space-padded),
        // and grows it only within the base's alignment block; the blob
        // start the reader derives from it must be the base's own.
        let header_len = (header_json.len() as u64).max(base.header_len());
        let header_end = PREFIX_LEN + header_len;
        let blob_start = (header_end + BLOB_ALIGNMENT - 1) & !(BLOB_ALIGNMENT - 1);
        if blob_start == base.blob_offset() {
            match clone_and_patch(
                &args.base,
                &tmp_out,
                &header_json,
                header_len,
                blob_start,
                &patches,
                &overlays,
                &replaced,
            ) {
                Ok(()) => {
                    move_into_place(&tmp_out, &args.output)?;
                    eprintln!(
                        "merge: {} tensors, {} replaced from {} overlay(s) patched into a clone of the base -> {} ({:.1}s)",
                        base.header().tensors.len(),
                        winner.len(),
                        overlays.len(),
                        args.output.display(),
                        t0.elapsed().as_secs_f64()
                    );
                    return Ok(());
                }
                Err(e) => {
                    let _ = std::fs::remove_file(&tmp_out);
                    eprintln!("  note: clone-and-patch unavailable ({e:#}); rewriting the bundle");
                }
            }
        }
    }

    // ── full rewrite: every tensor's bytes streamed into the output ──
    // Direct-write (the blob streams into the temp sibling at a fixed blob
    // start, the header is backfilled at finish) so the peak disk usage is
    // one bundle, not a bundle plus its `.blobtmp` sibling.
    header.tensors.clear();
    header.mmproj = None;
    header.speculator = None;
    let reserve = {
        let base_len = base
            .header()
            .to_canonical_json()
            .map(|j| j.len() as u64)
            .unwrap_or(0);
        base_len + 2 * base_len / 10 + (1u64 << 20)
    };
    let mut writer = BaseWriter::create_direct(&tmp_out, header, reserve)
        .with_context(|| format!("creating {}", tmp_out.display()))?;
    let mut n_mmproj = 0usize;
    let mut n_spec = 0usize;
    let mut n_slots = 0usize;
    let written = (|| -> Result<()> {
        // Entries and bytes verbatim (checksum, scale/bias offsets, residency,
        // algo …); the writer reassigns only offset/length. Each tensor's bytes
        // are touched once: mmap slice → payload Vec.
        for t in &base.header().tensors {
            let (entry, bytes) = match winner.get(t.name.as_str()) {
                Some(&i) => {
                    let ov = &overlays[i];
                    let e = ov
                        .header()
                        .tensors
                        .iter()
                        .find(|e| e.name == t.name)
                        .expect("winner index came from this overlay's table");
                    (e.clone(), ov.tensor_bytes(&t.name)?)
                }
                None => (t.clone(), base.tensor_bytes(&t.name)?),
            };
            writer.add_tensor(TensorPayload {
                entry,
                data: bytes.to_vec(),
            });
        }
        if let Some(mm) = base.header().mmproj.as_ref().filter(|_| !args.lm_only) {
            writer.set_mmproj_arch(mm.arch.clone());
            writer.set_mmproj_config(mm.config.clone());
            for t in &mm.tensors {
                writer.add_mmproj_tensor(TensorPayload {
                    entry: t.clone(),
                    data: base.mmproj_tensor_bytes(&t.name)?.to_vec(),
                });
                n_mmproj += 1;
            }
        }
        if let Some(sp) = base.header().speculator.as_ref().filter(|_| !args.lm_only) {
            writer.set_speculator_arch(sp.arch.clone());
            writer.set_speculator_config(sp.config.clone());
            for t in &sp.tensors {
                writer.add_speculator_tensor(TensorPayload {
                    entry: t.clone(),
                    data: base.tensor_bytes(&t.name)?.to_vec(),
                });
                n_spec += 1;
            }
        }
        let slots = base.slots().context("reading base slots")?;
        n_slots = slots.len();
        for s in slots {
            writer.add_slot(s);
        }
        writer.set_provenance(provenance(merge_prov("rewrite")));
        writer.finish().context("writing merged bundle")
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp_out);
        return Err(e);
    }
    move_into_place(&tmp_out, &args.output)?;
    eprintln!(
        "merge: {} tensors, {} replaced from {} overlay(s), {} mmproj, {} speculator, {} slots -> {} ({:.1}s)",
        base.header().tensors.len(),
        winner.len(),
        overlays.len(),
        n_mmproj,
        n_spec,
        n_slots,
        args.output.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Rename the finished temp sibling over the output; on failure the temp is
/// removed and the output is as it was.
fn move_into_place(tmp_out: &std::path::Path, output: &std::path::Path) -> Result<()> {
    std::fs::rename(tmp_out, output).with_context(|| {
        let _ = std::fs::remove_file(tmp_out);
        format!(
            "moving {} into place as {}",
            tmp_out.display(),
            output.display()
        )
    })
}

/// `basert merge`'s in-place path: `output` becomes a copy-on-write clone
/// of `base` (APFS `clonefile`; the blocks are shared until written), its
/// prefix + header region is rewritten with `header_json` (space-padded to
/// `header_len` like a direct-written header, then zero-padded to
/// `blob_start`, the base's own blob start), and each `(overlay, file
/// offset, length)` patch is written over the clone's bytes. Errors (no
/// clone support on this filesystem, a cross-volume output, …) leave the
/// caller to rewrite the bundle; the partial output (the temp sibling the
/// caller hands in) is removed by it.
#[allow(clippy::too_many_arguments)]
fn clone_and_patch(
    base: &std::path::Path,
    output: &std::path::Path,
    header_json: &[u8],
    header_len: u64,
    blob_start: u64,
    patches: &[(usize, u64, u64)],
    overlays: &[base_format::BaseReader],
    replaced: &[String],
) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::FileExt;
    if output.exists() {
        std::fs::remove_file(output).with_context(|| format!("removing {}", output.display()))?;
    }
    clone_file(base, output)
        .with_context(|| format!("cloning {} to {}", base.display(), output.display()))?;
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(output)
        .with_context(|| format!("opening the clone {}", output.display()))?;
    {
        let mut w = std::io::BufWriter::new(&mut f);
        w.write_all(&base_format::MAGIC)?;
        w.write_all(&base_format::FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&header_len.to_le_bytes())?;
        w.write_all(header_json)?;
        let mut fill = |byte: u8, mut n: usize| -> std::io::Result<()> {
            let buf = [byte; 8192];
            while n > 0 {
                let k = n.min(buf.len());
                w.write_all(&buf[..k])?;
                n -= k;
            }
            Ok(())
        };
        fill(b' ', (header_len - header_json.len() as u64) as usize)?;
        fill(0, (blob_start - PREFIX_LEN - header_len) as usize)?;
        w.flush()?;
    }
    for (k, &(i, at, len)) in patches.iter().enumerate() {
        let name = &replaced[k];
        let bytes = overlays[i].tensor_bytes(name)?;
        anyhow::ensure!(
            bytes.len() as u64 == len,
            "{name}: {} overlay bytes for {len} in the base",
            bytes.len()
        );
        f.write_all_at(bytes, at)
            .with_context(|| format!("{name}: writing {len} bytes at {at}"))?;
    }
    f.sync_all()?;
    Ok(())
}

/// A copy-on-write file clone (macOS `clonefile(2)`): instant, and the
/// clone shares the source's blocks until either is written.
#[cfg(target_os = "macos")]
fn clone_file(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::os::unix::ffi::OsStrExt;
    extern "C" {
        fn clonefile(src: *const c_char, dst: *const c_char, flags: c_int) -> c_int;
    }
    let s = CString::new(src.as_os_str().as_bytes())?;
    let d = CString::new(dst.as_os_str().as_bytes())?;
    // Safety: two NUL-terminated paths; the call creates `dst` or fails.
    if unsafe { clonefile(s.as_ptr(), d.as_ptr(), 0) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn clone_file(_src: &std::path::Path, _dst: &std::path::Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "file cloning is only implemented on macOS (APFS clonefile)",
    ))
}

/// Decode one bundle tensor to f32: GGUF super-blocks (`layout: gguf_super`)
/// through the GGUF reader's own dequant, the canonical-quant dtypes through
/// `unpack_rtn` (the packer's own inverse), f16 / bf16 / f32 directly.
pub(crate) fn dequant_entry(entry: &TensorEntry, bytes: &[u8]) -> Result<Vec<f32>> {
    let n: usize = entry.shape.iter().product::<u64>() as usize;
    // Before the base_qN test: a passed-through super-block tensor is filed
    // under the base_qN that sizes it, and `unpack_rtn` would read its
    // blocks as a slab.
    if entry.layout == Some(Layout::GgufSuper) {
        let code = entry.source_ggml_type.ok_or_else(|| {
            anyhow::anyhow!("{}: layout gguf_super without source_ggml_type", entry.name)
        })?;
        let info = TensorInfo {
            name: entry.name.clone(),
            shape: entry.shape.iter().rev().copied().collect(), // ne order

            ggml_type: GgmlType::from_u32(code)
                .with_context(|| format!("{}: source_ggml_type", entry.name))?,
            data_offset: 0,
        };
        return base_readers::gguf::dequant_to_f32(&info, bytes)
            .with_context(|| format!("{}: dequantizing GGUF super-blocks", entry.name));
    }
    if is_canonical_quant(entry.dtype) {
        let bits = entry
            .dtype
            .bits_per_weight()
            .expect("canonical quant dtype");
        let gs = entry
            .group_size
            .or_else(|| entry.dtype.default_group_size())
            .expect("canonical quant dtypes have a default group size");
        let slice = |off: Option<u64>, len: Option<u64>| -> Vec<u8> {
            match (off, len) {
                (Some(o), Some(l)) => bytes[o as usize..(o + l) as usize].to_vec(),
                _ => Vec::new(),
            }
        };
        let w_end = entry
            .scale_offset
            .map(|o| o as usize)
            .unwrap_or(bytes.len());
        let packed = base_quant::Packed {
            packed_weights: bytes[..w_end].to_vec(),
            scales: slice(entry.scale_offset, entry.scale_length),
            biases: slice(entry.bias_offset, entry.bias_length),
            group_size: gs,
            scale_dtype: entry.scale_dtype,
        };
        let cfg = base_quant::RtnConfig {
            bits,
            group_size: gs,
            symmetric: entry.symmetric,
            scale_dtype: entry.scale_dtype.unwrap_or(base_format::ScaleDtype::F16),
        };
        return Ok(base_quant::unpack_rtn(&packed, n, cfg));
    }
    Ok(match entry.dtype {
        TensorDtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        TensorDtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        TensorDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => bail!(
            "{}: dtype {other:?} is not supported by fakequant",
            entry.name
        ),
    })
}

/// The f32 image of one bundle tensor, as the runtime would see it, written
/// to `out`; the summary line both `fakequant` forms print.
fn write_fakequant_image(entry: &TensorEntry, bytes: &[u8], out: &std::path::Path) -> Result<()> {
    let deq = dequant_entry(entry, bytes)?;
    let mut image = Vec::with_capacity(deq.len() * 4);
    for v in &deq {
        image.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, &image).with_context(|| format!("writing {}", out.display()))?;
    println!(
        "{}",
        serde_json::json!({"shape": entry.shape, "elements": deq.len(), "dtype": entry.dtype,
            "group_size": entry.group_size, "scale_dtype": entry.scale_dtype,
            "layout": entry.layout, "source_ggml_type": entry.source_ggml_type})
    );
    eprintln!(
        "fakequant: {} {:?} -> {} ({} bytes)",
        entry.name,
        entry.dtype,
        out.display(),
        image.len()
    );
    Ok(())
}

pub fn cmd_fakequant(args: FakequantArgs) -> Result<()> {
    use base_format::{BaseReader, ScaleDtype};
    if let Some(bundle) = &args.bundle {
        refuse_output_aliasing_inputs("fakequant", &args.out, &[bundle.as_path()])?;
        let reader = BaseReader::open(bundle)
            .with_context(|| format!("opening bundle {}", bundle.display()))?;
        let entry = reader.find_entry(&args.tensor)?.clone();
        return write_fakequant_image(&entry, reader.tensor_bytes(&args.tensor)?, &args.out);
    }
    let input = args.input.as_ref().expect("clap: input or --bundle");
    let dtype_arg = args.dtype.as_ref().expect("clap: --dtype or --bundle");
    let dtype: TensorDtype = serde_json::from_value(serde_json::json!(dtype_arg))
        .with_context(|| format!("--dtype {dtype_arg:?}"))?;
    let scale_dtype: ScaleDtype = serde_json::from_value(serde_json::json!(args.scale_dtype))
        .with_context(|| format!("--scale-dtype {:?}", args.scale_dtype))?;
    if !is_canonical_quant(dtype) && !matches!(dtype, TensorDtype::F16 | TensorDtype::Bf16) {
        bail!("--dtype {dtype:?} is not supported by fakequant (base_q2..q8, f16, bf16)");
    }
    // One-entry overlay through convert's own per-tensor path (value hooks,
    // dtype policy, packer), read straight back: the image is exactly what a
    // bundle quantized this way holds. No AWQ fold (the plain RTN image).
    let entry = OverlaySpecEntry {
        name: args.tensor.clone(),
        dtype,
        group_size: args.group_size,
        scale_dtype: Some(scale_dtype),
        symmetric: if is_canonical_quant(dtype) {
            Some(args.symmetric)
        } else {
            None
        },
        algo: OverlayAlgo::Rtn,
        awq_alpha: None,
        codes: None,
        kquant: None,
        values: None,
    };
    let spec_json = serde_json::json!({"tensors": {args.tensor.clone(): {"dtype": dtype}}});
    let ctx = overlay_context(vec![entry], spec_json, None, 0, true)?;
    let dir = tempfile::tempdir().context("creating a temp dir for the fakequant overlay")?;
    let tmp = dir.path().join("fakequant.base");
    convert_hf(input, &tmp, &ctx)?;
    let reader = BaseReader::open(&tmp).context("reading the fakequant overlay back")?;
    let entry = reader.find_entry(&args.tensor)?.clone();
    write_fakequant_image(&entry, reader.tensor_bytes(&args.tensor)?, &args.out)
}

/// Minimal safetensors writer (sorted header, contiguous data) for tests.
#[cfg(test)]
pub(crate) fn write_st(path: &std::path::Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (n, dt, sh, b) in tensors {
        header.insert(
            n.to_string(),
            serde_json::json!({"dtype": dt, "shape": sh, "data_offsets": [off, off + b.len()]}),
        );
        off += b.len();
    }
    let h = serde_json::Value::Object(header).to_string();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(h.as_bytes());
    for (_, _, _, b) in tensors {
        out.extend_from_slice(b);
    }
    std::fs::write(path, out).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_format::ScaleDtype;

    fn randv(n: usize, seed: &mut u64, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                amp * (((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
            })
            .collect()
    }

    /// A gptq entry whose codes are RTN's own codes packs byte-identically
    /// to convert's quantized branch ([weights | scales | biases] from
    /// `pack_rtn`), for a fused 3-D expert tensor given as [E, N, K] or flat.
    #[test]
    fn gptq_3d_expert_codes_pack_like_convert_rtn() {
        let mut seed = 11u64;
        let (e_n, n, k, gs) = (3usize, 8usize, 128usize, 64usize);
        let w = randv(e_n * n * k, &mut seed, 0.7);
        let name = "layers.0.ffn_down_exps.weight";
        let shape = vec![e_n as u64, n as u64, k as u64];
        let cfg = base_quant::RtnConfig {
            bits: 4,
            group_size: gs as u32,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        };
        let p = base_quant::pack_rtn(&w, cfg);
        let mut want = p.packed_weights.clone();
        want.extend_from_slice(&p.scales);
        want.extend_from_slice(&p.biases);
        // the same codes, recovered from convert's own RTN
        let deq = base_quant::unpack_rtn(&p, w.len(), cfg);
        let dec = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let (scales, biases) = (dec(&p.scales), dec(&p.biases));
        let codes: Vec<u8> = deq
            .iter()
            .enumerate()
            .map(|(i, &v)| ((v - biases[i / gs]) / scales[i / gs]).round() as u8)
            .collect();
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let dir = tempfile::tempdir().unwrap();
        for (tag, cshape, sshape) in [
            (
                "3d",
                shape.clone(),
                vec![e_n as u64, n as u64, (k / gs) as u64],
            ),
            (
                "flat",
                vec![(e_n * n) as u64, k as u64],
                vec![(e_n * n) as u64, (k / gs) as u64],
            ),
        ] {
            let path = dir.path().join(format!("{tag}.safetensors"));
            write_st(
                &path,
                &[
                    (
                        &format!("{name}.biases"),
                        "F32",
                        sshape.clone(),
                        f32b(&biases),
                    ),
                    (&format!("{name}.codes"), "U8", cshape, codes.clone()),
                    (&format!("{name}.scales"), "F32", sshape, f32b(&scales)),
                ],
            );
            let st = SafetensorsFile::open(&path).unwrap();
            let e = OverlaySpecEntry {
                name: name.into(),
                dtype: TensorDtype::BaseQ4,
                group_size: Some(gs as u32),
                scale_dtype: Some(ScaleDtype::Bf16),
                symmetric: None,
                algo: OverlayAlgo::Gptq,
                awq_alpha: None,
                codes: Some(path.clone()),
                kquant: None,
                values: None,
            };
            let (entry, got) = emit_gptq_codes(&e, name, shape.clone(), &st).unwrap();
            assert_eq!(entry.shape, shape);
            assert_eq!(entry.algo.as_deref(), Some("gptq"));
            assert_eq!(got, want, "{tag}: gptq bytes differ from convert RTN");
            // and the bundle-side decoder reads the same values back
            assert_eq!(dequant_entry(&entry, &got).unwrap(), deq);
        }
    }

    #[test]
    fn spec_parser_rejects_inconsistent_entries() {
        let bad = [
            serde_json::json!({"tensors": {}}),
            serde_json::json!({"tensors": {"a": {"dtype": "base_q4", "algo": "gptq"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "algo": "awq"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "base_q4", "awq_alpha": 0.5}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "base_q4", "codes": "x.safetensors"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "base_q4", "bits": 4}}}),
        ];
        for b in bad {
            assert!(parse_overlay_spec(&b).is_err(), "{b}");
        }
        let ok = serde_json::json!({"tensors": {
            "layers.0.self_attn.q_proj.weight": {"dtype": "base_q4", "group_size": 64, "algo": "awq", "awq_alpha": 0.5},
            "layers.0.mlp.down_proj.weight": {"dtype": "base_q8"},
            "lm_head.weight": {"dtype": "base_q4", "algo": "gptq", "codes": "codes.safetensors"},
        }});
        let entries = parse_overlay_spec(&ok).unwrap();
        assert_eq!(entries.len(), 3);
        let gptq = entries.iter().find(|e| e.name == "lm_head.weight").unwrap();
        assert_eq!(gptq.algo, OverlayAlgo::Gptq);
        assert_eq!(
            gptq.codes.as_deref(),
            Some(std::path::Path::new("codes.safetensors"))
        );
    }

    /// The k-quant form parses to a prepacked entry under the passthrough's
    /// bundle dtype; the affine keys, `awq`, a missing `blocks`, and `blocks`
    /// on an affine dtype are refused.
    #[test]
    fn spec_parser_takes_the_kquant_form() {
        let ok = serde_json::json!({"tensors": {
            "layers.0.self_attn.q_proj.weight": {"dtype": "q4_k", "blocks": "b.safetensors"},
            "layers.0.mlp.down_proj.weight": {"dtype": "q6_k", "blocks": "b.safetensors", "algo": "gptq"},
            "layers.1.mlp.up_proj.weight": {"dtype": "iq4_xs", "blocks": "c.safetensors", "algo": "rtn"},
            "layers.1.mlp.gate_proj.weight": {"dtype": "base_q4", "group_size": 64},
        }});
        let entries = parse_overlay_spec(&ok).unwrap();
        let find = |n: &str| entries.iter().find(|e| e.name == n).unwrap();
        let q = find("layers.0.self_attn.q_proj.weight");
        assert_eq!(q.dtype, TensorDtype::BaseQ4);
        assert_eq!(q.algo, OverlayAlgo::Rtn);
        assert!(q.is_prepacked());
        assert_eq!(
            q.kquant,
            Some(KquantBlocks {
                ggml_type: GgmlType::Q4K,
                blocks: PathBuf::from("b.safetensors")
            })
        );
        assert!(q.group_size.is_none() && q.scale_dtype.is_none() && q.symmetric.is_none());
        assert!(q.codes.is_none() && q.awq_alpha.is_none());
        let d = find("layers.0.mlp.down_proj.weight");
        assert_eq!(d.dtype, TensorDtype::BaseQ6);
        assert_eq!(d.algo, OverlayAlgo::Gptq);
        assert_eq!(d.kquant.as_ref().unwrap().ggml_type, GgmlType::Q6K);
        let u = find("layers.1.mlp.up_proj.weight");
        assert_eq!(u.dtype, TensorDtype::BaseQ4);
        assert_eq!(u.kquant.as_ref().unwrap().ggml_type, GgmlType::Iq4Xs);
        let g = find("layers.1.mlp.gate_proj.weight");
        assert!(g.kquant.is_none() && !g.is_prepacked());
        for (name, ty) in KQUANT_TYPES {
            let one =
                serde_json::json!({"tensors": {"a": {"dtype": name, "blocks": "b.safetensors"}}});
            let e = parse_overlay_spec(&one).unwrap();
            assert_eq!(e[0].kquant.as_ref().unwrap().ggml_type, ty);
            assert_eq!(e[0].dtype, kquant_passthrough_dtype(ty).unwrap().0);
        }

        let bad = [
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": 3}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "group_size": 256}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "scale_dtype": "f16"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "symmetric": false}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "codes": "c"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "awq_alpha": 0.5}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "algo": "awq"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "bits": 4}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q8_k", "blocks": "b"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "iq2_xs", "blocks": "b"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "base_q4", "blocks": "b"}}}),
        ];
        for b in bad {
            assert!(parse_overlay_spec(&b).is_err(), "{b}");
        }
    }

    /// The dense-values form parses to a prepacked entry under its own half
    /// dtype; an affine or k-quant dtype with `values`, the affine keys or
    /// `blocks` with it, and `awq` are refused.
    #[test]
    fn spec_parser_takes_the_values_form() {
        let ok = serde_json::json!({"tensors": {
            "layers.0.mlp.experts.down_proj.weight": {"dtype": "f16", "values": "v.safetensors"},
            "layers.0.mlp.down_proj.weight": {"dtype": "bf16", "values": "v.safetensors", "algo": "gptq"},
            "layers.1.mlp.up_proj.weight": {"dtype": "f16", "values": "w.safetensors", "algo": "rtn"},
            "layers.1.mlp.gate_proj.weight": {"dtype": "f16"},
        }});
        let entries = parse_overlay_spec(&ok).unwrap();
        let find = |n: &str| entries.iter().find(|e| e.name == n).unwrap();
        let x = find("layers.0.mlp.experts.down_proj.weight");
        assert_eq!(x.dtype, TensorDtype::F16);
        assert_eq!(x.algo, OverlayAlgo::Rtn);
        assert!(x.is_prepacked());
        assert_eq!(
            x.values.as_deref(),
            Some(std::path::Path::new("v.safetensors"))
        );
        assert!(x.group_size.is_none() && x.scale_dtype.is_none() && x.symmetric.is_none());
        assert!(x.codes.is_none() && x.awq_alpha.is_none() && x.kquant.is_none());
        let d = find("layers.0.mlp.down_proj.weight");
        assert_eq!(d.dtype, TensorDtype::Bf16);
        assert_eq!(d.algo, OverlayAlgo::Gptq);
        assert!(d.values.is_some());
        assert_eq!(find("layers.1.mlp.up_proj.weight").algo, OverlayAlgo::Rtn);
        let g = find("layers.1.mlp.gate_proj.weight");
        assert!(g.values.is_none() && !g.is_prepacked());

        let bad = [
            serde_json::json!({"tensors": {"a": {"dtype": "base_q4", "values": "v"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f32", "values": "v"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "values": "v"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "q4_k", "blocks": "b", "values": "v"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": 3}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "group_size": 64}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "scale_dtype": "bf16"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "symmetric": false}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "codes": "c"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "awq_alpha": 0.5}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "blocks": "b"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "f16", "values": "v", "algo": "awq"}}}),
            serde_json::json!({"tensors": {"a": {"dtype": "bf16", "values": "v", "algo": "x"}}}),
        ];
        for b in bad {
            assert!(parse_overlay_spec(&b).is_err(), "{b}");
        }
    }

    /// A dense-values entry is the entry a `--profile` f16 / bf16 rule
    /// writes for the same tensor (no scales, no group, warm on the
    /// accelerator; the source shape kept for a fused 3-D expert stack given
    /// 3-D or flat), its bytes verbatim, and the bundle-side decoder reads
    /// the halves back.
    #[test]
    fn values_file_like_a_half_profile_rule() {
        let name = "layers.0.mlp.experts.down_proj.weight";
        let (e_n, n, k) = (2u64, 4u64, 512u64);
        let shape = vec![e_n, n, k];
        let flat = vec![e_n * n, k];
        let mut seed = 5u64;
        let vals = randv((e_n * n * k) as usize, &mut seed, 0.5);
        let f16b: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let bf16b: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let dir = tempfile::tempdir().unwrap();
        for (tag, st_dtype, dtype, bytes, st_shape) in [
            ("f16_3d", "F16", TensorDtype::F16, &f16b, shape.clone()),
            ("f16_flat", "F16", TensorDtype::F16, &f16b, flat.clone()),
            ("bf16_3d", "BF16", TensorDtype::Bf16, &bf16b, shape.clone()),
        ] {
            let path = dir.path().join(format!("{tag}.safetensors"));
            write_st(
                &path,
                &[(&format!("{name}.values"), st_dtype, st_shape, bytes.clone())],
            );
            let st = SafetensorsFile::open(&path).unwrap();
            let spec_dtype = serde_json::to_value(dtype).unwrap();
            let spec = serde_json::json!({"tensors": {name: {"dtype": spec_dtype, "values": path, "algo": "gptq"}}});
            let e = parse_overlay_spec(&spec).unwrap().remove(0);
            let (entry, got) = emit_values(&e, name, shape.clone(), &st).unwrap();
            assert_eq!(&got, bytes, "{tag}: bytes must be verbatim");
            assert_eq!(entry.dtype, dtype);
            assert_eq!(entry.shape, shape, "{tag}: the source shape is kept");
            assert_eq!(entry.length, bytes.len() as u64);
            assert!(entry.scale_offset.is_none() && entry.bias_offset.is_none());
            assert!(entry.group_size.is_none() && entry.scale_dtype.is_none());
            assert!(entry.layout.is_none() && entry.source_ggml_type.is_none());
            assert_eq!(entry.residency, Some(base_format::ResidencyHint::Warm));
            assert_eq!(
                entry.compute_region,
                base_format::ComputeRegion::Accelerator
            );
            assert_eq!(entry.algo.as_deref(), Some("gptq"));
            let deq = dequant_entry(&entry, &got).unwrap();
            let want: Vec<f32> = if dtype == TensorDtype::F16 {
                vals.iter()
                    .map(|&v| half::f16::from_f32(v).to_f32())
                    .collect()
            } else {
                vals.iter()
                    .map(|&v| half::bf16::from_f32(v).to_f32())
                    .collect()
            };
            assert_eq!(deq, want, "{tag}: dequant differs from the halves");
        }
        // an embedding is hot on the GPU, like convert's embedding branch
        let path = dir.path().join("emb.safetensors");
        write_st(
            &path,
            &[("lm_head.weight.values", "F16", flat.clone(), f16b.clone())],
        );
        let st = SafetensorsFile::open(&path).unwrap();
        let spec =
            serde_json::json!({"tensors": {"lm_head.weight": {"dtype": "f16", "values": path}}});
        let e = parse_overlay_spec(&spec).unwrap().remove(0);
        let (entry, _) = emit_values(&e, "lm_head.weight", flat.clone(), &st).unwrap();
        assert_eq!(entry.residency, Some(base_format::ResidencyHint::Hot));
        assert_eq!(entry.compute_region, base_format::ComputeRegion::Gpu);
        assert_eq!(entry.algo.as_deref(), Some("rtn"));

        // refusals: a missing / wrongly typed / wrongly shaped values tensor,
        // a 1-D source
        let path = dir.path().join("bad.safetensors");
        write_st(
            &path,
            &[
                (
                    &format!("{name}.values"),
                    "BF16",
                    shape.clone(),
                    bf16b.clone(),
                ),
                (
                    "short.values",
                    "F16",
                    vec![e_n * n, k - 2],
                    f16b[..f16b.len() - 4 * e_n as usize * n as usize].to_vec(),
                ),
                ("ok.values", "F16", shape.clone(), f16b.clone()),
            ],
        );
        let st = SafetensorsFile::open(&path).unwrap();
        let entry_for = |nm: &str| {
            parse_overlay_spec(
                &serde_json::json!({"tensors": {nm: {"dtype": "f16", "values": path}}}),
            )
            .unwrap()
            .remove(0)
        };
        assert!(emit_values(&entry_for(name), name, shape.clone(), &st).is_err()); // bf16 file, f16 spec
        assert!(emit_values(&entry_for("missing"), "missing", shape.clone(), &st).is_err());
        assert!(emit_values(&entry_for("short"), "short", shape.clone(), &st).is_err());
        assert!(emit_values(&entry_for("ok"), "ok", vec![e_n * n * k], &st).is_err());
        assert!(emit_values(&entry_for("ok"), "ok", flat.clone(), &st).is_err()); // 3-D file, 2-D source
        assert!(emit_values(&entry_for("ok"), "ok", shape.clone(), &st).is_ok());
    }

    /// A k-quant overlay entry is, field for field, the entry `convert
    /// --kquant-passthrough` writes for the same ggml type (plus `algo`),
    /// its bytes verbatim, and the bundle-side decoder reads it through the
    /// GGUF dequant — for every type on the menu.
    #[test]
    fn kquant_blocks_file_like_the_passthrough() {
        let (n, k) = (4u64, 512u64);
        let name = "layers.0.self_attn.q_proj.weight";
        let dir = tempfile::tempdir().unwrap();
        for (spec_name, ty) in KQUANT_TYPES {
            let (_, bb) = ty.block_geometry();
            let n_bytes = (n * (k / 256) * bb as u64) as usize;
            let mut seed = 7u64 + bb as u64;
            let blocks: Vec<u8> = randv(n_bytes, &mut seed, 127.0)
                .iter()
                .map(|v| (*v + 128.0) as u8)
                .collect();
            let path = dir.path().join(format!("{spec_name}.safetensors"));
            write_st(
                &path,
                &[(
                    &format!("{name}.blocks"),
                    "U8",
                    vec![n, (k / 256) * bb as u64],
                    blocks.clone(),
                )],
            );
            let st = SafetensorsFile::open(&path).unwrap();
            let spec = serde_json::json!({"tensors": {name: {"dtype": spec_name, "blocks": path, "algo": "gptq"}}});
            let e = parse_overlay_spec(&spec).unwrap().remove(0);
            let (entry, got) = emit_kquant_blocks(&e, name, vec![n, k], &st).unwrap();
            assert_eq!(got, blocks, "{spec_name}: bytes must be verbatim");

            // the same weight as a GGUF file lists it: ne = [k, n]
            let src = TensorInfo {
                name: name.to_string(),
                shape: vec![k, n],
                ggml_type: ty,
                data_offset: 0,
            };
            let (mut want, want_bytes) = kquant_passthrough_entry(&src, &blocks, name.to_string());
            assert_eq!(want_bytes, blocks);
            want.algo = Some("gptq".to_string());
            assert_eq!(
                serde_json::to_value(&entry).unwrap(),
                serde_json::to_value(&want).unwrap(),
                "{spec_name}: overlay entry differs from the passthrough entry"
            );
            let (want_dtype, want_code) = kquant_passthrough_dtype(ty).unwrap();
            assert_eq!(entry.dtype, want_dtype);
            assert_eq!(entry.source_ggml_type, Some(want_code));
            assert_eq!(entry.layout, Some(Layout::GgufSuper));
            assert_eq!(entry.group_size, Some(256));
            assert_eq!(entry.length, blocks.len() as u64);
            assert_eq!(entry.shape, vec![n, k], "{spec_name}: declared [out, in]");

            // the bundle-side decoder reads it as the GGUF reader does
            let deq = dequant_entry(&entry, &got).unwrap();
            let via_gguf = base_readers::gguf::dequant_to_f32(&src, &blocks).unwrap();
            assert_eq!(deq.len(), (n * k) as usize);
            assert!(
                deq.iter()
                    .zip(&via_gguf)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{spec_name}: dequant differs from dequant_to_f32"
            );
        }

        // refusals: a 3-D tensor in a type with no expert kernel, K not a
        // multiple of 256, a missing / wrongly typed / wrongly shaped blocks
        // tensor
        let path = dir.path().join("bad.safetensors");
        let (_, bb) = GgmlType::Q4K.block_geometry();
        write_st(
            &path,
            &[
                (
                    &format!("{name}.blocks"),
                    "U8",
                    vec![n, (k / 256) * bb as u64 + 1],
                    vec![0u8; (n * ((k / 256) * bb as u64 + 1)) as usize],
                ),
                (
                    "other.blocks",
                    "U8",
                    vec![n, (k / 256) * bb as u64],
                    vec![0u8; (n * (k / 256) * bb as u64) as usize],
                ),
                (
                    "f16.blocks",
                    "F16",
                    vec![n, (k / 256) * bb as u64 / 2],
                    vec![0u8; (n * (k / 256) * bb as u64) as usize],
                ),
                // a 2-expert stack in Q4_K (an expert type) and Q6_K (dense kernels only)
                (
                    "exps4.blocks",
                    "U8",
                    vec![2, n, (k / 256) * bb as u64],
                    vec![0u8; (2 * n * (k / 256) * bb as u64) as usize],
                ),
                (
                    "exps6.blocks",
                    "U8",
                    vec![2, n, (k / 256) * GgmlType::Q6K.block_geometry().1 as u64],
                    vec![
                        0u8;
                        (2 * n * (k / 256) * GgmlType::Q6K.block_geometry().1 as u64) as usize
                    ],
                ),
            ],
        );
        let st = SafetensorsFile::open(&path).unwrap();
        let entry_for = |nm: &str| {
            parse_overlay_spec(
                &serde_json::json!({"tensors": {nm: {"dtype": "q4_k", "blocks": path}}}),
            )
            .unwrap()
            .remove(0)
        };
        assert!(emit_kquant_blocks(&entry_for(name), name, vec![2, n, k], &st).is_err());
        assert!(emit_kquant_blocks(&entry_for(name), name, vec![n, 500], &st).is_err());
        assert!(emit_kquant_blocks(&entry_for(name), name, vec![n, k], &st).is_err()); // wrong shape
        assert!(emit_kquant_blocks(&entry_for("missing"), "missing", vec![n, k], &st).is_err());
        assert!(emit_kquant_blocks(&entry_for("f16"), "f16", vec![n, k], &st).is_err());
        assert!(emit_kquant_blocks(&entry_for("other"), "other", vec![n, k], &st).is_ok());
        // a 3-D expert stack: packs in Q4_K (expert_stack_kquant), refused in Q6_K
        assert!(emit_kquant_blocks(&entry_for("exps4"), "exps4", vec![2, n, k], &st).is_ok());
        let e6 = parse_overlay_spec(
            &serde_json::json!({"tensors": {"exps6": {"dtype": "q6_k", "blocks": path}}}),
        )
        .unwrap()
        .remove(0);
        let err = emit_kquant_blocks(&e6, "exps6", vec![2, n, k], &st)
            .unwrap_err()
            .to_string();
        assert!(err.contains("3-D expert stack"), "{err}");
    }
    /// A tiny base and an overlay replacing one of its tensors, both as
    /// bundles the merge can read.
    fn merge_fixtures(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        use crate::embed::fixtures::{header, payload};
        use base_format::BaseWriter;
        let cfg = serde_json::json!({"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 1});
        let base = dir.join("base.base");
        let mut w = BaseWriter::create(&base, header("llama", cfg.clone())).unwrap();
        w.add_tensor(payload("token_embedding.weight", 256, 0xA1));
        w.add_tensor(payload("layers.0.attention.q.weight", 100, 0xA2));
        w.finish().unwrap();
        let ov = dir.join("ov.base");
        let mut w = BaseWriter::create(&ov, header("llama", cfg)).unwrap();
        w.add_tensor(payload("layers.0.attention.q.weight", 100, 0xB7));
        w.finish().unwrap();
        (base, ov)
    }

    fn merge_args(
        base: &std::path::Path,
        ov: &std::path::Path,
        out: &std::path::Path,
    ) -> MergeArgs {
        MergeArgs {
            base: base.to_path_buf(),
            overlays: vec![ov.to_path_buf()],
            output: out.to_path_buf(),
            lm_only: false,
        }
    }

    #[test]
    fn merge_refuses_an_output_that_is_one_of_its_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let (base, ov) = merge_fixtures(dir.path());
        let base_bytes = std::fs::read(&base).unwrap();
        let ov_bytes = std::fs::read(&ov).unwrap();

        // The base itself, the overlay itself, and a symlink to the base.
        let err = cmd_merge(merge_args(&base, &ov, &base))
            .unwrap_err()
            .to_string();
        assert!(err.contains("same file as the input"), "{err}");
        let err = cmd_merge(merge_args(&base, &ov, &ov))
            .unwrap_err()
            .to_string();
        assert!(err.contains("same file as the input"), "{err}");
        #[cfg(unix)]
        {
            let link = dir.path().join("link.base");
            std::os::unix::fs::symlink(&base, &link).unwrap();
            let err = cmd_merge(merge_args(&base, &ov, &link))
                .unwrap_err()
                .to_string();
            assert!(err.contains("same file as the input"), "{err}");
        }

        // Nothing was truncated or rewritten, and no temp file is left.
        assert_eq!(std::fs::read(&base).unwrap(), base_bytes);
        assert_eq!(std::fs::read(&ov).unwrap(), ov_bytes);
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        #[cfg(unix)]
        assert_eq!(names, ["base.base", "link.base", "ov.base"]);
        #[cfg(not(unix))]
        assert_eq!(names, ["base.base", "ov.base"]);
    }

    #[test]
    fn merge_writes_the_output_whole_and_leaves_no_temp_behind() {
        use base_format::BaseReader;
        let dir = tempfile::tempdir().unwrap();
        let (base, ov) = merge_fixtures(dir.path());
        let out = dir.path().join("out.base");
        // A stale output is replaced, not appended to.
        std::fs::write(&out, b"stale").unwrap();
        cmd_merge(merge_args(&base, &ov, &out)).unwrap();

        let merged = BaseReader::open(&out).unwrap();
        assert_eq!(merged.header().tensors.len(), 2);
        assert_eq!(
            merged.tensor_bytes("layers.0.attention.q.weight").unwrap(),
            &vec![0xB7u8; 100][..]
        );
        assert_eq!(
            merged.tensor_bytes("token_embedding.weight").unwrap(),
            &vec![0xA1u8; 256][..]
        );
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["base.base", "out.base", "ov.base"]);
    }
}
