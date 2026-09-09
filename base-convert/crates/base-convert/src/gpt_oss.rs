//! gpt-oss conversion: HF `model_type = "gpt_oss"` MXFP4 checkpoints.
//!
//! This is a **mirror-policy transplant** path, the same class as the MLX
//! quantized-checkpoint path: every tensor the checkpoint stores quantized
//! (the MoE expert stacks, MXFP4 `*_blocks` + `*_scales`) is copied into
//! the bundle byte-for-byte — packed FP4 codes and E8M0 block scales
//! verbatim, never dequantized or requantized — and every tensor the
//! checkpoint keeps unquantized is carried at a dtype that represents it
//! losslessly (bf16 matrices stay bf16; 1-D / bias tensors are widened to
//! f32). The bundle therefore holds exactly the checkpoint's numbers, and
//! the header's `provenance` block records, per bundle tensor, which
//! checkpoint tensors it came from and how, so an independent gate can
//! check the claim.
//!
//! The HF generic path is not used because it funnels everything through
//! f32 + the target quantizer (which would requantize the experts) and its
//! safetensors reader has no route for the U8 block/scale tensors.
//!
//! Canonical bundle names (HF convention, `model.` stripped):
//!
//! | checkpoint                                   | bundle                                      | dtype  |
//! |----------------------------------------------|---------------------------------------------|--------|
//! | `model.embed_tokens.weight`                  | `embed_tokens.weight`                       | bf16   |
//! | `lm_head.weight`                             | `lm_head.weight`                            | bf16   |
//! | `model.norm.weight`                          | `final_norm.weight`                         | f32    |
//! | `…input_layernorm.weight`                    | `layers.N.input_norm.weight`                | f32    |
//! | `…post_attention_layernorm.weight`           | `layers.N.post_attn_norm.weight`            | f32    |
//! | `…self_attn.{q,k,v,o}_proj.weight`           | same                                        | bf16   |
//! | `…self_attn.{q,k,v,o}_proj.bias`             | same                                        | f32    |
//! | `…self_attn.sinks`                           | same                                        | f32    |
//! | `…mlp.router.weight`                         | same                                        | bf16   |
//! | `…mlp.router.bias`                           | same                                        | f32    |
//! | `…mlp.experts.gate_up_proj_blocks/_scales`   | `layers.N.mlp.experts.gate_up_proj.weight`  | mxfp4  |
//! | `…mlp.experts.gate_up_proj_bias`             | `layers.N.mlp.experts.gate_up_proj.bias`    | f32    |
//! | `…mlp.experts.down_proj_blocks/_scales`      | `layers.N.mlp.experts.down_proj.weight`     | mxfp4  |
//! | `…mlp.experts.down_proj_bias`                | `layers.N.mlp.experts.down_proj.bias`       | f32    |
//!
//! `--target base-q2..base-q8` narrows the bundle without touching the
//! checkpoint's quantized numbers: the attention projections are
//! RTN-quantized to the requested scheme — quant from full precision,
//! never quant-from-quant — while the MXFP4 expert stacks are still
//! transplanted verbatim. Embeddings and lm_head stay bf16: gpt-oss
//! embedding rows carry per-group outliers past ±17, and measured on the
//! PPL anchor even base-q8 groups cost +8% there. The router, norms and
//! biases stay full precision too (the router once set a whole run's
//! accuracy floor). Group size is the canonical one for the bit width,
//! dropped to the largest of 128/64/32 that divides the tensor's
//! in-features; a matrix nothing divides is carried bf16 with a note.
//!
//! The fused expert `gate_up_proj` keeps the checkpoint's row order
//! (gate = even rows, up = odd rows); the runtime's gpt-oss expert kernel
//! reads it that way, so nothing is permuted.
//!
//! MXFP4 payload layout (per tensor, `[n_experts, out, in]`): the packed
//! FP4 nibbles (`in/2` bytes per row, low nibble first — exactly the HF
//! `_blocks` bytes, `[E, out, in/32, 16]` flattened) followed at
//! `scale_offset` by one E8M0 byte per 32-value group (exactly the HF
//! `_scales` bytes). `group_size = 32`, `scale_dtype = e8m0`, no biases.

use anyhow::{bail, Context, Result};
use base_format::{
    AlignmentConfig, ComputeRegion, Header, HeaderFlags, LayerDescriptor, LayerKind,
    LayerPrecision, ModelConfig, QuantScheme, ResidencyHint, ScaleDtype, SourceInfo, TargetBackend,
    TensorDtype, TensorEntry, TensorFlags, TokenizerBlob,
};
use base_format::{BaseWriter, TensorPayload};
use base_readers::hf::HfDir;
use base_readers::safetensors::StDtype;
use serde_json::json;

use crate::QuantContext;

/// True when `config.json` describes a gpt-oss checkpoint this path handles.
pub(crate) fn is_gpt_oss(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(|v| v.as_str()) == Some("gpt_oss")
}

fn quant_method(config: &serde_json::Value) -> Option<String> {
    config
        .get("quantization_config")
        .and_then(|q| q.get("quant_method"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase())
}

fn entry(name: &str, dtype: TensorDtype, shape: Vec<u64>, len: u64, hot: bool) -> TensorEntry {
    TensorEntry {
        name: name.to_string(),
        dtype,
        shape,
        offset: 0,
        length: len,
        scale_offset: None,
        scale_length: None,
        bias_offset: None,
        bias_length: None,
        awq_scale_offset: None,
        awq_scale_length: None,
        group_size: None,
        layout: None,
        residency: Some(if hot {
            ResidencyHint::Hot
        } else {
            ResidencyHint::Warm
        }),
        compute_region: if hot {
            ComputeRegion::Gpu
        } else {
            ComputeRegion::Accelerator
        },
        scale_dtype: None,
        symmetric: false,
        flags: TensorFlags::empty(),
        checksum_xxh64: None,
        source_ggml_type: None,
    }
}

/// bf16 source bytes → f32 bytes (lossless widening).
fn bf16_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for c in bytes.chunks_exact(2) {
        let bits = (u16::from_le_bytes([c[0], c[1]]) as u32) << 16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn f16_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for c in bytes.chunks_exact(2) {
        let v = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// bf16/f16 source bytes → f32 values (for the RTN packer).
fn half_bytes_to_f32(bytes: &[u8], dtype: StDtype) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(match dtype {
            StDtype::Bf16 => f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16),
            _ => half::f16::from_le_bytes([c[0], c[1]]).to_f32(),
        });
    }
    out
}

/// The dense-quant request carried by `--target`: bit width + bundle
/// dtype + header scheme for base-q2..q8; None for the mirror targets.
/// Exact id of an added/special token from the checkpoint's `tokenizer.json`,
/// by literal content. Recovers the Harmony stop set for a checkpoint that
/// ships no `generation_config.json`; `None` when the file or the token is
/// absent, which the caller surfaces rather than silently accepting.
fn added_token_id(input: &std::path::Path, content: &str) -> Option<u32> {
    let bytes = std::fs::read(input.join("tokenizer.json")).ok()?;
    let tok: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    for t in tok.get("added_tokens")?.as_array()? {
        if t.get("content").and_then(|c| c.as_str()) == Some(content) {
            return t.get("id").and_then(|i| i.as_u64()).map(|v| v as u32);
        }
    }
    None
}

fn dense_quant(target: crate::TargetScheme) -> Option<(u32, TensorDtype, QuantScheme)> {
    use crate::TargetScheme as T;
    match target {
        T::BaseQ2 => Some((2, TensorDtype::BaseQ2, QuantScheme::BaseQ2)),
        T::BaseQ3 => Some((3, TensorDtype::BaseQ3, QuantScheme::BaseQ3)),
        T::BaseQ4 => Some((4, TensorDtype::BaseQ4, QuantScheme::BaseQ4)),
        T::BaseQ5 => Some((5, TensorDtype::BaseQ5, QuantScheme::BaseQ5)),
        T::BaseQ6 => Some((6, TensorDtype::BaseQ6, QuantScheme::BaseQ6)),
        T::BaseQ8 => Some((8, TensorDtype::BaseQ8, QuantScheme::BaseQ8)),
        _ => None,
    }
}

/// The canonical group size for the bit width, dropped to the largest of
/// 128/64/32 that divides `in_features` (groups must not straddle rows —
/// gpt-oss's hidden of 2880 rules out q8's canonical 128).
fn dense_group_size(bits: u32, in_features: u64) -> Option<u32> {
    let canonical = base_quant::rtn::RtnConfig::canonical(bits).group_size;
    [canonical, 64, 32]
        .into_iter()
        .find(|gs| *gs <= canonical && in_features % *gs as u64 == 0)
}

pub(crate) fn convert_gpt_oss(
    input: &std::path::Path,
    output: &std::path::Path,
    ctx: &QuantContext,
) -> Result<()> {
    use base_arch::hf_mapper_for_model_type;

    let hf = HfDir::open(input)?;
    let qm = quant_method(&hf.config);
    match qm.as_deref() {
        Some("mxfp4") => {}
        other => bail!(
            "gpt_oss: expected an MXFP4 checkpoint (quantization_config.quant_method = \"mxfp4\"), \
             found {:?} — only the MXFP4 transplant path is implemented",
            other
        ),
    }
    if ctx.profile.is_some() {
        bail!("gpt_oss: --profile is not applicable — the checkpoint's MXFP4 experts are transplanted verbatim (mirror policy)");
    }
    let dense = dense_quant(ctx.target);
    match (ctx.target, &dense) {
        (crate::TargetScheme::Mxfp4, _) | (crate::TargetScheme::Bf16, _) => {}
        (_, Some((bits, _, _))) => eprintln!(
            "  note:    --target {:?} quantizes the attention projections to {bits}-bit RTN; the MXFP4 \
             experts are transplanted verbatim and embed/lm_head/router stay full precision (outliers)",
            ctx.target
        ),
        (other, None) => eprintln!(
            "  note:    --target {:?} ignored for gpt_oss — the bundle mirrors the checkpoint (mxfp4 experts, bf16 dense)",
            other
        ),
    }

    let mapper = hf_mapper_for_model_type("gpt_oss").expect("gpt_oss mapper registered");
    let mut config = mapper.config_from_hf(&hf.config)?;
    // generation_config.json carries the full harmony stop set
    // (`<|return|>`, `<|endoftext|>`, `<|call|>`); config.json only names
    // the first. Merge the rest into `eos_token_ids` so the runtime stops
    // on every end-of-turn marker.
    let gc_path = input.join("generation_config.json");
    match std::fs::read(&gc_path) {
        Ok(bytes) => {
            // A file that exists but will not parse is a broken checkpoint, not
            // an absent one: silently dropping it produces a bundle that looks
            // fine and runs past `<|call|>`.
            let gc: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", gc_path.display()))?;
            let ids: Vec<u32> = match gc.get("eos_token_id") {
                Some(serde_json::Value::Number(n)) => {
                    n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
                }
                Some(serde_json::Value::Array(a)) => a
                    .iter()
                    .filter_map(|v| v.as_u64().map(|x| x as u32))
                    .collect(),
                _ => Vec::new(),
            };
            for id in ids {
                if id != config.eos_token_id && !config.eos_token_ids.contains(&id) {
                    config.eos_token_ids.push(id);
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No generation_config.json: config.json names only `<|return|>`,
            // so derive the rest of the Harmony stop set from the tokenizer.
            // Without `<|call|>` the runtime keeps generating after a finished
            // tool call until some other stop or the token limit.
            // Every marker must resolve. A missing tokenizer.json, a missing
            // added_tokens array, or an absent `<|call|>` would otherwise be
            // accepted silently and the converter would report success while
            // writing the same incomplete stop set this branch exists to
            // repair — the bundle then runs past finished tool calls.
            let mut missing: Vec<&str> = Vec::new();
            for marker in ["<|return|>", "<|call|>", "<|endoftext|>"] {
                match added_token_id(input, marker) {
                    Some(id) => {
                        if id != config.eos_token_id && !config.eos_token_ids.contains(&id) {
                            config.eos_token_ids.push(id);
                        }
                    }
                    None => missing.push(marker),
                }
            }
            if !missing.is_empty() {
                bail!(
                    "gpt-oss checkpoint has no generation_config.json and its tokenizer.json does not \
                     define the Harmony stop token(s) {missing:?}; the bundle would generate past a \
                     finished tool call. Supply generation_config.json or a complete tokenizer.json."
                );
            }
            eprintln!(
                "  note:    no generation_config.json — Harmony stops derived from the tokenizer ({:?})",
                config.eos_token_ids
            );
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", gc_path.display())),
    }
    let n_layers = config.num_hidden_layers as usize;
    let n_experts = config.num_experts as u64;
    let hidden = config.hidden_size as u64;
    let ffn = config.moe_intermediate_size as u64;
    eprintln!(
        "  arch:    gpt_oss (MXFP4 transplant) hidden={} layers={} heads={}/{} experts={} top_k={} ffn={} vocab={}",
        hidden,
        n_layers,
        config.num_attention_heads,
        config.num_kv_heads,
        n_experts,
        config.num_experts_per_tok,
        ffn,
        config.vocab_size
    );

    // ── header ───────────────────────────────────────────────────────
    let mut config_map = config.to_config_map();
    config_map.insert("model_type".into(), json!("gpt_oss"));
    let header = Header {
        schema: 1,
        arch: "gpt_oss".to_string(),
        // With a dense target the header names the requested scheme (the
        // per-tensor dtypes stay authoritative — experts remain mxfp4).
        quant_scheme: dense.as_ref().map_or(QuantScheme::Mxfp4, |(_, _, s)| *s),
        min_hw: "apple_m1".to_string(),
        created: crate::chrono_now(),
        base_rt_version: env!("CARGO_PKG_VERSION").to_string(),
        source: SourceInfo {
            // A safetensors checkpoint directory that the MLX reference loads
            // natively, converted under the mirror policy (quantized tensors
            // transplanted verbatim, the rest carried losslessly) — the same
            // contract as the MLX quantized-checkpoint path, hence its label.
            format: "mlx_safetensors".to_string(),
            sha256: "".to_string(),
            filename: input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
        },
        tokenizer: TokenizerBlob {
            fields: crate::tokenizer_from_hf(&hf),
        },
        config: ModelConfig { fields: config_map },
        metadata: Default::default(),
        target_backend: TargetBackend::Metal,
        quant_profile: String::new(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::QUANTIZED | HeaderFlags::HAS_MOE,
        layers: (0..n_layers)
            .map(|_| LayerDescriptor {
                kind: LayerKind::AttentionGqa,
                moe_n_experts: config.num_experts as u16,
                moe_n_active: config.num_experts_per_tok as u16,
                shared_attn_layer: None,
                compute_hint: Some(ComputeRegion::Accelerator),
                precision: LayerPrecision::default(),
            })
            .collect(),
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
        provenance: None, // filled below via the writer's header
    };

    // ── tensor plan ──────────────────────────────────────────────────
    // (bundle name, source tensor, kind)
    #[derive(Clone, Copy)]
    enum Kind {
        /// bf16 matrix carried verbatim as bf16.
        Matrix,
        /// bf16/f16 small tensor widened to f32 (norms, biases, sinks).
        Widen,
    }
    let mut plan: Vec<(String, String, Kind, bool)> = vec![
        (
            "embed_tokens.weight".into(),
            "model.embed_tokens.weight".into(),
            Kind::Matrix,
            true,
        ),
        (
            "lm_head.weight".into(),
            "lm_head.weight".into(),
            Kind::Matrix,
            false,
        ),
        (
            "final_norm.weight".into(),
            "model.norm.weight".into(),
            Kind::Widen,
            true,
        ),
    ];
    // (bundle name, blocks source, scales source, expected [E, out, in])
    let mut mx_plan: Vec<(String, String, String, [u64; 3])> = Vec::new();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}.");
        let b = format!("layers.{l}.");
        plan.push((
            format!("{b}input_norm.weight"),
            format!("{p}input_layernorm.weight"),
            Kind::Widen,
            true,
        ));
        plan.push((
            format!("{b}post_attn_norm.weight"),
            format!("{p}post_attention_layernorm.weight"),
            Kind::Widen,
            true,
        ));
        for proj in ["q", "k", "v", "o"] {
            plan.push((
                format!("{b}self_attn.{proj}_proj.weight"),
                format!("{p}self_attn.{proj}_proj.weight"),
                Kind::Matrix,
                false,
            ));
            plan.push((
                format!("{b}self_attn.{proj}_proj.bias"),
                format!("{p}self_attn.{proj}_proj.bias"),
                Kind::Widen,
                true,
            ));
        }
        plan.push((
            format!("{b}self_attn.sinks"),
            format!("{p}self_attn.sinks"),
            Kind::Widen,
            true,
        ));
        plan.push((
            format!("{b}mlp.router.weight"),
            format!("{p}mlp.router.weight"),
            Kind::Matrix,
            true,
        ));
        plan.push((
            format!("{b}mlp.router.bias"),
            format!("{p}mlp.router.bias"),
            Kind::Widen,
            true,
        ));
        plan.push((
            format!("{b}mlp.experts.gate_up_proj.bias"),
            format!("{p}mlp.experts.gate_up_proj_bias"),
            Kind::Widen,
            true,
        ));
        plan.push((
            format!("{b}mlp.experts.down_proj.bias"),
            format!("{p}mlp.experts.down_proj_bias"),
            Kind::Widen,
            true,
        ));
        mx_plan.push((
            format!("{b}mlp.experts.gate_up_proj.weight"),
            format!("{p}mlp.experts.gate_up_proj_blocks"),
            format!("{p}mlp.experts.gate_up_proj_scales"),
            [n_experts, 2 * ffn, hidden],
        ));
        mx_plan.push((
            format!("{b}mlp.experts.down_proj.weight"),
            format!("{p}mlp.experts.down_proj_blocks"),
            format!("{p}mlp.experts.down_proj_scales"),
            [n_experts, hidden, ffn],
        ));
    }

    // Coverage: every checkpoint tensor must be claimed by the plan.
    let mut claimed = std::collections::BTreeSet::new();
    for (_, src, _, _) in &plan {
        claimed.insert(src.clone());
    }
    for (_, blocks, scales, _) in &mx_plan {
        claimed.insert(blocks.clone());
        claimed.insert(scales.clone());
    }
    let unclaimed: Vec<String> = hf
        .tensor_names()
        .filter(|n| !claimed.contains(*n))
        .map(|s| s.to_string())
        .collect();
    if !unclaimed.is_empty() {
        bail!(
            "gpt_oss: {} checkpoint tensor(s) not understood by the converter (first: {:?}) — refusing to \
             produce a bundle that silently drops weights",
            unclaimed.len(),
            unclaimed.first()
        );
    }

    let mut writer = BaseWriter::create(output, header).context("create writer")?;
    let mut prov_tensors = serde_json::Map::new();

    let pb = indicatif::ProgressBar::new((plan.len() + mx_plan.len()) as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template("  transplanting [{bar:28}] {pos}/{len} {msg}")
            .expect("valid progress template")
            .progress_chars("=>-"),
    );

    // ── carried tensors ──────────────────────────────────────────────
    for (name, src, kind, hot) in &plan {
        pb.set_message(name.clone());
        let info = hf
            .tensor_info(src)
            .with_context(|| format!("gpt_oss: checkpoint tensor {src} missing"))?
            .clone();
        let bytes = hf
            .tensor_bytes(src)
            .expect("tensor_bytes after tensor_info");
        // The router stays full precision under every target — it is
        // tiny, and quantizing it once set a whole run's accuracy floor.
        // The embedding table and lm_head stay bf16 too: gpt-oss embedding
        // rows carry per-group outliers past +-17, and measured on the
        // anchor even base-q8 groups cost +8% PPL there while the
        // attention projections quantize cleanly.
        let quantize = dense
            .as_ref()
            .filter(|_| {
                matches!(kind, Kind::Matrix)
                    && !name.ends_with("mlp.router.weight")
                    && name != "embed_tokens.weight"
                    && name != "lm_head.weight"
            })
            .and_then(|(bits, dtype, _)| {
                let in_f = *info.shape.last().unwrap_or(&0);
                match dense_group_size(*bits, in_f) {
                    Some(gs) => Some((*bits, *dtype, gs)),
                    None => {
                        eprintln!(
                            "    note: {name} in_features={in_f} fits no group size — carried bf16"
                        );
                        None
                    }
                }
            });
        if let Some((bits, qdtype, gs)) = quantize {
            let w = match info.dtype {
                StDtype::Bf16 | StDtype::F16 => half_bytes_to_f32(bytes, info.dtype),
                StDtype::F32 => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => bail!("gpt_oss: unexpected dtype {:?} for {src}", other),
            };
            let cfg = base_quant::RtnConfig {
                bits,
                group_size: gs,
                symmetric: false,
                scale_dtype: ScaleDtype::Bf16,
            };
            let packed = base_quant::rtn::pack(&w, cfg);
            let mut data = Vec::with_capacity(
                packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
            );
            data.extend_from_slice(&packed.packed_weights);
            let scale_off = data.len() as u64;
            data.extend_from_slice(&packed.scales);
            let bias_off = data.len() as u64;
            data.extend_from_slice(&packed.biases);
            let mut e = entry(name, qdtype, info.shape.clone(), data.len() as u64, *hot);
            e.scale_offset = Some(scale_off);
            e.scale_length = Some(packed.scales.len() as u64);
            if !packed.biases.is_empty() {
                e.bias_offset = Some(bias_off);
                e.bias_length = Some(packed.biases.len() as u64);
            }
            e.group_size = Some(gs);
            e.scale_dtype = Some(ScaleDtype::Bf16);
            writer.add_tensor(TensorPayload { entry: e, data });
            prov_tensors.insert(
                name.clone(),
                json!({ "src": [src], "quantized_to": format!("base_q{bits}"),
                        "transform": "rtn", "group_size": gs }),
            );
            pb.inc(1);
            continue;
        }
        let (data, dtype): (Vec<u8>, TensorDtype) = match (kind, info.dtype) {
            (Kind::Matrix, StDtype::Bf16) => (bytes.to_vec(), TensorDtype::Bf16),
            (Kind::Matrix, StDtype::F16) => (bytes.to_vec(), TensorDtype::F16),
            (Kind::Matrix, StDtype::F32) => (bytes.to_vec(), TensorDtype::F32),
            (Kind::Widen, StDtype::Bf16) => (bf16_to_f32_bytes(bytes), TensorDtype::F32),
            (Kind::Widen, StDtype::F16) => (f16_to_f32_bytes(bytes), TensorDtype::F32),
            (Kind::Widen, StDtype::F32) => (bytes.to_vec(), TensorDtype::F32),
            (_, other) => bail!("gpt_oss: unexpected dtype {:?} for {src}", other),
        };
        let e = entry(name, dtype, info.shape.clone(), data.len() as u64, *hot);
        writer.add_tensor(TensorPayload { entry: e, data });
        prov_tensors.insert(name.clone(), json!({ "src": [src] }));
        pb.inc(1);
    }

    // ── MXFP4 expert stacks (verbatim transplant) ────────────────────
    for (name, blocks_src, scales_src, dims) in &mx_plan {
        pb.set_message(name.clone());
        let binfo = hf
            .tensor_info(blocks_src)
            .with_context(|| format!("gpt_oss: checkpoint tensor {blocks_src} missing"))?
            .clone();
        let sinfo = hf
            .tensor_info(scales_src)
            .with_context(|| format!("gpt_oss: checkpoint tensor {scales_src} missing"))?
            .clone();
        if binfo.dtype != StDtype::U8 || sinfo.dtype != StDtype::U8 {
            bail!("gpt_oss: {blocks_src}/{scales_src} must be U8 (MXFP4 blocks + E8M0 scales)");
        }
        let [e, n, k] = *dims;
        let groups = k / 32;
        let want_blocks = vec![e, n, groups, 16];
        let want_scales = vec![e, n, groups];
        if binfo.shape != want_blocks {
            bail!(
                "gpt_oss: {blocks_src} shape {:?}, expected {:?}",
                binfo.shape,
                want_blocks
            );
        }
        if sinfo.shape != want_scales {
            bail!(
                "gpt_oss: {scales_src} shape {:?}, expected {:?}",
                sinfo.shape,
                want_scales
            );
        }
        let blocks = hf.tensor_bytes(blocks_src).expect("blocks bytes");
        let scales = hf.tensor_bytes(scales_src).expect("scales bytes");
        let mut data = Vec::with_capacity(blocks.len() + scales.len());
        data.extend_from_slice(blocks);
        let scale_off = data.len() as u64;
        data.extend_from_slice(scales);
        let mut te = entry(
            name,
            TensorDtype::Mxfp4,
            vec![e, n, k],
            data.len() as u64,
            false,
        );
        te.scale_offset = Some(scale_off);
        te.scale_length = Some(scales.len() as u64);
        te.group_size = Some(32);
        te.scale_dtype = Some(ScaleDtype::E8m0);
        te.symmetric = true;
        writer.add_tensor(TensorPayload { entry: te, data });
        // The tensor is the checkpoint's expert stack copied verbatim: FP4
        // codes from `_blocks`, E8M0 block exponents from `_scales`. Both
        // sources are listed; the stack note says how many experts it holds.
        prov_tensors.insert(
            name.clone(),
            json!({
                "src": [blocks_src, scales_src],
                "transplanted_mxfp4": true,
                "stack": { "pattern": blocks_src, "count": e },
                "scales": scales_src,
            }),
        );
        pb.inc(1);
    }
    pb.finish_and_clear();

    writer.set_provenance(json!({
        "tensors": prov_tensors,
        "dropped": [],
        "policy": match &dense {
            Some((bits, _, _)) => format!(
                "mirror + dense target: checkpoint-quantized tensors transplanted verbatim (mxfp4 codes + e8m0 scales); \
                 attention projections RTN-quantized to base_q{bits} (per-tensor `quantized_to` entries); \
                 embed/lm_head/router/norms/biases carried full precision"),
            None => "mirror: checkpoint-quantized tensors transplanted verbatim (mxfp4 codes + e8m0 scales), \
                     unquantized tensors carried losslessly (bf16 matrices as bf16, 1-D/bias tensors widened to f32)".to_string(),
        },
    }));
    eprintln!(
        "  wrote {} carried + {} mxfp4 expert tensors",
        plan.len(),
        mx_plan.len()
    );
    writer.finish().context("finish")?;
    Ok(())
}
