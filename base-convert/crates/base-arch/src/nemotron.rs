//! Nemotron-H GGUF → canonical `.base` mapping.
//!
//! Covers the hybrid Mamba-2 + attention decoder in both its MoE form
//! (`nemotron_h_moe`, e.g. Nemotron 3 Nano 30B-A3B) and the dense form
//! (`nemotron_h`). Unlike a standard transformer, each block has exactly
//! ONE mixer: a Mamba-2 SSM, a GQA attention, or an (MoE) FFN. The GGUF
//! metadata encodes the schedule via per-layer arrays:
//!
//!   `{arch}.attention.head_count_kv` — nonzero ⇒ attention block
//!   `{arch}.feed_forward_length`     — nonzero ⇒ FFN (MoE) block
//!   both zero                        ⇒ Mamba-2 block
//!
//! The schedule is emitted as `layer_types` ("mamba" | "attention" |
//! "moe" | "mlp") in the `.base` header config, following the Qwen3.5
//! hybrid precedent of flat config scalars + a `layer_types` array.
//!
//! MoE FFN blocks use DeepSeek-style routing: sigmoid scores + a
//! selection-only correction bias (`exp_probs_b.bias`), top-k with
//! renormalization (`expert_weights_norm`) and a routed scaling factor
//! (`expert_weights_scale`), plus one always-on shared expert. The FFN
//! itself is squared-ReLU up/down — there is no `ffn_gate` tensor.

use crate::{ArchConfig, GgufMapper};
use anyhow::{Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct NemotronHMapper;

fn u32_array(m: &BTreeMap<String, KvValue>, k: &str) -> Option<Vec<u32>> {
    match m.get(k)? {
        KvValue::Array(a) => Some(
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|n| n as u32)
                .collect(),
        ),
        _ => None,
    }
}

impl GgufMapper for NemotronHMapper {
    fn canonical_arch(&self) -> &'static str {
        "nemotron_h_moe"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        let prefix = if m.keys().any(|k| k.starts_with("nemotron_h_moe.")) {
            "nemotron_h_moe"
        } else {
            "nemotron_h"
        };

        let u32_key = |k: &str| {
            m.get(&format!("{prefix}.{k}"))
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("missing metadata key: {prefix}.{k}"))
        };
        let f32_key = |k: &str| m.get(&format!("{prefix}.{k}")).and_then(|v| v.as_f32());
        let bool_key = |k: &str| {
            m.get(&format!("{prefix}.{k}")).and_then(|v| match v {
                KvValue::Bool(b) => Some(*b),
                _ => None,
            })
        };

        let hidden_size = u32_key("embedding_length")?;
        let num_hidden_layers = u32_key("block_count")?;
        let num_attention_heads = u32_key("attention.head_count")?;

        // Per-layer arrays encode the block schedule (see module doc).
        // `head_count_kv` is an array on hybrid checkpoints; tolerate a
        // scalar for hypothetical homogeneous ones.
        let kv_per_layer = u32_array(m, &format!("{prefix}.attention.head_count_kv"))
            .unwrap_or_else(|| {
                let scalar = u32_key("attention.head_count_kv").unwrap_or(num_attention_heads);
                vec![scalar; num_hidden_layers as usize]
            });
        let ffn_per_layer =
            u32_array(m, &format!("{prefix}.feed_forward_length")).unwrap_or_else(|| {
                let scalar = u32_key("feed_forward_length").unwrap_or(0);
                vec![scalar; num_hidden_layers as usize]
            });
        let num_kv_heads = kv_per_layer.iter().copied().max().unwrap_or(0);

        // MoE topology.
        let num_experts = u32_key("expert_count").unwrap_or(0);
        let num_experts_per_tok = u32_key("expert_used_count").unwrap_or(0);
        let moe_intermediate_size = u32_key("expert_feed_forward_length").unwrap_or(0);
        let num_shared_experts = u32_key("expert_shared_count").unwrap_or(0);

        let layer_types: Vec<String> = (0..num_hidden_layers as usize)
            .map(|i| {
                if kv_per_layer.get(i).copied().unwrap_or(0) > 0 {
                    "attention"
                } else if ffn_per_layer.get(i).copied().unwrap_or(0) > 0 {
                    if num_experts > 0 {
                        "moe"
                    } else {
                        "mlp"
                    }
                } else {
                    "mamba"
                }
                .to_string()
            })
            .collect();

        // Dense-slot FFN width: the shared expert on MoE checkpoints
        // (Qwen3.5-MoE precedent: `intermediate_size` = shared/dense
        // width, `moe_intermediate_size` = per-routed-expert width);
        // the widest per-layer FFN otherwise.
        let intermediate_size = u32_key("expert_shared_feed_forward_length")
            .ok()
            .filter(|&v| v > 0)
            .or_else(|| ffn_per_layer.iter().copied().max().filter(|&v| v > 0))
            .context("neither expert_shared_feed_forward_length nor feed_forward_length set")?;

        let vocab_size = u32_key("vocab_size").or_else(|_| {
            m.get("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    KvValue::Array(a) => Some(a.len() as u32),
                    _ => None,
                })
                .context("no vocab_size and no tokenizer.ggml.tokens")
        })?;
        let head_dim = u32_key("attention.key_length").unwrap_or(hidden_size / num_attention_heads);

        let rope_theta = f32_key("rope.freq_base").unwrap_or(10_000.0);
        let rope_scale = f32_key("rope.scaling.factor").unwrap_or(1.0);
        let rms_norm_eps = f32_key("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        // Partial RoPE on the attention blocks: only the first
        // `rope.dimension_count` of `head_dim` dims rotate (Nemotron 3
        // Nano: 84 / 128 = 0.65625).
        let partial_rotary_factor = match u32_key("rope.dimension_count") {
            Ok(d) if d > 0 && d < head_dim => d as f32 / head_dim as f32,
            _ => 0.0,
        };
        let max_position_embeddings = u32_key("context_length").unwrap_or(0);

        // Mamba-2 mixer geometry. `time_step_rank` carries the SSM head
        // count on Mamba-2 GGUFs (llama.cpp convention).
        let ssm_state_size = u32_key("ssm.state_size").unwrap_or(0);
        let ssm_conv_kernel = u32_key("ssm.conv_kernel").unwrap_or(0);
        let ssm_num_groups = u32_key("ssm.group_count").unwrap_or(0);
        let ssm_inner_size = u32_key("ssm.inner_size").unwrap_or(0);
        let ssm_num_heads = u32_key("ssm.time_step_rank").unwrap_or(0);

        Ok(ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta,
            rope_scale,
            rms_norm_eps,
            partial_rotary_factor,
            max_position_embeddings,
            tie_word_embeddings: false,
            n_kv_heads_per_layer: kv_per_layer,
            layer_types,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            num_shared_experts,
            // Sigmoid routing with renormalized top-k and routed scaling
            // (DeepSeek-style). `expert_weights_norm` defaults true for
            // this family.
            norm_topk_prob: bool_key("expert_weights_norm").unwrap_or(true),
            expert_gating: if num_experts > 0 { 1 } else { 0 },
            expert_weights_scale: f32_key("expert_weights_scale").unwrap_or(0.0),
            ssm_state_size,
            ssm_conv_kernel,
            ssm_num_groups,
            ssm_inner_size,
            ssm_num_heads,
            ..ArchConfig::default()
        })
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        // Router selection-bias (DeepSeek-style `e_score_correction_bias`)
        // — not part of the shared llama-style table.
        if let Some(rest) = n.strip_prefix("blk.") {
            if let Some((layer_str, suffix)) = rest.split_once('.') {
                if suffix == "exp_probs_b.bias" {
                    let layer: u32 = layer_str.parse().ok()?;
                    return Some(format!("layers.{layer}.mlp.router.e_score_correction_bias"));
                }
            }
        }
        crate::llama::map_llama_style(n)
    }
}

/// Nemotron-H HF / MLX-safetensors → canonical `.base` mapping.
///
/// The HF checkpoint names everything under `backbone.` and gives every
/// block the same `mixer.` prefix regardless of what the mixer *is* —
/// the block schedule lives in `hybrid_override_pattern`, not in the
/// tensor names. So `mixer.in_proj` (Mamba), `mixer.q_proj` (attention)
/// and `mixer.gate` (MoE router) are siblings, and the rename table
/// below is what splits them back into the canonical `ssm.*`,
/// `self_attn.*` and `mlp.*` families the runtime expects.
///
/// Targets exactly the names the GGUF mapper emits, so a bundle
/// converted from an MLX checkpoint and one converted from the GGUF are
/// interchangeable as far as the runtime is concerned.
pub struct NemotronHHfMapper;

/// HF/MLX tensor name → canonical `.base` name. `None` drops the tensor.
pub fn nemotron_hf_rename(name: &str) -> Option<String> {
    match name {
        "backbone.embeddings.weight" => return Some("embed_tokens.weight".into()),
        "backbone.norm_f.weight" => return Some("final_norm.weight".into()),
        "lm_head.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }

    let rest = name.strip_prefix("backbone.layers.")?;
    let (layer_str, tail) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;

    // Every block's pre-mixer norm. Named `norm.weight` directly under
    // the layer — distinct from `mixer.norm.weight`, which is the
    // Mamba-2 grouped gated norm *inside* the SSM mixer.
    if tail == "norm.weight" {
        return Some(format!("layers.{layer}.input_norm.weight"));
    }

    let mixer = tail.strip_prefix("mixer.")?;
    let canonical_tail = match mixer {
        // Attention blocks.
        "q_proj.weight" => "self_attn.q_proj.weight",
        "k_proj.weight" => "self_attn.k_proj.weight",
        "v_proj.weight" => "self_attn.v_proj.weight",
        "o_proj.weight" => "self_attn.o_proj.weight",

        // Mamba-2 blocks.
        "in_proj.weight" => "ssm.in_proj.weight",
        "out_proj.weight" => "ssm.out_proj.weight",
        "conv1d.weight" => "ssm.conv1d.weight",
        "conv1d.bias" => "ssm.conv1d.bias",
        "A_log" => "ssm.a_log",
        "D" => "ssm.d",
        "dt_bias" => "ssm.dt_bias",
        // The grouped RMSNorm applied to the scan output.
        "norm.weight" => "ssm.norm.weight",

        // MoE blocks. HF calls the router `gate`; the runtime's `mlp.router`
        // is the same matrix. `switch_mlp.fc1/fc2` are MLX's stacked
        // per-expert projections — this family has no gate projection
        // (squared-ReLU, not SwiGLU), so fc1/fc2 are up/down.
        "gate.weight" => "mlp.router.weight",
        "gate.e_score_correction_bias" => "mlp.router.e_score_correction_bias",
        "switch_mlp.fc1.weight" => "mlp.experts.up_proj.weight",
        "switch_mlp.fc2.weight" => "mlp.experts.down_proj.weight",
        "shared_experts.up_proj.weight" => "mlp.shared_expert.up_proj.weight",
        "shared_experts.down_proj.weight" => "mlp.shared_expert.down_proj.weight",

        // NVFP4 checkpoints (per-expert tensors arrive pre-stacked as the
        // virtual `experts.<proj>` names). The fp4 code bytes + e4m3 block
        // scales are transplanted into the `.weight` tensor; the global
        // scale and the calibrated activation scale ride along as f32
        // sidecar tensors.
        "experts.up_proj.weight" => "mlp.experts.up_proj.weight",
        "experts.down_proj.weight" => "mlp.experts.down_proj.weight",
        "experts.up_proj.weight_scale_2" => "mlp.experts.up_proj.weight_scale_2",
        "experts.down_proj.weight_scale_2" => "mlp.experts.down_proj.weight_scale_2",
        "experts.up_proj.input_scale" => "mlp.experts.up_proj.input_scale",
        "experts.down_proj.input_scale" => "mlp.experts.down_proj.input_scale",
        "in_proj.weight_scale_2" => "ssm.in_proj.weight_scale_2",
        "in_proj.input_scale" => "ssm.in_proj.input_scale",
        "out_proj.weight_scale_2" => "ssm.out_proj.weight_scale_2",
        "out_proj.input_scale" => "ssm.out_proj.input_scale",
        "shared_experts.up_proj.weight_scale_2" => "mlp.shared_expert.up_proj.weight_scale_2",
        "shared_experts.down_proj.weight_scale_2" => "mlp.shared_expert.down_proj.weight_scale_2",
        "shared_experts.up_proj.input_scale" => "mlp.shared_expert.up_proj.input_scale",
        "shared_experts.down_proj.input_scale" => "mlp.shared_expert.down_proj.input_scale",

        _ => return None,
    };
    Some(format!("layers.{layer}.{canonical_tail}"))
}

impl crate::HfMapper for NemotronHHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "nemotron_h_moe"
    }

    fn value_transform(&self, canonical: &str) -> Option<crate::ValueTransform> {
        // The HF checkpoint stores the Mamba-2 state-transition matrix
        // as `A_log`; `modeling_nemotron_h.py` computes
        // `A = -exp(A_log)` on every forward. The GGUF conversion bakes
        // that in, and the runtime's scan kernel consumes `A` directly
        // — it expects negative values, since `dA = exp(dt * A)` has to
        // decay. Copying `A_log` through verbatim leaves the scan with
        // positive transitions and the model emits one token forever.
        if canonical.ends_with(".ssm.a_log") {
            Some(crate::ValueTransform::NegExp)
        } else {
            None
        }
    }

    fn shape_fastest_first(&self) -> bool {
        // Match the GGUF mapper's `ne`-style shape notation ([in, out])
        // so both conversion paths describe this architecture the same
        // way. The bytes are identical either way — HF C-order
        // [out, in] and GGUF [in, out] are the same buffer, in
        // contiguous along the reduction dim — this only fixes how the
        // header *reports* it.
        true
    }

    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        let u32_key = |k: &str| c.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| c.get(k).and_then(|v| v.as_f64()).map(|n| n as f32);
        let req = |k: &str| u32_key(k).with_context(|| format!("config.json missing {k}"));

        let hidden_size = req("hidden_size")?;
        let num_hidden_layers = req("num_hidden_layers")?;
        let num_attention_heads = req("num_attention_heads")?;
        let num_kv_heads = u32_key("num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim = u32_key("head_dim").unwrap_or(hidden_size / num_attention_heads);

        // The block schedule. `hybrid_override_pattern` is one character
        // per layer: M = Mamba-2, * = attention, E/- = the FFN slot.
        let pattern = c
            .get("hybrid_override_pattern")
            .and_then(|v| v.as_str())
            .context("config.json missing hybrid_override_pattern (the block schedule)")?;
        if pattern.chars().count() != num_hidden_layers as usize {
            anyhow::bail!(
                "hybrid_override_pattern has {} entries but num_hidden_layers is {}",
                pattern.chars().count(),
                num_hidden_layers
            );
        }
        let num_experts = u32_key("n_routed_experts").unwrap_or(0);
        let layer_types: Vec<String> = pattern
            .chars()
            .map(|ch| {
                match ch {
                    'M' => "mamba",
                    '*' => "attention",
                    _ if num_experts > 0 => "moe",
                    _ => "mlp",
                }
                .to_string()
            })
            .collect();
        let n_kv_heads_per_layer: Vec<u32> = layer_types
            .iter()
            .map(|t| if t == "attention" { num_kv_heads } else { 0 })
            .collect();

        // `intermediate_size` is the *routed* expert width on this
        // checkpoint; the dense-slot width the runtime wants is the
        // shared expert's (Qwen3.5-MoE precedent, and what the GGUF
        // mapper reads out of expert_shared_feed_forward_length).
        let moe_intermediate_size = u32_key("moe_intermediate_size")
            .or_else(|| u32_key("intermediate_size"))
            .unwrap_or(0);
        let intermediate_size = u32_key("moe_shared_expert_intermediate_size")
            .or_else(|| u32_key("intermediate_size"))
            .context("neither moe_shared_expert_intermediate_size nor intermediate_size set")?;

        // Mamba-2 mixer geometry. `n_groups` is the SSM group count;
        // `n_group` (singular) is the DeepSeek routing group count and
        // is a different thing entirely — reading the wrong one gives 1
        // group instead of 8 and silently mis-shapes the scan.
        let ssm_num_heads = u32_key("mamba_num_heads").unwrap_or(0);
        let mamba_head_dim = u32_key("mamba_head_dim").unwrap_or(0);

        Ok(ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size: req("vocab_size")?,
            rope_theta: f32_key("rope_theta").unwrap_or(10_000.0),
            rope_scale: 1.0,
            rms_norm_eps: f32_key("norm_eps")
                .or_else(|| f32_key("layer_norm_epsilon"))
                .unwrap_or(1e-5),
            // Nemotron-H attention is NoPE — llama.cpp maps this
            // architecture to rope_type NONE regardless of the rope keys
            // the config carries, and the reference generations match
            // that. Recorded for completeness; the runtime does not
            // rotate these blocks.
            partial_rotary_factor: f32_key("partial_rotary_factor").unwrap_or(0.0),
            max_position_embeddings: u32_key("max_position_embeddings").unwrap_or(0),
            tie_word_embeddings: c
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            n_kv_heads_per_layer,
            layer_types,
            num_experts,
            num_experts_per_tok: u32_key("num_experts_per_tok").unwrap_or(0),
            moe_intermediate_size,
            num_shared_experts: u32_key("n_shared_experts").unwrap_or(0),
            norm_topk_prob: c
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            expert_gating: if num_experts > 0 { 1 } else { 0 },
            expert_weights_scale: f32_key("routed_scaling_factor").unwrap_or(0.0),
            ssm_state_size: u32_key("ssm_state_size").unwrap_or(0),
            ssm_conv_kernel: u32_key("conv_kernel").unwrap_or(0),
            ssm_num_groups: u32_key("n_groups").unwrap_or(0),
            ssm_inner_size: ssm_num_heads * mamba_head_dim,
            ssm_num_heads,
            ..ArchConfig::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic metadata mirroring the real Nemotron-3-Nano-30B-A3B
    /// GGUF (52 blocks: 23 Mamba-2, 23 MoE FFN, 6 attention; per-layer
    /// arrays carry the schedule).
    fn nano_metadata() -> BTreeMap<String, KvValue> {
        let mut m = BTreeMap::new();
        let p = "nemotron_h_moe";
        let u = KvValue::U32;
        m.insert("general.architecture".into(), KvValue::String(p.into()));
        m.insert(format!("{p}.block_count"), u(8));
        m.insert(format!("{p}.embedding_length"), u(2688));
        m.insert(format!("{p}.attention.head_count"), u(32));
        m.insert(format!("{p}.attention.key_length"), u(128));
        m.insert(format!("{p}.attention.value_length"), u(128));
        m.insert(
            format!("{p}.attention.head_count_kv"),
            KvValue::Array(vec![u(0), u(0), u(2), u(0), u(0), u(0), u(2), u(0)]),
        );
        m.insert(
            format!("{p}.feed_forward_length"),
            KvValue::Array(vec![
                u(0),
                u(1856),
                u(0),
                u(1856),
                u(0),
                u(1856),
                u(0),
                u(1856),
            ]),
        );
        m.insert(format!("{p}.vocab_size"), u(131072));
        m.insert(format!("{p}.context_length"), u(1048576));
        m.insert(format!("{p}.rope.freq_base"), KvValue::F32(10000.0));
        m.insert(format!("{p}.rope.dimension_count"), u(84));
        m.insert(
            format!("{p}.attention.layer_norm_rms_epsilon"),
            KvValue::F32(1e-5),
        );
        m.insert(format!("{p}.expert_count"), u(128));
        m.insert(format!("{p}.expert_used_count"), u(6));
        m.insert(format!("{p}.expert_feed_forward_length"), u(1856));
        m.insert(format!("{p}.expert_shared_feed_forward_length"), u(3712));
        m.insert(format!("{p}.expert_shared_count"), u(1));
        m.insert(format!("{p}.expert_weights_norm"), KvValue::Bool(true));
        m.insert(format!("{p}.expert_weights_scale"), KvValue::F32(2.5));
        m.insert(format!("{p}.ssm.conv_kernel"), u(4));
        m.insert(format!("{p}.ssm.state_size"), u(128));
        m.insert(format!("{p}.ssm.group_count"), u(8));
        m.insert(format!("{p}.ssm.inner_size"), u(4096));
        m.insert(format!("{p}.ssm.time_step_rank"), u(64));
        m
    }

    #[test]
    fn nano_config_from_gguf() {
        let c = NemotronHMapper.config_from_gguf(&nano_metadata()).unwrap();
        assert_eq!(c.hidden_size, 2688);
        assert_eq!(c.num_hidden_layers, 8);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_kv_heads, 2, "max of the per-layer kv-head array");
        assert_eq!(c.head_dim, 128);
        assert_eq!(
            c.intermediate_size, 3712,
            "dense slot = shared-expert width on MoE checkpoints"
        );
        assert_eq!(c.moe_intermediate_size, 1856);
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.num_experts_per_tok, 6);
        assert_eq!(c.num_shared_experts, 1);
        assert!(c.norm_topk_prob);
        assert_eq!(c.expert_gating, 1, "sigmoid routing");
        assert_eq!(c.expert_weights_scale, 2.5);
        assert_eq!(
            c.layer_types,
            vec![
                "mamba",
                "moe",
                "attention",
                "moe",
                "mamba",
                "moe",
                "attention",
                "moe"
            ]
        );
        assert_eq!(
            c.n_kv_heads_per_layer,
            vec![0, 0, 2, 0, 0, 0, 2, 0],
            "schedule array preserved verbatim"
        );
        assert_eq!(c.ssm_state_size, 128);
        assert_eq!(c.ssm_conv_kernel, 4);
        assert_eq!(c.ssm_num_groups, 8);
        assert_eq!(c.ssm_inner_size, 4096);
        assert_eq!(c.ssm_num_heads, 64);
        assert!((c.partial_rotary_factor - 0.65625).abs() < 1e-6);
        assert_eq!(c.max_position_embeddings, 1048576);
    }

    #[test]
    fn config_map_carries_ssm_and_routing_keys() {
        let c = NemotronHMapper.config_from_gguf(&nano_metadata()).unwrap();
        let map = c.to_config_map();
        assert_eq!(map["ssm_state_size"], serde_json::json!(128));
        assert_eq!(map["ssm_conv_kernel"], serde_json::json!(4));
        assert_eq!(map["ssm_num_groups"], serde_json::json!(8));
        assert_eq!(map["ssm_inner_size"], serde_json::json!(4096));
        assert_eq!(map["ssm_num_heads"], serde_json::json!(64));
        assert_eq!(map["expert_gating"], serde_json::json!(1));
        assert_eq!(map["expert_weights_scale"], serde_json::json!(2.5));
        assert_eq!(map["num_shared_experts"], serde_json::json!(1));
        assert!(map.contains_key("layer_types"));
        assert!(map.contains_key("n_kv_heads_per_layer"));
    }

    /// Every tensor pattern present in the real Nano GGUF must map — an
    /// unmapped name is silently dropped by convert_gguf, so this list is
    /// the converter-side completeness gate.
    #[test]
    fn nano_tensor_names_all_map() {
        let cases = [
            ("token_embd.weight", "embed_tokens.weight"),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "final_norm.weight"),
            ("blk.0.attn_norm.weight", "layers.0.input_norm.weight"),
            ("blk.5.attn_q.weight", "layers.5.self_attn.q_proj.weight"),
            ("blk.5.attn_k.weight", "layers.5.self_attn.k_proj.weight"),
            ("blk.5.attn_v.weight", "layers.5.self_attn.v_proj.weight"),
            (
                "blk.5.attn_output.weight",
                "layers.5.self_attn.o_proj.weight",
            ),
            (
                "blk.1.ffn_up_exps.weight",
                "layers.1.mlp.experts.up_proj.weight",
            ),
            (
                "blk.1.ffn_down_exps.weight",
                "layers.1.mlp.experts.down_proj.weight",
            ),
            ("blk.1.ffn_gate_inp.weight", "layers.1.mlp.router.weight"),
            (
                "blk.1.exp_probs_b.bias",
                "layers.1.mlp.router.e_score_correction_bias",
            ),
            (
                "blk.1.ffn_up_shexp.weight",
                "layers.1.mlp.shared_expert.up_proj.weight",
            ),
            (
                "blk.1.ffn_down_shexp.weight",
                "layers.1.mlp.shared_expert.down_proj.weight",
            ),
            ("blk.0.ssm_in.weight", "layers.0.ssm.in_proj.weight"),
            ("blk.0.ssm_out.weight", "layers.0.ssm.out_proj.weight"),
            ("blk.0.ssm_conv1d.weight", "layers.0.ssm.conv1d.weight"),
            ("blk.0.ssm_conv1d.bias", "layers.0.ssm.conv1d.bias"),
            ("blk.0.ssm_dt.bias", "layers.0.ssm.dt_bias"),
            ("blk.0.ssm_a", "layers.0.ssm.a_log"),
            ("blk.0.ssm_d", "layers.0.ssm.d"),
            ("blk.0.ssm_norm.weight", "layers.0.ssm.norm.weight"),
        ];
        for (gguf, canonical) in cases {
            assert_eq!(
                NemotronHMapper.map_tensor_name(gguf).as_deref(),
                Some(canonical),
                "mapping for {gguf}"
            );
        }
    }
}
