//! `basert convert-lora`: PEFT adapter directory → adapter `.base` bundle
//! (capability-audit: the LoRA endpoints existed with no way to author the
//! adapter files they load).
//!
//! Reads `adapter_config.json` + `adapter_model.safetensors` and emits the
//! `lora.<target>.{A,B}` tensor pairs the runtime loader consumes
//! (`src/lora/lora_loader.cpp`), plus `lora.rank` / `lora.alpha` /
//! `lora.max_seq` metadata. B is stored RAW — the loader pre-scales it by
//! alpha/rank at load time.
//!
//! Targets are keyed by the RUNTIME DISPATCH names (`layers.N.attention.q
//! .weight`, `layers.N.ffn.gate.weight`, …) — what `dispatch_gemm` receives
//! and `apply_delta` looks up — not the converter-canonical bundle names.
//! Two families of GEMM exist per projection group:
//!
//!   - the FUSED path (qkv in one GEMM carrying the q name with
//!     N = q+k+v; gate+up in one GEMM carrying the gate name with
//!     N = 2*ffn), used by quantized decode/prefill; and
//!   - the SPLIT path (per-projection GEMMs), used by mixed-dtype configs.
//!
//! The two are mutually exclusive at dispatch time, so we emit BOTH forms:
//! stacked block-diagonal pairs for the fused names AND per-projection
//! pairs for the split names. Whichever path runs finds its entry; the
//! other idles. Stacking needs the true per-projection output widths, which
//! come from `--base <model.base>` (also validating that the adapter and
//! model agree on the hidden size).
//!
//! Scope: the Llama-shaped families (llama / mistral / phi3 — the runtime
//! names above). Other architectures refuse loudly rather than emitting
//! entries that would silently never match a dispatch name.

use anyhow::{bail, Context, Result};
use base_format::{Header, TensorDtype, TensorEntry, TensorPayload};
use base_readers::safetensors::{SafetensorsFile, StDtype};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
pub struct ConvertLoraArgs {
    /// PEFT adapter directory (adapter_config.json + adapter_model.safetensors).
    pub input: PathBuf,
    /// The `.base` model bundle this adapter patches. Supplies the fused-GEMM
    /// output widths for stacking and validates the hidden size.
    #[arg(long)]
    pub base: PathBuf,
    /// Output adapter `.base` file. Defaults to `<input>/adapter.base`.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// `lora.max_seq`: the largest per-dispatch row count the runtime's
    /// delta scratch is sized for (prefill chunks must fit).
    #[arg(long, default_value_t = 4096)]
    pub max_seq: u32,
}

/// One projection's A/B pair in f32 (row-major).
struct Pair {
    a: Vec<f32>, // [r, in]
    b: Vec<f32>, // [out, r]
    r: usize,
    d_in: usize,
    d_out: usize,
}

fn to_f32(st: &SafetensorsFile, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    let info = st
        .tensors
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("adapter tensor {name} missing"))?
        .clone();
    let bytes = st.tensor_bytes(&info);
    let n: usize = info.shape.iter().product::<u64>() as usize;
    let v = match info.dtype {
        StDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StDtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        StDtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        other => bail!("adapter tensor {name}: unsupported dtype {other:?}"),
    };
    let v: Vec<f32> = v;
    if v.len() != n {
        bail!("adapter tensor {name}: byte length disagrees with shape");
    }
    Ok((info.shape.clone(), v))
}

fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
    }
    out
}

/// Shapes of the base model's canonical projection tensors, per layer:
/// (name suffix → [out, in]).
type BaseInfo = (
    BTreeMap<String, Vec<u64>>,
    BTreeMap<String, serde_json::Value>,
);

fn base_shapes(base: &PathBuf) -> Result<BaseInfo> {
    let f = std::fs::read(base).with_context(|| format!("read {}", base.display()))?;
    if f.len() < 16 || &f[0..4] != b"BASE" {
        bail!("{} is not a .base bundle", base.display());
    }
    let hlen = u64::from_le_bytes(f[8..16].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&f[16..16 + hlen])?;
    let arch = h["arch"].as_str().unwrap_or("");
    if arch != "llama" {
        bail!(
            "convert-lora currently maps the Llama-shaped runtime dispatch names; the base model's \
             arch is {arch:?}. Refusing to emit entries that would silently never match."
        );
    }
    let mut m = BTreeMap::new();
    for t in h["tensors"]
        .as_array()
        .context(".base header has no tensors")?
    {
        let name = t["name"].as_str().unwrap_or("");
        if name.contains(".self_attn.") || name.contains(".mlp.") {
            let shape: Vec<u64> = t["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                .unwrap_or_default();
            m.insert(name.to_string(), shape);
        }
    }
    // The engine's weight store refuses a degenerate config (dim=0 …), so
    // the adapter carries the BASE model's core dims — which doubles as
    // provenance for which model this adapter patches.
    let mut cfg = BTreeMap::new();
    if let Some(c) = h["config"].as_object() {
        for k in [
            "architecture",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "intermediate_size",
        ] {
            if let Some(v) = c.get(k) {
                cfg.insert(k.to_string(), v.clone());
            }
        }
    }
    Ok((m, cfg))
}

/// Stack member pairs block-diagonally into one fused-target pair.
/// `members`: (pair, out_offset) in fused-row order; `total_out` covers the
/// full fused GEMM N so absent members contribute zero rows.
fn stack(members: &[(&Pair, usize)], total_out: usize, d_in: usize) -> Pair {
    let eff_r: usize = members.iter().map(|(p, _)| p.r).sum();
    let mut a = vec![0f32; eff_r * d_in];
    let mut b = vec![0f32; total_out * eff_r];
    let mut r_off = 0usize;
    for (p, out_off) in members {
        for r in 0..p.r {
            a[(r_off + r) * d_in..(r_off + r + 1) * d_in]
                .copy_from_slice(&p.a[r * d_in..(r + 1) * d_in]);
        }
        for o in 0..p.d_out {
            for r in 0..p.r {
                b[(out_off + o) * eff_r + (r_off + r)] = p.b[o * p.r + r];
            }
        }
        r_off += p.r;
    }
    Pair {
        a,
        b,
        r: eff_r,
        d_in,
        d_out: total_out,
    }
}

pub fn cmd_convert_lora(args: ConvertLoraArgs) -> Result<()> {
    let cfg_path = args.input.join("adapter_config.json");
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?,
    )?;
    let rank = cfg["r"].as_u64().context("adapter_config.json missing r")? as usize;
    let alpha = cfg["lora_alpha"]
        .as_f64()
        .context("adapter_config.json missing lora_alpha")? as f32;
    let st = SafetensorsFile::open(args.input.join("adapter_model.safetensors"))?;
    let (shapes, base_cfg) = base_shapes(&args.base)?;
    let out = args
        .output
        .clone()
        .unwrap_or_else(|| args.input.join("adapter.base"));

    // Collect per-(layer, projection) pairs from the PEFT names.
    // `base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`.
    let mut pairs: BTreeMap<(u32, String), Pair> = BTreeMap::new();
    let names: Vec<String> = st.tensors.iter().map(|t| t.name.clone()).collect();
    for name in &names {
        let Some(rest) = name.split(".layers.").nth(1) else {
            continue;
        };
        if !name.ends_with(".lora_A.weight") {
            continue;
        }
        let mut it = rest.splitn(2, '.');
        let layer: u32 = it.next().unwrap_or("").parse().context("layer index")?;
        let proj_path = it.next().unwrap_or(""); // "self_attn.q_proj.lora_A.weight"
        let proj = proj_path
            .trim_end_matches(".lora_A.weight")
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_string(); // "q_proj"
        let b_name = name.replace(".lora_A.weight", ".lora_B.weight");
        let (a_shape, a) = to_f32(&st, name)?;
        let (b_shape, b) = to_f32(&st, &b_name)?;
        if a_shape.len() != 2
            || b_shape.len() != 2
            || a_shape[0] as usize != rank
            || b_shape[1] as usize != rank
        {
            bail!(
                "{name}: expected A [r,in] / B [out,r] with r={rank}, got {a_shape:?}/{b_shape:?}"
            );
        }
        pairs.insert(
            (layer, proj),
            Pair {
                a,
                b,
                r: rank,
                d_in: a_shape[1] as usize,
                d_out: b_shape[0] as usize,
            },
        );
    }
    if pairs.is_empty() {
        bail!("no lora_A/lora_B pairs found in adapter_model.safetensors");
    }

    // Validate against the base model and emit runtime-target pairs.
    let layers: std::collections::BTreeSet<u32> = pairs.keys().map(|(l, _)| *l).collect();
    let mut targets: Vec<(String, Pair)> = Vec::new();
    let shape_of = |layer: u32, canon: &str| -> Option<(usize, usize)> {
        shapes
            .get(&format!("layers.{layer}.{canon}.weight"))
            .and_then(|s| (s.len() == 2).then(|| (s[0] as usize, s[1] as usize)))
    };
    for layer in &layers {
        let get = |p: &str| pairs.get(&(*layer, p.to_string()));
        // Per-projection sanity + split-path entries.
        let singles = [
            ("q_proj", "attention.q"),
            ("k_proj", "attention.k"),
            ("v_proj", "attention.v"),
            ("o_proj", "attention.output"),
            ("gate_proj", "ffn.gate"),
            ("up_proj", "ffn.up"),
            ("down_proj", "ffn.down"),
        ];
        for (peft, runtime) in singles {
            let Some(p) = get(peft) else { continue };
            let canon = match peft {
                "o_proj" => "self_attn.o_proj".to_string(),
                "q_proj" | "k_proj" | "v_proj" => format!("self_attn.{peft}"),
                _ => format!("mlp.{peft}"),
            };
            if let Some((out_n, in_k)) = shape_of(*layer, &canon) {
                if p.d_in != in_k || p.d_out != out_n {
                    bail!(
                        "layer {layer} {peft}: adapter is [{}x{}] but the base tensor is [{out_n}x{in_k}] — \
                         wrong-model adapter?",
                        p.d_out, p.d_in
                    );
                }
            }
            targets.push((
                format!("layers.{layer}.{runtime}.weight"),
                Pair {
                    a: p.a.clone(),
                    b: p.b.clone(),
                    r: p.r,
                    d_in: p.d_in,
                    d_out: p.d_out,
                },
            ));
        }
        // Fused qkv (dispatch name = the q name, N = q+k+v) — emitted only
        // when the fused dispatch name differs from the split one, which it
        // does not for q ("attention.q" carries both) … it DOES differ in N:
        // the fused GEMM's tensor_name is STILL "layers.N.attention.q.weight"
        // but with N = q+k+v, so the SAME key must carry the STACKED pair for
        // the fused path to apply all three deltas — and the split path's
        // q-GEMM then finds a [q+k+v, ..]-shaped B whose N mismatches and
        // (by the loader's shape guard) skips. One key cannot serve both.
        // The fused path is what every quantized bundle runs, so the stacked
        // pair WINS the key; the split-path singles above keep k/v/o and the
        // FFN entries live.
        let (q, k, v) = (get("q_proj"), get("k_proj"), get("v_proj"));
        if q.is_some() || k.is_some() || v.is_some() {
            let (qn, d) = shape_of(*layer, "self_attn.q_proj")
                .with_context(|| format!("layer {layer}: base has no q_proj shape"))?;
            let (kn, _) = shape_of(*layer, "self_attn.k_proj")
                .with_context(|| format!("layer {layer}: base has no k_proj shape"))?;
            let (vn, _) = shape_of(*layer, "self_attn.v_proj")
                .with_context(|| format!("layer {layer}: base has no v_proj shape"))?;
            let mut members: Vec<(&Pair, usize)> = Vec::new();
            if let Some(p) = q {
                members.push((p, 0));
            }
            if let Some(p) = k {
                members.push((p, qn));
            }
            if let Some(p) = v {
                members.push((p, qn + kn));
            }
            let stacked = stack(&members, qn + kn + vn, d);
            // Replace the single-q entry for this key (see comment above).
            targets.retain(|(n, _)| n != &format!("layers.{layer}.attention.q.weight"));
            targets.push((format!("layers.{layer}.attention.q.weight"), stacked));
        }
        // Fused gate+up (dispatch name = the gate name, N = 2*ffn).
        let (g, u) = (get("gate_proj"), get("up_proj"));
        if g.is_some() || u.is_some() {
            let (gn, d) = shape_of(*layer, "mlp.gate_proj")
                .with_context(|| format!("layer {layer}: base has no gate_proj shape"))?;
            let (un, _) = shape_of(*layer, "mlp.up_proj")
                .with_context(|| format!("layer {layer}: base has no up_proj shape"))?;
            let mut members: Vec<(&Pair, usize)> = Vec::new();
            if let Some(p) = g {
                members.push((p, 0));
            }
            if let Some(p) = u {
                members.push((p, gn));
            }
            let stacked = stack(&members, gn + un, d);
            targets.retain(|(n, _)| n != &format!("layers.{layer}.ffn.gate.weight"));
            targets.push((format!("layers.{layer}.ffn.gate.weight"), stacked));
        }
    }

    // Write the adapter bundle: f16 A/B pairs + flat lora.* metadata.
    let mut metadata = BTreeMap::new();
    metadata.insert("lora.rank".to_string(), serde_json::json!(rank));
    metadata.insert("lora.alpha".to_string(), serde_json::json!(alpha));
    metadata.insert("lora.max_seq".to_string(), serde_json::json!(args.max_seq));
    let header = Header {
        schema: 1,
        arch: "lora-adapter".to_string(),
        quant_scheme: base_format::QuantScheme::F16,
        min_hw: "apple_m1".to_string(),
        created: crate::chrono_now(),
        base_rt_version: env!("CARGO_PKG_VERSION").to_string(),
        source: base_format::SourceInfo {
            format: "peft_safetensors".to_string(),
            sha256: String::new(),
            filename: args
                .input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("adapter")
                .to_string(),
        },
        tokenizer: base_format::TokenizerBlob {
            fields: BTreeMap::new(),
        },
        config: base_format::ModelConfig { fields: base_cfg },
        metadata,
        target_backend: base_format::TargetBackend::Metal,
        quant_profile: String::new(),
        alignment: Default::default(),
        flags: base_format::HeaderFlags::HAS_LORA,
        layers: Vec::new(),
        tensors: Vec::new(),
        calibration: None,
        mmproj: None,
        speculator: None,
        // A LoRA adapter carries its own f16 A/B pairs verbatim from the PEFT
        // checkpoint; there is no source-tensor transplant to record.
        provenance: None,
        sig: None,
    };
    let mut w = base_format::BaseWriter::create(&out, header)?;
    let mut n_pairs = 0usize;
    for (target, p) in &targets {
        for (suffix, data, shape) in [
            ("A", &p.a, vec![p.r as u64, p.d_in as u64]),
            ("B", &p.b, vec![p.d_out as u64, p.r as u64]),
        ] {
            w.add_tensor(TensorPayload {
                entry: TensorEntry {
                    name: format!("lora.{target}.{suffix}"),
                    dtype: TensorDtype::F16,
                    shape,
                    offset: 0,
                    length: 0,
                    scale_offset: None,
                    scale_length: None,
                    bias_offset: None,
                    bias_length: None,
                    awq_scale_offset: None,
                    awq_scale_length: None,
                    group_size: None,
                    scale_dtype: None,
                    symmetric: false,
                    layout: None,
                    residency: None,
                    compute_region: Default::default(),
                    flags: base_format::TensorFlags::empty(),
                    checksum_xxh64: None,
                    source_ggml_type: None,
                },
                data: f32s_to_f16_bytes(data),
            });
        }
        n_pairs += 1;
    }
    w.finish()?;
    println!(
        "wrote {} ({} targets, rank {}, alpha {})",
        out.display(),
        n_pairs,
        rank,
        alpha
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block-diagonal stacking: member deltas must land in their own output
    /// rows and rank columns; absent members contribute zero rows.
    #[test]
    fn stack_places_members_block_diagonally() {
        let q = Pair {
            a: vec![1.0, 2.0],
            b: vec![10.0, 20.0],
            r: 1,
            d_in: 2,
            d_out: 2,
        };
        let v = Pair {
            a: vec![3.0, 4.0],
            b: vec![30.0],
            r: 1,
            d_in: 2,
            d_out: 1,
        };
        // Fused layout [q(2) | k(1, absent) | v(1)] → total_out 4.
        let s = stack(&[(&q, 0), (&v, 3)], 4, 2);
        assert_eq!(s.r, 2);
        assert_eq!(s.d_out, 4);
        // A rows: q's then v's.
        assert_eq!(s.a, vec![1.0, 2.0, 3.0, 4.0]);
        // B [4 rows x 2 ranks]: q rows use col 0, absent k row all-zero,
        // v row uses col 1.
        assert_eq!(s.b, vec![10.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 30.0]);
    }
}
