//! `--mlx-passthrough` on a synthetic MLX Qwen3-MoE checkpoint whose
//! router and shared-expert gate are 8-bit inside a 4-bit source — the
//! layout mlx-community ships for Qwen3.6-35B-A3B (BAS-960). The 4-bit
//! experts must transplant byte-verbatim; the 8-bit tensors the runtime
//! reads at f16 must dequantize exactly and store as f16, and the
//! `--validate` gate must accept the result.

use base_format::{BaseReader, TensorDtype};
use std::path::Path;
use std::process::Command;

const HIDDEN: u64 = 64;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const HEAD_DIM: u64 = 32;
const EXPERTS: u64 = 4;
const MOE_FFN: u64 = 64;
const VOCAB: u64 = 128;
const GS: u64 = 64;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_basert")
}

/// One MLX affine-quantized tensor: packed codes, f16 scales and biases,
/// and the values they decode to (`q * scale + bias`, f32 math), in
/// row-major order over `shape`.
struct Quant {
    shape: Vec<u64>,
    bits: u32,
    packed: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    values: Vec<f32>,
}

/// Deterministic codes/scales/biases from a seed: the test needs exact
/// bytes to compare, not realistic weights.
fn quant(seed: u32, shape: &[u64], bits: u32) -> Quant {
    let in_features = *shape.last().unwrap() as usize;
    assert_eq!(in_features as u64 % GS, 0);
    let rows: usize = shape[..shape.len() - 1].iter().product::<u64>() as usize;
    let per_word = 32 / bits as usize;
    let words_per_row = in_features / per_word;
    let groups_per_row = in_features / GS as usize;
    let mut state = seed.wrapping_mul(2654435761) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mask = (1u32 << bits) - 1;
    let mut packed = vec![0u32; rows * words_per_row];
    let mut scales = Vec::with_capacity(rows * groups_per_row);
    let mut biases = Vec::with_capacity(rows * groups_per_row);
    let mut values = Vec::with_capacity(rows * in_features);
    for r in 0..rows {
        let mut sc = Vec::new();
        let mut bi = Vec::new();
        for _ in 0..groups_per_row {
            let s = half::f16::from_f32(0.001 + (next() % 1000) as f32 * 1e-4);
            let b = half::f16::from_f32(-0.5 + (next() % 1000) as f32 * 1e-3);
            scales.push(s.to_bits());
            biases.push(b.to_bits());
            sc.push(s.to_f32());
            bi.push(b.to_f32());
        }
        for j in 0..in_features {
            let q = next() & mask;
            let word = r * words_per_row + j / per_word;
            let shift = (j % per_word) * bits as usize;
            packed[word] |= q << shift;
            let g = j / GS as usize;
            values.push(q as f32 * sc[g] + bi[g]);
        }
    }
    Quant {
        shape: shape.to_vec(),
        bits,
        packed,
        scales,
        biases,
        values,
    }
}

struct St {
    header: serde_json::Map<String, serde_json::Value>,
    blobs: Vec<Vec<u8>>,
    offset: u64,
}

impl St {
    fn new() -> Self {
        Self {
            header: serde_json::Map::new(),
            blobs: Vec::new(),
            offset: 0,
        }
    }
    fn push(&mut self, name: &str, dtype: &str, shape: &[u64], bytes: Vec<u8>) {
        let end = self.offset + bytes.len() as u64;
        self.header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [self.offset, end] }),
        );
        self.offset = end;
        self.blobs.push(bytes);
    }
    fn f16(&mut self, name: &str, shape: &[u64], v: f32) {
        let n: u64 = shape.iter().product();
        let bytes: Vec<u8> = (0..n)
            .flat_map(|_| half::f16::from_f32(v).to_le_bytes())
            .collect();
        self.push(name, "F16", shape, bytes);
    }
    fn quant(&mut self, name: &str, q: &Quant) {
        let per_word = 32 / q.bits as u64;
        let mut pshape = q.shape.clone();
        *pshape.last_mut().unwrap() /= per_word;
        let mut sshape = q.shape.clone();
        *sshape.last_mut().unwrap() /= GS;
        let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        self.push(&format!("{name}.weight"), "U32", &pshape, u32s(&q.packed));
        self.push(&format!("{name}.scales"), "F16", &sshape, u16s(&q.scales));
        self.push(&format!("{name}.biases"), "F16", &sshape, u16s(&q.biases));
    }
    fn write(self, path: &Path) {
        let hdr = serde_json::to_vec(&serde_json::Value::Object(self.header)).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        file.extend_from_slice(&hdr);
        for b in &self.blobs {
            file.extend_from_slice(b);
        }
        std::fs::write(path, file).unwrap();
    }
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

#[test]
fn eight_bit_router_and_shared_gate_store_as_f16_beside_verbatim_q4_experts() {
    let tmp = std::env::temp_dir().join(format!("basert-mlx-pt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let dir = tmp.join("src");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": MOE_FFN,
            "moe_intermediate_size": MOE_FFN,
            "num_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
            "tie_word_embeddings": false,
            // The mlx-community layout: a 4-bit checkpoint whose router
            // and shared-expert gate are overridden to 8-bit.
            "quantization": {
                "bits": 4, "group_size": GS,
                "model.layers.0.mlp.gate": {"bits": 8, "group_size": GS},
                "model.layers.0.mlp.shared_expert_gate": {"bits": 8, "group_size": GS},
            },
        })
        .to_string(),
    )
    .unwrap();

    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let router = quant(7, &[EXPERTS, HIDDEN], 8);
    let shexp_gate = quant(8, &[1, HIDDEN], 8);
    let exp_gate = quant(9, &[EXPERTS, MOE_FFN, HIDDEN], 4);
    let exp_up = quant(10, &[EXPERTS, MOE_FFN, HIDDEN], 4);
    let exp_down = quant(11, &[EXPERTS, HIDDEN, MOE_FFN], 4);
    let mut st = St::new();
    st.quant("model.embed_tokens", &quant(1, &[VOCAB, HIDDEN], 4));
    st.quant("lm_head", &quant(2, &[VOCAB, HIDDEN], 4));
    st.f16("model.norm.weight", &[HIDDEN], 1.0);
    let l = "model.layers.0";
    st.f16(&format!("{l}.input_layernorm.weight"), &[HIDDEN], 1.0);
    st.f16(
        &format!("{l}.post_attention_layernorm.weight"),
        &[HIDDEN],
        1.0,
    );
    st.f16(&format!("{l}.self_attn.q_norm.weight"), &[HEAD_DIM], 1.0);
    st.f16(&format!("{l}.self_attn.k_norm.weight"), &[HEAD_DIM], 1.0);
    st.quant(
        &format!("{l}.self_attn.q_proj"),
        &quant(3, &[q_dim, HIDDEN], 4),
    );
    st.quant(
        &format!("{l}.self_attn.k_proj"),
        &quant(4, &[kv_dim, HIDDEN], 4),
    );
    st.quant(
        &format!("{l}.self_attn.v_proj"),
        &quant(5, &[kv_dim, HIDDEN], 4),
    );
    st.quant(
        &format!("{l}.self_attn.o_proj"),
        &quant(6, &[HIDDEN, q_dim], 4),
    );
    st.quant(&format!("{l}.mlp.gate"), &router);
    st.quant(&format!("{l}.mlp.shared_expert_gate"), &shexp_gate);
    st.quant(&format!("{l}.mlp.switch_mlp.gate_proj"), &exp_gate);
    st.quant(&format!("{l}.mlp.switch_mlp.up_proj"), &exp_up);
    st.quant(&format!("{l}.mlp.switch_mlp.down_proj"), &exp_down);
    st.write(&dir.join("model.safetensors"));

    let out = tmp.join("out.base");
    let output = Command::new(bin())
        .arg("convert")
        .arg(&dir)
        .arg("-o")
        .arg(&out)
        .arg("--mlx-passthrough")
        .arg("--validate")
        .output()
        .expect("run basert convert");
    assert!(
        output.status.success(),
        "convert --mlx-passthrough --validate failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reader = BaseReader::open(&out).unwrap();
    let entry = |name: &str| {
        reader
            .header()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "{name} missing; have {:?}",
                    reader
                        .header()
                        .tensors
                        .iter()
                        .map(|t| &t.name)
                        .collect::<Vec<_>>()
                )
            })
            .clone()
    };

    // The 8-bit tensors the runtime reads at f16: exact dequant, f16.
    for (name, q) in [
        ("layers.0.mlp.gate.weight", &router),
        ("layers.0.mlp.shared_expert_gate.weight", &shexp_gate),
    ] {
        let e = entry(name);
        assert_eq!(e.dtype, TensorDtype::F16, "{name} dtype");
        assert_eq!(
            reader.tensor_bytes(name).unwrap(),
            f16_bytes(&q.values),
            "{name} payload"
        );
    }

    // The 4-bit experts: transplanted packed codes, byte-verbatim.
    for (name, q) in [
        ("layers.0.ffn_gate_exps.weight", &exp_gate),
        ("layers.0.ffn_up_exps.weight", &exp_up),
        ("layers.0.ffn_down_exps.weight", &exp_down),
    ] {
        let e = entry(name);
        assert_eq!(e.dtype, TensorDtype::BaseQ4, "{name} dtype");
        let data = reader.tensor_bytes(name).unwrap();
        let scale_off = e.scale_offset.unwrap() as usize;
        let want: Vec<u8> = q.packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(&data[..scale_off], &want[..], "{name} packed codes");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
