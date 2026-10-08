//! HuggingFace safetensors directory reader.
//!
//! A HF model directory looks like:
//! ```text
//!   model_dir/
//!     config.json                      (required)
//!     tokenizer.json                   (preferred)
//!     tokenizer_config.json            (optional)
//!     model.safetensors                (single-shard case)
//!   OR
//!     model-00001-of-00004.safetensors (sharded)
//!     ...
//!     model.safetensors.index.json     (maps tensor name → shard)
//! ```
//!
//! This reader opens all shards mmap-style and presents a unified
//! tensor listing. Safetensors from HF is mostly unquantized
//! (F32/F16/BF16); MLX-quantized safetensors go through the `mlx`
//! module instead. The one quantized HF layout read here is block FP8
//! (DeepSeek-V3 / GLM-5.3 style: `quantization_config.quant_method =
//! "fp8"`, e4m3 `<stem>.weight` + f32 `<stem>.weight_scale_inv`, one
//! scale per `weight_block_size` tile): `tensor_to_f32` dequantizes the
//! weight × its block scale, and the scale tensors are hidden from
//! `tensor_names` so the converter sees a plain unquantized checkpoint.

use crate::safetensors::{SafetensorsFile, StDtype, StTensorInfo};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub struct HfDir {
    pub model_dir: PathBuf,
    pub config: serde_json::Value,
    pub tokenizer_json: Option<serde_json::Value>,
    pub tokenizer_config: Option<serde_json::Value>,
    /// Contents of `chat_template.jinja` if the HF checkpoint stores its
    /// chat template as a separate file (Gemma 4 family does this — the
    /// template uses `<|turn>` / `<|channel>` markers with conditional
    /// thinking-mode logic that doesn't fit the JSON `chat_template`
    /// string convention older Gemmas used). The converter copies this
    /// verbatim into the bundle's tokenizer header so the runtime can
    /// reach it without re-downloading the source dir.
    pub chat_template_jinja: Option<String>,
    shards: Vec<SafetensorsFile>,
    /// name → (shard_idx, tensor_idx)
    lookup: BTreeMap<String, (usize, usize)>,
    /// Block-FP8 checkpoints: e4m3 weight name → its `weight_scale_inv`.
    fp8_scales: BTreeMap<String, String>,
    /// The paired scale tensors, hidden from `tensor_names`.
    fp8_scale_names: BTreeSet<String>,
    /// `quantization_config.weight_block_size` ([rows, cols]) of a
    /// block-FP8 checkpoint.
    fp8_block: Option<[u64; 2]>,
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

impl HfDir {
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            bail!("{:?} is not a directory", dir);
        }

        let config_path = dir.join("config.json");
        let config_bytes =
            std::fs::read(&config_path).with_context(|| format!("reading {:?}", config_path))?;
        let (config_bytes, n_nonfinite) = sanitize_python_json(&config_bytes);
        if n_nonfinite > 0 {
            eprintln!(
                "  note: config.json contains {n_nonfinite} bare Infinity/NaN literal(s) \
                 (Python's json.dump emits these; they are not valid JSON) — read as null"
            );
        }
        let config: serde_json::Value =
            serde_json::from_slice(&config_bytes).context("parsing config.json")?;

        let tokenizer_json = read_optional_json(&dir.join("tokenizer.json"))?;
        let tokenizer_config = read_optional_json(&dir.join("tokenizer_config.json"))?;
        // HF stores the chat template in one of two places. Recent checkpoints
        // (Gemma 4 onwards) put it in a standalone `chat_template.jinja` file —
        // useful when the template wants multi-line Jinja or conditional
        // thinking-mode logic that doesn't survive JSON string escaping.
        // Older checkpoints (Qwen, Llama, Mistral, Gemma 3) keep it inside
        // `tokenizer_config.json` under the `chat_template` key. Read either,
        // preferring the standalone file when both exist.
        let chat_template_jinja = {
            let p = dir.join("chat_template.jinja");
            if p.exists() {
                Some(std::fs::read_to_string(&p).with_context(|| format!("reading {:?}", p))?)
            } else {
                tokenizer_config
                    .as_ref()
                    .and_then(|tc| tc.get("chat_template"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            }
        };

        // Discover shards. Prefer the index.json path if present.
        let index_path = dir.join("model.safetensors.index.json");
        let (shard_paths, routing): (Vec<PathBuf>, Option<BTreeMap<String, String>>) =
            if index_path.exists() {
                let idx_bytes = std::fs::read(&index_path)?;
                let idx: ShardIndex =
                    serde_json::from_slice(&idx_bytes).context("parsing shard index")?;
                let mut shards: Vec<PathBuf> = idx
                    .weight_map
                    .values()
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .map(|name| dir.join(name))
                    .collect();
                shards.sort();
                (shards, Some(idx.weight_map))
            } else {
                // Glob for model*.safetensors.
                let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| {
                        p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                            && p.file_name()
                                .and_then(|n| n.to_str())
                                .map(|n| n.starts_with("model"))
                                .unwrap_or(false)
                    })
                    .collect();
                paths.sort();
                if paths.is_empty() {
                    bail!("no .safetensors shards in {:?}", dir);
                }
                (paths, None)
            };

        let mut shards = Vec::with_capacity(shard_paths.len());
        for p in &shard_paths {
            shards
                .push(SafetensorsFile::open(p).with_context(|| format!("opening shard {:?}", p))?);
        }

        // Build a name → (shard_idx, tensor_idx) lookup.
        let mut lookup = BTreeMap::new();
        if let Some(map) = &routing {
            for (tensor_name, shard_name) in map {
                let shard_idx = shard_paths
                    .iter()
                    .position(|p| {
                        p.file_name().and_then(|n| n.to_str()) == Some(shard_name.as_str())
                    })
                    .with_context(|| format!("shard {shard_name} not found"))?;
                let tensor_idx = shards[shard_idx]
                    .tensors
                    .iter()
                    .position(|t| t.name == *tensor_name)
                    .with_context(|| {
                        format!("tensor {tensor_name} declared in index but missing in shard")
                    })?;
                lookup.insert(tensor_name.clone(), (shard_idx, tensor_idx));
            }
        } else {
            for (si, shard) in shards.iter().enumerate() {
                for (ti, t) in shard.tensors.iter().enumerate() {
                    if lookup.insert(t.name.clone(), (si, ti)).is_some() {
                        bail!("duplicate tensor across shards: {}", t.name);
                    }
                }
            }
        }

        let mut hf = Self {
            model_dir: dir,
            config,
            tokenizer_json,
            tokenizer_config,
            chat_template_jinja,
            shards,
            lookup,
            fp8_scales: BTreeMap::new(),
            fp8_scale_names: BTreeSet::new(),
            fp8_block: None,
        };
        hf.pair_fp8_block_scales()?;
        Ok(hf)
    }

    /// Pair every e4m3 `<stem>.weight` with its `<stem>.weight_scale_inv`
    /// and check each scale covers its weight at the declared block size,
    /// so a malformed checkpoint fails at open rather than mid-convert.
    fn pair_fp8_block_scales(&mut self) -> Result<()> {
        let qc = self.config.get("quantization_config");
        let block = match qc {
            Some(qc) if qc.get("quant_method").and_then(|v| v.as_str()) == Some("fp8") => {
                if let Some(fmt) = qc.get("fmt").and_then(|v| v.as_str()) {
                    if fmt != "e4m3" {
                        bail!("fp8 checkpoint with fmt {fmt:?}: only e4m3 is supported");
                    }
                }
                let bs: Vec<u64> = qc
                    .get("weight_block_size")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                    .unwrap_or_default();
                if bs.len() != 2 || bs.contains(&0) {
                    bail!(
                        "fp8 checkpoint: quantization_config.weight_block_size must be \
                         [rows, cols], got {:?}",
                        qc.get("weight_block_size")
                    );
                }
                Some([bs[0], bs[1]])
            }
            _ => None,
        };
        for (name, &(si, ti)) in &self.lookup {
            let info = &self.shards[si].tensors[ti];
            if info.dtype != StDtype::F8E4m3 {
                continue;
            }
            let Some(stem) = name.strip_suffix(".weight") else {
                continue;
            };
            let scale = format!("{stem}.weight_scale_inv");
            let Some(&(ssi, sti)) = self.lookup.get(&scale) else {
                continue;
            };
            let Some([bo, bi]) = block else {
                bail!(
                    "{name} is F8_E4M3 with a weight_scale_inv sibling, but config.json \
                     declares no fp8 quantization_config (weight_block_size unknown)"
                );
            };
            if info.shape.len() != 2 {
                bail!(
                    "fp8 block weight {name} has shape {:?}; only 2-D is supported",
                    info.shape
                );
            }
            let want = [info.shape[0].div_ceil(bo), info.shape[1].div_ceil(bi)];
            let sinfo = &self.shards[ssi].tensors[sti];
            if sinfo.shape != want {
                bail!(
                    "{scale} has shape {:?}; {name} {:?} at block [{bo}, {bi}] needs {:?}",
                    sinfo.shape,
                    info.shape,
                    want
                );
            }
            self.fp8_scales.insert(name.clone(), scale.clone());
            self.fp8_scale_names.insert(scale);
        }
        if !self.fp8_scales.is_empty() {
            self.fp8_block = block;
        }
        Ok(())
    }

    /// `[rows, cols]` block size when this is a block-FP8 checkpoint whose
    /// e4m3 weights `tensor_to_f32` dequantizes.
    pub fn fp8_block_size(&self) -> Option<[u64; 2]> {
        self.fp8_block
    }

    /// Number of e4m3 weights dequantized through a block scale.
    pub fn fp8_weight_count(&self) -> usize {
        self.fp8_scales.len()
    }

    /// Every tensor the checkpoint holds, minus the block-FP8 scales that
    /// `tensor_to_f32` folds into their weights.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.lookup
            .keys()
            .filter(|n| !self.fp8_scale_names.contains(*n))
            .map(|s| s.as_str())
    }

    pub fn tensor_info(&self, name: &str) -> Option<&StTensorInfo> {
        let (si, ti) = *self.lookup.get(name)?;
        Some(&self.shards[si].tensors[ti])
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let (si, ti) = *self.lookup.get(name)?;
        let info = &self.shards[si].tensors[ti];
        Some(self.shards[si].tensor_bytes(info))
    }

    pub fn model_type(&self) -> Option<&str> {
        self.config.get("model_type").and_then(|v| v.as_str())
    }

    /// True when config.json declares MLX-style quantization (bits +
    /// group_size), meaning tensor data is MLX-packed and needs the
    /// `mlx` reader to dequant rather than the plain safetensors path.
    pub fn is_mlx_quantized(&self) -> bool {
        self.config
            .get("quantization")
            .and_then(|v| v.get("bits"))
            .is_some()
    }

    /// Dequant a tensor's bytes to f32 using the declared safetensors
    /// dtype. Does not handle MLX-packed uint32 tensors; those are in
    /// the `mlx` module.
    pub fn tensor_to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor_info(name)
            .with_context(|| format!("tensor {name} not found"))?;
        let bytes = self
            .tensor_bytes(name)
            .with_context(|| format!("tensor_bytes({name}) missing after tensor_info()"))?;
        let n: usize = info.shape.iter().product::<u64>() as usize;
        use half::{bf16, f16};
        Ok(match info.dtype {
            StDtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            StDtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            StDtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            // Index tables (EAGLE-3's d2t offsets) and masks: exact in f32
            // for the magnitudes they hold (token ids < 2^24).
            StDtype::I64 => bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            StDtype::I32 => bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            StDtype::Bool => bytes
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect(),
            StDtype::F8E4m3 => {
                // Unscaled e4m3 is off by the block scale (often 1e-3..1e-1),
                // so a weight without one is refused, never decoded raw.
                let scale_name = self.fp8_scales.get(name).with_context(|| {
                    format!(
                        "tensor {name} is F8_E4M3 with no <stem>.weight_scale_inv block \
                         scale to dequantize it"
                    )
                })?;
                let [bo, bi] = self.fp8_block.expect("set whenever a scale is paired");
                let scale = self.tensor_to_f32(scale_name)?;
                dequant_fp8_block(bytes, &info.shape, &scale, bo as usize, bi as usize)
            }
            other => bail!(
                "tensor_to_f32: unsupported safetensors dtype {:?} (tensor {name}, {n} values)",
                other
            ),
        })
    }
}

/// OCP e4m3fn (the `F8_E4M3` safetensors dtype): 1 sign, 4 exponent
/// (bias 7), 3 mantissa bits, no infinities; S.1111.111 is NaN, so the
/// largest finite magnitude is 448.
pub fn e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0x0F && b & 0x07 == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        // Subnormal: 0.mmm × 2^-6.
        return sign * (man / 8.0) * 2f32.powi(-6);
    }
    sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
}

/// Dequantize a row-major `[rows, cols]` e4m3 weight against its
/// `[ceil(rows/bo), ceil(cols/bi)]` block scale: `w[r][c] = e4m3(q[r][c])
/// × scale[r/bo][c/bi]`. (The checkpoint calls it `weight_scale_inv`,
/// but it is the factor the stored codes are multiplied by — DeepSeek's
/// own `weight_dequant` is `x * s`.)
fn dequant_fp8_block(bytes: &[u8], shape: &[u64], scale: &[f32], bo: usize, bi: usize) -> Vec<f32> {
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let scale_cols = cols.div_ceil(bi);
    debug_assert_eq!(bytes.len(), rows * cols);
    debug_assert_eq!(scale.len(), rows.div_ceil(bo) * scale_cols);
    let lut: [f32; 256] = std::array::from_fn(|i| e4m3fn_to_f32(i as u8));
    let mut out = Vec::with_capacity(rows * cols);
    for (r, row) in bytes.chunks_exact(cols).enumerate() {
        let srow = &scale[(r / bo) * scale_cols..][..scale_cols];
        out.extend(
            row.iter()
                .enumerate()
                .map(|(c, &q)| lut[q as usize] * srow[c / bi]),
        );
    }
    out
}

fn read_optional_json(path: &Path) -> Result<Option<serde_json::Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    let (bytes, _) = sanitize_python_json(&bytes);
    Ok(Some(serde_json::from_slice(&bytes)?))
}

/// Rewrite Python's non-standard JSON literals (`Infinity`, `-Infinity`,
/// `NaN`) to `null` so a strict parser accepts the document.
///
/// `json.dump` emits these by default and plenty of published checkpoints
/// carry them — NVIDIA's Nemotron-H config has
/// `"time_step_limit": [0.0, Infinity]`. They are not valid JSON (RFC 8259
/// has no non-finite numbers) and `serde_json` rejects the whole file, so
/// without this the model cannot be read at all.
///
/// `null` rather than a large finite float: consumers read this config as
/// a `serde_json::Value` and query keys with `.as_f64()`, so null reads as
/// "absent" and a consumer that actually needs the value sees nothing
/// rather than a silently wrong number.
///
/// Replacement happens only outside string literals, so a key or value
/// whose *text* contains `NaN` is untouched. Returns the rewritten bytes
/// and how many literals were replaced.
pub fn sanitize_python_json(bytes: &[u8]) -> (Vec<u8>, usize) {
    const LITERALS: [&[u8]; 3] = [b"-Infinity", b"Infinity", b"NaN"];

    // Fast path: nothing to do for the overwhelming majority of files.
    if !bytes.windows(3).any(|w| w == b"NaN") && !bytes.windows(8).any(|w| w == b"Infinity") {
        return (bytes.to_vec(), 0);
    }

    let mut out = Vec::with_capacity(bytes.len());
    let mut count = 0usize;
    let mut i = 0usize;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b);
            if b == b'\\' && i + 1 < bytes.len() {
                // Copy the escaped character verbatim so an escaped quote
                // does not look like the end of the string.
                out.push(bytes[i + 1]);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push(b);
            i += 1;
            continue;
        }
        if let Some(lit) = LITERALS.iter().find(|lit| bytes[i..].starts_with(lit)) {
            out.extend_from_slice(b"null");
            count += 1;
            i += lit.len();
            continue;
        }
        out.push(b);
        i += 1;
    }
    (out, count)
}

#[cfg(test)]
mod sanitize_tests {
    use super::sanitize_python_json;

    fn s(input: &str) -> (String, usize) {
        let (bytes, n) = sanitize_python_json(input.as_bytes());
        (String::from_utf8(bytes).unwrap(), n)
    }

    #[test]
    fn rewrites_nemotron_time_step_limit() {
        let (out, n) = s(r#"{"time_step_limit": [0.0, Infinity]}"#);
        assert_eq!(out, r#"{"time_step_limit": [0.0, null]}"#);
        assert_eq!(n, 1);
        serde_json::from_str::<serde_json::Value>(&out).unwrap();
    }

    #[test]
    fn rewrites_negative_infinity_and_nan() {
        let (out, n) = s(r#"{"a": -Infinity, "b": NaN}"#);
        assert_eq!(out, r#"{"a": null, "b": null}"#);
        assert_eq!(n, 2);
    }

    #[test]
    fn leaves_strings_alone() {
        // The words appear inside string literals, where they are data.
        let (out, n) = s(r#"{"note": "NaN and Infinity", "x": Infinity}"#);
        assert_eq!(out, r#"{"note": "NaN and Infinity", "x": null}"#);
        assert_eq!(n, 1);
    }

    #[test]
    fn escaped_quote_does_not_end_the_string() {
        let (out, n) = s(r#"{"note": "a \" NaN", "x": 1}"#);
        assert_eq!(out, r#"{"note": "a \" NaN", "x": 1}"#);
        assert_eq!(n, 0);
    }

    #[test]
    fn valid_json_is_untouched() {
        let src = r#"{"a": 1.0, "b": [1, 2], "c": "text"}"#;
        let (out, n) = s(src);
        assert_eq!(out, src);
        assert_eq!(n, 0);
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::{e4m3fn_to_f32, HfDir};
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn e4m3fn_edge_values() {
        assert_eq!(e4m3fn_to_f32(0x00), 0.0);
        assert_eq!(e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(e4m3fn_to_f32(0xB8), -1.0);
        assert_eq!(e4m3fn_to_f32(0x7E), 448.0); // largest finite
        assert_eq!(e4m3fn_to_f32(0xFE), -448.0);
        assert_eq!(e4m3fn_to_f32(0x08), 2f32.powi(-6)); // smallest normal
        assert_eq!(e4m3fn_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(e4m3fn_to_f32(0x07), 7.0 * 2f32.powi(-9));
        assert_eq!(e4m3fn_to_f32(0x3C), 1.5);
        assert!(e4m3fn_to_f32(0x7F).is_nan());
        assert!(e4m3fn_to_f32(0xFF).is_nan());
        // Every other code is finite.
        let finite = (0u8..=255)
            .filter(|&b| e4m3fn_to_f32(b).is_finite())
            .count();
        assert_eq!(finite, 254);
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[u64], Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + bytes.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, end] }),
            );
            offset = end;
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hdr).unwrap();
        for (_, _, _, bytes) in tensors {
            f.write_all(bytes).unwrap();
        }
    }

    fn refusal<T>(r: anyhow::Result<T>) -> String {
        match r {
            Ok(_) => panic!("must refuse"),
            Err(e) => e.to_string(),
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn fp8_config(dir: &Path, block: serde_json::Value) {
        let config = serde_json::json!({
            "model_type": "test",
            "quantization_config": {
                "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                "weight_block_size": block,
            },
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    /// A 3×5 weight at block [2, 2]: ragged edge tiles in both
    /// dimensions, a distinct scale per tile, and a bf16 tensor alongside
    /// that must pass through untouched.
    #[test]
    fn dequantizes_weight_times_block_scale() {
        let tmp = tempfile::tempdir().unwrap();
        fp8_config(tmp.path(), serde_json::json!([2, 2]));
        // Codes: 1.0 everywhere except a 1.5 and a -448.
        let mut q = vec![0x38u8; 15];
        q[4] = 0x3C; // [0][4] = 1.5
        q[10] = 0xFE; // [2][0] = -448
                      // scale [2, 3]: tile (r/2, c/2).
        let scale = [1.0f32, 2.0, 4.0, 0.5, 0.25, 0.125];
        let norm: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
            .collect();
        write_safetensors(
            &tmp.path().join("model.safetensors"),
            &[
                ("l.weight", "F8_E4M3", &[3, 5], q),
                ("l.weight_scale_inv", "F32", &[2, 3], f32_bytes(&scale)),
                ("n.weight", "BF16", &[2], norm),
            ],
        );
        let hf = HfDir::open(tmp.path()).unwrap();
        assert_eq!(hf.fp8_block_size(), Some([2, 2]));
        assert_eq!(hf.fp8_weight_count(), 1);
        let names: Vec<&str> = hf.tensor_names().collect();
        assert_eq!(
            names,
            ["l.weight", "n.weight"],
            "the scale is folded, not listed"
        );
        let w = hf.tensor_to_f32("l.weight").unwrap();
        #[rustfmt::skip]
        let want = [
            1.0,  1.0,  2.0,   2.0,   4.0 * 1.5,
            1.0,  1.0,  2.0,   2.0,   4.0,
            -448.0 * 0.5, 0.5, 0.25, 0.25, 0.125,
        ];
        assert_eq!(w, want);
        assert_eq!(hf.tensor_to_f32("n.weight").unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn bf16_block_scales_are_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        fp8_config(tmp.path(), serde_json::json!([128, 128]));
        let scale: Vec<u8> = half::bf16::from_f32(0.5).to_le_bytes().to_vec();
        write_safetensors(
            &tmp.path().join("model.safetensors"),
            &[
                ("l.weight", "F8_E4M3", &[2, 3], vec![0x38; 6]),
                ("l.weight_scale_inv", "BF16", &[1, 1], scale),
            ],
        );
        let hf = HfDir::open(tmp.path()).unwrap();
        assert_eq!(hf.tensor_to_f32("l.weight").unwrap(), [0.5; 6]);
    }

    #[test]
    fn scale_shape_mismatch_fails_at_open() {
        let tmp = tempfile::tempdir().unwrap();
        fp8_config(tmp.path(), serde_json::json!([2, 2]));
        write_safetensors(
            &tmp.path().join("model.safetensors"),
            &[
                ("l.weight", "F8_E4M3", &[3, 5], vec![0x38; 15]),
                ("l.weight_scale_inv", "F32", &[1, 3], f32_bytes(&[1.0; 3])),
            ],
        );
        let err = refusal(HfDir::open(tmp.path()));
        assert!(err.contains("needs [2, 3]"), "{err}");
    }

    #[test]
    fn unscaled_fp8_is_refused_not_decoded_raw() {
        let tmp = tempfile::tempdir().unwrap();
        fp8_config(tmp.path(), serde_json::json!([2, 2]));
        write_safetensors(
            &tmp.path().join("model.safetensors"),
            &[("l.weight", "F8_E4M3", &[2, 2], vec![0x38; 4])],
        );
        let hf = HfDir::open(tmp.path()).unwrap();
        assert_eq!(hf.fp8_block_size(), None);
        let err = refusal(hf.tensor_to_f32("l.weight"));
        assert!(err.contains("no <stem>.weight_scale_inv"), "{err}");
    }

    #[test]
    fn paired_scale_without_fp8_config_fails_at_open() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), br#"{"model_type": "test"}"#).unwrap();
        write_safetensors(
            &tmp.path().join("model.safetensors"),
            &[
                ("l.weight", "F8_E4M3", &[2, 2], vec![0x38; 4]),
                ("l.weight_scale_inv", "F32", &[1, 1], f32_bytes(&[1.0])),
            ],
        );
        let err = refusal(HfDir::open(tmp.path()));
        assert!(err.contains("weight_block_size unknown"), "{err}");
    }
}
