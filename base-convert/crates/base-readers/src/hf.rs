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
//! tensor listing. It does not dequantize — safetensors from HF is
//! almost always unquantized (F32/F16/BF16). MLX-quantized safetensors
//! go through `mlx` module instead.

use crate::safetensors::{SafetensorsFile, StDtype, StTensorInfo};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
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

        Ok(Self {
            model_dir: dir,
            config,
            tokenizer_json,
            tokenizer_config,
            chat_template_jinja,
            shards,
            lookup,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.lookup.keys().map(|s| s.as_str())
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
            other => bail!(
                "tensor_to_f32: unsupported safetensors dtype {:?} (tensor {name}, {n} values)",
                other
            ),
        })
    }
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
