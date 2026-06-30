//! Input-format readers: GGUF, MLX-safetensors, HF-safetensors.
//!
//! Each reader exposes:
//! - A streaming metadata/header parse (no dequant)
//! - Per-tensor raw byte access (zero-copy from mmap)
//! - A `dequant_to_f32` helper that materializes a full-precision view
//!   of a tensor, used by the converter before re-quantizing to a
//!   canonical `.base` scheme
//!
//! GGUF is the first reader. MLX and HF land in a later commit.

pub mod gguf;
pub mod hf;
pub mod mlx;
pub mod safetensors;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Gguf,
    MlxSafetensors,
    HfSafetensors,
}

/// Detect source format from a path. Directories are resolved as HF or
/// MLX depending on config.json content.
pub fn detect_format(path: &std::path::Path) -> anyhow::Result<SourceFormat> {
    if path.is_dir() {
        let cfg_path = path.join("config.json");
        if !cfg_path.exists() {
            anyhow::bail!("{:?} is a directory without config.json", path);
        }
        let bytes = std::fs::read(&cfg_path)?;
        let cfg: serde_json::Value = serde_json::from_slice(&bytes)?;
        if cfg.get("quantization").and_then(|v| v.get("bits")).is_some() {
            Ok(SourceFormat::MlxSafetensors)
        } else {
            Ok(SourceFormat::HfSafetensors)
        }
    } else {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => Ok(SourceFormat::Gguf),
            Some("safetensors") => Ok(SourceFormat::HfSafetensors),
            _ => anyhow::bail!("unrecognized source path: {:?}", path),
        }
    }
}
