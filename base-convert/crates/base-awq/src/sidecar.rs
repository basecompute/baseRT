//! Activation-statistics sidecar I/O.
//!
//! Per the canonical-quant migration: no Python deps. The sidecar is
//! produced by baseRT running in fp16 calibration mode (see
//! `baseRT_calibrate` tool, future) and consumed by the converter
//! at convert time via `--awq-profile <path>`.
//!
//! On-disk format: JSON. Keys are canonical .base tensor names of
//! linear-layer weights; values are per-input-channel absmax floats
//! covering the calibration set. The sidecar's `source_fingerprint`
//! must match the model's fingerprint (sha256 of source weights) to
//! guarantee profile/model alignment — the converter rejects mismatch.

use crate::AwqProfile;
use anyhow::{anyhow, Context, Result};
use std::path::Path;

impl AwqProfile {
    /// Read the sidecar JSON at `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading AWQ profile {}", path.display()))?;
        let p: AwqProfile = serde_json::from_slice(&bytes)
            .context("parsing AWQ profile JSON")?;
        Ok(p)
    }

    /// Write the profile to `path` (canonical JSON, sorted keys).
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .context("serializing AWQ profile")?;
        std::fs::write(path, json)
            .with_context(|| format!("writing AWQ profile {}", path.display()))?;
        Ok(())
    }

    /// Lookup the per-input-channel absmax vector for a tensor name.
    /// Returns None if the profile lacks an entry — callers fall back
    /// to plain RTN for that tensor.
    pub fn absmax(&self, tensor_name: &str) -> Option<&[f32]> {
        self.per_tensor_absmax.get(tensor_name).map(|v| v.as_slice())
    }

    /// Validate the profile against an expected source fingerprint
    /// (typically sha256 of the .base header without the sig field,
    /// or sha256 of the source weights). Fails if mismatch.
    pub fn check_fingerprint(&self, expected: &str) -> Result<()> {
        match &self.source_fingerprint {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(anyhow!(
                "AWQ profile fingerprint {actual:?} does not match model fingerprint {expected:?}"
            )),
            None => Err(anyhow!(
                "AWQ profile has no source_fingerprint; cannot validate against model {expected:?}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile() -> AwqProfile {
        let mut p = AwqProfile {
            source_fingerprint: Some("abc123".into()),
            calib_tokens: 512,
            ..AwqProfile::default()
        };
        p.per_tensor_absmax.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![1.0, 2.5, 0.3, 5.0],
        );
        p
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let original = make_profile();
        original.save(tmp.path()).unwrap();
        let loaded = AwqProfile::load(tmp.path()).unwrap();
        assert_eq!(loaded.source_fingerprint, original.source_fingerprint);
        assert_eq!(loaded.calib_tokens, original.calib_tokens);
        assert_eq!(loaded.per_tensor_absmax.len(), 1);
        assert_eq!(
            loaded.absmax("model.layers.0.self_attn.q_proj.weight"),
            Some([1.0, 2.5, 0.3, 5.0].as_slice())
        );
    }

    #[test]
    fn lookup_missing_tensor_returns_none() {
        let p = make_profile();
        assert!(p.absmax("does.not.exist").is_none());
    }

    #[test]
    fn fingerprint_mismatch_errors() {
        let p = make_profile();
        let err = p.check_fingerprint("xyz789").unwrap_err();
        assert!(format!("{err}").contains("does not match"));
    }

    #[test]
    fn fingerprint_match_succeeds() {
        let p = make_profile();
        p.check_fingerprint("abc123").unwrap();
    }
}
