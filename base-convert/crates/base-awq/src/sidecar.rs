//! Activation-statistics sidecar I/O.
//!
//! Per the canonical-quant migration: no Python deps. The sidecar is
//! produced by baseRT running in fp16 calibration mode (see
//! `baseRT_calibrate` tool, future) and consumed by the converter
//! at convert time via `--awq-profile <path>`.
//!
//! On-disk format: JSON. `per_tensor_absmax`: keys are canonical .base
//! tensor names of linear-layer weights; values are per-input-channel
//! absmax floats covering the calibration set. Optional
//! `per_tensor_meansq` (same keys/lengths): per-channel mean of `x_j²`
//! over the calibration positions — when present it is the error weight
//! of the alpha search (see `activation_weights_from`); `absmax²` is the
//! fallback. Unknown extra keys are ignored.
//!
//! `source_fingerprint` contract (checked by base-convert for HF
//! safetensors sources): the string `inv-sha256:<hex>` where `<hex>` is
//! the SHA-256 of the source's tensor inventory — one line per tensor,
//! sorted by tensor name, `"{name}\t{d0,d1,...}\t{DTYPE}\n"` with the
//! safetensors header names verbatim (e.g.
//! `model.language_model.layers.0.mlp.up_proj.weight`), shape dims
//! joined by `,`, and the safetensors dtype string upper-cased
//! (`BF16`, `F16`, `F32`, ...). Every line ends with `\n`. A sidecar
//! whose fingerprint is `null` is accepted with a warning; a mismatch
//! is an error. Producers that can't compute it may leave it null.

use crate::AwqProfile;
use anyhow::{anyhow, Context, Result};
use std::path::Path;

impl AwqProfile {
    /// Read the sidecar JSON at `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading AWQ profile {}", path.display()))?;
        let p: AwqProfile = serde_json::from_slice(&bytes).context("parsing AWQ profile JSON")?;
        Ok(p)
    }

    /// Write the profile to `path` (canonical JSON, sorted keys).
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec_pretty(self).context("serializing AWQ profile")?;
        std::fs::write(path, json)
            .with_context(|| format!("writing AWQ profile {}", path.display()))?;
        Ok(())
    }

    /// Lookup the per-input-channel absmax vector for a tensor name.
    /// Returns None if the profile lacks an entry — callers fall back
    /// to plain RTN for that tensor.
    ///
    /// The calibrate tool records RUNTIME tensor names
    /// (`layers.N.attention.q.weight`, `layers.N.ffn.down.weight`) while
    /// the converter looks up CANONICAL names
    /// (`layers.N.self_attn.q_proj.weight`, `layers.N.mlp.down_proj.weight`),
    /// so an exact miss falls back to the canonical→runtime translation.
    /// `ffn.gate` serves BOTH `gate_proj` and `up_proj`: the runtime
    /// captures one absmax per matmul INPUT, and gate/up consume the same
    /// post-norm hidden state.
    pub fn absmax(&self, tensor_name: &str) -> Option<&[f32]> {
        if let Some(v) = self.per_tensor_absmax.get(tensor_name) {
            return Some(v.as_slice());
        }
        const MAP: [(&str, &str); 7] = [
            (".self_attn.q_proj.weight", ".attention.q.weight"),
            (".self_attn.k_proj.weight", ".attention.k.weight"),
            (".self_attn.v_proj.weight", ".attention.v.weight"),
            (".self_attn.o_proj.weight", ".attention.output.weight"),
            (".mlp.down_proj.weight", ".ffn.down.weight"),
            (".mlp.gate_proj.weight", ".ffn.gate.weight"),
            (".mlp.up_proj.weight", ".ffn.gate.weight"),
        ];
        for (canon, runtime) in MAP {
            if let Some(prefix) = tensor_name.strip_suffix(canon) {
                let alt = format!("{prefix}{runtime}");
                if let Some(v) = self.per_tensor_absmax.get(&alt) {
                    return Some(v.as_slice());
                }
            }
        }
        None
    }

    /// Lookup the optional per-input-channel `E[x_j²]` vector.
    pub fn meansq(&self, tensor_name: &str) -> Option<&[f32]> {
        self.per_tensor_meansq
            .get(tensor_name)
            .map(|v| v.as_slice())
    }

    /// Per-expert absmax rows `[E][K]` of a fused 3-D expert tensor.
    pub fn expert_absmax(&self, tensor_name: &str) -> Option<&[Vec<f32>]> {
        self.per_expert_absmax
            .get(tensor_name)
            .map(|v| v.as_slice())
    }

    /// Per-expert `E[x_j²]` rows `[E][K]` of a fused 3-D expert tensor.
    pub fn expert_meansq(&self, tensor_name: &str) -> Option<&[Vec<f32>]> {
        self.per_expert_meansq
            .get(tensor_name)
            .map(|v| v.as_slice())
    }

    /// Routed rows per expert `[E]` of a fused 3-D expert tensor.
    pub fn expert_rows(&self, tensor_name: &str) -> Option<&[u64]> {
        self.per_expert_rows.get(tensor_name).map(|v| v.as_slice())
    }

    /// Validate the profile against the source's inventory fingerprint
    /// (see the module doc for the contract). A mismatch is an error; a
    /// sidecar with no fingerprint is accepted with a warning so
    /// producers that cannot compute it still work.
    pub fn check_fingerprint(&self, expected: &str) -> Result<()> {
        match &self.source_fingerprint {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(anyhow!(
                "AWQ profile fingerprint {actual:?} does not match model fingerprint {expected:?}"
            )),
            None => {
                eprintln!(
                    "  awq:     warning: sidecar has no source_fingerprint; \
                     cannot verify it was measured on this checkpoint ({expected})"
                );
                Ok(())
            }
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
    fn per_expert_round_trip_and_legacy_parse() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut p = make_profile();
        let n = "layers.0.ffn_down_exps.weight".to_string();
        p.per_tensor_absmax.insert(n.clone(), vec![2.0, 3.0]);
        p.per_expert_absmax.insert(
            n.clone(),
            vec![vec![1.0, 3.0], vec![2.0, 0.5], vec![0.0, 0.0]],
        );
        p.per_expert_meansq.insert(
            n.clone(),
            vec![vec![0.5, 1.0], vec![1.5, 0.1], vec![0.0, 0.0]],
        );
        p.per_expert_rows.insert(n.clone(), vec![7, 9, 0]);
        p.save(tmp.path()).unwrap();
        let q = AwqProfile::load(tmp.path()).unwrap();
        assert_eq!(q.expert_absmax(&n).unwrap()[1], vec![2.0, 0.5]);
        assert_eq!(q.expert_meansq(&n).unwrap()[0], vec![0.5, 1.0]);
        assert_eq!(q.expert_rows(&n), Some([7u64, 9, 0].as_slice()));
        // a pre-MoE sidecar (no per_expert_* keys) still parses
        let legacy =
            br#"{"source_fingerprint":null,"calib_tokens":3,"per_tensor_absmax":{"a":[1.0]}}"#;
        let l: AwqProfile = serde_json::from_slice(legacy).unwrap();
        assert!(l.expert_absmax("a").is_none() && l.per_expert_rows.is_empty());
        // and an empty per-expert map is not written back
        let json = serde_json::to_string(&l).unwrap();
        assert!(!json.contains("per_expert"));
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

    #[test]
    fn fingerprint_absent_is_accepted() {
        let mut p = make_profile();
        p.source_fingerprint = None;
        p.check_fingerprint("anything").unwrap();
    }
}
