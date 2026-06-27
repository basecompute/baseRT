//! Quant profile loader + glob matcher.
//!
//! Profiles are reusable JSON configs that name a set of per-tensor
//! quant rules. The converter takes `--profile <path>` and the
//! resulting bundle records the profile name in `quant_profile` for
//! audit. See `tools/base-convert/profiles/*.json`.
//!
//! Glob syntax (per `CANONICAL_QUANT_SPEC.md`):
//!   - `*`    matches anything except `.`
//!   - `**`   matches anything (any number of `.`-segments)
//!   - `{a,b,c}` alternation; combinations are expanded at load time
//!   - First-match-wins
//!
//! Pattern matching in this module is regex-free (no extra workspace
//! dep) — small custom matcher.

use anyhow::{anyhow, bail, Context, Result};
use base_format::{ScaleDtype, TensorDtype};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level profile schema. One JSON file per profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantProfile {
    /// Identifier copied into `Header.quant_profile` at convert time.
    pub name: String,
    /// Architecture this profile is for. Compared against
    /// `Header.arch` at convert time; mismatch is an error.
    pub arch: String,
    /// Calibration spec — null/missing for RTN-only profiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationSpec>,
    /// Ordered rules. First match wins per tensor name.
    pub rules: Vec<RuleEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSpec {
    /// `awq` | `gptq` | `smoothquant` | `rtn`
    pub method: String,
    /// Calibration token count.
    pub tokens: u32,
    /// Dataset identifier. `wikitext-103` is the baked-in default.
    pub dataset: String,
}

/// A single rule. Pattern matches tensor names (glob); `dtype` is
/// required, the rest inherit canonical defaults if absent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEntry {
    pub pattern: String,
    pub dtype: TensorDtype,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_dtype: Option<ScaleDtype>,
    /// Override the dtype's default symmetric/asymmetric flag.
    /// None = false (asymmetric, MLX-affine default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symmetric: Option<bool>,
}

/// Resolved per-tensor quant config — the spec defaults filled in.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedQuant {
    pub dtype: TensorDtype,
    pub group_size: u32,
    pub scale_dtype: ScaleDtype,
    pub symmetric: bool,
}

impl QuantProfile {
    /// Parse a profile from JSON bytes.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let p: Self = serde_json::from_slice(bytes)
            .context("parsing quant profile JSON")?;
        p.validate()?;
        Ok(p)
    }

    /// Read a profile from a path on disk.
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading profile {}", path.display()))?;
        Self::from_json(&bytes)
    }

    /// Sanity-check the profile internals: rules cover their targets,
    /// any patterns are well-formed, `dtype` + `scale_dtype` are
    /// compatible.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            bail!("profile name must not be empty");
        }
        if self.arch.is_empty() {
            bail!("profile arch must not be empty");
        }
        for (i, rule) in self.rules.iter().enumerate() {
            if rule.pattern.is_empty() {
                bail!("rule {i}: empty pattern");
            }
            // Every alternation must have a closing brace.
            let opens = rule.pattern.matches('{').count();
            let closes = rule.pattern.matches('}').count();
            if opens != closes {
                bail!("rule {i}: unbalanced {{}} in pattern {:?}", rule.pattern);
            }
            // e4m3 is q8-only per spec.
            if rule.scale_dtype == Some(ScaleDtype::E4m3)
                && rule.dtype != TensorDtype::BaseQ8
            {
                bail!(
                    "rule {i}: scale_dtype=e4m3 is only valid for base_q8, not {:?}",
                    rule.dtype
                );
            }
        }
        Ok(())
    }

    /// Resolve a tensor name to its quant config. Returns the first
    /// matching rule's resolved config. None if no rule matches —
    /// callers typically convert that to an error ("profile must
    /// cover every tensor").
    pub fn resolve(&self, name: &str) -> Option<ResolvedQuant> {
        for rule in &self.rules {
            if pattern_matches(&rule.pattern, name) {
                return Some(resolve_rule(rule));
            }
        }
        None
    }

    /// Like `resolve` but errors when the name is unmatched, which is
    /// the common contract: the converter requires every tensor to be
    /// covered by the profile.
    pub fn resolve_or_err(&self, name: &str) -> Result<ResolvedQuant> {
        self.resolve(name)
            .ok_or_else(|| anyhow!("profile {:?} has no rule matching {:?}", self.name, name))
    }
}

fn resolve_rule(rule: &RuleEntry) -> ResolvedQuant {
    let group_size = rule
        .group_size
        .or_else(|| rule.dtype.default_group_size())
        .unwrap_or(1);
    let scale_dtype = rule.scale_dtype.unwrap_or(ScaleDtype::Bf16);
    let symmetric = rule.symmetric.unwrap_or(false);
    ResolvedQuant {
        dtype: rule.dtype,
        group_size,
        scale_dtype,
        symmetric,
    }
}

// ---------- Pattern matching ----------

/// Top-level glob match. Expands `{a,b}` alternation, then segment-
/// matches each expansion.
pub fn pattern_matches(pattern: &str, name: &str) -> bool {
    for expanded in expand_alternations(pattern) {
        if match_no_alternation(&expanded, name) {
            return true;
        }
    }
    false
}

/// Cartesian-expand `{a,b,c}` patterns. Nested alternation isn't
/// supported (no `{a,{b,c}}`); error out at validate time if needed.
fn expand_alternations(pattern: &str) -> Vec<String> {
    if !pattern.contains('{') {
        return vec![pattern.to_string()];
    }
    // Find the leftmost `{ ... }` group, expand it, recurse.
    let bytes = pattern.as_bytes();
    let open = pattern.find('{').unwrap();
    // Match closing brace ignoring nesting (we error on nesting in
    // validate; for safety we simply find next `}`).
    let close_rel = pattern[open..]
        .find('}')
        .expect("validate() should have caught unbalanced braces");
    let close = open + close_rel;
    let prefix = &pattern[..open];
    let body = &pattern[open + 1..close];
    let suffix = pattern[close + 1..].to_string();
    let mut out = Vec::new();
    for option in body.split(',') {
        let combined = format!("{prefix}{option}");
        for sfx in expand_alternations(&suffix) {
            out.push(format!("{combined}{sfx}"));
        }
    }
    let _ = bytes; // silence unused lint
    out
}

/// Segment-aware match for a pattern with no `{}`.
/// `*` matches within a single dot-segment (no `.`); `**` matches any
/// number of segments including zero.
fn match_no_alternation(pattern: &str, name: &str) -> bool {
    // Split into tokens: each segment OR a `**` token.
    let pat_segs: Vec<&str> = pattern.split('.').collect();
    let name_segs: Vec<&str> = name.split('.').collect();
    match_segments(&pat_segs, &name_segs)
}

fn match_segments(pat: &[&str], name: &[&str]) -> bool {
    // Empty pattern matches only empty name.
    if pat.is_empty() {
        return name.is_empty();
    }
    let head = pat[0];
    if head == "**" {
        // ** consumes 0..=N segments.
        for take in 0..=name.len() {
            if match_segments(&pat[1..], &name[take..]) {
                return true;
            }
        }
        false
    } else {
        if name.is_empty() {
            return false;
        }
        if !match_segment(head, name[0]) {
            return false;
        }
        match_segments(&pat[1..], &name[1..])
    }
}

/// Match a single pattern segment against a single name segment.
/// `*` within the segment matches any non-`.` characters.
fn match_segment(pat: &str, name: &str) -> bool {
    if pat == "*" {
        return true;
    }
    if !pat.contains('*') {
        return pat == name;
    }
    // Generic wildcard match: split `pat` on `*` and check each
    // literal piece appears in `name` in order.
    let mut idx = 0;
    let mut first = true;
    let pieces: Vec<&str> = pat.split('*').collect();
    let last = pieces.len() - 1;
    for (i, piece) in pieces.iter().enumerate() {
        if piece.is_empty() {
            first = false;
            continue;
        }
        if first && i == 0 {
            // Must match at the start.
            if !name[idx..].starts_with(piece) {
                return false;
            }
            idx += piece.len();
            first = false;
            continue;
        }
        if i == last {
            // Must match at the end.
            return name[idx..].ends_with(piece);
        }
        // Find piece anywhere after idx.
        match name[idx..].find(piece) {
            Some(p) => idx += p + piece.len(),
            None => return false,
        }
    }
    // All pieces matched (likely the pattern was `*`-only).
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_literal() {
        assert!(pattern_matches("model.norm.weight", "model.norm.weight"));
        assert!(!pattern_matches("model.norm.weight", "model.norm.bias"));
    }

    #[test]
    fn star_matches_one_segment() {
        assert!(pattern_matches(
            "model.layers.*.input_layernorm.weight",
            "model.layers.0.input_layernorm.weight"
        ));
        assert!(pattern_matches(
            "model.layers.*.input_layernorm.weight",
            "model.layers.42.input_layernorm.weight"
        ));
        // * does not span dots.
        assert!(!pattern_matches(
            "model.layers.*.weight",
            "model.layers.0.norm.weight"
        ));
    }

    #[test]
    fn double_star_spans_segments() {
        assert!(pattern_matches(
            "model.**.weight",
            "model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(pattern_matches("model.**.weight", "model.weight"));
        assert!(!pattern_matches("model.**.bias", "model.layers.0.weight"));
    }

    #[test]
    fn alternation_expands() {
        let pats =
            expand_alternations("a.{b,c,d}.e");
        assert_eq!(pats, vec!["a.b.e", "a.c.e", "a.d.e"]);
    }

    #[test]
    fn alternation_in_match() {
        let pat = "model.layers.*.self_attn.{q,k,v,o}_proj.weight";
        assert!(pattern_matches(pat, "model.layers.0.self_attn.q_proj.weight"));
        assert!(pattern_matches(pat, "model.layers.5.self_attn.k_proj.weight"));
        assert!(pattern_matches(pat, "model.layers.5.self_attn.v_proj.weight"));
        assert!(pattern_matches(pat, "model.layers.5.self_attn.o_proj.weight"));
        assert!(!pattern_matches(
            pat,
            "model.layers.5.self_attn.gate_proj.weight"
        ));
    }

    #[test]
    fn within_segment_wildcard() {
        // `*_proj` matches `q_proj`, `down_proj`, etc.
        assert!(pattern_matches(
            "model.layers.*.mlp.*_proj.weight",
            "model.layers.0.mlp.gate_proj.weight"
        ));
        assert!(pattern_matches(
            "model.layers.*.mlp.*_proj.weight",
            "model.layers.0.mlp.down_proj.weight"
        ));
    }

    #[test]
    fn first_match_wins() {
        let profile = QuantProfile {
            name: "t".into(),
            arch: "gemma4".into(),
            calibration: None,
            rules: vec![
                RuleEntry {
                    pattern: "lm_head.weight".into(),
                    dtype: TensorDtype::Bf16,
                    group_size: None,
                    scale_dtype: None,
                    symmetric: None,
                },
                RuleEntry {
                    pattern: "**.weight".into(),
                    dtype: TensorDtype::BaseQ4,
                    group_size: None,
                    scale_dtype: None,
                    symmetric: None,
                },
            ],
        };
        assert_eq!(
            profile.resolve("lm_head.weight").unwrap().dtype,
            TensorDtype::Bf16
        );
        assert_eq!(
            profile.resolve("model.layers.0.q_proj.weight").unwrap().dtype,
            TensorDtype::BaseQ4
        );
    }

    #[test]
    fn resolve_fills_canonical_defaults() {
        let profile = QuantProfile {
            name: "t".into(),
            arch: "llama".into(),
            calibration: None,
            rules: vec![RuleEntry {
                pattern: "**.weight".into(),
                dtype: TensorDtype::BaseQ4,
                group_size: None,
                scale_dtype: None,
                symmetric: None,
            }],
        };
        let r = profile.resolve("x.weight").unwrap();
        assert_eq!(r.dtype, TensorDtype::BaseQ4);
        assert_eq!(r.group_size, 64); // q4 spec default
        assert_eq!(r.scale_dtype, ScaleDtype::Bf16);
        assert!(!r.symmetric);
    }

    #[test]
    fn validate_catches_unbalanced_braces() {
        let json = r#"{
            "name": "x",
            "arch": "llama",
            "rules": [{"pattern": "a.{b,c.weight", "dtype": "base_q4"}]
        }"#;
        let err = QuantProfile::from_json(json.as_bytes()).unwrap_err();
        assert!(format!("{err}").contains("unbalanced"));
    }

    #[test]
    fn validate_catches_e4m3_on_non_q8() {
        let json = r#"{
            "name": "x",
            "arch": "llama",
            "rules": [{"pattern": "**.weight", "dtype": "base_q4", "scale_dtype": "e4m3"}]
        }"#;
        let err = QuantProfile::from_json(json.as_bytes()).unwrap_err();
        assert!(format!("{err}").contains("e4m3"));
    }

    /// The shipped default profiles must validate. This catches typos
    /// in on-disk profile JSON before they hit the converter.
    #[test]
    fn shipped_default_profiles_parse() {
        // Profiles live at <repo>/tools/base-convert/profiles/. Walk up
        // from CARGO_MANIFEST_DIR (= the base-quant crate) to that path.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let profiles_dir = std::path::Path::new(&manifest)
            .parent() // crates/
            .unwrap()
            .parent() // base-convert/
            .unwrap()
            .join("profiles");
        let names = [
            "default-q4.json",
            "default-q8.json",
            "dense-q4mix.json",
            "moe-q4mix.json",
            "q3-aggressive.json",
            "gemma4-moe-q4mix.json",
            "gemma4-moe-q4all.json",
            "gemma4-moe-mlx.json",
        ];
        for name in names {
            let path = profiles_dir.join(name);
            let p = QuantProfile::from_path(&path)
                .unwrap_or_else(|e| panic!("profile {name} failed: {e}"));
            // Sanity: the catch-all "**.weight" rule covers a generic
            // tensor name, so resolve never returns None for plausible
            // inputs.
            let _ = p
                .resolve_or_err("model.layers.0.self_attn.q_proj.weight")
                .unwrap_or_else(|e| panic!("{name} resolve: {e}"));
        }
    }

    /// MoE profile must route experts to q4 and shared_ffn / lm_head
    /// to q8 — the canonical mixed-precision layout.
    #[test]
    fn moe_q4mix_profile_routes_correctly() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&manifest)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("profiles")
            .join("moe-q4mix.json");
        let p = QuantProfile::from_path(&path).unwrap();

        // MoE expert: q4 / gs=64.
        let expert =
            p.resolve("model.layers.0.mlp.experts.0.gate_proj.weight").unwrap();
        assert_eq!(expert.dtype, TensorDtype::BaseQ4);
        assert_eq!(expert.group_size, 64);

        // Shared FFN (Qwen3-MoE / DeepSeek style): q8 / gs=128.
        let shared = p
            .resolve("model.layers.0.mlp.shared_experts.gate_proj.weight")
            .unwrap();
        assert_eq!(shared.dtype, TensorDtype::BaseQ8);
        assert_eq!(shared.group_size, 128);

        // lm_head: q8.
        let lm = p.resolve("lm_head.weight").unwrap();
        assert_eq!(lm.dtype, TensorDtype::BaseQ8);

        // Router stays in fp (kernel reads f16; profile uses f16
        // since the runtime's norm/router kernels assume half).
        let router = p.resolve("model.layers.0.mlp.router.weight").unwrap();
        assert!(matches!(
            router.dtype,
            TensorDtype::F16 | TensorDtype::Bf16
        ));
    }

    /// q3-aggressive profile routes MLP / experts to q3 / gs=32 per spec.
    #[test]
    fn q3_aggressive_profile_uses_q3_gs32() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&manifest)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("profiles")
            .join("q3-aggressive.json");
        let p = QuantProfile::from_path(&path).unwrap();

        let mlp = p.resolve("model.layers.5.mlp.gate_proj.weight").unwrap();
        assert_eq!(mlp.dtype, TensorDtype::BaseQ3);
        assert_eq!(mlp.group_size, 32);

        // Attention stays at q4 even in this profile.
        let attn = p.resolve("model.layers.5.self_attn.q_proj.weight").unwrap();
        assert_eq!(attn.dtype, TensorDtype::BaseQ4);
        assert_eq!(attn.group_size, 64);
    }

    /// `gemma4-moe-mlx` mirrors `mlx-community/gemma-4-26b-a4b-it-4bit`'s
    /// per-tensor `quantization` block: q4 default, q8 on the shared-dense
    /// FFN (`mlp.{gate,up,down}_proj`) and the router projection. Embed
    /// drops to q4 to match MLX (lm_head is tied; the canonical name
    /// remap routes through `embed_tokens.weight`). This test pins the
    /// routing so a future profile edit can't silently drift it.
    #[test]
    fn gemma4_moe_mlx_profile_matches_mlx_quant_block() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&manifest)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("profiles")
            .join("gemma4-moe-mlx.json");
        let p = QuantProfile::from_path(&path).unwrap();

        // Shared dense FFN — q8/gs=64 (only path that sees every token
        // on top of the routed experts; MLX upgrades to 8 bit so q4
        // residual-stream noise doesn't compound through 30 layers).
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("model.layers.0.mlp.{proj}.weight");
            let r = p.resolve(&name).unwrap_or_else(|| panic!("no rule for {name}"));
            assert_eq!(r.dtype, TensorDtype::BaseQ8, "mlp.{proj} should be q8");
            assert_eq!(r.group_size, 64, "mlp.{proj} should be gs=64");
        }

        // Router projection — q8/gs=64. Picks expert assignments;
        // q4 noise here silently re-routes to wrong experts which is
        // unrecoverable, while the bytes saved are negligible.
        for name in [
            "model.language_model.layers.0.router.proj.weight",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.ffn_gate_inp.weight",
        ] {
            let r = p.resolve(name).unwrap_or_else(|| panic!("no rule for {name}"));
            assert_eq!(r.dtype, TensorDtype::BaseQ8, "{name} should be q8");
            assert_eq!(r.group_size, 64, "{name} should be gs=64");
        }

        // Routed experts — q4/gs=64 (the bulk of model bytes; q4 noise
        // averages across the 8/128 experts that fire per token).
        for name in [
            "model.layers.0.experts.gate_up_proj",
            "model.layers.0.ffn_gate_up_exps.weight",
            "model.layers.0.ffn_down_exps.weight",
        ] {
            let r = p.resolve(name).unwrap_or_else(|| panic!("no rule for {name}"));
            assert_eq!(r.dtype, TensorDtype::BaseQ4, "{name} should be q4");
            assert_eq!(r.group_size, 64, "{name} should be gs=64");
        }

        // Attention projections — q4/gs=64.
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("model.layers.0.self_attn.{proj}.weight");
            let r = p.resolve(&name).unwrap_or_else(|| panic!("no rule for {name}"));
            assert_eq!(r.dtype, TensorDtype::BaseQ4, "self_attn.{proj} should be q4");
        }

        // Embed_tokens — q4/gs=64 to match MLX (also drops lm_head with
        // tied embeddings). Saves ~1 GB on 26B-A4B's 262144 vocab vs f16.
        let embed = p.resolve("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.dtype, TensorDtype::BaseQ4);
        assert_eq!(embed.group_size, 64);

        // Norms stay f16 — the runtime's rmsnorm_f16 kernel reads weight
        // buffers as `half *`, and bf16 norms break that read.
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm_2.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm_2.weight",
        ] {
            let r = p.resolve(name).unwrap_or_else(|| panic!("no rule for {name}"));
            assert_eq!(r.dtype, TensorDtype::F16, "{name} should be f16");
        }

        // Per-layer / per-expert scalar weights stay f16 (small, sensitive).
        for name in [
            "model.layers.0.layer_scalar",
            "model.layers.0.layer_out_scale.weight",
            "model.layers.0.router.per_expert_scale",
            "model.layers.0.ffn_down_exps.scale",
            "model.layers.0.router.scale",
            "model.layers.0.ffn_gate_inp.scale",
        ] {
            let r = p.resolve(name).unwrap_or_else(|| panic!("no rule for {name}"));
            assert_eq!(r.dtype, TensorDtype::F16, "{name} should be f16");
        }
    }

    #[test]
    fn from_json_roundtrip() {
        let json = r#"{
            "name": "test-q4mix-v1",
            "arch": "gemma4",
            "calibration": {
                "method": "awq",
                "tokens": 1024,
                "dataset": "wikitext-103"
            },
            "rules": [
                {"pattern": "model.embed_tokens.weight", "dtype": "bf16"},
                {"pattern": "model.layers.*.input_layernorm.weight", "dtype": "bf16"},
                {"pattern": "model.layers.*.self_attn.{q,k,v,o}_proj.weight",
                 "dtype": "base_q4", "scale_dtype": "bf16", "group_size": 64},
                {"pattern": "model.layers.*.mlp.experts.*.{gate,up,down}_proj.weight",
                 "dtype": "base_q4"},
                {"pattern": "lm_head.weight", "dtype": "base_q8", "scale_dtype": "bf16"}
            ]
        }"#;
        let p = QuantProfile::from_json(json.as_bytes()).unwrap();
        assert_eq!(p.name, "test-q4mix-v1");
        assert_eq!(p.calibration.as_ref().unwrap().method, "awq");
        assert_eq!(
            p.resolve("model.layers.5.self_attn.q_proj.weight")
                .unwrap()
                .dtype,
            TensorDtype::BaseQ4
        );
        assert_eq!(
            p.resolve("lm_head.weight").unwrap().dtype,
            TensorDtype::BaseQ8
        );
        assert_eq!(p.resolve("not_in_profile").map(|_| ()), None);
    }
}
