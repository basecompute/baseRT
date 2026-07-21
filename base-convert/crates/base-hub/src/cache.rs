//! Canonical on-disk layout for hub-managed models.
//!
//! ```text
//! $BASERT_MODELS_DIR/  (default ~/.cache/baseRT/models)
//!   <namespace>/<repo>/<variant>/model.base   ← artifact the runtime loads
//!   <namespace>/<repo>/<variant>/hub.json      ← provenance sidecar
//!   .src/hf/models--<org>--<repo>/             ← hf-hub download staging
//! ```
//!
//! The fixed `model.base` filename makes server discovery a trivial
//! `**/model.base` glob, and the per-`<variant>` directory lets multiple
//! quantizations of the same source repo coexist.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

/// Fixed artifact filename inside every variant directory.
pub const ARTIFACT_NAME: &str = "model.base";
/// Provenance sidecar filename inside every variant directory.
pub const SIDECAR_NAME: &str = "hub.json";
/// Sub-tree holding in-flight HF downloads prior to installation.
pub const SRC_STAGING: &str = ".src";

/// Resolve the models root: `$BASERT_MODELS_DIR` or `~/.cache/baseRT/models`.
pub fn models_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("BASERT_MODELS_DIR") {
        let p = PathBuf::from(dir);
        if p.as_os_str().is_empty() {
            bail!("BASERT_MODELS_DIR is set but empty");
        }
        return Ok(p);
    }
    let cache = dirs::cache_dir()
        .context("could not determine a cache directory (set BASERT_MODELS_DIR)")?;
    Ok(cache.join("baseRT").join("models"))
}

/// Convert a model id (`org/model`, possibly with extra path segments) into a
/// safe relative path. Rejects absolute paths, `..` traversal, Windows
/// prefixes, and empty segments so a hostile id can never escape the cache
/// root.
pub fn id_to_relpath(id: &str) -> Result<PathBuf> {
    let id = id.trim();
    if id.is_empty() {
        bail!("empty model id");
    }
    if id.starts_with('/') || id.starts_with('\\') {
        bail!("model id must not be absolute: {id:?}");
    }
    let mut out = PathBuf::new();
    let mut segments = 0;
    for raw in id.split(['/', '\\']) {
        if raw.is_empty() {
            bail!("model id has an empty path segment: {id:?}");
        }
        // Reject anything that isn't a plain normal component.
        let comp_path = Path::new(raw);
        let mut comps = comp_path.components();
        match (comps.next(), comps.next()) {
            (Some(Component::Normal(s)), None) if s == std::ffi::OsStr::new(raw) => {}
            _ => bail!("invalid segment {raw:?} in model id {id:?}"),
        }
        out.push(raw);
        segments += 1;
    }
    if segments < 1 {
        bail!("model id has no usable segments: {id:?}");
    }
    Ok(out)
}

/// Final directory for a given id + variant: `<root>/<id-relpath>/<variant>/`.
pub fn variant_dir(root: &Path, id: &str, variant: &str) -> Result<PathBuf> {
    let rel = id_to_relpath(id)?;
    let variant = sanitize_variant(variant)?;
    Ok(root.join(rel).join(variant))
}

/// The canonical artifact path inside a variant directory.
pub fn base_artifact_path(variant_dir: &Path) -> PathBuf {
    variant_dir.join(ARTIFACT_NAME)
}

/// Root for hf-hub downloads: `<root>/.src/hf`. Downloads land here (in
/// hf-hub's own `models--<org>--<repo>/{blobs,snapshots,refs}` layout) instead
/// of the user's global HuggingFace cache, so a pulled artifact is never
/// duplicated across two caches and a finished install is a same-filesystem
/// rename, not a copy. Kept apart from the canonical tree so an interrupted
/// download never pollutes it and `list` never trips over loose safetensors;
/// partial downloads left behind by a failed pull are resumed on retry.
pub fn hf_staging_dir(root: &Path) -> PathBuf {
    root.join(SRC_STAGING).join("hf")
}

/// A single path component derived from free-form text (variant / revision).
/// Replaces separators so the value can never introduce a new directory level
/// or escape upward.
fn sanitize_variant(s: &str) -> Result<String> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty variant/revision");
    }
    let cleaned: String = s
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '\0' => '-',
            c => c,
        })
        .collect();
    if cleaned == "." || cleaned == ".." {
        bail!("invalid variant/revision: {s:?}");
    }
    Ok(cleaned)
}

/// Provenance written next to every installed artifact. Distinct from the
/// `.base` header: it records *how* the model got here (resolution source,
/// upstream repo, revision), which the header doesn't carry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubSidecar {
    pub id: String,
    /// `catalog` | `huggingface` | `local`.
    pub source_kind: String,
    /// HF repo the `.base` (catalog) or source safetensors (huggingface)
    /// were fetched from.
    pub hf_repo: String,
    /// Original upstream model the artifact derives from, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
    pub revision: String,
    /// Quant profile name (or scheme) used / advertised.
    pub variant: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub pulled_at: String,
    /// sha256 of the installed `model.base`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_sha256: Option<String>,
}

/// Write the provenance sidecar into a variant directory (creates the dir).
pub fn write_sidecar(variant_dir: &Path, s: &HubSidecar) -> Result<()> {
    std::fs::create_dir_all(variant_dir)
        .with_context(|| format!("creating {}", variant_dir.display()))?;
    let path = variant_dir.join(SIDECAR_NAME);
    let json = serde_json::to_string_pretty(s)?;
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Read the provenance sidecar from a variant directory, if present.
pub fn read_sidecar(variant_dir: &Path) -> Result<Option<HubSidecar>> {
    let path = variant_dir.join(SIDECAR_NAME);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_dir_honors_env() {
        // SAFETY: single-threaded test process.
        std::env::set_var("BASERT_MODELS_DIR", "/tmp/baseRT-test-models");
        assert_eq!(
            models_dir().unwrap(),
            PathBuf::from("/tmp/baseRT-test-models")
        );
        std::env::remove_var("BASERT_MODELS_DIR");
    }

    #[test]
    fn id_to_relpath_accepts_normal_ids() {
        assert_eq!(
            id_to_relpath("basecompute/llama-3.2-1b-q4").unwrap(),
            PathBuf::from("basecompute").join("llama-3.2-1b-q4")
        );
        assert_eq!(
            id_to_relpath("meta-llama/Llama-3.2-1B").unwrap(),
            PathBuf::from("meta-llama").join("Llama-3.2-1B")
        );
    }

    #[test]
    fn id_to_relpath_rejects_traversal() {
        for bad in [
            "",
            "/abs/path",
            "..",
            "../escape",
            "basecompute/../../etc",
            "a//b",
            "basecompute/..",
            "\\\\windows",
        ] {
            assert!(id_to_relpath(bad).is_err(), "expected reject: {bad:?}");
        }
    }

    #[test]
    fn variant_dir_is_under_root() {
        let root = Path::new("/models");
        let d = variant_dir(root, "basecompute/llama", "default-q4").unwrap();
        assert_eq!(d, Path::new("/models/basecompute/llama/default-q4"));
        assert!(d.starts_with(root));
    }

    #[test]
    fn variant_with_slash_stays_one_level() {
        let root = Path::new("/models");
        // A scheme like "base_q4" or a colon'd value collapses to one segment.
        let d = variant_dir(root, "basecompute/llama", "q4:gs64/v2").unwrap();
        assert_eq!(d, Path::new("/models/basecompute/llama/q4-gs64-v2"));
    }

    #[test]
    fn hf_staging_is_under_dot_src() {
        let root = Path::new("/models");
        let d = hf_staging_dir(root);
        assert_eq!(d, Path::new("/models/.src/hf"));
        // Must stay inside the `.src` sub-tree that `list` skips.
        assert!(d.starts_with(root.join(SRC_STAGING)));
    }

    #[test]
    fn sidecar_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let vdir = tmp.path().join("basecompute/llama/default-q4");
        let s = HubSidecar {
            id: "basecompute/llama".into(),
            source_kind: "huggingface".into(),
            hf_repo: "meta-llama/Llama-3.2-1B".into(),
            source_repo: Some("meta-llama/Llama-3.2-1B".into()),
            revision: "main".into(),
            variant: "default-q4".into(),
            profile: Some("default-q4".into()),
            pulled_at: "2026-06-24T00:00:00Z".into(),
            base_sha256: None,
        };
        write_sidecar(&vdir, &s).unwrap();
        let back = read_sidecar(&vdir).unwrap().unwrap();
        assert_eq!(back.hf_repo, "meta-llama/Llama-3.2-1B");
        assert_eq!(back.variant, "default-q4");
        // Missing sidecar → None.
        assert!(read_sidecar(tmp.path()).unwrap().is_none());
    }
}
