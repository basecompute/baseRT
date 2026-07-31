//! Model resolution and enumeration.
//!
//! A user-supplied id resolves to a [`ModelRef`] describing where the model
//! comes from. [`MergedRegistry`] layers the three sources in priority order
//! — already-installed local, then the BaseRT catalog (pre-converted `.base`),
//! then an arbitrary HF repo (convert-on-pull) — mirroring uzu's
//! `MergedRegistry`.

use crate::cache;
use crate::catalog::Catalog;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Extract the quant bit-width token (`q4`, `q8`, `q6`, …) embedded in a quant
/// identity: `default-q4` → `q4`, `cuda-q4mix` → `q4`, `q8` → `q8`. A `q`
/// followed by digits, where the digits don't run into more digits, so `q4`
/// never spuriously matches inside `q40`. Returns `None` when there is no such
/// token (an opaque identity that only matches by exact string).
pub fn quant_bits(quant: &str) -> Option<&str> {
    let b = quant.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'q' && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            let start = i;
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            return Some(&quant[start..j]);
        }
        i += 1;
    }
    None
}

/// Where a resolved id ultimately comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRef {
    /// Pre-converted `.base` hosted in the basecompute HF org — download directly,
    /// no local conversion.
    Catalog {
        id: String,
        hf_repo: String,
        file: String,
        revision: String,
        variant: String,
        arch: Option<String>,
        size: Option<u64>,
        sha256: Option<String>,
    },
    /// Arbitrary HF repo of source safetensors — download and convert locally.
    HuggingFace {
        id: String,
        repo: String,
        revision: String,
    },
    /// Already present in the local cache.
    Local {
        id: String,
        variant: String,
        path: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Local,
    Catalog,
    HuggingFace,
}

/// One row for `basert list`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub variant: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    pub installed: bool,
    pub source_kind: SourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
}

/// Common interface implemented by each individual source.
pub trait Registry {
    fn list(&self) -> Result<Vec<ModelEntry>>;
}

// ---------------------------------------------------------------------------
// Local
// ---------------------------------------------------------------------------

/// Scans the cache root for installed `model.base` artifacts.
pub struct LocalRegistry {
    root: PathBuf,
}

impl LocalRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Walk the cache tree (skipping the `.src` staging sub-tree) and return
    /// every installed artifact as a row. Reads only the `.base` header.
    fn scan(&self) -> Result<Vec<ModelEntry>> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        let mut stack = vec![self.root.clone()];
        while let Some(dir) = stack.pop() {
            let rd = match std::fs::read_dir(&dir) {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            for entry in rd.flatten() {
                let path = entry.path();
                let ft = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(_) => continue,
                };
                if ft.is_dir() {
                    // Skip the source-snapshot staging tree.
                    if path.file_name() == Some(std::ffi::OsStr::new(cache::SRC_STAGING)) {
                        continue;
                    }
                    stack.push(path);
                } else if ft.is_file()
                    && path.file_name() == Some(std::ffi::OsStr::new(cache::ARTIFACT_NAME))
                {
                    if let Some(row) = self.entry_for_artifact(&path) {
                        out.push(row);
                    }
                }
            }
        }
        out.sort_by(|a, b| (&a.id, &a.variant).cmp(&(&b.id, &b.variant)));
        Ok(out)
    }

    /// Build a row from a discovered `model.base`. The id is every path
    /// segment between the root and the variant dir; the variant is the
    /// artifact's parent directory.
    fn entry_for_artifact(&self, artifact: &Path) -> Option<ModelEntry> {
        let variant_dir = artifact.parent()?;
        let variant = variant_dir.file_name()?.to_str()?.to_string();
        let id_dir = variant_dir.parent()?;
        let rel = id_dir.strip_prefix(&self.root).ok()?;
        if rel.as_os_str().is_empty() {
            return None;
        }
        let id = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect::<Vec<_>>()
            .join("/");

        let size_bytes = std::fs::metadata(artifact).ok().map(|m| m.len());
        let (arch, quant) = match base_format::BaseReader::read_header(artifact) {
            Ok(h) => {
                let quant = if h.quant_profile.is_empty() {
                    Some(format!("{:?}", h.quant_scheme))
                } else {
                    Some(h.quant_profile.clone())
                };
                (Some(h.arch.clone()), quant)
            }
            Err(_) => (None, Some(variant.clone())),
        };

        Some(ModelEntry {
            id,
            variant,
            arch,
            quant,
            size_bytes,
            installed: true,
            source_kind: SourceKind::Local,
            path: Some(artifact.to_path_buf()),
        })
    }

    /// Path to an installed artifact for `id`+`variant`, if it exists.
    pub fn installed_path(&self, id: &str, variant: &str) -> Option<PathBuf> {
        let vdir = cache::variant_dir(&self.root, id, variant).ok()?;
        let artifact = cache::base_artifact_path(&vdir);
        artifact.exists().then_some(artifact)
    }
}

impl Registry for LocalRegistry {
    fn list(&self) -> Result<Vec<ModelEntry>> {
        self.scan()
    }
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

/// The curated set of pre-converted basecompute-org models.
pub struct CatalogRegistry {
    catalog: Catalog,
}

impl CatalogRegistry {
    pub fn bundled() -> Result<Self> {
        Ok(Self {
            catalog: Catalog::bundled()?,
        })
    }

    /// Load the hosted catalog (cached under `cache_dir`), falling back to the
    /// bundled copy offline. Never fails — see [`Catalog::load`].
    pub fn load(cache_dir: &Path) -> Self {
        Self {
            catalog: Catalog::load(cache_dir),
        }
    }

    pub fn from_catalog(catalog: Catalog) -> Self {
        Self { catalog }
    }

    /// The backend this client build serves. The `basert` CLI ships per
    /// platform: Apple builds pair with the Metal runtime, everything
    /// else with CUDA. Entries whose `backend` field names a different
    /// backend are hidden from resolution (and marked in `list`).
    pub fn client_backend() -> &'static str {
        if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cuda"
        }
    }

    fn backend_ok(e: &crate::catalog::CatalogEntry) -> bool {
        e.backend
            .as_deref()
            .map(|b| b == Self::client_backend())
            .unwrap_or(true)
    }

    fn entry_to_ref(e: &crate::catalog::CatalogEntry) -> ModelRef {
        ModelRef::Catalog {
            id: e.id.clone(),
            hf_repo: e.hf_repo.clone(),
            file: e.file.clone(),
            revision: e.revision.clone(),
            variant: e.quant.clone(),
            arch: e.arch.clone(),
            size: e.size,
            sha256: e.sha256.clone(),
        }
    }

    /// True if `entry_quant` satisfies the caller's `want` quant token,
    /// comparing on the embedded bit-width so a bare `q4`, a universal
    /// `default-q4`, and a backend-native `cuda-q4mix` all match `want=q4`
    /// (see [`quant_bits`]). Falls back to an exact string match for identities
    /// with no `qN` token.
    fn quant_matches(entry_quant: &str, want: &str) -> bool {
        let w = quant_bits(want).unwrap_or(want);
        match quant_bits(entry_quant) {
            Some(b) => b == w,
            None => entry_quant == want,
        }
    }

    /// Quant-agnostic, backend-aware resolve.
    pub fn resolve(&self, id: &str) -> Option<ModelRef> {
        self.resolve_variant(id, None)
    }

    /// Backend- and quant-aware catalog resolution. Among the entries sharing
    /// `id` (one per quant, and now optionally a per-backend variant of each),
    /// keep only those THIS client can run — `backend == client_backend()` or
    /// the universal `backend == None` — and, when the caller names a quant,
    /// only that quant. A backend-native bundle (e.g. the CUDA `q4mix` packing)
    /// is PREFERRED over the universal one, so a CUDA client fetches the
    /// CUDA-native `.base` instead of the portable packing the runtime may not
    /// even be able to load; it falls back to the universal entry when no
    /// backend-native variant is published. Selection happens BEFORE download.
    pub fn resolve_variant(&self, id: &str, want_quant: Option<&str>) -> Option<ModelRef> {
        self.resolve_with_status(id, want_quant).0
    }

    /// Backend- and quant-aware resolution, returning `(resolved, backend_locked)`.
    ///
    /// Among the rows sharing `id` (one per quant, plus optional per-backend
    /// variants), keep those THIS client can run — `backend == client_backend()`
    /// or the universal `backend == None` — matching the requested quant by its
    /// bit-width (so `q4` selects `default-q4` OR the native `cuda-q4mix`). A
    /// backend-native bundle is PREFERRED over the universal fallback so a CUDA
    /// client fetches the CUDA-native `.base` instead of the portable packing the
    /// runtime may not even load; an exact-id row beats a case-insensitive alias
    /// so the requested identity stays authoritative. For a quant-agnostic
    /// resolve the requested quant defaults to the first id row's bits, keeping
    /// `resolve(id)` quant-stable.
    ///
    /// `backend_locked` is true when a matching `(id, quant)` row exists but ONLY
    /// for a foreign backend: the caller must refuse pre-download rather than
    /// fall through to a convert-on-pull of a bundle this client can't run. When
    /// the id+quant simply isn't published, both fields are `(None, false)` and
    /// the caller may convert-on-pull from the source repo.
    pub fn resolve_with_status(&self, id: &str, want_quant: Option<&str>) -> (Option<ModelRef>, bool) {
        let is_exact = |e: &crate::catalog::CatalogEntry| e.id == id;
        let is_id = |e: &crate::catalog::CatalogEntry| e.id == id || e.id.eq_ignore_ascii_case(id);

        // Default quant for a bare resolve = the first id row's bits (exact id
        // preferred), so resolve(id) keeps returning that quant, now with the
        // backend-native variant of it when one is published.
        let first = self
            .catalog
            .models
            .iter()
            .find(|e| is_exact(e))
            .or_else(|| self.catalog.models.iter().find(|e| is_id(e)));
        let first = match first {
            Some(f) => f,
            None => return (None, false), // unknown id — caller may raw-HF it
        };
        let want_bits: Option<String> = match want_quant {
            Some(w) => Some(quant_bits(w).unwrap_or(w).to_string()),
            None => quant_bits(&first.quant).map(|s| s.to_string()),
        };
        let quant_ok = |e: &crate::catalog::CatalogEntry| match want_bits.as_deref() {
            None => true,
            Some(w) => Self::quant_matches(&e.quant, w),
        };

        let matched: Vec<&crate::catalog::CatalogEntry> =
            self.catalog.models.iter().filter(|e| is_id(e) && quant_ok(e)).collect();
        if matched.is_empty() {
            return (None, false); // this id+quant isn't published — convert-on-pull
        }
        let mut runnable: Vec<&crate::catalog::CatalogEntry> =
            matched.iter().copied().filter(|e| Self::backend_ok(e)).collect();
        if runnable.is_empty() {
            // Published, but only for a backend this client can't run.
            if let Some(e) = matched.first() {
                eprintln!(
                    "hub: '{}' requires the {} backend (this client is {}) — not resolving",
                    id,
                    e.backend.as_deref().unwrap_or("?"),
                    Self::client_backend()
                );
            }
            return (None, true);
        }
        // Rank: exact-id before case-insensitive alias; then backend-native
        // before universal; stable so catalog order breaks any remaining tie.
        runnable.sort_by_key(|e| (u8::from(!is_exact(e)), u8::from(e.backend.is_none())));
        (Some(Self::entry_to_ref(runnable[0])), false)
    }
}

impl Registry for CatalogRegistry {
    fn list(&self) -> Result<Vec<ModelEntry>> {
        Ok(self
            .catalog
            .models
            .iter()
            // Only advertise rows THIS client can actually pull — a backend-
            // qualified row (backend=cuda) must not appear in `list --remote` on
            // Metal, where resolution would filter it out anyway (it'd promise an
            // artifact the client can't select). Universal rows (backend=None)
            // pass on every backend. Mirrors resolve_variant's backend_ok gate.
            .filter(|e| Self::backend_ok(e))
            .map(|e| ModelEntry {
                id: e.id.clone(),
                variant: e.quant.clone(),
                arch: e.arch.clone(),
                quant: Some(e.quant.clone()),
                size_bytes: e.size,
                installed: false,
                source_kind: SourceKind::Catalog,
                path: None,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Merged
// ---------------------------------------------------------------------------

/// Resolution order: Local (installed) → Catalog → HuggingFace passthrough.
pub struct MergedRegistry {
    pub root: PathBuf,
    pub local: LocalRegistry,
    pub catalog: CatalogRegistry,
}

impl MergedRegistry {
    pub fn new(root: impl Into<PathBuf>, catalog: CatalogRegistry) -> Self {
        let root = root.into();
        Self {
            local: LocalRegistry::new(root.clone()),
            catalog,
            root,
        }
    }

    /// Build from the cache root + bundled catalog. Used by tests and any
    /// offline-only path; production code paths should prefer [`Self::load`].
    pub fn bundled() -> Result<Self> {
        let root = cache::models_dir()?;
        Ok(Self::new(root, CatalogRegistry::bundled()?))
    }

    /// Build from the cache root + the hosted catalog (cached + bundled
    /// fallback). This is the CLI entry point so a catalog update is picked up
    /// without a new binary. Errors only if the models dir can't be resolved.
    pub fn load() -> Result<Self> {
        let root = cache::models_dir()?;
        let catalog = CatalogRegistry::load(&root);
        Ok(Self::new(root, catalog))
    }

    /// Resolve an id to a concrete [`ModelRef`].
    ///
    /// `want_quant` is the quant the caller is after (`q4`, `q8`, …); when set,
    /// the already-installed shortcut only fires for *that* quant — so asking
    /// for q8 never silently returns an installed q4. With `force`, the
    /// shortcut is skipped entirely so the model is re-fetched/re-converted.
    /// A bare HF `org/model` id that isn't in the catalog falls through to
    /// convert-on-pull.
    pub fn resolve(
        &self,
        id: &str,
        revision: &str,
        want_quant: Option<&str>,
        force: bool,
    ) -> Result<ModelRef> {
        // Resolve the catalog entry FIRST (backend- + quant-aware) so the
        // installed-shortcut checks the RIGHT variant: a CUDA client must not be
        // handed a cached universal `default-q4` when the catalog directs it to
        // the native `cuda-q4mix`, and a backend-locked model must be refused
        // BEFORE any multi-GB download or convert-on-pull.
        let (cat, backend_locked) = self.catalog.resolve_with_status(id, want_quant);
        if let Some(cref) = cat {
            if !force {
                if let ModelRef::Catalog { variant, .. } = &cref {
                    if let Some(path) = self.local.installed_path(id, variant) {
                        return Ok(ModelRef::Local {
                            id: id.to_string(),
                            variant: variant.clone(),
                            path,
                        });
                    }
                }
            }
            return Ok(cref);
        }
        if backend_locked {
            // Published for this id+quant, but only for another backend. Refuse
            // rather than converting-on-pull a bundle this client can't run.
            anyhow::bail!(
                "hub: {id:?} has no bundle for the {} backend this client runs",
                CatalogRegistry::client_backend()
            );
        }
        // Not published for this id+quant. Honour an already-installed variant
        // (legacy `default-<quant>` layout) before converting-on-pull.
        if !force {
            if let Some(want) = want_quant {
                let variant = format!("default-{want}");
                if let Some(path) = self.local.installed_path(id, &variant) {
                    return Ok(ModelRef::Local {
                        id: id.to_string(),
                        variant,
                        path,
                    });
                }
            }
        }
        // Fall through to a raw HF repo (convert-on-pull). Require an `org/model`
        // shape so a typo'd catalog id doesn't silently become a failing fetch.
        if id.split('/').filter(|s| !s.is_empty()).count() < 2 {
            anyhow::bail!(
                "unknown model id {id:?}: not in the catalog and not an `org/model` HF repo"
            );
        }
        Ok(ModelRef::HuggingFace {
            id: id.to_string(),
            repo: id.to_string(),
            revision: revision.to_string(),
        })
    }

    /// Union of installed + (optionally) catalog rows, deduped by
    /// `(id, variant)` with local winning.
    pub fn list(&self, include_remote: bool) -> Result<Vec<ModelEntry>> {
        let mut rows = self.local.list().context("scanning local models")?;
        if include_remote {
            let have: std::collections::HashSet<(String, String)> = rows
                .iter()
                .map(|r| (r.id.clone(), r.variant.clone()))
                .collect();
            for r in self.catalog.list()? {
                if !have.contains(&(r.id.clone(), r.variant.clone())) {
                    rows.push(r);
                }
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    fn catalog_with_one() -> CatalogRegistry {
        CatalogRegistry::from_catalog(
            Catalog::from_json(
                r#"{"schema":1,"updated":"x","models":[
                    {"id":"basecompute/demo","hf_repo":"basecompute/demo-base",
                     "arch":"llama","quant":"default-q4"}]}"#,
            )
            .unwrap(),
        )
    }

    #[test]
    fn resolve_prefers_catalog_then_hf() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = MergedRegistry::new(tmp.path(), catalog_with_one());

        match reg.resolve("basecompute/demo", "main", None, false).unwrap() {
            ModelRef::Catalog {
                hf_repo, variant, ..
            } => {
                assert_eq!(hf_repo, "basecompute/demo-base");
                assert_eq!(variant, "default-q4");
            }
            other => panic!("expected Catalog, got {other:?}"),
        }
        match reg
            .resolve("meta-llama/Llama-3.2-1B", "main", None, false)
            .unwrap()
        {
            ModelRef::HuggingFace { repo, .. } => assert_eq!(repo, "meta-llama/Llama-3.2-1B"),
            other => panic!("expected HuggingFace, got {other:?}"),
        }
        assert!(reg.resolve("single-segment", "main", None, false).is_err());
    }

    #[test]
    fn resolve_installed_shortcut_is_quant_aware() {
        let tmp = tempfile::tempdir().unwrap();
        // Catalog publishes BOTH quants under one id.
        let cat = CatalogRegistry::from_catalog(
            Catalog::from_json(
                r#"{"schema":1,"updated":"x","models":[
                    {"id":"basecompute/demo","hf_repo":"basecompute/demo","file":"demo-Q4.base","arch":"llama","quant":"default-q4"},
                    {"id":"basecompute/demo","hf_repo":"basecompute/demo","file":"demo-Q8.base","arch":"llama","quant":"default-q8"}]}"#,
            )
            .unwrap(),
        );
        let reg = MergedRegistry::new(tmp.path(), cat);
        // Only the q4 variant is on disk.
        let vdir = cache::variant_dir(tmp.path(), "basecompute/demo", "default-q4").unwrap();
        std::fs::create_dir_all(&vdir).unwrap();
        std::fs::write(cache::base_artifact_path(&vdir), b"installed-q4").unwrap();

        // Asking for q4 → the installed artifact.
        assert!(matches!(
            reg.resolve("basecompute/demo", "main", Some("q4"), false).unwrap(),
            ModelRef::Local { .. }
        ));
        // Asking for q8 must NOT return the installed q4 — quant-aware catalog
        // resolution fetches the RIGHT quant (`demo-Q8.base`), not the wrong one.
        match reg
            .resolve("basecompute/demo", "main", Some("q8"), false)
            .unwrap()
        {
            ModelRef::Catalog { file, variant, .. } => {
                assert_eq!(file, "demo-Q8.base");
                assert_eq!(variant, "default-q8");
            }
            other => panic!("expected the q8 catalog entry, got {other:?}"),
        }
    }

    // Build a CatalogRegistry from inline JSON (test helper).
    fn catalog_json(models: &str) -> CatalogRegistry {
        CatalogRegistry::from_catalog(
            Catalog::from_json(&format!(r#"{{"schema":1,"updated":"x","models":[{models}]}}"#))
                .unwrap(),
        )
    }

    #[test]
    fn quant_bits_extracts_bit_token() {
        assert_eq!(quant_bits("default-q4"), Some("q4"));
        assert_eq!(quant_bits("cuda-q4mix"), Some("q4"));
        assert_eq!(quant_bits("q8"), Some("q8"));
        assert_eq!(quant_bits("cuda-q4"), Some("q4"));
        assert_eq!(quant_bits("bf16"), None); // no qN token
        assert_eq!(quant_bits("q40"), Some("q40")); // whole digit run, not "q4"
    }

    #[test]
    fn resolve_prefers_backend_native_over_universal() {
        // Universal `default-q4` and a backend-native `<be>-q4` (distinct cache
        // identity, matched by bits). The client fetches the native bundle.
        let be = CatalogRegistry::client_backend(); // "cuda" on Linux CI, "metal" on Apple
        let uni = r#"{"id":"basecompute/hybrid","hf_repo":"basecompute/hybrid","file":"hybrid-Q4.base","arch":"qwen35","quant":"default-q4"}"#.to_string();
        let nat = format!(
            r#"{{"id":"basecompute/hybrid","hf_repo":"basecompute/hybrid","file":"hybrid-Q4-{be}.base","arch":"qwen35","quant":"{be}-q4","backend":"{be}"}}"#
        );
        // Native listed after the universal, and before it — the preference must
        // hold regardless of catalog order.
        for rows in [format!("{uni},{nat}"), format!("{nat},{uni}")] {
            let reg = catalog_json(&rows);
            reg.catalog.validate().unwrap(); // both rows coexist (backend in key)
            match reg.resolve_variant("basecompute/hybrid", Some("q4")).unwrap() {
                ModelRef::Catalog { file, variant, .. } => {
                    assert_eq!(file, format!("hybrid-Q4-{be}.base"));
                    assert_eq!(variant, format!("{be}-q4")); // distinct cache dir
                }
                other => panic!("expected the backend-native entry, got {other:?}"),
            }
        }
    }

    #[test]
    fn resolve_falls_back_to_universal_when_no_native() {
        let reg = catalog_json(
            r#"{"id":"basecompute/demo","hf_repo":"basecompute/demo","file":"demo-Q4.base","arch":"llama","quant":"default-q4"}"#,
        );
        match reg.resolve_variant("basecompute/demo", Some("q4")).unwrap() {
            ModelRef::Catalog { file, .. } => assert_eq!(file, "demo-Q4.base"),
            other => panic!("expected the universal entry, got {other:?}"),
        }
        // A quant that isn't published: absent (not backend-locked) → the merged
        // resolver may convert-on-pull.
        assert_eq!(reg.resolve_with_status("basecompute/demo", Some("q8")), (None, false));
    }

    #[test]
    fn resolve_backend_locked_reports_status_not_absent() {
        let foreign = if CatalogRegistry::client_backend() == "cuda" { "metal" } else { "cuda" };
        let reg = catalog_json(&format!(
            r#"{{"id":"basecompute/locked","hf_repo":"basecompute/locked","file":"locked-Q4.base","arch":"llama","quant":"{foreign}-q4","backend":"{foreign}"}}"#
        ));
        // Published for this id+quant, but only for a foreign backend → refuse,
        // and flag it distinctly from "absent" so the caller errors (below).
        assert_eq!(reg.resolve_with_status("basecompute/locked", Some("q4")), (None, true));
    }

    #[test]
    fn merged_resolve_refuses_backend_locked_no_hf_fallthrough() {
        let foreign = if CatalogRegistry::client_backend() == "cuda" { "metal" } else { "cuda" };
        let tmp = tempfile::tempdir().unwrap();
        let reg = MergedRegistry::new(
            tmp.path(),
            catalog_json(&format!(
                r#"{{"id":"basecompute/locked","hf_repo":"basecompute/locked","file":"l-Q4.base","arch":"llama","quant":"{foreign}-q4","backend":"{foreign}"}}"#
            )),
        );
        // Must ERROR (backend-locked), not fall through to a raw HF download of a
        // bundle this client can't run.
        assert!(reg.resolve("basecompute/locked", "main", Some("q4"), false).is_err());
    }

    #[test]
    fn merged_resolve_shortcut_uses_backend_resolved_variant() {
        // A CUDA-native variant installed under its own cache dir is found; a
        // cached universal default-q4 does NOT shadow the native preference.
        let be = CatalogRegistry::client_backend();
        let tmp = tempfile::tempdir().unwrap();
        let reg = MergedRegistry::new(
            tmp.path(),
            catalog_json(&format!(
                r#"{{"id":"basecompute/h","hf_repo":"basecompute/h","file":"h-Q4.base","arch":"qwen35","quant":"default-q4"}},
                   {{"id":"basecompute/h","hf_repo":"basecompute/h","file":"h-Q4-{be}.base","arch":"qwen35","quant":"{be}-q4","backend":"{be}"}}"#
            )),
        );
        // Only the universal is on disk → resolve directs to the native CATALOG
        // ref (download), NOT the cached universal.
        let uni = cache::variant_dir(tmp.path(), "basecompute/h", "default-q4").unwrap();
        std::fs::create_dir_all(&uni).unwrap();
        std::fs::write(cache::base_artifact_path(&uni), b"universal").unwrap();
        match reg.resolve("basecompute/h", "main", Some("q4"), false).unwrap() {
            ModelRef::Catalog { variant, .. } => assert_eq!(variant, format!("{be}-q4")),
            other => panic!("cached universal must not shadow the native pick, got {other:?}"),
        }
        // Now install the native variant → the shortcut returns it as Local.
        let nat = cache::variant_dir(tmp.path(), "basecompute/h", &format!("{be}-q4")).unwrap();
        std::fs::create_dir_all(&nat).unwrap();
        std::fs::write(cache::base_artifact_path(&nat), b"native").unwrap();
        assert!(matches!(
            reg.resolve("basecompute/h", "main", Some("q4"), false).unwrap(),
            ModelRef::Local { variant, .. } if variant == format!("{be}-q4")
        ));
    }

    #[test]
    fn resolve_quant_agnostic_keeps_default_quant() {
        // default-q4 listed first, an optional q8 later. resolve(id) (no quant)
        // must stay on the q4 family, not jump to q8.
        let reg = catalog_json(
            r#"{"id":"basecompute/m","hf_repo":"basecompute/m","file":"m-Q4.base","arch":"llama","quant":"default-q4"},
               {"id":"basecompute/m","hf_repo":"basecompute/m","file":"m-Q8.base","arch":"llama","quant":"default-q8"}"#,
        );
        match reg.resolve("basecompute/m").unwrap() {
            ModelRef::Catalog { file, .. } => assert_eq!(file, "m-Q4.base"),
            other => panic!("resolve(id) must keep the default (q4), got {other:?}"),
        }
        // Explicit q8 still selects q8.
        match reg.resolve_variant("basecompute/m", Some("q8")).unwrap() {
            ModelRef::Catalog { file, .. } => assert_eq!(file, "m-Q8.base"),
            other => panic!("expected q8, got {other:?}"),
        }
    }

    #[test]
    fn resolve_prefers_exact_id_over_case_alias() {
        // Two case-distinct ids: an exact request must resolve to the exact id's
        // row, not a differently-cased alias listed first.
        let reg = catalog_json(
            r#"{"id":"basecompute/CamelModel","hf_repo":"basecompute/alias","file":"alias-Q4.base","arch":"llama","quant":"default-q4"},
               {"id":"basecompute/camelmodel","hf_repo":"basecompute/exact","file":"exact-Q4.base","arch":"llama","quant":"default-q4"}"#,
        );
        match reg.resolve_variant("basecompute/camelmodel", Some("q4")).unwrap() {
            ModelRef::Catalog { hf_repo, .. } => assert_eq!(hf_repo, "basecompute/exact"),
            other => panic!("exact id must win, got {other:?}"),
        }
    }

    #[test]
    fn local_scan_walks_tree_and_skips_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // An installed (header-less / junk) artifact at a nested id.
        let vdir = cache::variant_dir(root, "basecompute/demo", "default-q4").unwrap();
        std::fs::create_dir_all(&vdir).unwrap();
        std::fs::write(cache::base_artifact_path(&vdir), b"junk-not-a-header").unwrap();
        // A staging artifact under .src must be ignored.
        let sdir = cache::hf_staging_dir(root).join("models--meta--Foo/snapshots/main");
        std::fs::create_dir_all(&sdir).unwrap();
        std::fs::write(cache::base_artifact_path(&sdir), b"junk").unwrap();

        let rows = LocalRegistry::new(root).list().unwrap();
        assert_eq!(rows.len(), 1, "staging tree must be skipped: {rows:?}");
        let r = &rows[0];
        assert_eq!(r.id, "basecompute/demo");
        assert_eq!(r.variant, "default-q4");
        assert!(r.installed);
        assert_eq!(r.source_kind, SourceKind::Local);
        // Unreadable header → arch falls back to None, quant to the variant.
        assert_eq!(r.arch, None);
        assert_eq!(r.quant.as_deref(), Some("default-q4"));
    }

    #[test]
    fn installed_local_shadows_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = MergedRegistry::new(tmp.path(), catalog_with_one());
        // Materialize an installed artifact for the catalog id+variant.
        let vdir = cache::variant_dir(tmp.path(), "basecompute/demo", "default-q4").unwrap();
        std::fs::create_dir_all(&vdir).unwrap();
        std::fs::write(cache::base_artifact_path(&vdir), b"not a real base").unwrap();

        match reg.resolve("basecompute/demo", "main", None, false).unwrap() {
            ModelRef::Local { variant, .. } => assert_eq!(variant, "default-q4"),
            other => panic!("expected Local, got {other:?}"),
        }
        // With force, the local shortcut is skipped.
        assert!(matches!(
            reg.resolve("basecompute/demo", "main", None, true).unwrap(),
            ModelRef::Catalog { .. }
        ));
    }
}
