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

    pub fn from_catalog(catalog: Catalog) -> Self {
        Self { catalog }
    }

    pub fn resolve(&self, id: &str) -> Option<ModelRef> {
        let e = self.catalog.find(id)?;
        Some(ModelRef::Catalog {
            id: e.id.clone(),
            hf_repo: e.hf_repo.clone(),
            file: e.file.clone(),
            revision: e.revision.clone(),
            variant: e.quant.clone(),
            arch: e.arch.clone(),
            size: e.size,
            sha256: e.sha256.clone(),
        })
    }
}

impl Registry for CatalogRegistry {
    fn list(&self) -> Result<Vec<ModelEntry>> {
        Ok(self
            .catalog
            .models
            .iter()
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

    /// Build from the cache root + bundled catalog.
    pub fn bundled() -> Result<Self> {
        let root = cache::models_dir()?;
        Ok(Self::new(root, CatalogRegistry::bundled()?))
    }

    /// Resolve an id to a concrete [`ModelRef`].
    ///
    /// With `force`, the already-installed shortcut is skipped so the model
    /// is re-fetched/re-converted. A bare HF `org/model` id that isn't in the
    /// catalog falls through to convert-on-pull.
    pub fn resolve(&self, id: &str, revision: &str, force: bool) -> Result<ModelRef> {
        if !force {
            if let Some(ModelRef::Catalog { variant, .. }) = self.catalog.resolve(id) {
                if let Some(path) = self.local.installed_path(id, &variant) {
                    return Ok(ModelRef::Local {
                        id: id.to_string(),
                        variant,
                        path,
                    });
                }
            }
        }
        if let Some(r) = self.catalog.resolve(id) {
            return Ok(r);
        }
        // Fall through to a raw HF repo. Require an `org/model` shape so a
        // typo'd catalog id doesn't silently become a (failing) HF fetch.
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

        match reg.resolve("basecompute/demo", "main", false).unwrap() {
            ModelRef::Catalog {
                hf_repo, variant, ..
            } => {
                assert_eq!(hf_repo, "basecompute/demo-base");
                assert_eq!(variant, "default-q4");
            }
            other => panic!("expected Catalog, got {other:?}"),
        }
        match reg
            .resolve("meta-llama/Llama-3.2-1B", "main", false)
            .unwrap()
        {
            ModelRef::HuggingFace { repo, .. } => assert_eq!(repo, "meta-llama/Llama-3.2-1B"),
            other => panic!("expected HuggingFace, got {other:?}"),
        }
        assert!(reg.resolve("single-segment", "main", false).is_err());
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
        let sdir = cache::src_staging_dir(root, "meta/Foo", "main").unwrap();
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

        match reg.resolve("basecompute/demo", "main", false).unwrap() {
            ModelRef::Local { variant, .. } => assert_eq!(variant, "default-q4"),
            other => panic!("expected Local, got {other:?}"),
        }
        // With force, the local shortcut is skipped.
        assert!(matches!(
            reg.resolve("basecompute/demo", "main", true).unwrap(),
            ModelRef::Catalog { .. }
        ));
    }
}
