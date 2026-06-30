//! The curated catalog of pre-converted `.base` models hosted in the basecompute
//! HF org.
//!
//! The catalog is fetched from a hosted URL at runtime ([`Catalog::load`]) so
//! it can be updated without shipping a new binary; the copy bundled via
//! `include_str!` is the offline/last-resort fallback. Resolution order is:
//! fresh on-disk cache → hosted fetch → stale cache → bundled.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Catalog shipped in the binary. Points only at public HF artifacts.
const BUNDLED: &str = include_str!("../catalog.json");

/// Hosted catalog, served raw from the public mirror's tree. Override with
/// `$BASERT_CATALOG_URL`; set `$BASERT_CATALOG_OFFLINE` to skip the network and
/// use the cache/bundled copy.
pub const DEFAULT_CATALOG_URL: &str =
    "https://raw.githubusercontent.com/basecompute/baseRT/main/base-convert/crates/base-hub/catalog.json";

/// Filename of the on-disk catalog cache, under the models dir.
const CACHE_FILE: &str = ".catalog-cache.json";
/// How long a cached catalog is served before re-fetching.
const CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
/// Network budget for a catalog fetch — kept tight so the CLI never hangs.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

fn catalog_url() -> String {
    std::env::var("BASERT_CATALOG_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_CATALOG_URL.to_string())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Catalog {
    pub schema: u32,
    #[serde(default)]
    pub updated: String,
    #[serde(default)]
    pub models: Vec<CatalogEntry>,
}

/// One pre-converted model the basecompute org hosts.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogEntry {
    /// Public id used with `basert pull` (e.g. `basecompute/llama-3.2-1b-q4`).
    pub id: String,
    /// HF repo the `.base` lives in.
    pub hf_repo: String,
    /// Filename of the `.base` within `hf_repo`.
    #[serde(default = "default_file")]
    pub file: String,
    #[serde(default = "default_revision")]
    pub revision: String,
    /// Original upstream model, informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// Quant identity → used as the on-disk variant directory.
    #[serde(default = "default_quant")]
    pub quant: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Optional integrity check for the downloaded `.base`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

fn default_file() -> String {
    "model.base".to_string()
}
fn default_revision() -> String {
    "main".to_string()
}
fn default_quant() -> String {
    "default".to_string()
}

impl Catalog {
    /// Parse the catalog bundled into the binary.
    pub fn bundled() -> Result<Self> {
        Self::from_json(BUNDLED).context("parsing bundled catalog.json")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Load the catalog for runtime use. Prefers a fresh hosted copy so the
    /// catalog can change without a new binary release. Never fails: any error
    /// (no network, bad JSON, …) falls through to the cache, then the bundled
    /// copy. `cache_dir` is typically the models dir.
    pub fn load(cache_dir: &Path) -> Self {
        let cache = cache_dir.join(CACHE_FILE);
        if std::env::var_os("BASERT_CATALOG_OFFLINE").is_some() {
            return Self::cache_or_bundled(&cache);
        }
        // A fresh cache short-circuits the network so most commands pay nothing.
        if let Some(c) = Self::fresh_cache(&cache) {
            return c;
        }
        match Self::fetch(&catalog_url()) {
            Some(c) => {
                let _ = Self::write_cache(&cache, &c);
                c
            }
            None => Self::cache_or_bundled(&cache),
        }
    }

    /// Parse the cache regardless of age, then fall back to the bundled copy,
    /// then to an empty catalog (load never fails).
    fn cache_or_bundled(cache: &Path) -> Self {
        Self::parse_file(cache)
            .or_else(|| Self::bundled().ok())
            .unwrap_or_else(|| Catalog {
                schema: 1,
                updated: String::new(),
                models: Vec::new(),
            })
    }

    fn fresh_cache(cache: &Path) -> Option<Self> {
        let age = std::fs::metadata(cache)
            .and_then(|m| m.modified())
            .ok()?
            .elapsed()
            .ok()?;
        (age < CACHE_TTL).then(|| Self::parse_file(cache)).flatten()
    }

    fn parse_file(path: &Path) -> Option<Self> {
        Self::from_json(&std::fs::read_to_string(path).ok()?).ok()
    }

    /// Write the cache atomically: serialize to a per-process temp file, then
    /// rename over the target. The rename is atomic on the same filesystem, so
    /// a concurrent `basert` process never observes a half-written cache and
    /// two writers just resolve to last-one-wins (both payloads are valid).
    fn write_cache(cache: &Path, cat: &Catalog) -> std::io::Result<()> {
        if let Some(dir) = cache.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let bytes = serde_json::to_vec_pretty(cat).unwrap_or_default();
        let tmp = cache.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, bytes)?;
        match std::fs::rename(&tmp, cache) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// Fetch + parse the hosted catalog. Returns `None` on any network/parse
    /// error, or if the payload isn't a schema-valid catalog (so we never cache
    /// or serve a CDN error page).
    fn fetch(url: &str) -> Option<Self> {
        let mut resp = ureq::get(url)
            .config()
            .timeout_global(Some(FETCH_TIMEOUT))
            .build()
            .call()
            .ok()?;
        let body = resp.body_mut().read_to_string().ok()?;
        Self::from_json(&body).ok().filter(|c| c.schema >= 1)
    }

    /// Find an entry by exact id, then case-insensitively.
    pub fn find(&self, id: &str) -> Option<&CatalogEntry> {
        self.models
            .iter()
            .find(|e| e.id == id)
            .or_else(|| self.models.iter().find(|e| e.id.eq_ignore_ascii_case(id)))
    }

    /// Check the catalog is internally consistent: every entry well-formed, and
    /// no duplicate `(id, quant)` pair (which would shadow in `find`/listing).
    /// Run by tests/CI to catch a malformed catalog edit before it ships and
    /// before clients fetch it.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for e in &self.models {
            if e.id.is_empty() {
                anyhow::bail!("catalog entry with empty id (hf_repo {:?})", e.hf_repo);
            }
            if e.hf_repo.is_empty() {
                anyhow::bail!("{}: empty hf_repo", e.id);
            }
            if !e.file.ends_with(".base") {
                anyhow::bail!("{}: file {:?} is not a .base", e.id, e.file);
            }
            if e.quant.is_empty() {
                anyhow::bail!("{}: empty quant", e.id);
            }
            if !seen.insert((e.id.as_str(), e.quant.as_str())) {
                anyhow::bail!("duplicate catalog entry: {} [{}]", e.id, e.quant);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_parses() {
        let cat = Catalog::bundled().expect("bundled catalog should parse");
        assert_eq!(cat.schema, 1);
    }

    #[test]
    fn bundled_catalog_is_valid() {
        Catalog::bundled()
            .unwrap()
            .validate()
            .expect("bundled catalog must be internally consistent");
    }

    #[test]
    fn validate_rejects_duplicate_id_quant() {
        let dup = Catalog::from_json(
            r#"{"schema":1,"models":[
                {"id":"basecompute/X","hf_repo":"basecompute/X","file":"a.base","quant":"default-q4"},
                {"id":"basecompute/X","hf_repo":"basecompute/X","file":"b.base","quant":"default-q4"}]}"#,
        )
        .unwrap();
        assert!(dup.validate().is_err(), "duplicate (id,quant) must fail");
    }

    #[test]
    fn cache_roundtrip_and_bundled_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join(".catalog-cache.json");

        // No cache yet → falls back to the populated bundled catalog.
        let c = Catalog::cache_or_bundled(&cache);
        assert!(!c.models.is_empty(), "bundled fallback should be populated");

        // Write a small catalog to the cache and read it back as fresh.
        let mini = Catalog::from_json(
            r#"{"schema":1,"updated":"x","models":[
                {"id":"basecompute/Test","hf_repo":"basecompute/Test",
                 "arch":"llama","quant":"default-q4"}]}"#,
        )
        .unwrap();
        Catalog::write_cache(&cache, &mini).unwrap();

        let fresh = Catalog::fresh_cache(&cache).expect("just-written cache is fresh");
        assert_eq!(fresh.models.len(), 1);
        assert_eq!(fresh.models[0].id, "basecompute/Test");

        // With a valid cache present, cache_or_bundled prefers it over bundled.
        assert_eq!(Catalog::cache_or_bundled(&cache).models.len(), 1);
    }

    #[test]
    fn bundled_catalog_is_populated_and_resolves_default_q4() {
        let cat = Catalog::bundled().unwrap();
        assert!(!cat.models.is_empty(), "bundled catalog should not be empty");
        // Every entry carries the fields the resolver/installer need.
        for e in &cat.models {
            assert!(e.id.starts_with("basecompute/"), "id: {}", e.id);
            assert!(e.file.ends_with(".base"), "file: {}", e.file);
            assert!(e.arch.is_some(), "arch missing for {}", e.id);
            assert!(e.sha256.is_some(), "sha256 missing for {}", e.id);
        }
        // find() returns the first id match — entries are ordered q4-first so
        // the recommended default is q4.
        let e = cat
            .find("basecompute/Llama-3.2-1B-Instruct")
            .expect("Llama-3.2-1B-Instruct should be catalogued");
        assert_eq!(e.quant, "default-q4");
        assert_eq!(e.arch.as_deref(), Some("llama"));
    }

    #[test]
    fn find_matches_exact_and_ci() {
        let cat = Catalog::from_json(
            r#"{"schema":1,"updated":"x","models":[
                {"id":"basecompute/llama-3.2-1b-q4","hf_repo":"basecompute/llama-3.2-1b-base",
                 "arch":"llama","quant":"default-q4"}]}"#,
        )
        .unwrap();
        assert!(cat.find("basecompute/llama-3.2-1b-q4").is_some());
        assert!(cat.find("BASECOMPUTE/Llama-3.2-1B-Q4").is_some());
        assert!(cat.find("nope").is_none());
        let e = cat.find("basecompute/llama-3.2-1b-q4").unwrap();
        assert_eq!(e.file, "model.base"); // default applied
        assert_eq!(e.revision, "main"); // default applied
    }

    #[test]
    fn malformed_catalog_errors() {
        assert!(Catalog::from_json("{ not json").is_err());
    }
}
