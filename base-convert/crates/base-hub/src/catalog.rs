//! The curated catalog of pre-converted `.base` models hosted in the basecompute
//! HF org. Bundled into the binary via `include_str!` and parsed at startup;
//! a future hosted catalog can serve the same schema at a URL.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Catalog shipped in the binary. Points only at public HF artifacts.
const BUNDLED: &str = include_str!("../catalog.json");

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

    /// Find an entry by exact id, then case-insensitively.
    pub fn find(&self, id: &str) -> Option<&CatalogEntry> {
        self.models
            .iter()
            .find(|e| e.id == id)
            .or_else(|| self.models.iter().find(|e| e.id.eq_ignore_ascii_case(id)))
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
