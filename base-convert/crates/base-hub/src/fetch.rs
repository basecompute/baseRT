//! File fetching, abstracted so tests can run without network.
//!
//! [`HfFetcher`] is the real implementation over hf-hub's blocking API; its
//! built-in progress bars cover downloads. [`MockFetcher`] copies from a
//! local fixture directory so the pull/convert pipeline can be exercised in
//! CI with no HuggingFace access.

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Download retries hf-hub performs on transient failures (peer disconnects,
/// truncated chunks — routine on multi-GB model pulls). hf-hub's own default is
/// `0`, which turns the very first network hiccup into a hard failure; its retry
/// loop resumes mid-file via Range headers, so opting in makes large pulls
/// resilient. Override with `$BASERT_HF_MAX_RETRIES` (`0` disables retries).
const DEFAULT_HF_MAX_RETRIES: usize = 5;

/// Resolve the retry count from `$BASERT_HF_MAX_RETRIES`, falling back to
/// [`DEFAULT_HF_MAX_RETRIES`] when the var is unset or unparseable.
fn resolve_max_retries() -> usize {
    std::env::var("BASERT_HF_MAX_RETRIES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_HF_MAX_RETRIES)
}

/// Fetches model files from a remote (or, for tests, a fixture) source.
pub trait Fetcher {
    /// Download `filename` from `repo` at `revision`; returns the local path.
    fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf>;

    /// List the filenames available in `repo` at `revision`.
    fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>>;
}

/// Real fetcher backed by hf-hub's synchronous API. Reads the HF token from
/// `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`, falling back to the cached login
/// token (`~/.cache/huggingface/token`).
pub struct HfFetcher {
    api: hf_hub::api::sync::Api,
}

impl HfFetcher {
    pub fn new() -> Result<Self> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .with_retries(resolve_max_retries());
        if let Some(tok) = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
            .filter(|s| !s.is_empty())
        {
            builder = builder.with_token(Some(tok));
        }
        let api = builder
            .build()
            .context("initializing HuggingFace API client")?;
        Ok(Self { api })
    }

    fn repo(&self, repo: &str, revision: &str) -> hf_hub::api::sync::ApiRepo {
        self.api.repo(hf_hub::Repo::with_revision(
            repo.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ))
    }
}

impl Fetcher for HfFetcher {
    fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
        self.repo(repo, revision)
            .get(filename)
            .with_context(|| format!("downloading {filename} from {repo}@{revision}"))
    }

    fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
        let info = self
            .repo(repo, revision)
            .info()
            .with_context(|| format!("fetching repo info for {repo}@{revision}"))?;
        Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
    }
}

/// Test fetcher that serves files from a local fixture directory laid out as
/// `<root>/<repo>/<filename>` (repo slashes become nested dirs).
pub struct MockFetcher {
    pub root: PathBuf,
}

impl MockFetcher {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn repo_dir(&self, repo: &str) -> PathBuf {
        let mut p = self.root.clone();
        for seg in repo.split('/') {
            p.push(seg);
        }
        p
    }
}

impl Fetcher for MockFetcher {
    fn get_file(&self, repo: &str, _revision: &str, filename: &str) -> Result<PathBuf> {
        let path = self.repo_dir(repo).join(filename);
        if !path.exists() {
            anyhow::bail!("mock fixture missing: {}", path.display());
        }
        Ok(path)
    }

    fn list_files(&self, repo: &str, _revision: &str) -> Result<Vec<String>> {
        let dir = self.repo_dir(repo);
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("listing mock repo {}", dir.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    out.push(name.to_string());
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All assertions live in one test: they mutate the shared process env, so
    // splitting them into separate `#[test]` fns would race under Rust's
    // parallel test runner. Sequential mutation within a single fn is safe.
    #[test]
    fn resolve_max_retries_reads_env_with_default_fallback() {
        let prev = std::env::var("BASERT_HF_MAX_RETRIES").ok();

        // Unset -> the opted-in default (must be > 0, else the retry loop that
        // makes multi-GB pulls resilient stays disabled — the bug this fixes).
        std::env::remove_var("BASERT_HF_MAX_RETRIES");
        assert!(DEFAULT_HF_MAX_RETRIES > 0, "retries must be opted in by default");
        assert_eq!(resolve_max_retries(), DEFAULT_HF_MAX_RETRIES);

        // A valid override is honored.
        std::env::set_var("BASERT_HF_MAX_RETRIES", "9");
        assert_eq!(resolve_max_retries(), 9);

        // "0" is a deliberate opt-out (fail fast), not a fallback.
        std::env::set_var("BASERT_HF_MAX_RETRIES", "0");
        assert_eq!(resolve_max_retries(), 0);

        // Surrounding whitespace is tolerated.
        std::env::set_var("BASERT_HF_MAX_RETRIES", "  3 ");
        assert_eq!(resolve_max_retries(), 3);

        // Garbage falls back to the default rather than panicking.
        std::env::set_var("BASERT_HF_MAX_RETRIES", "not-a-number");
        assert_eq!(resolve_max_retries(), DEFAULT_HF_MAX_RETRIES);

        match prev {
            Some(v) => std::env::set_var("BASERT_HF_MAX_RETRIES", v),
            None => std::env::remove_var("BASERT_HF_MAX_RETRIES"),
        }
    }
}
