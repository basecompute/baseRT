//! File fetching, abstracted so tests can run without network.
//!
//! [`HfFetcher`] is the real implementation over hf-hub's blocking API; its
//! built-in progress bars cover downloads. [`MockFetcher`] copies from a
//! local fixture directory so the pull/convert pipeline can be exercised in
//! CI with no HuggingFace access.

use anyhow::{Context, Result};
use std::path::PathBuf;

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
        let mut builder = hf_hub::api::sync::ApiBuilder::new().with_progress(true);
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
