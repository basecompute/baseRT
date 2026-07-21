//! File fetching, abstracted so tests can run without network.
//!
//! [`HfFetcher`] is the real implementation over hf-hub's blocking API; its
//! built-in progress bars cover downloads. [`MockFetcher`] copies from a
//! local fixture directory so the pull/convert pipeline can be exercised in
//! CI with no HuggingFace access.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

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

    /// The staging directory this fetcher owns for `repo` — every byte it
    /// downloaded for that repo lives under it, and nothing else does. `None`
    /// when the fetcher serves files it does not own (fixtures, a shared
    /// cache): those must be copied on install and never deleted.
    fn staging_dir(&self, repo: &str) -> Option<PathBuf> {
        let _ = repo;
        None
    }
}

/// Real fetcher backed by hf-hub's synchronous API. Reads the HF token from
/// `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`, falling back to the cached login
/// token (`~/.cache/huggingface/token`).
///
/// Downloads land in a private staging directory (normally
/// `<models root>/.src/hf` — see [`crate::cache::hf_staging_dir`]), NOT the
/// user's global HuggingFace cache: multi-GB `.base` artifacts would otherwise
/// persist there as a second copy after installation. Keeping staging on the
/// same filesystem as the models root also lets installs move (rename) the
/// downloaded bytes instead of copying them.
pub struct HfFetcher {
    api: hf_hub::api::sync::Api,
    staging_root: PathBuf,
}

impl HfFetcher {
    pub fn new(staging_root: impl Into<PathBuf>) -> Result<Self> {
        let staging_root = staging_root.into();
        // `ApiBuilder::new()` snapshots the login token from the default HF
        // cache location before `with_cache_dir` re-points downloads at our
        // staging dir, so `~/.cache/huggingface/token` keeps working.
        let mut builder = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .with_retries(resolve_max_retries())
            .with_cache_dir(staging_root.clone());
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
        Ok(Self { api, staging_root })
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

    fn staging_dir(&self, repo: &str) -> Option<PathBuf> {
        // hf-hub keeps everything for a repo under `models--<org>--<repo>`.
        let folder = hf_hub::Repo::model(repo.to_string()).folder_name();
        Some(self.staging_root.join(folder))
    }
}

/// Install a file returned by [`Fetcher::get_file`] at `dst`, leaving at most
/// one surviving copy of the bytes.
///
/// When `src` sits inside the fetcher's own staging tree for `repo`, the
/// underlying blob is *moved* (symlinks resolved first — hf-hub's snapshot
/// paths are pointers into `blobs/`), so no duplicate ever exists; a rename
/// that fails (e.g. across filesystems) degrades to a copy, and the source is
/// then reclaimed by [`cleanup_staging`]. Files the fetcher does not own
/// (fixtures, shared caches) are copied and left untouched.
pub fn install_file(fetcher: &dyn Fetcher, repo: &str, src: &Path, dst: &Path) -> Result<()> {
    let owned = fetcher
        .staging_dir(repo)
        .is_some_and(|dir| src.starts_with(&dir));
    if owned {
        // Resolve the snapshot symlink to the actual blob before renaming;
        // renaming the symlink itself would strand the payload in staging.
        let real = std::fs::canonicalize(src)
            .with_context(|| format!("resolving {}", src.display()))?;
        if std::fs::rename(&real, dst).is_ok() {
            return Ok(());
        }
        // Rename can fail across filesystems; fall through to a copy (the
        // staged source is removed later by `cleanup_staging`).
    }
    std::fs::copy(src, dst)
        .with_context(|| format!("installing {} into {}", src.display(), dst.display()))?;
    Ok(())
}

/// Delete everything the fetcher staged for `repo`. Call only once the
/// installed artifact is in place (or known-bad): partial downloads left
/// behind by a failed pull are exactly what makes resume-on-retry work, so
/// failures should skip this. A no-op for fetchers that own no staging.
pub fn cleanup_staging(fetcher: &dyn Fetcher, repo: &str) {
    if let Some(dir) = fetcher.staging_dir(repo) {
        if dir.exists() {
            let _ = std::fs::remove_dir_all(&dir);
        }
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

    /// Fetcher that owns an hf-hub-style staging tree:
    /// `<staging>/models--<org>--<repo>/blobs/<etag>` with
    /// `snapshots/<rev>/<file>` symlinks pointing at the blobs — the layout
    /// `HfFetcher` produces.
    struct StagedFetcher {
        staging: PathBuf,
    }

    impl StagedFetcher {
        fn repo_dir(&self, repo: &str) -> PathBuf {
            self.staging.join(format!("models--{}", repo.replace('/', "--")))
        }

        /// Materialize a staged download of `filename` with `bytes`.
        fn stage(&self, repo: &str, revision: &str, filename: &str, bytes: &[u8]) -> PathBuf {
            let rdir = self.repo_dir(repo);
            let blobs = rdir.join("blobs");
            let snap = rdir.join("snapshots").join(revision);
            std::fs::create_dir_all(&blobs).unwrap();
            std::fs::create_dir_all(&snap).unwrap();
            let blob = blobs.join(format!("etag-{filename}"));
            std::fs::write(&blob, bytes).unwrap();
            let pointer = snap.join(filename);
            std::os::unix::fs::symlink(&blob, &pointer).unwrap();
            pointer
        }
    }

    impl Fetcher for StagedFetcher {
        fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
            let p = self
                .repo_dir(repo)
                .join("snapshots")
                .join(revision)
                .join(filename);
            anyhow::ensure!(p.exists(), "not staged: {}", p.display());
            Ok(p)
        }

        fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
            let dir = self.repo_dir(repo).join("snapshots").join(revision);
            let mut out = Vec::new();
            for e in std::fs::read_dir(dir)? {
                out.push(e?.file_name().to_string_lossy().into_owned());
            }
            Ok(out)
        }

        fn staging_dir(&self, repo: &str) -> Option<PathBuf> {
            Some(self.repo_dir(repo))
        }
    }

    #[test]
    fn install_moves_owned_blob_then_cleanup_leaves_one_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let f = StagedFetcher { staging: tmp.path().join("staging") };
        let src = f.stage("org/m", "main", "m-Q4.base", b"payload");
        let dst = tmp.path().join("model.base");

        install_file(&f, "org/m", &src, &dst).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"payload");
        // The blob was moved, not copied: the staged payload is gone (only a
        // dangling pointer symlink may remain until cleanup).
        let blob = f.repo_dir("org/m").join("blobs").join("etag-m-Q4.base");
        assert!(!blob.exists(), "blob must be moved out of staging");

        cleanup_staging(&f, "org/m");
        assert!(!f.repo_dir("org/m").exists(), "staging tree must be removed");
        // Exactly one copy survives.
        assert_eq!(std::fs::read(&dst).unwrap(), b"payload");
    }

    #[test]
    fn install_overwrites_existing_artifact() {
        // `--force` re-pulls install over an existing model.base.
        let tmp = tempfile::tempdir().unwrap();
        let f = StagedFetcher { staging: tmp.path().join("staging") };
        let src = f.stage("org/m", "main", "m-Q4.base", b"new-bytes");
        let dst = tmp.path().join("model.base");
        std::fs::write(&dst, b"old-bytes").unwrap();

        install_file(&f, "org/m", &src, &dst).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"new-bytes");
    }

    #[test]
    fn install_copies_unowned_sources_and_preserves_them() {
        // MockFetcher owns no staging: fixtures must survive installation and
        // cleanup must be a no-op.
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("org").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let fixture = repo_dir.join("m.base");
        std::fs::write(&fixture, b"fixture-bytes").unwrap();
        let f = MockFetcher::new(tmp.path());

        let src = f.get_file("org/m", "main", "m.base").unwrap();
        let dst = tmp.path().join("model.base");
        install_file(&f, "org/m", &src, &dst).unwrap();
        cleanup_staging(&f, "org/m");

        assert_eq!(std::fs::read(&dst).unwrap(), b"fixture-bytes");
        assert!(fixture.exists(), "unowned source must not be deleted");
    }

    // All assertions live in one test: they mutate the shared process env, so
    // splitting them into separate `#[test]` fns would race under Rust's
    // parallel test runner. Sequential mutation within a single fn is safe.
    #[test]
    fn resolve_max_retries_reads_env_with_default_fallback() {
        let prev = std::env::var("BASERT_HF_MAX_RETRIES").ok();

        // Unset -> the opted-in default (must be > 0, else the retry loop that
        // makes multi-GB pulls resilient stays disabled — the bug this fixes).
        std::env::remove_var("BASERT_HF_MAX_RETRIES");
        const { assert!(DEFAULT_HF_MAX_RETRIES > 0, "retries must be opted in by default") };
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
