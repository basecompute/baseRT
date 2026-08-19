//! File fetching, abstracted so tests can run without network.
//!
//! [`HfFetcher`] is the real implementation over hf-hub's blocking API. Which
//! transport a file takes is the Hub's choice, not ours: Xet-backed files go
//! through hf-xet's chunk-deduplicated CAS path, and everything else through
//! the parallel, resumable range downloader in [`crate::download`]. Neither
//! the trait nor its callers see the difference. [`MockFetcher`] copies from a
//! local fixture directory so the pull/convert pipeline can be exercised in
//! CI with no HuggingFace access.

use anyhow::{Context, Result};
use hf_hub::progress::{DownloadEvent, FileStatus, Progress, ProgressEvent, ProgressHandler};
use hf_hub::repository::RepoTreeEntry;
use hf_hub::{HFClient, HFClientSync, HFRepositorySync, RepoTypeModel};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// Download retries hf-hub performs on transient failures (peer disconnects,
/// truncated chunks — routine on multi-GB model pulls). hf-hub's own default
/// is `3`; ours is higher because a pull that dies at 90% of a 30GB artifact
/// costs far more than a few extra backoffs. The same count bounds per-chunk
/// retries in [`crate::download`]. Override with `$BASERT_HF_MAX_RETRIES`
/// (`0` disables retries).
const DEFAULT_HF_MAX_RETRIES: usize = 5;

/// Files at or above this size go through the parallel, resumable path. Below
/// it the whole transfer is shorter than one retry's backoff, so hf-hub's own
/// single-stream download (and its snapshot/symlink cache bookkeeping) is the
/// better trade — this threshold separates `config.json` from a weight shard.
const RANGED_MIN_BYTES: u64 = 32 * 1024 * 1024;

/// Force every large file through the range downloader, even one the Hub
/// serves over Xet (`$BASERT_HF_FORCE_RANGED`).
///
/// Two uses. It is how the range path gets exercised against huggingface.co
/// at all — the Hub has migrated essentially every repository to Xet, so the
/// fallback would otherwise only ever run against a non-migrated endpoint or
/// an `$HF_ENDPOINT` mirror. And it is the escape hatch when hf-xet is the
/// thing going wrong on a given link: the same bytes are still reachable over
/// plain ranged HTTPS from the CDN.
fn force_ranged() -> bool {
    std::env::var("BASERT_HF_FORCE_RANGED")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

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

/// Terminal progress for one hf-hub-driven download.
///
/// Two event families have to land on the same bar. Plain HTTPS transfers
/// report per-file deltas ([`DownloadEvent::Progress`], only the files whose
/// counters moved), so the running total is a sum over remembered per-file
/// positions. Xet transfers report one aggregate for the in-flight batch with
/// no per-file breakdown ([`DownloadEvent::AggregateProgress`]), which sets
/// the position directly. A download uses one family or the other, never both.
///
/// hf-hub calls handlers from its transfer threads and forbids blocking, so
/// the state is a plain mutex held only long enough to update counters.
struct BarProgress {
    label: String,
    state: Mutex<BarState>,
}

#[derive(Default)]
struct BarState {
    bar: Option<ProgressBar>,
    /// Latest `bytes_completed` per file, for summing HTTPS deltas.
    per_file: HashMap<String, u64>,
}

impl BarProgress {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            state: Mutex::new(BarState::default()),
        }
    }
}

/// Shared bar rendering, so the hf-hub-driven and range-driven paths look
/// identical to whoever is watching the pull.
pub(crate) fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:.bold} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=> ")
}

impl ProgressHandler for BarProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        // A poisoned mutex here means a previous callback panicked; progress
        // display is not worth propagating that into the transfer.
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                let bar = ProgressBar::new(*total_bytes);
                bar.set_style(bar_style());
                bar.set_message(self.label.clone());
                state.bar = Some(bar);
            }
            DownloadEvent::Progress { files } => {
                for f in files {
                    // A `Complete` file with an unknown size (a cache hit) must
                    // not rewind the sum to zero.
                    let done = match f.status {
                        FileStatus::Complete if f.total_bytes == 0 => {
                            *state.per_file.get(&f.filename).unwrap_or(&0)
                        }
                        FileStatus::Complete => f.total_bytes,
                        _ => f.bytes_completed,
                    };
                    state.per_file.insert(f.filename.clone(), done);
                }
                let total: u64 = state.per_file.values().sum();
                if let Some(bar) = &state.bar {
                    bar.set_position(total);
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                if let Some(bar) = &state.bar {
                    // The xet batch total is authoritative once it is known:
                    // dedup means fewer bytes cross the wire than HEAD implied.
                    if *total_bytes > 0 {
                        bar.set_length(*total_bytes);
                    }
                    bar.set_position(*bytes_completed);
                }
            }
            DownloadEvent::Complete => {
                if let Some(bar) = state.bar.take() {
                    bar.finish_and_clear();
                }
            }
        }
    }
}

/// Build one hf-hub client pointed at `staging_root`.
///
/// The reqwest client is ours rather than hf-hub's default so it can carry
/// timeouts: `read_timeout` is what turns a connection that stops delivering
/// bytes — without ever closing — into an error the retry loop can act on,
/// instead of a pull that hangs until someone notices.
fn build_client(staging_root: &Path) -> Result<HFClientSync> {
    let http = reqwest::Client::builder()
        .read_timeout(crate::download::read_timeout())
        .connect_timeout(Duration::from_secs(30))
        .user_agent(concat!("baseRT/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building the HTTP client")?;
    let mut builder = HFClient::builder()
        // Points the whole cache — blobs, snapshots, refs — at our staging
        // tree instead of `~/.cache/huggingface/hub`.
        .cache_dir(staging_root.to_path_buf())
        .retry_max_attempts(resolve_max_retries())
        .client(http);
    // hf-hub resolves `$HF_TOKEN` itself but not the legacy
    // `$HUGGING_FACE_HUB_TOKEN`, so both are read here and passed explicitly;
    // an unset pair leaves hf-hub's own resolution in place.
    if let Some(tok) = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|s| !s.is_empty())
    {
        builder = builder.token(tok);
    }
    builder
        .build_sync()
        .context("initializing HuggingFace API client")
}

/// Real fetcher backed by hf-hub's blocking API. Reads the HF token from
/// `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`, falling back to hf-hub's own
/// resolution (`$HF_TOKEN_PATH`, then the cached login token under
/// `$HF_HOME`).
///
/// Downloads land in a private staging directory (normally
/// `<models root>/.src/hf` — see [`crate::cache::hf_staging_dir`]), NOT the
/// user's global HuggingFace cache: multi-GB `.base` artifacts would otherwise
/// persist there as a second copy after installation. Keeping staging on the
/// same filesystem as the models root also lets installs move (rename) the
/// downloaded bytes instead of copying them.
pub struct HfFetcher {
    client: HFClientSync,
    staging_root: PathBuf,
    /// Memoized tree listings, keyed by `(repo, revision)`. A pull asks about
    /// several files in the same repo — the artifact, then its sidecars — and
    /// the listing that answers "how big, and is it Xet?" is the same one
    /// `list_files` needs.
    trees: Mutex<HashMap<(String, String), Vec<RepoTreeEntry>>>,
}

/// What routing needs to know about one remote file.
struct FileFacts {
    size: u64,
    xet: bool,
    /// Content id to key the staged blob on — the LFS oid where there is one
    /// (a sha256 of the content), else the git object id.
    key: String,
}

impl HfFetcher {
    pub fn new(staging_root: impl Into<PathBuf>) -> Result<Self> {
        let staging_root = staging_root.into();
        let client = build_client(&staging_root)?;
        Ok(Self {
            client,
            staging_root,
            trees: Mutex::new(HashMap::new()),
        })
    }

    /// The repo's file tree at `revision`, fetched once and remembered.
    fn tree(&self, repo: &str, revision: &str) -> Result<Vec<RepoTreeEntry>> {
        let key = (repo.to_string(), revision.to_string());
        if let Some(hit) = self.trees.lock().unwrap().get(&key) {
            return Ok(hit.clone());
        }
        let entries = self
            .repo(repo)
            .list_tree()
            .revision(revision)
            .recursive(true)
            .send()
            .with_context(|| format!("listing files in {repo}@{revision}"))?;
        self.trees.lock().unwrap().insert(key, entries.clone());
        Ok(entries)
    }

    /// Size, transport and content id for one file, from the tree listing.
    ///
    /// The listing is the right source for this, not a HEAD on the resolve
    /// URL: `resolve/...` 302s to a CDN, and hf-hub's public
    /// `get_file_metadata` follows that redirect with its normal client, so
    /// the headers routing depends on — `X-Repo-Commit`, `X-Xet-Hash` — are
    /// gone from the response it reads. (It fails outright on Xet-backed
    /// files for exactly that reason.) The tree endpoint answers the same
    /// questions with no redirect in the way.
    fn facts(&self, repo: &str, revision: &str, filename: &str) -> Result<Option<FileFacts>> {
        Ok(self
            .tree(repo, revision)?
            .into_iter()
            .find_map(|e| match e {
                RepoTreeEntry::File {
                    path,
                    size,
                    oid,
                    lfs,
                    xet_hash,
                    ..
                } if path == filename => Some(FileFacts {
                    size: lfs.as_ref().and_then(|l| l.size).unwrap_or(size),
                    xet: xet_hash.is_some(),
                    key: lfs.and_then(|l| l.sha256).unwrap_or(oid),
                }),
                _ => None,
            }))
    }

    fn repo(&self, repo: &str) -> HFRepositorySync<RepoTypeModel> {
        let (owner, name) = hf_hub::split_id(repo);
        self.client.model(owner, name)
    }

    /// Where the parallel path parks a finished blob: the same `blobs/`
    /// directory hf-hub uses, so one `staging_dir` still covers every byte we
    /// fetched however it arrived.
    fn blob_path(&self, repo: &str, etag: &str) -> PathBuf {
        let safe: String = etag
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.staging_dir(repo)
            .unwrap_or_else(|| self.staging_root.clone())
            .join("blobs")
            .join(safe)
    }
}

impl Fetcher for HfFetcher {
    fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
        let handle = self.repo(repo);
        let facts = self.facts(repo, revision, filename)?;

        // Xet-backed: hand it to hf-hub, which routes to hf-xet — already
        // chunk-parallel, deduplicated against the local chunk cache, and
        // resumable. Re-implementing that on top of range requests would be
        // strictly worse. Small files take the same path: not worth a chunk
        // plan, and a retry costs less there than planning one. A file the
        // listing does not describe goes there too, rather than guessing.
        let facts = facts.filter(|f| (!f.xet || force_ranged()) && f.size >= RANGED_MIN_BYTES);
        let Some(facts) = facts else {
            return handle
                .download_file()
                .filename(filename)
                .revision(revision)
                .progress(Progress::new(BarProgress::new(filename.to_string())))
                .send()
                .with_context(|| format!("downloading {filename} from {repo}@{revision}"));
        };

        // Plain HTTPS blob, large: parallel and resumable (see `download`).
        let dst = self.blob_path(repo, &facts.key);
        if std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0) == facts.size {
            return Ok(dst);
        }
        let staging_root = self.staging_root.clone();
        let mint = move || build_client(&staging_root);
        crate::download::download_ranged(
            &mint,
            repo,
            revision,
            filename,
            facts.size,
            &dst,
            resolve_max_retries(),
        )
        .with_context(|| format!("downloading {filename} from {repo}@{revision}"))?;
        Ok(dst)
    }

    fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
        Ok(self
            .tree(repo, revision)?
            .into_iter()
            .filter_map(|e| match e {
                RepoTreeEntry::File { path, .. } => Some(path),
                _ => None,
            })
            .collect())
    }

    fn staging_dir(&self, repo: &str) -> Option<PathBuf> {
        // hf-hub keeps everything for a repo under `models--<org>--<repo>`.
        // The helper that spells this is crate-private, so the one-line
        // mapping is mirrored here.
        Some(
            self.staging_root
                .join(format!("models--{}", repo.replace('/', "--"))),
        )
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
        let real =
            std::fs::canonicalize(src).with_context(|| format!("resolving {}", src.display()))?;
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
            self.staging
                .join(format!("models--{}", repo.replace('/', "--")))
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
        let f = StagedFetcher {
            staging: tmp.path().join("staging"),
        };
        let src = f.stage("org/m", "main", "m-Q4.base", b"payload");
        let dst = tmp.path().join("model.base");

        install_file(&f, "org/m", &src, &dst).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"payload");
        // The blob was moved, not copied: the staged payload is gone (only a
        // dangling pointer symlink may remain until cleanup).
        let blob = f.repo_dir("org/m").join("blobs").join("etag-m-Q4.base");
        assert!(!blob.exists(), "blob must be moved out of staging");

        cleanup_staging(&f, "org/m");
        assert!(
            !f.repo_dir("org/m").exists(),
            "staging tree must be removed"
        );
        // Exactly one copy survives.
        assert_eq!(std::fs::read(&dst).unwrap(), b"payload");
    }

    #[test]
    fn install_moves_a_plain_staged_blob() {
        // The range downloader parks a plain file in `blobs/`, not a
        // snapshot symlink into it — the shape hf-hub's own path produces and
        // the one the other install tests cover. Install must move it just
        // the same, leaving nothing behind in staging.
        let tmp = tempfile::tempdir().unwrap();
        let f = StagedFetcher {
            staging: tmp.path().join("staging"),
        };
        let blobs = f.repo_dir("org/m").join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let src = blobs.join("sha256-of-the-content");
        std::fs::write(&src, b"ranged-payload").unwrap();
        let dst = tmp.path().join("model.base");

        install_file(&f, "org/m", &src, &dst).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"ranged-payload");
        assert!(!src.exists(), "the staged blob must be moved, not copied");

        cleanup_staging(&f, "org/m");
        assert!(!f.repo_dir("org/m").exists());
    }

    #[test]
    fn install_overwrites_existing_artifact() {
        // `--force` re-pulls install over an existing model.base.
        let tmp = tempfile::tempdir().unwrap();
        let f = StagedFetcher {
            staging: tmp.path().join("staging"),
        };
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
        const {
            assert!(
                DEFAULT_HF_MAX_RETRIES > 0,
                "retries must be opted in by default"
            )
        };
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
