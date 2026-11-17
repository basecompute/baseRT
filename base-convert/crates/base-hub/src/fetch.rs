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

/// Send large files over Xet's CAS path instead of ranged HTTPS
/// (`$BASERT_HF_XET`).
///
/// Off by default, which is the whole point of this knob's existence rather
/// than its opposite. Xet cannot resume: hf-hub 1.0 builds its `XetSession`
/// from a default `XetConfig` and exposes no way to pass one, so there is no
/// chunk cache to pick up from, and an interrupted 20GB pull starts over.
/// Measured on the 19.8GB Qwen3.5-35B-A3B bundle: killed at 2246MiB, the next
/// attempt was 12s in with 328MiB.
///
/// It is not a speed trade either, which is what made this an easy call. Over
/// three rounds on the same link the ranged path at its defaults matched or
/// beat Xet every time (Xet 47.3 / 28.8 MiB/s; ranged 24x16MB 54.4 / 81.7).
///
/// What Xet still has is cross-model deduplication: chunks shared with
/// something already fetched do not cross the wire at all. That does not show
/// up in a cold single-model pull like the one measured, so this stays
/// available for anyone pulling a family of related bundles.
fn prefer_xet() -> bool {
    std::env::var("BASERT_HF_XET")
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

    /// Pin `revision` to something immutable — the commit it names on the
    /// Hub — so that a sequence of requests against a moving branch all
    /// observe one publication. Sources with no such notion return the
    /// revision unchanged.
    fn resolve_revision(&self, repo: &str, revision: &str) -> Result<String> {
        let _ = repo;
        Ok(revision.to_string())
    }

    /// A stable identifier for the *content* of `filename` at `revision` —
    /// the LFS sha256 on the Hub — or `None` when the source has no such
    /// notion (fixtures). Lets a multi-file install notice that a file it
    /// already consumed has since been replaced under the same name.
    fn content_id(&self, repo: &str, revision: &str, filename: &str) -> Result<Option<String>> {
        let _ = (repo, revision, filename);
        Ok(None)
    }

    /// Read `range` of `filename` without downloading the rest.
    ///
    /// A `.base` header is a 16-byte prefix plus a JSON blob, both at the front
    /// of a file that may be tens of gigabytes. Two ranged reads are what let
    /// [`crate::scan`] learn a published bundle's arch and backend without
    /// fetching the bundle.
    fn read_range(
        &self,
        repo: &str,
        revision: &str,
        filename: &str,
        range: std::ops::Range<u64>,
    ) -> Result<Vec<u8>> {
        let _ = (repo, revision, filename, range);
        anyhow::bail!("this fetcher cannot read byte ranges")
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

/// The Hub token, if any: `$HF_TOKEN`, the legacy `$HUGGING_FACE_HUB_TOKEN`,
/// the file `$HF_TOKEN_PATH` names, then the cached login under `$HF_HOME`
/// (default `~/.cache/huggingface`). The same order hf-hub uses, spelled
/// out here because the raw tree listing below is not an hf-hub call.
pub(crate) fn resolve_token() -> Option<String> {
    let from_env = |k: &str| std::env::var(k).ok().filter(|s| !s.trim().is_empty());
    if let Some(t) = from_env("HF_TOKEN").or_else(|| from_env("HUGGING_FACE_HUB_TOKEN")) {
        return Some(t.trim().to_string());
    }
    let path = std::env::var_os("HF_TOKEN_PATH")
        .map(PathBuf::from)
        .or_else(|| hf_home().map(|h| h.join("token")))?;
    std::fs::read_to_string(path)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Where the cached login lives, in hf-hub's own order: `$HF_HOME`, then
/// `$XDG_CACHE_HOME/huggingface`, then `~/.cache/huggingface`.
fn hf_home() -> Option<PathBuf> {
    hf_home_from(
        std::env::var_os("HF_HOME").map(PathBuf::from),
        std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from),
        dirs::home_dir(),
    )
}

fn hf_home_from(
    hf_home: Option<PathBuf>,
    xdg_cache: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    let nonempty = |p: PathBuf| (!p.as_os_str().is_empty()).then_some(p);
    hf_home
        .and_then(nonempty)
        .or_else(|| xdg_cache.and_then(nonempty).map(|x| x.join("huggingface")))
        .or_else(|| home.map(|h| h.join(".cache").join("huggingface")))
}

/// Hub API base. `$HF_ENDPOINT` redirects everything at a mirror, the same
/// variable hf-hub honors for downloads.
pub(crate) fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// One file in a repo tree, as the Hub's own JSON describes it.
///
/// Read raw rather than through hf-hub's typed listing: that type expects
/// `sha256`/`pointer_size` under `lfs` while the Hub sends `oid`/`pointerSize`,
/// so the LFS hash — the one number that identifies a part's bytes — always
/// came back `None` and routing fell back to the git object id, which is
/// not a hash anything can check a download against.
#[derive(Debug, Clone)]
pub(crate) struct TreeEntry {
    pub path: String,
    /// Size of the content (the LFS payload where there is one).
    pub size: u64,
    /// The git object id — a sha1 over the pointer, not the content.
    pub git_oid: String,
    /// sha256 of the content, for LFS-backed files.
    pub lfs_sha256: Option<String>,
    /// Served through Xet.
    pub xet: bool,
}

/// Parse one page of `/api/models/<repo>/tree/<rev>` into file entries.
pub(crate) fn parse_tree(body: &str) -> Result<Vec<TreeEntry>> {
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(body).context("parsing the tree listing")?;
    Ok(entries
        .iter()
        .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("file"))
        .filter_map(|e| {
            let path = e.get("path")?.as_str()?.to_string();
            let git_oid = e.get("oid")?.as_str()?.to_string();
            let lfs = e.get("lfs").filter(|l| !l.is_null());
            Some(TreeEntry {
                size: lfs
                    .and_then(|l| l.get("size"))
                    .or_else(|| e.get("size"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                lfs_sha256: lfs
                    .and_then(|l| l.get("oid"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                xet: e.get("xetHash").is_some_and(|x| !x.is_null()),
                git_oid,
                path,
            })
        })
        .collect())
}

/// What a pinned revision may be: the commit the Hub reports for the
/// requested one, or the requested one itself when it is already a commit
/// id. A branch that resolves to nothing usable is refused rather than
/// handed on as if it were immutable — everything after the pin assumes it
/// cannot move.
pub(crate) fn pinned_revision(requested: &str, sha: Option<String>) -> Result<String> {
    let is_commit = |s: &str| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit());
    match sha {
        Some(sha) if is_commit(&sha) => Ok(sha),
        _ if is_commit(requested) => Ok(requested.to_string()),
        other => anyhow::bail!(
            "the Hub did not resolve {requested:?} to a commit (got {other:?}); pass a commit id as the revision"
        ),
    }
}

/// `revision` as one URL path segment: a ref like `feature/foo` or
/// `release#1` must not become a sub-path or a fragment.
pub(crate) fn encode_segment(s: &str) -> String {
    use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
    const KEEP: &AsciiSet = &NON_ALPHANUMERIC
        .remove(b'-')
        .remove(b'_')
        .remove(b'.')
        .remove(b'~');
    utf8_percent_encode(s, KEEP).to_string()
}

/// The whole tree of `repo` at `revision`, following the Hub's `Link`
/// pagination.
pub(crate) fn fetch_tree(repo: &str, revision: &str) -> Result<Vec<TreeEntry>> {
    let url = Some(format!(
        "{}/api/models/{repo}/tree/{}?recursive=true",
        endpoint(),
        encode_segment(revision)
    ));
    let token = resolve_token();
    // The same bounded reads the download client gets: a mirror that sends
    // headers and then goes quiet must fail, not hang the pull before it
    // starts.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(crate::download::read_timeout()))
        .timeout_recv_body(Some(crate::download::read_timeout()))
        .build()
        .into();
    collect_pages(url, |u| {
        let mut req = agent.get(u);
        if let Some(t) = &token {
            req = req.header("Authorization", &format!("Bearer {t}"));
        }
        let resp = req
            .call()
            .with_context(|| format!("listing files in {repo}@{revision}"))?;
        let next = resp
            .headers()
            .get("link")
            .and_then(|v| v.to_str().ok())
            .and_then(crate::scan::parse_next_link);
        let body = resp
            .into_body()
            .read_to_string()
            .with_context(|| format!("reading the file listing for {repo}@{revision}"))?;
        Ok((parse_tree(&body)?, next))
    })
}

/// Pages a listing may run to before it is treated as a misbehaving
/// mirror rather than a big repo (a page is up to 1000 entries).
const MAX_TREE_PAGES: usize = 100;

/// Follow `next` links from `first`, `fetch` returning one page's entries
/// and the link after it. A listing still pointing onward at the cap is an
/// error: a truncated tree would be memoized and make files vanish.
fn collect_pages<T>(
    first: Option<String>,
    mut fetch: impl FnMut(&str) -> Result<(Vec<T>, Option<String>)>,
) -> Result<Vec<T>> {
    let mut url = first;
    let mut out = Vec::new();
    for _ in 0..MAX_TREE_PAGES {
        let Some(u) = url.take() else {
            return Ok(out);
        };
        let (page, next) = fetch(&u)?;
        out.extend(page);
        url = next;
    }
    match url {
        None => Ok(out),
        Some(more) => anyhow::bail!(
            "the file listing did not end after {MAX_TREE_PAGES} pages (next: {more}); refusing a truncated tree"
        ),
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
    // `$HUGGING_FACE_HUB_TOKEN`; resolve once here (see `resolve_token`) and
    // pass it explicitly so every request agrees on who is asking.
    if let Some(tok) = resolve_token() {
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
    trees: Mutex<HashMap<(String, String), Vec<TreeEntry>>>,
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
    fn tree(&self, repo: &str, revision: &str) -> Result<Vec<TreeEntry>> {
        let key = (repo.to_string(), revision.to_string());
        if let Some(hit) = self.trees.lock().unwrap().get(&key) {
            return Ok(hit.clone());
        }
        let entries = fetch_tree(repo, revision)?;
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
            .find(|e| e.path == filename)
            .map(|e| FileFacts {
                size: e.size,
                xet: e.xet,
                key: e.lfs_sha256.unwrap_or(e.git_oid),
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
        // Large files take the resumable path whatever transport the Hub
        // offers. Xet is opt-in (see `prefer_xet`): it cannot resume, and a
        // 20GB pull that has to restart is the failure this release exists to
        // fix.
        let facts = facts.filter(|f| !(f.xet && prefer_xet()) && f.size >= RANGED_MIN_BYTES);
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
            .map(|e| e.path)
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

    fn resolve_revision(&self, repo: &str, revision: &str) -> Result<String> {
        let info = self
            .repo(repo)
            .info()
            .revision(revision.to_string())
            .send()
            .with_context(|| format!("looking up {repo}@{revision}"))?;
        pinned_revision(revision, info.sha)
    }

    fn content_id(&self, repo: &str, revision: &str, filename: &str) -> Result<Option<String>> {
        // Only the LFS hash is a hash of the bytes. A file git stores inline
        // has a sha1 over its header and content, which nothing downstream
        // can compare a download against, so it reports no id at all.
        Ok(self
            .tree(repo, revision)?
            .into_iter()
            .find(|e| e.path == filename)
            .and_then(|e| e.lfs_sha256))
    }

    fn read_range(
        &self,
        repo: &str,
        revision: &str,
        filename: &str,
        range: std::ops::Range<u64>,
    ) -> Result<Vec<u8>> {
        let bytes = self
            .repo(repo)
            .download_file_to_bytes()
            .filename(filename)
            .revision(revision)
            .range(range.clone())
            .send()
            .with_context(|| {
                format!(
                    "reading bytes {}..{} of {filename} from {repo}@{revision}",
                    range.start, range.end
                )
            })?;
        Ok(bytes.to_vec())
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

    #[test]
    fn the_cached_login_is_looked_for_where_hf_hub_puts_it() {
        let p = |s: &str| Some(PathBuf::from(s));
        assert_eq!(hf_home_from(p("/hf"), p("/xdg"), p("/home/u")), p("/hf"));
        assert_eq!(
            hf_home_from(None, p("/xdg"), p("/home/u")),
            p("/xdg/huggingface")
        );
        assert_eq!(
            hf_home_from(p(""), p("/xdg"), p("/home/u")),
            p("/xdg/huggingface")
        );
        assert_eq!(
            hf_home_from(None, None, p("/home/u")),
            p("/home/u/.cache/huggingface")
        );
        assert_eq!(hf_home_from(None, None, None), None);
    }

    #[test]
    fn a_listing_that_never_ends_is_refused_not_truncated() {
        // Every page points onward: exhausting the cap is an error, and
        // nothing partial is returned.
        let mut pages = 0;
        let err = collect_pages(Some("p0".to_string()), |_| {
            pages += 1;
            Ok((vec![pages], Some(format!("p{pages}"))))
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("truncated tree"), "{err}");
        assert_eq!(pages, MAX_TREE_PAGES);

        // A listing that ends is returned whole, however many pages.
        let got = collect_pages(Some("p0".to_string()), |u| {
            let n: usize = u[1..].parse().unwrap();
            Ok((vec![n], (n < 3).then(|| format!("p{}", n + 1))))
        })
        .unwrap();
        assert_eq!(got, vec![0, 1, 2, 3]);
        let none: Vec<u8> = collect_pages(None, |_| -> Result<(Vec<u8>, Option<String>)> {
            unreachable!()
        })
        .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn a_pin_is_a_commit_or_nothing() {
        let sha = "7c73ace2115ad5a838277152d555ec1229281c46".to_string();
        assert_eq!(pinned_revision("main", Some(sha.clone())).unwrap(), sha);
        // Asked for a commit already: it stands on its own.
        assert_eq!(pinned_revision(&sha, None).unwrap(), sha);
        // A branch the Hub cannot resolve is not quietly kept mutable.
        let err = pinned_revision("main", None).unwrap_err().to_string();
        assert!(err.contains("did not resolve"), "{err}");
        let err = pinned_revision("main", Some("not-a-sha".into()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("did not resolve"), "{err}");
    }

    #[test]
    fn a_revision_is_one_path_segment() {
        assert_eq!(encode_segment("main"), "main");
        assert_eq!(encode_segment("feature/foo"), "feature%2Ffoo");
        assert_eq!(encode_segment("release#1"), "release%231");
        assert_eq!(encode_segment("v1.2-rc_3~x"), "v1.2-rc_3~x");
    }

    #[test]
    fn tree_listing_keeps_the_lfs_hash_the_typed_client_drops() {
        // Verbatim shape of the Hub's answer: `lfs.oid` is the sha256 of the
        // content, top-level `oid` the git object id, and small files have
        // no `lfs` block at all.
        let body = r#"[
          {"type":"file","oid":"7bc52451a2e2576b266715c7898b9d79dbe0bb25","size":134,
           "lfs":{"oid":"cc36eece6e94329331b5c4abe4f0d31c06d56658c4360ebb1c4b8a974ed20bfe","size":48318382080,"pointerSize":134},
           "path":"parts/GLM-5.2-Q4.base.part-000"},
          {"type":"file","oid":"abc","size":12,"path":"README.md"},
          {"type":"directory","oid":"def","path":"parts"}
        ]"#;
        let got = parse_tree(body).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].path, "parts/GLM-5.2-Q4.base.part-000");
        assert_eq!(got[0].size, 48318382080);
        assert_eq!(
            got[0].lfs_sha256.as_deref(),
            Some("cc36eece6e94329331b5c4abe4f0d31c06d56658c4360ebb1c4b8a974ed20bfe")
        );
        assert_eq!(got[0].git_oid, "7bc52451a2e2576b266715c7898b9d79dbe0bb25");
        assert!(!got[0].xet);
        assert_eq!(got[1].size, 12);
        assert_eq!(got[1].lfs_sha256, None);
    }

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
