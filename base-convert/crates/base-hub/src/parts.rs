//! Split `.base` bundles: recognizing a part set in a repo listing and
//! reassembling one into a single artifact on install.
//!
//! The Hub caps a single file at 50 GB. A bundle past that ships as
//! `<name>.base.part-000`, `<name>.base.part-001`, … (a plain byte split,
//! the same convention as multi-part GGUFs), and the logical artifact is
//! `<name>.base`. Nothing else about the bundle changes: part 000 opens with
//! the ordinary header, so a ranged read of it still answers "what is this".
//!
//! Reassembly is streamed into `<dst>.partial` and renamed into place only
//! when every part has landed, so a half-built bundle never looks installed.
//! Each staged part is deleted as soon as it has been appended, which keeps
//! peak disk at one bundle plus one part rather than two bundles. A record
//! beside the partial says how many parts it holds, so a pull killed between
//! parts resumes at the next part instead of at byte zero.
//!
//! What makes the stitched result trustworthy, given that nothing on the Hub
//! carries a whole-file checksum for it:
//!
//! * a manifest beside the parts (`<name>.base.manifest.json`, see
//!   [`Manifest`]) says how many parts there are, how long each is, the
//!   sha256 of each, and the sha256 and length of the whole. Nothing in the
//!   part filenames says how many there should be, and a bundle may end in
//!   extension slots the header does not flag, so the manifest is the one
//!   thing that can prove the tail was not lost. It is required;
//! * the revision is pinned to a commit before the first byte moves, so a
//!   mutable branch advancing mid-pull cannot mix two publications;
//! * every part is hashed as it is appended and checked against the
//!   manifest, and the Hub's own per-part hashes must agree with it;
//! * the finished file is parsed as a `.base` as a second line of defence:
//!   its weights blob and any slot section have to fit;
//! * one process at a time works on a destination, held by a lock file.

use crate::fetch::{install_file, Fetcher};
use anyhow::{bail, Context, Result};
use base_format::{BLOB_ALIGNMENT, FORMAT_VERSION, MAGIC, PREFIX_LEN};
use indicatif::ProgressBar;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

/// A `.base` bundle as a repo hosts it: one file, or an ordered part set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    /// The logical `.base` path (`parts/GLM-5.2-Q4.base`), which is also the
    /// real path when `parts` is empty.
    pub name: String,
    /// Part paths in byte order; empty for a whole file.
    pub parts: Vec<String>,
}

impl Artifact {
    pub fn whole(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parts: Vec::new(),
        }
    }

    pub fn is_split(&self) -> bool {
        !self.parts.is_empty()
    }
}

/// `<name>.base.part-<digits>` → (`<name>.base`, index). Anything else is
/// not a part.
pub fn split_part_name(path: &str) -> Option<(String, u32)> {
    let (name, idx) = path.rsplit_once(".part-")?;
    if !name.ends_with(".base") || idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((name.to_string(), idx.parse().ok()?))
}

/// A repo listing sorted into what can be pulled and what cannot.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Grouped {
    pub artifacts: Vec<Artifact>,
    /// Logical names of part sets that are not usable, with the reason —
    /// a gap in the indices, typically a quant still being uploaded. Kept
    /// apart so one bad set does not take a repo's other bundles with it.
    pub malformed: Vec<(String, String)>,
}

/// Group a repo listing into artifacts: every whole `.base` file, plus one
/// artifact per part set. A set with a gap in its indices is reported as
/// malformed rather than accepted as a shorter bundle; a set that simply
/// stops early cannot be told apart here and is caught by the completeness
/// check on install. When a repo ships both a whole file and parts under
/// the same name, the whole file wins.
pub fn group(files: impl IntoIterator<Item = String>) -> Grouped {
    let mut whole = Vec::new();
    let mut sets: BTreeMap<String, BTreeMap<u32, String>> = BTreeMap::new();
    // Two spellings of one index (`part-000` and `part-0000`) are two files
    // claiming the same slot; neither can be picked over the other.
    let mut doubled: BTreeMap<String, String> = BTreeMap::new();
    for f in files {
        if let Some((name, idx)) = split_part_name(&f) {
            if let Some(other) = sets.entry(name.clone()).or_default().insert(idx, f.clone()) {
                doubled
                    .entry(name)
                    .or_insert_with(|| format!("index {idx} is listed twice ({other} and {f})"));
            }
        } else if f.ends_with(".base") {
            whole.push(f);
        }
    }
    let mut out = Grouped {
        artifacts: whole.into_iter().map(Artifact::whole).collect(),
        malformed: Vec::new(),
    };
    for (name, parts) in sets {
        if out.artifacts.iter().any(|a| a.name == name) {
            continue;
        }
        if let Some(why) = doubled.remove(&name) {
            out.malformed.push((name, why));
            continue;
        }
        let gap = (0u32..)
            .zip(parts.keys())
            .find(|(want, have)| *have != want)
            .map(|(want, _)| want);
        if let Some(want) = gap {
            out.malformed.push((
                name,
                format!(
                    "part set is missing part {want:03} (found {} parts)",
                    parts.len()
                ),
            ));
            continue;
        }
        out.artifacts.push(Artifact {
            name,
            parts: parts.into_values().collect(),
        });
    }
    out
}

/// The artifact `name` denotes in `repo`: the part set behind it when the
/// listing shows one, else the whole file. A name the listing does not know
/// is returned as a whole file so the download itself reports the miss; a
/// name that is a malformed part set is an error saying why.
pub fn find(fetcher: &dyn Fetcher, repo: &str, revision: &str, name: &str) -> Result<Artifact> {
    let files = fetcher
        .list_files(repo, revision)
        .with_context(|| format!("listing files in {repo}@{revision}"))?;
    let grouped = group(files);
    if let Some((_, why)) = grouped.malformed.iter().find(|(n, _)| n == name) {
        bail!("{repo}/{name}: {why}");
    }
    Ok(grouped
        .artifacts
        .into_iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| Artifact::whole(name)))
}

/// What a publisher states about a split bundle, in `<name>.manifest.json`
/// beside the parts.
///
/// ```json
/// {"size": 432223, "sha256": "…whole file…",
///  "parts": [{"name": "X.base.part-000", "size": 45, "sha256": "…"}, …]}
/// ```
///
/// `name` is the part's basename; parts are listed in byte order. Written
/// by `basert catalog-manifest` from the local parts, without reassembling
/// them: the whole-file hash is one sha256 run across the parts in order.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub size: u64,
    pub sha256: String,
    pub parts: Vec<ManifestPart>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestPart {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

/// `parts/X.base` → `parts/X.base.manifest.json`.
pub fn manifest_name(name: &str) -> String {
    format!("{name}.manifest.json")
}

impl Manifest {
    /// The per-part hashes, in order.
    pub fn part_ids(&self) -> Vec<String> {
        self.parts.iter().map(|p| p.sha256.clone()).collect()
    }

    /// Does this manifest describe exactly `parts` (paths in byte order)?
    /// Count, names and the size sum all have to line up, and every hash
    /// has to be one — a malformed whole-file hash would otherwise ride
    /// into the catalog and fail every pull only after the last byte.
    pub fn check_against(&self, parts: &[String]) -> Result<()> {
        if !is_sha256(&self.sha256) {
            bail!("the manifest's whole-file sha256 is not a sha256");
        }
        if self.parts.len() != parts.len() {
            bail!(
                "the manifest lists {} parts but the repo has {}",
                self.parts.len(),
                parts.len()
            );
        }
        for (m, p) in self.parts.iter().zip(parts) {
            let base = p.rsplit('/').next().unwrap_or(p);
            if m.name != base {
                bail!("the manifest names {} where the repo has {base}", m.name);
            }
            if !is_sha256(&m.sha256) {
                bail!("the manifest's sha256 for {} is not a sha256", m.name);
            }
        }
        let sum = self
            .parts
            .iter()
            .try_fold(0u64, |a, p| a.checked_add(p.size))
            .context("manifest part sizes overflow")?;
        if sum != self.size {
            bail!(
                "the manifest's parts sum to {sum} bytes but it says the whole is {}",
                self.size
            );
        }
        Ok(())
    }

    /// Do the per-part sizes match what the source reports for each part?
    /// Two wrong sizes can cancel out in the sum; they cannot here.
    pub fn check_sizes(&self, listed: &[u64]) -> Result<()> {
        if listed.len() != self.parts.len() {
            bail!(
                "the manifest lists {} parts but {} sizes were given",
                self.parts.len(),
                listed.len()
            );
        }
        for (m, have) in self.parts.iter().zip(listed) {
            if m.size != *have {
                bail!(
                    "the manifest says {} is {} bytes but the repo holds {have}",
                    m.name,
                    m.size
                );
            }
        }
        Ok(())
    }
}

/// Build a manifest from local part files, in the order given, hashing
/// each part and the whole in one pass.
pub fn build_manifest(parts: &[PathBuf]) -> Result<Manifest> {
    if parts.is_empty() {
        bail!("no parts to describe");
    }
    let mut whole = Sha256::new();
    let mut out = Manifest {
        size: 0,
        sha256: String::new(),
        parts: Vec::with_capacity(parts.len()),
    };
    let mut buf = vec![0u8; 8 << 20];
    for path in parts {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .with_context(|| format!("{}: not a file name", path.display()))?
            .to_string();
        let mut reader = BufReader::with_capacity(8 << 20, File::open(path)?);
        let mut one = Sha256::new();
        let mut size = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            one.update(&buf[..n]);
            whole.update(&buf[..n]);
            size += n as u64;
        }
        out.size += size;
        out.parts.push(ManifestPart {
            name,
            size,
            sha256: hex(one.finalize().as_slice()),
        });
    }
    out.sha256 = hex(whole.finalize().as_slice());
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A manifest is a few KB; anything claiming to be larger is not one, and
/// is refused before it is allocated. Shared with the catalog scan.
pub const MAX_MANIFEST_LEN: u64 = 1024 * 1024;

/// The manifest published beside `name`'s parts. Its absence is an error
/// that says what to publish: without it a lost tail is undetectable.
fn read_manifest(
    fetcher: &dyn Fetcher,
    repo: &str,
    revision: &str,
    name: &str,
) -> Result<Manifest> {
    let mname = manifest_name(name);
    let path = fetcher.get_file(repo, revision, &mname).with_context(|| {
        format!(
            "{repo}/{mname}: a split bundle needs its manifest beside the parts (publish one with `basert catalog-manifest`)"
        )
    })?;
    let len = std::fs::metadata(&path)
        .with_context(|| format!("sizing {}", path.display()))?
        .len();
    if len == 0 || len >= MAX_MANIFEST_LEN {
        bail!("{repo}/{mname}: implausible manifest size {len}");
    }
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {repo}/{mname}"))
}

/// Download `artifact` and install it at `dst`, reassembling a part set.
///
/// `expected_ids` is what a catalog row pinned for the parts (their sha256s,
/// in order). When given, the Hub's listing has to agree with it before any
/// byte is fetched, and each part is verified against it as it lands; a
/// disagreement means the bundle was republished since the row was written.
pub fn install(
    fetcher: &dyn Fetcher,
    repo: &str,
    revision: &str,
    artifact: &Artifact,
    dst: &Path,
    expected_ids: Option<&[String]>,
) -> Result<()> {
    // Pin the revision first: every later request names the commit, not
    // the branch, so a publish landing mid-pull cannot be half-observed.
    let pinned = fetcher
        .resolve_revision(repo, revision)
        .with_context(|| format!("resolving {repo}@{revision}"))?;
    if pinned != revision {
        eprintln!(
            "  pinned:  {revision} → {}",
            &pinned[..pinned.len().min(12)]
        );
    }
    // The caller's artifact came from whatever the branch pointed at when
    // it listed; look again at the pinned commit before deciding anything
    // — whether it is whole or split included — so the parts, the ids, and
    // the downloads all describe one publication.
    let artifact = find(fetcher, repo, &pinned, &artifact.name)?;
    if !artifact.is_split() {
        let src = fetcher.get_file(repo, &pinned, &artifact.name)?;
        return install_file(fetcher, repo, &src, dst);
    }
    let manifest = read_manifest(fetcher, repo, &pinned, &artifact.name)?;
    manifest.check_against(&artifact.parts).with_context(|| {
        format!(
            "{repo}/{}: manifest disagrees with the listing",
            artifact.name
        )
    })?;
    let ids = manifest.part_ids();
    // The Hub's own per-part hashes, where it has them, have to agree with
    // the manifest; and so does what the catalog pinned.
    for (part, id) in artifact.parts.iter().zip(&ids) {
        if let Some(listed) = fetcher.content_id(repo, &pinned, part)? {
            if !listed.eq_ignore_ascii_case(id) {
                bail!(
                    "{repo}/{part} is {listed} on the Hub but the manifest says {id}: the manifest is stale, republish it"
                );
            }
        }
    }
    if let Some(expected) = expected_ids {
        reconcile_ids(&artifact.parts, &ids, expected)?;
    }

    let lock = Lock::acquire(dst)?;
    Reassembly::open(dst, repo, &pinned, &artifact.parts, ids, manifest)?
        .run(fetcher, repo, &pinned)?;
    lock.release();
    Ok(())
}

/// What the catalog row pinned has to be what the manifest says now; a
/// difference means the bundle was republished since the row was written.
fn reconcile_ids(parts: &[String], manifest: &[String], expected: &[String]) -> Result<()> {
    if expected.len() != parts.len() {
        bail!(
            "the catalog pins {} parts for this bundle but the Hub lists {}: the bundle was republished, refresh the catalog",
            expected.len(),
            parts.len()
        );
    }
    for ((part, now), expected) in parts.iter().zip(manifest).zip(expected) {
        if !now.eq_ignore_ascii_case(expected) {
            bail!(
                "{part} is now {now}, not the {expected} the catalog pinned: the bundle was republished, refresh the catalog"
            );
        }
    }
    Ok(())
}

/// Where a part set is stitched together before it becomes `dst`.
pub fn partial_path(dst: &Path) -> PathBuf {
    with_suffix(dst, ".partial")
}

fn record_path(dst: &Path) -> PathBuf {
    with_suffix(dst, ".partial.json")
}

fn lock_path(dst: &Path) -> PathBuf {
    with_suffix(dst, ".partial.lock")
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Exclusive hold on one destination for the length of a reassembly, so two
/// pulls of the same variant cannot both append to one partial. Taken
/// without waiting: the second pull is told, not queued behind a 400 GB
/// download it would only duplicate.
struct Lock {
    file: File,
    path: PathBuf,
}

impl Lock {
    fn acquire(dst: &Path) -> Result<Self> {
        let path = lock_path(dst);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        // Safety: a valid fd for the lifetime of the call; flock has no
        // memory-safety preconditions beyond that.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
                bail!(
                    "another basert pull is already assembling {}; wait for it or stop it first",
                    dst.display()
                );
            }
            return Err(err).with_context(|| format!("locking {}", path.display()));
        }
        Ok(Self { file, path })
    }

    /// Remove the lock file once the install is in place. The lock is still
    /// held while the file is unlinked, so a pull arriving in that window
    /// creates a fresh lock file rather than sharing this one.
    fn release(self) {
        let _ = std::fs::remove_file(&self.path);
        drop(self.file);
    }
}

/// What the partial holds, persisted after every part so a restart can pick
/// up where the last one stopped.
///
/// Resuming is only safe when the remaining parts continue the same bytes
/// the partial already holds, so the record names where they came from:
/// the repo, the pinned commit, the part paths, and each part's content id.
/// A mutable revision republished between two attempts pins to a different
/// commit, and the partial is rebuilt rather than spliced.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct Record {
    repo: String,
    revision: String,
    parts: Vec<String>,
    /// Per-part sha256s, from the manifest.
    ids: Vec<String>,
    /// Parts fully appended, in order.
    appended: usize,
    /// Byte length of the partial once those parts were appended. Anything
    /// past it is a torn write from an interrupted append and is cut off.
    len: u64,
}

struct Reassembly {
    dst: PathBuf,
    partial: PathBuf,
    record_path: PathBuf,
    record: Record,
    manifest: Manifest,
    /// Some of the partial predates this process: its bytes were checked
    /// by an earlier attempt, not this one.
    resumed: bool,
    /// Running sha256 of everything this process appended, so a fresh
    /// install can compare the whole with the manifest without a second
    /// read of the file.
    whole: Sha256,
}

impl Reassembly {
    fn open(
        dst: &Path,
        repo: &str,
        revision: &str,
        parts: &[String],
        ids: Vec<String>,
        manifest: Manifest,
    ) -> Result<Self> {
        let partial = partial_path(dst);
        let record_path = record_path(dst);
        let fresh = Record {
            repo: repo.to_string(),
            revision: revision.to_string(),
            parts: parts.to_vec(),
            ids,
            appended: 0,
            len: 0,
        };
        let same_source = |r: &Record| {
            r.repo == fresh.repo
                && r.revision == fresh.revision
                && r.parts == fresh.parts
                && r.ids == fresh.ids
        };
        let record = match std::fs::read(&record_path) {
            Ok(bytes) => serde_json::from_slice::<Record>(&bytes)
                .ok()
                .filter(same_source)
                .filter(|r| std::fs::metadata(&partial).is_ok_and(|m| m.len() >= r.len))
                .unwrap_or(fresh),
            Err(_) => fresh,
        };
        if record.appended > 0 {
            eprintln!(
                "  resume:  {} of {} parts already assembled",
                record.appended,
                record.parts.len()
            );
            // Cut off a torn tail from an interrupted append.
            OpenOptions::new()
                .write(true)
                .open(&partial)
                .and_then(|f| f.set_len(record.len))
                .with_context(|| format!("truncating {}", partial.display()))?;
        } else {
            let _ = std::fs::remove_file(&partial);
            let _ = std::fs::remove_file(&record_path);
        }
        let resumed = record.appended > 0;
        Ok(Self {
            dst: dst.to_path_buf(),
            partial,
            record_path,
            record,
            manifest,
            resumed,
            whole: Sha256::new(),
        })
    }

    fn run(mut self, fetcher: &dyn Fetcher, repo: &str, revision: &str) -> Result<()> {
        let total = self.record.parts.len();
        for i in self.record.appended..total {
            let part = self.record.parts[i].clone();
            let src = fetcher.get_file(repo, revision, &part)?;
            let (digest, got) = append(
                &src,
                &self.partial,
                &format!("assemble {}/{total}", i + 1),
                &mut self.whole,
            )?;
            let want = &self.manifest.parts[i];
            let problem = if got != want.size {
                Some(format!(
                    "{part} is {got} bytes but the manifest says {}",
                    want.size
                ))
            } else if !digest.eq_ignore_ascii_case(&want.sha256) {
                Some(format!(
                    "{part} hashed to {digest} but the manifest says {}",
                    want.sha256
                ))
            } else {
                None
            };
            if let Some(problem) = problem {
                // Neither the staged part nor the partial can be trusted
                // past this point; the next attempt rebuilds. The staged
                // bytes go only if they are ours to drop.
                self.discard();
                discard_staged(fetcher, repo, &src);
                bail!("{problem}: the download was corrupted or the part was republished; run the pull again");
            }
            discard_staged(fetcher, repo, &src);
            self.record.appended = i + 1;
            self.record.len = std::fs::metadata(&self.partial)
                .with_context(|| format!("sizing {}", self.partial.display()))?
                .len();
            self.save()?;
        }

        // Every part matched the manifest, so the whole is the manifest's
        // size by construction; say so explicitly rather than trust it.
        if self.record.len != self.manifest.size {
            self.discard();
            bail!(
                "{total} parts reassemble to {} bytes but the manifest says {}",
                self.record.len,
                self.manifest.size
            );
        }
        // The whole has to be what the manifest says it is, not only each
        // part: a manifest with right parts and a wrong whole would
        // otherwise pass here and fail a catalog pull after the last byte.
        // A fresh install has hashed everything it appended; a resumed one
        // inherited bytes an earlier attempt verified, which anything that
        // touched the partial since (a torn write past the record, a disk
        // fault, a stray tool) could have changed, so it reads them again.
        let (kind, got) = if self.resumed {
            ("resumed", sha256_file(&self.partial, "verify")?)
        } else {
            ("reassembled", hex(self.whole.clone().finalize().as_slice()))
        };
        if !got.eq_ignore_ascii_case(&self.manifest.sha256) {
            self.discard();
            bail!(
                "the {kind} file hashed to {got} but the manifest says {}: the manifest is wrong or the partial was damaged; run the pull again",
                self.manifest.sha256
            );
        }
        // Second line of defence, against a manifest written from a broken
        // set: the file is parsed as a `.base` and its weights blob and any
        // slot section have to fit.
        if let Err(e) = check_complete(&self.partial, self.record.len) {
            self.discard();
            return Err(e.context(format!(
                "{total} parts reassemble to {} bytes but the bundle is not complete: the part set on the Hub is missing its tail",
                self.record.len
            )));
        }

        std::fs::rename(&self.partial, &self.dst).with_context(|| {
            format!(
                "installing {} as {}",
                self.partial.display(),
                self.dst.display()
            )
        })?;
        let _ = std::fs::remove_file(&self.record_path);
        Ok(())
    }

    fn save(&self) -> Result<()> {
        let bytes = serde_json::to_vec(&self.record)?;
        let tmp = with_suffix(&self.record_path, ".tmp");
        let mut f = File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, &self.record_path)
            .with_context(|| format!("writing {}", self.record_path.display()))?;
        Ok(())
    }

    /// Throw the partial away: what it holds has been shown to be wrong, so
    /// there is no resume value in it.
    fn discard(&self) {
        let _ = std::fs::remove_file(&self.partial);
        let _ = std::fs::remove_file(&self.record_path);
    }
}

fn is_sha256(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Append `src` to `dst` in bounded memory, hashing the bytes on the way
/// through — into `whole` too, the running hash of the file being built;
/// returns the hex sha256 of `src` and how many bytes it was. Shows a bar
/// in the same style as the download that preceded it.
fn append(src: &Path, dst: &Path, label: &str, whole: &mut Sha256) -> Result<(String, u64)> {
    let len = std::fs::metadata(src)?.len();
    let bar = ProgressBar::new(len);
    bar.set_style(crate::fetch::bar_style());
    bar.set_message(label.to_string());
    let mut reader = BufReader::with_capacity(8 << 20, File::open(src)?);
    let mut out = OpenOptions::new()
        .append(true)
        .create(true)
        .open(dst)
        .with_context(|| format!("opening {} for append", dst.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut total = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        whole.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        total += n as u64;
        bar.inc(n as u64);
    }
    out.sync_all()?;
    bar.finish_and_clear();
    Ok((hex(hasher.finalize().as_slice()), total))
}

/// sha256 of a whole file, in bounded memory, with a progress bar.
fn sha256_file(path: &Path, label: &str) -> Result<String> {
    let len = std::fs::metadata(path)?.len();
    let bar = ProgressBar::new(len);
    bar.set_style(crate::fetch::bar_style());
    bar.set_message(label.to_string());
    let mut reader = BufReader::with_capacity(8 << 20, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        bar.inc(n as u64);
    }
    bar.finish_and_clear();
    Ok(hex(hasher.finalize().as_slice()))
}

/// Drop a staged part that has been appended, so the staging tree never
/// holds more than the part in flight. Fixtures and shared caches the
/// fetcher does not own are left alone.
fn discard_staged(fetcher: &dyn Fetcher, repo: &str, src: &Path) {
    let owned = fetcher
        .staging_dir(repo)
        .is_some_and(|dir| src.starts_with(&dir));
    if !owned {
        return;
    }
    // hf-hub's snapshot path is a symlink into `blobs/`; remove both so the
    // bytes actually go away.
    if let Ok(real) = std::fs::canonicalize(src) {
        let _ = std::fs::remove_file(&real);
    }
    let _ = std::fs::remove_file(src);
}

/// Headers past this are corrupt or hostile: refuse rather than allocate.
/// The same bound the remote header scanner applies.
const MAX_HEADER_LEN: u64 = 256 * 1024 * 1024;

/// Flags that promise an extension-slot section after the weights blob.
const SLOT_FLAGS: base_format::HeaderFlags = base_format::HeaderFlags::HAS_LORA
    .union(base_format::HeaderFlags::HAS_SPECULATOR)
    .union(base_format::HeaderFlags::HAS_COMPUTE_GRAPH)
    .union(base_format::HeaderFlags::HAS_KV_WARMUP)
    .union(base_format::HeaderFlags::HAS_TRACE_REF)
    .union(base_format::HeaderFlags::ROPE_PRECOMPUTED);

/// Is a stitched `.base` of `len` bytes everything its header describes?
///
/// Two things have to hold. The weights blob must fit: the file is at least
/// the blob start plus the furthest tensor extent. And the slot section,
/// which follows the blob, must be whole: every slot record the count
/// promises lies inside the file, and when the header's flags advertise
/// slots there is at least one. A file cut exactly at the blob end with
/// slots the header never flagged is the one case nothing here can see.
///
/// The header comes from an arbitrary repository, so its numbers are not
/// trusted: the allocation is bounded and every sum is checked.
fn check_complete(path: &Path, len: u64) -> Result<()> {
    let mut f = File::open(path)?;
    let mut prefix = [0u8; PREFIX_LEN as usize];
    f.read_exact(&mut prefix)
        .context("shorter than a .base prefix")?;
    if prefix[0..4] != MAGIC {
        bail!("not a .base file (bad magic)");
    }
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        bail!("unsupported .base format version {version}");
    }
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    if header_len == 0 || header_len >= MAX_HEADER_LEN {
        bail!("implausible header length {header_len}");
    }
    let mut json = vec![0u8; header_len as usize];
    f.read_exact(&mut json)
        .context("file ends inside the header")?;
    let header = base_format::Header::from_json_bytes(&json).context("parsing the header")?;

    let overflow = || anyhow::anyhow!("tensor extents overflow: the header is corrupt");
    let blob_offset = (PREFIX_LEN + header_len)
        .checked_next_multiple_of(BLOB_ALIGNMENT)
        .ok_or_else(overflow)?;
    let mut blob_end = blob_offset;
    let mmproj = header.mmproj.iter().flat_map(|m| m.tensors.iter());
    for t in header.tensors.iter().chain(mmproj) {
        let regions = [
            (Some(t.offset), Some(t.length)),
            (t.scale_offset, t.scale_length),
            (t.bias_offset, t.bias_length),
            (t.awq_scale_offset, t.awq_scale_length),
        ];
        for (offset, length) in regions {
            let (Some(o), Some(l)) = (offset, length) else {
                continue;
            };
            let end = o
                .checked_add(l)
                .and_then(|e| e.checked_add(blob_offset))
                .ok_or_else(overflow)?;
            blob_end = blob_end.max(end);
        }
    }
    if len < blob_end {
        bail!("the weights blob needs {blob_end} bytes, the file has {len}");
    }

    let slots = walk_slots(&mut f, blob_end, len).context("reading the extension slots")?;
    if header.flags.intersects(SLOT_FLAGS) && slots == 0 {
        bail!(
            "the header advertises extension slots (flags {:?}) but the file ends at the weights blob",
            header.flags & SLOT_FLAGS
        );
    }
    Ok(())
}

/// Count the slot records after the blob, reading only their headers and
/// stepping over the payloads, so a multi-GB LoRA or speculator costs a
/// few seeks. Zero when nothing follows the blob. An error when the count
/// promises records the file does not hold.
fn walk_slots(f: &mut File, blob_end: u64, len: u64) -> Result<u32> {
    use std::io::{Seek, SeekFrom};
    // The section starts at the first 8-byte boundary after the blob.
    let start = blob_end
        .checked_next_multiple_of(8)
        .context("slot offset overflow")?;
    if start >= len {
        return Ok(0);
    }
    f.seek(SeekFrom::Start(start))?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4).context("truncated slot count")?;
    let n = u32::from_le_bytes(buf4);
    let mut pos = start + 4;
    for i in 0..n {
        // u16 kind, u16 flags, u64 payload_length, u64 xxh64.
        let mut rec = [0u8; 20];
        f.read_exact(&mut rec)
            .with_context(|| format!("slot {i} of {n}: truncated record header"))?;
        let payload_len = u64::from_le_bytes(rec[4..12].try_into().unwrap());
        let payload_end = pos
            .checked_add(20)
            .and_then(|p| p.checked_add(payload_len))
            .context("slot payload overflow")?;
        if payload_end > len {
            bail!("slot {i} of {n}: payload runs to byte {payload_end}, the file has {len}");
        }
        // The writer pads every payload to 8 bytes, the last one included,
        // and the canonical reader consumes that padding with read_exact.
        pos = payload_end
            .checked_next_multiple_of(8)
            .context("slot padding overflow")?;
        if pos > len {
            bail!("slot {i} of {n}: padding runs to byte {pos}, the file has {len}");
        }
        f.seek(SeekFrom::Start(pos))?;
    }
    Ok(n)
}

/// A real, tiny `.base`: a valid header describing one f32 tensor of
/// `blob_len` bytes, then that blob. Test support for anything that has to
/// feed [`install`] bytes it will accept, which is why it is public: the
/// pull command's tests live in another crate.
#[doc(hidden)]
pub fn synthetic_bundle(blob_len: u64) -> Vec<u8> {
    synthetic_bundle_with_slot(blob_len, None, false)
}

/// [`synthetic_bundle`] with, optionally, one extension slot holding
/// `slot` after the blob, and the header flagged as carrying LoRA when
/// `flagged` (whether or not a slot is actually written).
#[doc(hidden)]
pub fn synthetic_bundle_with_slot(blob_len: u64, slot: Option<&[u8]>, flagged: bool) -> Vec<u8> {
    let header = serde_json::json!({
        "schema": 1,
        "arch": "test",
        "quant_scheme": "base_q4",
        "min_hw": "apple_m1",
        "created": "2026-09-08T00:00:00Z",
        "baserT_version": "0.2.4",
        "source": { "format": "test", "sha256": "0".repeat(64), "filename": "x" },
        "tokenizer": {},
        "config": {},
        "flags": if flagged { "0x00000010" } else { "0x00000000" },
        "tensors": [{
            "name": "w", "dtype": "f32", "shape": [blob_len / 4],
            "offset": 0, "length": blob_len
        }]
    })
    .to_string();
    let mut v = MAGIC.to_vec();
    v.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    v.extend_from_slice(&(header.len() as u64).to_le_bytes());
    v.extend_from_slice(header.as_bytes());
    let blob_offset = (v.len() as u64).div_ceil(BLOB_ALIGNMENT) * BLOB_ALIGNMENT;
    v.resize(blob_offset as usize, 0);
    v.extend((0..blob_len).map(|i| (i % 251) as u8));
    if let Some(payload) = slot {
        // Slot section: pad to 8, u32 count, then one record (kind, flags,
        // payload length, xxh64 of zero = unchecked) and its payload.
        v.resize((v.len() as u64).div_ceil(8) as usize * 8, 0);
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&0x0001u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(payload);
        v.resize((v.len() as u64).div_ceil(8) as usize * 8, 0);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::MockFetcher;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn part_names_parse_and_reject() {
        assert_eq!(
            split_part_name("parts/GLM-5.2-Q4.base.part-007"),
            Some(("parts/GLM-5.2-Q4.base".to_string(), 7))
        );
        assert_eq!(split_part_name("m-Q4.base"), None);
        assert_eq!(split_part_name("m-Q4.base.part-"), None);
        assert_eq!(split_part_name("m-Q4.base.part-1a"), None);
        assert_eq!(split_part_name("m-Q4.safetensors.part-001"), None);
        // The manifest sits beside the parts and is neither.
        assert_eq!(split_part_name("m-Q4.base.manifest.json"), None);
    }

    #[test]
    fn group_orders_parts_and_keeps_whole_files() {
        // Listing order is not byte order; the group must be.
        let got = group(names(&[
            "README.md",
            "parts/g.base.part-002",
            "m-Q8.base",
            "parts/g.base.part-000",
            "parts/g.base.manifest.json",
            "parts/g.base.part-001",
        ]));
        assert_eq!(
            got.artifacts,
            vec![
                Artifact::whole("m-Q8.base"),
                Artifact {
                    name: "parts/g.base".into(),
                    parts: names(&[
                        "parts/g.base.part-000",
                        "parts/g.base.part-001",
                        "parts/g.base.part-002"
                    ]),
                },
            ]
        );
        assert!(got.malformed.is_empty());
    }

    #[test]
    fn group_isolates_a_gapped_set_and_prefers_a_whole_file() {
        // A quant mid-upload must not take the repo's other bundles with it.
        let got = group(names(&[
            "g.base.part-000",
            "g.base.part-002",
            "m-Q8.base",
            "h.base.part-000",
        ]));
        assert_eq!(
            got.artifacts,
            vec![
                Artifact::whole("m-Q8.base"),
                Artifact {
                    name: "h.base".into(),
                    parts: names(&["h.base.part-000"]),
                },
            ]
        );
        assert_eq!(got.malformed.len(), 1);
        assert_eq!(got.malformed[0].0, "g.base");
        assert!(
            got.malformed[0].1.contains("missing part 001"),
            "{}",
            got.malformed[0].1
        );

        let got = group(names(&["g.base", "g.base.part-000", "g.base.part-001"]));
        assert_eq!(got.artifacts, vec![Artifact::whole("g.base")]);
        assert!(got.malformed.is_empty());
    }

    #[test]
    fn group_rejects_two_spellings_of_one_index() {
        let got = group(names(&[
            "g.base.part-000",
            "g.base.part-0000",
            "g.base.part-001",
            "m-Q8.base",
        ]));
        assert_eq!(got.artifacts, vec![Artifact::whole("m-Q8.base")]);
        assert_eq!(got.malformed.len(), 1);
        assert_eq!(got.malformed[0].0, "g.base");
        assert!(
            got.malformed[0].1.contains("index 0 is listed twice"),
            "{}",
            got.malformed[0].1
        );
    }

    #[test]
    fn find_names_the_malformed_set_it_was_asked_for() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("org").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        for f in ["m-Q4.base.part-000", "m-Q4.base.part-002", "m-Q8.base"] {
            std::fs::write(repo_dir.join(f), b"x").unwrap();
        }
        let fetcher = MockFetcher::new(tmp.path());
        let err = find(&fetcher, "org/m", "main", "m-Q4.base")
            .unwrap_err()
            .to_string();
        assert!(err.contains("missing part 001"), "{err}");
        // The sibling is unaffected.
        assert_eq!(
            find(&fetcher, "org/m", "main", "m-Q8.base").unwrap(),
            Artifact::whole("m-Q8.base")
        );
    }

    /// Cut `bytes` into `n` parts of roughly equal size.
    fn split(bytes: &[u8], n: usize) -> Vec<Vec<u8>> {
        let each = bytes.len().div_ceil(n);
        bytes.chunks(each).map(<[u8]>::to_vec).collect()
    }

    fn sha(bytes: &[u8]) -> String {
        hex(Sha256::digest(bytes).as_slice())
    }

    /// A mock repo `org/m` holding `m-Q4.base.part-NNN` for each of `parts`,
    /// with a manifest built from exactly those files — what an honest
    /// publisher ships.
    fn fixture_repo(tmp: &Path, parts: &[Vec<u8>]) -> (MockFetcher, PathBuf) {
        let (fetcher, repo_dir) = fixture_repo_bare(tmp, parts);
        let paths: Vec<PathBuf> = (0..parts.len())
            .map(|i| repo_dir.join(format!("m-Q4.base.part-{i:03}")))
            .collect();
        let manifest = build_manifest(&paths).unwrap();
        write_manifest(&repo_dir, &manifest);
        (fetcher, repo_dir)
    }

    /// The same repo with no manifest at all.
    fn fixture_repo_bare(tmp: &Path, parts: &[Vec<u8>]) -> (MockFetcher, PathBuf) {
        let repo_dir = tmp.join("org").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        for (i, bytes) in parts.iter().enumerate() {
            std::fs::write(repo_dir.join(format!("m-Q4.base.part-{i:03}")), bytes).unwrap();
        }
        (MockFetcher::new(tmp), repo_dir)
    }

    fn write_manifest(repo_dir: &Path, manifest: &Manifest) {
        std::fs::write(
            repo_dir.join(manifest_name("m-Q4.base")),
            serde_json::to_vec_pretty(manifest).unwrap(),
        )
        .unwrap();
    }

    fn dst_in(tmp: &Path) -> PathBuf {
        let dst = tmp.join("out").join("model.base");
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        dst
    }

    fn record_for(art: &Artifact, manifest: &Manifest, appended: usize, len: u64) -> Record {
        Record {
            repo: "org/m".into(),
            revision: "main".into(),
            parts: art.parts.clone(),
            ids: manifest.part_ids(),
            appended,
            len,
        }
    }

    fn manifest_of(parts: &[Vec<u8>], tmp: &Path) -> Manifest {
        let dir = tmp.join("manifest-src");
        std::fs::create_dir_all(&dir).unwrap();
        let paths: Vec<PathBuf> = parts
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let p = dir.join(format!("m-Q4.base.part-{i:03}"));
                std::fs::write(&p, b).unwrap();
                p
            })
            .collect();
        build_manifest(&paths).unwrap()
    }

    #[test]
    fn build_manifest_hashes_each_part_and_the_whole_in_one_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let m = manifest_of(&parts, tmp.path());
        assert_eq!(m.size, whole.len() as u64);
        assert_eq!(m.sha256, sha(&whole));
        assert_eq!(m.parts.len(), 3);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(m.parts[i].name, format!("m-Q4.base.part-{i:03}"));
            assert_eq!(m.parts[i].size, p.len() as u64);
            assert_eq!(m.parts[i].sha256, sha(p));
        }
        m.check_against(&names(&[
            "parts/m-Q4.base.part-000",
            "parts/m-Q4.base.part-001",
            "parts/m-Q4.base.part-002",
        ]))
        .unwrap();
        let err = m
            .check_against(&names(&["m-Q4.base.part-000", "m-Q4.base.part-001"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("lists 3 parts but the repo has 2"), "{err}");

        // A whole-file hash that is not one is refused up front.
        let mut bad = m.clone();
        bad.sha256 = "not-a-digest".into();
        let err = bad
            .check_against(&names(&[
                "m-Q4.base.part-000",
                "m-Q4.base.part-001",
                "m-Q4.base.part-002",
            ]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole-file sha256"), "{err}");

        // Sizes are compared one by one, not only as a sum.
        let sizes: Vec<u64> = parts.iter().map(|p| p.len() as u64).collect();
        m.check_sizes(&sizes).unwrap();
        let mut swapped = sizes.clone();
        swapped[0] += 1;
        swapped[1] -= 1;
        let err = m.check_sizes(&swapped).unwrap_err().to_string();
        assert!(err.contains("bytes but the repo holds"), "{err}");
    }

    #[test]
    fn install_reassembles_parts_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let (fetcher, repo_dir) = fixture_repo(tmp.path(), &split(&whole, 3));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        assert_eq!(art.parts.len(), 3);

        let dst = dst_in(tmp.path());
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();

        assert_eq!(std::fs::read(&dst).unwrap(), whole);
        assert!(!partial_path(&dst).exists(), "partial must be renamed away");
        assert!(!record_path(&dst).exists(), "record must be cleared");
        assert!(!lock_path(&dst).exists(), "lock must be cleared");
        // Fixtures are not owned by the mock fetcher: never deleted.
        assert!(repo_dir.join("m-Q4.base.part-002").exists());
    }

    #[test]
    fn install_needs_a_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let (fetcher, _) = fixture_repo_bare(tmp.path(), &split(&whole, 2));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("needs its manifest"), "{err}");
        assert!(!dst.exists());
    }

    #[test]
    fn install_refuses_an_oversized_manifest_before_reading_it() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let (fetcher, repo_dir) = fixture_repo(tmp.path(), &split(&whole, 2));
        let big = vec![b' '; MAX_MANIFEST_LEN as usize + 1];
        std::fs::write(repo_dir.join(manifest_name("m-Q4.base")), &big).unwrap();
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("implausible manifest size"), "{err}");
    }

    #[test]
    fn install_resumes_from_the_recorded_part() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let (fetcher, _) = fixture_repo(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let manifest = manifest_of(&parts, tmp.path());
        let dst = dst_in(tmp.path());

        // Two parts landed, then the append of the third tore mid-way.
        let two: Vec<u8> = parts[..2].concat();
        let mut torn = two.clone();
        torn.extend_from_slice(b"XX");
        std::fs::write(partial_path(&dst), &torn).unwrap();
        let rec = record_for(&art, &manifest, 2, two.len() as u64);
        std::fs::write(record_path(&dst), serde_json::to_vec(&rec).unwrap()).unwrap();

        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_checks_the_whole_hash_on_a_fresh_install_too() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let (fetcher, repo_dir) = fixture_repo(tmp.path(), &parts);
        // Right parts, wrong whole: a manifest edited by hand, say.
        let mut m = manifest_of(&parts, tmp.path());
        m.sha256 = sha(b"not the whole");
        write_manifest(&repo_dir, &m);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = install(&fetcher, "org/m", "main", &art, &dst, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reassembled file hashed to"), "{err}");
        assert!(!dst.exists());
        assert!(!partial_path(&dst).exists());
    }

    #[test]
    fn install_rehashes_a_resumed_partial_against_the_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let (fetcher, _) = fixture_repo(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let manifest = manifest_of(&parts, tmp.path());
        let dst = dst_in(tmp.path());

        // Two parts were appended by an earlier attempt; since then one
        // byte inside them changed without changing the length, which the
        // record alone cannot see.
        let mut two: Vec<u8> = parts[..2].concat();
        two[100] ^= 0xff;
        std::fs::write(partial_path(&dst), &two).unwrap();
        let rec = record_for(&art, &manifest, 2, two.len() as u64);
        std::fs::write(record_path(&dst), serde_json::to_vec(&rec).unwrap()).unwrap();

        let err = install(&fetcher, "org/m", "main", &art, &dst, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("resumed file hashed to"), "{err}");
        assert!(!dst.exists());
        assert!(
            !partial_path(&dst).exists(),
            "a damaged partial is not kept"
        );

        // The next attempt starts clean and succeeds.
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_restarts_when_the_record_describes_other_parts() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 2);
        let (fetcher, _) = fixture_repo(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let manifest = manifest_of(&parts, tmp.path());
        let dst = dst_in(tmp.path());

        std::fs::write(partial_path(&dst), b"stale").unwrap();
        let rec = Record {
            parts: names(&["other.base.part-000", "other.base.part-001"]),
            ..record_for(&art, &manifest, 1, 5)
        };
        std::fs::write(record_path(&dst), serde_json::to_vec(&rec).unwrap()).unwrap();

        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    /// Mock whose parts carry content ids and whose branch resolves to a
    /// commit, like the Hub.
    struct HubLikeFetcher {
        inner: MockFetcher,
        ids: Vec<Option<String>>,
        commit: String,
    }

    impl Fetcher for HubLikeFetcher {
        fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
            assert_eq!(
                revision, self.commit,
                "files must be fetched at the pinned commit"
            );
            self.inner.get_file(repo, revision, filename)
        }
        fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
            self.inner.list_files(repo, revision)
        }
        fn resolve_revision(&self, _: &str, _: &str) -> Result<String> {
            Ok(self.commit.clone())
        }
        fn content_id(&self, _: &str, revision: &str, filename: &str) -> Result<Option<String>> {
            assert_eq!(revision, self.commit);
            let (_, idx) = split_part_name(filename).unwrap();
            Ok(self.ids[idx as usize].clone())
        }
    }

    fn hub_like(tmp: &Path, parts: &[Vec<u8>]) -> HubLikeFetcher {
        let (inner, _) = fixture_repo(tmp, parts);
        HubLikeFetcher {
            inner,
            ids: parts.iter().map(|p| Some(sha(p))).collect(),
            commit: "c0ffee".into(),
        }
    }

    #[test]
    fn install_pins_the_revision_and_verifies_each_part() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let fetcher = hub_like(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    /// Mock whose branch listing and pinned-commit listing differ: the
    /// branch gained a part after the caller listed it.
    struct AdvancingFetcher {
        inner: MockFetcher,
        commit: String,
    }

    impl Fetcher for AdvancingFetcher {
        fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
            assert_eq!(revision, self.commit);
            self.inner.get_file(repo, revision, filename)
        }
        fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
            let all = self.inner.list_files(repo, revision)?;
            if revision == self.commit {
                Ok(all)
            } else {
                // The branch, as the caller saw it: one part short.
                Ok(all
                    .into_iter()
                    .filter(|f| !f.ends_with("part-002"))
                    .collect())
            }
        }
        fn resolve_revision(&self, _: &str, _: &str) -> Result<String> {
            Ok(self.commit.clone())
        }
    }

    /// Mock whose branch listing shows a whole file where the pinned
    /// commit holds a part set: the publication changed shape after the
    /// caller listed.
    struct ResplitFetcher {
        inner: MockFetcher,
        commit: String,
    }

    impl Fetcher for ResplitFetcher {
        fn get_file(&self, repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
            assert_eq!(revision, self.commit);
            self.inner.get_file(repo, revision, filename)
        }
        fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
            if revision == self.commit {
                self.inner.list_files(repo, revision)
            } else {
                Ok(vec!["m-Q4.base".to_string()])
            }
        }
        fn resolve_revision(&self, _: &str, _: &str) -> Result<String> {
            Ok(self.commit.clone())
        }
    }

    #[test]
    fn install_decides_whole_or_split_at_the_pinned_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let (inner, _) = fixture_repo(tmp.path(), &split(&whole, 3));
        let fetcher = ResplitFetcher {
            inner,
            commit: "c0ffee".into(),
        };
        // The caller saw a whole file on the branch; the commit has parts.
        let seen = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        assert!(!seen.is_split());
        let dst = dst_in(tmp.path());
        install(&fetcher, "org/m", "main", &seen, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_relists_the_parts_at_the_pinned_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let (inner, _) = fixture_repo(tmp.path(), &split(&whole, 3));
        let fetcher = AdvancingFetcher {
            inner,
            commit: "c0ffee".into(),
        };
        // Listed from the branch: two parts. Installed from the pinned
        // commit: all three, so the bundle is whole.
        let stale = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        assert_eq!(stale.parts.len(), 2);
        let dst = dst_in(tmp.path());
        install(&fetcher, "org/m", "main", &stale, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_refuses_a_manifest_the_hub_disagrees_with() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let mut fetcher = hub_like(tmp.path(), &parts);
        // The Hub's LFS hash for part 1 is not what the manifest says: a
        // part was republished without republishing the manifest.
        fetcher.ids[1] = Some(sha(b"something else"));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = install(&fetcher, "org/m", "main", &art, &dst, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("manifest is stale"), "{err}");
        assert!(!dst.exists());
        assert!(!partial_path(&dst).exists(), "refused before any byte");
    }

    #[test]
    fn install_rejects_a_part_whose_bytes_do_not_match_the_manifest() {
        // Corruption the listing cannot see (the mock reports no ids):
        // the bytes that arrive differ from what the manifest promised.
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let (fetcher, repo_dir) = fixture_repo(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());

        // Same length, one byte flipped.
        let mut flipped = parts[1].clone();
        flipped[10] ^= 0xff;
        std::fs::write(repo_dir.join("m-Q4.base.part-001"), &flipped).unwrap();
        let err = install(&fetcher, "org/m", "main", &art, &dst, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("hashed to"), "{err}");
        assert!(
            !partial_path(&dst).exists(),
            "a bad partial has no resume value"
        );

        // Shorter than promised.
        std::fs::write(repo_dir.join("m-Q4.base.part-001"), &parts[1][..100]).unwrap();
        let err = install(&fetcher, "org/m", "main", &art, &dst, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bytes but the manifest says"), "{err}");
        assert!(!dst.exists());
    }

    #[test]
    fn install_restarts_when_a_part_was_republished_under_the_same_name() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let fetcher = hub_like(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let manifest = manifest_of(&parts, tmp.path());
        let dst = dst_in(tmp.path());

        // Two parts were assembled from the previous publication, whose
        // part 000 had different bytes. Splicing the new tail onto them
        // would install a bundle nobody published.
        let mut old_ids = manifest.part_ids();
        old_ids[0] = sha(b"previous part 0");
        std::fs::write(partial_path(&dst), b"OLD-HEAD").unwrap();
        let rec = Record {
            revision: fetcher.commit.clone(),
            ids: old_ids,
            ..record_for(&art, &manifest, 2, 8)
        };
        std::fs::write(record_path(&dst), serde_json::to_vec(&rec).unwrap()).unwrap();

        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);

        // Same ids and commit: the record is honored and the tail appended.
        std::fs::remove_file(&dst).unwrap();
        let two: Vec<u8> = parts[..2].concat();
        std::fs::write(partial_path(&dst), &two).unwrap();
        let rec = Record {
            ids: manifest.part_ids(),
            len: two.len() as u64,
            ..rec
        };
        std::fs::write(record_path(&dst), serde_json::to_vec(&rec).unwrap()).unwrap();
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_checks_the_manifest_against_a_catalog_row() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        let fetcher = hub_like(tmp.path(), &parts);
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let pinned: Vec<String> = parts.iter().map(|p| sha(p)).collect();

        // Agreeing row: installs.
        install(&fetcher, "org/m", "main", &art, &dst, Some(&pinned)).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
        std::fs::remove_file(&dst).unwrap();

        // The catalog pinned a different part 2: refused before any byte.
        let mut stale = pinned.clone();
        stale[2] = sha(b"older publication");
        let err = install(&fetcher, "org/m", "main", &art, &dst, Some(&stale))
            .unwrap_err()
            .to_string();
        assert!(err.contains("republished"), "{err}");
        assert!(!partial_path(&dst).exists());

        // A different part count is the same story.
        let err = install(&fetcher, "org/m", "main", &art, &dst, Some(&pinned[..2]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("pins 2 parts"), "{err}");
    }

    #[test]
    fn install_refuses_a_part_set_missing_its_tail() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        // The publisher wrote the manifest for three parts and uploaded
        // two: gap-free, and nothing in the names says a third was meant
        // to exist. The manifest does.
        let (fetcher, repo_dir) = fixture_repo(tmp.path(), &parts);
        std::fs::remove_file(repo_dir.join("m-Q4.base.part-002")).unwrap();
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        assert_eq!(art.parts.len(), 2);
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("lists 3 parts but the repo has 2"), "{err}");
        assert!(!dst.exists(), "a truncated bundle must not be installed");
        assert!(!partial_path(&dst).exists());
    }

    #[test]
    fn install_refuses_a_slot_only_tail_the_header_cannot_see() {
        // A bundle ending in an unflagged slot (calibration data, say),
        // cut so the last part is exactly that slot section, and that part
        // never uploaded. The header is silent about the slot; only the
        // manifest knows a third part existed.
        let tmp = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 300];
        let whole = synthetic_bundle_with_slot(1000, Some(&payload), false);
        let no_slot = synthetic_bundle_with_slot(1000, None, false);
        let mut parts = split(&no_slot, 2);
        parts.push(whole[no_slot.len()..].to_vec());
        assert_eq!(parts.concat(), whole);
        let (fetcher, repo_dir) = fixture_repo(tmp.path(), &parts);
        std::fs::remove_file(repo_dir.join("m-Q4.base.part-002")).unwrap();
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("lists 3 parts but the repo has 2"), "{err}");
        assert!(!dst.exists());
    }

    #[test]
    fn install_refuses_a_bundle_whose_advertised_slots_were_in_the_lost_tail() {
        // The header-level defence, for a manifest written from an already
        // broken set: the flags promise slots and the file ends at the blob.
        let tmp = tempfile::tempdir().unwrap();
        let no_slot = synthetic_bundle_with_slot(1000, None, true);
        let (fetcher, _) = fixture_repo(tmp.path(), &split(&no_slot, 2));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("advertises extension slots"), "{err}");
        assert!(!dst.exists());

        // With the slot present, the same bundle installs.
        let tmp = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 300];
        let whole = synthetic_bundle_with_slot(1000, Some(&payload), true);
        let (fetcher, _) = fixture_repo(tmp.path(), &split(&whole, 3));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_refuses_a_slot_section_missing_only_its_padding() {
        let tmp = tempfile::tempdir().unwrap();
        // A 301-byte payload is followed by 3 pad bytes; lose just those.
        let payload = vec![7u8; 301];
        let whole = synthetic_bundle_with_slot(1000, Some(&payload), false);
        assert_eq!(whole.len() % 8, 0);
        let cut = &whole[..whole.len() - 3];
        let (fetcher, _) = fixture_repo(tmp.path(), &split(cut, 2));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("padding runs to byte"), "{err}");
        assert!(!dst.exists());
    }

    #[test]
    fn install_refuses_a_slot_section_cut_short_even_when_unflagged() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 300];
        let whole = synthetic_bundle_with_slot(1000, Some(&payload), false);
        // Lose the last 100 bytes: inside the slot payload.
        let cut = &whole[..whole.len() - 100];
        let (fetcher, _) = fixture_repo(tmp.path(), &split(cut, 2));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());
        let err = format!(
            "{:#}",
            install(&fetcher, "org/m", "main", &art, &dst, None).unwrap_err()
        );
        assert!(err.contains("payload runs to byte"), "{err}");
        assert!(!dst.exists());
    }

    #[test]
    fn check_complete_bounds_the_header_and_checks_its_sums() {
        let tmp = tempfile::tempdir().unwrap();
        // A prefix claiming a 1 GiB header on a 16-byte file: refused
        // before any allocation.
        let mut v = MAGIC.to_vec();
        v.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        v.extend_from_slice(&(1u64 << 30).to_le_bytes());
        let p = tmp.path().join("huge.base");
        std::fs::write(&p, &v).unwrap();
        let err = check_complete(&p, v.len() as u64).unwrap_err().to_string();
        assert!(err.contains("implausible header length"), "{err}");

        // A tensor whose offset + length wraps u64: reported, not wrapped
        // into a small requirement.
        let mut whole = synthetic_bundle(64);
        let json_start = PREFIX_LEN as usize;
        let header_len = u64::from_le_bytes(whole[8..16].try_into().unwrap()) as usize;
        let json = String::from_utf8(whole[json_start..json_start + header_len].to_vec()).unwrap();
        let bad = json.replace("\"offset\":0", &format!("\"offset\":{}", u64::MAX - 8));
        assert_ne!(bad, json);
        let mut rebuilt = whole[..8].to_vec();
        rebuilt.extend_from_slice(&(bad.len() as u64).to_le_bytes());
        rebuilt.extend_from_slice(bad.as_bytes());
        let blob_offset = (rebuilt.len() as u64).div_ceil(BLOB_ALIGNMENT) * BLOB_ALIGNMENT;
        rebuilt.resize(blob_offset as usize, 0);
        rebuilt.extend_from_slice(&whole.split_off(whole.len() - 64));
        let p = tmp.path().join("wrap.base");
        std::fs::write(&p, &rebuilt).unwrap();
        let err = check_complete(&p, rebuilt.len() as u64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflow"), "{err}");
    }

    #[test]
    fn install_is_exclusive_per_destination() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = synthetic_bundle(1000);
        let (fetcher, _) = fixture_repo(tmp.path(), &split(&whole, 2));
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = dst_in(tmp.path());

        // Another pull holds the destination.
        let other = Lock::acquire(&dst).unwrap();
        let err = install(&fetcher, "org/m", "main", &art, &dst, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("another basert pull"), "{err}");
        other.release();

        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), whole);
    }

    #[test]
    fn install_of_a_whole_file_is_a_plain_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("org").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("m-Q4.base"), b"whole").unwrap();
        let fetcher = MockFetcher::new(tmp.path());
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        assert!(!art.is_split());
        let dst = tmp.path().join("model.base");
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"whole");
    }

    /// Owned staging tree in hf-hub's shape, so the space-bounding deletes
    /// can be observed.
    struct StagedFetcher {
        staging: PathBuf,
    }

    impl StagedFetcher {
        fn repo_dir(&self, repo: &str) -> PathBuf {
            self.staging
                .join(format!("models--{}", repo.replace('/', "--")))
        }

        fn stage(&self, repo: &str, revision: &str, filename: &str, bytes: &[u8]) {
            let rdir = self.repo_dir(repo);
            let blobs = rdir.join("blobs");
            let snap = rdir.join("snapshots").join(revision);
            std::fs::create_dir_all(&blobs).unwrap();
            std::fs::create_dir_all(&snap).unwrap();
            let blob = blobs.join(format!("etag-{filename}"));
            std::fs::write(&blob, bytes).unwrap();
            std::os::unix::fs::symlink(&blob, snap.join(filename)).unwrap();
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
    fn install_from_staging_frees_each_part_as_it_lands() {
        let tmp = tempfile::tempdir().unwrap();
        let fetcher = StagedFetcher {
            staging: tmp.path().join("staging"),
        };
        let whole = synthetic_bundle(1000);
        let parts = split(&whole, 3);
        for (i, b) in parts.iter().enumerate() {
            fetcher.stage("org/m", "main", &format!("m-Q4.base.part-{i:03}"), b);
        }
        let manifest = manifest_of(&parts, tmp.path());
        fetcher.stage(
            "org/m",
            "main",
            &manifest_name("m-Q4.base"),
            &serde_json::to_vec(&manifest).unwrap(),
        );
        let art = find(&fetcher, "org/m", "main", "m-Q4.base").unwrap();
        let dst = tmp.path().join("model.base");
        install(&fetcher, "org/m", "main", &art, &dst, None).unwrap();

        assert_eq!(std::fs::read(&dst).unwrap(), whole);
        let blobs = fetcher.repo_dir("org/m").join("blobs");
        let left: Vec<String> = std::fs::read_dir(&blobs)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.contains("manifest"))
            .collect();
        assert!(
            left.is_empty(),
            "every staged part must be consumed, found {left:?}"
        );
    }
}
