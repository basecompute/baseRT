//! Derive the catalog by scanning an organization's published bundles.
//!
//! [`crate::gen`] makes a catalog row from a `.base` file the publisher has on
//! disk. That covers the moment a bundle is built and misses everything after
//! it: a model published from another machine, or by another person, or a year
//! ago, never reaches the catalog at all unless someone remembers to hand-write
//! a row. The result is silent drift — at the time this module was written, 14
//! of the 37 repositories under `basecompute` were missing from the catalog,
//! including every Whisper bundle, so `basert list --remote` showed a third
//! less than was actually shipped.
//!
//! So the catalog is derived from the Hub instead, which is the thing that is
//! actually true. Three sources, none of which requires downloading an
//! artifact:
//!
//! * the org listing (`/api/models?author=…`) names the repositories;
//! * each repository's tree gives every file's path, size, and LFS sha256 —
//!   two of the five fields a row carries, straight from the Hub;
//! * two ranged reads of each `.base` give its header, which knows the arch,
//!   the backend it was packed for, and its quant profile.
//!
//! A 20GB bundle costs a few megabytes to describe. Rows are then built by
//! [`crate::gen::entry_from_header`] — the same function the local path uses,
//! so a scanned row and a published row cannot disagree.
//!
//! Scans are incremental against a known catalog: a file whose sha256 already
//! appears is described by the row that is already there, so a rescan reads
//! headers only for what is new or changed.

use crate::catalog::{Catalog, CatalogEntry};
use crate::fetch::Fetcher;
use anyhow::{Context, Result};
use base_format::{Header, PREFIX_LEN};

/// A file as the Hub describes it, before anything is downloaded.
#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    /// Content sha256 — the LFS `oid`, which is what a catalog row records.
    pub sha256: Option<String>,
}

/// Where the file list comes from. A trait so the scan is testable without a
/// network, and because the real implementation cannot use hf-hub's typed
/// tree listing: its `BlobLfsInfo` decodes `sha256`/`pointer_size` while the
/// Hub sends `oid`/`pointerSize`, so every hash comes back `None` and every
/// bundle would look unpinnable. Reading the tree directly is both correct and
/// one less layer.
pub trait RepoIndex {
    fn files(&self, repo: &str) -> Result<Vec<RemoteFile>>;
}

/// The real Hub.
pub struct HubApi;

impl RepoIndex for HubApi {
    fn files(&self, repo: &str) -> Result<Vec<RemoteFile>> {
        let url = format!("{}/api/models/{repo}/tree/main?recursive=true", endpoint());
        let body = ureq::get(&url)
            .call()
            .with_context(|| format!("listing files in {repo}"))?
            .into_body()
            .read_to_string()
            .with_context(|| format!("reading the file listing for {repo}"))?;
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(&body).with_context(|| format!("parsing {repo}'s tree"))?;
        Ok(entries
            .iter()
            .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("file"))
            .filter_map(|e| {
                let path = e.get("path")?.as_str()?.to_string();
                let lfs = e.get("lfs");
                Some(RemoteFile {
                    size: lfs
                        .and_then(|l| l.get("size"))
                        .or_else(|| e.get("size"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    sha256: lfs
                        .and_then(|l| l.get("oid"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    path,
                })
            })
            .collect())
    }
}

/// Hub API base. `$HF_ENDPOINT` redirects the whole scan at a mirror, the same
/// variable hf-hub honors for downloads.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// What a scan found, including what it deliberately did not publish.
#[derive(Debug, Default)]
pub struct ScanReport {
    /// Rows for every publishable bundle, sorted for a stable catalog file.
    pub entries: Vec<CatalogEntry>,
    /// Files that are `.base` but cannot be represented as a row, and why.
    /// Reported rather than dropped: a bundle that is published but
    /// uncatalogable is a fact someone needs to see.
    pub skipped: Vec<(String, String)>,
    /// Rows carried over from the known catalog because the file's sha256 was
    /// unchanged — no header read needed.
    pub reused: usize,
    /// Repositories that contained no `.base` file at all.
    pub empty_repos: Vec<String>,
}

impl ScanReport {
    /// Today's date, for the catalog's `updated` field. Kept here so every
    /// caller stamps it the same way.
    pub fn updated_stamp(&self) -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Civil date from a Unix timestamp (Howard Hinnant's algorithm), so
        // the catalog carries a real date without a chrono dependency.
        let days = (secs / 86_400) as i64;
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        format!("{y:04}-{m:02}-{d:02}")
    }
}

/// List every model repository under `org`, following pagination.
pub fn list_org_repos(org: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut url = format!("{}/api/models?author={org}&limit=100", endpoint());
    // The Hub paginates with a `Link: <…>; rel="next"` header. Bounded so a
    // malformed or cyclic Link chain cannot spin forever.
    for _ in 0..50 {
        let resp = ureq::get(&url)
            .call()
            .with_context(|| format!("listing models for {org}"))?;
        let next = resp
            .headers()
            .get("link")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_next_link);
        let body = resp
            .into_body()
            .read_to_string()
            .context("reading the model listing")?;
        let page: Vec<serde_json::Value> =
            serde_json::from_str(&body).context("parsing the model listing")?;
        if page.is_empty() {
            break;
        }
        out.extend(
            page.iter()
                .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string)),
        );
        match next {
            Some(n) => url = n,
            None => break,
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// Pull the `rel="next"` target out of a `Link` header.
fn parse_next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        if !part.contains("rel=\"next\"") {
            return None;
        }
        let start = part.find('<')? + 1;
        let end = part[start..].find('>')? + start;
        Some(part[start..end].to_string())
    })
}

/// The file already occupying `candidate`'s catalog identity, if any.
///
/// The catalog's uniqueness key is `(id, quant, backend)`. Two files claiming
/// it is a publishing mistake with no basis for picking a winner, so the second
/// is reported rather than emitted — and this is checked for carried-over rows
/// as well as freshly derived ones, since a duplicate that is already in the
/// catalog would otherwise be copied forward forever.
fn clashing_identity(entries: &[CatalogEntry], candidate: &CatalogEntry) -> Option<String> {
    entries
        .iter()
        .find(|e| {
            e.id == candidate.id && e.quant == candidate.quant && e.backend == candidate.backend
        })
        .map(|e| e.file.clone())
}

/// The `source_repo` a new row should carry: whatever a sibling row for the
/// same repository already records.
///
/// Which upstream model a bundle was converted from is a property of the
/// model, and no `.base` header records it. Without this, publishing a new
/// quant into an existing repo would regenerate its rows with the field
/// dropped.
fn inherited_source_repo(known: &Catalog, repo: &str) -> Option<String> {
    known
        .models
        .iter()
        .find(|m| m.hf_repo == repo && m.source_repo.is_some())
        .and_then(|m| m.source_repo.clone())
}

/// Read a published bundle's header without downloading the bundle.
///
/// Two reads, because the header's length is in the header: the 16-byte prefix
/// carries magic, format version, and the JSON length; the second read takes
/// exactly that many bytes. The JSON is large — several megabytes on a model
/// with a big tokenizer — but it is still four orders of magnitude smaller than
/// the artifact, and a fixed-size guess would be wrong in both directions.
pub fn read_remote_header(
    fetcher: &dyn Fetcher,
    repo: &str,
    revision: &str,
    filename: &str,
) -> Result<Header> {
    let prefix = fetcher.read_range(repo, revision, filename, 0..PREFIX_LEN)?;
    anyhow::ensure!(
        prefix.len() == PREFIX_LEN as usize,
        "{filename}: short prefix ({} bytes)",
        prefix.len()
    );
    anyhow::ensure!(&prefix[0..4] == b"BASE", "{filename}: not a .base file");
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    // A header claiming to be enormous is corrupt or hostile; refuse rather
    // than allocate it. 256MB is far past any real tokenizer.
    anyhow::ensure!(
        header_len > 0 && header_len < 256 * 1024 * 1024,
        "{filename}: implausible header length {header_len}"
    );
    let json = fetcher.read_range(
        repo,
        revision,
        filename,
        PREFIX_LEN..PREFIX_LEN + header_len,
    )?;
    Header::from_json_bytes(&json).with_context(|| format!("parsing {filename}'s header"))
}

/// Build catalog rows for every `.base` bundle published under `org`.
///
/// `known` supplies rows to reuse: when a file's sha256 is unchanged, its
/// existing row is kept verbatim, so the header read is skipped and any
/// hand-curated field on that row (`source_repo`, which no header knows)
/// survives the regeneration.
pub fn scan_org(
    fetcher: &dyn Fetcher,
    index: &dyn RepoIndex,
    org: &str,
    repos: &[String],
    known: &Catalog,
) -> Result<ScanReport> {
    let mut report = ScanReport::default();
    for repo in repos {
        let files = match index.files(repo) {
            Ok(f) => f,
            Err(e) => {
                report
                    .skipped
                    .push((repo.clone(), format!("listing failed: {e}")));
                continue;
            }
        };
        let bundles: Vec<_> = files
            .into_iter()
            .filter(|f| f.path.ends_with(".base"))
            .collect();
        if bundles.is_empty() {
            report.empty_repos.push(repo.clone());
            continue;
        }
        for f in bundles {
            let Some(sha256) = f.sha256.clone() else {
                report.skipped.push((
                    format!("{repo}/{}", f.path),
                    "the Hub reports no LFS sha256, so integrity could not be pinned".to_string(),
                ));
                continue;
            };
            // Unchanged file: keep the row that already describes it.
            if let Some(prev) = known.models.iter().find(|m| {
                m.hf_repo == *repo && m.file == f.path && m.sha256.as_deref() == Some(&sha256)
            }) {
                if let Some(clash) = clashing_identity(&report.entries, prev) {
                    report.skipped.push((
                        format!("{repo}/{}", f.path),
                        format!(
                            "claims the same catalog identity ({}, backend={:?}) as {}",
                            prev.quant, prev.backend, clash
                        ),
                    ));
                    continue;
                }
                report.entries.push(prev.clone());
                report.reused += 1;
                continue;
            }
            let header = match read_remote_header(fetcher, repo, "main", &f.path) {
                Ok(h) => h,
                Err(e) => {
                    report
                        .skipped
                        .push((format!("{repo}/{}", f.path), format!("{e:#}")));
                    continue;
                }
            };
            // `id` is the repo itself for a published bundle: the hub id and
            // the HF coordinates are the same thing on this org.
            match crate::gen::entry_from_header(&header, repo, repo, &f.path, f.size, sha256) {
                Ok(mut entry) => {
                    // Two files claiming one identity is a publishing mistake,
                    // not something to pick a winner for: the catalog's key is
                    // (id, quant, backend) and the resolver would have no basis
                    // to choose. Seen in the wild — `Qwen3-0.6B-cuda-q4.base`
                    // carries a Metal header, so it derives `default-q4` and
                    // collides with `Qwen3-0.6B-Q4.base`. Report both names so
                    // whoever published them can fix the bundle.
                    if let Some(clash) = clashing_identity(&report.entries, &entry) {
                        report.skipped.push((
                            format!("{repo}/{}", f.path),
                            format!(
                                "would claim the same catalog identity ({}, backend={:?}) as {} \
                                 — its header says target_backend={:?}, so check whether the \
                                 bundle or its filename is wrong",
                                entry.quant, entry.backend, clash, header.target_backend
                            ),
                        ));
                        continue;
                    }
                    // `source_repo` — which upstream model this was converted
                    // from — is a property of the model, and no header records
                    // it. Carry it across from any known row for the same repo
                    // so publishing a new quant does not quietly drop it from
                    // the ones already curated.
                    entry.source_repo = inherited_source_repo(known, repo);
                    report.entries.push(entry)
                }
                Err(e) => report
                    .skipped
                    .push((format!("{repo}/{}", f.path), format!("{e:#}"))),
            }
        }
    }
    let _ = org;
    // Stable order: a regenerated catalog should diff only where the Hub
    // changed, never because a listing came back in a different order.
    // Universal rows before backend-locked ones, then by quant. `Catalog::find`
    // returns the FIRST id match and the catalog's contract is that this is the
    // recommended default — a plain alphabetical sort puts `cuda-q4mix` ahead
    // of `default-q4` and quietly changes what `basert pull <id>` resolves to.
    report.entries.sort_by(|a, b| {
        (&a.id, a.backend.is_some(), &a.quant, &a.file).cmp(&(
            &b.id,
            b.backend.is_some(),
            &b.quant,
            &b.file,
        ))
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_link_is_extracted_and_absent_when_last_page() {
        let h = "<https://huggingface.co/api/models?cursor=abc>; rel=\"next\"";
        assert_eq!(
            parse_next_link(h).as_deref(),
            Some("https://huggingface.co/api/models?cursor=abc")
        );
        assert_eq!(parse_next_link("<https://x>; rel=\"prev\""), None);
        assert_eq!(parse_next_link(""), None);
    }

    /// A fetcher that serves bytes from an in-memory `.base` prefix, so header
    /// reading is testable without a network or a multi-GB fixture.
    struct BytesFetcher {
        files: Vec<(String, Vec<u8>, Option<String>)>,
    }

    impl Fetcher for BytesFetcher {
        fn get_file(&self, _: &str, _: &str, _: &str) -> Result<std::path::PathBuf> {
            anyhow::bail!("unused")
        }
        fn list_files(&self, _: &str, _: &str) -> Result<Vec<String>> {
            Ok(self.files.iter().map(|(n, _, _)| n.clone()).collect())
        }
        fn read_range(
            &self,
            _: &str,
            _: &str,
            filename: &str,
            range: std::ops::Range<u64>,
        ) -> Result<Vec<u8>> {
            let (_, bytes, _) = self
                .files
                .iter()
                .find(|(n, _, _)| n == filename)
                .ok_or_else(|| anyhow::anyhow!("no such file"))?;
            let start = range.start as usize;
            let end = (range.end as usize).min(bytes.len());
            anyhow::ensure!(start <= end, "bad range");
            Ok(bytes[start..end].to_vec())
        }
    }

    impl RepoIndex for BytesFetcher {
        fn files(&self, _: &str) -> Result<Vec<RemoteFile>> {
            Ok(self
                .files
                .iter()
                .map(|(n, b, sha)| RemoteFile {
                    path: n.clone(),
                    size: b.len() as u64,
                    sha256: sha.clone(),
                })
                .collect())
        }
    }

    fn base_bytes(header_json: &str) -> Vec<u8> {
        let mut v = b"BASE".to_vec();
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        v.extend_from_slice(header_json.as_bytes());
        v
    }

    #[test]
    fn a_non_base_file_is_rejected_before_any_second_read() {
        let f = BytesFetcher {
            files: vec![("junk.base".into(), b"NOTBASEnotbase!!".to_vec(), None)],
        };
        let err = read_remote_header(&f, "org/m", "main", "junk.base").unwrap_err();
        assert!(err.to_string().contains("not a .base file"), "{err}");
    }

    #[test]
    fn an_implausible_header_length_is_refused_not_allocated() {
        let mut bytes = b"BASE".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let f = BytesFetcher {
            files: vec![("huge.base".into(), bytes, None)],
        };
        let err = read_remote_header(&f, "org/m", "main", "huge.base").unwrap_err();
        assert!(
            err.to_string().contains("implausible header length"),
            "{err}"
        );
    }

    #[test]
    fn unchanged_files_reuse_their_row_and_read_no_header() {
        // The bundle's bytes are deliberately NOT a valid header: if the scan
        // tried to read one, this test would fail. Reuse must be decided by
        // the sha256 alone.
        let f = BytesFetcher {
            files: vec![(
                "m-Q4.base".into(),
                b"not a header at all".to_vec(),
                Some("abc123".into()),
            )],
        };
        let known = Catalog {
            schema: 1,
            updated: String::new(),
            models: vec![CatalogEntry {
                id: "basecompute/m".into(),
                hf_repo: "basecompute/m".into(),
                file: "m-Q4.base".into(),
                revision: "main".into(),
                source_repo: Some("Qwen/m".into()),
                arch: Some("qwen35".into()),
                quant: "default-q4".into(),
                size: Some(19),
                sha256: Some("abc123".into()),
                backend: None,
            }],
        };
        let r = scan_org(
            &f,
            &f,
            "basecompute",
            &["basecompute/m".to_string()],
            &known,
        )
        .unwrap();
        assert_eq!(r.reused, 1);
        assert_eq!(r.entries.len(), 1);
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
        // The curated field the header cannot know survives regeneration.
        assert_eq!(r.entries[0].source_repo.as_deref(), Some("Qwen/m"));
    }

    #[test]
    fn a_changed_sha_forces_a_reread_and_a_bad_header_is_reported_not_dropped() {
        let f = BytesFetcher {
            files: vec![(
                "m-Q4.base".into(),
                base_bytes("{ this is not valid json"),
                Some("newsha".into()),
            )],
        };
        let known = Catalog {
            schema: 1,
            updated: String::new(),
            models: vec![],
        };
        let r = scan_org(
            &f,
            &f,
            "basecompute",
            &["basecompute/m".to_string()],
            &known,
        )
        .unwrap();
        assert!(r.entries.is_empty());
        assert_eq!(r.skipped.len(), 1, "a broken bundle must be reported");
        assert!(r.skipped[0].0.contains("m-Q4.base"));
    }

    #[test]
    fn a_new_quant_inherits_the_repos_curated_source_repo() {
        // The rule a newly published quant depends on: no header knows which
        // upstream model it came from, so the value has to come from a sibling
        // row or it is silently lost on regeneration.
        let known = Catalog {
            schema: 1,
            updated: String::new(),
            models: vec![
                CatalogEntry {
                    id: "basecompute/m".into(),
                    hf_repo: "basecompute/m".into(),
                    file: "m-Q4.base".into(),
                    revision: "main".into(),
                    source_repo: Some("Qwen/m".into()),
                    arch: Some("qwen35".into()),
                    quant: "default-q4".into(),
                    size: Some(1),
                    sha256: Some("q4sha".into()),
                    backend: None,
                },
                CatalogEntry {
                    id: "basecompute/other".into(),
                    hf_repo: "basecompute/other".into(),
                    file: "other-Q4.base".into(),
                    revision: "main".into(),
                    source_repo: None,
                    arch: Some("llama".into()),
                    quant: "default-q4".into(),
                    size: Some(1),
                    sha256: Some("othersha".into()),
                    backend: None,
                },
            ],
        };
        assert_eq!(
            inherited_source_repo(&known, "basecompute/m").as_deref(),
            Some("Qwen/m")
        );
        // A repo whose rows never had one stays None rather than borrowing a
        // neighbour's.
        assert_eq!(inherited_source_repo(&known, "basecompute/other"), None);
        assert_eq!(inherited_source_repo(&known, "basecompute/unknown"), None);
    }

    #[test]
    fn a_repo_with_no_bundles_is_recorded_not_treated_as_failure() {
        let f = BytesFetcher {
            files: vec![("README.md".into(), b"hi".to_vec(), None)],
        };
        let known = Catalog {
            schema: 1,
            updated: String::new(),
            models: vec![],
        };
        let r = scan_org(
            &f,
            &f,
            "basecompute",
            &["basecompute/docs".to_string()],
            &known,
        )
        .unwrap();
        assert_eq!(r.empty_repos, vec!["basecompute/docs".to_string()]);
        assert!(r.entries.is_empty());
        assert!(r.skipped.is_empty());
    }
}
