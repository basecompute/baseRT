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
    /// The files of `repo` at `revision` — a commit, once the scan has
    /// pinned one, so the listing, the manifest and the header all describe
    /// one publication.
    fn files(&self, repo: &str, revision: &str) -> Result<Vec<RemoteFile>>;
}

/// The real Hub.
pub struct HubApi;

impl RepoIndex for HubApi {
    fn files(&self, repo: &str, revision: &str) -> Result<Vec<RemoteFile>> {
        // The same listing the pull uses: paginated, authenticated, timed,
        // and keeping the LFS hash. A tree read in one shot would stop at
        // the first page and make a bundle on the next one vanish from the
        // regenerated catalog.
        Ok(crate::fetch::fetch_tree(repo, revision)?
            .into_iter()
            .map(|e| RemoteFile {
                path: e.path,
                size: e.size,
                sha256: e.lfs_sha256,
            })
            .collect())
    }
}

use crate::fetch::endpoint;

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
pub(crate) fn parse_next_link(link: &str) -> Option<String> {
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
    read_header_via(
        &|range| fetcher.read_range(repo, revision, filename, range),
        filename,
    )
}

/// One logical byte range of a split bundle, read from whichever parts it
/// falls in. A header can straddle part 000 when the split is small or the
/// tokenizer large; the installer accepts any cut, so the scan has to.
pub fn read_split_range(
    fetcher: &dyn Fetcher,
    repo: &str,
    revision: &str,
    parts: &[(String, u64)],
    range: std::ops::Range<u64>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((range.end - range.start) as usize);
    let mut start = 0u64;
    for (path, size) in parts {
        let end = start.checked_add(*size).context("part sizes overflow")?;
        let lo = range.start.max(start);
        let hi = range.end.min(end);
        if lo < hi {
            out.extend(fetcher.read_range(repo, revision, path, lo - start..hi - start)?);
        }
        start = end;
        if start >= range.end {
            break;
        }
    }
    Ok(out)
}

/// The header of a `.base`, given a way to read byte ranges of it.
fn read_header_via(
    read: &dyn Fn(std::ops::Range<u64>) -> Result<Vec<u8>>,
    filename: &str,
) -> Result<Header> {
    let prefix = read(0..PREFIX_LEN)?;
    anyhow::ensure!(
        prefix.len() == PREFIX_LEN as usize,
        "{filename}: short prefix ({} bytes)",
        prefix.len()
    );
    anyhow::ensure!(&prefix[0..4] == b"BASE", "{filename}: not a .base file");
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    anyhow::ensure!(
        version == base_format::FORMAT_VERSION,
        "{filename}: unsupported .base format version {version}"
    );
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    // A header claiming to be enormous is corrupt or hostile; refuse rather
    // than allocate it. 256MB is far past any real tokenizer.
    anyhow::ensure!(
        header_len > 0 && header_len < 256 * 1024 * 1024,
        "{filename}: implausible header length {header_len}"
    );
    let json = read(PREFIX_LEN..PREFIX_LEN + header_len)?;
    anyhow::ensure!(
        json.len() as u64 == header_len,
        "{filename}: header truncated ({} of {header_len} bytes)",
        json.len()
    );
    Header::from_json_bytes(&json).with_context(|| format!("parsing {filename}'s header"))
}

/// One bundle as the scan sees it: the logical file, where its header can
/// be read from, and how many parts it is split into (0 for a whole file).
struct Bundle {
    file: RemoteFile,
    /// Where the header's bytes live: the file itself, or the parts in
    /// order with their sizes.
    header_from: Vec<(String, u64)>,
    parts: usize,
    /// The Hub's sha256 for each part, in order, `None` where it reports
    /// none (an inline part, a mirror without LFS metadata); empty for a
    /// whole file. Every value that is there gets compared.
    part_ids: Vec<Option<String>>,
    /// The part paths, in order; empty for a whole file.
    part_paths: Vec<String>,
    /// The Hub's size for each part, in order; empty for a whole file.
    part_sizes: Vec<u64>,
    /// The manifest published beside the parts, when there is one.
    manifest: Option<RemoteFile>,
}

/// Fold a repo listing into bundles. A part set becomes one logical file
/// whose size is the sum of its parts, with no sha256 of its own yet (the
/// manifest supplies that later) and its header behind part 000.
fn group_bundles(files: Vec<RemoteFile>) -> (Vec<Bundle>, Vec<(String, String)>) {
    let by_path: std::collections::HashMap<String, RemoteFile> =
        files.into_iter().map(|f| (f.path.clone(), f)).collect();
    let grouped = crate::parts::group(by_path.keys().cloned());
    let mut malformed = grouped.malformed;
    let mut out = Vec::with_capacity(grouped.artifacts.len());
    for a in grouped.artifacts {
        if a.is_split() {
            let manifest = by_path.get(&crate::parts::manifest_name(&a.name)).cloned();
            let files: Vec<&RemoteFile> = a.parts.iter().filter_map(|p| by_path.get(p)).collect();
            let part_sizes: Vec<u64> = files.iter().map(|f| f.size).collect();
            // A listing is untrusted input: sizes that do not add up are a
            // malformed set, not a panic that ends the scan.
            let Some(size) = part_sizes
                .iter()
                .try_fold(0u64, |acc, s| acc.checked_add(*s))
            else {
                malformed.push((a.name, "listed part sizes overflow".to_string()));
                continue;
            };
            let part_ids: Vec<Option<String>> = files.iter().map(|f| f.sha256.clone()).collect();
            out.push(Bundle {
                file: RemoteFile {
                    path: a.name,
                    size,
                    sha256: None,
                },
                header_from: a
                    .parts
                    .iter()
                    .cloned()
                    .zip(part_sizes.iter().copied())
                    .collect(),
                parts: a.parts.len(),
                part_ids,
                part_paths: a.parts,
                part_sizes,
                manifest,
            });
        } else if let Some(f) = by_path.get(&a.name) {
            out.push(Bundle {
                header_from: vec![(f.path.clone(), f.size)],
                file: f.clone(),
                parts: 0,
                part_ids: Vec::new(),
                part_paths: Vec::new(),
                part_sizes: Vec::new(),
                manifest: None,
            });
        }
    }
    // Listings are unordered maps here; keep the report deterministic.
    out.sort_by(|x, y| x.file.path.cmp(&y.file.path));
    (out, malformed)
}

/// Fetch and parse a split bundle's manifest with one ranged read.
fn read_remote_manifest(
    fetcher: &dyn Fetcher,
    repo: &str,
    revision: &str,
    file: &RemoteFile,
) -> Result<crate::parts::Manifest> {
    anyhow::ensure!(
        file.size > 0 && file.size < crate::parts::MAX_MANIFEST_LEN,
        "{}: implausible manifest size {}",
        file.path,
        file.size
    );
    let bytes = fetcher.read_range(repo, revision, &file.path, 0..file.size)?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", file.path))
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
        // One commit for everything read about this repo, so a publish
        // landing mid-scan cannot pair one publication's manifest with
        // another's header.
        let pinned = match fetcher.resolve_revision(repo, "main") {
            Ok(p) => p,
            Err(e) => {
                report
                    .skipped
                    .push((repo.clone(), format!("resolving main failed: {e:#}")));
                continue;
            }
        };
        let files = match index.files(repo, &pinned) {
            Ok(f) => f,
            Err(e) => {
                report
                    .skipped
                    .push((repo.clone(), format!("listing failed: {e}")));
                continue;
            }
        };
        let (bundles, malformed) = group_bundles(files);
        // A part set with a gap (a quant mid-upload, say) is reported on
        // its own; the repo's other bundles are scanned as usual.
        for (name, why) in malformed {
            report.skipped.push((format!("{repo}/{name}"), why));
        }
        if bundles.is_empty() {
            report.empty_repos.push(repo.clone());
            continue;
        }
        for Bundle {
            file: mut f,
            header_from,
            parts,
            part_ids,
            part_paths,
            part_sizes,
            manifest,
        } in bundles
        {
            // A split bundle has a sha256 per part on the Hub and none for
            // the whole. Its manifest supplies the whole-file hash and size,
            // and is believed only when the Hub's per-part hashes are the
            // ones it lists — a republished part with a stale manifest
            // would otherwise pin a hash no download can match.
            let mut parts_sha256 = None;
            if parts > 0 {
                let Some(m) = manifest else {
                    report.skipped.push((
                        format!("{repo}/{}", f.path),
                        format!(
                            "split into {parts} parts with no {} beside them; publish one with `basert catalog-manifest`",
                            crate::parts::manifest_name(&f.path)
                        ),
                    ));
                    continue;
                };
                let manifest = match read_remote_manifest(fetcher, repo, &pinned, &m) {
                    Ok(m) => m,
                    Err(e) => {
                        report
                            .skipped
                            .push((format!("{repo}/{}", f.path), format!("{e:#}")));
                        continue;
                    }
                };
                // The same check a pull runs, so a row is never written for
                // a manifest every pull would then refuse.
                if let Err(e) = manifest
                    .check_against(&part_paths)
                    .and_then(|()| manifest.check_sizes(&part_sizes))
                {
                    report.skipped.push((
                        format!("{repo}/{}", f.path),
                        format!("its manifest disagrees with the listing: {e:#}; republish the manifest"),
                    ));
                    continue;
                }
                let ids = manifest.part_ids();
                // Every hash the Hub does have must agree with the manifest;
                // hex case is not content, so the compare ignores it, as the
                // install's does.
                let same_ids = ids.len() == part_ids.len()
                    && ids
                        .iter()
                        .zip(&part_ids)
                        .all(|(a, b)| b.as_deref().is_none_or(|b| a.eq_ignore_ascii_case(b)));
                let hub_ids: Vec<&str> = part_ids
                    .iter()
                    .map(|p| p.as_deref().unwrap_or("?"))
                    .collect();
                if manifest.parts.len() != parts || !same_ids {
                    report.skipped.push((
                        format!("{repo}/{}", f.path),
                        format!(
                            "its manifest describes {} parts [{}] but the Hub holds {parts} [{}]: republish the manifest",
                            manifest.parts.len(),
                            ids.join(", "),
                            hub_ids.join(", ")
                        ),
                    ));
                    continue;
                }
                f.sha256 = Some(manifest.sha256.clone());
                f.size = manifest.size;
                parts_sha256 = Some(ids);
            }
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
                // The same bytes may have been re-split at other boundaries
                // since the row was written, which changes the parts without
                // changing the whole: the split metadata is today's.
                let mut row = prev.clone();
                if parts_sha256.is_some() {
                    row.parts_sha256 = parts_sha256;
                    row.size = Some(f.size);
                }
                report.entries.push(row);
                report.reused += 1;
                continue;
            }
            let header = match read_header_via(
                &|range| read_split_range(fetcher, repo, &pinned, &header_from, range),
                &f.path,
            ) {
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
                    entry.parts_sha256 = parts_sha256;
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
        fn files(&self, _: &str, _: &str) -> Result<Vec<RemoteFile>> {
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
                parts_sha256: None,
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
                    parts_sha256: None,
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
                    parts_sha256: None,
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
    fn a_split_bundle_is_catalogued_from_its_manifest() {
        use crate::parts::{synthetic_bundle, Manifest, ManifestPart};
        // Fake but well-formed hashes: the manifest check wants 64 hex.
        fn h(tag: &str) -> String {
            format!(
                "{:0>64}",
                tag.bytes().map(|b| format!("{b:02x}")).collect::<String>()
            )
        }
        // Part 000 carries a real header; the scan reads it from there.
        let whole = synthetic_bundle(1000);
        let (p0, p1) = whole.split_at(whole.len() / 2);
        let manifest = Manifest {
            size: whole.len() as u64,
            sha256: h("whole-sha"),
            parts: vec![
                ManifestPart {
                    name: "g.base.part-000".into(),
                    size: p0.len() as u64,
                    sha256: h("p0"),
                },
                ManifestPart {
                    name: "g.base.part-001".into(),
                    size: p1.len() as u64,
                    sha256: h("p1"),
                },
            ],
        };
        let repo = |manifest: Option<&Manifest>| {
            let mut files = vec![
                ("parts/g.base.part-000".into(), p0.to_vec(), Some(h("p0"))),
                ("parts/g.base.part-001".into(), p1.to_vec(), Some(h("p1"))),
            ];
            if let Some(m) = manifest {
                files.push((
                    "parts/g.base.manifest.json".into(),
                    serde_json::to_vec(m).unwrap(),
                    None,
                ));
            }
            BytesFetcher { files }
        };
        let none = Catalog {
            schema: 1,
            updated: String::new(),
            models: vec![],
        };
        let scan = |f: &BytesFetcher, known: &Catalog| {
            scan_org(f, f, "basecompute", &["basecompute/g".to_string()], known).unwrap()
        };

        // No manifest: the repo is not "empty", and the skip says what to
        // publish.
        let r = scan(&repo(None), &none);
        assert!(r.empty_repos.is_empty());
        assert!(r.entries.is_empty());
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert_eq!(r.skipped[0].0, "basecompute/g/parts/g.base");
        assert!(
            r.skipped[0].1.contains("catalog-manifest"),
            "{}",
            r.skipped[0].1
        );

        // With one: a row, whole-file hash and size from the manifest, the
        // per-part hashes pinned on it, header facts from part 000.
        let r = scan(&repo(Some(&manifest)), &none);
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
        assert_eq!(r.entries.len(), 1);
        let e = &r.entries[0];
        assert_eq!(e.file, "parts/g.base");
        assert_eq!(e.sha256, Some(h("whole-sha")));
        assert_eq!(e.size, Some(whole.len() as u64));
        assert_eq!(e.parts_sha256, Some(vec![h("p0"), h("p1")]));
        assert_eq!(e.arch.as_deref(), Some("test"));

        // A manifest with the right hashes but a wrong part name or size
        // would be accepted here and refused by every pull: it is refused
        // here instead.
        let mut misnamed = manifest.clone();
        misnamed.parts[1].name = "g.base.part-01".into();
        let r = scan(&repo(Some(&misnamed)), &none);
        assert!(r.entries.is_empty());
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert!(
            r.skipped[0].1.contains("disagrees with the listing"),
            "{}",
            r.skipped[0].1
        );
        let mut wrong_size = manifest.clone();
        wrong_size.parts[1].size += 1;
        let r = scan(&repo(Some(&wrong_size)), &none);
        assert!(r.entries.is_empty(), "{:?}", r.entries);
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        // Two wrong sizes that cancel in the sum are still wrong.
        let mut offset = manifest.clone();
        offset.parts[0].size += 1;
        offset.parts[1].size -= 1;
        let r = scan(&repo(Some(&offset)), &none);
        assert!(r.entries.is_empty(), "{:?}", r.entries);
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert!(
            r.skipped[0].1.contains("bytes but the repo holds"),
            "{}",
            r.skipped[0].1
        );

        // A part the Hub has no hash for does not blind the scan to the
        // others: part 1's Hub hash still has to match.
        let mut half_known = repo(Some(&manifest));
        half_known.files[0].2 = None;
        half_known.files[1].2 = Some(h("p1-hub"));
        let r = scan(&half_known, &none);
        assert!(r.entries.is_empty(), "{:?}", r.entries);
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert!(
            r.skipped[0].1.contains("republish the manifest"),
            "{}",
            r.skipped[0].1
        );
        // And with only part 0 unknown and part 1 agreeing, it is fine.
        let mut half_known = repo(Some(&manifest));
        half_known.files[0].2 = None;
        let r = scan(&half_known, &none);
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);

        // A parseable header of another format version is refused here,
        // not after a pull has reassembled the whole bundle.
        let mut v2 = repo(Some(&manifest));
        v2.files[0].1[4..8].copy_from_slice(&2u32.to_le_bytes());
        let r = scan(&v2, &none);
        assert!(r.entries.is_empty());
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert!(
            r.skipped[0]
                .1
                .contains("unsupported .base format version 2"),
            "{}",
            r.skipped[0].1
        );

        // A cut inside the prefix: the header is read across the parts.
        let (t0, t1) = whole.split_at(10);
        let tiny = Manifest {
            parts: vec![
                ManifestPart {
                    name: "g.base.part-000".into(),
                    size: t0.len() as u64,
                    sha256: h("t0"),
                },
                ManifestPart {
                    name: "g.base.part-001".into(),
                    size: t1.len() as u64,
                    sha256: h("t1"),
                },
            ],
            ..manifest.clone()
        };
        let straddling = BytesFetcher {
            files: vec![
                ("parts/g.base.part-000".into(), t0.to_vec(), Some(h("t0"))),
                ("parts/g.base.part-001".into(), t1.to_vec(), Some(h("t1"))),
                (
                    "parts/g.base.manifest.json".into(),
                    serde_json::to_vec(&tiny).unwrap(),
                    None,
                ),
            ],
        };
        let r = scan(&straddling, &none);
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
        assert_eq!(r.entries.len(), 1);
        assert_eq!(r.entries[0].arch.as_deref(), Some("test"));

        // Uppercase hex in the manifest is the same digest.
        let mut upper = manifest.clone();
        for p in &mut upper.parts {
            p.sha256 = p.sha256.to_ascii_uppercase();
        }
        let r = scan(&repo(Some(&upper)), &none);
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
        assert_eq!(r.entries.len(), 1);

        // A manifest the Hub disagrees with (part 1 republished): refused.
        let mut stale = manifest.clone();
        stale.parts[1].sha256 = "p1-old".into();
        let r = scan(&repo(Some(&stale)), &none);
        assert!(r.entries.is_empty());
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert!(
            r.skipped[0].1.contains("republish the manifest"),
            "{}",
            r.skipped[0].1
        );

        // An unchanged bundle reuses its row without a header read.
        let known = Catalog {
            schema: 1,
            updated: String::new(),
            models: scan(&repo(Some(&manifest)), &none).entries,
        };
        let r = scan(&repo(Some(&manifest)), &known);
        assert_eq!(r.reused, 1);
        assert_eq!(r.entries.len(), 1);

        // The same bytes re-split at other boundaries: the whole-file hash
        // is unchanged, so the row is reused, but its part hashes are
        // today's — a pull checks the manifest against the row.
        let (q0, q1) = whole.split_at(whole.len() / 3);
        let resplit = Manifest {
            parts: vec![
                ManifestPart {
                    name: "g.base.part-000".into(),
                    size: q0.len() as u64,
                    sha256: h("q0"),
                },
                ManifestPart {
                    name: "g.base.part-001".into(),
                    size: q1.len() as u64,
                    sha256: h("q1"),
                },
            ],
            ..manifest.clone()
        };
        let f = BytesFetcher {
            files: vec![
                ("parts/g.base.part-000".into(), q0.to_vec(), Some(h("q0"))),
                ("parts/g.base.part-001".into(), q1.to_vec(), Some(h("q1"))),
                (
                    "parts/g.base.manifest.json".into(),
                    serde_json::to_vec(&resplit).unwrap(),
                    None,
                ),
            ],
        };
        let r = scan(&f, &known);
        assert_eq!(r.reused, 1);
        assert_eq!(r.entries[0].parts_sha256, Some(vec![h("q0"), h("q1")]));
    }

    #[test]
    fn listed_part_sizes_that_overflow_are_a_skip_not_a_panic() {
        let files = vec![
            RemoteFile {
                path: "g.base.part-000".into(),
                size: u64::MAX,
                sha256: Some("p0".into()),
            },
            RemoteFile {
                path: "g.base.part-001".into(),
                size: 1,
                sha256: Some("p1".into()),
            },
            RemoteFile {
                path: "m-Q4.base".into(),
                size: 3,
                sha256: Some("m".into()),
            },
        ];
        let (bundles, malformed) = group_bundles(files);
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].file.path, "m-Q4.base");
        assert_eq!(malformed.len(), 1);
        assert_eq!(malformed[0].0, "g.base");
        assert!(malformed[0].1.contains("overflow"), "{}", malformed[0].1);
    }

    #[test]
    fn a_gapped_part_set_is_skipped_without_dropping_its_siblings() {
        let f = BytesFetcher {
            files: vec![
                (
                    "m-Q4.base".into(),
                    b"not a header at all".to_vec(),
                    Some("abc123".into()),
                ),
                (
                    "m-Q8.base.part-000".into(),
                    b"x".to_vec(),
                    Some("p0".into()),
                ),
                (
                    "m-Q8.base.part-002".into(),
                    b"x".to_vec(),
                    Some("p2".into()),
                ),
            ],
        };
        let known = Catalog {
            schema: 1,
            updated: String::new(),
            models: vec![CatalogEntry {
                id: "basecompute/m".into(),
                hf_repo: "basecompute/m".into(),
                file: "m-Q4.base".into(),
                revision: "main".into(),
                source_repo: None,
                arch: Some("qwen35".into()),
                quant: "default-q4".into(),
                size: Some(19),
                sha256: Some("abc123".into()),
                parts_sha256: None,
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
        // The good row survives; the half-uploaded quant is named.
        assert_eq!(r.entries.len(), 1);
        assert_eq!(r.entries[0].file, "m-Q4.base");
        assert_eq!(r.skipped.len(), 1, "{:?}", r.skipped);
        assert_eq!(r.skipped[0].0, "basecompute/m/m-Q8.base");
        assert!(
            r.skipped[0].1.contains("missing part 001"),
            "{}",
            r.skipped[0].1
        );
        assert!(r.empty_repos.is_empty());
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
