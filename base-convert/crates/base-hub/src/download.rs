//! Resumable, multi-connection download of a single Hub file.
//!
//! This exists because neither transport hf-hub offers is safe for the files
//! we pull. A `.base` bundle or a safetensors shard is tens of gigabytes over
//! a link that will not stay up for the whole transfer, and hf-hub 1.0's plain
//! HTTPS path issues one un-ranged `GET` into `File::create`: its retry only
//! wraps the *initial* request, so a peer that disconnects mid-body fails the
//! pull outright, and the next attempt restarts from byte zero. That is the
//! "download stopped halfway" failure.
//!
//! Xet-backed files do not come through here — hf-hub hands those to hf-xet,
//! which already transfers chunk-parallel and resumes from its local chunk
//! cache. This module is the fallback for everything the Hub still serves as
//! plain LFS/HTTPS blobs, and it gives that path the same two properties:
//!
//! * **Parallel.** The file is cut into fixed chunks fetched concurrently over
//!   `Range` requests, one connection per worker. A single TCP stream tops out
//!   well below the link on a long fat pipe to a CDN edge.
//! * **Resumable.** Chunk completion is recorded in a sidecar bitmap next to
//!   the partial file, fsynced as each chunk lands. A pull killed at 90% —
//!   crash, `^C`, or a dead link — resumes at 90%, across process restarts.
//!
//! Reads also carry a timeout (`$BASERT_HF_READ_TIMEOUT_SECS`), so a
//! connection that goes quiet without closing is failed and retried rather
//! than hanging the pull forever.
//!
//! Not locked against a second process pulling the same file at the same
//! time. Both would write identical bytes at identical offsets, so the result
//! is not corrupt, but the loser's final rename can fail once the winner has
//! moved the partial away. hf-hub 0.5 did not lock here either, so this is
//! not a regression; it is worth a lock file if concurrent pulls ever become
//! a normal thing to do.

use anyhow::{bail, Context, Result};
use hf_hub::{HFClientSync, RepoTypeModel};
use indicatif::ProgressBar;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Concurrent range requests.
///
/// Measured against the 19.8GB Qwen3.5-35B-A3B bundle, three rounds on the
/// same link (MiB/s, 15s warmup + 45s window, cold cache each run):
///
///   connections x chunk    r1     r2     r3     mean
///   8 x 32MB (old)        39.8     -      -
///   16 x 16MB             49.4   53.3     -
///   24 x 16MB             54.4   81.7   83.5    73.2
///   32 x 8MB              58.3   77.1   77.8    71.1
///
/// 24 and 32 are indistinguishable; 24 is chosen for the smaller connection
/// count at equal throughput. Eight — the previous default — is measurably
/// short of the link.
const DEFAULT_CONNECTIONS: usize = 24;
/// Bytes per range request. Large enough that per-request overhead vanishes
/// against a multi-GB file, small enough that a failed chunk is cheap to redo.
///
/// Peak memory is `connections * chunk` — 384MB at the defaults, since each
/// worker holds one chunk in flight. Lower either knob on a constrained box;
/// 16MB earned its place over 8MB in round 3 (83.5 vs 62.4 at 24 connections).
const DEFAULT_CHUNK_MB: u64 = 16;
/// Seconds a read may stall before the chunk is failed and retried.
const DEFAULT_READ_TIMEOUT_SECS: u64 = 60;

/// Read a positive `usize` knob from the environment.
fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Read a positive `u64` knob from the environment.
fn env_u64(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// How many range requests to run concurrently (`$BASERT_HF_CONNECTIONS`).
pub fn connections() -> usize {
    env_usize("BASERT_HF_CONNECTIONS", DEFAULT_CONNECTIONS)
}

/// Bytes per range request (`$BASERT_HF_CHUNK_MB`).
pub fn chunk_size() -> u64 {
    env_u64("BASERT_HF_CHUNK_MB", DEFAULT_CHUNK_MB) * 1024 * 1024
}

/// Per-read stall timeout (`$BASERT_HF_READ_TIMEOUT_SECS`).
pub fn read_timeout() -> Duration {
    Duration::from_secs(env_u64(
        "BASERT_HF_READ_TIMEOUT_SECS",
        DEFAULT_READ_TIMEOUT_SECS,
    ))
}

/// Records which chunks of a partial download are already on disk.
///
/// One byte per chunk in a sidecar file, synced as each chunk completes. A
/// byte-per-chunk (rather than a packed bitmap) keeps the update a single
/// positioned write with no read-modify-write race between workers.
struct ChunkMap {
    file: File,
    path: PathBuf,
    done: Vec<bool>,
}

impl ChunkMap {
    /// Open (or create) the sidecar for `chunks` chunks. A sidecar whose
    /// length disagrees with the chunk count is from a different plan — chunk
    /// size changed, or the file did — and is discarded rather than trusted.
    fn open(path: PathBuf, chunks: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening chunk map {}", path.display()))?;
        let len = file.metadata()?.len() as usize;
        let done = if len == chunks {
            let mut buf = vec![0u8; chunks];
            file.read_exact_at(&mut buf, 0)?;
            buf.into_iter().map(|b| b == 1).collect()
        } else {
            file.set_len(chunks as u64)?;
            file.write_all_at(&vec![0u8; chunks], 0)?;
            vec![false; chunks]
        };
        Ok(Self { file, path, done })
    }

    fn mark(&self, index: usize) -> Result<()> {
        self.file.write_all_at(&[1u8], index as u64)?;
        // Durability matters more than speed here: the whole point is that a
        // kill -9 mid-pull does not cost the bytes already on disk.
        self.file.sync_data()?;
        Ok(())
    }

    fn remove(self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Byte range covered by chunk `index`.
fn chunk_range(index: usize, chunk: u64, size: u64) -> std::ops::Range<u64> {
    let start = index as u64 * chunk;
    let end = (start + chunk).min(size);
    start..end
}

/// Download `filename` at `revision` into `dst`, in parallel and resumably.
///
/// `size` is the expected length from the Hub's file metadata; the result is
/// verified against it, so a short transfer is an error rather than a
/// truncated model that fails much later at load time with a confusing
/// message. `mint_client` produces one client per worker: `HFClientSync`
/// parks a single-threaded runtime on one background thread, so sharing a
/// client across workers would funnel every range request through one
/// reactor.
pub fn download_ranged(
    mint_client: &(dyn Fn() -> Result<HFClientSync> + Sync),
    repo: &str,
    revision: &str,
    filename: &str,
    size: u64,
    dst: &Path,
    max_retries: usize,
) -> Result<()> {
    let chunk = chunk_size();
    let chunks = size.div_ceil(chunk) as usize;
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let partial = PathBuf::from(format!("{}.partial", dst.display()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&partial)
        .with_context(|| format!("opening {}", partial.display()))?;
    // Preallocate so workers can write at their own offsets in any order.
    if file.metadata()?.len() != size {
        file.set_len(size)?;
    }

    let map = ChunkMap::open(PathBuf::from(format!("{}.chunks", dst.display())), chunks)?;
    let pending: Vec<usize> = (0..chunks).filter(|i| !map.done[*i]).collect();
    let resumed: u64 = (0..chunks)
        .filter(|i| map.done[*i])
        .map(|i| {
            let r = chunk_range(i, chunk, size);
            r.end - r.start
        })
        .sum();

    let bar = ProgressBar::new(size);
    bar.set_style(crate::fetch::bar_style());
    bar.set_message(filename.to_string());
    bar.set_position(resumed);
    if resumed > 0 {
        eprintln!(
            "{filename}: resuming at {:.1}% ({}/{} chunks already on disk)",
            resumed as f64 / size as f64 * 100.0,
            chunks - pending.len(),
            chunks
        );
    }

    let workers = connections().min(pending.len().max(1));
    let queue = Mutex::new(pending);
    let completed = AtomicU64::new(resumed);
    let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                // One client — and so one connection pool and one runtime —
                // per worker. Built inside the worker so a client that fails
                // to build only fails its own share of the chunks.
                let client = match mint_client() {
                    Ok(c) => c,
                    Err(e) => {
                        *failure.lock().unwrap() = Some(e);
                        return;
                    },
                };
                let (owner, name) = hf_hub::split_id(repo);
                let handle: hf_hub::HFRepositorySync<RepoTypeModel> = client.model(owner, name);
                loop {
                    if failure.lock().unwrap().is_some() {
                        return;
                    }
                    let Some(index) = queue.lock().unwrap().pop() else {
                        return;
                    };
                    let range = chunk_range(index, chunk, size);
                    match fetch_chunk(&handle, revision, filename, range.clone(), max_retries) {
                        Ok(bytes) => {
                            let expected = (range.end - range.start) as usize;
                            if bytes.len() != expected {
                                *failure.lock().unwrap() = Some(anyhow::anyhow!(
                                    "{filename}: range {}..{} returned {} bytes, expected {expected}",
                                    range.start,
                                    range.end,
                                    bytes.len()
                                ));
                                return;
                            }
                            if let Err(e) = file
                                .write_all_at(&bytes, range.start)
                                .with_context(|| format!("writing {filename} at {}", range.start))
                                .and_then(|()| map.mark(index))
                            {
                                *failure.lock().unwrap() = Some(e);
                                return;
                            }
                            let total = completed.fetch_add(expected as u64, Ordering::Relaxed)
                                + expected as u64;
                            bar.set_position(total);
                        },
                        Err(e) => {
                            *failure.lock().unwrap() = Some(e);
                            return;
                        },
                    }
                }
            });
        }
    });

    if let Some(e) = failure.into_inner().unwrap() {
        // The partial file and its chunk map are deliberately left behind:
        // they are what makes the next attempt resume instead of restart.
        bar.abandon();
        return Err(e);
    }

    file.sync_all()?;
    let written = file.metadata()?.len();
    if written != size {
        bail!("{filename}: downloaded {written} bytes, expected {size}");
    }
    drop(file);
    map.remove();
    std::fs::rename(&partial, dst).with_context(|| format!("finalizing {}", dst.display()))?;
    bar.finish_and_clear();
    Ok(())
}

/// Fetch one range, retrying transient failures with exponential backoff.
fn fetch_chunk(
    handle: &hf_hub::HFRepositorySync<RepoTypeModel>,
    revision: &str,
    filename: &str,
    range: std::ops::Range<u64>,
    max_retries: usize,
) -> Result<Vec<u8>> {
    let mut attempt = 0;
    loop {
        let result = handle
            .download_file_to_bytes()
            .filename(filename)
            .revision(revision)
            .range(range.clone())
            .send();
        match result {
            Ok(bytes) => return Ok(bytes.to_vec()),
            Err(e) => {
                if attempt >= max_retries {
                    return Err(anyhow::Error::new(e).context(format!(
                        "{filename}: range {}..{} failed after {} attempts",
                        range.start,
                        range.end,
                        attempt + 1
                    )));
                }
                // 100ms, 200ms, 400ms, … capped: a CDN edge that just dropped
                // us needs a moment, not a tight loop.
                let backoff = Duration::from_millis(100u64 << attempt.min(6));
                std::thread::sleep(backoff);
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ranges_tile_the_file_exactly() {
        let size = 100u64;
        let chunk = 30u64;
        let chunks = size.div_ceil(chunk) as usize;
        assert_eq!(chunks, 4);
        let ranges: Vec<_> = (0..chunks).map(|i| chunk_range(i, chunk, size)).collect();
        assert_eq!(ranges[0], 0..30);
        assert_eq!(ranges[3], 90..100, "last chunk must be clamped to the size");
        // No gaps, no overlap, full coverage.
        let mut next = 0;
        for r in &ranges {
            assert_eq!(r.start, next);
            next = r.end;
        }
        assert_eq!(next, size);
    }

    #[test]
    fn chunk_map_persists_completion_across_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("blob.chunks");
        let map = ChunkMap::open(path.clone(), 4).unwrap();
        assert_eq!(map.done, vec![false; 4]);
        map.mark(1).unwrap();
        map.mark(2).unwrap();
        drop(map);

        // Reopening is what a resumed pull does: the two finished chunks must
        // not be fetched again.
        let reopened = ChunkMap::open(path.clone(), 4).unwrap();
        assert_eq!(reopened.done, vec![false, true, true, false]);
        reopened.remove();
        assert!(!path.exists());
    }

    #[test]
    fn chunk_map_from_a_different_plan_is_discarded() {
        // A sidecar sized for another chunk plan must not be reinterpreted:
        // its bytes mean different ranges, and trusting them would leave holes
        // in the file.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("blob.chunks");
        let map = ChunkMap::open(path.clone(), 8).unwrap();
        map.mark(7).unwrap();
        drop(map);

        let reopened = ChunkMap::open(path, 4).unwrap();
        assert_eq!(reopened.done, vec![false; 4], "stale plan must reset");
    }
}
