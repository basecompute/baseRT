//! Generate catalog entries from `.base` files.
//!
//! The hub catalog has grown one hand-edited JSON row per (model, quant,
//! backend). That does not scale: the `size`/`sha256`/`backend` fields must
//! stay byte-exact with what clients download, and hand-maintaining them is
//! error-prone. Every `.base` header already records the `arch` and the
//! `target_backend` the bundle was packed for; combined with the file's size
//! and sha256 that is all a [`CatalogEntry`] needs except the publish
//! coordinates (`id`, `hf_repo`, quant identity). This module derives an entry
//! straight from the file so a publisher generates rows instead of editing
//! JSON — the catalog becomes a build artifact.

use crate::catalog::CatalogEntry;
use anyhow::{Context, Result};
use base_format::TargetBackend;
use std::io::Read;
use std::path::Path;

/// Map a bundle's packed `target_backend` to the catalog `backend` field the
/// resolver filters on. `None` = universal (resolves on every backend). The
/// Apple/MLX-affine packing is the portable base, so `Metal` maps to universal;
/// the CUDA / ROCm / CPU tile layouts are backend-locked and only resolve for a
/// client of that backend.
pub fn backend_tag(target: TargetBackend) -> Result<Option<String>> {
    Ok(match target {
        TargetBackend::Metal => None,
        // GB10 (sm121) serializes to `cuda_sm121`, which the CUDA runtime accepts
        // alongside a generic `cuda`; we advertise the generic tag. sm89/sm90 are
        // NOT accepted by the current reader, so refuse to generate an entry that
        // would download and then fail to load.
        TargetBackend::CudaSm121 => Some("cuda".to_string()),
        TargetBackend::CudaSm89 | TargetBackend::CudaSm90 => anyhow::bail!(
            "{target:?} is not supported for catalog publishing yet — the runtime accepts \
             cuda / cuda_sm121; rebuild the bundle with --target=cuda_sm121"
        ),
        TargetBackend::RocmCdna3 => Some("rocm".to_string()),
        TargetBackend::CpuAvx2 | TargetBackend::CpuNeon => Some("cpu".to_string()),
    })
}

/// sha256 of a file, streamed in 1 MiB chunks (bounded memory, no mmap dep).
fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Derive a catalog entry from a local `.base` file. The publisher supplies only
/// the coordinates the header cannot know — `id` and `hf_repo`. Everything the
/// resolver, cache layout, and integrity check depend on — `arch`, `backend`,
/// the quant IDENTITY, `size`, `sha256` — is read/derived from the bytes, so a
/// generated entry can never drift from what clients download.
///
/// The quant identity is `<slot>-<bits>` where `<bits>` is the header's quant
/// bit-token (`q4`/`q8`/…) and `<slot>` is `default` for a universal bundle or
/// the backend tag for a backend-locked one (`cuda-q4`). Keeping the bits as the
/// LAST segment lets `quant_tag`/`pull id:q4` match it; using a distinct slot
/// per backend gives universal and native variants distinct cache dirs and a
/// distinct catalog uniqueness key, so both can coexist for one model.
pub fn entry_from_base(path: &Path, id: &str, hf_repo: &str) -> Result<CatalogEntry> {
    let header = base_format::BaseReader::read_header(path)
        .with_context(|| format!("read .base header {}", path.display()))?;
    let backend = backend_tag(header.target_backend)?;
    let bits = crate::registry::quant_bits(&header.quant_profile)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{:?}", header.quant_scheme).to_ascii_lowercase());
    let quant = match &backend {
        Some(b) => format!("{b}-{bits}"),
        None => format!("default-{bits}"),
    };
    let size = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let sha256 = sha256_file(path)?;
    let file = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.base")
        .to_string();
    Ok(CatalogEntry {
        id: id.to_string(),
        hf_repo: hf_repo.to_string(),
        file,
        revision: "main".to_string(),
        source_repo: None,
        arch: Some(header.arch.clone()),
        quant,
        size: Some(size),
        sha256: Some(sha256),
        backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_tag_maps_target_to_catalog_field() {
        assert_eq!(backend_tag(TargetBackend::Metal).unwrap(), None); // universal base
        assert_eq!(
            backend_tag(TargetBackend::CudaSm121).unwrap().as_deref(),
            Some("cuda")
        );
        assert_eq!(
            backend_tag(TargetBackend::RocmCdna3).unwrap().as_deref(),
            Some("rocm")
        );
        assert_eq!(
            backend_tag(TargetBackend::CpuNeon).unwrap().as_deref(),
            Some("cpu")
        );
        // sm89 / sm90 aren't accepted by the runtime → refuse to generate an
        // entry that would download and then fail to load.
        assert!(backend_tag(TargetBackend::CudaSm89).is_err());
        assert!(backend_tag(TargetBackend::CudaSm90).is_err());
    }

    #[test]
    fn sha256_file_matches_known_vector() {
        // sha256("") = e3b0c442...
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("empty.base");
        std::fs::write(&p, b"").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
