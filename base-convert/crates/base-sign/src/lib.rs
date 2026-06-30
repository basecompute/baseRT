//! ed25519 signing + verification for `.base` manifests.
//!
//! The signed payload is:
//!
//! ```text
//!   canonical_json(header with "sig" field removed) || sha256(weights_blob)
//! ```
//!
//! The signature is placed back into the header's `sig` field (alg,
//! key_id, base64-encoded signature bytes). This shape keeps the
//! signature reproducible regardless of how the JSON is re-serialized
//! at verify time, as long as the canonicalizer is deterministic.

use anyhow::{Context, Result};
use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey, SECRET_KEY_LENGTH,
};
use sha2::{Digest, Sha256};

/// Sign a payload (canonical JSON || sha256(blob)) with an ed25519 key.
pub fn sign_payload(key: &SigningKey, payload: &[u8]) -> Signature {
    key.sign(payload)
}

/// Verify an ed25519 signature. Returns Ok(()) when valid, else Err.
pub fn verify_payload(key: &VerifyingKey, payload: &[u8], sig: &Signature) -> Result<()> {
    key.verify(payload, sig)
        .context("ed25519 signature verification failed")
}

/// Compute `sha256(weights_blob)`. Callers pass the raw blob bytes as
/// seen in the `.base` file (post-padding, pre-slots).
pub fn blob_sha256(blob: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(blob);
    let d = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

/// Build the signed payload: `canonical_header_json || blob_sha256`.
/// The caller is expected to have removed the `sig` field from the
/// header before canonicalizing.
pub fn build_payload(canonical_header_json: &[u8], blob_bytes: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(canonical_header_json.len() + 32);
    payload.extend_from_slice(canonical_header_json);
    payload.extend_from_slice(&blob_sha256(blob_bytes));
    payload
}

/// Load a SigningKey from the raw 32-byte secret key bytes. Callers
/// typically read these from a key file produced by
/// `ed25519-dalek`'s `SigningKey::to_bytes`.
pub fn signing_key_from_bytes(bytes: &[u8]) -> Result<SigningKey> {
    if bytes.len() != SECRET_KEY_LENGTH {
        anyhow::bail!(
            "ed25519 secret key must be {} bytes, got {}",
            SECRET_KEY_LENGTH,
            bytes.len()
        );
    }
    let mut arr = [0u8; SECRET_KEY_LENGTH];
    arr.copy_from_slice(bytes);
    Ok(SigningKey::from_bytes(&arr))
}

/// Base64 (standard, no padding stripped) helpers so the `sig.signature`
/// field in the JSON header is ASCII-safe.
pub fn b64_encode(bytes: &[u8]) -> String {
    // Minimal base64 implementation to avoid pulling in a crate just
    // for this. Standard alphabet, with `=` padding.
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        let b2 = bytes[i + 2];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[((b0 & 0b11) << 4 | b1 >> 4) as usize] as char);
        out.push(ALPHABET[((b1 & 0b1111) << 2 | b2 >> 6) as usize] as char);
        out.push(ALPHABET[(b2 & 0b111111) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let b0 = bytes[i];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[((b0 & 0b11) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[((b0 & 0b11) << 4 | b1 >> 4) as usize] as char);
        out.push(ALPHABET[((b1 & 0b1111) << 2) as usize] as char);
        out.push('=');
    }
    out
}

pub fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u8;
    for c in s.chars() {
        let v: u32 = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => anyhow::bail!("invalid base64 character: {:?}", c),
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    Ok(out)
}

/// Re-sign an existing `.base` file, producing a new signed file at
/// `output`. Reads the input file, recomputes the canonical JSON header
/// without a sig, computes blob sha256, signs, and writes a new file
/// with the sig field populated.
///
/// This is intentionally a post-process step: keeps `BaseWriter`
/// streaming-friendly (no memory-buffered blob) while centralizing
/// signing logic here.
pub fn sign_base_file<P: AsRef<std::path::Path>>(
    input: P,
    output: P,
    key: &SigningKey,
    key_id: &str,
) -> Result<()> {
    use base_format::{BaseReader, Header, Signature as BaseSig};
    use std::io::Write;

    // 1. Read input and extract blob bytes + header.
    let reader = BaseReader::open(input.as_ref())
        .with_context(|| format!("opening {:?}", input.as_ref()))?;

    // Compute end-of-blob as max(offset + length) over all tensors.
    let blob_end_rel = reader
        .header()
        .tensors
        .iter()
        .map(|t| t.offset + t.length)
        .max()
        .unwrap_or(0);
    let blob_start = reader.blob_offset();
    let blob_bytes: Vec<u8> = {
        let file_bytes = std::fs::read(input.as_ref())?;
        file_bytes[blob_start as usize..(blob_start + blob_end_rel) as usize].to_vec()
    };

    // 2. Build canonical JSON of header WITHOUT sig, compute payload,
    // and sign.
    let mut header: Header = reader.header().without_sig();
    let canonical = serde_json::to_vec(&header)?;
    let payload = build_payload(&canonical, &blob_bytes);
    let sig = sign_payload(key, &payload);
    let sig_b64 = b64_encode(&sig.to_bytes());

    header.sig = Some(BaseSig {
        alg: "ed25519".to_string(),
        key_id: key_id.to_string(),
        signature: sig_b64,
    });

    // 3. Re-serialize header with sig field and write the output file.
    //
    // Tensor offsets in the header are relative to blob_start, so they
    // stay valid as long as we recompute blob_start for the new header
    // length. We write: prefix + new_header + pad to 64 KiB + blob +
    // (whatever follows blob in the original, e.g. extension slots).
    let new_header = serde_json::to_vec(&header)?;
    let new_header_len = new_header.len() as u64;

    const MAGIC: [u8; 4] = *b"BASE";
    const FORMAT_VERSION: u32 = 1;
    const PREFIX_LEN: u64 = 16;
    const BLOB_ALIGNMENT: u64 = 64 * 1024;

    let header_end = PREFIX_LEN + new_header_len;
    let new_blob_start = (header_end + BLOB_ALIGNMENT - 1) & !(BLOB_ALIGNMENT - 1);

    let mut out = std::fs::File::create(output.as_ref())?;
    out.write_all(&MAGIC)?;
    out.write_all(&FORMAT_VERSION.to_le_bytes())?;
    out.write_all(&new_header_len.to_le_bytes())?;
    out.write_all(&new_header)?;
    let pad = (new_blob_start - header_end) as usize;
    if pad > 0 {
        out.write_all(&vec![0u8; pad])?;
    }
    out.write_all(&blob_bytes)?;

    // Copy any trailing bytes (extension slots) verbatim. These are
    // addressed by the slots_offset scan at read time, which uses the
    // max tensor offset + length — since tensor offsets are unchanged
    // relative to blob start, the slots section layout survives.
    let original_bytes = std::fs::read(input.as_ref())?;
    let tail_start = (blob_start + blob_end_rel) as usize;
    if tail_start < original_bytes.len() {
        let pad = (8 - (blob_bytes.len() % 8)) % 8;
        if pad > 0 {
            out.write_all(&vec![0u8; pad])?;
        }
        out.write_all(&original_bytes[tail_start..])?;
    }

    Ok(())
}

/// Verify a signed `.base` file. Reads the file, strips the sig from
/// the header, recomputes the payload, and verifies. Returns Ok(()) if
/// the file is either unsigned (nothing to verify) or signed + valid.
pub fn verify_base_file<P: AsRef<std::path::Path>>(path: P, key: &VerifyingKey) -> Result<()> {
    use base_format::BaseReader;

    let reader = BaseReader::open(path.as_ref())
        .with_context(|| format!("opening {:?}", path.as_ref()))?;
    let Some(recorded_sig) = reader.header().sig.clone() else {
        return Ok(());
    };
    if recorded_sig.alg != "ed25519" {
        anyhow::bail!(
            "unsupported signature algorithm: {:?}",
            recorded_sig.alg
        );
    }
    let sig_bytes = b64_decode(&recorded_sig.signature)?;
    if sig_bytes.len() != 64 {
        anyhow::bail!("ed25519 signature must be 64 bytes, got {}", sig_bytes.len());
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let sig = Signature::from_bytes(&sig_arr);

    let blob_end_rel = reader
        .header()
        .tensors
        .iter()
        .map(|t| t.offset + t.length)
        .max()
        .unwrap_or(0);
    let blob_start = reader.blob_offset();
    let file_bytes = std::fs::read(path.as_ref())?;
    let blob_bytes = &file_bytes[blob_start as usize..(blob_start + blob_end_rel) as usize];

    let canonical = serde_json::to_vec(&reader.header().without_sig())?;
    let payload = build_payload(&canonical, blob_bytes);
    verify_payload(key, &payload, &sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand_core::OsRng;

    #[test]
    fn sign_verify_round_trip() {
        let mut rng = OsRng;
        let key = SigningKey::generate(&mut rng);
        let vk = key.verifying_key();

        let payload = build_payload(br#"{"arch":"test"}"#, &[1u8, 2, 3, 4]);
        let sig = sign_payload(&key, &payload);
        verify_payload(&vk, &payload, &sig).unwrap();
    }

    #[test]
    fn tamper_detected() {
        let mut rng = OsRng;
        let key = SigningKey::generate(&mut rng);
        let vk = key.verifying_key();

        let payload = build_payload(br#"{"arch":"test"}"#, &[1u8, 2, 3, 4]);
        let sig = sign_payload(&key, &payload);

        let tampered = build_payload(br#"{"arch":"test"}"#, &[1u8, 2, 3, 5]);
        let err = verify_payload(&vk, &tampered, &sig).unwrap_err();
        assert!(err.to_string().contains("verification failed"));
    }

    #[test]
    fn base64_round_trip() {
        for size in 0..64 {
            let bytes: Vec<u8> = (0..size as u8).collect();
            let s = b64_encode(&bytes);
            let back = b64_decode(&s).unwrap();
            assert_eq!(bytes, back, "size={}", size);
        }
    }

    #[test]
    fn base64_matches_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(b64_encode(b""), "");
        assert_eq!(b64_encode(b"f"), "Zg==");
        assert_eq!(b64_encode(b"fo"), "Zm8=");
        assert_eq!(b64_encode(b"foo"), "Zm9v");
        assert_eq!(b64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(b64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(b64_encode(b"foobar"), "Zm9vYmFy");
    }
}
