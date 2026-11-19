//! `basert merge`: a dense overlay of the base's own dtypes is patched into
//! a copy-on-write clone of the base (no second bundle on disk); anything
//! else (a dtype change, a different length) rewrites the bundle. Either
//! way the output holds the overlay's bytes where it replaced a tensor and
//! the base's everywhere else, its slots and provenance included.
use base_format::{
    AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, ModelConfig,
    QuantScheme, Slot, SlotKind, SourceInfo, TargetBackend, TensorDtype, TensorEntry, TensorFlags,
    TensorPayload, TokenizerBlob,
};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

fn basert() -> Command {
    Command::new(env!("CARGO_BIN_EXE_basert"))
}

fn header(provenance: Option<serde_json::Value>) -> Header {
    Header {
        schema: 1,
        arch: "test".to_string(),
        quant_scheme: QuantScheme::BaseQ4,
        min_hw: "apple_m1".to_string(),
        created: "2026-04-24T00:00:00Z".to_string(),
        base_rt_version: "0.1.0-test".to_string(),
        source: SourceInfo {
            format: "test".to_string(),
            sha256: "0".repeat(64),
            filename: "synthetic".to_string(),
        },
        tokenizer: TokenizerBlob {
            fields: BTreeMap::new(),
        },
        config: ModelConfig {
            fields: BTreeMap::new(),
        },
        metadata: BTreeMap::new(),
        target_backend: TargetBackend::Metal,
        quant_profile: String::new(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::empty(),
        layers: vec![],
        tensors: vec![],
        mmproj: None,
        speculator: None,
        calibration: None,
        provenance,
        sig: None,
    }
}

fn entry(name: &str, dtype: TensorDtype, shape: Vec<u64>) -> TensorEntry {
    TensorEntry {
        name: name.to_string(),
        dtype,
        shape,
        offset: 0,
        length: 0,
        scale_offset: None,
        scale_length: None,
        bias_offset: None,
        bias_length: None,
        awq_scale_offset: None,
        awq_scale_length: None,
        group_size: None,
        layout: None,
        residency: None,
        compute_region: ComputeRegion::Gpu,
        scale_dtype: None,
        symmetric: false,
        flags: TensorFlags::empty(),
        checksum_xxh64: None,
        source_ggml_type: None,
        algo: None,
        awq_alpha: None,
    }
}

fn write(
    path: &Path,
    prov: Option<serde_json::Value>,
    tensors: &[(&str, TensorDtype, Vec<u64>, Vec<u8>)],
    slots: bool,
) {
    write_with(path, prov, tensors, slots, None);
}

/// `reserve`: `Some(n)` writes through `BaseWriter::create_direct` with an
/// n-byte header reserve (the `convert --direct-write` layout: header_len =
/// n, the JSON space-padded to it); `None` is the compact `create`.
fn write_with(
    path: &Path,
    prov: Option<serde_json::Value>,
    tensors: &[(&str, TensorDtype, Vec<u64>, Vec<u8>)],
    slots: bool,
    reserve: Option<u64>,
) {
    let mut w = match reserve {
        Some(n) => BaseWriter::create_direct(path, header(prov), n).unwrap(),
        None => BaseWriter::create(path, header(prov)).unwrap(),
    };
    for (name, dtype, shape, bytes) in tensors {
        w.add_tensor(TensorPayload {
            entry: entry(name, *dtype, shape.clone()),
            data: bytes.clone(),
        });
    }
    if slots {
        w.add_slot(Slot::new(SlotKind::LoraWeights, b"slot bytes".to_vec()));
    }
    w.finish().unwrap();
}

fn pattern(seed: u8, n: usize) -> Vec<u8> {
    (0..n).map(|i| seed.wrapping_add((i % 251) as u8)).collect()
}

fn merge(base: &Path, overlay: &Path, out: &Path) -> String {
    let o = basert()
        .args([
            "merge",
            base.to_str().unwrap(),
            overlay.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "merge failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn method(out: &Path) -> String {
    let r = BaseReader::open(out).unwrap();
    r.header().provenance.as_ref().unwrap()["merge"]["method"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn a_dense_overlay_of_the_same_dtype_is_patched_into_a_clone() {
    let dir = tempfile::tempdir().unwrap();
    let (base, overlay, out) = (
        dir.path().join("base.base"),
        dir.path().join("ov.base"),
        dir.path().join("out.base"),
    );
    let (a, b, c) = (
        pattern(1, 4096),
        pattern(2, 2 * 64 * 65),
        pattern(3, 4 * 33),
    );
    let prov = serde_json::json!({"tensors": {"a": {"src": ["a.src"]}}});
    write(
        &base,
        Some(prov),
        &[
            ("a", TensorDtype::Bf16, vec![32, 64], a.clone()),
            ("b", TensorDtype::Bf16, vec![64, 65], b.clone()),
            ("c", TensorDtype::F32, vec![33], c.clone()),
        ],
        true,
    );
    let b2 = pattern(9, b.len());
    write(
        &overlay,
        None,
        &[("b", TensorDtype::Bf16, vec![64, 65], b2.clone())],
        false,
    );
    let log = merge(&base, &overlay, &out);
    let r = BaseReader::open(&out).unwrap();
    assert_eq!(r.tensor_bytes("a").unwrap(), &a[..]);
    assert_eq!(r.tensor_bytes("b").unwrap(), &b2[..], "the overlay's bytes");
    assert_eq!(r.tensor_bytes("c").unwrap(), &c[..]);
    for n in ["a", "b", "c"] {
        r.verify_tensor(n).unwrap();
    }
    let slots = r.slots().unwrap();
    assert_eq!(slots.len(), 1, "the base's slots survive");
    assert_eq!(slots[0].payload, b"slot bytes");
    let prov = r.header().provenance.as_ref().unwrap();
    assert_eq!(
        prov["tensors"]["a"]["src"][0], "a.src",
        "the base's provenance is kept"
    );
    assert_eq!(prov["merge"]["replaced"], serde_json::json!(["b"]));
    if cfg!(target_os = "macos") {
        assert_eq!(prov["merge"]["method"], "clone_patch", "{log}");
        assert!(log.contains("patched into a clone"), "{log}");
        // the clone keeps the base's layout: same blob start, same offsets
        let rb = BaseReader::open(&base).unwrap();
        assert_eq!(r.blob_offset(), rb.blob_offset());
        for n in ["a", "b", "c"] {
            assert_eq!(
                r.tensor_file_offset(n).unwrap(),
                rb.tensor_file_offset(n).unwrap(),
                "{n}"
            );
        }
    } else {
        assert_eq!(prov["merge"]["method"], "rewrite");
    }
    // the base is untouched
    let rb = BaseReader::open(&base).unwrap();
    assert_eq!(rb.tensor_bytes("b").unwrap(), &b[..]);
    rb.verify_tensor("b").unwrap();
}

/// The 16-byte prefix's header_len field: what the reader parses as the
/// header, space padding included.
fn header_len(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    u64::from_le_bytes(bytes[8..16].try_into().unwrap())
}

#[test]
fn a_direct_written_base_keeps_its_reserved_header_in_the_clone() {
    // `convert --direct-write` reserves 64 MiB for the header and pads its
    // JSON with spaces; the merged header fits in the reserve with room to
    // spare, so the clone keeps header_len = reserve and the base's blob
    // start instead of rewriting a model-sized bundle.
    const RESERVE: u64 = 64 * 1024 * 1024;
    let dir = tempfile::tempdir().unwrap();
    let (base, overlay, out) = (
        dir.path().join("base.base"),
        dir.path().join("ov.base"),
        dir.path().join("out.base"),
    );
    let (a, b) = (pattern(1, 4096), pattern(2, 2 * 64 * 65));
    write_with(
        &base,
        None,
        &[
            ("a", TensorDtype::Bf16, vec![32, 64], a.clone()),
            ("b", TensorDtype::Bf16, vec![64, 65], b.clone()),
        ],
        true,
        Some(RESERVE),
    );
    assert_eq!(header_len(&base), RESERVE);
    let rb = BaseReader::open(&base).unwrap();
    assert_eq!(rb.blob_offset(), RESERVE + 64 * 1024);
    let b2 = pattern(9, b.len());
    write(
        &overlay,
        None,
        &[("b", TensorDtype::Bf16, vec![64, 65], b2.clone())],
        false,
    );
    let log = merge(&base, &overlay, &out);
    let r = BaseReader::open(&out).unwrap();
    assert_eq!(r.tensor_bytes("a").unwrap(), &a[..]);
    assert_eq!(r.tensor_bytes("b").unwrap(), &b2[..], "the overlay's bytes");
    r.verify_tensor("a").unwrap();
    r.verify_tensor("b").unwrap();
    assert_eq!(r.slots().unwrap().len(), 1);
    assert_eq!(
        r.header().provenance.as_ref().unwrap()["merge"]["replaced"],
        serde_json::json!(["b"])
    );
    if cfg!(target_os = "macos") {
        assert_eq!(method(&out), "clone_patch", "{log}");
        assert_eq!(header_len(&out), RESERVE, "the reserve survives the merge");
        assert_eq!(r.blob_offset(), rb.blob_offset());
        for n in ["a", "b"] {
            assert_eq!(
                r.tensor_file_offset(n).unwrap(),
                rb.tensor_file_offset(n).unwrap(),
                "{n}"
            );
        }
    } else {
        assert_eq!(method(&out), "rewrite");
    }
}

#[test]
fn a_dtype_change_rewrites_the_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let (base, overlay, out) = (
        dir.path().join("base.base"),
        dir.path().join("ov.base"),
        dir.path().join("out.base"),
    );
    let (a, b) = (pattern(1, 4096), pattern(2, 2 * 64 * 65));
    write(
        &base,
        None,
        &[
            ("a", TensorDtype::Bf16, vec![32, 64], a.clone()),
            ("b", TensorDtype::Bf16, vec![64, 65], b.clone()),
        ],
        true,
    );
    // f16 has bf16's length: the dtype alone sends the merge down the rewrite path
    let b2 = pattern(9, b.len());
    write(
        &overlay,
        None,
        &[("b", TensorDtype::F16, vec![64, 65], b2.clone())],
        false,
    );
    merge(&base, &overlay, &out);
    assert_eq!(method(&out), "rewrite");
    let r = BaseReader::open(&out).unwrap();
    assert_eq!(r.tensor_bytes("a").unwrap(), &a[..]);
    assert_eq!(r.tensor_bytes("b").unwrap(), &b2[..]);
    assert_eq!(r.find_entry("b").unwrap().dtype, TensorDtype::F16);
    r.verify_tensor("a").unwrap();
    r.verify_tensor("b").unwrap();
    assert_eq!(r.slots().unwrap().len(), 1);
    assert!(
        !dir.path().join("out.base.blobtmp").exists(),
        "direct write: no temp blob"
    );
}

#[test]
fn a_shape_mismatch_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (base, overlay, out) = (
        dir.path().join("base.base"),
        dir.path().join("ov.base"),
        dir.path().join("out.base"),
    );
    write(
        &base,
        None,
        &[("a", TensorDtype::Bf16, vec![32, 64], pattern(1, 4096))],
        false,
    );
    write(
        &overlay,
        None,
        &[("a", TensorDtype::Bf16, vec![64, 32], pattern(2, 4096))],
        false,
    );
    let o = basert()
        .args([
            "merge",
            base.to_str().unwrap(),
            overlay.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("shape"));
}
