use base_format::{
    AlignmentConfig, BaseWriter, ComputeRegion, Header, HeaderFlags, ModelConfig, QuantScheme,
    SourceInfo, TargetBackend, TensorDtype, TensorEntry, TensorFlags, TensorPayload, TokenizerBlob,
};
use base_sign::{sign_base_file, verify_base_file};
use ed25519_dalek::SigningKey;
use rand_core::OsRng;
use std::collections::BTreeMap;

fn make_header() -> Header {
    Header {
        schema: 1,
        arch: "test".to_string(),
        quant_scheme: QuantScheme::BaseQ4,
        min_hw: "apple_m1".to_string(),
        created: "2026-04-24T00:00:00Z".to_string(),
        base_rt_version: "0.1.0".to_string(),
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
        target_backend: TargetBackend::Metal,
        quant_profile: String::new(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::empty(),
        layers: vec![],
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
    }
}

fn entry(name: &str) -> TensorEntry {
    TensorEntry {
        name: name.to_string(),
        dtype: TensorDtype::F32,
        shape: vec![64],
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
    }
}

#[test]
fn sign_and_verify_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let unsigned = dir.path().join("unsigned.base");
    let signed = dir.path().join("signed.base");

    // Build an unsigned .base file.
    {
        let mut w = BaseWriter::create(&unsigned, make_header()).unwrap();
        w.add_tensor(TensorPayload {
            entry: entry("t1"),
            data: vec![0xABu8; 256],
        });
        w.add_tensor(TensorPayload {
            entry: entry("t2"),
            data: vec![0xCDu8; 128],
        });
        w.finish().unwrap();
    }

    let mut rng = OsRng;
    let key = SigningKey::generate(&mut rng);
    let vk = key.verifying_key();

    sign_base_file(&unsigned, &signed, &key, "test-key-2026").unwrap();
    verify_base_file(&signed, &vk).expect("freshly signed file should verify");
}

#[test]
fn verify_detects_tampered_blob() {
    let dir = tempfile::tempdir().unwrap();
    let unsigned = dir.path().join("unsigned.base");
    let signed = dir.path().join("signed.base");

    {
        let mut w = BaseWriter::create(&unsigned, make_header()).unwrap();
        w.add_tensor(TensorPayload {
            entry: entry("t1"),
            data: vec![0xABu8; 256],
        });
        w.finish().unwrap();
    }

    let mut rng = OsRng;
    let key = SigningKey::generate(&mut rng);
    let vk = key.verifying_key();
    sign_base_file(&unsigned, &signed, &key, "test-key").unwrap();

    // Tamper with a byte inside the blob.
    let mut bytes = std::fs::read(&signed).unwrap();
    let tamper_at = bytes.len() - 100; // somewhere in the blob
    bytes[tamper_at] ^= 0x01;
    std::fs::write(&signed, &bytes).unwrap();

    let err = verify_base_file(&signed, &vk).expect_err("tampered file should not verify");
    assert!(err.to_string().contains("verification failed"));
}

#[test]
fn verify_detects_wrong_key() {
    let dir = tempfile::tempdir().unwrap();
    let unsigned = dir.path().join("unsigned.base");
    let signed = dir.path().join("signed.base");

    {
        let mut w = BaseWriter::create(&unsigned, make_header()).unwrap();
        w.add_tensor(TensorPayload {
            entry: entry("t1"),
            data: vec![0xABu8; 256],
        });
        w.finish().unwrap();
    }

    let mut rng = OsRng;
    let signing_key = SigningKey::generate(&mut rng);
    let impostor_key = SigningKey::generate(&mut rng);
    sign_base_file(&unsigned, &signed, &signing_key, "real-key").unwrap();

    let err = verify_base_file(&signed, &impostor_key.verifying_key())
        .expect_err("wrong key must not verify");
    assert!(err.to_string().contains("verification failed"));
}

#[test]
fn unsigned_file_verifies_as_noop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsigned.base");

    {
        let mut w = BaseWriter::create(&path, make_header()).unwrap();
        w.add_tensor(TensorPayload {
            entry: entry("t1"),
            data: vec![0xABu8; 256],
        });
        w.finish().unwrap();
    }

    let mut rng = OsRng;
    let key = SigningKey::generate(&mut rng);
    verify_base_file(&path, &key.verifying_key()).expect("unsigned file = noop verify");
}
