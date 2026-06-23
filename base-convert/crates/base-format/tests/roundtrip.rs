use base_format::{
    AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, LayerDescriptor,
    LayerKind, LayerPrecision, Layout, ModelConfig, QuantScheme, ResidencyHint, Slot, SlotKind,
    SourceInfo, TargetBackend, TensorDtype, TensorEntry, TensorFlags, TensorPayload, TokenizerBlob,
};
use std::collections::BTreeMap;

fn make_header() -> Header {
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

fn entry(name: &str, shape: Vec<u64>) -> TensorEntry {
    TensorEntry {
        name: name.to_string(),
        dtype: TensorDtype::F32,
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
    }
}

#[test]
fn roundtrip_two_tensors() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Tensors: sized so alignment matters. 17 bytes of 0xAA then 9 bytes
    // of 0x55 — second will be 64-byte aligned inside the blob.
    let payload_a: Vec<u8> = vec![0xAA; 17];
    let payload_b: Vec<u8> = vec![0x55; 9];

    let header = make_header();
    let mut writer = BaseWriter::create(&path, header).unwrap();
    writer.add_tensor(TensorPayload {
        entry: entry("a", vec![17]),
        data: payload_a.clone(),
    });
    writer.add_tensor(TensorPayload {
        entry: entry("b", vec![9]),
        data: payload_b.clone(),
    });
    writer.finish().unwrap();

    // Read back.
    let reader = BaseReader::open(&path).unwrap();
    assert_eq!(reader.header().arch, "test");
    assert_eq!(reader.header().tensors.len(), 2);
    assert_eq!(reader.tensor_bytes("a").unwrap(), payload_a.as_slice());
    assert_eq!(reader.tensor_bytes("b").unwrap(), payload_b.as_slice());

    // Tensor B should be placed at a 64-byte boundary within the blob.
    let entry_b = reader
        .header()
        .tensors
        .iter()
        .find(|t| t.name == "b")
        .unwrap();
    assert_eq!(entry_b.offset % 64, 0);

    // Blob itself is 64 KiB aligned within the file.
    assert_eq!(reader.blob_offset() % (64 * 1024), 0);
}

fn expect_err(result: Result<BaseReader, base_format::Error>) -> base_format::Error {
    match result {
        Ok(_) => panic!("expected error, got Ok"),
        Err(e) => e,
    }
}

#[test]
fn rejects_bad_magic() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"NOPExxxxxxxxxxxxxxxxxxxxxxxx").unwrap();
    let err = expect_err(BaseReader::open(tmp.path()));
    assert!(matches!(err, base_format::Error::BadMagic(_)));
}

#[test]
fn rejects_truncated() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"BASE").unwrap();
    let err = expect_err(BaseReader::open(tmp.path()));
    assert!(matches!(err, base_format::Error::Truncated));
}

#[test]
fn compute_regions_get_per_region_alignment() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let mut accel = entry("accel_tensor", vec![8]);
    accel.compute_region = ComputeRegion::Accelerator;
    let mut gpu = entry("gpu_tensor", vec![8]);
    gpu.compute_region = ComputeRegion::Gpu;
    let mut cpu = entry("cpu_tensor", vec![8]);
    cpu.compute_region = ComputeRegion::Cpu;

    let mut writer = BaseWriter::create(&path, make_header()).unwrap();
    // Interleave regions on write — the writer must still respect each
    // tensor's region-specific alignment.
    writer.add_tensor(TensorPayload {
        entry: accel.clone(),
        data: vec![0x01; 17],
    });
    writer.add_tensor(TensorPayload {
        entry: gpu.clone(),
        data: vec![0x02; 17],
    });
    writer.add_tensor(TensorPayload {
        entry: cpu.clone(),
        data: vec![0x03; 17],
    });
    writer.add_tensor(TensorPayload {
        entry: gpu.clone(), // second GPU tensor — must re-align to 16 KiB
        data: vec![0x04; 17],
    });
    writer.finish().unwrap();

    let reader = BaseReader::open(&path).unwrap();
    let find = |name: &str| {
        reader
            .header()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap()
            .clone()
    };

    // Accelerator region: 64 B aligned (default accel_align_log2=6).
    let a = find("accel_tensor");
    assert_eq!(a.offset % 64, 0);

    // GPU region: 16 KiB aligned (default gpu_page_log2=14).
    let g = reader
        .header()
        .tensors
        .iter()
        .filter(|t| t.name == "gpu_tensor")
        .collect::<Vec<_>>();
    assert_eq!(g.len(), 2);
    for t in g {
        assert_eq!(t.offset % (16 * 1024), 0);
    }

    // CPU region: 64 B aligned.
    let c = find("cpu_tensor");
    assert_eq!(c.offset % 64, 0);

    // Absolute file offsets for GPU-region tensors must be 16 KiB-aligned
    // (not just blob-relative) — this is what MTLBuffer.makeBufferWith-
    // BytesNoCopy requires. We verify via the reader's eligibility check.
    assert!(reader.tensor_is_zero_copy_eligible("gpu_tensor").unwrap());
    assert!(reader.tensor_is_zero_copy_eligible("accel_tensor").unwrap());
    assert!(reader.tensor_is_zero_copy_eligible("cpu_tensor").unwrap());
}

#[test]
fn roundtrips_layout_and_residency() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let payload: Vec<u8> = vec![0x42; 128];

    let mut entry_with_hints = entry("qkv", vec![256, 128]);
    entry_with_hints.dtype = TensorDtype::BaseQ4;
    entry_with_hints.layout = Some(Layout::Tile8x8Mlx);
    entry_with_hints.residency = Some(ResidencyHint::Warm);
    entry_with_hints.group_size = Some(64);

    let mut writer = BaseWriter::create(tmp.path(), make_header()).unwrap();
    writer.add_tensor(TensorPayload {
        entry: entry_with_hints,
        data: payload.clone(),
    });
    writer.finish().unwrap();

    let reader = BaseReader::open(tmp.path()).unwrap();
    let t = &reader.header().tensors[0];
    assert_eq!(t.layout, Some(Layout::Tile8x8Mlx));
    assert_eq!(t.residency, Some(ResidencyHint::Warm));
    assert_eq!(t.group_size, Some(64));
    assert!(matches!(t.dtype, TensorDtype::BaseQ4));
}

#[test]
fn writer_fills_xxhash64_and_reader_verifies() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let payload: Vec<u8> = (0..256u16).map(|i| (i & 0xff) as u8).collect();
    let expected = xxhash_rust::xxh64::xxh64(&payload, 0);

    let mut writer = BaseWriter::create(tmp.path(), make_header()).unwrap();
    writer.add_tensor(TensorPayload {
        entry: entry("checked", vec![256]),
        data: payload.clone(),
    });
    writer.finish().unwrap();

    let reader = BaseReader::open(tmp.path()).unwrap();
    let t = &reader.header().tensors[0];
    assert_eq!(t.checksum_xxh64, Some(expected));
    reader.verify_tensor("checked").expect("checksum should match");
}

#[test]
fn reader_detects_tensor_corruption() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let payload: Vec<u8> = vec![0xAA; 64];

    let mut writer = BaseWriter::create(&path, make_header()).unwrap();
    let mut e = entry("t", vec![64]);
    e.compute_region = ComputeRegion::Cpu;
    writer.add_tensor(TensorPayload {
        entry: e,
        data: payload,
    });
    writer.finish().unwrap();

    // Corrupt one byte inside the tensor data region. Find the tensor's
    // absolute offset via the reader, then flip a bit.
    let offset = {
        let r = BaseReader::open(&path).unwrap();
        r.tensor_file_offset("t").unwrap()
    };
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[offset as usize] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let reader = BaseReader::open(&path).unwrap();
    let err = reader
        .verify_tensor("t")
        .expect_err("should detect corruption");
    assert!(matches!(err, base_format::Error::ChecksumMismatch { .. }));
}

#[test]
fn rejects_ssm_a_matrix_outside_cpu_f32() {
    let tmp = tempfile::NamedTempFile::new().unwrap();

    // SSM A-matrix placed in GPU region — should be rejected at open.
    let mut bad = entry("layers.0.A_log", vec![16]);
    bad.dtype = TensorDtype::F32;
    bad.compute_region = ComputeRegion::Gpu;
    bad.flags = TensorFlags::SSM_A_MATRIX;

    let mut writer = BaseWriter::create(tmp.path(), make_header()).unwrap();
    writer.add_tensor(TensorPayload {
        entry: bad,
        data: vec![0u8; 64],
    });
    writer.finish().unwrap();

    let err = match BaseReader::open(tmp.path()) {
        Ok(_) => panic!("expected InvalidSsmAMatrix"),
        Err(e) => e,
    };
    assert!(matches!(err, base_format::Error::InvalidSsmAMatrix { .. }));
}

#[test]
fn accepts_ssm_a_matrix_when_cpu_f32() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut good = entry("layers.0.A_log", vec![16]);
    good.dtype = TensorDtype::F32;
    good.compute_region = ComputeRegion::Cpu;
    good.flags = TensorFlags::SSM_A_MATRIX;

    let mut writer = BaseWriter::create(tmp.path(), make_header()).unwrap();
    writer.add_tensor(TensorPayload {
        entry: good,
        data: vec![0u8; 64],
    });
    writer.finish().unwrap();

    let reader = BaseReader::open(tmp.path()).unwrap();
    assert!(reader.header().tensors[0]
        .flags
        .contains(TensorFlags::SSM_A_MATRIX));
}

#[test]
fn roundtrips_header_flags_and_layer_map() {
    let tmp = tempfile::NamedTempFile::new().unwrap();

    let mut header = make_header();
    header.flags = HeaderFlags::QUANTIZED
        | HeaderFlags::HAS_MOE
        | HeaderFlags::TIED_EMBEDDINGS
        | HeaderFlags::SIGNED;
    header.layers = vec![
        LayerDescriptor {
            kind: LayerKind::AttentionGqa,
            moe_n_experts: 0,
            moe_n_active: 0,
            shared_attn_layer: None,
            compute_hint: Some(ComputeRegion::Accelerator),
            precision: LayerPrecision::default(),
        },
        LayerDescriptor {
            kind: LayerKind::AttentionMoe,
            moe_n_experts: 128,
            moe_n_active: 8,
            shared_attn_layer: None,
            compute_hint: None,
            precision: LayerPrecision::default(),
        },
        LayerDescriptor {
            kind: LayerKind::Ssm,
            moe_n_experts: 0,
            moe_n_active: 0,
            shared_attn_layer: None,
            compute_hint: Some(ComputeRegion::Cpu),
            precision: LayerPrecision {
                force_fp32_attn: false,
                force_fp32_ssm: true,
                no_quantize: false,
            },
        },
        LayerDescriptor {
            kind: LayerKind::AttentionGqa,
            moe_n_experts: 0,
            moe_n_active: 0,
            shared_attn_layer: Some(0), // Zamba-style share
            compute_hint: None,
            precision: LayerPrecision::default(),
        },
    ];

    let mut writer = BaseWriter::create(tmp.path(), header).unwrap();
    writer.add_tensor(TensorPayload {
        entry: entry("dummy", vec![4]),
        data: vec![0xCC; 16],
    });
    writer.finish().unwrap();

    let reader = BaseReader::open(tmp.path()).unwrap();
    let h = reader.header();
    assert!(h.flags.contains(HeaderFlags::HAS_MOE));
    assert!(h.flags.contains(HeaderFlags::TIED_EMBEDDINGS));
    assert_eq!(h.layers.len(), 4);
    assert_eq!(h.layers[1].moe_n_experts, 128);
    assert_eq!(h.layers[1].moe_n_active, 8);
    assert!(h.layers[2].precision.force_fp32_ssm);
    assert_eq!(h.layers[3].shared_attn_layer, Some(0));
}

#[test]
fn extension_slots_roundtrip() {
    let tmp = tempfile::NamedTempFile::new().unwrap();

    let mut writer = BaseWriter::create(tmp.path(), make_header()).unwrap();
    writer.add_tensor(TensorPayload {
        entry: entry("dummy", vec![4]),
        data: vec![0xDD; 64],
    });
    // Two slots: a well-known one (RopeTables) and an unknown one that
    // should still roundtrip (forward compat).
    writer.add_slot(Slot::new(SlotKind::RopeTables, vec![0xAB; 200]));
    writer.add_slot(Slot {
        kind_raw: 0x9999, // unknown to current code
        flags: base_format::SlotFlags::empty(),
        payload: vec![0xCD; 37],
        payload_xxh64: xxhash_rust::xxh64::xxh64(&[0xCDu8; 37], 0),
    });
    writer.finish().unwrap();

    let reader = BaseReader::open(tmp.path()).unwrap();
    let slots = reader.slots().unwrap();
    assert_eq!(slots.len(), 2);

    assert_eq!(slots[0].kind(), Some(SlotKind::RopeTables));
    assert_eq!(slots[0].payload.len(), 200);
    assert!(slots[0].payload.iter().all(|&b| b == 0xAB));

    // Unknown slot preserved for forward compat — kind_raw retained,
    // kind() returns None but the slot is still visible.
    assert_eq!(slots[1].kind_raw, 0x9999);
    assert_eq!(slots[1].kind(), None);
    assert_eq!(slots[1].payload.len(), 37);
}

#[test]
fn extension_slot_corruption_detected() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let mut writer = BaseWriter::create(&path, make_header()).unwrap();
    writer.add_tensor(TensorPayload {
        entry: entry("dummy", vec![4]),
        data: vec![0xDD; 64],
    });
    writer.add_slot(Slot::new(SlotKind::KvWarmup, vec![0xEF; 100]));
    writer.finish().unwrap();

    // Flip a byte somewhere late in the file (inside the slot payload).
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 20;
    bytes[last] ^= 0x80;
    std::fs::write(&path, &bytes).unwrap();

    let reader = BaseReader::open(&path).unwrap();
    let err = reader.slots().expect_err("should detect slot corruption");
    assert!(matches!(err, base_format::Error::ChecksumMismatch { .. }));
}

#[test]
fn rejects_unknown_version() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"BASE");
    bytes.extend_from_slice(&999u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    std::fs::write(tmp.path(), &bytes).unwrap();
    let err = expect_err(BaseReader::open(tmp.path()));
    assert!(matches!(err, base_format::Error::UnsupportedVersion(999, _)));
}
