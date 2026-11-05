//! `basert add-speculator`: embed a converted drafter sidecar in a target
//! bundle as its built-in speculator head (`header.speculator`), the way
//! `convert` embeds a Qwen3.5 checkpoint's own MTP head. Google ships Gemma
//! 4's head as a separate checkpoint (`google/gemma-4-<size>-it-assistant`);
//! embedding it gives a Gemma 4 bundle the same `--speculate mtp-head` /
//! `auto` story as a Qwen one, with nothing to pair at serve time.
//!
//! Byte-level: the target's weights blob is copied verbatim, so every main
//! and mmproj tensor keeps its bytes, offsets and checksums; the head's
//! tensors are appended behind it, each placed as the writer places a
//! tensor of its region by the stricter of the two files' alignment rules,
//! the first of them on a GPU page (the runtime wraps the section as one
//! buffer from its first entry); the header gains the section and
//! `HAS_SPECULATOR`. Nothing is re-quantized, and a reconverted target is
//! not needed: the catalog's files are embedded as they are. The head is
//! embedded as it was converted, f16 or quantized (`convert
//! --quantize-drafter`): a quantized entry's scales and biases sit inside
//! its own payload (`length` spans them, their offsets are relative to the
//! entry's), so the entry moves whole and the runtime reads it from the
//! section as it reads it from the sidecar.
//!
//! The section is the sidecar's own header, not a Qwen-style one: `arch` is
//! the sidecar's (`gemma4-mtp`), `config` is the sidecar's whole config (the
//! head's own geometry — a 4-layer, narrow Gemma 4 decoder over the target's
//! KV, plus `backbone_hidden_size`) with `kind` added, and the tensors keep
//! the sidecar's names (`output.weight`, `layers.0.*`, `fc_embed.weight`,
//! …, no `mtp.` prefix). The runtime opens the section as if it were the
//! sidecar file. `kind` is the sidecar arch rather than `mtp` on purpose: a
//! runtime or superfluid from before this head existed reads `kind`, sees it
//! is not the one-layer MTP head it knows, and leaves the section alone
//! (`auto` decodes plainly, bare `mtp-head` refuses cleanly) instead of
//! loading the target a second time as a head it cannot run.
use anyhow::{bail, Context, Result};
use base_format::{AlignmentConfig, ComputeRegion, HeaderFlags, BLOB_ALIGNMENT, PREFIX_LEN};
use clap::Parser;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
pub struct AddSpeculatorArgs {
    /// Target `.base` bundle: the model the head drafts for.
    pub target: PathBuf,
    /// The head, converted on its own (`basert convert
    /// google/gemma-4-E4B-it-assistant` writes a `gemma4-mtp` sidecar), f16
    /// or quantized (`--quantize-drafter --target base-q8`).
    #[arg(long)]
    pub head: PathBuf,
    /// Where to write the embedded bundle (refused if it exists).
    #[arg(short, long)]
    pub output: PathBuf,
    /// A version for the head, recorded as `head_version` in the section
    /// (a catalog row carries it as `speculator_version`): a later head for
    /// the same model gets a new one, and `basert serve --speculate` pulls a
    /// file whose head is older than the catalog's. Any string; absent means
    /// unversioned.
    #[arg(long, value_name = "V")]
    pub head_version: Option<String>,
}

pub fn cmd_add_speculator(args: AddSpeculatorArgs) -> Result<()> {
    let s = add_speculator(
        &args.target,
        &args.head,
        &args.output,
        args.head_version.as_deref(),
    )?;
    eprintln!(
        "{}: {} head ({} tensors, {} bytes) embedded as `{}`; {} bytes written",
        args.output.display(),
        s.head_arch,
        s.n_head_tensors,
        s.head_bytes,
        s.kind,
        s.out_bytes
    );
    Ok(())
}

#[derive(Debug)]
pub struct Summary {
    pub head_arch: String,
    pub kind: String,
    pub n_head_tensors: usize,
    pub head_bytes: u64,
    pub out_bytes: u64,
}

/// A bundle's header as the open JSON it is on disk, plus where its blob
/// starts and how long the file is.
struct Opened {
    path: PathBuf,
    header: serde_json::Value,
    blob_start: u64,
    file_size: u64,
}

fn open_header(path: &Path) -> Result<Opened> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_size = f.metadata()?.len();
    let mut prefix = [0u8; PREFIX_LEN as usize];
    f.read_exact(&mut prefix)
        .with_context(|| format!("{}: not a .base file (truncated prefix)", path.display()))?;
    if prefix[0..4] != base_format::MAGIC {
        bail!("{}: not a .base file (bad magic)", path.display());
    }
    let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
    if version != base_format::FORMAT_VERSION {
        bail!(
            "{}: format version {version}, expected {}",
            path.display(),
            base_format::FORMAT_VERSION
        );
    }
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    if header_len == 0 || PREFIX_LEN + header_len > file_size {
        bail!("{}: implausible header length {header_len}", path.display());
    }
    let mut json = vec![0u8; header_len as usize];
    f.read_exact(&mut json)?;
    let header: serde_json::Value = serde_json::from_slice(&json)
        .with_context(|| format!("{}: header is not JSON", path.display()))?;
    if !header.is_object() {
        bail!("{}: header is not a JSON object", path.display());
    }
    Ok(Opened {
        path: path.to_path_buf(),
        header,
        blob_start: align_up(PREFIX_LEN + header_len, BLOB_ALIGNMENT),
        file_size,
    })
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn tensors_of(section: &serde_json::Value) -> Vec<&serde_json::Value> {
    section
        .get("tensors")
        .and_then(|t| t.as_array())
        .map(|a| a.iter().collect())
        .unwrap_or_default()
}

fn u64_of(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

/// The last blob byte any tensor of the header (main, mmproj, speculator)
/// reaches. An entry's `length` spans its weights and, when quantized, its
/// scales and biases (sub-region offsets are relative to the entry's), so
/// the entry's own range is the whole of it.
fn blob_end(header: &serde_json::Value) -> u64 {
    let mut end = 0u64;
    let mut visit = |t: &serde_json::Value| {
        if let (Some(o), Some(l)) = (u64_of(t, "offset"), u64_of(t, "length")) {
            end = end.max(o + l);
        }
    };
    for t in tensors_of(header) {
        visit(t);
    }
    for sec in ["mmproj", "speculator"] {
        if let Some(s) = header.get(sec) {
            for t in tensors_of(s) {
                visit(t);
            }
        }
    }
    end
}

fn parse_flags(header: &serde_json::Value) -> Result<HeaderFlags> {
    match header.get("flags") {
        None => Ok(HeaderFlags::empty()),
        Some(v) => serde_json::from_value(v.clone()).context("header flags"),
    }
}

/// A quantized entry's sub-regions (scales, biases, AWQ scales), each
/// `(offset, length)` relative to the entry's own payload.
const SUB_REGIONS: [(&str, &str, &str); 3] = [
    ("scale_offset", "scale_length", "scales"),
    ("bias_offset", "bias_length", "biases"),
    ("awq_scale_offset", "awq_scale_length", "AWQ scales"),
];

/// Why a sub-region of `t` reaches outside its payload, if one does: the
/// runtime refuses such an entry when it opens the section, so refuse it
/// before the copy. The sum is checked without overflowing.
fn sub_region_overrun(t: &serde_json::Value) -> Option<String> {
    let len = u64_of(t, "length").unwrap_or(0);
    for (off, sub_len, what) in SUB_REGIONS {
        if let (Some(o), Some(l)) = (u64_of(t, off), u64_of(t, sub_len)) {
            if l > 0 && (l > len || o > len - l) {
                return Some(format!(
                    "{what} [{o}, {}) outside its payload of {len} bytes",
                    o.saturating_add(l)
                ));
            }
        }
    }
    None
}

/// Embed the head sidecar at `head` in the bundle at `target`, writing the
/// result to `out`; `head_version`, when given, is recorded in the section.
/// See the module doc for the layout.
pub fn add_speculator(
    target: &Path,
    head: &Path,
    out: &Path,
    head_version: Option<&str>,
) -> Result<Summary> {
    if out.exists() {
        bail!("{}: exists; refusing to overwrite it", out.display());
    }
    let t = open_header(target)?;
    let h = open_header(head)?;

    // ── The target: a plain bundle with room behind its last tensor ──
    if let Some(s) = t.header.get("speculator") {
        bail!(
            "{}: already carries a speculator head (arch {}); refusing to replace it",
            target.display(),
            s.get("arch").and_then(|a| a.as_str()).unwrap_or("?")
        );
    }
    let t_flags = parse_flags(&t.header)?;
    let slot_flags = HeaderFlags::HAS_LORA
        | HeaderFlags::HAS_COMPUTE_GRAPH
        | HeaderFlags::HAS_KV_WARMUP
        | HeaderFlags::HAS_TRACE_REF;
    if t_flags.intersects(slot_flags) {
        bail!(
            "{}: carries extension slots ({:?}); embedding a head behind them is not supported",
            target.display(),
            t_flags
        );
    }
    if t_flags.contains(HeaderFlags::SIGNED) || t.header.get("sig").is_some() {
        bail!("{}: is signed; the embedded bundle would need signing again (strip the signature first)", target.display());
    }
    let t_blob_len = t.file_size - t.blob_start;
    let t_end = blob_end(&t.header);
    if t_end != t_blob_len {
        bail!(
            "{}: {} bytes follow the last tensor (extension slots?); refusing to append behind them",
            target.display(),
            t_blob_len as i128 - t_end as i128
        );
    }

    // ── The head: an MTP sidecar of this target's backbone, f16 or quantized ──
    let head_arch = h
        .header
        .get("arch")
        .and_then(|a| a.as_str())
        .unwrap_or("")
        .to_string();
    if !base_arch::dflash::is_mtp_sidecar_arch(&head_arch) {
        bail!(
            "{}: header arch `{head_arch}` is not an MTP head sidecar (expected gemma4-mtp)",
            head.display()
        );
    }
    let backbone = head_arch.strip_suffix("-mtp").unwrap_or(&head_arch);
    let t_arch = t.header.get("arch").and_then(|a| a.as_str()).unwrap_or("");
    if t_arch != backbone {
        bail!(
            "{}: arch `{t_arch}`, but the head drafts for `{backbone}`",
            target.display()
        );
    }
    if h.header.get("mmproj").is_some() || h.header.get("speculator").is_some() {
        bail!(
            "{}: a head sidecar carries no sub-bundles of its own",
            head.display()
        );
    }
    let h_cfg = h
        .header
        .get("config")
        .filter(|c| c.is_object())
        .context("head config")?;
    let t_cfg = t
        .header
        .get("config")
        .filter(|c| c.is_object())
        .context("target config")?;
    for (want, key_h, key_t) in [
        ("hidden width", "backbone_hidden_size", "hidden_size"),
        ("vocabulary", "vocab_size", "vocab_size"),
    ] {
        let (a, b) = (u64_of(h_cfg, key_h), u64_of(t_cfg, key_t));
        if a.is_none() || a != b {
            bail!(
                "the head's {want} ({key_h} {}) does not match the target's ({key_t} {})",
                a.map_or("missing".to_string(), |v| v.to_string()),
                b.map_or("missing".to_string(), |v| v.to_string())
            );
        }
    }
    let h_tensors = tensors_of(&h.header);
    if h_tensors.is_empty() {
        bail!("{}: no tensors", head.display());
    }
    // A quantized head's entries move whole (see the module doc); only a
    // sub-region that reaches past its payload is refused.
    for t in &h_tensors {
        if let Some(why) = sub_region_overrun(t) {
            bail!(
                "{}: tensor {} has its {why}",
                head.display(),
                t.get("name").and_then(|n| n.as_str()).unwrap_or("?")
            );
        }
    }
    let h_blob_len = h.file_size - h.blob_start;
    if blob_end(&h.header) > h_blob_len {
        bail!(
            "{}: a tensor reaches past the end of the file",
            head.display()
        );
    }

    // ── Place the head's tensors behind the target's blob ──
    // Each by the stricter of the two files' rules for its region: the
    // target's, which placed the target's own tensors, and the head's,
    // which placed it as a sidecar (a Metal head's converter puts
    // accelerator-region tensors on 16 KiB pages; a catalog target from an
    // older converter asks 64 B of them). The first one starts a GPU page
    // whatever the rules say: the runtime wraps the section as one buffer
    // from its first entry, and Metal needs that base page-aligned — at
    // 64 B the whole head falls back to per-tensor copies.
    let alignment_of = |o: &Opened, what: &str| -> Result<AlignmentConfig> {
        Ok(match o.header.get("alignment") {
            Some(a) => {
                serde_json::from_value(a.clone()).with_context(|| format!("{what} alignment"))?
            }
            None => AlignmentConfig::default(),
        })
    };
    let (t_align, h_align) = (alignment_of(&t, "target")?, alignment_of(&h, "head")?);
    let align_for =
        |region: ComputeRegion| t_align.align_for(region).max(h_align.align_for(region));
    let mut cursor = t_blob_len;
    let mut placed: Vec<(serde_json::Value, u64, u64, u64)> = Vec::with_capacity(h_tensors.len()); // (entry, src offset, dst offset, length)
    for (i, src) in h_tensors.iter().enumerate() {
        let region: ComputeRegion = match src.get("compute_region") {
            Some(r) => serde_json::from_value(r.clone()).context("head tensor compute_region")?,
            None => ComputeRegion::default(),
        };
        let (off, len) = (
            u64_of(src, "offset").context("head tensor offset")?,
            u64_of(src, "length").context("head tensor length")?,
        );
        let mut align = align_for(region);
        if i == 0 {
            align = align.max(align_for(ComputeRegion::Gpu));
        }
        let dst = align_up(cursor, align);
        let mut entry = (*src).clone();
        entry["offset"] = serde_json::json!(dst);
        placed.push((entry, off, dst, len));
        cursor = dst + len;
    }
    let head_bytes: u64 = placed.iter().map(|p| p.3).sum();

    // ── The new header: the target's, plus the section and its flag ──
    let kind = head_arch.clone();
    let mut config: std::collections::BTreeMap<String, serde_json::Value> = h_cfg
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    config.insert("kind".to_string(), serde_json::json!(kind));
    // Where the head came from and, when the caller says, which one it is:
    // the sidecar's source checkpoint name, and a version a catalog row can
    // repeat (`speculator_version`) so a client knows its file's head is old.
    if let Some(src) = h
        .header
        .get("source")
        .and_then(|s| s.get("filename"))
        .and_then(|f| f.as_str())
    {
        if !src.is_empty() {
            config.insert("head_source".to_string(), serde_json::json!(src));
        }
    }
    if let Some(v) = head_version {
        config.insert("head_version".to_string(), serde_json::json!(v));
    }
    let section = serde_json::json!({
        "arch": head_arch,
        "config": config,
        "tensors": placed.iter().map(|p| p.0.clone()).collect::<Vec<_>>(),
    });
    let flags = t_flags | HeaderFlags::HAS_SPECULATOR;
    // The writer emits the struct's field order (`mmproj`, `speculator`,
    // `calibration`, …); keep the section where a converted bundle has it.
    let mut header = serde_json::Map::new();
    let mut inserted = false;
    for (k, v) in t.header.as_object().unwrap() {
        if !inserted && matches!(k.as_str(), "calibration" | "sig" | "provenance") {
            header.insert("speculator".to_string(), section.clone());
            inserted = true;
        }
        if k == "flags" {
            header.insert(k.clone(), serde_json::to_value(flags)?);
        } else {
            header.insert(k.clone(), v.clone());
        }
    }
    if !inserted {
        header.insert("speculator".to_string(), section);
    }
    if !header.contains_key("flags") {
        header.insert("flags".to_string(), serde_json::to_value(flags)?);
    }
    let header_json = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let header_len = header_json.len() as u64;
    let blob_start = align_up(PREFIX_LEN + header_len, BLOB_ALIGNMENT);

    // ── Write: prefix, header, pad, the target's blob verbatim, the head ──
    let mut w = std::io::BufWriter::with_capacity(
        8 << 20,
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out)
            .with_context(|| format!("creating {}", out.display()))?,
    );
    w.write_all(&base_format::MAGIC)?;
    w.write_all(&base_format::FORMAT_VERSION.to_le_bytes())?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_json)?;
    write_zeros(&mut w, blob_start - PREFIX_LEN - header_len)?;
    {
        let mut src = std::fs::File::open(&t.path)?;
        src.seek(SeekFrom::Start(t.blob_start))?;
        let copied = std::io::copy(&mut src.take(t_blob_len), &mut w)?;
        if copied != t_blob_len {
            bail!(
                "{}: short read ({copied} of {t_blob_len} blob bytes)",
                target.display()
            );
        }
    }
    let mut written = t_blob_len;
    {
        let mut src = std::fs::File::open(&h.path)?;
        for (_, off, dst, len) in &placed {
            write_zeros(&mut w, dst - written)?;
            src.seek(SeekFrom::Start(h.blob_start + off))?;
            let copied = std::io::copy(&mut (&src).take(*len), &mut w)?;
            if copied != *len {
                bail!(
                    "{}: short read ({copied} of {len} bytes at {off})",
                    head.display()
                );
            }
            written = dst + len;
        }
    }
    w.flush()?;
    let out_bytes = blob_start + written;
    Ok(Summary {
        head_arch: head_arch.clone(),
        kind,
        n_head_tensors: placed.len(),
        head_bytes,
        out_bytes,
    })
}

fn write_zeros<W: Write>(w: &mut W, n: u64) -> Result<()> {
    static Z: [u8; 65536] = [0u8; 65536];
    let mut left = n;
    while left > 0 {
        let k = left.min(Z.len() as u64) as usize;
        w.write_all(&Z[..k])?;
        left -= k as u64;
    }
    Ok(())
}

/// Tiny bundles for tests: a header of one arch and config, an f16 entry,
/// a filled payload. Shared with hub.rs's tests.
#[cfg(test)]
pub(crate) mod fixtures {
    use base_format::{
        AlignmentConfig, ComputeRegion, Header, HeaderFlags, ModelConfig, QuantScheme, SourceInfo,
        TargetBackend, TensorDtype, TensorEntry, TensorFlags, TensorPayload, TokenizerBlob,
    };
    use std::collections::BTreeMap;

    pub(crate) fn header(arch: &str, cfg: serde_json::Value) -> Header {
        Header {
            schema: 1,
            arch: arch.to_string(),
            quant_scheme: QuantScheme::F16,
            min_hw: "apple_m1".to_string(),
            created: "2026-10-29T00:00:00Z".to_string(),
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
                fields: cfg
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
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
            provenance: None,
            sig: None,
        }
    }

    pub(crate) fn entry(name: &str, shape: Vec<u64>) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            dtype: TensorDtype::F16,
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

    pub(crate) fn payload(name: &str, n: usize, fill: u8) -> TensorPayload {
        TensorPayload {
            entry: entry(name, vec![n as u64 / 2]),
            data: vec![fill; n],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use base_format::{BaseReader, BaseWriter, TensorDtype, TensorEntry, TensorPayload};

    /// A tiny Gemma-4-shaped target (two main tensors, one mmproj tensor)
    /// and a head sidecar over it.
    fn fixtures(dir: &Path) -> (PathBuf, PathBuf) {
        let target = dir.join("target.base");
        let mut w = BaseWriter::create(
            &target,
            header(
                "gemma4",
                serde_json::json!({"hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 2}),
            ),
        )
        .unwrap();
        w.add_tensor(payload("token_embedding.weight", 256, 0xA1));
        w.add_tensor(payload("layers.0.attention.q.weight", 100, 0xA2));
        w.set_mmproj_arch("gemma4-vision");
        w.add_mmproj_tensor(payload("vision.patch.weight", 50, 0xB1));
        w.finish().unwrap();

        let head = dir.join("head.base");
        let mut w = BaseWriter::create(
            &head,
            header(
                "gemma4-mtp",
                serde_json::json!({"hidden_size": 4, "backbone_hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 4, "speculator_kind": "mtp-shared-kv"}),
            ),
        )
        .unwrap();
        w.add_tensor(payload("output.weight", 128, 0xC1));
        w.add_tensor(payload("layers.0.input_norm.weight", 8, 0xC2));
        w.add_tensor(payload("fc_embed.weight", 64, 0xC3));
        w.finish().unwrap();
        (target, head)
    }

    fn bytes_of(path: &Path, blob_start: u64, e: &TensorEntry) -> Vec<u8> {
        let mut f = std::fs::File::open(path).unwrap();
        f.seek(SeekFrom::Start(blob_start + e.offset)).unwrap();
        let mut v = vec![0u8; e.length as usize];
        f.read_exact(&mut v).unwrap();
        v
    }

    #[test]
    fn embeds_the_head_behind_an_untouched_target_blob() {
        let dir = tempfile::tempdir().unwrap();
        let (target, head) = fixtures(dir.path());
        let out = dir.path().join("out.base");
        let s = add_speculator(&target, &head, &out, Some("1")).unwrap();
        assert_eq!(
            (
                s.head_arch.as_str(),
                s.kind.as_str(),
                s.n_head_tensors,
                s.head_bytes
            ),
            ("gemma4-mtp", "gemma4-mtp", 3, 200)
        );
        assert_eq!(s.out_bytes, std::fs::metadata(&out).unwrap().len());

        let before = BaseReader::open(&target).unwrap();
        let after = BaseReader::open(&out).unwrap();
        let (hb, ha) = (before.header(), after.header());
        // The target's own sections are the same entries over the same bytes.
        let same = |a: &TensorEntry, b: &TensorEntry| {
            serde_json::to_string(a).unwrap() == serde_json::to_string(b).unwrap()
        };
        assert_eq!(hb.tensors.len(), ha.tensors.len());
        for (a, b) in hb.tensors.iter().zip(&ha.tensors) {
            assert!(same(a, b), "{}", a.name);
            assert_eq!(
                before.tensor_bytes(&a.name).unwrap(),
                after.tensor_bytes(&b.name).unwrap()
            );
        }
        let (ma, mb) = (hb.mmproj.as_ref().unwrap(), ha.mmproj.as_ref().unwrap());
        assert_eq!(
            (ma.arch.as_str(), ma.tensors.len()),
            (mb.arch.as_str(), mb.tensors.len())
        );
        assert!(same(&ma.tensors[0], &mb.tensors[0]));
        assert_eq!(hb.config.fields, ha.config.fields);
        assert_eq!(hb.arch, ha.arch);
        assert!(
            ha.flags.contains(HeaderFlags::HAS_SPECULATOR)
                && !hb.flags.contains(HeaderFlags::HAS_SPECULATOR)
        );

        // The section is the sidecar's header, `kind` added, names kept,
        // bytes and checksums intact, placed by the target's alignment.
        let spec = ha.speculator.as_ref().unwrap();
        assert_eq!(spec.arch, "gemma4-mtp");
        assert_eq!(spec.config["kind"], serde_json::json!("gemma4-mtp"));
        assert_eq!(spec.config["backbone_hidden_size"], serde_json::json!(8));
        assert_eq!(
            spec.config["speculator_kind"],
            serde_json::json!("mtp-shared-kv")
        );
        let hh = BaseReader::open(&head).unwrap();
        let t_blob_end = hb
            .mmproj
            .as_ref()
            .unwrap()
            .tensors
            .iter()
            .chain(&hb.tensors)
            .map(|e| e.offset + e.length)
            .max()
            .unwrap();
        let mut cursor = t_blob_end;
        for (src, dst) in hh.header().tensors.iter().zip(&spec.tensors) {
            assert_eq!(src.name, dst.name);
            assert_eq!(src.checksum_xxh64, dst.checksum_xxh64);
            assert_eq!(src.length, dst.length);
            assert_eq!(
                dst.offset,
                align_up(cursor, ha.alignment.align_for(ComputeRegion::Gpu)),
                "{}",
                dst.name
            );
            assert_eq!(
                bytes_of(&head, hh.blob_offset(), src),
                bytes_of(&out, after.blob_offset(), dst),
                "{}",
                dst.name
            );
            after.verify_tensor(&dst.name).unwrap();
            cursor = dst.offset + dst.length;
        }
        assert_eq!(
            after.blob_offset() + cursor,
            s.out_bytes,
            "nothing follows the head"
        );
    }

    /// A head converted with `--quantize-drafter`: its entries carry scale
    /// and bias sub-regions inside their payloads, and they are embedded as
    /// they are — the entry equal apart from its new offset, the sub-region
    /// offsets still relative to it, the bytes the sidecar's. Its
    /// accelerator-region tensors keep the page alignment the head's own
    /// converter gave them, though the (older) target asks only 64 B.
    #[test]
    fn embeds_a_quantized_head_with_its_sub_regions() {
        let dir = tempfile::tempdir().unwrap();
        let (target, _) = fixtures(dir.path());
        let head = dir.path().join("q8head.base");
        let mut hdr = header(
            "gemma4-mtp",
            serde_json::json!({"hidden_size": 4, "backbone_hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 4}),
        );
        hdr.alignment.accel_align_log2 = 14;
        let mut w = BaseWriter::create(&head, hdr).unwrap();
        // 64 q8 weights, then 8 bytes of scales and 8 of biases, one payload.
        let q8 = |name: &str, first: u8| {
            let mut p = TensorPayload {
                entry: entry(name, vec![16, 4]),
                data: (first..first + 80).collect(),
            };
            p.entry.dtype = TensorDtype::BaseQ8;
            p.entry.compute_region = ComputeRegion::Accelerator;
            p.entry.scale_offset = Some(64);
            p.entry.scale_length = Some(8);
            p.entry.bias_offset = Some(72);
            p.entry.bias_length = Some(8);
            p.entry.group_size = Some(128);
            p
        };
        w.add_tensor(q8("output.weight", 0));
        w.add_tensor(payload("output_norm.weight", 8, 0xC2));
        w.add_tensor(q8("layers.0.mlp.down_proj.weight", 100));
        w.finish().unwrap();

        let out = dir.path().join("out.base");
        let s = add_speculator(&target, &head, &out, Some("q8")).unwrap();
        assert_eq!((s.n_head_tensors, s.head_bytes), (3, 168));
        let hh = BaseReader::open(&head).unwrap();
        let after = BaseReader::open(&out).unwrap();
        let spec = after.header().speculator.as_ref().unwrap();
        assert_eq!(spec.config["head_version"], serde_json::json!("q8"));
        for (src, dst) in hh.header().tensors.iter().zip(&spec.tensors) {
            let mut moved = src.clone();
            moved.offset = dst.offset;
            assert_eq!(
                serde_json::to_string(&moved).unwrap(),
                serde_json::to_string(dst).unwrap(),
                "{}: the entry moves whole",
                src.name
            );
            assert_eq!(
                bytes_of(&head, hh.blob_offset(), src),
                bytes_of(&out, after.blob_offset(), dst),
                "{}",
                dst.name
            );
            after.verify_tensor(&dst.name).unwrap();
        }
        let q = &spec.tensors[0];
        assert_eq!(
            (
                q.dtype,
                q.scale_offset,
                q.scale_length,
                q.bias_offset,
                q.bias_length
            ),
            (TensorDtype::BaseQ8, Some(64), Some(8), Some(72), Some(8))
        );
        let bytes = bytes_of(&out, after.blob_offset(), q);
        assert_eq!(
            &bytes[64..72],
            &(64u8..72).collect::<Vec<_>>()[..],
            "the scales"
        );
        assert_eq!(
            &bytes[72..80],
            &(72u8..80).collect::<Vec<_>>()[..],
            "the biases"
        );
        // The target's rule (64 B) would have put the second q8 entry right
        // behind the norm; the head's (a page) wins, and so does the first
        // entry's page start.
        assert_eq!(after.header().alignment.accel_align_log2, 6);
        assert_eq!(
            spec.tensors[0].offset % 16384,
            0,
            "the first entry starts a GPU page"
        );
        assert_eq!(
            spec.tensors[2].offset % 16384,
            0,
            "an accelerator tensor keeps the head's page alignment"
        );
        assert_eq!(
            spec.tensors[1].offset,
            spec.tensors[0].offset + 16384,
            "the norm follows on the next page (gpu)"
        );
    }

    #[test]
    fn refuses_what_it_cannot_embed_into_or_from() {
        let dir = tempfile::tempdir().unwrap();
        let (target, head) = fixtures(dir.path());
        let out = dir.path().join("out.base");

        // Output in the way.
        std::fs::write(&out, b"x").unwrap();
        assert!(add_speculator(&target, &head, &out, None)
            .unwrap_err()
            .to_string()
            .contains("refusing to overwrite"));
        std::fs::remove_file(&out).unwrap();

        // A head whose width is not the target's, and a target that is not
        // the head's backbone.
        let other = dir.path().join("other.base");
        let mut w = BaseWriter::create(
            &other,
            header(
                "gemma4-mtp",
                serde_json::json!({"hidden_size": 4, "backbone_hidden_size": 12, "vocab_size": 16}),
            ),
        )
        .unwrap();
        w.add_tensor(payload("output.weight", 64, 1));
        w.finish().unwrap();
        assert!(add_speculator(&target, &other, &out, None)
            .unwrap_err()
            .to_string()
            .contains("hidden width"));
        assert!(add_speculator(&head, &head, &out, None)
            .unwrap_err()
            .to_string()
            .contains("but the head drafts for"));

        // A quantized head whose scales reach past the entry's payload: the
        // runtime would refuse the section, so the embed refuses the head.
        let q = dir.path().join("q.base");
        let mut w = BaseWriter::create(
            &q,
            header(
                "gemma4-mtp",
                serde_json::json!({"hidden_size": 4, "backbone_hidden_size": 8, "vocab_size": 16}),
            ),
        )
        .unwrap();
        let mut p = payload("output.weight", 64, 1);
        p.entry.dtype = TensorDtype::BaseQ4;
        p.entry.scale_offset = Some(60);
        p.entry.scale_length = Some(8);
        w.add_tensor(p);
        w.finish().unwrap();
        assert!(add_speculator(&target, &q, &out, None)
            .unwrap_err()
            .to_string()
            .contains("scales [60, 68) outside its payload of 64 bytes"));

        // Embedding twice.
        add_speculator(&target, &head, &out, None).unwrap();
        let twice = dir.path().join("twice.base");
        assert!(add_speculator(&out, &head, &twice, None)
            .unwrap_err()
            .to_string()
            .contains("already carries"));
    }
}
