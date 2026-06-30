use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod hub;

/// Listed in `basert --help` so the dispatched runtime tools are discoverable
/// (clap's external-subcommand catch-all is otherwise invisible in help).
const RUNTIME_TOOLS_HELP: &str = "\
Runtime tools (forwarded to the matching `basert-<cmd>` binary):
  serve       Start the OpenAI-compatible HTTP server
  chat        Interactive chat
  complete    One-shot text completion
  bench       Throughput benchmark
  profile     Profile prefill/decode timing
  transcribe  Audio transcription

Run `basert <tool> --help` for a tool's own options.";

/// The `basert` CLI: the model hub (pull/list from HuggingFace), the offline
/// GGUF / MLX-safetensors / HF-safetensors → `.base` converter, and a
/// launcher for the runtime tools (`basert serve`, `basert chat`, …).
#[derive(Parser, Debug)]
#[command(name = "basert", version, about, after_help = RUNTIME_TOOLS_HELP)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Convert a source model to `.base`.
    Convert(ConvertArgs),
    /// Sign an existing unsigned `.base` file.
    Sign(SignArgs),
    /// Verify a signed `.base` file.
    Verify(VerifyArgs),
    /// Inspect a `.base` file (summary of header + tensors + slots).
    Inspect(InspectArgs),
    /// Generate an ed25519 keypair for signing.
    Keygen(KeygenArgs),
    /// Download a model into the local hub cache (pre-converted `.base` from
    /// the BaseRT catalog, or an HF repo converted on the fly).
    Pull(PullArgs),
    /// List models in the local hub cache (and, with `--remote`, the catalog).
    List(ListArgs),
    /// Runtime tools — `serve`, `chat`, `complete`, `bench`, … — forwarded to
    /// the matching engine binary (`basert-<cmd>`).
    #[command(external_subcommand)]
    External(Vec<String>),
}

#[derive(Parser, Debug)]
struct ConvertArgs {
    /// Input model path. With `--synthetic`, this is a bundle name only —
    /// no file is read; a deterministic dummy model is generated.
    input: PathBuf,

    /// Output `.base` file. Defaults to <input>.base.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Target quant scheme.
    #[arg(long, value_enum, default_value_t = TargetScheme::BaseQ4)]
    target: TargetScheme,

    /// Calibration text file (UTF-8). Required for AWQ.
    #[arg(long)]
    calib: Option<PathBuf>,

    /// Number of calibration tokens.
    #[arg(long, default_value_t = 512)]
    calib_tokens: u32,

    /// AWQ calibration mode.
    #[arg(long, value_enum, default_value_t = AwqMode::Full)]
    awq_mode: AwqMode,

    /// Generate a synthetic end-to-end bundle instead of reading a real
    /// model. Used to exercise the full pipeline (quant → write → sign)
    /// in CI without depending on HuggingFace downloads.
    #[arg(long)]
    synthetic: bool,

    /// Canonical-quant profile JSON (e.g.
    /// `tools/base-convert/profiles/dense-q4mix.json`). When set,
    /// per-tensor quant decisions come from the profile rules; the
    /// `--target` flag becomes the fallback for tensors the profile's
    /// catch-all `**.weight` rule should otherwise have covered. Sets
    /// `Header.target_backend = metal` and records the profile name in
    /// `Header.quant_profile`.
    #[arg(long)]
    profile: Option<PathBuf>,

    /// AWQ activation-stats sidecar JSON (produced by baseRT
    /// calibration mode). When set, every tensor whose profile rule
    /// dispatches to a canonical `base_qN` dtype runs AWQ search +
    /// rotation before the RTN pack. Tensors whose profile says
    /// `dtype: bf16` are unaffected.
    #[arg(long)]
    awq_profile: Option<PathBuf>,

    /// Override the spec-mandated "fp16/bf16/fp32 source only" check
    /// to allow `--profile` on already-quantized inputs (GGUF
    /// Q4_K_M / Q5_0 / Q8_0, MLX 4-bit/8-bit). Quant-from-quant
    /// compounds error vs converting from the original fp16; this
    /// flag is the explicit acknowledgment for users who don't have
    /// the fp16 checkpoint locally.
    #[arg(long)]
    allow_quant_from_quant: bool,
}

#[derive(Parser, Debug)]
struct SignArgs {
    /// Unsigned input `.base` file.
    input: PathBuf,
    /// Signed output `.base` file.
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// Path to a 32-byte ed25519 secret key.
    #[arg(long)]
    key: PathBuf,
    /// Human-readable key identifier stored in the signed file.
    #[arg(long, default_value = "baseRT-default")]
    key_id: String,
}

#[derive(Parser, Debug)]
struct VerifyArgs {
    /// Input `.base` file.
    input: PathBuf,
    /// Path to a 32-byte ed25519 public (verifying) key.
    #[arg(long)]
    pubkey: PathBuf,
}

#[derive(Parser, Debug)]
struct KeygenArgs {
    /// Output directory. Files written: <name>.secret (32 B) and
    /// <name>.pub (32 B).
    #[arg(short = 'o', long)]
    out: PathBuf,
    /// Base name for the key files.
    #[arg(long, default_value = "baseRT-key")]
    name: String,
}

#[derive(Parser, Debug)]
struct InspectArgs {
    /// Input `.base` file.
    input: PathBuf,
    /// Also verify per-tensor xxhash64 (slow for large models).
    #[arg(long)]
    verify_checksums: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TargetScheme {
    BaseQ2,
    BaseQ3,
    BaseQ4,
    BaseQ5,
    BaseQ6,
    BaseQ8,
    Bf16,
    Mxfp4,
    Nvfp4,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AwqMode {
    Full,
    Lite,
    None,
}

#[derive(Parser, Debug)]
struct PullArgs {
    /// Model id: `basecompute/<name>` (pre-converted, from the catalog) or
    /// `org/model` (a raw HF repo, downloaded + converted locally).
    id: String,
    /// Override the auto-selected quant profile (path to a profile JSON).
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Target quant scheme for convert-on-pull when no profile applies.
    #[arg(long, value_enum, default_value_t = TargetScheme::BaseQ4)]
    target: TargetScheme,
    /// HF revision / branch / tag.
    #[arg(long, default_value = "main")]
    revision: String,
    /// Re-download / re-convert even if already cached.
    #[arg(long)]
    force: bool,
    /// Resolve and print the plan without downloading anything.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Parser, Debug)]
struct ListArgs {
    /// Also list catalog models that aren't installed yet.
    #[arg(long)]
    remote: bool,
    /// Emit JSON instead of a table.
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Convert(a) => cmd_convert(a),
        Cmd::Sign(a) => cmd_sign(a),
        Cmd::Verify(a) => cmd_verify(a),
        Cmd::Inspect(a) => cmd_inspect(a),
        Cmd::Keygen(a) => cmd_keygen(a),
        Cmd::Pull(a) => hub::cmd_pull(a),
        Cmd::List(a) => hub::cmd_list(a),
        Cmd::External(argv) => hub::dispatch_external(argv),
    }
}

fn cmd_keygen(args: KeygenArgs) -> Result<()> {
    use ed25519_dalek::SigningKey;
    use rand_core::OsRng;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {:?}", args.out))?;
    let mut rng = OsRng;
    let sk = SigningKey::generate(&mut rng);
    let vk = sk.verifying_key();
    let sec_path = args.out.join(format!("{}.secret", args.name));
    let pub_path = args.out.join(format!("{}.pub", args.name));
    std::fs::write(&sec_path, sk.to_bytes())?;
    std::fs::write(&pub_path, vk.to_bytes())?;
    eprintln!("secret: {}", sec_path.display());
    eprintln!("public: {}", pub_path.display());
    Ok(())
}

fn cmd_convert(args: ConvertArgs) -> Result<()> {
    eprintln!("base-convert v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  input:   {}", args.input.display());
    let output = args.output.clone().unwrap_or_else(|| {
        let mut p = args.input.clone();
        p.set_extension("base");
        p
    });
    eprintln!("  output:  {}", output.display());

    let ctx = QuantContext::from_args(&args)?;
    if let Some(name) = ctx.profile_name() {
        eprintln!("  profile: {}", name);
    } else {
        eprintln!("  target:  {:?}", args.target);
    }
    if let Some(awq) = ctx.awq_profile.as_ref() {
        eprintln!("  awq:     sidecar tokens={}", awq.calib_tokens);
    }

    if args.synthetic {
        return convert_synthetic_with_ctx(&output, &ctx);
    }

    // Detect source format: GGUF / HF-safetensors / MLX-safetensors.
    use base_readers::SourceFormat;
    let fmt = base_readers::detect_format(&args.input)
        .with_context(|| format!("detecting format for {:?}", args.input))?;

    match fmt {
        SourceFormat::Gguf => convert_gguf(&args.input, &output, &ctx),
        SourceFormat::HfSafetensors => convert_hf(&args.input, &output, &ctx),
        SourceFormat::MlxSafetensors => convert_mlx(&args.input, &output, &ctx),
    }
}

/// Real-model conversion: read a GGUF, dequant per-tensor to f32,
/// remap tensor names to canonical .base convention, re-quantize to the
/// target scheme, write the .base file.
///
/// Plain round-trip-through-f32 then re-quantize (no AWQ on this path).
fn convert_gguf(
    input: &std::path::Path,
    output: &std::path::Path,
    ctx: &QuantContext,
) -> Result<()> {
    use base_arch::source_mapper_for_gguf;
    use base_format::{
        AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, LayerKind,
        LayerDescriptor, LayerPrecision, ModelConfig, QuantScheme, SourceInfo, TargetBackend, TensorDtype,
        TensorFlags, TensorPayload, TokenizerBlob,
    };
    use base_readers::gguf::{dequant_to_f32, ggml_type_name, GgmlType, GgufFile};
    let target = ctx.target;

    let gguf = GgufFile::open(input)
        .with_context(|| format!("opening GGUF {:?}", input))?;
    let arch = gguf
        .arch()
        .ok_or_else(|| anyhow::anyhow!("GGUF missing general.architecture"))?;
    eprintln!("  arch:    {}", arch);

    // Per CANONICAL_QUANT_SPEC.md: profile-driven canonical-quant
    // requires fp16/bf16/fp32 source. A GGUF with quantized weight
    // tensors (Q4_0/Q5_0/Q4_K/Q8_0/...) is already-quantized;
    // dequant→requant via the canonical profile compounds error.
    // Reject by default; users explicitly opt in via
    // `--allow-quant-from-quant`. F16/BF16/F32 only-tensor GGUFs
    // (rare; usually only norms are non-quant) pass through silently.
    if ctx.profile.is_some() && !ctx.allow_quant_from_quant {
        for tensor in gguf.tensors.iter() {
            if !matches!(
                tensor.ggml_type,
                GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
            ) {
                bail!(
                    "GGUF source contains quantized tensor `{}` (ggml_type={}). \
                     Canonical-quant requires fp16/bf16/fp32 source. \
                     Re-fetch the bf16 HF safetensors checkpoint, or pass \
                     --allow-quant-from-quant to accept the compounded quant error.",
                    tensor.name,
                    ggml_type_name(tensor.ggml_type)
                );
            }
        }
    }

    let mapper = source_mapper_for_gguf(arch)
        .ok_or_else(|| anyhow::anyhow!("arch {:?} not supported yet", arch))?;
    let config = mapper.config_from_gguf(&gguf.metadata)?;

    eprintln!(
        "  config:  hidden={}, layers={}, heads={}/{}, ffn={}, vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_kv_heads,
        config.intermediate_size,
        config.vocab_size
    );

    let quant_scheme = match target {
        TargetScheme::BaseQ2 => QuantScheme::BaseQ2,
        TargetScheme::BaseQ3 => QuantScheme::BaseQ3,
        TargetScheme::BaseQ4 => QuantScheme::BaseQ4,
        TargetScheme::BaseQ5 => QuantScheme::BaseQ5,
        TargetScheme::BaseQ6 => QuantScheme::BaseQ6,
        TargetScheme::BaseQ8 => QuantScheme::BaseQ8,
        TargetScheme::Bf16 => QuantScheme::Bf16,
        TargetScheme::Mxfp4 => QuantScheme::Mxfp4,
        TargetScheme::Nvfp4 => QuantScheme::Nvfp4,
    };

    let mut header = Header {
        schema: 1,
        arch: mapper.canonical_arch().to_string(),
        quant_scheme,
        min_hw: "apple_m1".to_string(),
        created: chrono_now(),
        base_rt_version: env!("CARGO_PKG_VERSION").to_string(),
        source: SourceInfo {
            format: "gguf".to_string(),
            sha256: compute_sha256_streaming(input)?,
            filename: input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
        },
        tokenizer: TokenizerBlob {
            fields: base_arch::tokenizer::extract_from_gguf(&gguf.metadata),
        },
        config: ModelConfig {
            fields: config.to_config_map(),
        },
        target_backend: TargetBackend::Metal,
        quant_profile: ctx.profile_name().unwrap_or("").to_string(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::QUANTIZED,
        layers: (0..config.num_hidden_layers)
            .map(|_| LayerDescriptor {
                kind: LayerKind::AttentionGqa,
                moe_n_experts: 0,
                moe_n_active: 0,
                shared_attn_layer: None,
                compute_hint: Some(ComputeRegion::Accelerator),
                precision: LayerPrecision::default(),
            })
            .collect(),
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
    };

    if config.tie_word_embeddings {
        header.flags |= HeaderFlags::TIED_EMBEDDINGS;
    }
    // Heuristics: detect MoE / SSM / hybrid structure from tensor names.
    let has_moe = gguf
        .tensors
        .iter()
        .any(|t| t.name.contains("_exps") || t.name.contains("_shexp"));
    let has_ssm = gguf.tensors.iter().any(|t| t.name.contains("ssm_"));
    let has_attn = gguf
        .tensors
        .iter()
        .any(|t| t.name.contains("attn_q") || t.name.contains("attn_qkv"));
    if has_moe {
        header.flags |= HeaderFlags::HAS_MOE;
    }
    if has_ssm {
        header.flags |= HeaderFlags::HAS_SSM;
        if has_attn {
            header.flags |= HeaderFlags::HAS_HYBRID;
        }
    }

    let mut writer = BaseWriter::create(output, header).context("create writer")?;

    // Walk tensors in GGUF order (which roughly matches layer-major).
    // Norms → CPU region, large weights → Accelerator region, embeddings
    // → GPU region.
    let mut dropped = 0usize;
    let mut kept = 0usize;
    for info in gguf.tensors.iter() {
        let Some(canonical) = mapper.map_tensor_name(&info.name) else {
            dropped += 1;
            continue;
        };
        kept += 1;

        let bytes = gguf
            .tensor_bytes(info)
            .with_context(|| format!("reading {:?}", info.name))?;

        let f32s = dequant_to_f32(info, bytes).with_context(|| {
            format!(
                "dequant {:?} type={}",
                info.name,
                ggml_type_name(info.ggml_type)
            )
        })?;

        // Route by tensor role, not by GGUF type. Norms / rope_freqs
        // → CPU region f32. Everything else → Accelerator region, which
        // gets quantized to the target scheme unless it's 1D (norm-like).
        //
        // SSM A-matrix (state transition) MUST be f32 in CPU region —
        // quantizing it produces NaN after ~100 recurrent steps.
        let is_ssm_a = canonical == "ssm.a_log"
            || canonical.ends_with(".ssm.a_log")
            || info.name.ends_with(".ssm_a");
        let is_ssm_sensitive = is_ssm_a
            || canonical.ends_with(".ssm.dt_bias")
            || canonical.ends_with(".ssm.d");
        // SSM A-matrix (and adjacent SSM scalars) MUST stay f32 in CPU
        // region — quantizing them produces NaN after ~100 recurrent
        // steps. Regular 1-D norm weights and the embed/lm_head pair
        // get pre-converted to f16 in GPU region so the runtime can
        // hand out zero-copy views without a load-time conversion +
        // duplicate allocation (the source of the Phase-2/3 .base 3x
        // memory bloat).
        let is_norm_like = info.shape.len() == 1;
        let is_embedding = canonical == "embed_tokens.weight" || canonical == "lm_head.weight";

        let (entry, data) = if is_ssm_sensitive {
            let mut flags = TensorFlags::empty();
            if is_ssm_a {
                flags |= TensorFlags::SSM_A_MATRIX;
            }
            let mut entry = base_format::TensorEntry {
                name: canonical,
                dtype: TensorDtype::F32,
                shape: info.shape.clone(),
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
                residency: Some(base_format::ResidencyHint::Hot),
                compute_region: ComputeRegion::Cpu,
                scale_dtype: None,
                symmetric: false,
                flags,
                checksum_xxh64: None,
            source_ggml_type: None,
};
            let data: Vec<u8> = f32s.iter().flat_map(|f| f.to_le_bytes()).collect();
            entry.length = data.len() as u64;
            (entry, data)
        } else if is_norm_like {
            // 1-D norms (and biases caught by the same shape check) are
            // always emitted at f16. A profile's catch-all `**.weight`
            // rule typically targets a quant bit-width; quantizing a
            // per-channel norm-gain wrecks the model. Override the
            // profile here so a profile that omits explicit norm
            // patterns still produces a working bundle.
            let (bytes, dtype) = (
                f32s.iter()
                    .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                    .collect::<Vec<u8>>(),
                TensorDtype::F16,
            );
            let entry = base_format::TensorEntry {
                name: canonical,
                dtype,
                shape: info.shape.clone(),
                offset: 0,
                length: bytes.len() as u64,
                scale_offset: None,
                scale_length: None,
                bias_offset: None,
                bias_length: None,
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: None,
                layout: None,
                residency: Some(base_format::ResidencyHint::Hot),
                compute_region: ComputeRegion::Gpu,
                scale_dtype: None,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            (entry, bytes)
        } else if is_embedding {
            // Embed/lm_head: pack at the target quant scheme. Earlier
            // revisions left these at f16/bf16 to "skip runtime conversion",
            // but a 525 MB f16 embed table burns L2/L3 cache during decode
            // (Llama-3.2-1B at MLX-direct ships embed at 4-bit ≈ 131 MB —
            // cache-friendly). Quantizing embed matches the source format
            // and is what the embedding_lookup_q4 kernel expects.
            let (packed, dtype) = if ctx.profile.is_some() {
                let in_features = info.shape.last().copied().map(|d| d as usize);
                ctx.pack_tensor(&canonical, &f32s, in_features)?
            } else {
                pack_for_target(&f32s, target)?
            };
            let mut data = Vec::with_capacity(
                packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
            );
            data.extend_from_slice(&packed.packed_weights);
            let scale_off = data.len() as u64;
            data.extend_from_slice(&packed.scales);
            let bias_off = data.len() as u64;
            data.extend_from_slice(&packed.biases);
            let mut entry = base_format::TensorEntry {
                name: canonical,
                dtype,
                shape: info.shape.clone(),
                offset: 0,
                length: 0,
                scale_offset: if !packed.scales.is_empty() {
                    Some(scale_off)
                } else {
                    None
                },
                scale_length: if !packed.scales.is_empty() {
                    Some(packed.scales.len() as u64)
                } else {
                    None
                },
                bias_offset: if !packed.biases.is_empty() {
                    Some(bias_off)
                } else {
                    None
                },
                bias_length: if !packed.biases.is_empty() {
                    Some(packed.biases.len() as u64)
                } else {
                    None
                },
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: if packed.group_size > 0 {
                    Some(packed.group_size)
                } else {
                    None
                },
                layout: None,
                residency: Some(base_format::ResidencyHint::Hot),
                compute_region: ComputeRegion::Gpu,
                scale_dtype: packed.scale_dtype,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            entry.length = data.len() as u64;
            (entry, data)
        } else {
            // Quantize to target scheme, Accelerator region.
            let (packed, dtype) = if ctx.profile.is_some() {
                let in_features = info.shape.last().copied().map(|d| d as usize);
                ctx.pack_tensor(&canonical, &f32s, in_features)?
            } else {
                pack_for_target(&f32s, target)?
            };
            let mut data = Vec::with_capacity(
                packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
            );
            data.extend_from_slice(&packed.packed_weights);
            let scale_off = data.len() as u64;
            data.extend_from_slice(&packed.scales);
            let bias_off = data.len() as u64;
            data.extend_from_slice(&packed.biases);

            let mut entry = base_format::TensorEntry {
                name: canonical,
                dtype,
                shape: info.shape.clone(),
                offset: 0,
                length: packed.packed_weights.len() as u64,
                scale_offset: if !packed.scales.is_empty() {
                    Some(scale_off)
                } else {
                    None
                },
                scale_length: if !packed.scales.is_empty() {
                    Some(packed.scales.len() as u64)
                } else {
                    None
                },
                bias_offset: if !packed.biases.is_empty() {
                    Some(bias_off)
                } else {
                    None
                },
                bias_length: if !packed.biases.is_empty() {
                    Some(packed.biases.len() as u64)
                } else {
                    None
                },
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: if packed.group_size > 0 {
                    Some(packed.group_size)
                } else {
                    None
                },
                layout: None,
                residency: Some(base_format::ResidencyHint::Warm),
                compute_region: ComputeRegion::Accelerator,
                scale_dtype: packed.scale_dtype,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            let _ = entry.length; // length will be overwritten by writer
            entry.length = data.len() as u64;
            (entry, data)
        };

        writer.add_tensor(TensorPayload { entry, data });
    }
    eprintln!("  mapped:  {} tensors kept, {} dropped", kept, dropped);

    writer.finish().context("writing bundle")?;

    // Sanity-check: reopen and verify zero-copy alignment.
    let reader = BaseReader::open(output).context("reopen for verification")?;
    for t in reader.header().tensors.iter() {
        if t.compute_region == ComputeRegion::Gpu
            && !reader.tensor_is_zero_copy_eligible(&t.name)?
        {
            bail!("tensor {:?} failed zero-copy alignment check", t.name);
        }
    }
    eprintln!(
        "  wrote {} tensors ({} MB)",
        reader.header().tensors.len(),
        std::fs::metadata(output)?.len() / (1024 * 1024)
    );
    Ok(())
}

/// Convert from an HF safetensors directory.
fn convert_hf(
    input: &std::path::Path,
    output: &std::path::Path,
    ctx: &QuantContext,
) -> Result<()> {
    use base_arch::hf_mapper_for_model_type;
    use base_readers::hf::HfDir;
    let hf = HfDir::open(input)?;
    let model_type = hf
        .model_type()
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?;
    eprintln!("  arch:    {}", model_type);
    let mapper = hf_mapper_for_model_type(model_type)
        .ok_or_else(|| anyhow::anyhow!("HF model_type {:?} not supported yet", model_type))?;
    let config = mapper.config_from_hf(&hf.config)?;
    eprintln!(
        "  config:  hidden={}, layers={}, heads={}/{}, ffn={}, vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_kv_heads,
        config.intermediate_size,
        config.vocab_size
    );
    let provider = HfTensorProvider { hf: &hf };
    let mmproj_cfg = mmproj_config_from_hf(&hf);
    convert_generic(
        input,
        output,
        ctx,
        mapper.canonical_arch(),
        "hf_safetensors",
        config,
        &provider,
        hf.tensor_names().map(|s| s.to_string()).collect(),
        &tokenizer_from_hf(&hf),
        mmproj_cfg,
        &|n| mapper.norm_shift(n),
    )
}

/// Convert from an MLX-quantized safetensors directory.
fn convert_mlx(
    input: &std::path::Path,
    output: &std::path::Path,
    ctx: &QuantContext,
) -> Result<()> {
    use base_arch::hf_mapper_for_model_type;
    use base_readers::mlx::MlxDir;
    let mlx = MlxDir::open(input)?;
    let model_type = mlx
        .hf
        .model_type()
        .ok_or_else(|| anyhow::anyhow!("config.json missing model_type"))?;
    eprintln!(
        "  arch:    {} (MLX-quantized bits={} group_size={})",
        model_type, mlx.quant.bits, mlx.quant.group_size
    );
    // Per `CANONICAL_QUANT_SPEC.md`: profile-driven canonical-quant
    // requires fp16/bf16/fp32 source. MLX 4-bit/8-bit checkpoints
    // are already-quantized; quant-from-quant compounds error and
    // contradicts the bit-budget contract. Reject by default; users
    // who explicitly accept the lossy path opt in via
    // `--allow-quant-from-quant`.
    if ctx.profile.is_some() && !ctx.allow_quant_from_quant {
        bail!(
            "MLX-quantized source ({}-bit, gs={}) is not a valid input for --profile (per \
             CANONICAL_QUANT_SPEC.md, canonical-quant requires fp16/bf16/fp32 source). \
             Re-fetch the fp16/bf16 HF checkpoint, or pass --allow-quant-from-quant to \
             accept the compounded quant error.",
            mlx.quant.bits, mlx.quant.group_size
        );
    }
    let mapper = hf_mapper_for_model_type(model_type)
        .ok_or_else(|| anyhow::anyhow!("model_type {:?} not supported yet", model_type))?;
    let config = mapper.config_from_hf(&mlx.hf.config)?;
    eprintln!(
        "  config:  hidden={}, layers={}, heads={}/{}, ffn={}, vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_kv_heads,
        config.intermediate_size,
        config.vocab_size
    );
    let names: Vec<String> = mlx
        .hf
        .tensor_names()
        .filter(|n| !n.ends_with(".scales") && !n.ends_with(".biases"))
        .map(|s| s.to_string())
        .collect();
    let provider = MlxTensorProvider { mlx: &mlx };
    let mmproj_cfg = mmproj_config_from_hf(&mlx.hf);
    convert_generic(
        input,
        output,
        ctx,
        mapper.canonical_arch(),
        "mlx_safetensors",
        config,
        &provider,
        names,
        &tokenizer_from_hf(&mlx.hf),
        mmproj_cfg,
        &|n| mapper.norm_shift(n),
    )
}

fn tokenizer_from_hf(hf: &base_readers::hf::HfDir) -> std::collections::BTreeMap<String, serde_json::Value> {
    use serde_json::json;
    let mut m = std::collections::BTreeMap::new();
    m.insert("tokenizer_type".into(), json!("hf"));
    if let Some(tj) = &hf.tokenizer_json {
        m.insert("tokenizer.json".into(), tj.clone());
    }
    if let Some(tc) = &hf.tokenizer_config {
        m.insert("tokenizer_config.json".into(), tc.clone());
    }
    // Gemma 4 (and some other recent models) ship the chat template as a
    // separate `chat_template.jinja` file rather than inside
    // `tokenizer_config.json`. Mirror it into the runtime-visible
    // `tokenizer.chat_template` slot the BaseWeightStore already reads
    // (`base_weight_store.cpp:459`). Without this the runtime falls
    // back to its hardcoded chat template, which on Gemma 4 sends
    // `<start_of_turn>` markers the model has never seen during
    // training and silently corrupts long generations.
    if let Some(jinja) = &hf.chat_template_jinja {
        m.insert("tokenizer.chat_template".into(), json!(jinja));
    }
    m
}

/// Extract the mmproj config block from an HF directory's `config.json`
/// (and `processor_config.json`, when present). Returns an empty map for
/// text-only models. Captures the bits the runtime needs to drive the
/// vision / audio prefill paths: tower configs, multimodal token IDs,
/// soft-token counts, and image-pooling parameters.
fn mmproj_config_from_hf(hf: &base_readers::hf::HfDir) -> std::collections::BTreeMap<String, serde_json::Value> {
    let mut m = std::collections::BTreeMap::new();
    let cfg = &hf.config;

    // Multimodal sub-configs — passed through verbatim. Runtime parses
    // hidden_size / num_hidden_layers / patch_size / etc. from these.
    for key in [
        "vision_config",
        "audio_config",
    ] {
        if let Some(v) = cfg.get(key) {
            m.insert(key.into(), v.clone());
        }
    }

    // Multimodal token IDs and soft-token counts (Gemma 4 family).
    for key in [
        "image_token_id",
        "boi_token_id",
        "eoi_token_id",
        "audio_token_id",
        "boa_token_id",
        "eoa_token_id",
        "vision_soft_tokens_per_image",
    ] {
        if let Some(v) = cfg.get(key) {
            m.insert(key.into(), v.clone());
        }
    }

    // processor_config.json — image_processor.pooling_kernel_size and
    // audio sequence length live here, not in config.json. Read it from
    // the model_dir; tolerate absence (older checkpoints may omit it).
    let processor_path = hf.model_dir.join("processor_config.json");
    if let Ok(bytes) = std::fs::read(&processor_path) {
        if let Ok(pcfg) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            for key in ["audio_seq_length", "audio_ms_per_token", "image_seq_length"] {
                if let Some(v) = pcfg.get(key) {
                    m.insert(key.into(), v.clone());
                }
            }
            // Pooling kernel size lives nested under image_processor.
            if let Some(ip) = pcfg.get("image_processor") {
                if let Some(pk) = ip.get("pooling_kernel_size") {
                    m.insert("pooling_kernel_size".into(), pk.clone());
                }
                if let Some(ps) = ip.get("patch_size") {
                    // Same key already present in vision_config; expose
                    // the processor-level one as a top-level fallback for
                    // simpler runtime parsing.
                    m.insert("processor_patch_size".into(), ps.clone());
                }
            }
        }
    }

    m
}

/// Abstraction over GGUF vs HF vs MLX so the shared convert logic
/// doesn't care where bytes come from. All sources are dequantized to
/// f32 and re-packed via the profile-driven canonical path.
trait TensorProvider {
    fn source_shape(&self, name: &str) -> Result<Vec<u64>>;
    fn to_f32(&self, name: &str) -> Result<Vec<f32>>;
}

struct HfTensorProvider<'a> {
    hf: &'a base_readers::hf::HfDir,
}
impl<'a> TensorProvider for HfTensorProvider<'a> {
    fn source_shape(&self, name: &str) -> Result<Vec<u64>> {
        self.hf
            .tensor_info(name)
            .map(|t| t.shape.clone())
            .ok_or_else(|| anyhow::anyhow!("tensor {name} missing"))
    }
    fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.hf.tensor_to_f32(name)
    }
}

struct MlxTensorProvider<'a> {
    mlx: &'a base_readers::mlx::MlxDir,
}
impl<'a> TensorProvider for MlxTensorProvider<'a> {
    fn source_shape(&self, name: &str) -> Result<Vec<u64>> {
        if let Some(s) = self.mlx.unpacked_shape(name) {
            return Ok(s);
        }
        self.mlx
            .hf
            .tensor_info(name)
            .map(|t| t.shape.clone())
            .ok_or_else(|| anyhow::anyhow!("tensor {name} missing"))
    }
    fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.mlx.tensor_to_f32(name)
    }
}

#[allow(clippy::too_many_arguments)]
fn convert_generic(
    input: &std::path::Path,
    output: &std::path::Path,
    ctx: &QuantContext,
    canonical_arch: &str,
    source_format: &str,
    config: base_arch::ArchConfig,
    provider: &dyn TensorProvider,
    source_names: Vec<String>,
    tokenizer_fields: &std::collections::BTreeMap<String, serde_json::Value>,
    mmproj_config: std::collections::BTreeMap<String, serde_json::Value>,
    norm_shift: &dyn Fn(&str) -> f32,
) -> Result<()> {
    use base_format::{
        AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, LayerKind,
        LayerDescriptor, LayerPrecision, ModelConfig, QuantScheme, SourceInfo, TargetBackend, TensorDtype,
        TensorFlags, TensorPayload, TokenizerBlob,
    };
    let target = ctx.target;
    let quant_scheme = match target {
        TargetScheme::BaseQ2 => QuantScheme::BaseQ2,
        TargetScheme::BaseQ3 => QuantScheme::BaseQ3,
        TargetScheme::BaseQ4 => QuantScheme::BaseQ4,
        TargetScheme::BaseQ5 => QuantScheme::BaseQ5,
        TargetScheme::BaseQ6 => QuantScheme::BaseQ6,
        TargetScheme::BaseQ8 => QuantScheme::BaseQ8,
        TargetScheme::Bf16 => QuantScheme::Bf16,
        TargetScheme::Mxfp4 => QuantScheme::Mxfp4,
        TargetScheme::Nvfp4 => QuantScheme::Nvfp4,
    };

    // Map source names → canonical. Use the same llama-style map that
    // GGUF uses for blk.N.* tensors, plus a HF-style pass-through for
    // `model.layers.N.*` already-canonical names. Multimodal towers
    // get tagged for the mmproj sub-bundle so the runtime can choose
    // whether to materialize them.
    let mut mapped: Vec<(String, String)> = Vec::new();
    let mut mmproj_mapped: Vec<(String, String)> = Vec::new();
    let mut dropped = 0usize;
    for n in &source_names {
        match to_canonical_name(n, canonical_arch) {
            Some(Canonical::Main(c)) => mapped.push((n.clone(), c)),
            Some(Canonical::Mmproj(c)) => mmproj_mapped.push((n.clone(), c)),
            None => dropped += 1,
        }
    }
    eprintln!(
        "  mapped:  {} tensors kept, {} mmproj, {} dropped",
        mapped.len(),
        mmproj_mapped.len(),
        dropped
    );

    let mut tok_fields = tokenizer_fields.clone();
    if !tok_fields.contains_key("tokenizer_type") {
        tok_fields.insert("tokenizer_type".into(), serde_json::json!("hf"));
    }

    // Per-layer FFN width: derive from each layer's down_proj inner dim.
    // HF text_config exposes only a single `intermediate_size`, but Gemma 4
    // E2B has heterogeneous FFN (6144 for own-KV layers, 12288 for shared-KV
    // late layers). Without this, the runtime falls back to the uniform
    // `intermediate_size` and dispatches FFN GEMMs at the wrong dimension
    // for the late layers — model output is garbage. Auto-derive from the
    // actual MLX/HF tensor shapes the source ships.
    let mut config = config;
    if config.per_layer_ffn.is_empty() && config.num_hidden_layers > 0 {
        let mut per_layer_ffn = vec![0u32; config.num_hidden_layers as usize];
        let mut found_any = false;
        for (src_name, canonical) in &mapped {
            // canonical = "layers.{N}.mlp.down_proj.weight"
            let Some(rest) = canonical.strip_prefix("layers.") else { continue };
            let Some((idx_str, tail)) = rest.split_once('.') else { continue };
            if tail != "mlp.down_proj.weight" { continue }
            let Ok(layer) = idx_str.parse::<usize>() else { continue };
            if layer >= per_layer_ffn.len() { continue }
            // down_proj shape (HF unpacked): [hidden_size, ffn_size]
            if let Ok(shape) = provider.source_shape(src_name) {
                if shape.len() == 2 {
                    per_layer_ffn[layer] = shape[1] as u32;
                    found_any = true;
                }
            }
        }
        if found_any {
            // Backfill any zero entries with the uniform intermediate_size
            // so the runtime has a value for every layer.
            let fallback = config.intermediate_size;
            for v in per_layer_ffn.iter_mut() {
                if *v == 0 { *v = fallback; }
            }
            // Only emit per_layer_ffn when FFN width actually varies
            // layer-to-layer (Gemma 4 E2B: 6144 for own-KV, 12288 for
            // shared-KV). When every layer has the same width the
            // metadata is redundant and may push the runtime onto a
            // per-layer dispatch path that homogeneous archs (Gemma 3,
            // Llama, Qwen) don't expect.
            let uniform = per_layer_ffn.iter().all(|&w| w == per_layer_ffn[0]);
            if !uniform {
                config.per_layer_ffn = per_layer_ffn;
            }
        }
    }

    let mut header = Header {
        schema: 1,
        arch: canonical_arch.to_string(),
        quant_scheme,
        min_hw: "apple_m1".to_string(),
        created: chrono_now(),
        base_rt_version: env!("CARGO_PKG_VERSION").to_string(),
        source: SourceInfo {
            format: source_format.to_string(),
            sha256: "".to_string(), // streaming hash over a directory is a follow-up
            filename: input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
        },
        tokenizer: TokenizerBlob { fields: tok_fields },
        config: ModelConfig {
            fields: config.to_config_map(),
        },
        target_backend: TargetBackend::Metal,
        quant_profile: ctx.profile_name().unwrap_or("").to_string(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::QUANTIZED,
        layers: (0..config.num_hidden_layers)
            .map(|_| LayerDescriptor {
                kind: LayerKind::AttentionGqa,
                moe_n_experts: 0,
                moe_n_active: 0,
                shared_attn_layer: None,
                compute_hint: Some(ComputeRegion::Accelerator),
                precision: LayerPrecision::default(),
            })
            .collect(),
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
    };

    if config.tie_word_embeddings {
        header.flags |= HeaderFlags::TIED_EMBEDDINGS;
    }
    let has_moe = source_names
        .iter()
        .any(|n| n.contains("experts") || n.contains("_exps") || n.contains("_shexp"));
    let has_ssm = source_names.iter().any(|n| n.contains(".ssm.") || n.contains("ssm_"));
    let has_attn = source_names.iter().any(|n| n.contains("self_attn") || n.contains("attn_q"));
    if has_moe {
        header.flags |= HeaderFlags::HAS_MOE;
    }
    if has_ssm {
        header.flags |= HeaderFlags::HAS_SSM;
        if has_attn {
            header.flags |= HeaderFlags::HAS_HYBRID;
        }
    }

    let mut writer = BaseWriter::create(output, header).context("create writer")?;

    let pb = indicatif::ProgressBar::new(mapped.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template("  quantizing [{bar:28}] {pos}/{len} {msg}")
            .expect("valid progress template")
            .progress_chars("=>-"),
    );
    for (src_name, canonical) in &mapped {
        pb.set_message(canonical.clone());
        let shape = provider.source_shape(src_name)?;

        let mut f32s = provider
            .to_f32(src_name)
            .with_context(|| format!("reading {src_name}"))?;

        // Per-arch hook: bake the +1 unit-offset into Gemma 3's
        // zero-centered RMSNorm gamma so the runtime can use the plain
        // `rmsnorm(x) * weight` kernel and still produce
        // `rmsnorm(x) * (1 + weight)`. Mirrors `convert_hf_to_gguf.py`'s
        // `Gemma3Model.norm_shift`. Other archs no-op via the trait
        // default (returns 0.0).
        if shape.len() == 1 {
            let s = norm_shift(canonical);
            if s != 0.0 {
                for v in f32s.iter_mut() {
                    *v += s;
                }
            }
        }

        let f32s_for = || -> &[f32] { &f32s };

        let is_ssm_a = canonical == "ssm.a_log"
            || canonical.ends_with(".ssm.a_log")
            || src_name.ends_with(".ssm_a");
        let is_ssm_sensitive = is_ssm_a
            || canonical.ends_with(".ssm.dt_bias")
            || canonical.ends_with(".ssm.d");
        // See GGUF path above for rationale on the f16/GPU route.
        let is_norm_like = shape.len() == 1;
        let is_embedding =
            canonical == "embed_tokens.weight" || canonical == "lm_head.weight";

        let (entry, data) = if is_ssm_sensitive {
            let mut flags = TensorFlags::empty();
            if is_ssm_a {
                flags |= TensorFlags::SSM_A_MATRIX;
            }
            let mut entry = base_format::TensorEntry {
                name: canonical.clone(),
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
                residency: Some(base_format::ResidencyHint::Hot),
                compute_region: ComputeRegion::Cpu,
                scale_dtype: None,
                symmetric: false,
                flags,
                checksum_xxh64: None,
            source_ggml_type: None,
};
            let data: Vec<u8> = f32s_for().iter().flat_map(|f| f.to_le_bytes()).collect();
            entry.length = data.len() as u64;
            (entry, data)
        } else if is_norm_like {
            // 1-D norms (and biases caught by the same shape check) are
            // always emitted at f16. A profile's catch-all `**.weight`
            // rule typically targets a quant bit-width; quantizing a
            // per-channel norm-gain wrecks the model. Override the
            // profile here so a profile that omits explicit norm
            // patterns still produces a working bundle.
            let (bytes, dtype) = (
                f32s_for()
                    .iter()
                    .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                    .collect::<Vec<u8>>(),
                TensorDtype::F16,
            );
            let entry = base_format::TensorEntry {
                name: canonical.clone(),
                dtype,
                shape,
                offset: 0,
                length: bytes.len() as u64,
                scale_offset: None,
                scale_length: None,
                bias_offset: None,
                bias_length: None,
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: None,
                layout: None,
                residency: Some(base_format::ResidencyHint::Hot),
                compute_region: ComputeRegion::Gpu,
                scale_dtype: None,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            (entry, bytes)
        } else if is_embedding {
            // See GGUF path above for rationale on embed quantization.
            let in_features = shape.last().copied().map(|d| d as usize);
            let (packed, dtype) = if ctx.profile.is_some() {
                ctx.pack_tensor(canonical, f32s_for(), in_features)?
            } else {
                pack_for_target(f32s_for(), target)?
            };
            let mut data = Vec::with_capacity(
                packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
            );
            data.extend_from_slice(&packed.packed_weights);
            let scale_off = data.len() as u64;
            data.extend_from_slice(&packed.scales);
            let bias_off = data.len() as u64;
            data.extend_from_slice(&packed.biases);
            let mut entry = base_format::TensorEntry {
                name: canonical.clone(),
                dtype,
                shape,
                offset: 0,
                length: 0,
                scale_offset: if !packed.scales.is_empty() {
                    Some(scale_off)
                } else {
                    None
                },
                scale_length: if !packed.scales.is_empty() {
                    Some(packed.scales.len() as u64)
                } else {
                    None
                },
                bias_offset: if !packed.biases.is_empty() {
                    Some(bias_off)
                } else {
                    None
                },
                bias_length: if !packed.biases.is_empty() {
                    Some(packed.biases.len() as u64)
                } else {
                    None
                },
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: if packed.group_size > 0 {
                    Some(packed.group_size)
                } else {
                    None
                },
                layout: None,
                residency: Some(base_format::ResidencyHint::Hot),
                compute_region: ComputeRegion::Gpu,
                scale_dtype: packed.scale_dtype,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            entry.length = data.len() as u64;
            (entry, data)
        } else {
            let in_features = shape.last().copied().map(|d| d as usize);
            let (packed, dtype) = if ctx.profile.is_some() {
                ctx.pack_tensor(canonical, f32s_for(), in_features)?
            } else {
                pack_for_target(f32s_for(), target)?
            };
            let mut data = Vec::with_capacity(
                packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
            );
            data.extend_from_slice(&packed.packed_weights);
            let scale_off = data.len() as u64;
            data.extend_from_slice(&packed.scales);
            let bias_off = data.len() as u64;
            data.extend_from_slice(&packed.biases);

            let mut entry = base_format::TensorEntry {
                name: canonical.clone(),
                dtype,
                shape,
                offset: 0,
                length: data.len() as u64,
                scale_offset: if !packed.scales.is_empty() {
                    Some(scale_off)
                } else {
                    None
                },
                scale_length: if !packed.scales.is_empty() {
                    Some(packed.scales.len() as u64)
                } else {
                    None
                },
                bias_offset: if !packed.biases.is_empty() {
                    Some(bias_off)
                } else {
                    None
                },
                bias_length: if !packed.biases.is_empty() {
                    Some(packed.biases.len() as u64)
                } else {
                    None
                },
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: if packed.group_size > 0 {
                    Some(packed.group_size)
                } else {
                    None
                },
                layout: None,
                residency: Some(base_format::ResidencyHint::Warm),
                compute_region: ComputeRegion::Accelerator,
                scale_dtype: packed.scale_dtype,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            entry.length = data.len() as u64;
            (entry, data)
        };

        writer.add_tensor(TensorPayload { entry, data });
        pb.inc(1);
    }
    pb.finish_and_clear();
    eprintln!("  quantized {} tensors", mapped.len());

    // Multimodal towers — preserve their HF names verbatim and route
    // them into the mmproj sub-bundle. Tower weights stay on the same
    // f16 → bf16 / target-quant routing as the LM body so a future
    // multimodal runtime can dispatch on the same kernel set. We tag
    // them as `Accelerator` since they're large 2-D weight matrices
    // with the same memory characteristics as the LM weights.
    if !mmproj_mapped.is_empty() {
        writer.set_mmproj_arch(format!("{canonical_arch}_mm"));
        if !mmproj_config.is_empty() {
            writer.set_mmproj_config(mmproj_config);
        }
    }
    for (src_name, canonical) in &mmproj_mapped {
        let shape = provider.source_shape(src_name)?;
        let f32s = provider
            .to_f32(src_name)
            .with_context(|| format!("reading mmproj {src_name}"))?;
        let f32s_for = || -> &[f32] { &f32s };
        let logical_count: usize = shape.iter().map(|d| *d as usize).product();
        // A tower checkpoint sprays many small auxiliary tensors
        // (scalars, per-channel offsets, conv biases, lookup tables)
        // that aren't group-aligned. Route anything that isn't a plain
        // 2-D linear weight through the f16 path so the runtime can
        // read them via `tensor_raw_ptr` without dequant. The runtime
        // expects pos_embed / per_dim_scale / clip-bounds in raw
        // BF16/F16 — quantizing them silently breaks the vision encoder.
        let is_small_or_unaligned = shape.len() != 2
            || logical_count < 64
            || logical_count % 64 != 0
            || canonical.contains("position_embedding")
            || canonical.contains("pos_embed")
            || canonical.contains(".per_dim_scale")
            || canonical.ends_with(".bias")
            || canonical.contains(".relative_k_proj");
        let (entry, data) = if is_small_or_unaligned {
            let bytes: Vec<u8> = f32s_for()
                .iter()
                .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                .collect();
            let entry = base_format::TensorEntry {
                name: canonical.clone(),
                dtype: TensorDtype::F16,
                shape,
                offset: 0,
                length: bytes.len() as u64,
                scale_offset: None,
                scale_length: None,
                bias_offset: None,
                bias_length: None,
                awq_scale_offset: None,
                awq_scale_length: None,
                group_size: None,
                layout: None,
                residency: Some(base_format::ResidencyHint::Cold),
                compute_region: ComputeRegion::Gpu,
                scale_dtype: None,
                symmetric: false,
                flags: TensorFlags::empty(),
                checksum_xxh64: None,
            source_ggml_type: None,
};
            (entry, bytes)
        } else {
            // Pad if needed so pack's group-size invariant holds for
            // tower tensors of arbitrary shapes. Towers are rare and
            // large; group-aligned by construction in our current set,
            // but the runtime won't care if we drop a few mis-aligned
            // ones rather than break the convert.
            //
            // Profile-driven path takes priority — mmproj tensors are
            // matched against the profile rules just like LM tensors,
            // so vision/audio towers get the same canonical-quant
            // treatment (typically the catch-all `**.weight` rule).
            let in_features = shape.last().copied().map(|d| d as usize);
            let pack_n: Result<_> = if ctx.profile.is_some() {
                ctx.pack_tensor(canonical, f32s_for(), in_features)
            } else {
                pack_for_target(f32s_for(), target)
            };
            match pack_n {
                Ok((packed, dtype)) => {
                    let mut data = Vec::with_capacity(
                        packed.packed_weights.len()
                            + packed.scales.len()
                            + packed.biases.len(),
                    );
                    data.extend_from_slice(&packed.packed_weights);
                    let scale_off = data.len() as u64;
                    data.extend_from_slice(&packed.scales);
                    let bias_off = data.len() as u64;
                    data.extend_from_slice(&packed.biases);
                    let mut entry = base_format::TensorEntry {
                        name: canonical.clone(),
                        dtype,
                        shape,
                        offset: 0,
                        length: 0,
                        scale_offset: if !packed.scales.is_empty() {
                            Some(scale_off)
                        } else {
                            None
                        },
                        scale_length: if !packed.scales.is_empty() {
                            Some(packed.scales.len() as u64)
                        } else {
                            None
                        },
                        bias_offset: if !packed.biases.is_empty() {
                            Some(bias_off)
                        } else {
                            None
                        },
                        bias_length: if !packed.biases.is_empty() {
                            Some(packed.biases.len() as u64)
                        } else {
                            None
                        },
                        awq_scale_offset: None,
                        awq_scale_length: None,
                        group_size: if packed.group_size > 0 {
                            Some(packed.group_size)
                        } else {
                            None
                        },
                        layout: None,
                        residency: Some(base_format::ResidencyHint::Cold),
                        compute_region: ComputeRegion::Accelerator,
                        scale_dtype: packed.scale_dtype,
                        symmetric: false,
                        flags: TensorFlags::empty(),
                        checksum_xxh64: None,
                    source_ggml_type: None,
};
                    entry.length = data.len() as u64;
                    (entry, data)
                }
                Err(_) => {
                    // Fallback: emit as f16 GPU. Tower tensor whose
                    // last dim isn't group-aligned. Decode quality is
                    // not gated on this — runtime support is the
                    // bottleneck.
                    let bytes: Vec<u8> = f32s_for()
                        .iter()
                        .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                        .collect();
                    let entry = base_format::TensorEntry {
                        name: canonical.clone(),
                        dtype: TensorDtype::F16,
                        shape,
                        offset: 0,
                        length: bytes.len() as u64,
                        scale_offset: None,
                        scale_length: None,
                        bias_offset: None,
                        bias_length: None,
                        awq_scale_offset: None,
                        awq_scale_length: None,
                        group_size: None,
                        layout: None,
                        residency: Some(base_format::ResidencyHint::Cold),
                        compute_region: ComputeRegion::Gpu,
                        scale_dtype: None,
                        symmetric: false,
                        flags: TensorFlags::empty(),
                        checksum_xxh64: None,
                    source_ggml_type: None,
};
                    (entry, bytes)
                }
            }
        };
        writer.add_mmproj_tensor(TensorPayload { entry, data });
    }

    writer.finish().context("writing bundle")?;

    let reader = BaseReader::open(output).context("reopen for verification")?;
    for t in reader.header().tensors.iter() {
        if t.compute_region == ComputeRegion::Gpu
            && !reader.tensor_is_zero_copy_eligible(&t.name)?
        {
            bail!("tensor {:?} failed zero-copy alignment check", t.name);
        }
    }
    eprintln!(
        "  wrote {} tensors ({} MB)",
        reader.header().tensors.len(),
        std::fs::metadata(output)?.len() / (1024 * 1024)
    );
    Ok(())
}

/// Routing decision for an HF/MLX tensor name.
#[derive(Debug, Clone)]
enum Canonical {
    /// Goes into the main LM bundle (header.tensors).
    Main(String),
    /// Goes into the mmproj sub-bundle (header.mmproj.tensors). The
    /// inner string is the name as it should appear in the manifest;
    /// for now we keep the HF prefix verbatim so the future runtime can
    /// dispatch on the canonical multimodal layout.
    Mmproj(String),
}

/// nomic-bert HF safetensors → canonical `.base` names. Targets the
/// same names the GGUF `NomicBertMapper` emits so the runtime
/// (`src/core/models/bert.cpp`) sees one form regardless of source.
fn nomic_bert_hf_rename(name: &str) -> Option<String> {
    match name {
        "embeddings.word_embeddings.weight" => return Some("embed_tokens.weight".into()),
        "embeddings.token_type_embeddings.weight" => return Some("token_types.weight".into()),
        "emb_ln.weight" => return Some("token_embd_norm.weight".into()),
        "emb_ln.bias" => return Some("token_embd_norm.bias".into()),
        // Pooling / classifier heads from sentence-transformers wrappers
        // — nomic-embed-text uses mean pooling, no learned head.
        "embeddings.position_embeddings.weight" => return None,
        _ => {}
    }
    let rest = name.strip_prefix("encoder.layers.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;
    let canonical = match suffix {
        "attn.Wqkv.weight" => "self_attn.qkv_proj.weight",
        "attn.out_proj.weight" => "self_attn.o_proj.weight",
        "mlp.fc11.weight" => "mlp.gate_proj.weight",
        "mlp.fc12.weight" => "mlp.up_proj.weight",
        "mlp.fc2.weight" => "mlp.down_proj.weight",
        // Nomic block ordering is post-norm BERT-style: `norm1` wraps
        // attention, `norm2` wraps MLP. The runtime resolves the
        // cross-layer alias so blk.N's `layer_output_norm` is read as
        // layer N+1's input norm; converter just keeps the GGUF names.
        "norm1.weight" => "attn_output_norm.weight",
        "norm1.bias" => "attn_output_norm.bias",
        "norm2.weight" => "layer_output_norm.weight",
        "norm2.bias" => "layer_output_norm.bias",
        _ => return None,
    };
    Some(format!("layers.{layer}.{canonical}"))
}

/// Map an HF-style or MLX-style tensor name to canonical `.base` name.
/// HF already uses the canonical `model.layers.N.*` convention, so we
/// mostly strip the `model.` prefix. For anything not in HF convention,
/// fall back to the llama-style GGUF mapper.
fn to_canonical_name(name: &str, arch: &str) -> Option<Canonical> {
    // Multimodal wrappers (Gemma-4) put the LLM under `language_model.model.*`
    // and the audio/vision towers under siblings. The towers ship in
    // the same .base bundle but live under `header.mmproj.tensors` so
    // a text-only runtime can skip them entirely.
    //
    // Apply the Gemma-4 mmproj canonical rename here so the runtime
    // sees runtime-canonical names (`vision.layers.N.attention.q.weight`,
    // `audio.sscp.layer0.conv.weight`, …) directly. Without this the
    // BaseWeightStore would have to carry a parallel rule table.
    // Tensors that don't match a known rule are passed through verbatim
    // (the runtime will simply not look them up).
    // HF mainline (Gemma 4 26B-A4B) wraps everything under `model.*`, so the
    // tower roots show up as `model.vision_tower.*` / `model.audio_tower.*` /
    // `model.embed_vision.*`. Strip a leading `model.` before the tower-prefix
    // check so HF-mainline towers route to mmproj just like MLX-source towers.
    let mm_name = name.strip_prefix("model.").unwrap_or(name);
    if mm_name.starts_with("audio_tower.")
        || mm_name.starts_with("vision_tower.")
        || mm_name.starts_with("embed_audio")
        || mm_name.starts_with("embed_vision")
        || mm_name.starts_with("multi_modal_projector")
    {
        let canonical = base_arch::gemma::map_gemma4_mmproj_name(mm_name).unwrap_or_else(|| mm_name.to_string());
        return Some(Canonical::Mmproj(canonical));
    }

    // nomic-bert HF safetensors use BERT-flavored names that don't match
    // the LLaMA `model.layers.N.*` convention the rest of this function
    // assumes. Handle them inline; the rename targets are the same
    // canonical names the GGUF nomic-bert mapper emits, so the runtime
    // path is identical regardless of source format.
    if arch == "nomic-bert" {
        return nomic_bert_hf_rename(name).map(Canonical::Main);
    }

    // Strip HF naming prefix. Accept multiple multimodal wrapper
    // orderings:
    //   - `model.language_model.*` (Gemma 4 26B-A4B, mainline HF)
    //   - `language_model.model.*` (older/alternative wrapping)
    //   - `language_model.*`
    //   - `model.*` (text-only models)
    let stripped = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("language_model.model."))
        .or_else(|| name.strip_prefix("language_model."))
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    // HF native names → canonical.
    if stripped != name {
        if stripped == "rotary_emb.inv_freq" {
            return None;
        }
        // The HF model has `model.norm.weight` for the final pre-output
        // norm; the GGUF / runtime canonical name is `final_norm.weight`
        // (Llama mapper renames `output_norm.weight → final_norm.weight`,
        // and the runtime alias `output_norm ↔ final_norm` resolves the
        // runtime side). Without this rename, HF-sourced .base bundles
        // ship `norm.weight` and the runtime gets a null buffer for the
        // final norm, leaving the LM-head GEMV to read uninitialized
        // hidden state.
        if stripped == "norm.weight" {
            return Some(Canonical::Main("final_norm.weight".to_string()));
        }
        // Gemma 4 PLE globals — HF name → canonical name the runtime reads.
        if stripped == "embed_tokens_per_layer.weight" {
            return Some(Canonical::Main("per_layer_token_embd.weight".to_string()));
        }
        if stripped == "per_layer_model_projection.weight" {
            return Some(Canonical::Main("per_layer_model_proj.weight".to_string()));
        }
        if stripped == "per_layer_projection_norm.weight" {
            return Some(Canonical::Main("per_layer_proj_norm.weight".to_string()));
        }
        // MLX collapses per-expert MoE weights into a stacked
        // [n_experts, out, in] tensor. Different MLX checkpoints use
        // different parent paths:
        //   - Qwen3-MoE:   `mlp.switch_mlp.{gate,up,down}_proj.weight`
        //   - Gemma 4 MoE: `experts.switch_glu.{gate,up,down}_proj.weight`
        //                  (NO `mlp.` prefix; sibling `router.proj` instead
        //                   of `mlp.router`.)
        // Canonicalize both into the same `mlp.experts.{...}` shape so
        // the runtime's name-remap table only sees one form.
        let mut canon = stripped.replace(".mlp.switch_mlp.", ".mlp.experts.");
        canon = canon.replace(".experts.switch_glu.", ".mlp.experts.");
        // Gemma 4 MLX router: `router.proj.weight` is the routing
        // matrix; map to `.mlp.router.weight` (= GGUF .base form, which
        // base_weight_store remaps to runtime canonical
        // `ffn_gate_inp.weight`). `router.scale` is the per-layer
        // router input-RMS-norm scalar that gemma4.cpp reads directly
        // by its GGUF-style name `ffn_gate_inp.scale`. HF mainline
        // also has `router.per_expert_scale` (per-expert post-down
        // multiplier) which maps to `ffn_down_exps.scale`.
        canon = canon
            .replace(".router.proj.weight", ".mlp.router.weight")
            .replace(".router.per_expert_scale", ".ffn_down_exps.scale")
            .replace(".router.scale", ".ffn_gate_inp.scale");
        // Two MoE expert source-name conventions to canonicalize:
        //
        //   HF mainline (Gemma 4 26B-A4B):
        //     `.experts.gate_up_proj`  (bare, no `.weight` — 3D fused tensor)
        //     `.experts.down_proj`     (bare, no `.weight`)
        //   MLX / split convention:
        //     `.experts.gate_proj.weight`
        //     `.experts.up_proj.weight`
        //     `.experts.down_proj.weight`
        //
        // Match the longer-suffix `.weight` form first so the bare-suffix
        // rename doesn't double up the `.weight` tail. Pre-fix produced
        // `mlp.ffn_gate_exps.weight.weight` for MLX-style sources. Use
        // `.mlp.experts.X_proj` → `.ffn_X_exps` (drop the now-redundant
        // `.mlp.` prefix) so the runtime's BaseWeightStore alias catches
        // the canonical name without going through the legacy
        // `mlp.experts.X_proj.weight ↔ ffn_X_exps.weight` rule.
        canon = canon
            .replace(".mlp.experts.gate_up_proj.weight", ".ffn_gate_up_exps.weight")
            .replace(".mlp.experts.down_proj.weight", ".ffn_down_exps.weight")
            .replace(".mlp.experts.gate_proj.weight", ".ffn_gate_exps.weight")
            .replace(".mlp.experts.up_proj.weight", ".ffn_up_exps.weight")
            .replace(".experts.gate_up_proj.weight", ".ffn_gate_up_exps.weight")
            .replace(".experts.down_proj.weight", ".ffn_down_exps.weight")
            .replace(".experts.gate_proj.weight", ".ffn_gate_exps.weight")
            .replace(".experts.up_proj.weight", ".ffn_up_exps.weight")
            .replace(".experts.gate_up_proj", ".ffn_gate_up_exps.weight")
            .replace(".experts.down_proj", ".ffn_down_exps.weight")
            .replace(".experts.gate_proj", ".ffn_gate_exps.weight")
            .replace(".experts.up_proj", ".ffn_up_exps.weight");
        // Gemma 4 PLE per-layer tensors. HF uses descriptive names; the
        // runtime reads the GGUF-flavored canonical names.
        canon = canon
            .replace(".per_layer_input_gate.", ".per_layer_inp_gate.")
            .replace(".per_layer_projection.", ".per_layer_proj.")
            .replace(".post_per_layer_input_norm.", ".per_layer_post_norm.");
        // Gemma 4 has four distinct per-layer norms (HF naming):
        //   input_layernorm                — pre-attention norm
        //   post_attention_layernorm       — post-attention pre-residual
        //   pre_feedforward_layernorm      — pre-FFN norm (= GGUF ffn_norm)
        //   post_feedforward_layernorm     — post-FFN pre-residual
        // Map all four to the GGUF-style canonicals that the runtime's
        // alias map resolves (`input_norm`, `post_attention_norm`,
        // `post_attn_norm`, `post_ffw_norm`). Without the rename the MLX
        // .base ships `input_layernorm` / `post_feedforward_layernorm` —
        // the runtime alias table accepts both forms, but emitting the
        // GGUF-canonical names keeps MLX-sourced and GGUF-sourced .base
        // bundles header-equivalent so per-tensor diffs surface real
        // structural divergence rather than naming noise.
        canon = canon.replace(".input_layernorm.", ".input_norm.");
        // `post_attention_layernorm` is one HF name with two meanings:
        //   - Llama / Qwen / Mistral: pre-FFN norm (= GGUF `ffn_norm`,
        //     base canonical `post_attn_norm`).
        //   - Gemma 4: post-attention pre-residual norm (Gemma-specific,
        //     kept verbatim in .base, paired with a SEPARATE pre-FFN
        //     `pre_feedforward_layernorm`).
        // Rename per-arch so a Llama-style HF/MLX bundle ships
        // `post_attn_norm.weight` (which kBasePerLayerRules then
        // remaps to runtime canonical `ffn_norm.weight`) while a
        // Gemma 4 bundle ships `post_attention_norm.weight` directly.
        // Pre-arch-aware code did the Gemma rename unconditionally,
        // silently breaking Llama-style MLX MoE (the pre-FFN norm
        // ended up under a name kBasePerLayerRules doesn't know).
        if arch == "gemma4" || arch == "gemma3" {
            // Gemma 3 and Gemma 4 both have four per-layer norms
            // (input + post-attn + pre-FFN + post-FFN). HF and GGUF use
            // different names; map HF → GGUF-canonical so the runtime's
            // alias table resolves identically regardless of source. The
            // earlier code skipped Gemma 3's `pre_feedforward_layernorm`
            // and `post_feedforward_layernorm` renames (they only fired
            // for arch == "gemma4"), so HF-derived Gemma 3 .base files
            // carried the Gemma-4-incompatible names and the runtime's
            // norm pipeline missed them.
            canon = canon
                .replace(".post_attention_layernorm.", ".post_attention_norm.")
                .replace(".pre_feedforward_layernorm.", ".post_attn_norm.")
                .replace(".post_feedforward_layernorm.", ".post_ffw_norm.")
                // Gemma 4 26B-A4B MoE has TWO parallel FFN streams
                // (shared dense + routed experts) with separate pre/post
                // norms for each — the HF names carry _1/_2 suffixes;
                // the runtime reads them under the GGUF-canonical names.
                .replace(".post_feedforward_layernorm_1.", ".post_ffw_norm_1.")
                .replace(".post_feedforward_layernorm_2.", ".post_ffw_norm_2.")
                .replace(".pre_feedforward_layernorm_2.", ".pre_ffw_norm_2.");
        } else {
            canon = canon.replace(".post_attention_layernorm.", ".post_attn_norm.");
        }
        // HF `layers.N.layer_scalar` (shape [1] BF16) ↔ GGUF
        // `blk.N.layer_output_scale.weight` ↔ canonical
        // `layers.N.layer_out_scale.weight`.
        if let Some(rest) = canon.strip_prefix("layers.") {
            if let Some((idx, tail)) = rest.split_once('.') {
                if tail == "layer_scalar" && idx.parse::<u32>().is_ok() {
                    return Some(Canonical::Main(format!(
                        "layers.{idx}.layer_out_scale.weight"
                    )));
                }
            }
        }
        return Some(Canonical::Main(canon));
    }
    // Top-level lm_head / embed_tokens / etc.
    if name == "lm_head.weight" || name == "embed_tokens.weight" {
        return Some(Canonical::Main(name.to_string()));
    }
    // Fall back to GGUF-style mapping (blk.N.*).
    base_arch::llama::map_llama_style(name).map(Canonical::Main)
}

fn compute_sha256_streaming(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let f = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&f)? };
    let mut h = Sha256::new();
    // Hash in 1 MiB chunks to bound peak memory.
    for chunk in mmap.chunks(1024 * 1024) {
        h.update(chunk);
    }
    let digest = h.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// Pipeline exercise: build a deterministic 4-layer transformer-shaped
/// Synthetic-bundle generator with QuantContext (canonical-quant
/// path). When the context carries a profile, per-tensor quant is
/// driven by profile.resolve; otherwise falls back to convert_synthetic
/// for v1.0 behavior. Useful for end-to-end smoke testing the canonical
/// pipeline without a real model checkpoint.
fn convert_synthetic_with_ctx(
    output: &std::path::Path,
    ctx: &QuantContext,
) -> Result<()> {
    if ctx.profile.is_none() {
        return convert_synthetic(output, ctx.target);
    }
    use base_format::{
        AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, LayerKind,
        LayerDescriptor, LayerPrecision, ModelConfig, QuantScheme, SourceInfo, TargetBackend,
        TensorFlags, TensorPayload, TokenizerBlob,
    };
    use std::collections::BTreeMap;

    let hidden = 256usize;
    let n_layers = 4usize;
    let vocab = 512usize;

    // Bundle-default scheme echoes the most common dtype the profile
    // emits — informational, the runtime keys per-tensor on TensorEntry.dtype.
    let bundle_scheme = QuantScheme::BaseQ4;

    let header = Header {
        schema: 1,
        arch: "synthetic".to_string(),
        quant_scheme: bundle_scheme,
        min_hw: "apple_m1".to_string(),
        created: chrono_now(),
        base_rt_version: env!("CARGO_PKG_VERSION").to_string(),
        source: SourceInfo {
            format: "synthetic".to_string(),
            sha256: "0".repeat(64),
            filename: "synthetic".to_string(),
        },
        tokenizer: TokenizerBlob {
            fields: BTreeMap::new(),
        },
        config: ModelConfig {
            fields: {
                let mut f = BTreeMap::new();
                f.insert("hidden_size".into(), serde_json::json!(hidden));
                f.insert("num_hidden_layers".into(), serde_json::json!(n_layers));
                f.insert("vocab_size".into(), serde_json::json!(vocab));
                f
            },
        },
        target_backend: TargetBackend::Metal,
        quant_profile: ctx
            .profile_name()
            .unwrap_or("")
            .to_string(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::QUANTIZED | HeaderFlags::TIED_EMBEDDINGS,
        layers: (0..n_layers)
            .map(|_| LayerDescriptor {
                kind: LayerKind::AttentionGqa,
                moe_n_experts: 0,
                moe_n_active: 0,
                shared_attn_layer: None,
                compute_hint: Some(ComputeRegion::Accelerator),
                precision: LayerPrecision::default(),
            })
            .collect(),
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
    };

    let mut writer = BaseWriter::create(output, header.clone()).context("create writer")?;

    // Helper: pack via QuantContext + emit a TensorEntry with the
    // weights+scales+biases combined into one TensorPayload.
    let emit = |writer: &mut BaseWriter<_>,
                name: &str,
                dtype_intent: &str,
                weights: &[f32],
                shape: Vec<u64>,
                in_features: Option<usize>,
                region: ComputeRegion|
     -> Result<()> {
        let _ = dtype_intent; // dtype comes back from pack_tensor
        let (packed, dtype) = ctx.pack_tensor(name, weights, in_features)?;
        let mut data = Vec::with_capacity(
            packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
        );
        data.extend_from_slice(&packed.packed_weights);
        let scale_off = data.len() as u64;
        data.extend_from_slice(&packed.scales);
        let bias_off = data.len() as u64;
        data.extend_from_slice(&packed.biases);

        let mut entry = base_format::TensorEntry {
            name: name.to_string(),
            dtype,
            shape,
            offset: 0,
            length: 0,
            scale_offset: if !packed.scales.is_empty() {
                Some(scale_off)
            } else {
                None
            },
            scale_length: if !packed.scales.is_empty() {
                Some(packed.scales.len() as u64)
            } else {
                None
            },
            bias_offset: if !packed.biases.is_empty() {
                Some(bias_off)
            } else {
                None
            },
            bias_length: if !packed.biases.is_empty() {
                Some(packed.biases.len() as u64)
            } else {
                None
            },
            awq_scale_offset: None,
            awq_scale_length: None,
            group_size: if packed.group_size > 0 {
                Some(packed.group_size)
            } else {
                None
            },
            scale_dtype: packed.scale_dtype,
            symmetric: false,
            layout: None,
            residency: Some(base_format::ResidencyHint::Warm),
            compute_region: region,
            flags: TensorFlags::empty(),
            checksum_xxh64: None,
            source_ggml_type: None,
        };
        entry.length = data.len() as u64;
        writer.add_tensor(TensorPayload { entry, data });
        Ok(())
    };

    // Embedding (always bf16 per default profiles).
    let embed: Vec<f32> = (0..vocab * hidden).map(|i| ((i as f32) % 17.0) * 0.01).collect();
    emit(
        &mut writer,
        "model.embed_tokens.weight",
        "bf16",
        &embed,
        vec![vocab as u64, hidden as u64],
        Some(hidden),
        ComputeRegion::Gpu,
    )?;

    // Per-layer weights routed through the profile.
    for layer in 0..n_layers {
        // Attention projection (q): [hidden, hidden]
        let w: Vec<f32> = (0..hidden * hidden)
            .map(|i| (((layer * 7 + i) as f32) % 31.0 - 15.0) * 0.02)
            .collect();
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            emit(
                &mut writer,
                &format!("model.layers.{layer}.self_attn.{proj}.weight"),
                "qkvo",
                &w,
                vec![hidden as u64, hidden as u64],
                Some(hidden),
                ComputeRegion::Accelerator,
            )?;
        }
        // MLP gate/up/down — gate/up are [hidden*2, hidden]; down is [hidden, hidden*2].
        let mlp_w: Vec<f32> = (0..hidden * hidden * 2)
            .map(|i| (((layer * 13 + i) as f32) % 23.0 - 11.0) * 0.03)
            .collect();
        emit(
            &mut writer,
            &format!("model.layers.{layer}.mlp.gate_proj.weight"),
            "mlp",
            &mlp_w,
            vec![(hidden * 2) as u64, hidden as u64],
            Some(hidden),
            ComputeRegion::Accelerator,
        )?;
        emit(
            &mut writer,
            &format!("model.layers.{layer}.mlp.up_proj.weight"),
            "mlp",
            &mlp_w,
            vec![(hidden * 2) as u64, hidden as u64],
            Some(hidden),
            ComputeRegion::Accelerator,
        )?;
        let down_w: Vec<f32> = (0..hidden * hidden * 2)
            .map(|i| (((layer * 17 + i) as f32) % 19.0 - 9.0) * 0.025)
            .collect();
        emit(
            &mut writer,
            &format!("model.layers.{layer}.mlp.down_proj.weight"),
            "mlp",
            &down_w,
            vec![hidden as u64, (hidden * 2) as u64],
            Some(hidden * 2),
            ComputeRegion::Accelerator,
        )?;
        // Norms (bf16 per profile defaults).
        let norm: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();
        emit(
            &mut writer,
            &format!("model.layers.{layer}.input_layernorm.weight"),
            "norm",
            &norm,
            vec![hidden as u64],
            Some(hidden),
            ComputeRegion::Cpu,
        )?;
        emit(
            &mut writer,
            &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            "norm",
            &norm,
            vec![hidden as u64],
            Some(hidden),
            ComputeRegion::Cpu,
        )?;
    }

    // Final norm.
    let norm: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();
    emit(
        &mut writer,
        "model.norm.weight",
        "norm",
        &norm,
        vec![hidden as u64],
        Some(hidden),
        ComputeRegion::Cpu,
    )?;

    // lm_head.
    let lm_w: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i as f32) % 11.0) * 0.04)
        .collect();
    emit(
        &mut writer,
        "lm_head.weight",
        "lm_head",
        &lm_w,
        vec![vocab as u64, hidden as u64],
        Some(hidden),
        ComputeRegion::Gpu,
    )?;

    writer.finish().context("writing canonical synthetic bundle")?;

    // Read it back; verify the canonical fields populated correctly.
    let reader = BaseReader::open(output).context("reopen canonical bundle")?;
    let h = reader.header();
    assert_eq!(h.target_backend, TargetBackend::Metal);
    assert_eq!(h.quant_profile, ctx.profile_name().unwrap_or(""));
    eprintln!(
        "  ok: wrote {} tensors, profile={}",
        h.tensors.len(),
        h.quant_profile
    );
    Ok(())
}

/// bundle with synthetic weights, quantize per `--target`, write to
/// disk, and verify it reads back.
fn convert_synthetic(output: &std::path::Path, target: TargetScheme) -> Result<()> {
    use base_format::{
        AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, LayerKind,
        LayerDescriptor, LayerPrecision, ModelConfig, QuantScheme, SourceInfo, TargetBackend, TensorDtype,
        TensorFlags, TensorPayload, TokenizerBlob,
    };
    use std::collections::BTreeMap;

    let hidden = 256usize;
    let n_layers = 4usize;
    let vocab = 512usize;

    let quant_scheme = match target {
        TargetScheme::BaseQ2 => QuantScheme::BaseQ2,
        TargetScheme::BaseQ3 => QuantScheme::BaseQ3,
        TargetScheme::BaseQ4 => QuantScheme::BaseQ4,
        TargetScheme::BaseQ5 => QuantScheme::BaseQ5,
        TargetScheme::BaseQ6 => QuantScheme::BaseQ6,
        TargetScheme::BaseQ8 => QuantScheme::BaseQ8,
        TargetScheme::Bf16 => QuantScheme::Bf16,
        TargetScheme::Mxfp4 => QuantScheme::Mxfp4,
        TargetScheme::Nvfp4 => QuantScheme::Nvfp4,
    };

    let header = Header {
        schema: 1,
        arch: "synthetic".to_string(),
        quant_scheme,
        min_hw: "apple_m1".to_string(),
        created: chrono_now(),
        base_rt_version: env!("CARGO_PKG_VERSION").to_string(),
        source: SourceInfo {
            format: "synthetic".to_string(),
            sha256: "0".repeat(64),
            filename: "synthetic".to_string(),
        },
        tokenizer: TokenizerBlob {
            fields: BTreeMap::new(),
        },
        config: ModelConfig {
            fields: {
                let mut f = BTreeMap::new();
                f.insert("hidden_size".into(), serde_json::json!(hidden));
                f.insert("num_hidden_layers".into(), serde_json::json!(n_layers));
                f.insert("vocab_size".into(), serde_json::json!(vocab));
                f
            },
        },
        target_backend: TargetBackend::Metal,
        quant_profile: String::new(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::QUANTIZED | HeaderFlags::TIED_EMBEDDINGS,
        layers: (0..n_layers)
            .map(|_| LayerDescriptor {
                kind: LayerKind::AttentionGqa,
                moe_n_experts: 0,
                moe_n_active: 0,
                shared_attn_layer: None,
                compute_hint: Some(ComputeRegion::Accelerator),
                precision: LayerPrecision::default(),
            })
            .collect(),
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
    };

    let mut writer = BaseWriter::create(output, header.clone()).context("create writer")?;

    // Embedding (GPU region, bf16).
    let embed: Vec<f32> = (0..vocab * hidden).map(|i| ((i as f32) % 17.0) * 0.01).collect();
    let embed_bytes: Vec<u8> = embed
        .iter()
        .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
        .collect();
    writer.add_tensor(TensorPayload {
        entry: tensor_entry(
            "embed_tokens.weight",
            TensorDtype::Bf16,
            vec![vocab as u64, hidden as u64],
            ComputeRegion::Gpu,
            None,
            None,
            None,
        ),
        data: embed_bytes,
    });

    // Per-layer attention + MLP, quantized.
    for layer in 0..n_layers {
        // A deterministic weight matrix for each tensor.
        let w_attn: Vec<f32> = (0..hidden * hidden * 3)
            .map(|i| (((layer * 7 + i) as f32) % 31.0 - 15.0) * 0.02)
            .collect();
        let (packed, dtype) = pack_for_target(&w_attn, target)?;

        let packed_len = packed.packed_weights.len() as u64;
        let scales_len = packed.scales.len() as u64;
        let biases_len = packed.biases.len() as u64;

        // Concatenate weights || scales || biases into one TensorPayload,
        // with offsets pointing into the single payload. The writer
        // assigns a single offset; we record sub-offsets relative to it.
        let mut data = Vec::with_capacity(
            packed.packed_weights.len() + packed.scales.len() + packed.biases.len(),
        );
        data.extend_from_slice(&packed.packed_weights);
        let scale_off = data.len() as u64;
        data.extend_from_slice(&packed.scales);
        let bias_off = data.len() as u64;
        data.extend_from_slice(&packed.biases);

        let mut entry = tensor_entry(
            &format!("layers.{layer}.attn.qkv_proj.weight"),
            dtype,
            vec![(hidden * 3) as u64, hidden as u64],
            ComputeRegion::Accelerator,
            Some(packed.group_size),
            Some(packed_len),
            None,
        );
        entry.scale_offset = Some(scale_off);
        entry.scale_length = Some(scales_len);
        if biases_len > 0 {
            entry.bias_offset = Some(bias_off);
            entry.bias_length = Some(biases_len);
        }
        entry.flags = TensorFlags::empty();
        writer.add_tensor(TensorPayload { entry, data });
    }

    // Final norm — f32, CPU region.
    let norm: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let norm_bytes: Vec<u8> = norm.iter().flat_map(|f| f.to_le_bytes()).collect();
    writer.add_tensor(TensorPayload {
        entry: tensor_entry(
            "final_norm.weight",
            TensorDtype::F32,
            vec![hidden as u64],
            ComputeRegion::Cpu,
            None,
            None,
            None,
        ),
        data: norm_bytes,
    });

    writer.finish().context("writing synthetic bundle")?;

    // Read it back and assert invariants.
    let reader = BaseReader::open(output).context("reopen for verification")?;
    assert_eq!(reader.header().tensors.len(), n_layers + 2);
    for t in reader.header().tensors.iter() {
        assert!(
            reader.tensor_is_zero_copy_eligible(&t.name)?,
            "tensor {:?} not zero-copy aligned",
            t.name
        );
    }
    eprintln!(
        "  wrote {} tensors across {} layers ({} bytes on disk)",
        reader.header().tensors.len(),
        n_layers,
        std::fs::metadata(output)?.len()
    );
    let _ = header;
    Ok(())
}

fn tensor_entry(
    name: &str,
    dtype: base_format::TensorDtype,
    shape: Vec<u64>,
    region: base_format::ComputeRegion,
    group_size: Option<u32>,
    _weight_len: Option<u64>,
    _layout: Option<base_format::Layout>,
) -> base_format::TensorEntry {
    base_format::TensorEntry {
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
        group_size,
        scale_dtype: None,
        symmetric: false,
        layout: None,
        residency: None,
        compute_region: region,
        flags: base_format::TensorFlags::empty(),
        checksum_xxh64: None,
        source_ggml_type: None,
    }
}

fn pack_for_target(
    weights: &[f32],
    target: TargetScheme,
) -> Result<(base_quant::Packed, base_format::TensorDtype)> {
    use base_format::TensorDtype;
    match target {
        TargetScheme::BaseQ2 => Ok((base_quant::base_q2::pack(weights), TensorDtype::BaseQ2)),
        TargetScheme::BaseQ3 => Ok((base_quant::base_q3::pack(weights), TensorDtype::BaseQ3)),
        TargetScheme::BaseQ4 => Ok((base_quant::base_q4::pack(weights), TensorDtype::BaseQ4)),
        TargetScheme::BaseQ5 => Ok((base_quant::base_q5::pack(weights), TensorDtype::BaseQ5)),
        TargetScheme::BaseQ6 => Ok((base_quant::base_q6::pack(weights), TensorDtype::BaseQ6)),
        TargetScheme::BaseQ8 => Ok((base_quant::base_q8::pack(weights), TensorDtype::BaseQ8)),
        TargetScheme::Bf16 => Ok((pack_bf16(weights), TensorDtype::Bf16)),
        TargetScheme::Mxfp4 => Ok((base_quant::mxfp4::pack(weights), TensorDtype::Mxfp4)),
        TargetScheme::Nvfp4 => Ok((base_quant::nvfp4::pack(weights), TensorDtype::Nvfp4)),
    }
}

/// Wrap fp32 weights as bf16 raw bytes (no quant, no scales).
fn pack_bf16(weights: &[f32]) -> base_quant::Packed {
    let bytes: Vec<u8> = weights
        .iter()
        .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
        .collect();
    base_quant::Packed {
        packed_weights: bytes,
        scales: vec![],
        biases: vec![],
        group_size: 0,
        scale_dtype: None,
    }
}

/// Profile-driven per-tensor quant context.
///
/// Holds the loaded profile + optional AWQ sidecar; produces
/// per-tensor packed bytes for canonical-quant bundles.
struct QuantContext {
    profile: Option<base_quant::QuantProfile>,
    awq_profile: Option<base_awq::AwqProfile>,
    awq_config: base_awq::AwqConfig,
    /// Fallback target when no profile is set.
    target: TargetScheme,
    /// Bypass the spec's already-quantized-source rejection.
    allow_quant_from_quant: bool,
}

impl QuantContext {
    fn from_args(args: &ConvertArgs) -> Result<Self> {
        let profile = match &args.profile {
            Some(p) => Some(
                base_quant::QuantProfile::from_path(p)
                    .with_context(|| format!("loading profile {}", p.display()))?,
            ),
            None => None,
        };
        let awq_profile = match &args.awq_profile {
            Some(p) => Some(
                base_awq::AwqProfile::load(p)
                    .with_context(|| format!("loading AWQ sidecar {}", p.display()))?,
            ),
            None => None,
        };
        if awq_profile.is_some() && profile.is_none() {
            bail!(
                "--awq-profile requires --profile (AWQ only applies to canonical bit-widths from a profile)"
            );
        }
        Ok(Self {
            profile,
            awq_profile,
            awq_config: base_awq::AwqConfig::default(),
            target: args.target,
            allow_quant_from_quant: args.allow_quant_from_quant,
        })
    }

    fn profile_name(&self) -> Option<&str> {
        self.profile.as_ref().map(|p| p.name.as_str())
    }

    /// Pack a single tensor. With a profile set, looks up the tensor
    /// name and applies the profile's per-tensor rule. Without a
    /// profile, falls back to the CLI `--target`. With AWQ
    /// sidecar + a canonical bit-width rule, runs AWQ search +
    /// rotation before pack_rtn.
    ///
    /// `in_features` is the in-features dim of the tensor (last dim
    /// for row-major weight matrices). Used for AWQ which operates
    /// per-input-channel.
    fn pack_tensor(
        &self,
        name: &str,
        weights: &[f32],
        in_features: Option<usize>,
    ) -> Result<(base_quant::Packed, base_format::TensorDtype)> {
        use base_format::TensorDtype;
        // Without a profile: legacy uniform-target behavior.
        let Some(profile) = &self.profile else {
            return pack_for_target(weights, self.target);
        };
        let resolved = profile
            .resolve_or_err(name)
            .with_context(|| format!("profile lookup for tensor {name:?}"))?;

        match resolved.dtype {
            TensorDtype::Bf16 => Ok((pack_bf16(weights), TensorDtype::Bf16)),
            TensorDtype::F16 => {
                let bytes: Vec<u8> = weights
                    .iter()
                    .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                    .collect();
                Ok((
                    base_quant::Packed {
                        packed_weights: bytes,
                        scales: vec![],
                        biases: vec![],
                        group_size: 0,
                        scale_dtype: None,
                    },
                    TensorDtype::F16,
                ))
            }
            TensorDtype::F32 => {
                let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok((
                    base_quant::Packed {
                        packed_weights: bytes,
                        scales: vec![],
                        biases: vec![],
                        group_size: 0,
                        scale_dtype: None,
                    },
                    TensorDtype::F32,
                ))
            }
            TensorDtype::Mxfp4 => {
                Ok((base_quant::mxfp4::pack(weights), TensorDtype::Mxfp4))
            }
            TensorDtype::Nvfp4 => {
                Ok((base_quant::nvfp4::pack(weights), TensorDtype::Nvfp4))
            }
            dtype @ (TensorDtype::BaseQ2
            | TensorDtype::BaseQ3
            | TensorDtype::BaseQ4
            | TensorDtype::BaseQ5
            | TensorDtype::BaseQ6
            | TensorDtype::BaseQ8) => self.pack_canonical(name, weights, in_features, dtype, resolved),
        }
    }

    fn pack_canonical(
        &self,
        name: &str,
        weights: &[f32],
        in_features: Option<usize>,
        dtype: base_format::TensorDtype,
        resolved: base_quant::ResolvedQuant,
    ) -> Result<(base_quant::Packed, base_format::TensorDtype)> {
        let bits = dtype.bits_per_weight().unwrap();
        let cfg = base_quant::RtnConfig {
            bits,
            group_size: resolved.group_size,
            symmetric: resolved.symmetric,
            scale_dtype: resolved.scale_dtype,
        };
        // Group-size divisibility check: tiny 1-D tensors (logit
        // softcap scalars, sliding-window mask params, etc.) can be
        // matched by a profile's catch-all rule that targets a quant
        // bit-width; we fall back to bf16 for those rather than
        // emitting an unloadable bundle.
        if weights.len() % cfg.group_size as usize != 0 {
            eprintln!(
                "    note: {name} has {} elements (not a multiple of gs={}); falling back to bf16",
                weights.len(),
                cfg.group_size
            );
            return Ok((pack_bf16(weights), base_format::TensorDtype::Bf16));
        }
        // Per-row divisibility check. Runtime GEMV/GEMM kernels assume
        // groups don't span row boundaries (each row is `K` weights +
        // `K/gs` scales). When K isn't a multiple of gs the flat pack
        // would silently produce groups that span rows; dequant would
        // read the wrong scale → offset-shifted output. Tiny model-glue
        // tensors (Gemma 3n AltUp coefs, K=4) can't satisfy gs=64 — fall
        // back to bf16 (no group structure) for those rather than
        // refusing the whole conversion. Earlier behavior bailed
        // outright; that broke Gemma 3n which has K=4 / K=16 AltUp
        // tensors that still need to land in the bundle.
        if let Some(k) = in_features {
            if k > 0 && k % cfg.group_size as usize != 0 {
                eprintln!(
                    "    note: {name} has in_features={k} (not a multiple of gs={}); \
                     falling back to bf16 — quant grouping would misalign scales.",
                    cfg.group_size
                );
                return Ok((pack_bf16(weights), base_format::TensorDtype::Bf16));
            }
        }

        // AWQ pre-process: only when sidecar carries an absmax for
        // this tensor and we know the in_features dim.
        let weights_for_pack: Vec<f32> = match (&self.awq_profile, in_features) {
            (Some(awq), Some(in_feat)) => {
                if let Some(absmax) = awq.absmax(name) {
                    if absmax.len() == in_feat {
                        let plan = self
                            .awq_config
                            .search(weights, in_feat, absmax, bits, cfg.group_size, cfg.symmetric);
                        // Rotate weights. The runtime undoes the rotation by
                        // pre-multiplying activations with `plan.scales`;
                        // those scales are stored alongside the rotated
                        // tensor in the `.base` header so the runtime can
                        // recover the original output.
                        eprintln!(
                            "    awq: {name} α={:.2} mse={:.4e}",
                            plan.alpha, plan.mse
                        );
                        base_awq::awq_apply(weights, in_feat, &plan.scales)
                    } else {
                        eprintln!(
                            "    awq: skipping {name} — sidecar absmax len {} != in_features {}",
                            absmax.len(),
                            in_feat
                        );
                        weights.to_vec()
                    }
                } else {
                    // Tensor not in sidecar → plain RTN.
                    weights.to_vec()
                }
            }
            _ => weights.to_vec(),
        };

        let packed = base_quant::pack_rtn(&weights_for_pack, cfg);
        Ok((packed, dtype))
    }
}

fn cmd_sign(args: SignArgs) -> Result<()> {
    let key_bytes = std::fs::read(&args.key)
        .with_context(|| format!("reading key {:?}", args.key))?;
    let key = base_sign::signing_key_from_bytes(&key_bytes)?;
    base_sign::sign_base_file(&args.input, &args.output, &key, &args.key_id)?;
    eprintln!("signed -> {}", args.output.display());
    Ok(())
}

fn cmd_verify(args: VerifyArgs) -> Result<()> {
    use ed25519_dalek::VerifyingKey;
    let bytes = std::fs::read(&args.pubkey)
        .with_context(|| format!("reading pubkey {:?}", args.pubkey))?;
    if bytes.len() != 32 {
        bail!("ed25519 public key must be 32 bytes, got {}", bytes.len());
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    let vk = VerifyingKey::from_bytes(&arr).context("parsing pubkey")?;
    base_sign::verify_base_file(&args.input, &vk)?;
    eprintln!("OK — {} verifies", args.input.display());
    Ok(())
}

fn cmd_inspect(args: InspectArgs) -> Result<()> {
    use base_format::BaseReader;
    let reader = BaseReader::open(&args.input)?;
    let h = reader.header();
    println!("arch:          {}", h.arch);
    println!("quant_scheme:  {:?}", h.quant_scheme);
    println!("min_hw:        {}", h.min_hw);
    println!("base_rt:       {}", h.base_rt_version);
    println!("created:       {}", h.created);
    println!("flags:         {:?}", h.flags);
    println!("n_layers:      {}", h.layers.len());
    println!("n_tensors:     {}", h.tensors.len());
    println!("signed:        {}", h.sig.is_some());

    let mut total: u64 = 0;
    for t in h.tensors.iter() {
        total += t.length;
    }
    println!("weights bytes: {}", total);

    let slots = reader.slots()?;
    println!("n_slots:       {}", slots.len());
    for s in &slots {
        println!("  slot kind={:?} raw=0x{:04x} len={}", s.kind(), s.kind_raw, s.payload.len());
    }

    if args.verify_checksums {
        eprintln!("verifying all tensor checksums...");
        for t in h.tensors.iter() {
            reader
                .verify_tensor(&t.name)
                .with_context(|| format!("verifying {:?}", t.name))?;
        }
        eprintln!("all checksums OK");
    }
    Ok(())
}

/// Minimal "current timestamp" without pulling in chrono. ISO 8601 UTC.
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", secs) // unix epoch string; real chrono formatting is a post-MVP nicety
}

#[cfg(test)]
mod canonical_name_tests {
    use super::{to_canonical_name, Canonical};

    fn main_canon(name: &str, arch: &str) -> Option<String> {
        match to_canonical_name(name, arch)? {
            Canonical::Main(s) => Some(s),
            Canonical::Mmproj(_) => None,
        }
    }

    // MLX Qwen3-MoE: `model.layers.N.mlp.switch_mlp.X_proj.weight`
    // (per-expert stack) must canonicalize to `layers.N.ffn_X_exps.weight`
    // without doubling the `.weight` suffix.
    #[test]
    fn mlx_qwen3_moe_split_experts_no_double_weight() {
        assert_eq!(
            main_canon("model.layers.5.mlp.switch_mlp.gate_proj.weight", "qwen3"),
            Some("layers.5.ffn_gate_exps.weight".to_string()),
        );
        assert_eq!(
            main_canon("model.layers.5.mlp.switch_mlp.up_proj.weight", "qwen3"),
            Some("layers.5.ffn_up_exps.weight".to_string()),
        );
        assert_eq!(
            main_canon("model.layers.5.mlp.switch_mlp.down_proj.weight", "qwen3"),
            Some("layers.5.ffn_down_exps.weight".to_string()),
        );
    }

    // MLX Gemma 4 MoE: `model.layers.N.experts.switch_glu.X_proj.weight`
    // (no `mlp.` parent). Switch_glu→experts canonicalization plus the
    // .weight-suffix-aware experts rename should produce the canonical
    // ffn_*_exps name.
    #[test]
    fn mlx_gemma4_moe_switch_glu_split_experts() {
        assert_eq!(
            main_canon(
                "model.layers.7.experts.switch_glu.gate_proj.weight",
                "gemma4",
            ),
            Some("layers.7.ffn_gate_exps.weight".to_string()),
        );
        assert_eq!(
            main_canon(
                "model.layers.7.experts.switch_glu.down_proj.weight",
                "gemma4",
            ),
            Some("layers.7.ffn_down_exps.weight".to_string()),
        );
    }

    // HF mainline Gemma 4 26B-A4B uses `model.language_model.layers.N.experts.X_proj`
    // bare (no `.weight` suffix; gate+up are 3D fused into a single tensor).
    #[test]
    fn hf_mainline_gemma4_moe_fused_bare() {
        assert_eq!(
            main_canon(
                "model.language_model.layers.0.experts.gate_up_proj",
                "gemma4",
            ),
            Some("layers.0.ffn_gate_up_exps.weight".to_string()),
        );
        assert_eq!(
            main_canon("model.language_model.layers.0.experts.down_proj", "gemma4"),
            Some("layers.0.ffn_down_exps.weight".to_string()),
        );
    }

    // The shared-expert path (Qwen2-MoE / some Qwen3 variants) uses
    // `mlp.shared_expert.X_proj.weight`. The experts-rename rules must NOT
    // touch it — shared-expert tensors map through `mlp.shared_expert.*`
    // aliases on the runtime side.
    #[test]
    fn shared_expert_not_rewritten_by_experts_rules() {
        assert_eq!(
            main_canon(
                "model.layers.3.mlp.shared_expert.gate_proj.weight",
                "qwen3",
            ),
            Some("layers.3.mlp.shared_expert.gate_proj.weight".to_string()),
        );
    }

    // Dense Llama-style FFN (`mlp.gate_proj.weight`, no `experts.` segment)
    // must not be confused with MoE experts.
    #[test]
    fn dense_mlp_not_rewritten_by_experts_rules() {
        assert_eq!(
            main_canon("model.layers.2.mlp.gate_proj.weight", "llama"),
            Some("layers.2.mlp.gate_proj.weight".to_string()),
        );
    }

    // Router naming is preserved on the canonical side: HF `mlp.gate.weight`
    // (Qwen3-MoE) and Gemma 4 `router.proj.weight` both end up at
    // `mlp.router.weight`, which the runtime alias resolves to
    // `ffn_gate_inp.weight`. HF mainline `router.per_expert_scale` becomes
    // `ffn_down_exps.scale` directly.
    #[test]
    fn moe_router_renames() {
        assert_eq!(
            main_canon("model.layers.4.router.proj.weight", "gemma4"),
            Some("layers.4.mlp.router.weight".to_string()),
        );
        assert_eq!(
            main_canon("model.layers.4.router.per_expert_scale", "gemma4"),
            Some("layers.4.ffn_down_exps.scale".to_string()),
        );
        assert_eq!(
            main_canon("model.layers.4.router.scale", "gemma4"),
            Some("layers.4.ffn_gate_inp.scale".to_string()),
        );
    }

    /// nomic-bert HF safetensors use BERT-flavored names (`emb_ln`,
    /// `encoder.layers.N.attn.Wqkv`, `mlp.fc11/fc12/fc2`, `norm1/2`).
    /// Each must land at the same canonical name the GGUF NomicBertMapper
    /// emits so the runtime path is identical for both sources.
    #[test]
    fn nomic_bert_hf_canonical_names() {
        let cases: &[(&str, &str)] = &[
            ("embeddings.word_embeddings.weight", "embed_tokens.weight"),
            ("embeddings.token_type_embeddings.weight", "token_types.weight"),
            ("emb_ln.weight", "token_embd_norm.weight"),
            ("emb_ln.bias", "token_embd_norm.bias"),
            (
                "encoder.layers.0.attn.Wqkv.weight",
                "layers.0.self_attn.qkv_proj.weight",
            ),
            (
                "encoder.layers.11.attn.out_proj.weight",
                "layers.11.self_attn.o_proj.weight",
            ),
            (
                "encoder.layers.5.mlp.fc11.weight",
                "layers.5.mlp.gate_proj.weight",
            ),
            (
                "encoder.layers.5.mlp.fc12.weight",
                "layers.5.mlp.up_proj.weight",
            ),
            (
                "encoder.layers.5.mlp.fc2.weight",
                "layers.5.mlp.down_proj.weight",
            ),
            (
                "encoder.layers.0.norm1.weight",
                "layers.0.attn_output_norm.weight",
            ),
            (
                "encoder.layers.0.norm1.bias",
                "layers.0.attn_output_norm.bias",
            ),
            (
                "encoder.layers.0.norm2.weight",
                "layers.0.layer_output_norm.weight",
            ),
            (
                "encoder.layers.0.norm2.bias",
                "layers.0.layer_output_norm.bias",
            ),
        ];
        for (src, want) in cases {
            assert_eq!(
                main_canon(src, "nomic-bert"),
                Some(want.to_string()),
                "nomic rename for {src}",
            );
        }
        // Unrecognized sentence-transformers head tensors drop to None.
        assert_eq!(
            main_canon("embeddings.position_embeddings.weight", "nomic-bert"),
            None,
        );
        assert_eq!(main_canon("0.auto_model.pooler.dense.weight", "nomic-bert"), None);
    }
}
