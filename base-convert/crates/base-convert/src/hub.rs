//! `basert pull` / `basert list` — the model-hub CLI surface — plus
//! `dispatch_external`, the launcher that forwards `basert <cmd>` (serve, chat,
//! …) to the matching `basert-<cmd>` runtime binary.
//!
//! Resolution and storage live in the `base-hub` crate; this module is the
//! glue that drives it from the CLI and, for convert-on-pull, hands the
//! downloaded snapshot to the existing `cmd_convert` pipeline.

use crate::{AwqMode, ConvertArgs, ListArgs, PullArgs, TargetScheme};
use anyhow::{bail, Context, Result};
use base_hub::cache::{self, HubSidecar};
use base_hub::fetch::{self, Fetcher, HfFetcher};
use base_hub::registry::{MergedRegistry, ModelEntry, ModelRef, Registry, SourceKind};
use std::path::{Path, PathBuf};
use std::process::Command;

// Only generic, arch-agnostic profiles ship in the (public) binary. Tuned
// per-arch profiles are intentionally NOT bundled — tuned quality reaches
// users through pre-converted catalog artifacts, not by exposing the recipe.
const PROFILE_Q4: &str = include_str!("../../../profiles/default-q4.json");
const PROFILE_Q6: &str = include_str!("../../../profiles/default-q6.json");
const PROFILE_Q8: &str = include_str!("../../../profiles/default-q8.json");

fn bundled_profile_for_target(target: TargetScheme) -> Option<(&'static str, &'static str)> {
    match target {
        TargetScheme::BaseQ4 => Some(("default-q4", PROFILE_Q4)),
        TargetScheme::BaseQ6 => Some(("default-q6", PROFILE_Q6)),
        TargetScheme::BaseQ8 => Some(("default-q8", PROFILE_Q8)),
        _ => None,
    }
}

fn target_str(target: TargetScheme) -> &'static str {
    match target {
        TargetScheme::BaseQ2 => "base_q2",
        TargetScheme::BaseQ3 => "base_q3",
        TargetScheme::BaseQ4 => "base_q4",
        TargetScheme::BaseQ5 => "base_q5",
        TargetScheme::BaseQ6 => "base_q6",
        TargetScheme::BaseQ8 => "base_q8",
        TargetScheme::Bf16 => "bf16",
        TargetScheme::Mxfp4 => "mxfp4",
        TargetScheme::Nvfp4 => "nvfp4",
    }
}

/// Source files worth downloading for a safetensors conversion. Mirrors
/// mlx-lm's allow_patterns; deliberately excludes `.bin`/`.pth`/`.gguf`.
fn want_source_file(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    base.ends_with(".safetensors")
        || base.ends_with(".json")
        || base.ends_with(".jinja")
        || base.ends_with(".txt")
        || base.ends_with(".model")
        || base.starts_with("tokenizer")
}

/// Lowercased quant tag — the segment after the last `-`/`_`. Maps a profile
/// name, `--target` string, or `.base` filename stem to a comparable token:
/// `default-q4` → `q4`, `base_q8` → `q8`, `Llama-3.2-1B-Instruct-Q4` → `q4`,
/// `bf16` → `bf16`.
fn quant_tag(s: &str) -> String {
    s.rsplit(['-', '_']).next().unwrap_or(s).to_ascii_lowercase()
}

/// True when `tag` names a quant scheme we know how to label on disk.
fn is_quant_tag(tag: &str) -> bool {
    matches!(
        tag,
        "q2" | "q3" | "q4" | "q5" | "q6" | "q8" | "bf16" | "mxfp4" | "nvfp4"
    )
}

/// The quant the user is asking for, derived from `--profile` (its filename)
/// or, failing that, `--target`.
fn quant_token(args: &PullArgs) -> String {
    if let Some(p) = &args.profile {
        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
            return quant_tag(stem);
        }
    }
    quant_tag(target_str(args.target))
}

/// `path/to/Foo-Q4.base` → `Foo-Q4`.
fn base_file_stem(f: &str) -> String {
    let b = f.rsplit('/').next().unwrap_or(f);
    b.strip_suffix(".base").unwrap_or(b).to_string()
}

/// List the `.base` artifacts a repo hosts (empty when it ships none).
fn list_base_files(fetcher: &dyn Fetcher, repo: &str, revision: &str) -> Result<Vec<String>> {
    let files = fetcher
        .list_files(repo, revision)
        .with_context(|| format!("listing files in {repo}@{revision}"))?;
    Ok(files.into_iter().filter(|f| f.ends_with(".base")).collect())
}

/// Pick the `.base` whose quant tag matches `want`. Falls back to the sole
/// artifact when the repo has exactly one; errors when several are present and
/// none match.
fn select_base_file<'a>(files: &'a [String], want: &str) -> Result<&'a String> {
    if let Some(f) = files.iter().find(|f| quant_tag(&base_file_stem(f)) == want) {
        return Ok(f);
    }
    if let [only] = files {
        return Ok(only);
    }
    let avail: Vec<String> = files.iter().map(|f| quant_tag(&base_file_stem(f))).collect();
    bail!(
        "no pre-converted .base for quant {want:?} in this repo; it offers: {}",
        avail.join(", ")
    )
}

/// Decide whether `token` should be treated as a hub model id rather than a
/// filesystem path. Hub ids are HuggingFace-style `namespace/name` (optionally
/// `:variant`); anything that already exists on disk, names a `.base` file, or
/// is an explicit relative/absolute/home path is left for the runtime to open.
fn looks_like_hub_id(token: &str) -> bool {
    if token.is_empty() || Path::new(token).exists() {
        return false;
    }
    if token.starts_with(['.', '/', '~']) || token.ends_with(".base") {
        return false;
    }
    let id = token.split_once(':').map_or(token, |(id, _)| id);
    id.contains('/')
}

/// Map a requested id to the id we should actually pull, preferring a
/// pre-converted artifact in the basecompute org. `Qwen/Qwen3-0.6B` →
/// `basecompute/Qwen3-0.6B` when that's catalogued; otherwise the id is
/// returned unchanged (convert-on-pull from the source repo).
fn preconverted_id(reg: &MergedRegistry, id: &str) -> String {
    if id.starts_with("basecompute/") {
        return id.to_string();
    }
    let name = id.rsplit('/').next().unwrap_or(id);
    let candidate = format!("basecompute/{name}");
    if reg.catalog.resolve(&candidate).is_some() {
        candidate
    } else {
        id.to_string()
    }
}

/// Map a quant tag (`q4`, `q8`, …) to the matching convert target.
fn target_from_quant(tag: &str) -> TargetScheme {
    match tag {
        "q2" => TargetScheme::BaseQ2,
        "q3" => TargetScheme::BaseQ3,
        "q5" => TargetScheme::BaseQ5,
        "q6" => TargetScheme::BaseQ6,
        "q8" => TargetScheme::BaseQ8,
        "bf16" => TargetScheme::Bf16,
        "mxfp4" => TargetScheme::Mxfp4,
        "nvfp4" => TargetScheme::Nvfp4,
        _ => TargetScheme::BaseQ4,
    }
}

/// The installed artifact path for `id`, if exactly one variant is present.
/// `Ok(None)` when not installed; errors when several variants exist (the
/// caller must disambiguate with `id:<variant>`).
fn installed_single_path(reg: &MergedRegistry, id: &str) -> Result<Option<PathBuf>> {
    let installed = reg.local.list()?;
    let hits: Vec<&ModelEntry> = installed.iter().filter(|r| r.id == id).collect();
    match hits.as_slice() {
        [] => Ok(None),
        [one] => Ok(Some(
            one.path
                .clone()
                .with_context(|| format!("installed model `{id}` has no artifact path"))?,
        )),
        many => {
            let variants = many.iter().map(|r| r.variant.as_str()).collect::<Vec<_>>().join(", ");
            bail!("model `{id}` has multiple installed variants ({variants}) — specify one as `{id}:<variant>`")
        }
    }
}

/// Fetch a not-yet-installed model on demand, then return its artifact path.
/// Prefers the pre-converted basecompute mirror; otherwise converts the source
/// repo on pull. Progress (download + quantization) is shown by `cmd_pull`.
fn auto_pull_and_resolve(reg: &MergedRegistry, id: &str, want_variant: Option<&str>) -> Result<PathBuf> {
    let pull_id = preconverted_id(reg, id);
    let target = want_variant
        .map(|v| target_from_quant(&quant_tag(v)))
        .unwrap_or(TargetScheme::BaseQ4);

    if pull_id != id {
        eprintln!("{id}: not installed — using pre-converted {pull_id}");
    } else {
        eprintln!("{id}: not installed — fetching (download + convert if needed)");
    }

    cmd_pull(PullArgs {
        id: pull_id.clone(),
        profile: None,
        target,
        revision: "main".to_string(),
        force: false,
        dry_run: false,
    })
    .with_context(|| format!("auto-fetching {pull_id}"))?;

    // Re-scan and return the freshly-installed artifact (registered under the
    // pulled id, which may differ from the requested one).
    let reg2 = MergedRegistry::load()?;
    if let Some(v) = want_variant {
        if let Some(p) = reg2.local.installed_path(&pull_id, v) {
            return Ok(p);
        }
    }
    installed_single_path(&reg2, &pull_id)?
        .with_context(|| format!("fetched {pull_id} but could not locate the installed artifact"))
}

/// Resolve a hub model reference to the installed `.base` artifact path.
///
/// The variant (quant build) is chosen by an inline `org/model:default-q8` on
/// the token, or by the `--variant` launcher flag (`default_variant`) for
/// tokens without an inline `:variant` — the inline form wins. With neither,
/// the sole installed variant is used (and it's an error to resolve a model
/// with several installed without naming one). An uninstalled model is fetched
/// on demand — preferring the pre-converted basecompute mirror, else converting
/// the source repo — so `basert chat`/`serve <id>` Just Works.
fn resolve_hub_model(token: &str, default_variant: Option<&str>) -> Result<PathBuf> {
    let (id, inline) = token.split_once(':').map_or((token, None), |(i, v)| (i, Some(v)));

    // A trailing/empty `:` is a typo, not "default variant" — fail loudly so it
    // doesn't silently resolve to q4.
    if inline == Some("") {
        bail!(
            "empty variant after ':' in `{token}` — name a variant \
             (e.g. `{id}:default-q8`) or drop the colon. Run `basert list` to \
             see installed variants."
        );
    }

    // Inline `:variant` wins over the `--variant` flag.
    let variant = inline.or(default_variant);
    let reg = MergedRegistry::load()?;

    if let Some(v) = variant {
        if let Some(p) = reg.local.installed_path(id, v) {
            return Ok(p);
        }
        return auto_pull_and_resolve(&reg, id, Some(v));
    }

    if let Some(p) = installed_single_path(&reg, id)? {
        return Ok(p);
    }
    auto_pull_and_resolve(&reg, id, None)
}

/// Rewrite every hub-id-shaped argument to the installed `.base` path so that
/// `basert chat org/model` (positional) and `basert serve --model org/model`
/// (repeatable flag) both Just Work after `basert pull`. `default_variant`
/// comes from the `--variant` launcher flag and applies to ids without an
/// inline `:variant`.
fn resolve_model_args(rest: &[String], default_variant: Option<&str>) -> Result<Vec<String>> {
    rest.iter()
        .map(|arg| {
            if looks_like_hub_id(arg) {
                Ok(resolve_hub_model(arg, default_variant)?.to_string_lossy().into_owned())
            } else {
                Ok(arg.clone())
            }
        })
        .collect()
}

/// Pull a `--variant <v>` / `--variant=<v>` selector out of the forwarded args,
/// returning the variant (if any) and the args with it removed — the runtime
/// binary never sees a flag it doesn't understand. The selector applies to
/// hub-id model args that don't already carry an inline `:variant`.
fn extract_variant_flag(rest: &[String]) -> Result<(Option<String>, Vec<String>)> {
    let mut variant: Option<String> = None;
    let mut out: Vec<String> = Vec::with_capacity(rest.len());
    let mut i = 0;
    while i < rest.len() {
        let a = &rest[i];
        if a == "--variant" {
            let v = rest
                .get(i + 1)
                .context("`--variant` needs a value, e.g. `--variant default-q8`")?;
            if v.is_empty() {
                bail!("`--variant` value is empty");
            }
            variant = Some(v.clone());
            i += 2;
            continue;
        }
        if let Some(v) = a.strip_prefix("--variant=") {
            if v.is_empty() {
                bail!("`--variant=` value is empty");
            }
            variant = Some(v.to_string());
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    Ok((variant, out))
}

/// Forward `basert <cmd> [args…]` to the matching runtime binary. Searches for
/// `basert-<cmd>` (release layout) then `baseRT_<cmd>` (local dev build),
/// looking next to this executable first and then on `PATH`. On success the
/// current process is replaced (`exec`), so this only returns on failure.
pub fn dispatch_external(argv: Vec<String>) -> Result<()> {
    use std::os::unix::process::CommandExt;

    let (cmd, rest) = argv.split_first().context("no command given")?;
    // `--variant <v>` is a launcher-level model selector; strip it before
    // forwarding (the runtime binary doesn't know it) and apply it during
    // hub-id resolution.
    let (variant_flag, rest) = extract_variant_flag(rest)?;
    let rest = resolve_model_args(&rest, variant_flag.as_deref())?;
    let candidates = [format!("basert-{cmd}"), format!("baseRT_{cmd}")];

    // Prefer a binary sitting next to `basert` (how the release ships); fall
    // back to a bare name, which `Command` resolves against `PATH`.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let mut targets: Vec<PathBuf> = Vec::new();
    if let Some(dir) = &exe_dir {
        for name in &candidates {
            let p = dir.join(name);
            if p.is_file() {
                targets.push(p);
            }
        }
    }
    targets.extend(candidates.iter().map(PathBuf::from));

    for target in &targets {
        // exec() returns only on failure; ENOENT means try the next candidate.
        let err = Command::new(target).args(&rest).exec();
        if err.kind() != std::io::ErrorKind::NotFound {
            return Err(err).with_context(|| format!("launching {}", target.display()));
        }
    }
    if RUNTIME_COMMANDS.contains(&cmd.as_str()) {
        bail!(
            "`basert {cmd}` needs the BaseRT runtime, which wasn't found in this \
             installation.\n\
             Reinstall the full BaseRT package, or run `basert --help`."
        )
    }
    bail!("unknown command `basert {cmd}`.\nRun `basert --help` to see available commands.")
}

/// Commands served by the BaseRT runtime rather than this binary. Used only
/// to shape the not-found error above; dispatch itself is name-driven, so
/// commands absent from this list (or from `basert --help`) still dispatch.
const RUNTIME_COMMANDS: [&str; 6] =
    ["serve", "chat", "complete", "bench", "transcribe", "profile"];

pub fn cmd_pull(args: PullArgs) -> Result<()> {
    let reg = MergedRegistry::load()?;
    let root = reg.root.clone();
    let want = quant_token(&args);
    let r = reg.resolve(&args.id, &args.revision, Some(&want), args.force)?;

    if args.dry_run {
        print_plan(&r, &want);
        return Ok(());
    }

    match &r {
        ModelRef::Local { id, variant, path } => {
            eprintln!(
                "{id} [{variant}] already installed: {}\n  (use --force to re-pull)",
                path.display()
            );
            Ok(())
        }
        // The catalog advertises one quant per id (the recommended default).
        // Serve it directly when it's what the user wants; otherwise grab the
        // requested quant straight from the same repo rather than silently
        // handing back the cataloged one.
        ModelRef::Catalog { id, hf_repo, revision, variant, .. } => {
            if quant_tag(variant) == want {
                pull_catalog(&root, &r)
            } else {
                let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;
                let base_files = list_base_files(&fetcher, hf_repo, revision)?;
                pull_base_direct(&root, &args, id, hf_repo, revision, &fetcher, &base_files)
            }
        }
        // A raw HF repo: if it already hosts pre-converted `.base` artifacts
        // (e.g. the basecompute org), download the matching one directly — no
        // local conversion. Otherwise treat it as source safetensors.
        ModelRef::HuggingFace { id, repo, revision } => {
            let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;
            let base_files = list_base_files(&fetcher, repo, revision)?;
            if base_files.is_empty() {
                pull_and_convert(&root, &args, id, repo, revision)
            } else {
                pull_base_direct(&root, &args, id, repo, revision, &fetcher, &base_files)
            }
        }
    }
}

fn print_plan(r: &ModelRef, want: &str) {
    match r {
        ModelRef::Local { id, variant, path } => {
            println!("plan: {id} [{variant}] already installed at {}", path.display())
        }
        ModelRef::Catalog { id, hf_repo, file, revision, variant, .. } => {
            if quant_tag(variant) == want {
                println!(
                    "plan: pull pre-converted {id} [{variant}] from {hf_repo}/{file}@{revision} (no conversion)"
                )
            } else {
                println!(
                    "plan: pull pre-converted {id} from {hf_repo}@{revision} \
                     selecting the {want} .base (catalog default is {variant}; no conversion)"
                )
            }
        }
        ModelRef::HuggingFace { id, repo, revision } => println!(
            "plan: fetch {repo}@{revision} → {id} \
             (download pre-converted .base if the repo has one, else convert source locally)"
        ),
    }
}

/// Fast path: download a pre-converted `.base` from the catalog.
fn pull_catalog(root: &Path, r: &ModelRef) -> Result<()> {
    let ModelRef::Catalog {
        id,
        hf_repo,
        file,
        revision,
        variant,
        sha256,
        ..
    } = r
    else {
        unreachable!("pull_catalog called with non-catalog ref");
    };
    eprintln!("basert pull v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  catalog: {hf_repo}/{file}@{revision} (pre-converted)");

    let fetcher = HfFetcher::new(cache::hf_staging_dir(root))?;
    let src = fetcher.get_file(hf_repo, revision, file)?;

    let vdir = cache::variant_dir(root, id, variant)?;
    std::fs::create_dir_all(&vdir)?;
    let out = cache::base_artifact_path(&vdir);
    // Moves the staged download into place (same filesystem), so the pulled
    // artifact exists exactly once on disk.
    fetch::install_file(&fetcher, hf_repo, &src, &out)?;

    let got_sha = crate::compute_sha256_streaming(&out)?;
    if let Some(expected) = sha256 {
        if !got_sha.eq_ignore_ascii_case(expected) {
            std::fs::remove_file(&out).ok();
            // The staged bytes are equally bad — no resume value in them, so
            // clear them and let a retry start from scratch.
            fetch::cleanup_staging(&fetcher, hf_repo);
            bail!("sha256 mismatch for {id}: expected {expected}, got {got_sha}");
        }
        eprintln!("  sha256:  verified");
    }

    write_sidecar_for(
        &vdir,
        id,
        "catalog",
        hf_repo,
        None,
        revision,
        variant,
        None,
        Some(got_sha),
    )?;
    fetch::cleanup_staging(&fetcher, hf_repo);
    eprintln!("installed {id} [{variant}] → {}", out.display());
    Ok(())
}

/// Fast path for any HF repo that already hosts pre-converted `.base`
/// artifacts (the basecompute org, or anyone's): download the one matching
/// the requested quant directly, no local conversion. This is what makes
/// `basert pull <org>/<model>` work for `.base`-only repos that carry no
/// safetensors + `config.json`.
#[allow(clippy::too_many_arguments)]
fn pull_base_direct(
    root: &Path,
    args: &PullArgs,
    id: &str,
    repo: &str,
    revision: &str,
    fetcher: &dyn Fetcher,
    base_files: &[String],
) -> Result<()> {
    eprintln!("basert pull v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  source:  {repo}@{revision} (HuggingFace, pre-converted .base)");

    let want = quant_token(args);
    let file = select_base_file(base_files, &want)?;
    // Label the on-disk variant by the artifact's own quant when it carries
    // one (so a `-Q8.base` never lands in a `default-q4` dir); otherwise fall
    // back to what was requested.
    let file_tag = quant_tag(&base_file_stem(file));
    let variant_tag = if is_quant_tag(&file_tag) { file_tag } else { want };
    let variant = format!("default-{variant_tag}");
    eprintln!("  variant: {variant}");
    eprintln!("  file:    {file}");

    let vdir = cache::variant_dir(root, id, &variant)?;
    std::fs::create_dir_all(&vdir)?;
    let out = cache::base_artifact_path(&vdir);

    let src = fetcher.get_file(repo, revision, file)?;
    // Moves the staged download into place (same filesystem), so the pulled
    // artifact exists exactly once on disk.
    fetch::install_file(fetcher, repo, &src, &out)?;

    let sha = crate::compute_sha256_streaming(&out).ok();
    write_sidecar_for(&vdir, id, "huggingface", repo, None, revision, &variant, None, sha)?;
    fetch::cleanup_staging(fetcher, repo);
    eprintln!("installed {id} [{variant}] → {}", out.display());
    Ok(())
}

/// Convert-on-pull: download an HF repo's source safetensors and run the
/// existing conversion pipeline into the cache.
fn pull_and_convert(
    root: &Path,
    args: &PullArgs,
    id: &str,
    repo: &str,
    revision: &str,
) -> Result<()> {
    eprintln!("basert pull v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  source:  {repo}@{revision} (HuggingFace, convert-on-pull)");

    // Pick the profile + variant name.
    let (variant, profile_path, _tmp) = choose_profile(args)?;
    eprintln!("  variant: {variant}");

    let vdir = cache::variant_dir(root, id, &variant)?;
    let out = cache::base_artifact_path(&vdir);
    std::fs::create_dir_all(&vdir)?;

    let fetcher = HfFetcher::new(cache::hf_staging_dir(root))?;
    let snapshot = download_source(repo, revision, &fetcher)?;

    let conv = ConvertArgs {
        input: snapshot,
        output: Some(out.clone()),
        target: args.target,
        calib: None,
        calib_tokens: 512,
        awq_mode: AwqMode::Full,
        synthetic: false,
        profile: profile_path,
        awq_profile: None,
        allow_quant_from_quant: false,
    };
    crate::cmd_convert(conv).with_context(|| format!("converting {repo}"))?;

    let sha = crate::compute_sha256_streaming(&out).ok();
    write_sidecar_for(
        &vdir,
        id,
        "huggingface",
        repo,
        Some(repo),
        revision,
        &variant,
        Some(&variant),
        sha,
    )?;
    // The staged source snapshot is now a redundant multi-GB copy of a model
    // we keep in converted form (provenance lives in hub.json; a re-convert
    // can re-fetch). Drop it unless the user opted to keep sources — keeping
    // them makes a later pull of another quant of the same repo skip the
    // re-download.
    if keep_hf_sources() {
        eprintln!(
            "  keeping HF source snapshot under {} (BASERT_KEEP_HF_SOURCES)",
            cache::hf_staging_dir(root).display()
        );
    } else {
        fetch::cleanup_staging(&fetcher, repo);
    }
    eprintln!("installed {id} [{variant}] → {}", out.display());
    Ok(())
}

/// Whether convert-on-pull should keep the downloaded HF source snapshot
/// after a successful conversion. Off by default: the converted `.base` is
/// the artifact users asked for, and the safetensors sources would otherwise
/// linger as a second multi-GB copy. `BASERT_KEEP_HF_SOURCES=1` (anything but
/// `0`/`false`/empty) trades that disk for faster re-pulls of other quants.
fn keep_hf_sources() -> bool {
    std::env::var("BASERT_KEEP_HF_SOURCES")
        .map(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Returns `(variant, profile_path, tempfile_guard)`. The guard keeps a
/// bundled-profile tempfile alive until conversion finishes.
fn choose_profile(
    args: &PullArgs,
) -> Result<(String, Option<PathBuf>, Option<tempfile::NamedTempFile>)> {
    if let Some(p) = &args.profile {
        let variant = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("custom")
            .to_string();
        return Ok((variant, Some(p.clone()), None));
    }
    if let Some((name, json)) = bundled_profile_for_target(args.target) {
        let mut tf = tempfile::Builder::new()
            .prefix("base-profile-")
            .suffix(".json")
            .tempfile()?;
        std::io::Write::write_all(&mut tf, json.as_bytes())?;
        let path = tf.path().to_path_buf();
        return Ok((name.to_string(), Some(path), Some(tf)));
    }
    // No generic profile for this scheme → flat --target, variant = scheme.
    Ok((target_str(args.target).to_string(), None, None))
}

/// Read `model_type` from a fetched `config.json` and fail early when no
/// converter supports it — so an unsupported repo errors before its weights
/// are downloaded. `text_config.model_type` is consulted as a fallback for
/// multimodal configs that nest the language-model arch there.
fn check_supported_arch(config_path: &Path, repo: &str, revision: &str) -> Result<()> {
    let bytes = std::fs::read(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as JSON", config_path.display()))?;

    let model_type = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| cfg.pointer("/text_config/model_type").and_then(|v| v.as_str()))
        .ok_or_else(|| {
            anyhow::anyhow!("{repo}@{revision}: config.json has no model_type field")
        })?;

    if base_arch::hf_mapper_for_model_type(model_type).is_some() {
        return Ok(());
    }

    bail!(
        "{repo}@{revision}: model_type {model_type:?} is not supported by convert-on-pull.\n\
         Supported architectures: {}.\n\
         Pre-converted models are available via `basert list --remote`.",
        base_arch::SUPPORTED_HF_MODEL_TYPES.join(", ")
    )
}

/// Download the wanted source files into hf-hub's snapshot dir and return it.
fn download_source(repo: &str, revision: &str, fetcher: &dyn Fetcher) -> Result<PathBuf> {
    let files = fetcher.list_files(repo, revision)?;
    let wanted: Vec<&String> = files.iter().filter(|f| want_source_file(f)).collect();
    if !wanted.iter().any(|f| f.ends_with("config.json")) {
        bail!("{repo}@{revision} has no config.json — not an HF safetensors model");
    }
    if !wanted.iter().any(|f| f.ends_with(".safetensors")) {
        bail!(
            "{repo}@{revision} has no .safetensors weights (only safetensors source is supported)"
        );
    }

    // Anchor the snapshot dir on config.json, then fetch the rest beside it.
    let cfg = fetcher.get_file(repo, revision, "config.json")?;
    // Pre-flight: reject unsupported architectures from config.json alone,
    // before downloading multi-GB safetensors. Mirrors the model_type read
    // in `convert_hf` so the gate matches the eventual conversion.
    check_supported_arch(&cfg, repo, revision)?;
    let snapshot = cfg
        .parent()
        .map(|p| p.to_path_buf())
        .context("config.json has no parent dir")?;
    for f in &wanted {
        if f.as_str() == "config.json" {
            continue;
        }
        fetcher.get_file(repo, revision, f)?;
    }
    Ok(snapshot)
}

#[allow(clippy::too_many_arguments)]
fn write_sidecar_for(
    vdir: &Path,
    id: &str,
    source_kind: &str,
    hf_repo: &str,
    source_repo: Option<&str>,
    revision: &str,
    variant: &str,
    profile: Option<&str>,
    base_sha256: Option<String>,
) -> Result<()> {
    cache::write_sidecar(
        vdir,
        &HubSidecar {
            id: id.to_string(),
            source_kind: source_kind.to_string(),
            hf_repo: hf_repo.to_string(),
            source_repo: source_repo.map(|s| s.to_string()),
            revision: revision.to_string(),
            variant: variant.to_string(),
            profile: profile.map(|s| s.to_string()),
            pulled_at: crate::chrono_now(),
            base_sha256,
        },
    )
}

pub fn cmd_list(args: ListArgs) -> Result<()> {
    let reg = MergedRegistry::load()?;
    let rows = reg.list(args.remote)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if rows.is_empty() {
        eprintln!("No models in {}.", reg.root.display());
        eprintln!("Pull one with:  basert pull <org/model>");
        return Ok(());
    }
    print_table(&rows);
    Ok(())
}

fn print_table(rows: &[ModelEntry]) {
    let dash = "-".to_string();
    let fmt_size = |b: Option<u64>| match b {
        Some(n) => human_size(n),
        None => dash.clone(),
    };
    let status = |r: &ModelEntry| match r.source_kind {
        SourceKind::Local => "installed".to_string(),
        SourceKind::Catalog => "available".to_string(),
        SourceKind::HuggingFace => "remote".to_string(),
    };

    let mut id_w = "ID".len();
    let mut arch_w = "ARCH".len();
    let mut quant_w = "QUANT".len();
    let mut size_w = "SIZE".len();
    for r in rows {
        id_w = id_w.max(r.id.len());
        arch_w = arch_w.max(r.arch.as_deref().unwrap_or("-").len());
        quant_w = quant_w.max(r.quant.as_deref().unwrap_or("-").len());
        size_w = size_w.max(fmt_size(r.size_bytes).len());
    }
    println!(
        "{:<id_w$}  {:<arch_w$}  {:<quant_w$}  {:>size_w$}  STATUS",
        "ID", "ARCH", "QUANT", "SIZE"
    );
    for r in rows {
        println!(
            "{:<id_w$}  {:<arch_w$}  {:<quant_w$}  {:>size_w$}  {}",
            r.id,
            r.arch.as_deref().unwrap_or("-"),
            r.quant.as_deref().unwrap_or("-"),
            fmt_size(r.size_bytes),
            status(r),
        );
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_hub::fetch::MockFetcher;

    #[test]
    fn quant_tag_extracts_last_segment() {
        assert_eq!(quant_tag("default-q4"), "q4");
        assert_eq!(quant_tag("base_q8"), "q8");
        assert_eq!(quant_tag("Llama-3.2-1B-Instruct-Q4"), "q4");
        assert_eq!(quant_tag("bf16"), "bf16");
        assert_eq!(quant_tag("model"), "model");
    }

    #[test]
    fn base_file_stem_strips_dir_and_ext() {
        assert_eq!(base_file_stem("Foo-Q4.base"), "Foo-Q4");
        assert_eq!(base_file_stem("sub/dir/Foo-Q8.base"), "Foo-Q8");
        assert_eq!(base_file_stem("model.base"), "model");
    }

    #[test]
    fn select_base_file_matches_quant_then_falls_back() {
        let files = vec![
            "Llama-3.2-1B-Instruct-Q4.base".to_string(),
            "Llama-3.2-1B-Instruct-Q8.base".to_string(),
        ];
        assert_eq!(select_base_file(&files, "q4").unwrap(), &files[0]);
        assert_eq!(select_base_file(&files, "q8").unwrap(), &files[1]);
        // No match among several → error that lists what's available.
        let err = select_base_file(&files, "q2").unwrap_err().to_string();
        assert!(err.contains("q4") && err.contains("q8"), "{err}");
        // Sole artifact → used regardless of requested quant.
        let one = vec!["model.base".to_string()];
        assert_eq!(select_base_file(&one, "q4").unwrap(), &one[0]);
    }

    #[test]
    fn list_base_files_filters_to_base_only() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("basecompute").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        for f in ["m-Q4.base", "m-Q8.base", "README.md", ".gitattributes"] {
            std::fs::write(repo_dir.join(f), b"x").unwrap();
        }
        let fetcher = MockFetcher::new(tmp.path());
        let mut got = list_base_files(&fetcher, "basecompute/m", "main").unwrap();
        got.sort();
        assert_eq!(got, vec!["m-Q4.base".to_string(), "m-Q8.base".to_string()]);
    }

    #[test]
    fn pull_base_direct_installs_requested_quant() {
        let tmp = tempfile::tempdir().unwrap();
        // Fake repo with both quants.
        let repo_dir = tmp.path().join("basecompute").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("m-Q4.base"), b"q4-bytes").unwrap();
        std::fs::write(repo_dir.join("m-Q8.base"), b"q8-bytes").unwrap();
        let fetcher = MockFetcher::new(tmp.path());
        let base_files = list_base_files(&fetcher, "basecompute/m", "main").unwrap();

        let root = tmp.path().join("cache");
        let args = PullArgs {
            id: "basecompute/m".into(),
            profile: None,
            target: TargetScheme::BaseQ8,
            revision: "main".into(),
            force: false,
            dry_run: false,
        };
        pull_base_direct(&root, &args, "basecompute/m", "basecompute/m", "main", &fetcher, &base_files)
            .unwrap();

        // The Q8 artifact landed under the default-q8 variant dir.
        let out = root.join("basecompute/m/default-q8/model.base");
        assert!(out.exists(), "missing {}", out.display());
        assert_eq!(std::fs::read(&out).unwrap(), b"q8-bytes");
        // A provenance sidecar was written.
        assert!(root.join("basecompute/m/default-q8/hub.json").exists());
        // MockFetcher owns no staging: its fixtures are copied, never moved.
        assert!(repo_dir.join("m-Q8.base").exists(), "fixture must survive");
    }

    /// Fetcher owning an hf-hub-style staging tree (blob + snapshot symlink),
    /// mirroring what `HfFetcher` leaves on disk mid-pull.
    struct StagedFetcher {
        staging: PathBuf,
    }

    impl StagedFetcher {
        fn repo_dir(&self, repo: &str) -> PathBuf {
            self.staging.join(format!("models--{}", repo.replace('/', "--")))
        }

        fn stage(&self, repo: &str, revision: &str, filename: &str, bytes: &[u8]) {
            let rdir = self.repo_dir(repo);
            let blobs = rdir.join("blobs");
            let snap = rdir.join("snapshots").join(revision);
            std::fs::create_dir_all(&blobs).unwrap();
            std::fs::create_dir_all(&snap).unwrap();
            let blob = blobs.join(format!("etag-{filename}"));
            std::fs::write(&blob, bytes).unwrap();
            std::os::unix::fs::symlink(&blob, snap.join(filename)).unwrap();
        }
    }

    impl Fetcher for StagedFetcher {
        fn get_file(&self, repo: &str, revision: &str, filename: &str) -> anyhow::Result<PathBuf> {
            let p = self.repo_dir(repo).join("snapshots").join(revision).join(filename);
            anyhow::ensure!(p.exists(), "not staged: {}", p.display());
            Ok(p)
        }

        fn list_files(&self, repo: &str, revision: &str) -> anyhow::Result<Vec<String>> {
            let dir = self.repo_dir(repo).join("snapshots").join(revision);
            let mut out = Vec::new();
            for e in std::fs::read_dir(dir)? {
                out.push(e?.file_name().to_string_lossy().into_owned());
            }
            Ok(out)
        }

        fn staging_dir(&self, repo: &str) -> Option<PathBuf> {
            Some(self.repo_dir(repo))
        }
    }

    #[test]
    fn pull_base_direct_from_staging_leaves_exactly_one_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("cache");
        let fetcher = StagedFetcher {
            staging: root.join(".src").join("hf"),
        };
        fetcher.stage("basecompute/m", "main", "m-Q4.base", b"q4-bytes");
        let base_files = list_base_files(&fetcher, "basecompute/m", "main").unwrap();

        let args = PullArgs {
            id: "basecompute/m".into(),
            profile: None,
            target: TargetScheme::BaseQ4,
            revision: "main".into(),
            force: false,
            dry_run: false,
        };
        pull_base_direct(&root, &args, "basecompute/m", "basecompute/m", "main", &fetcher, &base_files)
            .unwrap();

        let out = root.join("basecompute/m/default-q4/model.base");
        assert_eq!(std::fs::read(&out).unwrap(), b"q4-bytes");
        // The staged download was consumed: no second copy anywhere.
        assert!(
            !fetcher.repo_dir("basecompute/m").exists(),
            "staging tree must be cleaned after a successful install"
        );
    }

    #[test]
    fn keep_hf_sources_reads_env() {
        // Single test for all cases: mutates shared process env (see the
        // matching pattern in base-hub's fetch.rs tests).
        let prev = std::env::var("BASERT_KEEP_HF_SOURCES").ok();

        std::env::remove_var("BASERT_KEEP_HF_SOURCES");
        assert!(!keep_hf_sources(), "default must be: clean up sources");
        for off in ["", "0", "false", "FALSE", " 0 "] {
            std::env::set_var("BASERT_KEEP_HF_SOURCES", off);
            assert!(!keep_hf_sources(), "{off:?} should not keep sources");
        }
        for on in ["1", "true", "yes"] {
            std::env::set_var("BASERT_KEEP_HF_SOURCES", on);
            assert!(keep_hf_sources(), "{on:?} should keep sources");
        }

        match prev {
            Some(v) => std::env::set_var("BASERT_KEEP_HF_SOURCES", v),
            None => std::env::remove_var("BASERT_KEEP_HF_SOURCES"),
        }
    }

    #[test]
    fn preconverted_id_prefers_basecompute_mirror() {
        let reg = MergedRegistry::bundled().unwrap();
        // A source-org id with a catalogued basecompute counterpart maps to it.
        assert_eq!(
            preconverted_id(&reg, "Qwen/Qwen3-0.6B"),
            "basecompute/Qwen3-0.6B"
        );
        // An already-basecompute id is unchanged.
        assert_eq!(
            preconverted_id(&reg, "basecompute/Qwen3-0.6B"),
            "basecompute/Qwen3-0.6B"
        );
        // No catalogued counterpart → fall back to the source repo (convert-on-pull).
        assert_eq!(
            preconverted_id(&reg, "someorg/Definitely-Not-Catalogued-XYZ"),
            "someorg/Definitely-Not-Catalogued-XYZ"
        );
    }

    #[test]
    fn target_from_quant_maps_known_tags() {
        assert!(matches!(target_from_quant("q8"), TargetScheme::BaseQ8));
        assert!(matches!(target_from_quant("q4"), TargetScheme::BaseQ4));
        assert!(matches!(target_from_quant("bf16"), TargetScheme::Bf16));
        // Unknown → q4 default.
        assert!(matches!(target_from_quant("weird"), TargetScheme::BaseQ4));
    }

    #[test]
    fn want_source_file_filters_safetensors_only() {
        for ok in [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            assert!(want_source_file(ok), "should want {ok}");
        }
        for no in [
            "pytorch_model.bin",
            "model.pth",
            "weights.gguf",
            "README.md",
        ] {
            assert!(!want_source_file(no), "should skip {no}");
        }
    }

    #[test]
    fn download_source_resolves_snapshot_and_filters() {
        let tmp = tempfile::tempdir().unwrap();
        // Lay out a fake HF repo: <root>/org/model/<files>.
        let repo_dir = tmp.path().join("org").join("model");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        for f in ["model.safetensors", "tokenizer.json", "pytorch_model.bin"] {
            std::fs::write(repo_dir.join(f), b"x").unwrap();
        }
        let fetcher = MockFetcher::new(tmp.path());

        let snapshot = download_source("org/model", "main", &fetcher).unwrap();
        assert_eq!(snapshot, repo_dir);
    }

    #[test]
    fn download_source_rejects_unsupported_arch_before_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("org").join("exotic");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("config.json"), br#"{"model_type":"mamba"}"#).unwrap();
        std::fs::write(repo_dir.join("model.safetensors"), b"x").unwrap();
        let fetcher = MockFetcher::new(tmp.path());

        let err = download_source("org/exotic", "main", &fetcher).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mamba"), "{msg}");
        assert!(msg.contains("not supported"), "{msg}");
        // The supported set is surfaced so the user knows what works.
        assert!(msg.contains("llama"), "{msg}");
    }

    #[test]
    fn check_supported_arch_reads_nested_text_config() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("config.json");
        std::fs::write(&cfg, br#"{"text_config":{"model_type":"gemma4_text"}}"#).unwrap();
        check_supported_arch(&cfg, "org/m", "main").unwrap();
    }

    #[test]
    fn download_source_rejects_non_safetensors_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("org").join("ggufonly");
        std::fs::create_dir_all(&repo_dir).unwrap();
        std::fs::write(repo_dir.join("config.json"), b"{}").unwrap();
        std::fs::write(repo_dir.join("model.gguf"), b"x").unwrap();
        let fetcher = MockFetcher::new(tmp.path());

        let err = download_source("org/ggufonly", "main", &fetcher).unwrap_err();
        assert!(err.to_string().contains("safetensors"), "{err}");
    }

    #[test]
    fn looks_like_hub_id_distinguishes_ids_from_paths() {
        // Hub-id shaped: namespace/name, optionally :variant.
        assert!(looks_like_hub_id("Qwen/Qwen3-0.6B"));
        assert!(looks_like_hub_id("meta-llama/Llama-3.2-1B"));
        assert!(looks_like_hub_id("Qwen/Qwen3-0.6B:default-q4"));
        // Explicit paths / files are never reinterpreted as ids.
        assert!(!looks_like_hub_id("models/your-model.base"));
        assert!(!looks_like_hub_id("./Qwen/Qwen3-0.6B"));
        assert!(!looks_like_hub_id("/abs/path/model.base"));
        assert!(!looks_like_hub_id("~/cache/model.base"));
        // No slash → bare filename, leave for the runtime.
        assert!(!looks_like_hub_id("model.base"));
        assert!(!looks_like_hub_id("qwen"));
        assert!(!looks_like_hub_id(""));
    }

    #[test]
    fn resolve_model_args_passes_through_non_ids() {
        // Flags and plain paths are untouched (and never hit the registry).
        let args = vec![
            "models/m.base".to_string(),
            "--max-tokens".to_string(),
            "32".to_string(),
        ];
        let out = resolve_model_args(&args, None).unwrap();
        assert_eq!(out, args);
    }

    #[test]
    fn resolve_hub_model_rejects_empty_variant() {
        // A trailing `:` (empty variant) is a typo — error before any pull,
        // rather than silently resolving to the q4 default.
        let err = resolve_hub_model("basecompute/Gemma-4-E2B-it:", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty variant"), "{err}");
    }

    #[test]
    fn extract_variant_flag_parses_all_forms() {
        // `--variant <v>`
        let (v, rest) = extract_variant_flag(&[
            "--model".into(),
            "org/m".into(),
            "--variant".into(),
            "default-q8".into(),
            "--port".into(),
            "8080".into(),
        ])
        .unwrap();
        assert_eq!(v.as_deref(), Some("default-q8"));
        assert_eq!(rest, vec!["--model", "org/m", "--port", "8080"]);

        // `--variant=<v>`
        let (v, rest) = extract_variant_flag(&["--variant=q8".into(), "org/m".into()]).unwrap();
        assert_eq!(v.as_deref(), Some("q8"));
        assert_eq!(rest, vec!["org/m"]);

        // Absent → None, args untouched.
        let (v, rest) = extract_variant_flag(&["org/m".into()]).unwrap();
        assert!(v.is_none());
        assert_eq!(rest, vec!["org/m"]);

        // Missing / empty value → error.
        assert!(extract_variant_flag(&["--variant".into()]).is_err());
        assert!(extract_variant_flag(&["--variant=".into()]).is_err());
    }

    #[test]
    fn choose_profile_uses_bundled_for_q4_q6_and_q8() {
        let mk = |t| PullArgs {
            id: "x/y".into(),
            profile: None,
            target: t,
            revision: "main".into(),
            force: false,
            dry_run: false,
        };
        let (variant, path, guard) = choose_profile(&mk(TargetScheme::BaseQ4)).unwrap();
        assert_eq!(variant, "default-q4");
        assert!(path.is_some() && guard.is_some());

        let (variant, path, guard) = choose_profile(&mk(TargetScheme::BaseQ6)).unwrap();
        assert_eq!(variant, "default-q6");
        assert!(path.is_some() && guard.is_some());

        let (variant, _, _) = choose_profile(&mk(TargetScheme::BaseQ8)).unwrap();
        assert_eq!(variant, "default-q8");

        // No bundled generic profile for bf16 → flat target, no profile.
        let (variant, path, guard) = choose_profile(&mk(TargetScheme::Bf16)).unwrap();
        assert_eq!(variant, "bf16");
        assert!(path.is_none() && guard.is_none());
    }
}
