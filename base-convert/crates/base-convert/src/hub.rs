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
use base_hub::fetch::{Fetcher, HfFetcher};
use base_hub::registry::{MergedRegistry, ModelEntry, ModelRef, Registry, SourceKind};
use std::path::{Path, PathBuf};
use std::process::Command;

// Only generic, arch-agnostic profiles ship in the (public) binary. Tuned
// per-arch profiles are intentionally NOT bundled — tuned quality reaches
// users through pre-converted catalog artifacts, not by exposing the recipe.
const PROFILE_Q4: &str = include_str!("../../../profiles/default-q4.json");
const PROFILE_Q8: &str = include_str!("../../../profiles/default-q8.json");

fn bundled_profile_for_target(target: TargetScheme) -> Option<(&'static str, &'static str)> {
    match target {
        TargetScheme::BaseQ4 => Some(("default-q4", PROFILE_Q4)),
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

/// Resolve a hub model reference (`org/model` or `org/model:variant`) to the
/// installed `.base` artifact path. Errors when the id is hub-shaped but not
/// installed, or ambiguous across variants, so the user gets an actionable
/// message instead of a runtime "file not found".
fn resolve_hub_model(token: &str) -> Result<PathBuf> {
    let (id, variant) = token.split_once(':').map_or((token, None), |(i, v)| (i, Some(v)));
    let reg = MergedRegistry::bundled()?;

    if let Some(v) = variant {
        return reg.local.installed_path(id, v).with_context(|| {
            format!("model `{id}:{v}` is not installed — run `basert pull {id}` or see `basert list`")
        });
    }

    let installed = reg.local.list()?;
    let hits: Vec<&ModelEntry> = installed.iter().filter(|r| r.id == id).collect();
    match hits.as_slice() {
        [] => bail!("model `{id}` is not installed — run `basert pull {id}` or see `basert list`"),
        [one] => one
            .path
            .clone()
            .with_context(|| format!("installed model `{id}` has no artifact path")),
        many => {
            let variants = many.iter().map(|r| r.variant.as_str()).collect::<Vec<_>>().join(", ");
            bail!("model `{id}` has multiple installed variants ({variants}) — specify one as `{id}:<variant>`")
        }
    }
}

/// Rewrite every hub-id-shaped argument to the installed `.base` path so that
/// `basert chat org/model` (positional) and `basert serve --model org/model`
/// (repeatable flag) both Just Work after `basert pull`.
fn resolve_model_args(rest: &[String]) -> Result<Vec<String>> {
    rest.iter()
        .map(|arg| {
            if looks_like_hub_id(arg) {
                Ok(resolve_hub_model(arg)?.to_string_lossy().into_owned())
            } else {
                Ok(arg.clone())
            }
        })
        .collect()
}

/// Forward `basert <cmd> [args…]` to the matching runtime binary. Searches for
/// `basert-<cmd>` (release layout) then `baseRT_<cmd>` (local dev build),
/// looking next to this executable first and then on `PATH`. On success the
/// current process is replaced (`exec`), so this only returns on failure.
pub fn dispatch_external(argv: Vec<String>) -> Result<()> {
    use std::os::unix::process::CommandExt;

    let (cmd, rest) = argv
        .split_first()
        .context("no subcommand given to launcher")?;
    let rest = resolve_model_args(rest)?;
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
    bail!(
        "unknown command `basert {cmd}` — no `basert-{cmd}` runtime tool found \
         next to this binary or on PATH.\n\
         Install the BaseRT runtime alongside the CLI, or run `basert --help`."
    )
}

pub fn cmd_pull(args: PullArgs) -> Result<()> {
    let reg = MergedRegistry::bundled()?;
    let root = reg.root.clone();
    let r = reg.resolve(&args.id, &args.revision, args.force)?;

    if args.dry_run {
        print_plan(&r);
        return Ok(());
    }

    match r {
        ModelRef::Local { id, variant, path } => {
            eprintln!(
                "✓ {id} [{variant}] already installed: {}\n  (use --force to re-pull)",
                path.display()
            );
            Ok(())
        }
        ModelRef::Catalog { .. } => pull_catalog(&root, &r),
        ModelRef::HuggingFace { id, repo, revision } => {
            pull_and_convert(&root, &args, &id, &repo, &revision)
        }
    }
}

fn print_plan(r: &ModelRef) {
    match r {
        ModelRef::Local { id, variant, path } => {
            println!("plan: {id} [{variant}] already installed at {}", path.display())
        }
        ModelRef::Catalog { id, hf_repo, file, revision, variant, .. } => println!(
            "plan: pull pre-converted {id} [{variant}] from {hf_repo}/{file}@{revision} (no conversion)"
        ),
        ModelRef::HuggingFace { id, repo, revision } => {
            println!("plan: download {repo}@{revision} source + convert locally → {id}")
        }
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

    let fetcher = HfFetcher::new()?;
    let src = fetcher.get_file(hf_repo, revision, file)?;

    let vdir = cache::variant_dir(root, id, variant)?;
    std::fs::create_dir_all(&vdir)?;
    let out = cache::base_artifact_path(&vdir);
    std::fs::copy(&src, &out).with_context(|| format!("installing into {}", out.display()))?;

    let got_sha = crate::compute_sha256_streaming(&out)?;
    if let Some(expected) = sha256 {
        if !got_sha.eq_ignore_ascii_case(expected) {
            std::fs::remove_file(&out).ok();
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
    eprintln!("✓ installed {id} [{variant}] → {}", out.display());
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

    let fetcher = HfFetcher::new()?;
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
    eprintln!("✓ installed {id} [{variant}] → {}", out.display());
    Ok(())
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
    let reg = MergedRegistry::bundled()?;
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
        for f in [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "pytorch_model.bin",
        ] {
            std::fs::write(repo_dir.join(f), b"x").unwrap();
        }
        let fetcher = MockFetcher::new(tmp.path());

        let snapshot = download_source("org/model", "main", &fetcher).unwrap();
        assert_eq!(snapshot, repo_dir);
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
        let out = resolve_model_args(&args).unwrap();
        assert_eq!(out, args);
    }

    #[test]
    fn choose_profile_uses_bundled_for_q4_and_q8() {
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

        let (variant, _, _) = choose_profile(&mk(TargetScheme::BaseQ8)).unwrap();
        assert_eq!(variant, "default-q8");

        // No bundled generic profile for bf16 → flat target, no profile.
        let (variant, path, guard) = choose_profile(&mk(TargetScheme::Bf16)).unwrap();
        assert_eq!(variant, "bf16");
        assert!(path.is_none() && guard.is_none());
    }
}
