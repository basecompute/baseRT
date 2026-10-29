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
use base_hub::catalog::CatalogEntry;
use base_hub::fetch::{self, Fetcher, HfFetcher};
use base_hub::parts::{self, Artifact, Grouped};
use base_hub::registry::{
    CatalogRegistry, MergedRegistry, ModelEntry, ModelRef, Registry, SourceKind,
};
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
    s.rsplit(['-', '_'])
        .next()
        .unwrap_or(s)
        .to_ascii_lowercase()
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

/// List the `.base` artifacts a repo hosts (empty when it ships none). A
/// bundle the Hub's 50 GB file cap forced into `.base.part-NNN` pieces shows
/// up once, under its logical `.base` name.
fn list_base_files(fetcher: &dyn Fetcher, repo: &str, revision: &str) -> Result<Grouped> {
    let files = fetcher
        .list_files(repo, revision)
        .with_context(|| format!("listing files in {repo}@{revision}"))?;
    let grouped = parts::group(files);
    for (name, why) in &grouped.malformed {
        eprintln!("  skipping {name}: {why}");
    }
    Ok(grouped)
}

/// Pick the `.base` whose quant tag matches `want`. A malformed part set
/// carrying that quant is an error saying why, never a fall-through to a
/// different quant. Falls back to the sole artifact when the repo has
/// exactly one and nothing malformed; errors when several are present and
/// none match.
fn select_base_file<'a>(files: &'a Grouped, want: &str) -> Result<&'a Artifact> {
    if let Some(f) = files
        .artifacts
        .iter()
        .find(|f| quant_tag(&base_file_stem(&f.name)) == want)
    {
        return Ok(f);
    }
    if let Some((name, why)) = files
        .malformed
        .iter()
        .find(|(n, _)| quant_tag(&base_file_stem(n)) == want)
    {
        bail!("{name}: the {want} publication is incomplete ({why})");
    }
    if let ([only], []) = (files.artifacts.as_slice(), files.malformed.as_slice()) {
        return Ok(only);
    }
    let avail: Vec<String> = files
        .artifacts
        .iter()
        .map(|f| quant_tag(&base_file_stem(&f.name)))
        .chain(
            files
                .malformed
                .iter()
                .map(|(n, _)| format!("{} (incomplete)", quant_tag(&base_file_stem(n)))),
        )
        .collect();
    bail!(
        "no pre-converted .base for quant {want:?} in this repo; it offers: {} \
         (or take one by name with --file: {})",
        avail.join(", "),
        file_names(files)
    )
}

/// Whether `name`, as given to `--file`, names the repo file `f`: by its
/// whole path when `name` has a directory, else by its file name; the
/// `.base` extension is optional either way.
fn names_file(name: &str, f: &str) -> bool {
    let bare = |s: &str| s.strip_suffix(".base").unwrap_or(s).to_string();
    if name.contains('/') {
        bare(name) == bare(f)
    } else {
        base_file_stem(f) == bare(name)
    }
}

/// Pick the `.base` that `name` names (see [`names_file`]). A file name
/// shared by several directories must be given with its path; a malformed
/// part set by that name is an error saying why; no match lists the names
/// there are.
fn select_named_base_file<'a>(files: &'a Grouped, name: &str) -> Result<&'a Artifact> {
    let hits: Vec<&Artifact> = files
        .artifacts
        .iter()
        .filter(|f| names_file(name, &f.name))
        .collect();
    let broken: Vec<&(String, String)> = files
        .malformed
        .iter()
        .filter(|(n, _)| names_file(name, n))
        .collect();
    match (hits.as_slice(), broken.as_slice()) {
        ([one], []) => Ok(one),
        ([], [(n, why)]) => bail!("{n}: this publication is incomplete ({why})"),
        ([], []) => bail!(
            "no .base named {name:?} in this repo; it has: {}",
            file_names(files)
        ),
        _ => bail!(
            "{name:?} names several files in this repo ({}); give its path",
            hits.iter()
                .map(|f| f.name.clone())
                .chain(broken.iter().map(|(n, _)| format!("{n} (incomplete)")))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn file_names(files: &Grouped) -> String {
    files
        .artifacts
        .iter()
        .map(|f| f.name.clone())
        .chain(
            files
                .malformed
                .iter()
                .map(|(n, _)| format!("{n} (incomplete)")),
        )
        .collect::<Vec<_>>()
        .join(", ")
}

/// The variant a `.base` picked by quant installs under: the artifact's own
/// quant when it carries one (so a `-Q8.base` never lands in a `default-q4`
/// dir); otherwise what was requested.
fn base_variant(file: &str, want: &str) -> String {
    let file_tag = quant_tag(&base_file_stem(file));
    if is_quant_tag(&file_tag) {
        format!("default-{file_tag}")
    } else {
        format!("default-{want}")
    }
}

/// Variants of files taken by `--file` that no catalog row publishes start
/// with this, so none is ever a catalog row's or a quant's: the resolver
/// reaches one only by its exact name.
pub(crate) const NAMED_VARIANT_PREFIX: &str = "file-";

/// The variant a `--file` pull installs a file the catalog does not publish
/// under: the prefix, its path in the repo made readable (`Foo-Medium.base`
/// → `file-foo-medium-…`) and cut to 64 bytes so the directory name stays
/// well inside a filesystem's 255-byte limit, and the first 8 hex digits of the
/// path's sha256, so paths the readable part folds together (`a/model.base`,
/// `a-model.base`, `A-Model.base`, two long paths sharing a prefix) still get
/// variants of their own.
fn named_variant(file: &str) -> String {
    use sha2::{Digest, Sha256};
    let path = file.strip_suffix(".base").unwrap_or(file);
    let mut readable = path.replace('/', "-").to_ascii_lowercase();
    let mut end = readable.len().min(64);
    while !readable.is_char_boundary(end) {
        end -= 1;
    }
    readable.truncate(end);
    let digest = Sha256::digest(file.as_bytes());
    let short: String = digest[..4].iter().map(|b| format!("{b:02x}")).collect();
    format!("{NAMED_VARIANT_PREFIX}{readable}-{short}")
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
        [one] => Ok(Some(one.path.clone().with_context(|| {
            format!("installed model `{id}` has no artifact path")
        })?)),
        many => {
            // Multiple variants installed — e.g. a universal `default-q4` cached
            // before the backend-qualified catalog entries existed, plus a native
            // `cuda-q4mix` pulled after. Rather than erroring, apply the same
            // backend preference the catalog resolver uses (backend-native >
            // universal) so bare `serve <id>` keeps Just Working. The installed
            // variant dir name carries the backend slot (`cuda-*`/`rocm-*`/`cpu-*`
            // vs the universal `default-*`); on a backend that can't load a
            // foreign-native bundle it simply won't be present.
            let backend = base_hub::registry::CatalogRegistry::client_backend();
            let native_prefix = format!("{backend}-");
            let native: Vec<&ModelEntry> = many
                .iter()
                .copied()
                .filter(|r| r.variant.starts_with(&native_prefix))
                .collect();
            let pick = if native.len() == 1 {
                Some(native[0])
            } else if native.is_empty() {
                let uni: Vec<&ModelEntry> = many
                    .iter()
                    .copied()
                    .filter(|r| r.variant.starts_with("default-"))
                    .collect();
                (uni.len() == 1).then(|| uni[0])
            } else {
                None
            };
            if let Some(one) = pick {
                return Ok(Some(one.path.clone().with_context(|| {
                    format!("installed model `{id}` has no artifact path")
                })?));
            }
            let variants = many
                .iter()
                .map(|r| r.variant.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!("model `{id}` has multiple installed variants ({variants}) — specify one as `{id}:<variant>`")
        }
    }
}

/// Installed artifact for `id` matching the requested variant `want`. Tries an
/// EXACT variant-dir match first (honors `:default-q4` / `:cuda-q4mix`); failing
/// that, matches by BIT WIDTH so the documented short `:q4` finds an installed
/// `cuda-q4mix` or `default-q4` (the exact dir is named for the quant slot, not
/// the bare bits), preferring the backend-native slot over universal — the same
/// rule the catalog resolver's quant_matches + backend preference apply. Returns
/// `None` when nothing installed matches (caller then auto-pulls).
fn installed_best_variant(reg: &MergedRegistry, id: &str, want: &str) -> Option<PathBuf> {
    if let Some(p) = reg.local.installed_path(id, want) {
        return Some(p); // exact variant name present
    }
    let want_bits = base_hub::registry::quant_bits(want).unwrap_or(want);
    let backend = base_hub::registry::CatalogRegistry::client_backend();
    let native_prefix = format!("{backend}-");
    // A foreign-native slot (e.g. `cuda-q4mix` on macOS) must be EXCLUDED, not
    // just deprioritized: on a copied/shared cache it can't be loaded, and a bit-
    // width tie would otherwise pick the lexicographically-earlier CUDA artifact
    // over the runnable universal one. Runnable = universal (`default-*`), this
    // client's native slot (`<backend>-*`), or a bare bit-width dir (no slot).
    const KNOWN_BACKENDS: [&str; 4] = ["cuda", "rocm", "cpu", "metal"];
    let runnable = |v: &str| -> bool {
        match v.split_once('-') {
            Some((slot, _)) if KNOWN_BACKENDS.contains(&slot) => slot == backend, // native only
            _ => true, // default-*, bare bits, etc.
        }
    };
    let installed = reg.local.list().ok()?;
    let mut matches: Vec<&ModelEntry> = installed
        .iter()
        .filter(|r| {
            r.id == id
                && runnable(&r.variant)
                // a file taken by name answers only to its exact variant
                && !r.variant.starts_with(NAMED_VARIANT_PREFIX)
                && base_hub::registry::quant_bits(&r.variant).unwrap_or(r.variant.as_str())
                    == want_bits
        })
        .collect();
    if matches.is_empty() {
        return None;
    }
    // Backend-native slot first (stable so a deterministic pick when both exist).
    matches.sort_by_key(|r| u8::from(!r.variant.starts_with(&native_prefix)));
    matches[0].path.clone()
}

/// Fetch a not-yet-installed model on demand, then return its artifact path.
/// Prefers the pre-converted basecompute mirror; otherwise converts the source
/// repo on pull. Progress (download + quantization) is shown by `cmd_pull`.
fn auto_pull_and_resolve(
    reg: &MergedRegistry,
    id: &str,
    want_variant: Option<&str>,
) -> Result<PathBuf> {
    let pull_id = preconverted_id(reg, id);

    if pull_id != id {
        eprintln!("{id}: not installed — using pre-converted {pull_id}");
    } else {
        eprintln!("{id}: not installed — fetching (download + convert if needed)");
    }

    // The variant goes with the id: the pull `basert pull <id>:<variant>`
    // makes, so a catalog variant or a `file-…` one is fetched as named, not
    // as the quant its last segment happens to spell.
    cmd_pull(PullArgs {
        id: want_variant.map_or_else(|| pull_id.clone(), |v| format!("{pull_id}:{v}")),
        profile: None,
        target: TargetScheme::BaseQ4,
        revision: "main".to_string(),
        file: None,
        force: false,
        dry_run: false,
    })
    .with_context(|| format!("auto-fetching {pull_id}"))?;

    // Re-scan and return the freshly-installed artifact (registered under the
    // pulled id, which may differ from the requested one): the variant as
    // named, or for a quant (`:q8`) the slot it installed under.
    let reg2 = MergedRegistry::load()?;
    if let Some(v) = want_variant {
        if let Some(p) = installed_best_variant(&reg2, &pull_id, v) {
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
/// What `--speculate` asks of the model's own MTP head in a runtime
/// command's args: bare `mtp-head`, or `auto`, which takes that head before
/// any drafter. None when speculation is off or not asked for, a drafter is
/// named, or the strategy needs no head (`prompt-lookup`).
pub(crate) fn mtp_speculation_asked(rest: &[String]) -> Option<&str> {
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let v = if a == "--speculate" {
            it.next().map(String::as_str)
        } else {
            a.strip_prefix("--speculate=")
        };
        if let Some(v) = v {
            return matches!(v, "mtp-head" | "auto").then_some(v);
        }
    }
    None
}

/// An installed catalog file that cannot serve the speculation asked for,
/// though the catalog's current file could: it carries no MTP head, or one
/// of another version than the row's `speculator_version`, while the row for
/// the same id and variant says the published file has one (`speculator:
/// mtp`, or `gemma4-mtp` for Gemma 4's embedded head) and is another file
/// (its sha256 differs). Only a file the catalog installed qualifies —
/// pulling it again replaces like with like; a conversion or a file from
/// elsewhere is left alone. Returns the variant to pull again.
fn stale_head_variant(reg: &MergedRegistry, id: &str, path: &Path) -> Option<String> {
    let sidecar = cache::read_sidecar(path.parent()?).ok()??;
    if sidecar.source_kind != "catalog" {
        return None;
    }
    let row = reg.catalog.rows(id).find(|e| e.quant == sidecar.variant)?;
    if !matches!(row.speculator.as_deref(), Some("mtp" | "gemma4-mtp")) {
        return None;
    }
    let same_file = matches!((&row.sha256, &sidecar.base_sha256), (Some(want), Some(got)) if want.eq_ignore_ascii_case(got));
    if same_file {
        return None;
    }
    let header = base_format::BaseReader::read_header(path).ok()?;
    let stale = match (&header.speculator, &row.speculator_version) {
        // No head at all.
        (None, _) => true,
        // A head of another version than the one the catalog publishes.
        (Some(have), Some(want)) => {
            have.config.get("head_version").and_then(|v| v.as_str()) != Some(want.as_str())
        }
        // A head, and the catalog names no version to hold it to.
        (Some(_), None) => false,
    };
    stale.then_some(sidecar.variant)
}

/// The installed artifact to serve, pulled again first when `--speculate
/// <directive>` asks for its MTP head and the catalog's current file has one
/// this file lacks (stale_head_variant). Without a speculation request the
/// file is served as it is, with no word said.
fn serve_ready(
    reg: &MergedRegistry,
    id: &str,
    path: PathBuf,
    mtp: Option<&str>,
) -> Result<PathBuf> {
    let Some(directive) = mtp else {
        return Ok(path);
    };
    let Some(variant) = stale_head_variant(reg, id, &path) else {
        return Ok(path);
    };
    eprintln!(
        "{id} [{variant}]: the installed file carries no MTP head and the catalog's current one does — \
         pulling it again for --speculate {directive}"
    );
    cmd_pull(PullArgs {
        id: format!("{id}:{variant}"),
        profile: None,
        target: TargetScheme::BaseQ4,
        revision: "main".to_string(),
        file: None,
        force: true,
        dry_run: false,
    })
    .with_context(|| format!("pulling {id}:{variant} again"))?;
    let reg2 = MergedRegistry::load()?;
    installed_best_variant(&reg2, id, &variant).with_context(|| {
        format!("pulled {id}:{variant} again but could not locate the installed artifact")
    })
}

fn resolve_hub_model(
    token: &str,
    default_variant: Option<&str>,
    mtp: Option<&str>,
) -> Result<PathBuf> {
    let (id, inline) = token
        .split_once(':')
        .map_or((token, None), |(i, v)| (i, Some(v)));

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
        if let Some(p) = installed_best_variant(&reg, id, v) {
            return serve_ready(&reg, id, p, mtp);
        }
        return auto_pull_and_resolve(&reg, id, Some(v));
    }

    if let Some(p) = installed_single_path(&reg, id)? {
        return serve_ready(&reg, id, p, mtp);
    }
    auto_pull_and_resolve(&reg, id, None)
}

/// Rewrite every hub-id-shaped argument to the installed `.base` path so that
/// `basert chat org/model` (positional) and `basert serve --model org/model`
/// (repeatable flag) both Just Work after `basert pull`. `default_variant`
/// comes from the `--variant` launcher flag and applies to ids without an
/// inline `:variant`.
fn resolve_model_args(rest: &[String], default_variant: Option<&str>) -> Result<Vec<String>> {
    // `--speculate mtp-head` / `auto` asks for the model's own head: a stale
    // catalog file without one is pulled again on the way (serve_ready).
    let mtp = mtp_speculation_asked(rest);
    rest.iter()
        .map(|arg| {
            if looks_like_hub_id(arg) {
                Ok(resolve_hub_model(arg, default_variant, mtp)?
                    .to_string_lossy()
                    .into_owned())
            } else if let Some((kind, id)) = speculate_drafter_id(arg) {
                // `--speculate dflash:<id>` takes the SAME road as the model
                // it drafts for: catalogue lookup, pull, convert-on-pull,
                // variant selection, hub cache. A drafter was the one model a
                // user had to fetch and convert by hand and then name by path.
                let path = resolve_hub_model(id, default_variant, None)?;
                Ok(format!("{kind}:{}", path.to_string_lossy()))
            } else {
                Ok(arg.clone())
            }
        })
        .collect()
}

/// `<strategy>:<hub id>` for the strategies that take a drafter artifact —
/// `dflash:basecompute/Qwen3-4B-DFlash`, `eagle3:org/head:default-q4`. Returns
/// None for a path (`dflash:./x.base`, which keeps working unchanged), for a
/// strategy that takes no artifact (`prompt-lookup`, `mtp-head`), and for
/// anything that is not id-shaped.
fn speculate_drafter_id(token: &str) -> Option<(&str, &str)> {
    let (kind, rest) = token.split_once(':')?;
    if !matches!(kind, "dflash" | "dspark" | "eagle3" | "draft-model") {
        return None;
    }
    looks_like_hub_id(rest).then_some((kind, rest))
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

const COMPUTEARENA_HARNESS_ENV: &str = "COMPUTEARENA_BASERT_HARNESS";
const LEGACY_COMPUTEARENA_HARNESS_ENV: &str = "BASERT_COMPUTEARENA_HARNESS";
const BENCHMARK_HARNESS_BINARY: &str = "basert-benchmark-harness";

fn bundled_computearena_harness(
    exe_dir: Option<&Path>,
    environment_override: bool,
) -> Option<PathBuf> {
    if environment_override {
        return None;
    }
    let exe_dir = exe_dir?;

    // Published BaseRT packages keep the launcher and harness together.
    let sibling = exe_dir.join(BENCHMARK_HARNESS_BINARY);
    if sibling.is_file() {
        return Some(sibling);
    }

    // A source build places the Rust launcher under
    // tools/base-convert/target/{debug,release}, while CMake publishes the
    // native harness under build/. Recognize that layout so the documented
    // developer build works exactly like an installed release.
    exe_dir.ancestors().find_map(|directory| {
        let workspace_manifest = directory.join("tools/base-convert/Cargo.toml");
        let harness = directory.join("build").join(BENCHMARK_HARNESS_BINARY);
        (workspace_manifest.is_file() && harness.is_file()).then_some(harness)
    })
}

/// Forward `basert <cmd> [args…]` to the matching runtime binary. Searches for
/// `basert-<cmd>` (release layout) then `baseRT_<cmd>` (local dev build),
/// looking next to this executable first and then on `PATH`. `basert serve`
/// has no runtime binary: it runs superfluid (`crate::serve`). On success the
/// current process is replaced (`exec`), so this only returns on failure.
pub fn dispatch_external(argv: Vec<String>) -> Result<()> {
    use std::os::unix::process::CommandExt;

    let (cmd, rest) = argv.split_first().context("no command given")?;
    let is_computearena = cmd == "computearena";
    let (candidates, rest) = if is_computearena {
        // ComputeArena is independently distributed and owns runtime
        // selection. Preserve its arguments and select the BaseRT adapter —
        // no variant resolution.
        let mut forwarded = Vec::with_capacity(rest.len() + 1);
        forwarded.push("basert".to_string());
        forwarded.extend_from_slice(rest);
        (vec!["computearena".to_string()], forwarded)
    } else {
        // `--variant <v>` is a launcher-level model selector; strip it before
        // forwarding (the runtime binary doesn't know it) and apply it during
        // hub-id resolution.
        let (variant_flag, rest) = extract_variant_flag(rest)?;
        let rest = resolve_model_args(&rest, variant_flag.as_deref())?;
        if cmd == "serve" {
            return crate::serve::run(&rest);
        }
        (vec![format!("basert-{cmd}"), format!("baseRT_{cmd}")], rest)
    };

    // Prefer a binary sitting next to `basert` (how the release ships; for the
    // versions layout, in the version this launcher runs from); fall back to a
    // bare name, which `Command` resolves against `PATH`.
    let exe_dir = crate::selfupdate::bundle_dir().ok();
    let harness_environment_override = std::env::var_os(COMPUTEARENA_HARNESS_ENV).is_some()
        || std::env::var_os(LEGACY_COMPUTEARENA_HARNESS_ENV).is_some();
    let bundled_harness = if is_computearena {
        bundled_computearena_harness(exe_dir.as_deref(), harness_environment_override)
    } else {
        None
    };
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
        let mut command = Command::new(target);
        command.args(&rest);
        if let Some(harness) = &bundled_harness {
            command.env(COMPUTEARENA_HARNESS_ENV, harness);
        }
        let err = command.exec();
        if err.kind() != std::io::ErrorKind::NotFound {
            return Err(err).with_context(|| format!("launching {}", target.display()));
        }
    }
    if is_computearena {
        bail!(
            "ComputeArena is not installed.\n\
             Install it from https://computearena.ai/quickstart, then run \
             `basert computearena` again."
        )
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
const RUNTIME_COMMANDS: [&str; 5] = ["chat", "complete", "bench", "transcribe", "profile"];

pub fn cmd_pull(args: PullArgs) -> Result<()> {
    let reg = MergedRegistry::load()?;
    // Resolved against whichever catalog the attempt is given: the cached one
    // first, the live one if the cached rows prove to be behind the Hub.
    with_fresh_catalog_on_mismatch(&reg, MergedRegistry::load_fresh, |reg| {
        if let Some((id, variant)) = args.id.split_once(':') {
            let fetcher = HfFetcher::new(cache::hf_staging_dir(&reg.root))?;
            return pull_variant(reg, &args, id, variant, &fetcher);
        }
        pull_id(reg, &args)
    })
}

/// Where `basert pull <id>:<variant>` gets the variant from (see
/// [`variant_source`]).
#[derive(Debug, PartialEq)]
enum VariantSource {
    /// The catalog row the variant names, pulled as a plain pull of it is.
    Catalog(ModelRef),
    /// A `.base` in the id's repo that no catalog row publishes, pulled as
    /// `--file` would pull it.
    File(String),
    /// A quant, pulled as `--target` would pull it.
    Quant(TargetScheme),
}

/// What a pull says of a variant installed already.
fn say_installed(id: &str, variant: &str, path: &Path, dry_run: bool) {
    if dry_run {
        println!(
            "plan: {id} [{variant}] already installed at {}",
            path.display()
        );
    } else {
        eprintln!(
            "{id} [{variant}] already installed: {}\n  (use --force to re-pull)",
            path.display()
        );
    }
}

/// `basert pull <id>:<variant>`: the variant as `basert list` names it, the
/// form `basert chat` and `serve` already take. An installed variant is left
/// as it is (no request to the Hub); otherwise it is pulled from where
/// [`variant_source`] says. `--file`, `--profile` and `--target` each name what
/// the variant names, so none goes with one.
fn pull_variant(
    reg: &MergedRegistry,
    args: &PullArgs,
    id: &str,
    variant: &str,
    fetcher: &dyn Fetcher,
) -> Result<()> {
    if variant.is_empty() {
        bail!(
            "empty variant after ':' in `{}` — name a variant (e.g. `{id}:default-q8`) \
             or drop the colon. Run `basert list` to see installed variants.",
            args.id
        );
    }
    let given = [
        args.file.as_ref().map(|_| "--file"),
        args.profile.as_ref().map(|_| "--profile"),
        (args.target != TargetScheme::BaseQ4).then_some("--target"),
    ];
    if let Some(flag) = given.into_iter().flatten().next() {
        bail!(
            "`{}` names the variant, and so does {flag}: give one",
            args.id
        );
    }
    if !args.force {
        if let Some(path) = reg.local.installed_path(id, variant) {
            say_installed(id, variant, &path, args.dry_run);
            return Ok(());
        }
    }
    let pull = |file: Option<String>, target: TargetScheme| PullArgs {
        id: id.to_string(),
        profile: None,
        target,
        revision: args.revision.clone(),
        file,
        force: args.force,
        dry_run: args.dry_run,
    };
    match variant_source(reg, id, variant, &args.revision, fetcher)? {
        VariantSource::Catalog(r) => pull_catalog_row(reg, args, &r, fetcher),
        VariantSource::File(file) => {
            let named = pull(Some(file.clone()), args.target);
            pull_named_file(reg, &named, &file, fetcher)
        }
        VariantSource::Quant(target) => pull_id(reg, &pull(None, target)),
    }
}

/// The catalog row a variant named, pulled as a plain pull of it is: left as
/// it is when installed, else its file from its own repo and revision,
/// checked against its checksums. The row is carried through, never looked
/// up again by its file name, which other rows of the id may share.
fn pull_catalog_row(
    reg: &MergedRegistry,
    args: &PullArgs,
    r: &ModelRef,
    fetcher: &dyn Fetcher,
) -> Result<()> {
    let ModelRef::Catalog { id, variant, .. } = r else {
        unreachable!("pull_catalog_row called with non-catalog ref");
    };
    if !args.force {
        if let Some(path) = reg.local.installed_path(id, variant) {
            say_installed(id, variant, &path, args.dry_run);
            return Ok(());
        }
    }
    if args.dry_run {
        print_plan(r, variant);
        return Ok(());
    }
    pull_catalog(&reg.root, r, fetcher)
}

/// The `--target` a variant spells, taken whole: a quant tag (`q8`, `bf16`),
/// bare or under the prefix a converted variant or a `--target` value carries
/// (`default-q8`, `base-q8`, `base_q8`). `None` for anything else, so a name
/// that only ends in a quant (`default-q8-custom`, `typo-q8`) is not one.
fn target_alias(variant: &str) -> Option<TargetScheme> {
    let v = variant.to_ascii_lowercase();
    let tag = ["default-", "base-", "base_"]
        .iter()
        .find_map(|p| v.strip_prefix(p))
        .unwrap_or(&v);
    is_quant_tag(tag).then(|| target_from_quant(tag))
}

/// Where the variant `variant` of `id` comes from, in this order: a catalog
/// row of the id by that name (`default-q8`, `cuda-q4mix`), as `basert list
/// --remote` shows it, refused when it is for another backend; the repo file
/// a `file-…` variant was named for, looked for where `--file` looks; for a
/// width named on its own (`q8`, `f16`), the catalog's row of it for this
/// client; a quant `--target` takes, spelled as [`target_alias`] reads it.
/// Anything else, a name that merely contains a width included, is an error
/// naming the catalog's variants.
fn variant_source(
    reg: &MergedRegistry,
    id: &str,
    variant: &str,
    revision: &str,
    fetcher: &dyn Fetcher,
) -> Result<VariantSource> {
    // A row this client can run before one it can't, which is refused
    // saying which backend it is for.
    let mut rows: Vec<&CatalogEntry> = reg
        .catalog
        .rows(id)
        .filter(|e| e.quant == variant)
        .collect();
    rows.sort_by_key(|e| !CatalogRegistry::backend_ok(e));
    if let Some(e) = rows.first() {
        if !CatalogRegistry::backend_ok(e) {
            bail!(
                "{variant} is the {} bundle of {}; this client runs {}",
                e.backend.as_deref().unwrap_or_default(),
                e.id,
                CatalogRegistry::client_backend()
            );
        }
        return Ok(VariantSource::Catalog(CatalogRegistry::entry_to_ref(e)));
    }
    if variant.starts_with(NAMED_VARIANT_PREFIX) {
        return named_variant_file(reg, id, variant, revision, fetcher).map(VariantSource::File);
    }
    // A width named on its own: the catalog's row of it for this client,
    // native before universal, as a plain pull picks one. The whole name
    // must be the width; a `file-…` one can spell one (`file-model-q4-…`).
    if base_hub::registry::quant_bits(variant) == Some(variant) {
        match reg.catalog.resolve_with_status(id, Some(variant)) {
            (Some(r), _) => return Ok(VariantSource::Catalog(r)),
            (None, true) => bail!(
                "{id} has no {variant} bundle for the {} backend this client runs",
                CatalogRegistry::client_backend()
            ),
            (None, false) => {}
        }
    }
    if let Some(target) = target_alias(variant) {
        return Ok(VariantSource::Quant(target));
    }
    let listed: Vec<String> = reg
        .catalog
        .rows(id)
        .filter(|e| CatalogRegistry::backend_ok(e))
        .map(|e| e.quant.clone())
        .collect();
    let listed = if listed.is_empty() {
        String::new()
    } else {
        format!("; the catalog lists {}", listed.join(", "))
    };
    bail!(
        "{id} has no variant {variant:?}{listed}. Name one `basert list --remote` shows, \
         a quant (q4, q8, bf16, …), or a file with --file"
    )
}

/// The repo file the `file-…` variant `variant` was named for (see
/// [`named_variant`]), looked for where `--file` looks: the repos the id's
/// catalog rows use, else the id's own. A file the catalog publishes installs
/// under its row's variant, never a `file-…` one, so it is not that file.
fn named_variant_file(
    reg: &MergedRegistry,
    id: &str,
    variant: &str,
    revision: &str,
    fetcher: &dyn Fetcher,
) -> Result<String> {
    let mut sources: Vec<(String, String)> = Vec::new();
    for e in reg.catalog.rows(id) {
        let at = (e.hf_repo.clone(), e.revision.clone());
        if !sources.contains(&at) {
            sources.push(at);
        }
    }
    if sources.is_empty() {
        sources.push((id.to_string(), revision.to_string()));
    }
    let mut names = Vec::new();
    for (repo, revision) in &sources {
        let files = list_base_files(fetcher, repo, revision)?;
        names.extend(files.artifacts.iter().map(|f| f.name.clone()));
    }
    let Some(file) = names.iter().find(|f| named_variant(f) == variant) else {
        bail!(
            "no .base in {} is the file of {variant}; there are: {}",
            sources
                .iter()
                .map(|(r, v)| format!("{r}@{v}"))
                .collect::<Vec<_>>()
                .join(", "),
            names.join(", ")
        );
    };
    if let Some(e) = reg.catalog.rows(id).find(|e| e.file == *file) {
        bail!(
            "{file} is the catalog's {} of {}: pull {}:{}",
            e.quant,
            e.id,
            e.id,
            e.quant
        );
    }
    Ok(file.clone())
}

/// `basert pull <id>`, with no `:variant`.
fn pull_id(reg: &MergedRegistry, args: &PullArgs) -> Result<()> {
    let root = reg.root.clone();
    let want = quant_token(args);
    if let Some(file) = &args.file {
        let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;
        return pull_named_file(reg, args, file, &fetcher);
    }
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
        ModelRef::Catalog {
            id,
            hf_repo,
            revision,
            variant,
            ..
        } => {
            // Match on the QUANT BITS, not the raw variant string: the resolver
            // may hand back a backend-native variant (e.g. `cuda-q4mix`) that
            // satisfies a `q4` request but whose `quant_tag` ("q4mix") isn't the
            // literal `want` ("q4"). A raw `quant_tag(variant) == want` there
            // wrongly fell through to pull_base_direct and grabbed the universal
            // f16-scale `default-q4` — which a CUDA client can't load. Mirror
            // installed_best_variant / the catalog resolver's bit-aware match.
            let want_bits = base_hub::registry::quant_bits(&want).unwrap_or(&want);
            if base_hub::registry::quant_bits(variant).unwrap_or(variant) == want_bits {
                let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;
                pull_catalog(&root, &r, &fetcher)
            } else {
                let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;
                let base_files = list_base_files(&fetcher, hf_repo, revision)?;
                pull_base_direct(
                    &root,
                    args,
                    id,
                    hf_repo,
                    revision,
                    &fetcher,
                    &base_files,
                    None,
                    None,
                )
            }
        }
        // A raw HF repo: if it already hosts pre-converted `.base` artifacts
        // (e.g. the basecompute org), download the matching one directly — no
        // local conversion. Otherwise treat it as source safetensors.
        ModelRef::HuggingFace { id, repo, revision } => {
            let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;
            let base_files = list_base_files(&fetcher, repo, revision)?;
            // A repo holding only a half-uploaded split bundle is still a
            // `.base` repo: it gets the incomplete-publication error, not a
            // conversion attempt.
            if base_files.artifacts.is_empty() && base_files.malformed.is_empty() {
                pull_and_convert(&root, args, id, repo, revision)
            } else {
                pull_base_direct(
                    &root,
                    args,
                    id,
                    repo,
                    revision,
                    &fetcher,
                    &base_files,
                    None,
                    None,
                )
            }
        }
    }
}

/// `pull --file`: download the named `.base` from the id's repo (the catalog's
/// repo for a catalog id, else the id itself) into its own variant.
fn pull_named_file(
    reg: &MergedRegistry,
    args: &PullArgs,
    file: &str,
    fetcher: &dyn Fetcher,
) -> Result<()> {
    // A file a catalog row of the id publishes is that row's: taken from its
    // repo and revision, refused when it is for another backend, and
    // installed under its variant (`cuda-q4mix`), as a catalog pull of it
    // would be. Any other file comes from the id's repo (its catalog repo,
    // else the id itself), whatever quant or backend the catalog lists.
    let rows: Vec<&CatalogEntry> = reg
        .catalog
        .rows(&args.id)
        .filter(|e| names_file(file, &e.file))
        .collect();
    let listed = match rows.as_slice() {
        [] => None,
        [one] => Some(*one),
        many => bail!(
            "{file:?} names several files the catalog lists for {} ({}); give its path",
            args.id,
            many.iter()
                .map(|e| format!("{} in {}", e.file, e.hf_repo))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    if let Some(e) = listed.filter(|e| !CatalogRegistry::backend_ok(e)) {
        bail!(
            "{} is the {} bundle of {}; this client runs {}",
            e.file,
            e.backend.as_deref().unwrap_or_default(),
            e.id,
            CatalogRegistry::client_backend()
        );
    }
    // Where to look: the listing row's repo; else every repo the id's rows
    // keep files in (a name found in more than one must be given its row);
    // else the id itself as a Hub repo.
    let id = listed
        .or_else(|| reg.catalog.entry(&args.id))
        .map_or_else(|| args.id.clone(), |e| e.id.clone());
    let mut sources: Vec<(String, String)> = Vec::new();
    let rows = listed
        .into_iter()
        .chain(reg.catalog.rows(&args.id).filter(|_| listed.is_none()));
    for e in rows {
        let at = (e.hf_repo.clone(), e.revision.clone());
        if !sources.contains(&at) {
            sources.push(at);
        }
    }
    if sources.is_empty() {
        if args.id.split('/').filter(|s| !s.is_empty()).count() < 2 {
            bail!(
                "unknown model id {:?}: not in the catalog and not an `org/model` HF repo",
                args.id
            );
        }
        sources.push((args.id.clone(), args.revision.clone()));
    }
    // The name is resolved against the repo every time, cached or not, so a
    // cache hit answers exactly what a fresh pull would take.
    let mut found = Vec::new();
    for (repo, revision) in &sources {
        let base_files = list_base_files(fetcher, repo, revision)?;
        let names = |n: &str| names_file(file, n);
        if base_files.artifacts.iter().any(|f| names(&f.name))
            || base_files.malformed.iter().any(|(n, _)| names(n))
        {
            found.push((repo.clone(), revision.clone(), base_files));
        } else if sources.len() == 1 {
            // one place to look: its own error lists what is there
            select_named_base_file(&base_files, file)?;
        }
    }
    let (repo, revision, base_files) = match found.len() {
        1 => found.remove(0),
        0 => bail!(
            "no .base named {file:?} in {}",
            sources
                .iter()
                .map(|(r, v)| format!("{r}@{v}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => bail!(
            "{file:?} is in several repos of {id} ({}); pull it from its repo's id",
            found
                .iter()
                .map(|(r, v, _)| format!("{r}@{v}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let artifact = select_named_base_file(&base_files, file)?;
    if let Some(e) = listed.filter(|e| e.file != artifact.name) {
        bail!(
            "{file:?} names {} in the repo, but the catalog of {id} lists {}",
            artifact.name,
            e.file
        );
    }
    let variant = listed.map_or_else(|| named_variant(&artifact.name), |e| e.quant.clone());

    if let Some(path) = reg
        .local
        .installed_path(&id, &variant)
        .filter(|_| !args.force)
    {
        // Only this file's own install answers for it: one that records it,
        // or (for the catalog's file) a catalog pull of it from before files
        // were recorded. Anything else there is refused, not handed back.
        let vdir = cache::variant_dir(&reg.root, &id, &variant)?;
        let sidecar = cache::read_sidecar(&vdir)?;
        let held = sidecar.as_ref().and_then(|s| {
            s.file.clone().or_else(|| {
                // a catalog pull from before files were recorded is the row's
                // file only if its bytes are the row's
                listed
                    .filter(|e| {
                        s.source_kind == "catalog"
                            && matches!((&s.base_sha256, &e.sha256), (Some(got), Some(want)) if got.eq_ignore_ascii_case(want))
                    })
                    .map(|e| e.file.clone())
            })
        });
        match held {
            Some(held) if held == artifact.name => {}
            Some(held) => bail!(
                "{id} [{variant}] holds {held}, not {} (--force replaces it)",
                artifact.name
            ),
            None => bail!(
                "{id} [{variant}] holds {}, not {} (--force replaces it)",
                match sidecar.and_then(|s| s.profile) {
                    Some(_) => "a conversion",
                    None => "an install from another source",
                },
                artifact.name
            ),
        }
        if args.dry_run {
            println!(
                "plan: {id} [{variant}] already installed at {}",
                path.display()
            );
            return Ok(());
        }
        // a catalog pull from before files were recorded is now known as this one
        if let Some(mut s) = cache::read_sidecar(&vdir)?.filter(|s| s.file.is_none()) {
            s.file = Some(artifact.name.clone());
            cache::write_sidecar(&vdir, &s)?;
        }
        eprintln!(
            "{id} [{variant}] already installed: {}\n  (use --force to re-pull)",
            path.display()
        );
        return Ok(());
    }
    if args.dry_run {
        println!(
            "plan: pull {} from {repo}@{revision} → {id} [{variant}] (no conversion)",
            artifact.name
        );
        return Ok(());
    }
    pull_base_direct(
        &reg.root,
        args,
        &id,
        &repo,
        &revision,
        fetcher,
        &base_files,
        listed,
        Some(&variant),
    )
}

fn print_plan(r: &ModelRef, want: &str) {
    match r {
        ModelRef::Local { id, variant, path } => {
            println!(
                "plan: {id} [{variant}] already installed at {}",
                path.display()
            )
        }
        ModelRef::Catalog {
            id,
            hf_repo,
            file,
            revision,
            variant,
            ..
        } => {
            let want_bits = base_hub::registry::quant_bits(want).unwrap_or(want);
            if base_hub::registry::quant_bits(variant).unwrap_or(variant) == want_bits {
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
/// A pulled file whose bytes are not the catalog row's (`expected`): typed,
/// so a caller can tell a catalog that is behind the Hub from a bad download
/// (with_fresh_catalog_on_mismatch).
#[derive(Debug)]
struct ShaMismatch {
    id: String,
    file: String,
    expected: String,
    got: String,
}

impl std::fmt::Display for ShaMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sha256 mismatch for {} ({}): expected {}, got {}",
            self.id, self.file, self.expected, self.got
        )
    }
}

impl std::error::Error for ShaMismatch {}

/// Runs `attempt` against `reg`; when it fails because the file's bytes are
/// not the row's sha256, fetches the live catalog past the cache (`refetch`)
/// and runs it once more if the live row names another sha256. The cached
/// catalog is served under its TTL without a look at the hosted one, and a
/// bundle's row and file change together on the Hub, so between that change
/// and the cache's expiry a fresh pull downloads the new bytes and checks
/// them against the old row. The live catalog settles it: a Hub file that
/// disagrees with the live row too stays an error, as does a cache that
/// cannot be refreshed.
fn with_fresh_catalog_on_mismatch(
    reg: &MergedRegistry,
    refetch: impl FnOnce() -> Option<MergedRegistry>,
    attempt: impl Fn(&MergedRegistry) -> Result<()>,
) -> Result<()> {
    let err = match attempt(reg) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    let Some(m) = err.downcast_ref::<ShaMismatch>() else {
        return Err(err);
    };
    let Some(fresh) = refetch() else {
        return Err(err);
    };
    let changed = fresh.catalog.rows(&m.id).any(|e| {
        e.file == m.file
            && e.sha256
                .as_deref()
                .is_some_and(|s| !s.eq_ignore_ascii_case(&m.expected))
    });
    if !changed {
        return Err(err);
    }
    eprintln!(
        "  catalog: the cached catalog was behind the Hub ({}'s sha256 changed); pulling again from the live one",
        m.file
    );
    attempt(&fresh)
}

/// Where a pull puts the downloaded `.base` until it has passed the checks
/// that make it the installed one: beside it (`model.base.pending`), on the
/// same filesystem, so the final move is a rename.
fn pending_path(out: &Path) -> PathBuf {
    let mut p = out.as_os_str().to_owned();
    p.push(".pending");
    PathBuf::from(p)
}

fn pull_catalog(root: &Path, r: &ModelRef, fetcher: &dyn Fetcher) -> Result<()> {
    let ModelRef::Catalog {
        id,
        hf_repo,
        file,
        revision,
        variant,
        sha256,
        parts_sha256,
        ..
    } = r
    else {
        unreachable!("pull_catalog called with non-catalog ref");
    };
    eprintln!("basert pull v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  catalog: {hf_repo}/{file}@{revision} (pre-converted)");

    // The catalog names the logical `.base`; the repo may hold it as parts.
    let artifact = parts::find(fetcher, hf_repo, revision, file)?;

    let vdir = cache::variant_dir(root, id, variant)?;
    std::fs::create_dir_all(&vdir)?;
    let out = cache::base_artifact_path(&vdir);
    announce_parts(&artifact, &out);
    // The download lands beside the installed file (same filesystem, moved
    // from the staging area, so it exists once on disk) and replaces it only
    // once its sha256 is the row's: a forced pull that fails the check — a
    // `basert serve` refreshing a file for its head, `pull --force` — leaves
    // the file and its sidecar as they were rather than an empty slot.
    let pending = pending_path(&out);
    parts::install(
        fetcher,
        hf_repo,
        revision,
        &artifact,
        &pending,
        parts_sha256.as_deref(),
    )?;

    let got_sha = crate::compute_sha256_streaming(&pending)?;
    if let Some(expected) = sha256 {
        if !got_sha.eq_ignore_ascii_case(expected) {
            std::fs::remove_file(&pending).ok();
            // The staged bytes are equally bad — no resume value in them, so
            // clear them and let a retry start from scratch.
            fetch::cleanup_staging(fetcher, hf_repo);
            return Err(ShaMismatch {
                id: id.clone(),
                file: file.clone(),
                expected: expected.clone(),
                got: got_sha,
            }
            .into());
        }
        eprintln!("  sha256:  verified");
    }
    std::fs::rename(&pending, &out)
        .with_context(|| format!("moving {} into place", pending.display()))?;

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
        Some(file.as_str()),
    )?;
    count_download(fetcher, hf_repo, revision);
    fetch::cleanup_staging(fetcher, hf_repo);
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
    base_files: &Grouped,
    listed: Option<&CatalogEntry>,
    variant: Option<&str>,
) -> Result<()> {
    eprintln!("basert pull v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  source:  {repo}@{revision} (HuggingFace, pre-converted .base)");

    let want = quant_token(args);
    let artifact = match &args.file {
        Some(name) => select_named_base_file(base_files, name)?,
        None => select_base_file(base_files, &want)?,
    };
    let file = &artifact.name;
    // The catalog's own file keeps the catalog's variant and checksums,
    // however it was asked for.
    let listed = listed.filter(|e| e.file == *file);
    let variant = match variant {
        Some(v) => v.to_string(),
        None => listed.map_or_else(|| base_variant(file, &want), |e| e.quant.clone()),
    };
    eprintln!("  variant: {variant}");
    eprintln!("  file:    {file}");

    let vdir = cache::variant_dir(root, id, &variant)?;
    std::fs::create_dir_all(&vdir)?;
    let out = cache::base_artifact_path(&vdir);
    announce_parts(artifact, &out);

    // As pull_catalog: the download lands beside the installed file (moved
    // from the staging area; a part set is reassembled there first) and
    // replaces it only once it passes the catalog's sha256, so a failed
    // forced pull leaves the installed file and its sidecar as they were.
    let parts_sha256 = listed.and_then(|e| e.parts_sha256.as_deref());
    let pending = pending_path(&out);
    parts::install(fetcher, repo, revision, artifact, &pending, parts_sha256)?;

    let sha = crate::compute_sha256_streaming(&pending).ok();
    if let Some(expected) = listed.and_then(|e| e.sha256.as_deref()) {
        let got = sha.as_deref().unwrap_or("nothing readable");
        if !got.eq_ignore_ascii_case(expected) {
            std::fs::remove_file(&pending).ok();
            fetch::cleanup_staging(fetcher, repo);
            return Err(ShaMismatch {
                id: id.to_string(),
                file: file.clone(),
                expected: expected.to_string(),
                got: got.to_string(),
            }
            .into());
        }
        eprintln!("  sha256:  verified");
    }
    std::fs::rename(&pending, &out)
        .with_context(|| format!("moving {} into place", pending.display()))?;
    write_sidecar_for(
        &vdir,
        id,
        if listed.is_some() {
            "catalog"
        } else {
            "huggingface"
        },
        repo,
        None,
        revision,
        &variant,
        None,
        sha,
        Some(file.as_str()),
    )?;
    count_download(fetcher, repo, revision);
    fetch::cleanup_staging(fetcher, repo);
    eprintln!("installed {id} [{variant}] → {}", out.display());
    Ok(())
}

/// Hugging Face counts a repo's downloads by requests for its `config.json`.
/// Fetched for basecompute's own repos when one is published; a failure does
/// not fail the pull.
fn count_download(fetcher: &dyn Fetcher, repo: &str, revision: &str) {
    if repo.starts_with("basecompute/")
        && fetcher
            .list_files(repo, revision)
            .is_ok_and(|files| files.iter().any(|f| f == "config.json"))
    {
        let _ = fetcher.get_file(repo, revision, "config.json");
    }
}

/// One line about a split bundle, so the user knows why the install step
/// runs on after the download bar finishes and what the disk needs.
fn announce_parts(artifact: &Artifact, out: &Path) {
    if !artifact.is_split() {
        return;
    }
    // A re-pull keeps the installed bundle until the new one is complete,
    // so the honest figure for that path is one bundle more.
    let replacing = if out.exists() {
        " plus the installed bundle it replaces, kept until the new one is complete"
    } else {
        ""
    };
    eprintln!(
        "  parts:   {} (Hub-split; reassembled on install, needs bundle + one part of free disk{replacing})",
        artifact.parts.len()
    );
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

    let (quantize_drafter, keep_drafter_fc_f16) =
        pull_drafter_flags(drafter_sidecar_arch(&snapshot));
    let conv = ConvertArgs {
        input: snapshot,
        output: Some(out.clone()),
        mtp_head: None,
        mtp_head_version: None,
        target: args.target,
        calib: None,
        calib_tokens: 512,
        awq_mode: AwqMode::Full,
        synthetic: false,
        profile: profile_path,
        awq_profile: None,
        allow_quant_from_quant: false,
        no_mlx_passthrough: false,
        // Drafter sidecars (DFlash / DSpark / EAGLE-3) are pulled at the
        // target scheme, as the catalog's default-q4 drafters are; which taps
        // follow is pull_drafter_flags' call. A target bundle keeps the
        // previous behaviour — the flag would also quantize an in-bundle MTP
        // head, which has no acceptance measurement.
        quantize_drafter,
        keep_drafter_fc_f16,
        speculator_lm_head: false,
        mtp_draft_ranking: None,
        mtp_draft_vocab: None,
        // Convert-on-pull always produces a canonical-quant bundle;
        // k-quant passthrough stays an explicit `convert` opt-in.
        kquant_passthrough: false,
        // Convert-on-pull follows the converter default: a tied model gets
        // its head materialized so the profile head rule sets logit
        // precision (`plan_tied_lm_head`); the opt-out is a `convert` flag.
        no_synth_tied_head: false,
        // Convert-on-pull sources HF checkpoints, where the perception
        // tower already lives in the same snapshot — the mmproj flags only
        // apply to GGUF sources, which ship the tower separately.
        mmproj: None,
        mmproj_config: None,
        mlx_passthrough: false,
        validate: false,
        direct_write: false,
        imatrix: false,
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
        None,
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

/// The drafter sidecar architecture the snapshot's `config.json` names
/// (`dflash`, `dspark` or `eagle3`): the same detection `convert_hf` routes
/// on. None for an ordinary LM, and when the config cannot be read, so an odd
/// snapshot converts exactly as before.
fn drafter_sidecar_arch(snapshot: &Path) -> Option<&'static str> {
    let bytes = std::fs::read(snapshot.join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    base_arch::dflash::hf_mapper_for_architectures(&cfg)
        .or_else(|| base_arch::dflash::hf_mapper_for_eagle3(&cfg))
        .map(base_arch::HfMapper::canonical_arch)
}

/// `(quantize_drafter, keep_drafter_fc_f16)` for a convert-on-pull of a
/// snapshot whose sidecar arch is `sidecar` (None: a target model). A
/// sidecar's transformer layers follow the target scheme, as the catalog's
/// default-q4 drafters do. Its `fc.{i}` taps follow it for the DFlash family
/// only, where 4-bit taps were measured (acceptance unchanged on
/// Qwen3.8-27B-DFlash2 and Qwen3.6-35B-A3B-DFlash2, 2026-10-19); DSpark and
/// EAGLE-3 taps stay f16 until measured. `base-convert --quantize-drafter`
/// quantizes them regardless.
fn pull_drafter_flags(sidecar: Option<&str>) -> (bool, bool) {
    match sidecar {
        None => (false, false),
        Some("dflash") => (true, false),
        Some(_) => (true, true),
    }
}

/// Read `model_type` from a fetched `config.json` and fail early when no
/// converter supports it — so an unsupported repo errors before its weights
/// are downloaded. `text_config.model_type` is consulted as a fallback for
/// multimodal configs that nest the language-model arch there.
fn check_supported_arch(config_path: &Path, repo: &str, revision: &str) -> Result<()> {
    let bytes =
        std::fs::read(config_path).with_context(|| format!("reading {}", config_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {} as JSON", config_path.display()))?;

    let model_type = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| {
            cfg.pointer("/text_config/model_type")
                .and_then(|v| v.as_str())
        })
        .ok_or_else(|| anyhow::anyhow!("{repo}@{revision}: config.json has no model_type field"))?;

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
        let src = fetcher.get_file(repo, revision, f)?;
        link_into_snapshot(&snapshot, f, &src)?;
    }
    Ok(snapshot)
}

/// Make `filename` resolvable under `snapshot`, the dir the converter reads.
///
/// hf-hub's own download path writes a `snapshots/<rev>/<file>` pointer into
/// `blobs/`, but the resumable range path (every file of 32MB or more, i.e.
/// every safetensors shard) parks its result at `blobs/<etag>` and returns
/// that path with no pointer. Without this link the snapshot holds config
/// and tokenizer but no weights, and conversion fails with "no .safetensors
/// shards" on a pull that downloaded everything.
fn link_into_snapshot(snapshot: &Path, filename: &str, src: &Path) -> Result<()> {
    // `filename` comes from the remote repo's file listing. Reuse the cache's
    // rule so a hostile or malformed entry (`..`, an absolute path, a Windows
    // prefix) cannot place a symlink outside the snapshot — `Path::join`
    // would otherwise honour a leading `/` and discard the snapshot prefix.
    let rel = cache::id_to_relpath(filename)
        .with_context(|| format!("refusing to link unsafe source path {filename}"))?;
    let pointer = snapshot.join(rel);
    // A symlink target is stored verbatim and resolved RELATIVE TO THE LINK's
    // directory, not the process CWD. `BASERT_MODELS_DIR` is used exactly as
    // given (it is not canonicalised), so a relative models root would other-
    // wise write a target that dangles the moment it is read from inside the
    // snapshot directory. Absolutise before linking.
    let src = std::path::absolute(src)
        .with_context(|| format!("resolving fetched path {}", src.display()))?;
    let src = src.as_path();
    if pointer == src {
        return Ok(());
    }
    if let Ok(meta) = std::fs::symlink_metadata(&pointer) {
        let same = match (std::fs::canonicalize(&pointer), std::fs::canonicalize(src)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if same {
            return Ok(());
        }
        if !meta.file_type().is_symlink() {
            bail!(
                "{} exists and is not the fetched {}",
                pointer.display(),
                src.display()
            );
        }
        // A stale or dangling pointer from an earlier pull: replace it.
        std::fs::remove_file(&pointer)
            .with_context(|| format!("replacing stale {}", pointer.display()))?;
    }
    if let Some(parent) = pointer.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::os::unix::fs::symlink(src, &pointer)
        .with_context(|| format!("linking {} -> {}", pointer.display(), src.display()))
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
    file: Option<&str>,
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
            file: file.map(|s| s.to_string()),
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

/// `basert catalog-scan` — regenerate the catalog from what an organization
/// has actually published.
///
/// The catalog used to be hand-edited, one JSON row per (model, quant,
/// backend), and drifted from the Hub the moment anyone published without
/// remembering to add a row. This derives it instead: the org listing names
/// the repos, each repo's tree gives size and sha256, and two ranged reads of
/// each bundle give the header that knows its arch, backend and quant. A row
/// whose sha256 is unchanged is carried over untouched, so a rescan is cheap
/// and hand-curated fields (`source_repo`) survive.
pub fn cmd_catalog_scan(org: String, out: Option<PathBuf>, dry_run: bool) -> Result<()> {
    let root = cache::models_dir()?;
    let fetcher = HfFetcher::new(cache::hf_staging_dir(&root))?;

    let known = base_hub::catalog::Catalog::bundled()
        .context("reading the bundled catalog to reuse unchanged rows")?;
    eprintln!("scanning {org} …");
    let repos = base_hub::scan::list_org_repos(&org)?;
    eprintln!("  {} repositories", repos.len());

    let report = base_hub::scan::scan_org(&fetcher, &base_hub::scan::HubApi, &org, &repos, &known)?;

    // What the scan decided, before anything is written. Silence here would
    // make a dropped bundle indistinguishable from one that does not exist.
    eprintln!(
        "  {} rows ({} reused unchanged), {} repos with no bundle",
        report.entries.len(),
        report.reused,
        report.empty_repos.len()
    );
    for (what, why) in &report.skipped {
        eprintln!("  SKIPPED {what}: {why}");
    }

    let before: std::collections::BTreeSet<_> = known
        .models
        .iter()
        .map(|m| (m.hf_repo.clone(), m.file.clone()))
        .collect();
    let after: std::collections::BTreeSet<_> = report
        .entries
        .iter()
        .map(|m| (m.hf_repo.clone(), m.file.clone()))
        .collect();
    for (repo, file) in after.difference(&before) {
        eprintln!("  + {repo}/{file}");
    }
    for (repo, file) in before.difference(&after) {
        eprintln!("  - {repo}/{file}  (no longer published)");
    }

    let path = out.unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|exe| workspace_catalog(&exe))
            .unwrap_or_else(|| PathBuf::from("catalog.json"))
    });
    if dry_run {
        eprintln!("dry run — {} not written", path.display());
        return Ok(());
    }

    let doc = base_hub::catalog::Catalog {
        schema: known.schema,
        updated: report.updated_stamp(),
        models: report.entries,
    };
    let json = serde_json::to_string_pretty(&doc)? + "\n";
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// The bundled catalog's source in the workspace a source-built launcher sits
/// in (`<workspace>/target/<profile>/basert`): `catalog-scan`'s in-place
/// default. Found at run time because `env!("CARGO_MANIFEST_DIR")` compiled
/// the build machine's checkout path into every release binary.
fn workspace_catalog(exe: &Path) -> Option<PathBuf> {
    exe.ancestors()
        .map(|dir| dir.join("crates/base-hub/catalog.json"))
        .find(|catalog| catalog.is_file())
}

/// Write the manifest a split bundle needs beside its parts.
///
/// `bundle` is the logical `.base` path the parts are named after
/// (`parts/GLM-5.2-Q4.base`, which need not exist); its `.part-NNN` siblings
/// are hashed in order, each and as one whole, and the result lands at
/// `<bundle>.manifest.json` ready to upload with the parts.
pub fn cmd_catalog_manifest(bundle: PathBuf) -> Result<()> {
    let dir = bundle
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = bundle
        .file_name()
        .and_then(|n| n.to_str())
        .context("bundle path has no file name")?
        .to_string();
    // Every entry is read or the command fails: an entry lost to an I/O
    // error could be the terminal part, and a manifest written without it
    // would describe a truncated bundle as complete.
    let mut parts: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&dir).with_context(|| format!("listing {}", dir.display()))? {
        let entry = entry.with_context(|| format!("reading an entry of {}", dir.display()))?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if let Some((logical, idx)) = base_hub::parts::split_part_name(&name) {
            if logical == stem {
                parts.push((idx, entry.path()));
            }
        }
    }
    parts.sort();
    if parts.is_empty() {
        bail!("no {stem}.part-NNN files in {}", dir.display());
    }
    for (want, (have, path)) in (0u32..).zip(&parts) {
        if *have != want {
            bail!(
                "part set is missing part {want:03} (next is {})",
                path.display()
            );
        }
    }
    let paths: Vec<PathBuf> = parts.into_iter().map(|(_, p)| p).collect();
    eprintln!("hashing {} parts of {stem} …", paths.len());
    let manifest = base_hub::parts::build_manifest(&paths)?;
    let out = dir.join(base_hub::parts::manifest_name(&stem));
    std::fs::write(&out, serde_json::to_string_pretty(&manifest)? + "\n")
        .with_context(|| format!("writing {}", out.display()))?;
    eprintln!(
        "  size:   {} bytes\n  sha256: {}\nwrote {}",
        manifest.size,
        manifest.sha256,
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_hub::fetch::MockFetcher;

    #[test]
    fn a_pulled_snapshot_is_a_drafter_sidecar_by_what_its_config_names() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(drafter_sidecar_arch(tmp.path()), None, "no config.json");
        let config = tmp.path().join("config.json");
        for (text, arch) in [
            (
                r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}"#,
                None,
            ),
            (
                r#"{"architectures": ["DFlashDraftModel"], "model_type": "qwen3"}"#,
                Some("dflash"),
            ),
            (
                r#"{"architectures": ["DFlash2DraftModel"]}"#,
                Some("dflash"),
            ),
            (r#"{"architectures": ["Qwen3DSparkModel"]}"#, Some("dspark")),
            (
                r#"{"architectures": ["LlamaForCausalLMEagle3"]}"#,
                Some("eagle3"),
            ),
            // nebius' EAGLE-3 heads say LlamaForCausalLM: one layer with a
            // draft vocabulary.
            (
                r#"{"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1, "draft_vocab_size": 32000}"#,
                Some("eagle3"),
            ),
            ("not json", None),
        ] {
            std::fs::write(&config, text).unwrap();
            assert_eq!(drafter_sidecar_arch(tmp.path()), arch, "{text}");
        }
    }

    #[test]
    fn a_pull_quantizes_a_sidecars_layers_and_only_the_measured_fc_taps() {
        assert_eq!(pull_drafter_flags(None), (false, false));
        assert_eq!(pull_drafter_flags(Some("dflash")), (true, false));
        assert_eq!(pull_drafter_flags(Some("dspark")), (true, true));
        assert_eq!(pull_drafter_flags(Some("eagle3")), (true, true));
    }

    #[test]
    fn bundled_computearena_harness_requires_a_sibling_and_respects_overrides() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(bundled_computearena_harness(Some(tmp.path()), false), None);

        let harness = tmp.path().join(BENCHMARK_HARNESS_BINARY);
        std::fs::write(&harness, b"fixture").unwrap();
        assert_eq!(
            bundled_computearena_harness(Some(tmp.path()), false),
            Some(harness)
        );
        assert_eq!(bundled_computearena_harness(Some(tmp.path()), true), None);
        assert_eq!(bundled_computearena_harness(None, false), None);
    }

    #[test]
    fn catalog_scan_writes_the_bundled_catalog_of_a_source_built_launcher() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().join("baseRT/tools/base-convert");
        let catalog = workspace.join("crates/base-hub/catalog.json");
        std::fs::create_dir_all(catalog.parent().unwrap()).unwrap();
        std::fs::create_dir_all(workspace.join("target/release")).unwrap();
        std::fs::write(&catalog, b"{}").unwrap();
        assert_eq!(
            workspace_catalog(&workspace.join("target/release/basert")),
            Some(catalog)
        );
        // An installed launcher has no workspace around it.
        assert_eq!(workspace_catalog(&tmp.path().join("bin/basert")), None);
    }

    #[test]
    fn bundled_computearena_harness_supports_the_source_build_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("baseRT");
        let exe_dir = repo.join("tools/base-convert/target/release");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::write(repo.join("tools/base-convert/Cargo.toml"), b"[workspace]\n").unwrap();

        let harness = repo.join("build").join(BENCHMARK_HARNESS_BINARY);
        std::fs::create_dir_all(harness.parent().unwrap()).unwrap();
        std::fs::write(&harness, b"fixture").unwrap();

        assert_eq!(
            bundled_computearena_harness(Some(&exe_dir), false),
            Some(harness)
        );
    }

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
            Artifact::whole("Llama-3.2-1B-Instruct-Q4.base"),
            Artifact::whole("Llama-3.2-1B-Instruct-Q8.base"),
        ];
        let files = Grouped {
            artifacts: files,
            malformed: vec![],
        };
        assert_eq!(select_base_file(&files, "q4").unwrap(), &files.artifacts[0]);
        assert_eq!(select_base_file(&files, "q8").unwrap(), &files.artifacts[1]);
        // No match among several → error that lists what's available.
        let err = select_base_file(&files, "q2").unwrap_err().to_string();
        assert!(err.contains("q4") && err.contains("q8"), "{err}");
        // Sole artifact → used regardless of requested quant.
        let one = Grouped {
            artifacts: vec![Artifact::whole("model.base")],
            malformed: vec![],
        };
        assert_eq!(select_base_file(&one, "q4").unwrap(), &one.artifacts[0]);
    }

    #[test]
    fn select_base_file_reports_an_incomplete_requested_quant() {
        // A gapped Q4 upload beside a complete Q8: asking for Q4 is told
        // the Q4 publication is incomplete, not quietly handed Q8.
        let files = Grouped {
            artifacts: vec![Artifact::whole("m-Q8.base")],
            malformed: vec![(
                "m-Q4.base".to_string(),
                "part set is missing part 001 (found 2 parts)".to_string(),
            )],
        };
        let err = select_base_file(&files, "q4").unwrap_err().to_string();
        assert!(err.contains("incomplete"), "{err}");
        assert!(err.contains("missing part 001"), "{err}");
        // Q8 is still there for whoever asks for it.
        assert_eq!(select_base_file(&files, "q8").unwrap().name, "m-Q8.base");
        // And a third quant sees both, the broken one marked.
        let err = select_base_file(&files, "q2").unwrap_err().to_string();
        assert!(
            err.contains("q4 (incomplete)") && err.contains("q8"),
            "{err}"
        );
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
        let mut got: Vec<String> = list_base_files(&fetcher, "basecompute/m", "main")
            .unwrap()
            .artifacts
            .into_iter()
            .map(|a| a.name)
            .collect();
        got.sort();
        assert_eq!(got, vec!["m-Q4.base".to_string(), "m-Q8.base".to_string()]);
    }

    #[test]
    fn list_base_files_groups_a_part_set_under_its_logical_name() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("basecompute").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        for f in ["m-Q4.base.part-000", "m-Q4.base.part-001", "README.md"] {
            std::fs::write(repo_dir.join(f), b"x").unwrap();
        }
        let fetcher = MockFetcher::new(tmp.path());
        let got = list_base_files(&fetcher, "basecompute/m", "main").unwrap();
        assert_eq!(got.artifacts.len(), 1);
        assert_eq!(got.artifacts[0].name, "m-Q4.base");
        assert_eq!(got.artifacts[0].parts.len(), 2);
        // The logical name carries the quant tag the selector matches on.
        assert_eq!(select_base_file(&got, "q4").unwrap().name, "m-Q4.base");
    }

    #[test]
    fn pull_base_direct_reassembles_a_split_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_dir = tmp.path().join("basecompute").join("m");
        std::fs::create_dir_all(&repo_dir).unwrap();
        // A real tiny bundle, cut in two: the install checks the stitched
        // file against its own header, so arbitrary bytes will not do.
        let whole = base_hub::parts::synthetic_bundle(1000);
        let (head, tail) = whole.split_at(whole.len() / 2);
        std::fs::write(repo_dir.join("m-Q4.base.part-000"), head).unwrap();
        std::fs::write(repo_dir.join("m-Q4.base.part-001"), tail).unwrap();
        // The manifest the publisher ships beside the parts, made the way
        // they would make it.
        cmd_catalog_manifest(repo_dir.join("m-Q4.base")).unwrap();
        assert!(repo_dir.join("m-Q4.base.manifest.json").exists());
        let fetcher = MockFetcher::new(tmp.path());
        let base_files = list_base_files(&fetcher, "basecompute/m", "main").unwrap();

        let root = tmp.path().join("cache");
        let args = PullArgs {
            id: "basecompute/m".into(),
            profile: None,
            target: TargetScheme::BaseQ4,
            revision: "main".into(),
            file: None,
            force: false,
            dry_run: false,
        };
        pull_base_direct(
            &root,
            &args,
            "basecompute/m",
            "basecompute/m",
            "main",
            &fetcher,
            &base_files,
            None,
            None,
        )
        .unwrap();

        let vdir = root.join("basecompute/m/default-q4");
        assert_eq!(std::fs::read(vdir.join("model.base")).unwrap(), whole);
        assert!(vdir.join("hub.json").exists());
        // No reassembly leftovers beside the installed artifact.
        let stray: Vec<_> = std::fs::read_dir(&vdir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("partial"))
            .collect();
        assert!(stray.is_empty(), "leftovers: {stray:?}");
    }

    #[test]
    fn config_json_is_requested_only_from_a_basecompute_repo_that_has_one() {
        struct Recording(std::cell::RefCell<Vec<String>>, Vec<&'static str>);
        impl Fetcher for Recording {
            fn get_file(&self, _repo: &str, _revision: &str, filename: &str) -> Result<PathBuf> {
                self.0.borrow_mut().push(filename.to_string());
                Ok(PathBuf::from(filename))
            }
            fn list_files(&self, _repo: &str, _revision: &str) -> Result<Vec<String>> {
                Ok(self.1.iter().map(|f| f.to_string()).collect())
            }
        }
        let with = Recording(Default::default(), vec!["m-Q4.base", "config.json"]);
        count_download(&with, "basecompute/m", "main");
        assert_eq!(*with.0.borrow(), ["config.json"]);
        let without = Recording(Default::default(), vec!["m-Q4.base"]);
        count_download(&without, "basecompute/m", "main");
        assert!(without.0.borrow().is_empty());
        let elsewhere = Recording(Default::default(), vec!["m-Q4.base", "config.json"]);
        count_download(&elsewhere, "someone/m", "main");
        assert!(elsewhere.0.borrow().is_empty());
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
            file: None,
            force: false,
            dry_run: false,
        };
        pull_base_direct(
            &root,
            &args,
            "basecompute/m",
            "basecompute/m",
            "main",
            &fetcher,
            &base_files,
            None,
            None,
        )
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

    #[test]
    fn select_named_base_file_matches_by_name_and_lists_the_rest() {
        let files = Grouped {
            artifacts: vec![
                Artifact::whole("m-Small.base"),
                Artifact::whole("sub/m-Large.base"),
            ],
            malformed: vec![(
                "m-Medium.base".into(),
                "part set is missing part 001".into(),
            )],
        };
        assert_eq!(
            select_named_base_file(&files, "m-Small.base").unwrap().name,
            "m-Small.base"
        );
        assert_eq!(
            select_named_base_file(&files, "m-Large").unwrap().name,
            "sub/m-Large.base"
        );
        let err = select_named_base_file(&files, "m-Medium.base")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("incomplete") && err.contains("missing part 001"),
            "{err}"
        );
        let err = select_named_base_file(&files, "m-Huge.base")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("m-Small.base") && err.contains("sub/m-Large.base"),
            "{err}"
        );
    }

    #[test]
    fn variants_of_quant_picked_and_named_files() {
        assert_eq!(base_variant("m-Q8.base", "q4"), "default-q8");
        assert_eq!(base_variant("model.base", "q4"), "default-q4");
        let medium = named_variant("m-Medium.base");
        assert!(
            medium.starts_with("file-m-medium-") && medium.len() == "file-m-medium-".len() + 8,
            "{medium}"
        );
        // paths the readable part folds together keep variants of their own
        let folded = ["a/model.base", "a-model.base", "A-Model.base"].map(named_variant);
        assert!(
            folded.iter().all(|v| v.starts_with("file-a-model-")),
            "{folded:?}"
        );
        for wide in ["é", "𝔸"] {
            let long = format!("{}/model.base", wide.repeat(200));
            let v = named_variant(&long);
            assert!(v.len() <= "file-".len() + 64 + 9, "{} bytes", v.len());
            assert_ne!(
                v,
                named_variant(&format!("{}/other.base", wide.repeat(200)))
            );
        }
        assert!(
            folded[0] != folded[1] && folded[1] != folded[2] && folded[0] != folded[2],
            "{folded:?}"
        );
    }

    #[test]
    fn a_path_picks_among_files_that_share_a_name() {
        let files = Grouped {
            artifacts: vec![
                Artifact::whole("a/model.base"),
                Artifact::whole("b/model.base"),
            ],
            malformed: vec![],
        };
        assert_eq!(
            select_named_base_file(&files, "b/model.base").unwrap().name,
            "b/model.base"
        );
        assert_eq!(
            select_named_base_file(&files, "a/model").unwrap().name,
            "a/model.base"
        );
        let err = select_named_base_file(&files, "model.base")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("names several files") && err.contains("b/model.base"),
            "{err}"
        );
        assert!(select_named_base_file(&files, "c/model.base").is_err());

        // An incomplete upload by that name counts toward the ambiguity.
        let files = Grouped {
            artifacts: vec![Artifact::whole("a/model.base")],
            malformed: vec![("b/model.base".into(), "part set is missing part 001".into())],
        };
        let err = select_named_base_file(&files, "model.base")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("names several files") && err.contains("b/model.base (incomplete)"),
            "{err}"
        );
        let err = select_named_base_file(&files, "b/model.base")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("incomplete") && err.contains("missing part 001"),
            "{err}"
        );
        assert_eq!(
            select_named_base_file(&files, "a/model.base").unwrap().name,
            "a/model.base"
        );
    }

    fn named(file: &str, dry_run: bool, id: &str) -> PullArgs {
        PullArgs {
            id: id.into(),
            profile: None,
            target: TargetScheme::BaseQ4,
            revision: "main".into(),
            file: Some(file.into()),
            force: false,
            dry_run,
        }
    }

    fn pull_named(reg: &MergedRegistry, fetcher: &dyn Fetcher, args: &PullArgs) -> Result<()> {
        pull_named_file(reg, args, args.file.as_deref().unwrap(), fetcher)
    }

    fn catalog(json_models: &str) -> base_hub::registry::CatalogRegistry {
        base_hub::registry::CatalogRegistry::from_catalog(
            base_hub::catalog::Catalog::from_json(&format!(
                r#"{{"schema":1,"models":[{json_models}]}}"#
            ))
            .unwrap(),
        )
    }

    /// A flat fixture repo `MockFetcher` serves from `root`.
    fn fixture_repo(root: &Path, repo: &str, files: &[(&str, &[u8])]) {
        let dir = root.join(repo);
        std::fs::create_dir_all(&dir).unwrap();
        for (f, bytes) in files {
            std::fs::write(dir.join(f), bytes).unwrap();
        }
    }

    /// A repo listing and nothing to download: what the cache checks see.
    struct Listing(Vec<&'static str>);

    impl Fetcher for Listing {
        fn get_file(&self, _: &str, _: &str, f: &str) -> Result<PathBuf> {
            bail!("no download expected ({f})")
        }
        fn list_files(&self, _: &str, _: &str) -> Result<Vec<String>> {
            Ok(self.0.iter().map(|f| f.to_string()).collect())
        }
    }

    /// An installed variant as an earlier pull (of any kind) would leave it.
    fn install(
        reg: &MergedRegistry,
        id: &str,
        variant: &str,
        file: Option<&str>,
        kind: &str,
        profile: Option<&str>,
    ) {
        let vdir = cache::variant_dir(&reg.root, id, variant).unwrap();
        std::fs::create_dir_all(&vdir).unwrap();
        std::fs::write(cache::base_artifact_path(&vdir), b"x").unwrap();
        write_sidecar_for(
            &vdir, id, kind, id, None, "main", variant, profile, None, file,
        )
        .unwrap();
    }

    fn held(reg: &MergedRegistry, id: &str, variant: &str) -> Option<String> {
        let vdir = cache::variant_dir(&reg.root, id, variant).unwrap();
        cache::read_sidecar(&vdir).unwrap().unwrap().file
    }

    /// A repo whose `.base` files are named by size, not quant (the BaseQuant
    /// releases): no quant picks one, `--file` does, each into a variant of
    /// its own that records the file, which `basert list` reports.
    #[test]
    fn a_size_named_repo_installs_each_file_under_its_own_variant() {
        let tmp = tempfile::tempdir().unwrap();
        fixture_repo(
            tmp.path(),
            "basecompute/m",
            &[
                ("m-Small.base", b"small"),
                ("m-Medium.base", b"medium"),
                ("m-Medium.gguf", b"gguf"),
            ],
        );
        let fetcher = MockFetcher::new(tmp.path());
        let base_files = list_base_files(&fetcher, "basecompute/m", "main").unwrap();
        let err = select_base_file(&base_files, "q4").unwrap_err().to_string();
        assert!(
            err.contains("--file") && err.contains("m-Medium.base"),
            "{err}"
        );

        let reg = MergedRegistry::new(tmp.path().join("cache"), catalog(""));
        pull_named(
            &reg,
            &fetcher,
            &named("m-Medium.base", false, "basecompute/m"),
        )
        .unwrap();
        pull_named(&reg, &fetcher, &named("m-Small", false, "basecompute/m")).unwrap();
        let m = reg.root.join("basecompute/m");
        assert_eq!(
            std::fs::read(m.join(named_variant("m-Medium.base")).join("model.base")).unwrap(),
            b"medium"
        );
        assert_eq!(
            std::fs::read(m.join(named_variant("m-Small.base")).join("model.base")).unwrap(),
            b"small"
        );
        assert_eq!(
            held(&reg, "basecompute/m", &named_variant("m-Medium.base")).as_deref(),
            Some("m-Medium.base")
        );
        let rows = reg.local.list().unwrap();
        let files: Vec<_> = rows.iter().filter_map(|r| r.file.as_deref()).collect();
        assert_eq!(files, ["m-Medium.base", "m-Small.base"]);
        // and a quant never reaches one by its bits
        assert!(installed_best_variant(&reg, "basecompute/m", "default-q4").is_none());
    }

    /// A cached file answers only for itself: the name is resolved against
    /// the repo first, and whatever else is in the variant is refused.
    #[test]
    fn a_cached_file_answers_only_for_itself() {
        let tmp = tempfile::tempdir().unwrap();
        fixture_repo(tmp.path(), "basecompute/m", &[("m-Medium.base", b"medium")]);
        let fetcher = MockFetcher::new(tmp.path());
        let reg = MergedRegistry::new(tmp.path().join("cache"), catalog(""));
        let medium = reg
            .root
            .join("basecompute/m")
            .join(named_variant("m-Medium.base"))
            .join("model.base");

        pull_named(
            &reg,
            &fetcher,
            &named("m-Medium.base", true, "basecompute/m"),
        )
        .unwrap();
        assert!(!medium.exists(), "a dry run installs nothing");
        pull_named(
            &reg,
            &fetcher,
            &named("m-Medium.base", false, "basecompute/m"),
        )
        .unwrap();
        std::fs::write(&medium, b"cached").unwrap();
        for dry in [false, true] {
            pull_named(&reg, &fetcher, &named("m-Medium", dry, "basecompute/m")).unwrap();
        }
        assert_eq!(
            std::fs::read(&medium).unwrap(),
            b"cached",
            "a cache hit downloads nothing"
        );
        let err = pull_named(
            &reg,
            &fetcher,
            &named("Typo-Medium.base", true, "basecompute/m"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("no .base named"), "{err}");

        for (file, profile, says) in [
            (Some("other.base"), None, "holds other.base"),
            (None, Some("default-q4"), "holds a conversion"),
            (None, None, "an install from another source"),
        ] {
            install(
                &reg,
                "basecompute/m",
                &named_variant("m-Medium.base"),
                file,
                "huggingface",
                profile,
            );
            for dry in [false, true] {
                let err = pull_named(
                    &reg,
                    &fetcher,
                    &named("m-Medium.base", dry, "basecompute/m"),
                )
                .unwrap_err();
                assert!(
                    err.to_string().contains(says) && err.to_string().contains("--force"),
                    "{err}"
                );
            }
        }
        let mut force = named("m-Medium.base", false, "basecompute/m");
        force.force = true;
        pull_named(&reg, &fetcher, &force).unwrap();
        assert_eq!(std::fs::read(&medium).unwrap(), b"medium");
    }

    /// A bare name is resolved against the repo listing whether or not its
    /// file is cached, so it is ambiguous exactly when a fresh pull says so;
    /// a path picks among same-named files, which get variants of their own.
    #[test]
    fn a_bare_name_is_resolved_against_the_repo_even_when_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = MergedRegistry::new(tmp.path(), catalog(""));
        install(
            &reg,
            "basecompute/m",
            &named_variant("model.base"),
            Some("model.base"),
            "huggingface",
            None,
        );
        install(
            &reg,
            "basecompute/m",
            &named_variant("a/model.base"),
            Some("a/model.base"),
            "huggingface",
            None,
        );
        pull_named(
            &reg,
            &Listing(vec!["model.base"]),
            &named("model.base", false, "basecompute/m"),
        )
        .unwrap();
        let grown = Listing(vec!["model.base", "a/model.base", "b/model.base"]);
        for bare in ["model.base", "model"] {
            let err = pull_named(&reg, &grown, &named(bare, false, "basecompute/m")).unwrap_err();
            assert!(err.to_string().contains("names several files"), "{err}");
        }
        pull_named(&reg, &grown, &named("a/model.base", true, "basecompute/m")).unwrap();
        let err =
            pull_named(&reg, &grown, &named("b/model.base", false, "basecompute/m")).unwrap_err();
        assert!(
            err.to_string().contains("no download expected"),
            "b/model is not cached: {err}"
        );
        // an incomplete upload by that name counts toward the ambiguity too
        let files = Grouped {
            artifacts: vec![Artifact::whole("a/model.base")],
            malformed: vec![("b/model.base".into(), "part set is missing part 001".into())],
        };
        let err = select_named_base_file(&files, "model.base")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("names several files") && err.contains("b/model.base (incomplete)"),
            "{err}"
        );
    }

    /// A file a catalog row publishes is that row's: from its repo, under its
    /// variant, refused for another backend, answered by an older catalog
    /// pull of it; a bare name two rows publish must be given with its path.
    #[test]
    fn a_catalog_file_is_its_rows() {
        let here = CatalogRegistry::client_backend();
        let there = if here == "cuda" { "metal" } else { "cuda" };
        let tmp = tempfile::tempdir().unwrap();
        let mine = format!("cat-{here}-q4mix.base");
        let theirs = format!("cat-{there}-q4mix.base");
        fixture_repo(
            tmp.path(),
            "basecompute/cat-a",
            &[
                ("cat-Q4.base", b"q4"),
                (&mine, b"mine"),
                (&theirs, b"theirs"),
                ("Experimental-Q4.base", b"exp"),
                ("Both.base", b"a"),
            ],
        );
        fixture_repo(
            tmp.path(),
            "basecompute/cat-b",
            &[
                ("cat-Q8.base", b"q8"),
                ("Only-B.base", b"b"),
                ("Both.base", b"b"),
            ],
        );
        let row = |repo: &str, quant: &str, file: &str, backend: &str| {
            format!(
                r#"{{"id":"basecompute/cat","hf_repo":"basecompute/{repo}","quant":"{quant}","file":"{file}"{backend}}}"#
            )
        };
        let reg = MergedRegistry::new(
            tmp.path().join("cache"),
            catalog(
                &[
                    row("cat-a", "default-q4", "cat-Q4.base", r#","sha256":"112f2dfa31205df3f5f9db109460c8b85067d7dc27f70781c2e4ef903ee9f26a""#),
                    row("cat-b", "default-q8", "cat-Q8.base", ""),
                    row(
                        "cat-a",
                        &format!("{here}-q4mix"),
                        &mine,
                        &format!(r#","backend":"{here}""#),
                    ),
                    row(
                        "cat-a",
                        &format!("{there}-q4mix"),
                        &theirs,
                        &format!(r#","backend":"{there}""#),
                    ),
                ]
                .join(","),
            ),
        );
        let fetcher = MockFetcher::new(tmp.path());
        let cat = reg.root.join("basecompute/cat");

        pull_named(
            &reg,
            &fetcher,
            &named("cat-Q8.base", false, "basecompute/cat"),
        )
        .unwrap();
        assert_eq!(
            std::fs::read(cat.join("default-q8/model.base")).unwrap(),
            b"q8",
            "from its own row's repo"
        );
        pull_named(&reg, &fetcher, &named(&mine, false, "basecompute/cat")).unwrap();
        assert_eq!(
            std::fs::read(cat.join(format!("{here}-q4mix/model.base"))).unwrap(),
            b"mine"
        );
        let err =
            pull_named(&reg, &fetcher, &named(&theirs, false, "basecompute/cat")).unwrap_err();
        assert!(
            err.to_string().contains(&format!("the {there} bundle")),
            "{err}"
        );
        assert!(!cat.join(format!("{there}-q4mix")).exists());

        // an unlisted file never takes a catalog variant, now or later
        pull_named(
            &reg,
            &fetcher,
            &named("Experimental-Q4.base", false, "basecompute/cat"),
        )
        .unwrap();
        assert!(
            cat.join(named_variant("Experimental-Q4.base"))
                .join("model.base")
                .exists()
                && !cat.join("default-q4").exists()
        );

        // an unlisted file is looked for in every repo the id's rows use
        pull_named(
            &reg,
            &fetcher,
            &named("Only-B.base", false, "basecompute/cat"),
        )
        .unwrap();
        let only_b = cat.join(named_variant("Only-B.base")).join("model.base");
        assert_eq!(std::fs::read(only_b).unwrap(), b"b");
        let err =
            pull_named(&reg, &fetcher, &named("Both.base", true, "basecompute/cat")).unwrap_err();
        assert!(
            err.to_string().contains("several repos") && err.to_string().contains("cat-b"),
            "{err}"
        );

        // a catalog pull from before files were recorded is its row's file
        // only when its bytes are the row's
        let legacy = |sha: &str| {
            let vdir = cache::variant_dir(&reg.root, "basecompute/cat", "default-q4").unwrap();
            std::fs::create_dir_all(&vdir).unwrap();
            std::fs::write(cache::base_artifact_path(&vdir), b"q4").unwrap();
            let id = "basecompute/cat";
            write_sidecar_for(
                &vdir,
                id,
                "catalog",
                id,
                None,
                "main",
                "default-q4",
                None,
                Some(sha.into()),
                None,
            )
            .unwrap();
        };
        legacy(&"0".repeat(64));
        let err = pull_named(
            &reg,
            &fetcher,
            &named("cat-Q4.base", false, "basecompute/cat"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        legacy("112f2dfa31205df3f5f9db109460c8b85067d7dc27f70781c2e4ef903ee9f26a");
        pull_named(
            &reg,
            &fetcher,
            &named("cat-Q4.base", false, "basecompute/cat"),
        )
        .unwrap();
        assert_eq!(
            held(&reg, "basecompute/cat", "default-q4").as_deref(),
            Some("cat-Q4.base")
        );

        let two = MergedRegistry::new(
            tmp.path().join("two"),
            catalog(
                &[
                    row("cat-a", "default-q4", "a/model.base", ""),
                    row("cat-b", "default-q8", "b/model.base", ""),
                ]
                .join(","),
            ),
        );
        let err = pull_named(
            &two,
            &Listing(vec!["a/model.base"]),
            &named("model.base", true, "basecompute/cat"),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("names several files the catalog lists"),
            "{err}"
        );
    }

    /// A cached catalog behind the Hub — the row's sha256 is the old file's,
    /// the Hub serves the new one — fails a fresh pull's checksum; the pull
    /// then fetches the live catalog and, when the live row names the new
    /// sha256, pulls once more against it. A live row that agrees with the
    /// stale one (the Hub really disagrees), or no live catalog at all, leaves
    /// the error, after one attempt.
    #[test]
    fn a_stale_catalog_row_is_retried_from_the_live_catalog() {
        use sha2::{Digest, Sha256};
        let tmp = tempfile::tempdir().unwrap();
        fixture_repo(
            tmp.path(),
            "basecompute/cat-files",
            &[("cat-Medium.base", b"medium")],
        );
        let fetcher = MockFetcher::new(tmp.path());
        let real = format!("{:x}", Sha256::digest(b"medium"));
        let row = |sha: &str| {
            catalog(&format!(
                r#"{{"id":"basecompute/cat","hf_repo":"basecompute/cat-files","quant":"default-medium","file":"cat-Medium.base","sha256":"{sha}"}}"#
            ))
        };
        let root = tmp.path().join("hub");
        let stale = MergedRegistry::new(root.clone(), row(&"0".repeat(64)));
        let out = root.join("basecompute/cat/default-medium/model.base");
        let attempts = std::cell::Cell::new(0);
        let attempt = |reg: &MergedRegistry| {
            attempts.set(attempts.get() + 1);
            let r = reg
                .resolve("basecompute/cat", "main", Some("default-medium"), false)
                .unwrap();
            pull_catalog(&reg.root, &r, &fetcher)
        };

        // The live catalog knows the new file: the retry installs it.
        with_fresh_catalog_on_mismatch(
            &stale,
            || Some(MergedRegistry::new(root.clone(), row(&real))),
            attempt,
        )
        .unwrap();
        assert_eq!(attempts.get(), 2);
        assert_eq!(std::fs::read(&out).unwrap(), b"medium");
        let s = cache::read_sidecar(out.parent().unwrap()).unwrap().unwrap();
        assert_eq!(s.base_sha256.as_deref(), Some(real.as_str()));
        std::fs::remove_dir_all(&root).unwrap();

        // The live row agrees with the stale one: the Hub is wrong, one attempt.
        attempts.set(0);
        let err = with_fresh_catalog_on_mismatch(
            &stale,
            || Some(MergedRegistry::new(root.clone(), row(&"0".repeat(64)))),
            attempt,
        )
        .unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert_eq!(attempts.get(), 1);
        assert!(!out.exists());

        // No live catalog to be had: the error as it was.
        attempts.set(0);
        let err = with_fresh_catalog_on_mismatch(&stale, || None, attempt).unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert_eq!(attempts.get(), 1);
    }

    /// A pull that fails the catalog's sha256 leaves an installed file and
    /// its sidecar as they were, on both pull paths: the download is checked
    /// beside them and moved into place only when it is the row's file. A
    /// forced pull is what `basert serve` runs to refresh a file for its MTP
    /// head, so a catalog and a Hub file that disagree must not cost the
    /// model that was being served.
    #[test]
    fn a_failed_checksum_leaves_the_installed_file_alone() {
        let tmp = tempfile::tempdir().unwrap();
        fixture_repo(
            tmp.path(),
            "basecompute/cat-files",
            &[("cat-Medium.base", b"medium")],
        );
        let fetcher = MockFetcher::new(tmp.path());
        let reg = MergedRegistry::new(
            tmp.path().join("hub"),
            catalog(&format!(
                r#"{{"id":"basecompute/cat","hf_repo":"basecompute/cat-files","quant":"default-medium","file":"cat-Medium.base","sha256":"{}"}}"#,
                "0".repeat(64)
            )),
        );
        // What an earlier pull installed, with the sidecar it recorded.
        let vdir = cache::variant_dir(&reg.root, "basecompute/cat", "default-medium").unwrap();
        std::fs::create_dir_all(&vdir).unwrap();
        let out = cache::base_artifact_path(&vdir);
        std::fs::write(&out, b"the installed one").unwrap();
        write_sidecar_for(
            &vdir,
            "basecompute/cat",
            "catalog",
            "basecompute/cat-files",
            None,
            "main",
            "default-medium",
            None,
            Some("installed-sha".into()),
            Some("cat-Medium.base"),
        )
        .unwrap();
        let untouched = |what: &str| {
            assert_eq!(
                std::fs::read(&out).unwrap(),
                b"the installed one",
                "{what}: the file"
            );
            let s = cache::read_sidecar(&vdir).unwrap().unwrap();
            assert_eq!(
                s.base_sha256.as_deref(),
                Some("installed-sha"),
                "{what}: the sidecar"
            );
            assert!(
                !pending_path(&out).exists(),
                "{what}: no download left beside it"
            );
        };

        // The catalog path, forced (the refresh `basert serve` runs).
        let r = reg
            .resolve("basecompute/cat", "main", Some("default-medium"), true)
            .unwrap();
        assert!(matches!(r, ModelRef::Catalog { .. }));
        let err = pull_catalog(&reg.root, &r, &fetcher).unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        untouched("catalog path");

        // The direct path, by file name, forced.
        let args = PullArgs {
            force: true,
            ..named("cat-Medium.base", false, "basecompute/cat")
        };
        let err = pull_named(&reg, &fetcher, &args).unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        untouched("direct path");
    }

    /// The catalog's own file is held to the catalog's sha256 when named.
    #[test]
    fn a_named_catalog_file_is_checked_against_the_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        fixture_repo(
            tmp.path(),
            "basecompute/cat-files",
            &[("cat-Medium.base", b"medium")],
        );
        let fetcher = MockFetcher::new(tmp.path());
        let row = |sha: &str| {
            catalog(&format!(
                r#"{{"id":"basecompute/cat","hf_repo":"basecompute/cat-files","quant":"default-medium","file":"cat-Medium.base","sha256":"{sha}"}}"#
            ))
        };
        let out = |reg: &MergedRegistry| reg.root.join("basecompute/cat/default-medium/model.base");

        let bad = MergedRegistry::new(tmp.path().join("bad"), row(&"0".repeat(64)));
        let err = pull_named(
            &bad,
            &fetcher,
            &named("cat-Medium.base", false, "basecompute/cat"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert!(
            !out(&bad).exists(),
            "a mismatched download is not installed"
        );

        let good = MergedRegistry::new(
            tmp.path().join("good"),
            row("c082456a7766e23a18db084cd34b6ff510baef506548b897cc80e9b7d3e121c8"),
        );
        pull_named(
            &good,
            &fetcher,
            &named("cat-Medium.base", false, "basecompute/cat"),
        )
        .unwrap();
        assert_eq!(std::fs::read(out(&good)).unwrap(), b"medium");
    }

    /// A Hub that must not be asked anything.
    struct Offline;

    impl Fetcher for Offline {
        fn get_file(&self, _: &str, _: &str, f: &str) -> Result<PathBuf> {
            bail!("no download expected ({f})")
        }
        fn list_files(&self, repo: &str, _: &str) -> Result<Vec<String>> {
            bail!("no listing expected ({repo})")
        }
    }

    fn with_variant(id: &str, dry_run: bool) -> PullArgs {
        PullArgs {
            id: id.into(),
            profile: None,
            target: TargetScheme::BaseQ4,
            revision: "main".into(),
            file: None,
            force: false,
            dry_run,
        }
    }

    fn pull_with_variant(
        reg: &MergedRegistry,
        fetcher: &dyn Fetcher,
        args: &PullArgs,
    ) -> Result<()> {
        let (id, variant) = args.id.split_once(':').unwrap();
        pull_variant(reg, args, id, variant, fetcher)
    }

    /// `basert pull <id>:<variant>` takes the variant as `basert list` names
    /// it: an installed one is left as it is without asking the Hub, a
    /// catalog one is its row's file (refused for another backend), a
    /// `file-…` one is the file it was named for, a quant is a `--target`,
    /// and anything else is an error naming the catalog's variants.
    #[test]
    fn a_pull_names_a_variant_as_basert_list_does() {
        let here = CatalogRegistry::client_backend();
        let there = if here == "cuda" { "metal" } else { "cuda" };
        let tmp = tempfile::tempdir().unwrap();
        let theirs = format!("cat-{there}-q4mix.base");
        fixture_repo(
            tmp.path(),
            "basecompute/cat",
            &[
                ("cat-Q4.base", b"q4"),
                ("cat-Q8.base", b"q8"),
                ("cat-F16.base", b"f16"),
                (&theirs, b"theirs"),
                ("cat-Medium.base", b"medium"),
                ("Extra-Q4.base", b"extra"),
            ],
        );
        let row = |quant: &str, file: &str, backend: &str| {
            format!(
                r#"{{"id":"basecompute/cat","hf_repo":"basecompute/cat","quant":"{quant}","file":"{file}"{backend}}}"#
            )
        };
        let reg = MergedRegistry::new(
            tmp.path().join("cache"),
            catalog(
                &[
                    row("default-q4", "cat-Q4.base", ""),
                    row("default-q8", "cat-Q8.base", ""),
                    row("default-f16", "cat-F16.base", ""),
                    row(
                        &format!("{there}-q4mix"),
                        &theirs,
                        &format!(r#","backend":"{there}""#),
                    ),
                    row(
                        &format!("{there}-q6"),
                        "cat-Q6.base",
                        &format!(r#","backend":"{there}""#),
                    ),
                ]
                .join(","),
            ),
        );
        let fetcher = MockFetcher::new(tmp.path());
        let cat = reg.root.join("basecompute/cat");
        let q8 = cat.join("default-q8/model.base");

        // a catalog variant: its row's file, under its name
        let args = with_variant("basecompute/cat:default-q8", false);
        pull_with_variant(&reg, &fetcher, &args).unwrap();
        assert_eq!(std::fs::read(&q8).unwrap(), b"q8");
        assert!(
            !cat.join("default-q4").exists(),
            "not the catalog's default"
        );

        // installed: left as it is, the Hub unasked, unless --force
        std::fs::write(&q8, b"cached").unwrap();
        for dry in [false, true] {
            let args = with_variant("basecompute/cat:default-q8", dry);
            pull_with_variant(&reg, &Offline, &args).unwrap();
        }
        assert_eq!(std::fs::read(&q8).unwrap(), b"cached");
        let mut force = with_variant("basecompute/cat:default-q8", false);
        force.force = true;
        pull_with_variant(&reg, &fetcher, &force).unwrap();
        assert_eq!(std::fs::read(&q8).unwrap(), b"q8");

        // another backend's variant is refused, and nothing is downloaded
        let args = with_variant(&format!("basecompute/cat:{there}-q4mix"), false);
        let err = pull_with_variant(&reg, &fetcher, &args).unwrap_err();
        assert!(
            err.to_string().contains(&format!("the {there} bundle")),
            "{err}"
        );
        assert!(!cat.join(format!("{there}-q4mix")).exists());

        // a `file-…` variant: the file it was named for
        let medium = named_variant("cat-Medium.base");
        let args = with_variant(&format!("basecompute/cat:{medium}"), false);
        pull_with_variant(&reg, &fetcher, &args).unwrap();
        assert_eq!(
            std::fs::read(cat.join(&medium).join("model.base")).unwrap(),
            b"medium"
        );
        let args = with_variant("basecompute/cat:file-cat-large-00000000", true);
        let err = pull_with_variant(&reg, &fetcher, &args).unwrap_err();
        assert!(err.to_string().contains("cat-Medium.base"), "{err}");
        // the catalog's own file is its row's variant, never a `file-…` one
        let args = with_variant(
            &format!("basecompute/cat:{}", named_variant("cat-Q4.base")),
            true,
        );
        let err = pull_with_variant(&reg, &fetcher, &args).unwrap_err();
        assert!(
            err.to_string().contains("pull basecompute/cat:default-q4"),
            "{err}"
        );

        // a `file-…` variant whose name spells a width is still its file
        let extra = named_variant("Extra-Q4.base");
        assert_eq!(
            variant_source(&reg, "basecompute/cat", &extra, "main", &fetcher).unwrap(),
            VariantSource::File("Extra-Q4.base".into())
        );

        // a width named on its own is the catalog's row of it for this
        // client, f16 included; one published only for another backend is
        // refused
        let row_of =
            |v: &str| match variant_source(&reg, "basecompute/cat", v, "main", &Offline).unwrap() {
                VariantSource::Catalog(ModelRef::Catalog { variant, file, .. }) => (variant, file),
                other => panic!("{v}: {other:?}"),
            };
        for (v, row) in [
            ("q4", "default-q4"),
            ("q8", "default-q8"),
            ("f16", "default-f16"),
        ] {
            let file = format!("cat-{}.base", v.to_ascii_uppercase());
            assert_eq!(row_of(v), (row.to_string(), file));
        }
        let err = variant_source(&reg, "basecompute/cat", "q6", "main", &Offline)
            .unwrap_err()
            .to_string();
        assert!(err.contains(&format!("the {here} backend")), "{err}");

        // a quant is a --target, as --target spells it
        for (v, t) in [
            ("q8", TargetScheme::BaseQ8),
            ("default-q6", TargetScheme::BaseQ6),
            ("base_q2", TargetScheme::BaseQ2),
            ("base-q3", TargetScheme::BaseQ3),
            ("bf16", TargetScheme::Bf16),
        ] {
            assert_eq!(
                variant_source(&reg, "org/model", v, "main", &Offline).unwrap(),
                VariantSource::Quant(t)
            );
        }
        // anything else names the variants this client can pull, a name
        // that merely contains or ends in a width included
        for v in [
            "large",
            "default-q8-custom",
            "not-a-variant-f16",
            "typo-q8",
            "q8-custom",
        ] {
            let err = variant_source(&reg, "basecompute/cat", v, "main", &Offline)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("the catalog lists default-q4, default-q8, default-f16")
                    && !err.contains(there),
                "{v}: {err}"
            );
            let err = variant_source(&reg, "org/model", v, "main", &Offline).unwrap_err();
            assert!(err.to_string().contains("has no variant"), "{v}: {err}");
        }

        // a flag that names the variant too is refused, as is an empty one
        let mut file = with_variant("basecompute/cat:default-q8", true);
        file.file = Some("cat-Q8.base".into());
        let mut target = with_variant("basecompute/cat:default-q8", true);
        target.target = TargetScheme::BaseQ8;
        let mut profile = with_variant("basecompute/cat:default-q8", true);
        profile.profile = Some("default-q8.json".into());
        for (args, flag) in [
            (file, "--file"),
            (target, "--target"),
            (profile, "--profile"),
        ] {
            let err = pull_with_variant(&reg, &Offline, &args).unwrap_err();
            assert!(err.to_string().contains(flag), "{err}");
        }
        let args = with_variant("basecompute/cat:", true);
        let err = pull_with_variant(&reg, &Offline, &args).unwrap_err();
        assert!(err.to_string().contains("empty variant"), "{err}");
    }

    /// The row a variant names is the one pulled, from its own repo, even
    /// when another row of the id publishes a file of the same path: by its
    /// name, by its width, and by a re-pull.
    #[test]
    fn a_variant_pulls_its_row_when_rows_share_a_file_name() {
        let tmp = tempfile::tempdir().unwrap();
        fixture_repo(tmp.path(), "basecompute/cat-a", &[("model.base", b"a")]);
        fixture_repo(tmp.path(), "basecompute/cat-b", &[("model.base", b"b")]);
        let row = |repo: &str, quant: &str| {
            format!(
                r#"{{"id":"basecompute/cat","hf_repo":"basecompute/{repo}","quant":"{quant}","file":"model.base"}}"#
            )
        };
        let reg = MergedRegistry::new(
            tmp.path().join("cache"),
            catalog(&[row("cat-a", "default-q4"), row("cat-b", "default-q8")].join(",")),
        );
        let fetcher = MockFetcher::new(tmp.path());
        let installed = |variant: &str| {
            std::fs::read(
                reg.root
                    .join("basecompute/cat")
                    .join(variant)
                    .join("model.base"),
            )
            .unwrap()
        };

        let args = with_variant("basecompute/cat:default-q8", false);
        pull_with_variant(&reg, &fetcher, &args).unwrap();
        assert_eq!(installed("default-q8"), b"b");
        let args = with_variant("basecompute/cat:q4", false);
        pull_with_variant(&reg, &fetcher, &args).unwrap();
        assert_eq!(installed("default-q4"), b"a");
        assert_eq!(
            held(&reg, "basecompute/cat", "default-q8").as_deref(),
            Some("model.base")
        );

        let mut force = with_variant("basecompute/cat:default-q8", false);
        force.force = true;
        std::fs::write(
            reg.root.join("basecompute/cat/default-q8/model.base"),
            b"stale",
        )
        .unwrap();
        pull_with_variant(&reg, &fetcher, &force).unwrap();
        assert_eq!(installed("default-q8"), b"b");
        for dry in [false, true] {
            let args = with_variant("basecompute/cat:q8", dry);
            pull_with_variant(&reg, &Offline, &args).unwrap();
        }
    }

    /// Fetcher owning an hf-hub-style staging tree (blob + snapshot symlink),
    /// mirroring what `HfFetcher` leaves on disk mid-pull.
    struct StagedFetcher {
        staging: PathBuf,
    }

    impl StagedFetcher {
        fn repo_dir(&self, repo: &str) -> PathBuf {
            self.staging
                .join(format!("models--{}", repo.replace('/', "--")))
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
            let p = self
                .repo_dir(repo)
                .join("snapshots")
                .join(revision)
                .join(filename);
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
            file: None,
            force: false,
            dry_run: false,
        };
        pull_base_direct(
            &root,
            &args,
            "basecompute/m",
            "basecompute/m",
            "main",
            &fetcher,
            &base_files,
            None,
            None,
        )
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

    /// Serves small files from `snapshots/main/` but large ones as bare
    /// `blobs/` paths with no snapshot pointer — the shape `HfFetcher`'s
    /// resumable range path returns.
    struct BlobFetcher {
        root: PathBuf,
    }

    impl Fetcher for BlobFetcher {
        fn get_file(&self, _repo: &str, _revision: &str, filename: &str) -> Result<PathBuf> {
            if filename.ends_with(".safetensors") {
                Ok(self.root.join("blobs").join(filename.replace('/', "_")))
            } else {
                Ok(self.root.join("snapshots").join("main").join(filename))
            }
        }

        fn list_files(&self, _repo: &str, _revision: &str) -> Result<Vec<String>> {
            Ok([
                "config.json",
                "tokenizer.json",
                "model.safetensors",
                "sub/model-2.safetensors",
            ]
            .map(String::from)
            .to_vec())
        }
    }

    #[test]
    fn download_source_links_blob_only_files_into_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = tmp.path().join("snapshots").join("main");
        let blobs = tmp.path().join("blobs");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(snap.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();
        std::fs::write(snap.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(blobs.join("model.safetensors"), b"w1").unwrap();
        std::fs::write(blobs.join("sub_model-2.safetensors"), b"w2").unwrap();
        // A dangling pointer left by an earlier, cleaned-up pull.
        std::os::unix::fs::symlink(tmp.path().join("gone"), snap.join("model.safetensors"))
            .unwrap();
        let fetcher = BlobFetcher {
            root: tmp.path().to_path_buf(),
        };

        // Twice: a re-pull over an already-linked snapshot is a no-op.
        for _ in 0..2 {
            let snapshot = download_source("org/model", "main", &fetcher).unwrap();
            assert_eq!(snapshot, snap);
            assert_eq!(
                std::fs::read(snap.join("model.safetensors")).unwrap(),
                b"w1"
            );
            assert_eq!(
                std::fs::read(snap.join("sub/model-2.safetensors")).unwrap(),
                b"w2"
            );
        }
    }

    // A symlink target is resolved relative to the LINK's directory, so a
    // relative `src` (which a relative BASERT_MODELS_DIR produces, since the
    // env var is used verbatim) must be absolutised before it is stored —
    // otherwise every shard pointer dangles and conversion dies on open.
    // The source path here is deliberately relative to the crate's CWD; no
    // chdir, so this stays safe under the parallel test runner.
    #[test]
    fn link_into_snapshot_absolutises_a_relative_source() {
        let rel_dir = Path::new("target").join("link-into-snapshot-relative-src");
        std::fs::create_dir_all(&rel_dir).unwrap();
        let rel_src = rel_dir.join("blob.safetensors");
        std::fs::write(&rel_src, b"w1").unwrap();
        assert!(rel_src.is_relative());

        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("snapshots").join("main");
        std::fs::create_dir_all(&snapshot).unwrap();

        link_into_snapshot(&snapshot, "model.safetensors", &rel_src).unwrap();

        let pointer = snapshot.join("model.safetensors");
        assert!(
            std::fs::read_link(&pointer).unwrap().is_absolute(),
            "symlink target must be absolute or it dangles from inside the snapshot"
        );
        // The real symptom: reading THROUGH the link from the snapshot.
        assert_eq!(std::fs::read(&pointer).unwrap(), b"w1");

        std::fs::remove_dir_all(&rel_dir).ok();
    }

    // The file list comes from the remote repo, so a traversing entry must be
    // refused rather than placing a symlink outside the snapshot.
    #[test]
    fn link_into_snapshot_rejects_traversing_filenames() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("snapshots").join("main");
        std::fs::create_dir_all(&snapshot).unwrap();
        let src = tmp.path().join("blob.safetensors");
        std::fs::write(&src, b"w").unwrap();

        for bad in [
            "../escaped.safetensors",
            "/etc/escaped.safetensors",
            "a/../../b.safetensors",
        ] {
            let err = link_into_snapshot(&snapshot, bad, &src).unwrap_err();
            assert!(
                err.to_string().contains("unsafe source path"),
                "{bad}: {err}"
            );
        }
        assert!(!tmp.path().join("escaped.safetensors").exists());
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
    fn speculate_drafter_id_splits_ids_but_not_paths() {
        assert_eq!(
            speculate_drafter_id("dflash:basecompute/Qwen3-4B-DFlash"),
            Some(("dflash", "basecompute/Qwen3-4B-DFlash"))
        );
        // An inline variant rides along: resolve_hub_model parses it.
        assert_eq!(
            speculate_drafter_id("eagle3:org/head:default-q4"),
            Some(("eagle3", "org/head:default-q4"))
        );
        // Paths keep working: the artifact may be a local build.
        assert_eq!(speculate_drafter_id("dflash:./drafters/x.base"), None);
        assert_eq!(speculate_drafter_id("dspark:/abs/x.base"), None);
        assert_eq!(speculate_drafter_id("dflash:x.base"), None);
        // Strategies without an artifact, and unknown prefixes.
        assert_eq!(speculate_drafter_id("prompt-lookup"), None);
        assert_eq!(speculate_drafter_id("mtp-head"), None);
        assert_eq!(speculate_drafter_id("nope:org/model"), None);
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
        let err = resolve_hub_model("basecompute/Gemma-4-E2B-it:", None, None)
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
            "8453".into(),
        ])
        .unwrap();
        assert_eq!(v.as_deref(), Some("default-q8"));
        assert_eq!(rest, vec!["--model", "org/m", "--port", "8453"]);

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
            file: None,
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

    #[test]
    fn mtp_speculation_is_asked_by_mtp_head_or_auto() {
        let asked = |v: &[&str]| {
            mtp_speculation_asked(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .map(str::to_string)
        };
        assert_eq!(
            asked(&["m.base", "--speculate", "mtp-head"]).as_deref(),
            Some("mtp-head")
        );
        assert_eq!(
            asked(&["--speculate=auto", "m.base"]).as_deref(),
            Some("auto")
        );
        assert_eq!(asked(&["m.base", "--speculate", "dflash:org/d"]), None);
        assert_eq!(asked(&["m.base", "--speculate", "off"]), None);
        assert_eq!(asked(&["m.base", "--speculate", "prompt-lookup"]), None);
        assert_eq!(asked(&["m.base", "--speculate"]), None);
        assert_eq!(asked(&["m.base"]), None);
    }

    /// A catalog file without the head its row now promises is pulled again
    /// for speculation; one with a head, one the catalog promises no head
    /// for, the catalog's own file, or an install from elsewhere are not.
    #[test]
    fn a_catalog_file_without_the_head_its_row_promises_is_stale() {
        use base_format::{SpeculatorBundle, FORMAT_VERSION, MAGIC};
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let id = "basecompute/gemma-4-E4B-it";
        let (old, new) = ("b".repeat(64), "a".repeat(64));
        // A header-only bundle (read_header never reaches the blob) and the
        // sidecar a pull records beside it.
        // `head`: None for no head, Some(version) for one (None = unversioned).
        let install = |variant: &str, head: Option<Option<&str>>, source_kind: &str| -> PathBuf {
            let vdir = cache::variant_dir(&root, id, variant).unwrap();
            std::fs::create_dir_all(&vdir).unwrap();
            let path = cache::base_artifact_path(&vdir);
            let mut h =
                crate::embed::fixtures::header("gemma4", serde_json::json!({"hidden_size": 8}));
            if let Some(version) = head {
                let mut config: std::collections::BTreeMap<String, serde_json::Value> =
                    [("kind".to_string(), serde_json::json!("gemma4-mtp"))]
                        .into_iter()
                        .collect();
                if let Some(v) = version {
                    config.insert("head_version".to_string(), serde_json::json!(v));
                }
                h.speculator = Some(SpeculatorBundle {
                    arch: "gemma4-mtp".into(),
                    config,
                    tensors: vec![],
                });
            }
            let json = h.to_canonical_json().unwrap();
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&MAGIC).unwrap();
            f.write_all(&FORMAT_VERSION.to_le_bytes()).unwrap();
            f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
            f.write_all(&json).unwrap();
            cache::write_sidecar(
                &vdir,
                &HubSidecar {
                    id: id.into(),
                    source_kind: source_kind.into(),
                    hf_repo: id.into(),
                    source_repo: None,
                    revision: "main".into(),
                    variant: variant.into(),
                    profile: None,
                    pulled_at: "0".into(),
                    base_sha256: Some(old.clone()),
                    file: Some(format!("{variant}.base")),
                },
            )
            .unwrap();
            path
        };
        let reg = |variant: &str, speculator: &str, sha: &str| {
            MergedRegistry::new(
                root.clone(),
                catalog(&format!(
                    r#"{{"id":"{id}","hf_repo":"{id}","file":"{variant}.base","quant":"{variant}","sha256":"{sha}"{speculator}}}"#
                )),
            )
        };
        let promised = r#","speculator":"gemma4-mtp""#;

        let headless = install("default-q8", None, "catalog");
        assert_eq!(
            stale_head_variant(&reg("default-q8", promised, &new), id, &headless).as_deref(),
            Some("default-q8")
        );
        assert_eq!(
            stale_head_variant(
                &reg("default-q8", r#","speculator":"mtp""#, &new),
                id,
                &headless
            )
            .as_deref(),
            Some("default-q8")
        );
        assert_eq!(
            stale_head_variant(&reg("default-q8", "", &new), id, &headless),
            None,
            "the catalog promises no head"
        );
        assert_eq!(
            stale_head_variant(&reg("default-q8", promised, &old), id, &headless),
            None,
            "the catalog's file is this one"
        );
        assert_eq!(
            stale_head_variant(&reg("default-q4", promised, &new), id, &headless),
            None,
            "another variant's row"
        );

        let with_head = install("default-q4", Some(None), "catalog");
        assert_eq!(
            stale_head_variant(&reg("default-q4", promised, &new), id, &with_head),
            None,
            "it has its head"
        );

        // A versioned head: stale only when the catalog names another version.
        let v1 = install("cuda-q8", Some(Some("1")), "catalog");
        let versioned =
            |v: &str| format!(r#","speculator":"gemma4-mtp","speculator_version":"{v}""#);
        assert_eq!(
            stale_head_variant(&reg("cuda-q8", &versioned("1"), &new), id, &v1),
            None,
            "the catalog's head is this version"
        );
        assert_eq!(
            stale_head_variant(&reg("cuda-q8", &versioned("2"), &new), id, &v1).as_deref(),
            Some("cuda-q8"),
            "the catalog's head is a newer version"
        );
        assert_eq!(
            stale_head_variant(&reg("cuda-q8", promised, &new), id, &v1),
            None,
            "the catalog names no version to hold it to"
        );
        assert_eq!(
            stale_head_variant(&reg("default-q4", &versioned("1"), &new), id, &with_head)
                .as_deref(),
            Some("default-q4"),
            "an unversioned head against a versioned row"
        );

        let converted = install("default-f16", None, "huggingface");
        assert_eq!(
            stale_head_variant(&reg("default-f16", promised, &new), id, &converted),
            None,
            "not the catalog's install"
        );

        // Without a speculation request the file is served as it is.
        let r = reg("default-q8", promised, &new);
        assert_eq!(
            serve_ready(&r, id, headless.clone(), None).unwrap(),
            headless
        );
    }
}
