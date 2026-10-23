//! `basert serve`: the command line basert-serve has always had (the C++
//! server's, flag for flag), run on superfluid. The launcher reads the
//! command line, finds the engine library and a superfluid beside itself
//! (installing the newest release if asked, see [`crate::superfluid`]), and
//! execs `superfluid serve` with baseRT's models only, nothing fetched, and a
//! request log that starts fresh with each start: the server basert-serve has
//! always been.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::superfluid;

pub const USAGE: &str = "\
Usage: basert serve <model> [--model <model2> ...] [options]

  <model>  a .base file path, or a hub id (e.g. basecompute/Qwen3-4B) that
           `basert serve` pulls/converts on first use. The first non-flag
           argument is treated as a model (same as `basert chat`).

  The server is superfluid (https://github.com/basecompute/superfluid). basert
  runs the one installed beside it or in its own data directory, installing
  the newest release that serves this engine when there is none, or when the
  one found cannot serve it (the release basert was tested with when GitHub
  cannot be read): asked first on a terminal, without asking otherwise, never
  with BASERT_SUPERFLUID_AUTO_INSTALL=0.
  `basert superfluid install` does it ahead of time, and
  `basert superfluid status` shows what is installed and whether it can serve.

Options:
  --model <model>       Additional .base model or hub id. Repeatable (multi-model serving).
  --verbose, -v         Show engine diagnostics (load, GPU, dispatch table)
  --host <addr>         Bind address (default: 127.0.0.1)
  --port <N>            Bind port (default: 8453)
  --max-context <N>     Max context window in tokens. Default: auto-sized for this
                        machine — the GPU memory budget less the weights and
                        scratch headroom, over what a token of KV costs across the
                        configured lanes, capped only at the model's trained window.
                        Determines KV cache allocation up front; requests past it
                        are rejected with context_length_exceeded.
  --max-tokens <N>      Cap on generated tokens for a request that names none.
                        Default: fill the rest of the context window.
  --rate-limit <N>      Requests per minute per client (0 = unlimited)
  --api-key <key>       Require Bearer token authentication
  --sessions <path>     Request-log directory (default: per-port under the state directory).
                        The request log is rotated at startup; Local owns chat history.
  --idle-timeout <N>    Auto-unload idle models after N seconds (0 = disabled)
  --model-dir <path>    Autoload: register every *.base in <path> as a known-but-
                        unloaded model, loaded into VRAM on first request (and freed
                        again by --idle-timeout). Pairs well with --idle-timeout.
  --request-timeout <ms> Accepted; it bounds FIM completions, not chat completions,
                        which this server runs to their token limit.
  --drain-timeout <N>   Seconds to wait for in-flight requests on SIGUSR1 (default: 60)
                        SIGUSR1 stops accepting connections and drains; pair with an
                        external supervisor for rolling restarts. SIGINT/SIGTERM still
                        exit immediately without draining.
  --files-max-bytes <N> Reject uploads that would push total stored bytes above N
  --files-expiry <N>    Auto-remove files older than N seconds (background sweep)
  --files-sweep <N>     Sweep cadence in seconds (default 300, when --files-expiry set)
  --log-file <path>     Redirect stderr (access log + diagnostics) to this file.
                        Use external rotation: `logrotate copytruncate` (Linux) or
                        `newsyslog` with the F flag (macOS). Default: stderr.
  --kv-bits 4|8|16|84   KV cache element width (84 = mixed K=Q8/V=Q4) (default: per-model auto)
  --max-batch-size N    Max concurrent sequences (default 1)
  --max-batch N         Alias for --max-batch-size (used by Local)
  --continuous-batching [N]  Serve up to N requests at once through one shared
                        forward pass (default 8). Set N >= your peak concurrent
                        requests, or the surplus queues (higher TTFT, no extra
                        throughput).
  --prefill-budget N    Max prompt tokens ingested per engine tick (64-4096). Lower
                        it to trade first-token latency for smoother concurrent decode.
  --basert-lib <dir>    The directory holding libbaseRT to serve with (default: beside
                        basert, or ../lib, or $BASERT_LIB)
  --print-command       Print the superfluid command line this start would run, and exit
  --version             Print the version and exit

Any other --flag is handed to `superfluid serve` as it is, with the word after
it as its value unless that word is a model (`superfluid serve --help` lists
them); everything after `--` is handed over untouched.

Accepted so that existing command lines keep working, and no longer used:
  --paged-kv, --prefix-cache   Paged KV and prefix reuse are always on.
  --prefix-cache-file <path>, --prefix-cache-save-interval <sec>
                        The prefix cache is not kept on disk.
  --files-dir <path>    Uploaded files are kept in the server's state directory.
  --media-dir <path>    Images and audio arrive as data: URIs or uploads; no local
                        path is read.
  --metallib <path>, --prefill-chunk N, --paged-weights, --no-paged-weights-retry,
  --gpu-wait-timeout-ms N, --decode-replay 0|1|2, --no-baked-decode
                        Engine settings this server does not expose.

State (the request log, uploaded files) is kept under $BASERT_STATE_DIR, by
default the serve/ directory beside the model cache, one directory per port.
The engine library is found beside basert (or in ../lib); BASERT_LIB names
another directory. BASERT_SUPERFLUID_DIR names where superfluid is installed;
BASERT_SUPERFLUID_RELEASES another release listing (a URL or a file);
BASERT_SUPERFLUID_SOURCE a directory holding the tested release's tarball,
read instead of GitHub; BASERT_SUPERFLUID_AUTO_INSTALL=0 forbids a download
at a start.
";

/// Flags every start hands superfluid; one given on the command line too is
/// not handed over twice.
const ALWAYS: [&str; 5] = [
    "--offline",
    "--no-park",
    "--sessions-cache",
    "--no-tui",
    "--no-http-qos-header",
];

/// How deep `--model-dir` is looked into for a model to start with; a hub
/// cache nests one three levels down (`<org>/<repo>/<variant>/model.base`).
const MODEL_DIR_DEPTH: usize = 8;

/// What a command line asks for.
#[derive(Debug, Default, PartialEq)]
pub struct Options {
    pub models: Vec<String>,
    pub model_dir: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    /// `None` sizes the window for this device.
    pub max_context: Option<u32>,
    /// `None` fills the rest of the window.
    pub max_tokens: Option<u32>,
    pub rate_limit: u32,
    pub api_key: Option<String>,
    pub sessions: Option<String>,
    pub idle_timeout: u64,
    pub request_timeout_ms: u64,
    pub drain_timeout: Option<u64>,
    pub files_max_bytes: Option<u64>,
    pub files_expiry: Option<u64>,
    pub files_sweep: Option<u64>,
    pub log_file: Option<String>,
    pub kv_bits: Option<u32>,
    pub lanes: u32,
    pub prefill_budget: Option<u32>,
    pub verbose: bool,
    /// `--basert-lib`: the engine library directory, superfluid's own flag,
    /// in place of the one found beside basert.
    pub basert_lib: Option<String>,
    /// superfluid's own flags, handed over after the translated ones.
    pub passthrough: Vec<String>,
    /// Print the superfluid command line and exit.
    pub print_command: bool,
    /// Said on stderr before serving: flags that are no longer used, and
    /// values read as something else.
    pub notes: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum Parsed {
    Serve(Box<Options>),
    Help,
    Version,
}

fn number<T: std::str::FromStr>(flag: &str, raw: &str) -> Result<T, String> {
    raw.trim()
        .parse()
        .map_err(|_| format!("Error: {flag} expects a number, got {raw:?}"))
}

/// Reads a command line (after `serve`) the way the C++ server did.
pub fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut o = Options::default();
    // The C++ server's two batch settings: `--max-batch-size` set the lanes
    // the load installs, `--continuous-batching [N]` (default 8) set them
    // too, and the larger of the two won.
    let mut max_batch_size: u32 = 1;
    let mut continuous: Option<u32> = None;
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        let flag = a.as_str();
        // A flag that takes a value and has none was an unknown option.
        let value =
            |it: &mut std::iter::Peekable<std::slice::Iter<String>>| -> Result<String, String> {
                it.next()
                    .cloned()
                    .ok_or_else(|| format!("Unknown option: {flag} (see `basert serve --help`)"))
            };
        match flag {
            "--help" | "-h" => return Ok(Parsed::Help),
            "--version" | "-V" => return Ok(Parsed::Version),
            "--model" => o.models.push(value(&mut it)?),
            "--model-dir" => o.model_dir = Some(value(&mut it)?),
            "--host" => o.host = Some(value(&mut it)?),
            "--port" => o.port = Some(number(flag, &value(&mut it)?)?),
            "--max-context" => {
                let v = value(&mut it)?;
                let n: i64 = if v == "auto" { 0 } else { number(flag, &v)? };
                if n > 0 {
                    o.max_context = Some(n.min(u32::MAX as i64) as u32);
                } else {
                    if v != "auto" {
                        o.notes.push(format!("WARNING: --max-context {v} is not a positive token count; auto-sizing instead"));
                    }
                    o.max_context = None;
                }
            }
            "--max-tokens" => {
                let n: i64 = number(flag, &value(&mut it)?)?;
                o.max_tokens = (n > 0).then(|| n.min(u32::MAX as i64) as u32);
            }
            "--rate-limit" => {
                o.rate_limit =
                    number::<i64>(flag, &value(&mut it)?)?.clamp(0, u32::MAX as i64) as u32
            }
            "--api-key" => o.api_key = Some(value(&mut it)?),
            "--sessions" => o.sessions = Some(value(&mut it)?),
            "--idle-timeout" => {
                o.idle_timeout = number::<i64>(flag, &value(&mut it)?)?.max(0) as u64
            }
            "--request-timeout" => {
                o.request_timeout_ms = number::<i64>(flag, &value(&mut it)?)?.max(0) as u64;
                // superfluid's completion deadline bounds its FIM path; a
                // chat completion has no deadline, and the C++ server's did.
                if o.request_timeout_ms > 0 {
                    o.notes.push(format!(
                        "basert serve: {flag} bounds FIM completions only: a chat completion runs to its token limit"
                    ));
                }
            }
            "--drain-timeout" => {
                o.drain_timeout = Some(number::<i64>(flag, &value(&mut it)?)?.max(0) as u64)
            }
            "--files-max-bytes" => {
                o.files_max_bytes = Some(number::<i64>(flag, &value(&mut it)?)?.max(0) as u64)
            }
            "--files-expiry" => {
                o.files_expiry = Some(number::<i64>(flag, &value(&mut it)?)?.max(0) as u64)
            }
            "--files-sweep" => {
                o.files_sweep = Some(number::<i64>(flag, &value(&mut it)?)?.max(0) as u64)
            }
            "--log-file" => o.log_file = Some(value(&mut it)?),
            "--kv-bits" => {
                let bits: i64 = number(flag, &value(&mut it)?)?;
                if ![4, 8, 16, 84].contains(&bits) {
                    return Err(format!("Error: --kv-bits must be 4, 8, 16, or 84 (84 = mixed K=Q8/V=Q4) (got {bits})"));
                }
                o.kv_bits = Some(bits as u32);
            }
            "--max-batch-size" | "--max-batch" => {
                let b: i64 = number(flag, &value(&mut it)?)?;
                if b < 1 {
                    return Err(format!("Error: {flag} must be >= 1 (got {b})"));
                }
                max_batch_size = b.min(u32::MAX as i64) as u32;
            }
            "--continuous-batching" => {
                // The width is optional: taken when the next word starts with a digit.
                let mut lanes = continuous.unwrap_or(8);
                if let Some(next) = it
                    .peek()
                    .filter(|n| n.starts_with(|c: char| c.is_ascii_digit()))
                {
                    let b: i64 = number(flag, next)?;
                    it.next();
                    if b >= 1 {
                        lanes = b.min(u32::MAX as i64) as u32;
                    }
                }
                continuous = Some(lanes);
                max_batch_size = lanes;
            }
            "--prefill-budget" => {
                let v: i64 = number(flag, &value(&mut it)?)?;
                if v > 0 {
                    let kept = v.clamp(64, 4096) as u32;
                    if kept as i64 != v {
                        o.notes.push(format!(
                            "basert serve: --prefill-budget {v} is outside 64-4096; using {kept}"
                        ));
                    }
                    o.prefill_budget = Some(kept);
                }
            }
            "--verbose" | "-v" => o.verbose = true,
            "--paged-kv" | "--prefix-cache" => o.notes.push(format!(
                "basert serve: {flag} is no longer used: paged KV and prefix reuse are always on"
            )),
            "--prefix-cache-file" | "--prefix-cache-save-interval" => {
                value(&mut it)?;
                o.notes.push(format!(
                    "basert serve: {flag} is no longer used: the prefix cache is not kept on disk"
                ));
            }
            "--files-dir" => {
                value(&mut it)?;
                o.notes.push(format!("basert serve: {flag} is no longer used: uploaded files are kept in the server's state directory"));
            }
            "--media-dir" => {
                value(&mut it)?;
                o.notes.push(format!(
                    "basert serve: {flag} is no longer used: images and audio arrive as data: URIs or uploads, and no local path is read"
                ));
            }
            "--metallib" | "--prefill-chunk" | "--gpu-wait-timeout-ms" | "--decode-replay" => {
                value(&mut it)?;
                o.notes.push(format!("basert serve: {flag} is no longer used: an engine setting this server does not expose"));
            }
            "--paged-weights" | "--no-paged-weights-retry" | "--no-baked-decode" => {
                o.notes.push(format!("basert serve: {flag} is no longer used: an engine setting this server does not expose"));
            }
            "--print-command" => o.print_command = true,
            "--basert-lib" => o.basert_lib = Some(value(&mut it)?),
            "--runtime" => {
                return Err("Error: --runtime is not accepted: basert serve always serves through superfluid's basert runtime".into())
            }
            "--" => o.passthrough.extend(it.by_ref().cloned()),
            other if !other.starts_with('-') => o.models.push(other.to_string()),
            other if ALWAYS.contains(&other) => {}
            other => {
                // superfluid's own flag: handed over as it is, with the word
                // after it as its value unless that word is a model.
                o.passthrough.push(other.to_string());
                if let Some(next) = it
                    .peek()
                    .map(|n| n.as_str())
                    .filter(|n| !n.starts_with('-') && !looks_like_model(n))
                {
                    o.passthrough.push(next.to_string());
                    it.next();
                }
            }
        }
    }
    if o.models.is_empty() && o.model_dir.is_none() {
        return Err("Error: at least one --model (or --model-dir) is required".into());
    }
    o.lanes = max_batch_size.max(continuous.unwrap_or(1));
    Ok(Parsed::Serve(Box::new(o)))
}

/// The `superfluid serve` command line for `o`, after `serve`. `sessions` is
/// this server's state directory.
pub fn superfluid_args(o: &Options, sessions: &Path) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let mut push = |k: &str, v: String| {
        a.push(k.to_string());
        a.push(v);
    };
    for m in &o.models {
        push("--model", m.clone());
    }
    if let Some(d) = &o.model_dir {
        push("--model-dir", d.clone());
    }
    // baseRT's models only, never fetched, and a request log that starts
    // fresh with each start: the server basert-serve has always been.
    push("--runtime", "basert".into());
    push(
        "--sessions",
        o.sessions
            .clone()
            .unwrap_or_else(|| sessions.display().to_string()),
    );
    push(
        "--host",
        o.host.clone().unwrap_or_else(|| "127.0.0.1".into()),
    );
    push("--port", o.port.unwrap_or(8453).to_string());
    push("--max-batch", o.lanes.to_string());
    if let Some(n) = o.max_context {
        push("--max-context", n.to_string());
    }
    if let Some(n) = o.max_tokens {
        push("--max-tokens", n.to_string());
    }
    if o.rate_limit > 0 {
        push("--rate-limit", o.rate_limit.to_string());
    }
    if let Some(k) = &o.api_key {
        push("--api-key", k.clone());
    }
    if o.idle_timeout > 0 {
        push("--idle-timeout", o.idle_timeout.to_string());
    }
    if o.request_timeout_ms > 0 {
        push("--completion-deadline-ms", o.request_timeout_ms.to_string());
    }
    if let Some(n) = o.drain_timeout {
        push("--drain-timeout", n.to_string());
    }
    if let Some(n) = o.files_max_bytes {
        push("--files-max-bytes", n.to_string());
    }
    if let Some(n) = o.files_expiry {
        push("--files-expiry", n.to_string());
    }
    if let Some(n) = o.files_sweep {
        push("--files-sweep", n.to_string());
    }
    if let Some(p) = &o.log_file {
        push("--log-file", p.clone());
    }
    if let Some(b) = o.kv_bits {
        push("--kv-bits", b.to_string());
    }
    if let Some(b) = o.prefill_budget {
        push("--prefill-budget", b.to_string());
    }
    if let Some(dir) = &o.basert_lib {
        push("--basert-lib", dir.clone());
    }
    a.extend(ALWAYS.map(String::from));
    a.extend(o.passthrough.iter().cloned());
    a
}

/// A word on the command line that is a model, not a flag's value: a `.base`
/// path, or a hub id (`org/name`, with a variant after `:` or not).
fn looks_like_model(word: &str) -> bool {
    word.ends_with(".base")
        || (!word.starts_with(['/', '.', '~']) && word.matches('/').count() == 1)
}

/// `basert serve <args>`, the args' hub ids already resolved to paths. Returns
/// only when superfluid could not be started: on success the process is
/// replaced (`exec`).
pub fn run(rest: &[String]) -> Result<()> {
    let parsed = parse(rest);
    if parsed == Ok(Parsed::Version) {
        println!("basert serve {}", env!("CARGO_PKG_VERSION"));
        match superfluid::Install::locate()? {
            Some(i) => println!(
                "superfluid {} ({})",
                i.installed_release()
                    .unwrap_or_else(|| "of a release it does not say".to_string()),
                i.binary().display()
            ),
            None => println!(
                "superfluid not installed (`basert superfluid install` installs the newest release; tested with {})",
                superfluid::RELEASE
            ),
        }
        return Ok(());
    }
    crate::print_banner();
    if rest.is_empty() {
        print!("{USAGE}");
        std::process::exit(1);
    }
    let mut o = match parsed {
        Ok(Parsed::Serve(o)) => *o,
        Ok(_) => {
            print!("{USAGE}");
            return Ok(());
        }
        Err(e) => fail(&e),
    };
    for note in &o.notes {
        eprintln!("{note}");
    }
    for m in &o.models {
        if !Path::new(m).exists() {
            fail(&format!(
                "Failed to load model {m}: no such file (a .base path, or a hub id that `basert serve` pulls on first use)"
            ));
        }
    }
    if o.models.is_empty() {
        // superfluid starts with a model loaded; the C++ server could start
        // with none and load the first on first request.
        let dir = o.model_dir.clone().unwrap_or_default();
        let Some(first) = first_model_under(Path::new(&dir), MODEL_DIR_DEPTH) else {
            fail(&format!("Error: --model-dir {dir} holds no .base model"));
        };
        eprintln!(
            "basert serve: --model-dir without --model: {} is loaded at startup, the others on first use",
            first.display()
        );
        o.models.push(first.display().to_string());
    }
    let state = state_dir()?;
    let sessions = o
        .sessions
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| state.join(format!("basert-serve-{}", o.port.unwrap_or(8453))));
    std::fs::create_dir_all(&sessions).with_context(|| {
        format!(
            "Error: could not create the state directory {}",
            sessions.display()
        )
    })?;
    // The engine library: the user's `--basert-lib`, else the one found
    // beside basert, which superfluid is told of the same way.
    let engine = match &o.basert_lib {
        Some(dir) => Some(PathBuf::from(dir)),
        None => engine_dir()?,
    };
    // superfluid keeps what it installs and its session socket under
    // SUPERFLUID_HOME: here, beside this server's state, never in
    // ~/.superfluid, so a superfluid of the user's own is never touched.
    let home = state.join("superfluid");
    let mut tail: Vec<String> = Vec::new();
    if let (Some(dir), None) = (&engine, &o.basert_lib) {
        tail.push("--basert-lib".to_string());
        tail.push(dir.display().to_string());
    }
    if o.print_command {
        let binary = match superfluid::Install::locate()? {
            Some(i) => i.binary().display().to_string(),
            None => "superfluid (not installed; the newest release would be)".to_string(),
        };
        let mut words = vec![binary, "serve".to_string()];
        words.extend(superfluid_args(&o, &sessions));
        words.extend(tail);
        println!(
            "SUPERFLUID_HOME={} SUPERFLUID_NO_UPDATE_CHECK=1{} {}",
            shell_word(&home.display().to_string()),
            if o.verbose {
                " SUPERFLUID_ENGINE_VERBOSE=1"
            } else {
                ""
            },
            words
                .iter()
                .map(|w| shell_word(w))
                .collect::<Vec<_>>()
                .join(" ")
        );
        return Ok(());
    }
    let binary = superfluid::ensure(engine.as_deref(), &home, &mut |line| {
        eprintln!("basert serve: {line}")
    })?;
    std::fs::create_dir_all(&home)
        .with_context(|| format!("Error: could not create {}", home.display()))?;

    let mut command = Command::new(&binary);
    command
        .arg("serve")
        .args(superfluid_args(&o, &sessions))
        .args(&tail);
    // A superfluid of basert's is updated by basert, never by itself: no
    // daily check, no notice in this server's output.
    command
        .env("SUPERFLUID_HOME", &home)
        .env("SUPERFLUID_NO_UPDATE_CHECK", "1");
    if o.verbose {
        command.env("SUPERFLUID_ENGINE_VERBOSE", "1");
    }
    use std::os::unix::process::CommandExt;
    let err = command.exec();
    Err(err).with_context(|| format!("launching {}", binary.display()))
}

/// `word` as a shell would need it: quoted when it holds a space or a quote.
fn shell_word(word: &str) -> String {
    if word.is_empty() || word.contains(|c: char| c.is_whitespace() || c == '\'' || c == '"') {
        format!("'{}'", word.replace('\'', "'\\''"))
    } else {
        word.to_string()
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// `$BASERT_STATE_DIR`, else `serve/` beside the model cache, as `basert
/// serve` has always had it.
pub(crate) fn superfluid_home() -> Result<PathBuf> {
    Ok(state_dir()?.join("superfluid"))
}

fn state_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("BASERT_STATE_DIR").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let models = base_hub::models_dir()
        .context("no cache directory to keep the server's state in (set BASERT_STATE_DIR)")?;
    Ok(models
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(models)
        .join("serve"))
}

/// Where libbaseRT is, for superfluid's basert worker: `None` when `$BASERT_LIB`
/// names it (superfluid reads that itself), else beside this launcher (the
/// release bundle), in `../lib` (the SDK bundle), or in `build/` of the source
/// tree this launcher was built in.
pub(crate) fn engine_dir() -> Result<Option<PathBuf>> {
    if std::env::var_os("BASERT_LIB").is_some_and(|v| !v.is_empty()) {
        return Ok(None);
    }
    let exe = std::env::current_exe().context("this program's own path")?;
    let dir = exe.parent().context("this program's own directory")?;
    let mut candidates = vec![dir.to_path_buf(), dir.join("..").join("lib")];
    candidates.extend(
        dir.ancestors()
            .filter(|a| a.join("tools/base-convert/Cargo.toml").is_file())
            .map(|a| a.join("build")),
    );
    for candidate in candidates {
        if ["libbaseRT.dylib", "libbaseRT.so"]
            .iter()
            .any(|name| candidate.join(name).exists())
        {
            return Ok(Some(std::fs::canonicalize(&candidate).unwrap_or(candidate)));
        }
    }
    bail!(
        "libbaseRT was not found beside {} or in its ../lib; set BASERT_LIB to the directory holding it",
        exe.display()
    )
}

/// The first `.base` under `dir` in name order, looking into directories.
fn first_model_under(dir: &Path, depth: usize) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    entries.into_iter().find_map(|p| {
        if p.is_file() && p.extension().is_some_and(|e| e == "base") {
            Some(p)
        } else if depth > 0 && p.is_dir() {
            first_model_under(&p, depth - 1)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    fn serve(args: &[&str]) -> Options {
        match parse(&line(args)) {
            Ok(Parsed::Serve(o)) => *o,
            other => panic!("{args:?}: {other:?}"),
        }
    }

    #[test]
    fn a_model_is_a_bare_word_or_a_model_flag_in_order() {
        assert_eq!(
            serve(&["a.base", "--model", "b.base", "c.base"]).models,
            ["a.base", "b.base", "c.base"]
        );
        assert_eq!(
            serve(&["--model-dir", "/m"]).model_dir.as_deref(),
            Some("/m")
        );
    }

    #[test]
    fn what_the_cpp_server_refused_is_refused_in_its_words() {
        for (args, said) in [
            (
                &[][..],
                "Error: at least one --model (or --model-dir) is required",
            ),
            (
                &["m.base", "--runtime", "llamacpp"],
                "Error: --runtime is not accepted: basert serve always serves through superfluid's basert runtime",
            ),
            (
                &["m.base", "--port"],
                "Unknown option: --port (see `basert serve --help`)",
            ),
            (
                &["m.base", "--kv-bits", "5"],
                "Error: --kv-bits must be 4, 8, 16, or 84 (84 = mixed K=Q8/V=Q4) (got 5)",
            ),
            (
                &["m.base", "--max-batch-size", "0"],
                "Error: --max-batch-size must be >= 1 (got 0)",
            ),
            (
                &["m.base", "--port", "nope"],
                "Error: --port expects a number, got \"nope\"",
            ),
        ] {
            assert_eq!(parse(&line(args)).unwrap_err(), said, "{args:?}");
        }
    }

    #[test]
    fn lanes_are_one_unless_batching_asks_for_more() {
        assert_eq!(serve(&["m.base"]).lanes, 1);
        assert_eq!(serve(&["m.base", "--continuous-batching"]).lanes, 8);
        assert_eq!(serve(&["m.base", "--continuous-batching", "4"]).lanes, 4);
        assert_eq!(serve(&["m.base", "--max-batch-size", "3"]).lanes, 3);
        // The larger of the two settings, whichever came last.
        assert_eq!(
            serve(&[
                "m.base",
                "--continuous-batching",
                "4",
                "--max-batch-size",
                "2"
            ])
            .lanes,
            4
        );
        assert_eq!(
            serve(&[
                "m.base",
                "--max-batch-size",
                "16",
                "--continuous-batching",
                "4"
            ])
            .lanes,
            4
        );
        // A model named after the flag is not taken for its width.
        let o = serve(&["--continuous-batching", "m.base"]);
        assert_eq!(
            (o.lanes, o.models.as_slice()),
            (8, &["m.base".to_string()][..])
        );
    }

    #[test]
    fn auto_and_unset_values_leave_the_default() {
        let o = serve(&[
            "m.base",
            "--max-context",
            "auto",
            "--max-tokens",
            "0",
            "--prefill-budget",
            "0",
        ]);
        assert_eq!(
            (o.max_context, o.max_tokens, o.prefill_budget),
            (None, None, None)
        );
        assert!(o.notes.is_empty(), "{:?}", o.notes);
        let o = serve(&["m.base", "--max-context", "-5"]);
        assert_eq!(o.max_context, None);
        assert_eq!(
            o.notes,
            ["WARNING: --max-context -5 is not a positive token count; auto-sizing instead"]
        );
    }

    #[test]
    fn local_launches_keep_the_session_path_and_batch_width() {
        for lanes in ["1", "2", "4"] {
            let o = serve(&[
                "/cache/model.base",
                "--port",
                "8453",
                "--host",
                "0.0.0.0",
                "--api-key",
                "local-test",
                "--sessions",
                "/app support/Local/engine",
                "--max-batch",
                lanes,
                "--max-context",
                "8192",
                "--kv-bits",
                "84",
            ]);
            assert_eq!(o.sessions.as_deref(), Some("/app support/Local/engine"));
            assert_eq!(o.lanes, lanes.parse::<u32>().unwrap());
            let args = superfluid_args(&o, Path::new("/fallback"));
            assert_eq!(args.iter().filter(|a| *a == "--sessions").count(), 1);
            assert!(args
                .windows(2)
                .any(|a| a == ["--sessions", "/app support/Local/engine"]));
            assert!(args.windows(2).any(|a| a == ["--max-batch", lanes]));
        }
        assert!(parse(&line(&["m.base", "--max-batch", "0"])).is_err());
        assert!(parse(&line(&["m.base", "--sessions"])).is_err());
        assert_eq!(
            serve(&["m.base", "--continuous-batching", "4", "--max-batch", "2"]).lanes,
            4
        );
    }

    #[test]
    fn retired_flags_are_taken_with_their_values_and_said_to_be_unused() {
        let o = serve(&[
            "m.base",
            "--paged-kv",
            "--prefix-cache",
            "--prefix-cache-file",
            "/c",
            "--media-dir",
            "/m",
            "--files-dir",
            "/f",
            "--metallib",
            "x.metallib",
            "--prefill-chunk",
            "256",
            "--decode-replay",
            "1",
            "--no-baked-decode",
        ]);
        assert_eq!(o.models, ["m.base"]);
        assert_eq!(o.notes.len(), 9, "{:?}", o.notes);
        assert!(
            o.notes.iter().all(|n| n.contains(" is no longer used: ")),
            "{:?}",
            o.notes
        );
    }

    #[test]
    fn the_superfluid_command_line_serves_basert_models_only_and_keeps_nothing() {
        let o = serve(&[
            "m.base",
            "--port",
            "9000",
            "--api-key",
            "k",
            "--request-timeout",
            "60000",
            "--kv-bits",
            "8",
        ]);
        let a = superfluid_args(&o, Path::new("/state/basert-serve-9000"));
        let joined = a.join(" ");
        for want in [
            "--model m.base",
            "--runtime basert",
            "--sessions /state/basert-serve-9000",
            "--host 127.0.0.1",
            "--port 9000",
            "--max-batch 1",
            "--api-key k",
            "--completion-deadline-ms 60000",
            "--kv-bits 8",
            "--offline",
            "--no-park",
            "--sessions-cache",
            "--no-tui",
        ] {
            assert!(joined.contains(want), "{want:?} in {joined}");
        }
        assert!(
            !joined.contains("--max-context"),
            "auto sizing is superfluid's default: {joined}"
        );
    }

    #[test]
    fn superfluids_own_flags_are_handed_over_as_they_are() {
        let o = serve(&[
            "m.base",
            "--prefix-blob-budget-mb",
            "512",
            "--speculate",
            "auto",
            "--no-tui",
            "--offline",
            "--basert-lib",
            "/engine",
            "--",
            "--anything",
            "goes",
            "--here",
        ]);
        assert_eq!(o.models, ["m.base"]);
        assert_eq!(o.basert_lib.as_deref(), Some("/engine"));
        assert_eq!(
            o.passthrough,
            [
                "--prefix-blob-budget-mb",
                "512",
                "--speculate",
                "auto",
                "--anything",
                "goes",
                "--here"
            ]
        );
        // A model after a flag that takes no value stays a model; a flag every
        // start hands over anyway is not handed over twice.
        let o = serve(&[
            "--no-http-qos-header",
            "--tui",
            "basecompute/Qwen3-4B:default-q8",
            "x.base",
        ]);
        assert_eq!(o.models, ["basecompute/Qwen3-4B:default-q8", "x.base"]);
        assert_eq!(o.passthrough, ["--tui"]);
        assert!(looks_like_model("/models/m.base") && looks_like_model("org/name"));
        assert!(
            !looks_like_model("/tmp/log") && !looks_like_model("auto") && !looks_like_model("512")
        );
        // Handed over last, after the flags every start sets; the user's
        // engine directory is superfluid's flag, said once.
        let o = serve(&["m.base", "--basert-lib", "/engine", "--speculate", "auto"]);
        let joined = superfluid_args(&o, Path::new("/s")).join(" ");
        assert!(
            joined.ends_with("--no-http-qos-header --speculate auto"),
            "{joined}"
        );
        assert_eq!(
            joined.matches("--basert-lib /engine").count(),
            1,
            "{joined}"
        );
        assert_eq!(shell_word("plain"), "plain");
        assert_eq!(shell_word("with space"), "'with space'");
        assert_eq!(shell_word("it's"), "'it'\\''s'");
    }

    #[test]
    fn the_first_model_under_a_directory_is_found_through_a_hub_cache_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let deep = tmp.path().join("org").join("repo").join("variant");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("model.base"), b"").unwrap();
        std::fs::write(tmp.path().join("zzz.txt"), b"").unwrap();
        assert_eq!(
            first_model_under(tmp.path(), MODEL_DIR_DEPTH),
            Some(deep.join("model.base"))
        );
        assert_eq!(first_model_under(tmp.path(), 0), None);
    }
}
