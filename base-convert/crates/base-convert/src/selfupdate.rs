//! `basert self update|check|list|use|install`: basert updating itself.
//!
//! The install script's layout: each release unpacked whole under
//! `<root>/versions/<v>/`, a `versions/current` link to the one in use, and in
//! `<root>` itself (the directory on PATH) a link per file of the release
//! through `versions/current`. An update unpacks the new release beside the
//! old, checks that its launcher runs and that a superfluid serves its engine,
//! then turns the one link: the launcher, the runtime tools, the engine
//! library and its kernel files change together, and a process already running
//! keeps the files it started with. The version replaced stays, for `basert
//! self use`; older ones are removed, unless a server still runs on one or it
//! holds a `keep` file.
//!
//! A copy the install script placed before this layout (the release's files
//! directly in `<root>`, beside a `.release` stamp) is moved under it by the
//! first update, as the version to go back to. A copy a package manager placed
//! (an `installed-by` file beside it says which), cargo's or a source build is
//! updated the way it was installed, and these commands say how.
//!
//! superfluid follows the engine. When the superfluid `basert serve` runs
//! cannot serve the new engine, the newest superfluid release that states the
//! engine's ABI number is downloaded and asked before anything changes; when
//! no release does, basert stays where it is. When it can, a newer superfluid
//! that also serves the engine is installed as well.
//!
//! A command run on a terminal looks for a newer release once a day, in a
//! process of its own so that nothing waits on it, and a later command prints
//! one line when there is one (a few days after it is published, so that
//! early reports land first). `BASERT_NO_UPDATE_CHECK=1` turns this off.

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde_json::{json, Value};

use crate::superfluid;

/// The releases the newest is chosen from: the open repo's, where the
/// install script downloads from.
const RELEASES: &str = "https://api.github.com/repos/basecompute/baseRT/releases?per_page=30";
/// Another release listing in GitHub's form, a URL or a file: a mirror's, or
/// the tests'. Its assets may be named by path.
pub const RELEASES_ENV: &str = "BASERT_RELEASES";
/// Set (and not `0`), no command looks for a newer release by itself.
pub const NO_CHECK_ENV: &str = "BASERT_NO_UPDATE_CHECK";
/// The file beside a launcher a package manager installed, one line
/// `<manager>: <command>`: the copy is that manager's to update, with that
/// command.
pub const INSTALLED_BY: &str = "installed-by";
/// The release a directory holds, as its tag: in the root, the one in use (the
/// install script reads it to skip a download); in a version, that version.
const STAMP: &str = ".release";
/// A version holding this file is never removed.
const KEEP: &str = "keep";
/// In a version, a file per `basert serve` running on its engine, named by pid.
const SERVING: &str = ".serving";
/// A release younger than this is not announced.
const HOLD_OFF: Duration = Duration::from_secs(3 * 24 * 60 * 60);
/// How often a command looks for a newer release, and how often it says so.
const CHECK_EVERY: Duration = Duration::from_secs(24 * 60 * 60);
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `basert self <action>`.
#[derive(Args, Debug)]
pub struct SelfArgs {
    #[command(subcommand)]
    pub action: Action,
}

#[derive(Subcommand, Debug)]
pub enum Action {
    /// Install the newest release for this machine and make it the one in use.
    ///
    /// The superfluid `basert serve` runs is brought to the newest release
    /// that serves the new engine; when none does yet, nothing changes.
    Update {
        /// Go to this release instead of the newest (an older one too), e.g. 0.4.3.
        #[arg(long = "version", value_name = "V")]
        version: Option<String>,
        /// Say what would be done, and change nothing.
        #[arg(long)]
        dry_run: bool,
        /// For a copy Homebrew, npm, pipx, uv or cargo installed: run that
        /// manager's update without asking (needed without a terminal).
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Say whether a newer release exists, without installing it.
    Check {
        /// Record the answer for the notice a later command prints, and print nothing.
        #[arg(long, hide = true)]
        record: bool,
        /// Start `--record` in a process of its own and return at once.
        #[arg(long, hide = true)]
        background: bool,
    },
    /// The versions installed here, and which is in use.
    List,
    /// Make an installed version the one in use (the one it replaces stays).
    Use {
        /// The version, as `basert self list` shows it.
        version: String,
    },
    /// Place an unpacked release bundle here and make it the one in use.
    ///
    /// What the install script runs; a copy it placed before is kept under
    /// versions/ as the version to go back to.
    Install {
        /// The unpacked bundle (moved into place).
        #[arg(long, value_name = "DIR")]
        from: PathBuf,
        /// The install directory (default: $BASERT_INSTALL_DIR, else ~/.basert).
        #[arg(long, value_name = "DIR")]
        root: Option<PathBuf>,
    },
}

pub fn cmd(args: SelfArgs) -> Result<()> {
    let mut say = |line: &str| eprintln!("{line}");
    match args.action {
        Action::Update {
            version,
            dry_run,
            yes,
        } => update(version.as_deref(), dry_run, yes, &mut say),
        Action::Check { record: true, .. } => {
            record();
            Ok(())
        }
        Action::Check {
            background: true, ..
        } => background(),
        Action::Check { .. } => check(),
        Action::List => list(),
        Action::Use { version } => use_version(&version, &mut say),
        Action::Install { from, root } => install_from(&from, root, &mut say),
    }
}

// --- where this basert came from ---------------------------------------------

/// How the running basert was installed, which decides who updates it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Installed {
    /// `<root>/versions/<v>/basert`: the versions layout, this command's to update.
    Managed { root: PathBuf, version: String },
    /// The install script's earlier layout: the release's files directly in
    /// `root`, beside a `.release` stamp.
    InPlace { root: PathBuf },
    /// A package manager's copy, cargo's, a source build: `how` it is updated,
    /// and the command that does it when there is one.
    Other {
        exe: PathBuf,
        how: String,
        command: Option<String>,
    },
}

impl Installed {
    /// Whether a package manager (cargo among them) placed this copy, so its
    /// folder is that manager's: an upgrade replaces it, an uninstall leaves
    /// behind what basert put there.
    pub fn by_package_manager(&self) -> bool {
        matches!(
            self,
            Installed::Other {
                command: Some(_),
                ..
            }
        )
    }
}

pub fn installed_as(exe: &Path) -> Installed {
    let real = fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf());
    let dir = real.parent().unwrap_or(Path::new("/")).to_path_buf();
    if let (Some(version), Some(versions)) =
        (dir.file_name().and_then(|n| n.to_str()), dir.parent())
    {
        if versions.file_name().is_some_and(|n| n == "versions") && version_key(version).is_some() {
            if let Some(root) = versions.parent() {
                return Installed::Managed {
                    root: root.to_path_buf(),
                    version: version.to_string(),
                };
            }
        }
    }
    // A package keeps the release's layout too, so it is told apart by its
    // file, not by where it is.
    let named = package_manager(&dir);
    if named.is_none() && dir.join(STAMP).is_file() {
        return Installed::InPlace { root: dir };
    }
    match delegate(&real, named) {
        Some(d) => Installed::Other {
            exe: real,
            how: format!("{} installed it: `{}` updates it", d.manager(), d.command()),
            command: Some(d.command().to_string()),
        },
        None => Installed::Other {
            how: match real.to_string_lossy().contains("/target/") {
                true => "it is a source build: build it again",
                false => "the install script did not place it: update it the way it was installed",
            }
            .to_string(),
            exe: real,
            command: None,
        },
    }
}

/// The manager and command an `installed-by` file in `dir` names.
fn package_manager(dir: &Path) -> Option<(String, String)> {
    let line = fs::read_to_string(dir.join(INSTALLED_BY)).ok()?;
    let (manager, command) = line.trim().split_once(": ")?;
    Some((manager.trim().to_string(), command.trim().to_string()))
}

// --- a copy a package manager installed ----------------------------------------

/// How a copy another tool installed is updated: by that tool's own command,
/// which `self update` runs (after asking), or which it only names, saying
/// why. The command is built here from the manager's name, never read from
/// the `installed-by` file: that file's own text is for showing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delegate {
    Run {
        manager: String,
        /// The command as a user would type it.
        command: String,
        /// What is run: the manager's binary as found from the copy's own
        /// path, so it is the installation that owns the copy.
        argv: Vec<String>,
        /// Put first on PATH (npm's `node`, beside it).
        path_first: Option<PathBuf>,
    },
    Name {
        manager: String,
        command: String,
        why: String,
    },
}

impl Delegate {
    fn manager(&self) -> &str {
        match self {
            Delegate::Run { manager, .. } | Delegate::Name { manager, .. } => manager,
        }
    }

    fn command(&self) -> &str {
        match self {
            Delegate::Run { command, .. } | Delegate::Name { command, .. } => command,
        }
    }
}

/// The delegate for the launcher at the real path `exe`, from the manager its
/// `installed-by` file names, or from where it is for packages published
/// before they carried the file; `None` for a copy no package manager placed.
fn delegate(exe: &Path, named: Option<(String, String)>) -> Option<Delegate> {
    let p = exe.to_string_lossy().into_owned();
    let dir = exe.parent().unwrap_or(Path::new("/"));
    // pipx and uv keep the wheel in a venv of their own, whose file says pip.
    if p.contains("/pipx/venvs/") {
        return Some(on_path("pipx", &["upgrade", "basert"]));
    }
    if p.contains("/uv/tools/") {
        return Some(on_path("uv", &["tool", "upgrade", "basert"]));
    }
    let manager = named.as_ref().map(|(m, _)| m.to_ascii_lowercase());
    Some(match manager.as_deref() {
        Some("homebrew" | "brew") => homebrew(&p),
        Some("npm") => npm(&p),
        Some("pip") => pip(&p),
        Some("cargo") => cargo(dir),
        Some(_) => {
            let (manager, command) = named?;
            Delegate::Name {
                manager,
                command,
                why: "basert runs only Homebrew, npm, pipx, uv and cargo itself".to_string(),
            }
        }
        None if p.contains("/Cellar/") => homebrew(&p),
        None if p.contains("/node_modules/") => npm(&p),
        None if p.contains("/site-packages/") => pip(&p),
        None if p.contains("/.cargo/bin/") => cargo(dir),
        None => return None,
    })
}

/// Whether this user may write `path`, as the manager would need to.
fn writable(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `c` is a NUL-terminated path that outlives the call.
    unsafe { libc::access(c.as_ptr(), libc::W_OK) == 0 }
}

/// `tool args…` with `tool` from PATH, or named when PATH has none.
fn on_path(tool: &str, args: &[&str]) -> Delegate {
    let command = std::iter::once(tool)
        .chain(args.iter().copied())
        .collect::<Vec<_>>()
        .join(" ");
    let found = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|d| d.join(tool))
            .find(|p| superfluid::is_executable(p))
    });
    match found {
        Some(bin) => Delegate::Run {
            manager: tool.to_string(),
            command,
            argv: std::iter::once(bin.display().to_string())
                .chain(args.iter().map(|a| a.to_string()))
                .collect(),
            path_first: None,
        },
        None => Delegate::Name {
            manager: tool.to_string(),
            why: format!("no {tool} on PATH"),
            command,
        },
    }
}

/// `brew upgrade <formula>` by the Homebrew whose Cellar holds the copy,
/// unless the formula is pinned or that Homebrew is another user's.
fn homebrew(p: &str) -> Delegate {
    let manager = "Homebrew".to_string();
    let Some(i) = p.find("/Cellar/") else {
        return Delegate::Name {
            manager,
            command: "brew upgrade basert".to_string(),
            why: "the Homebrew it came from cannot be told from where it is".to_string(),
        };
    };
    let prefix = Path::new(&p[..i]);
    let formula = p[i + "/Cellar/".len()..]
        .split('/')
        .next()
        .unwrap_or("basert");
    let command = format!("brew upgrade {formula}");
    let brew = prefix.join("bin").join("brew");
    let why = if !superfluid::is_executable(&brew) {
        format!("there is no {}", brew.display())
    } else if prefix.join("var/homebrew/pinned").join(formula).exists() {
        format!("it is pinned in Homebrew: `brew unpin {formula}` first")
    } else if !writable(&prefix.join("Cellar")) {
        format!("the Homebrew in {} is another user's", prefix.display())
    } else {
        return Delegate::Run {
            manager,
            command,
            argv: vec![brew.display().to_string(), "upgrade".into(), formula.into()],
            path_first: None,
        };
    };
    Delegate::Name {
        manager,
        command,
        why,
    }
}

/// `npm install --global` by the npm whose global folder holds the copy; a
/// project's dependency (or pnpm's, yarn's, bun's) is that project's to update.
fn npm(p: &str) -> Delegate {
    let manager = "npm".to_string();
    let command = "npm install --global @basecompute/basert@latest".to_string();
    let Some(i) = p.find("/lib/node_modules/") else {
        return Delegate::Name {
            manager,
            command: "npm install @basecompute/basert@latest".to_string(),
            why: "it is a project's dependency (or pnpm's, yarn's or bun's): update it in that project"
                .to_string(),
        };
    };
    let prefix = Path::new(&p[..i]);
    let bin = prefix.join("bin");
    let npm = bin.join("npm");
    let why = if !superfluid::is_executable(&npm) {
        format!("there is no {}", npm.display())
    } else if !writable(&prefix.join("lib").join("node_modules")) {
        format!(
            "npm's global folder in {} needs root to write: run it yourself",
            prefix.display()
        )
    } else {
        return Delegate::Run {
            manager,
            command,
            argv: vec![
                npm.display().to_string(),
                "install".into(),
                "--global".into(),
                "@basecompute/basert@latest".into(),
            ],
            path_first: Some(bin),
        };
    };
    Delegate::Name {
        manager,
        command,
        why,
    }
}

/// pip is only named: the environment may be a project's, which a lock file
/// (Poetry, uv, pip-tools) pins, or the system's, which refuses or needs root.
fn pip(p: &str) -> Delegate {
    // .../<env>/lib/python3.x/site-packages/baseRT/_engine/basert
    let python = p
        .find("/lib/python")
        .map(|i| Path::new(&p[..i]).join("bin").join("python"))
        .filter(|py| superfluid::is_executable(py));
    Delegate::Name {
        manager: "pip".to_string(),
        command: match python {
            Some(py) => format!("{} -m pip install --upgrade basert", py.display()),
            None => "pip install --upgrade basert".to_string(),
        },
        why: "its environment may be a project's, pinned by a lock file, or the system's: \
              run it in that environment yourself"
            .to_string(),
    }
}

/// `cargo install basert-cli` by the cargo beside the copy.
fn cargo(dir: &Path) -> Delegate {
    let cargo = dir.join("cargo");
    if !superfluid::is_executable(&cargo) {
        return on_path("cargo", &["install", "basert-cli"]);
    }
    Delegate::Run {
        manager: "cargo".to_string(),
        command: "cargo install basert-cli".to_string(),
        argv: vec![
            cargo.display().to_string(),
            "install".into(),
            "basert-cli".into(),
        ],
        path_first: Some(dir.to_path_buf()),
    }
}

/// `self update` of a copy another tool installed: its manager's command, run
/// after asking (or with `--yes`) when it is one basert runs, else named.
fn update_by_manager(
    exe: &Path,
    version: Option<&str>,
    dry_run: bool,
    yes: bool,
    say: &mut dyn FnMut(&str),
) -> Result<()> {
    let real = fs::canonicalize(exe).unwrap_or_else(|_| exe.to_path_buf());
    let dir = real.parent().unwrap_or(Path::new("/")).to_path_buf();
    let Some(plan) = delegate(&real, package_manager(&dir)) else {
        let Installed::Other { how, .. } = installed_as(&real) else {
            bail!("{} is not a package manager's copy", real.display());
        };
        bail!(
            "{} was not installed by the install script or a package manager, so `basert self` does not change it; {how}",
            real.display()
        );
    };
    let (manager, command, argv, path_first) = match (plan, version) {
        (
            Delegate::Name {
                manager,
                command,
                why,
            },
            _,
        ) => bail!(
            "{} was installed by {manager}, which updates it: run `{command}` ({why})",
            real.display()
        ),
        (
            Delegate::Run {
                manager, command, ..
            },
            Some(_),
        ) => bail!(
            "{} was installed by {manager}, which chooses its version: `{command}` updates it, \
             and a version is pinned there",
            real.display()
        ),
        (
            Delegate::Run {
                manager,
                command,
                argv,
                path_first,
            },
            None,
        ) => (manager, command, argv, path_first),
    };
    if dry_run {
        say(&format!(
            "{} was installed by {manager}: `self update` would run {}",
            real.display(),
            argv.join(" ")
        ));
        return Ok(());
    }
    // The manager replaces this directory: a server's workers load the engine
    // from it again with each model. cargo replaces the launcher alone; its
    // engine is wherever BASERT_LIB names.
    let busy = match manager.as_str() {
        "cargo" => Vec::new(),
        _ => servers_on_package(&dir),
    };
    if !busy.is_empty() {
        let pids = busy
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "a server runs on the engine in {} (pid {pids}), which `{command}` replaces: stop it, then run this again",
            dir.display()
        );
    }
    if !yes {
        if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
            bail!(
                "{} was installed by {manager}: run `{command}`, or `basert self update --yes` to let basert run it",
                real.display()
            );
        }
        if !superfluid::confirm(&format!(
            "{} was installed by {manager}. Run `{}`? [Y/n] ",
            real.display(),
            argv.join(" ")
        ))? {
            bail!("nothing was run; `{command}` updates it");
        }
    }
    if manager == "cargo" {
        say("cargo builds basert from source: this takes a few minutes");
    }
    say(&format!("running {}", argv.join(" ")));
    let mut run = Command::new(&argv[0]);
    run.args(&argv[1..]);
    if let Some(first) = path_first {
        let rest = std::env::var_os("PATH").unwrap_or_default();
        let path =
            std::env::join_paths(std::iter::once(first).chain(std::env::split_paths(&rest)))?;
        run.env("PATH", path);
    }
    let status = run
        .status()
        .with_context(|| format!("running {}", argv[0]))?;
    if !status.success() {
        bail!("`{}` failed ({status})", argv.join(" "));
    }
    say("the next `basert serve` brings superfluid to a release that serves the new engine (`basert superfluid install` does it now)");
    Ok(())
}

/// The servers on the engine in a package's directory `dir`: basert's and
/// superfluid's programs run from it (only theirs: a cargo install shares
/// `~/.cargo/bin` with other tools), and those `basert serve` recorded in the
/// state directory.
fn servers_on_package(dir: &Path) -> Vec<i32> {
    let ours = |exe: &Path| {
        exe.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("basert") || n.starts_with("superfluid"))
    };
    let mut pids = processes_running(dir, &ours);
    if let Some(records) = serving_records() {
        for pid in servers_recorded(&records, dir) {
            if !pids.contains(&pid) {
                pids.push(pid);
            }
        }
    }
    pids
}

fn this_exe() -> Result<PathBuf> {
    std::env::current_exe().context("this program's own path")
}

/// The directory the install script made: `<root>` for the versions layout,
/// else this launcher's own directory as it was started. superfluid is looked
/// for beside it.
pub fn install_dir() -> Result<PathBuf> {
    let exe = this_exe()?;
    if let Installed::Managed { root, .. } = installed_as(&exe) {
        return Ok(root);
    }
    exe.parent()
        .map(Path::to_path_buf)
        .context("this program's own directory")
}

/// The directory this launcher's release files are in: for the versions
/// layout the version it runs from, a real path no update changes under a
/// running server; else its own directory as it was started.
pub fn bundle_dir() -> Result<PathBuf> {
    let exe = this_exe()?;
    if let Installed::Managed { root, version } = installed_as(&exe) {
        return Ok(Layout { root }.version_dir(&version));
    }
    exe.parent()
        .map(Path::to_path_buf)
        .context("this program's own directory")
}

// --- releases ----------------------------------------------------------------

/// The engine bundle name's platform for this machine:
/// `basert-engine-<platform>-<v>.tar.gz`.
pub fn platform() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("macos-arm64"),
        ("linux", "aarch64") => Ok("linux-arm64-cuda"),
        (os, arch) => bail!("no basert release is published for {os}/{arch}"),
    }
}

/// A basert release for this machine, as a listing names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    /// `0.4.3`.
    pub version: String,
    /// `v0.4.3`.
    pub tag: String,
    pub published: String,
    /// The engine ABI number the release states beside its bundles
    /// (`basert-engine-abi-<N>.txt`), from 0.4.0 on.
    pub abi: Option<u32>,
    tarball: String,
    tarball_url: String,
    checksum_url: String,
}

/// `0.4.3` (or `v0.4.3`) as `[0, 4, 3]`; `None` for anything else.
pub fn version_key(v: &str) -> Option<Vec<u64>> {
    v.strip_prefix('v')
        .unwrap_or(v)
        .split('.')
        .map(|part| part.parse().ok())
        .collect()
}

/// The releases in a listing of GitHub's form that are neither drafts nor
/// pre-releases and publish this platform's bundle and its checksum, newest
/// first.
pub fn releases_in(listing: &Value, platform: &str) -> Vec<Release> {
    let mut found: Vec<Release> = listing
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| !r["draft"].as_bool().unwrap_or(false))
        .filter(|r| !r["prerelease"].as_bool().unwrap_or(false))
        .filter_map(|r| {
            let tag = r["tag_name"].as_str()?;
            let version = tag.strip_prefix('v').unwrap_or(tag);
            version_key(version)?;
            let assets = r["assets"].as_array()?;
            let url = |name: &str| {
                assets.iter().find(|a| a["name"] == name)?["browser_download_url"]
                    .as_str()
                    .map(str::to_string)
            };
            let tarball = format!("basert-engine-{platform}-{version}.tar.gz");
            let abi = assets.iter().find_map(|a| {
                a["name"]
                    .as_str()?
                    .strip_prefix("basert-engine-abi-")?
                    .strip_suffix(".txt")?
                    .parse()
                    .ok()
            });
            Some(Release {
                version: version.to_string(),
                tag: tag.to_string(),
                published: r["published_at"].as_str().unwrap_or_default().to_string(),
                abi,
                tarball_url: url(&tarball)?,
                checksum_url: url(&format!("{tarball}.sha256"))?,
                tarball,
            })
        })
        .collect();
    found.sort_by_key(|r| std::cmp::Reverse(version_key(&r.version)));
    found
}

/// The release listing: `$BASERT_RELEASES`, else the open repo's.
fn listing() -> Result<Value> {
    let named = std::env::var(RELEASES_ENV).ok().filter(|v| !v.is_empty());
    superfluid::read_listing(named.as_deref().unwrap_or(RELEASES))
}

/// The engine ABI number the header in a release directory states.
fn abi_in(dir: &Path) -> Option<u32> {
    fs::read_to_string(dir.join("include").join("baseRT").join("baseRT.h"))
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("#define BASERT_ABI_VERSION "))?
        .trim()
        .parse()
        .ok()
}

/// What the launcher `basert` says its version is (`basert 0.4.3`).
fn launcher_version(launcher: &Path) -> Result<String> {
    let out = Command::new(launcher)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running {} --version", launcher.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    match text.split_whitespace().collect::<Vec<_>>()[..] {
        ["basert", v, ..] if out.status.success() && version_key(v).is_some() => Ok(v.to_string()),
        _ => bail!(
            "{} --version answered {:?}, not `basert <version>`",
            launcher.display(),
            text.trim()
        ),
    }
}

// --- the versions layout -----------------------------------------------------

/// A release file, by name: what the install script replaces in its install
/// directory. Anything else there (superfluid's directory, the user's own) is
/// left alone.
fn is_bundle_file(name: &str) -> bool {
    matches!(name, "basert" | "include" | "LICENSE" | "NOTICE")
        || name.starts_with("basert-")
        || name.starts_with("baseRT_")
        || name.starts_with("libbaseRT")
        || (name.starts_with("baseRT")
            && (name.ends_with(".metallib") || name.ends_with(".fatbin")))
}

/// What `<root>` links to through `versions/current`: every name in the
/// release but its dot files and a `keep` file.
fn members(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().to_str().map(str::to_string))
                .filter(|n| !n.starts_with('.') && n != KEEP)
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub root: PathBuf,
}

impl Layout {
    fn versions(&self) -> PathBuf {
        self.root.join("versions")
    }

    pub fn version_dir(&self, v: &str) -> PathBuf {
        self.versions().join(v)
    }

    fn launcher(&self, v: &str) -> PathBuf {
        self.version_dir(v).join("basert")
    }

    pub fn current(&self) -> Option<String> {
        let target = fs::read_link(self.versions().join("current")).ok()?;
        target.file_name()?.to_str().map(str::to_string)
    }

    fn previous(&self) -> Option<String> {
        let v = fs::read_to_string(self.versions().join("previous")).ok()?;
        let v = v.trim();
        (!v.is_empty() && self.launcher(v).is_file()).then(|| v.to_string())
    }

    /// The complete versions here, newest first.
    pub fn on_disk(&self) -> Vec<String> {
        let mut all: Vec<String> = fs::read_dir(self.versions())
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| e.file_name().to_str().map(str::to_string))
                    .filter(|n| version_key(n).is_some() && self.launcher(n).is_file())
                    .collect()
            })
            .unwrap_or_default();
        all.sort_by_key(|v| std::cmp::Reverse(version_key(v)));
        all
    }

    /// The version the root's `.release` stamp names.
    fn stamped(&self) -> Option<String> {
        let tag = fs::read_to_string(self.root.join(STAMP)).ok()?;
        let v = tag.trim().trim_start_matches('v');
        version_key(v).map(|_| v.to_string())
    }

    /// The root's stamp: the release in use, by its tag.
    fn stamp(&self, tag: &str) -> Result<()> {
        let at = self.root.join(STAMP);
        fs::write(&at, format!("{tag}\n")).with_context(|| format!("writing {}", at.display()))
    }

    /// Point `current` at `v`, in one rename, remembering what it pointed at.
    fn set_current(&self, v: &str) -> Result<()> {
        if !superfluid::is_executable(&self.launcher(v)) {
            bail!("{v} is not installed here (`basert self list` shows what is)");
        }
        let was = self.current();
        relink(&self.versions().join("current"), Path::new(v))?;
        if let Some(was) = was.filter(|was| was != v) {
            self.set_previous(&was)?;
        }
        Ok(())
    }

    fn set_previous(&self, v: &str) -> Result<()> {
        let at = self.versions().join("previous");
        fs::write(&at, format!("{v}\n")).with_context(|| format!("writing {}", at.display()))
    }

    /// The root's links: one per file of the version in use, through
    /// `versions/current`. A link to a file that version does not have is
    /// removed, and so is a file the install script placed in the root before
    /// (kept under `versions/` by then): a tool left behind would load the new
    /// version's kernel files. Anything else in the way is left as it is, and
    /// said.
    fn link(&self) -> Result<Vec<String>> {
        let Some(v) = self.current() else {
            bail!("no version is in use in {}", self.versions().display());
        };
        let through = Path::new("versions").join("current");
        let mut notes = Vec::new();
        let names = members(&self.version_dir(&v));
        for name in &names {
            let link = self.root.join(name);
            let target = through.join(name);
            match fs::read_link(&link) {
                Ok(to) if to == target => continue,
                Ok(to) if to.starts_with("versions") => {}
                // A file or link is renamed over in one step; a directory
                // (include/) has to go first.
                _ if placed_by_script(&link, name) => {
                    if fs::symlink_metadata(&link).is_ok_and(|m| m.is_dir()) {
                        remove_placed(&link)?;
                    }
                }
                Ok(to) => {
                    notes.push(format!(
                        "{} links to {}, not into versions/: left as it is",
                        link.display(),
                        to.display()
                    ));
                    continue;
                }
                Err(_) if fs::symlink_metadata(&link).is_ok() => {
                    notes.push(format!(
                        "{} is not a file of a release: left as it is",
                        link.display()
                    ));
                    continue;
                }
                Err(_) => {}
            }
            relink(&link, &target)?;
        }
        for entry in fs::read_dir(&self.root)?.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if names.contains(&name) {
                continue;
            }
            let path = entry.path();
            match fs::read_link(&path) {
                Ok(to) if to.starts_with(&through) && !path.exists() => fs::remove_file(&path)
                    .with_context(|| format!("removing {}", path.display()))?,
                _ if placed_by_script(&path, &name) => remove_placed(&path)?,
                _ => {}
            }
        }
        Ok(notes)
    }

    /// Removes every version but the one in use, the one before it, one
    /// holding a `keep` file and one a server still runs on.
    fn prune(&self) -> Result<Vec<String>> {
        let (current, previous) = (self.current(), self.previous());
        let mut gone = Vec::new();
        for v in self.on_disk() {
            let dir = self.version_dir(&v);
            if Some(&v) == current.as_ref()
                || Some(&v) == previous.as_ref()
                || dir.join(KEEP).exists()
                || serving(&dir)
            {
                continue;
            }
            fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
            gone.push(v);
        }
        Ok(gone)
    }
}

/// Whether `path` in the root is the install script's earlier copy of a
/// release file: a file or directory of a release's name, or a link of one to
/// a file beside it (the library's versioned names).
fn placed_by_script(path: &Path, name: &str) -> bool {
    is_bundle_file(name)
        && match fs::read_link(path) {
            Ok(to) => to.components().count() == 1 && !to.starts_with("versions"),
            Err(_) => fs::symlink_metadata(path).is_ok(),
        }
}

fn remove_placed(path: &Path) -> Result<()> {
    let is_dir = fs::symlink_metadata(path).is_ok_and(|m| m.is_dir());
    match is_dir {
        true => fs::remove_dir_all(path),
        false => fs::remove_file(path),
    }
    .with_context(|| format!("removing {}", path.display()))
}

/// `link` made to point at `target`, in one rename.
fn relink(link: &Path, target: &Path) -> Result<()> {
    let name = link.file_name().and_then(|n| n.to_str()).unwrap_or("link");
    let tmp = link.with_file_name(format!(".{name}.tmp"));
    let _ = fs::remove_file(&tmp);
    std::os::unix::fs::symlink(target, &tmp)
        .with_context(|| format!("creating {}", tmp.display()))?;
    fs::rename(&tmp, link).with_context(|| format!("replacing {}", link.display()))
}

/// A server records the engine it runs on, so that no update removes or
/// replaces it while it runs: `basert serve` calls this just before it becomes
/// superfluid (same pid). A version of the layout holds the record itself; a
/// package's directory is the package manager's, so its servers are recorded
/// in the state directory instead, a file per pid naming the engine.
pub fn note_serving(engine: Option<&Path>) {
    let Some(engine) = engine else { return };
    // By its real path, and a library named by its file as its directory:
    // the version is found however `$BASERT_LIB` or `--basert-lib` names it.
    let engine = canonical(engine);
    let engine = match engine.is_dir() {
        true => engine,
        false => match engine.parent() {
            Some(dir) => dir.to_path_buf(),
            None => return,
        },
    };
    let pid = std::process::id().to_string();
    let in_layout = engine
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|n| n == "versions");
    let (dir, text) = match in_layout {
        true => (engine.join(SERVING), String::new()),
        false => match serving_records() {
            Some(records) => (records, format!("{}\n", engine.display())),
            None => return,
        },
    };
    if fs::create_dir_all(&dir).is_ok() {
        let _ = fs::write(dir.join(pid), text);
    }
}

/// Where `basert serve` records the servers on a package's engine.
fn serving_records() -> Option<PathBuf> {
    crate::serve::state_dir().ok().map(|d| d.join("serving"))
}

fn canonical(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The live servers `records` names on the engine in `dir`; the records of
/// servers that stopped are removed.
fn servers_recorded(records: &Path, dir: &Path) -> Vec<i32> {
    let dir = canonical(dir);
    let mut pids = Vec::new();
    for entry in fs::read_dir(records).into_iter().flatten().flatten() {
        let pid = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<i32>().ok());
        match pid {
            Some(pid) if alive(pid) => {
                let on = fs::read_to_string(entry.path()).unwrap_or_default();
                if Path::new(on.trim()) == dir {
                    pids.push(pid);
                }
            }
            _ => {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    pids
}

/// Whether a server still runs on the version in `dir`; the records of
/// servers that stopped are removed.
fn serving(dir: &Path) -> bool {
    !servers_on_version(dir).is_empty()
}

/// The live servers recorded on the version in `dir`; the records of servers
/// that stopped are removed.
fn servers_on_version(dir: &Path) -> Vec<i32> {
    let mut pids = Vec::new();
    for entry in fs::read_dir(dir.join(SERVING))
        .into_iter()
        .flatten()
        .flatten()
    {
        match entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<i32>().ok())
        {
            Some(pid) if alive(pid) => pids.push(pid),
            _ => {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    pids
}

/// The servers on the engine in `root`, the install script's copy from before
/// the layout, whose files moving it under the layout turns into links: a
/// server started from it reads its engine from `root` by that path, so its
/// next model would load the new release's. Those `basert serve` recorded on
/// it, and superfluid and the runtime tools run from it (not the launcher,
/// which loads no engine, and runs this).
fn servers_on_flat(root: &Path) -> Vec<i32> {
    let on_engine = |exe: &Path| {
        exe.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("superfluid") || n.starts_with("basert-"))
    };
    let mut pids = processes_running(root, &on_engine);
    if let Some(records) = serving_records() {
        for pid in servers_recorded(&records, root) {
            if !pids.contains(&pid) {
                pids.push(pid);
            }
        }
    }
    pids
}

/// That no server runs on the flat copy in `root` (see [`servers_on_flat`]).
fn flat_unused(root: &Path) -> Result<()> {
    let pids = servers_on_flat(root);
    if pids.is_empty() {
        return Ok(());
    }
    bail!(
        "a server runs on the engine in {} (pid {}), whose files this would replace: stop it, \
         then run this again",
        root.display(),
        pid_list(&pids)
    )
}

fn pid_list(pids: &[i32]) -> String {
    pids.iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 delivers nothing; it only asks whether the process exists.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Only one `basert self` changes a layout at a time.
struct Lock(PathBuf);

impl Lock {
    fn take(layout: &Layout) -> Result<Lock> {
        let versions = layout.versions();
        fs::create_dir_all(&versions)
            .with_context(|| format!("creating {}", versions.display()))?;
        let path = versions.join(".lock");
        for _ in 0..2 {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    use std::io::Write;
                    let _ = writeln!(f, "{}", std::process::id());
                    return Ok(Lock(path));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let holder = fs::read_to_string(&path)
                        .ok()
                        .and_then(|s| s.trim().parse::<i32>().ok());
                    if let Some(pid) = holder.filter(|&p| alive(p)) {
                        bail!(
                            "another `basert self` (pid {pid}) is changing {}; run this again once it is done",
                            layout.root.display()
                        );
                    }
                    let _ = fs::remove_file(&path);
                }
                Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
            }
        }
        bail!("could not take {}", path.display())
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// `from` copied to `to`, links as links.
fn copy_entry(from: &Path, to: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(from).with_context(|| format!("reading {}", from.display()))?;
    if meta.file_type().is_symlink() {
        std::os::unix::fs::symlink(fs::read_link(from)?, to)?;
    } else if meta.is_dir() {
        fs::create_dir_all(to)?;
        for entry in fs::read_dir(from)?.flatten() {
            copy_entry(&entry.path(), &to.join(entry.file_name()))?;
        }
    } else {
        fs::copy(from, to).with_context(|| format!("copying {}", from.display()))?;
    }
    Ok(())
}

/// The install script's earlier copy, kept under the layout as the version to
/// go back to. It is copied beside the layout and moved in whole, so a copy
/// cut short is never taken for the version; the files in the root stay until
/// the links replace them.
fn keep_in_place(layout: &Layout, say: &mut dyn FnMut(&str)) -> Result<String> {
    let old = match layout.stamped() {
        Some(v) => v,
        None => launcher_version(&layout.root.join("basert"))?,
    };
    let dest = layout.version_dir(&old);
    if superfluid::is_executable(&dest.join("basert")) {
        return Ok(old);
    }
    let staging = layout
        .versions()
        .join(".staging")
        .join(format!("kept-{old}"));
    let _ = fs::remove_dir_all(&staging);
    let copied = (|| -> Result<()> {
        fs::create_dir_all(&staging)?;
        for entry in fs::read_dir(&layout.root)?.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if is_bundle_file(&name) {
                copy_entry(&entry.path(), &staging.join(&name))?;
            }
        }
        fs::write(staging.join(STAMP), format!("v{old}\n"))?;
        let _ = fs::remove_dir_all(&dest);
        fs::rename(&staging, &dest)
            .with_context(|| format!("{} -> {}", staging.display(), dest.display()))
    })();
    if copied.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    copied?;
    say(&format!(
        "the release in place in {} ({old}) is kept as {}",
        layout.root.display(),
        dest.display()
    ));
    Ok(old)
}

/// Download `release`, check its SHA-256 against the one it publishes, unpack
/// it, check its launcher runs and is that version, and move it in as
/// `versions/<v>`. A complete copy already there is used as it is. Whether
/// anything was downloaded.
fn place(layout: &Layout, release: &Release, say: &mut dyn FnMut(&str)) -> Result<bool> {
    let dest = layout.version_dir(&release.version);
    let stamped = fs::read_to_string(dest.join(STAMP)).unwrap_or_default();
    if superfluid::is_executable(&dest.join("basert")) && stamped.trim() == release.tag {
        return Ok(false);
    }
    // Another copy is there (cut short, or another build of the tag): a server
    // on it reads its engine by that path.
    let pids = servers_on_version(&dest);
    if !pids.is_empty() {
        bail!(
            "{} holds another copy of basert {} and a server runs on it (pid {}): stop it, \
             then run this again",
            dest.display(),
            release.version,
            pid_list(&pids)
        );
    }
    let staging = layout.versions().join(".staging");
    fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    // Beside the layout, so the final rename stays on one filesystem.
    let stage = tempfile::Builder::new()
        .prefix("new-")
        .tempdir_in(&staging)
        .with_context(|| format!("creating a staging directory in {}", staging.display()))?;
    let tarball = stage.path().join(&release.tarball);
    let checksum = stage.path().join(format!("{}.sha256", release.tarball));
    say(&format!(
        "downloading basert {} ({}) from {}",
        release.version, release.tarball, release.tarball_url
    ));
    superfluid::download(&release.tarball_url, &tarball)?;
    superfluid::download(&release.checksum_url, &checksum)?;
    superfluid::check_sha256(&tarball, &checksum)?;
    let unpacked = stage.path().join("unpacked");
    fs::create_dir(&unpacked)?;
    superfluid::untar(&tarball, &unpacked)?;
    let said = launcher_version(&unpacked.join("basert"))
        .with_context(|| format!("{} holds no basert launcher that runs", release.tarball))?;
    if said != release.version {
        bail!(
            "{}: its launcher says it is {said}, not {}",
            release.tarball,
            release.version
        );
    }
    fs::write(unpacked.join(STAMP), format!("{}\n", release.tag))?;
    if dest.exists() {
        fs::remove_dir_all(&dest).with_context(|| format!("removing {}", dest.display()))?;
    }
    fs::rename(&unpacked, &dest)
        .with_context(|| format!("moving the release into {}", dest.display()))?;
    Ok(true)
}

// --- superfluid --------------------------------------------------------------

/// What superfluid needs for an engine, found before anything changes.
enum Pairing {
    /// None is installed: `basert serve` installs one at its first start.
    NotInstalled,
    /// The installed one serves the engine (and stays).
    Serves,
    /// A release, downloaded and asked, to take the installed one's place.
    Replace {
        installed: superfluid::Install,
        staged: superfluid::Staged,
        /// This install's turn among the installs into that directory,
        /// kept until the swap.
        turn: Option<superfluid::Held>,
    },
}

/// What superfluid needs to serve the engine in `engine`, whose ABI number is
/// `abi`. When the installed one cannot, the newest release that states the
/// number is downloaded and asked, whatever its order (going back to an older
/// basert may take an older superfluid); that no release can is an error. With
/// `forward`, a newer release that serves it is taken even when the installed
/// one can. Nothing is replaced under a running server.
fn pair_superfluid(
    engine: &Path,
    abi: Option<u32>,
    forward: bool,
    say: &mut dyn FnMut(&str),
) -> Result<Pairing> {
    let Some(installed) = superfluid::Install::locate()? else {
        return Ok(Pairing::NotInstalled);
    };
    installed.claim();
    // Installs into it take turns; what is there is asked once this one's
    // turn has come.
    let turn = installed.take_turn(say)?;
    let home = crate::serve::superfluid_home().ok();
    let have = installed
        .installed_release()
        .unwrap_or_else(|| "of a release it does not say".to_string());
    let ready = installed.probe(Some(engine), home.as_deref()).ready;
    if ready.is_ok() && !forward {
        return Ok(Pairing::Serves);
    }
    let found = match abi {
        Some(abi) => superfluid::newest_for(abi).map_err(|e| format!("{e:#}")),
        None => Err("its engine states no ABI number".to_string()),
    };
    let release = match (&ready, found) {
        (Err(_), Ok(Some(release))) => release,
        (Ok(()), Ok(Some(release))) if superfluid::newer(Some(&release.tag), &have) => release,
        (Ok(()), _) => return Ok(Pairing::Serves),
        (Err(why), Ok(None)) => bail!(
            "superfluid {have} at {} cannot serve this engine ({why}), and no superfluid release \
             states its ABI number {} yet",
            installed.root.display(),
            abi.unwrap_or_default()
        ),
        (Err(why), Err(lookup)) => bail!(
            "superfluid {have} at {} cannot serve this engine ({why}), and no release that can \
             was found ({lookup})",
            installed.root.display()
        ),
    };
    let tag = release.tag.clone();
    let busy = running_from(&installed.root);
    if !busy.is_empty() {
        let pids = busy
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        if let Err(why) = &ready {
            bail!(
                "superfluid {have} at {} cannot serve this engine ({why}) and {tag} can, but a \
                 server runs from it (pid {pids}): stop it and run this again",
                installed.root.display()
            );
        }
        say(&format!(
            "superfluid {tag} is out, but a server runs from {} (pid {pids}): {have} stays until the next update",
            installed.root.display()
        ));
        return Ok(Pairing::Serves);
    }
    if !installed.writable() {
        if let Err(why) = &ready {
            bail!(
                "superfluid {have} at {} cannot serve this engine ({why}), and that directory \
                 cannot be written to put {tag} in its place",
                installed.root.display()
            );
        }
        return Ok(Pairing::Serves);
    }
    let staged = installed.stage(&superfluid::Source::GitHub, &release, say)?;
    match staged.fresh.probe(Some(engine), home.as_deref()).ready {
        // Replaced, its directory goes whole: one that holds more than
        // superfluid (a prefix `$BASERT_SUPERFLUID_DIR` names) stays.
        Ok(()) => match (installed.replaceable(&staged.fresh.root), &ready) {
            (Ok(()), _) => Ok(Pairing::Replace {
                installed,
                staged,
                turn,
            }),
            (Err(e), Ok(())) => {
                say(&format!("superfluid {tag} is out, but {have} stays: {e:#}"));
                Ok(Pairing::Serves)
            }
            (Err(e), Err(why)) => bail!(
                "superfluid {have} at {} cannot serve this engine ({why}) and {tag} can, but {e:#}",
                installed.root.display()
            ),
        },
        Err(why) if ready.is_err() => {
            bail!("superfluid {tag} cannot serve this engine either: {why}")
        }
        Err(why) => {
            say(&format!(
                "superfluid {tag} cannot serve this engine ({why}): {have} stays"
            ));
            Ok(Pairing::Serves)
        }
    }
}

fn apply(pairing: Pairing, say: &mut dyn FnMut(&str)) -> Result<()> {
    match pairing {
        Pairing::Replace {
            installed,
            staged,
            turn,
        } => {
            let swapped = installed.swap_in(staged, say);
            drop(turn);
            swapped
        }
        Pairing::NotInstalled | Pairing::Serves => Ok(()),
    }
}

// --- the commands --------------------------------------------------------------

/// The layout the running basert belongs to, and whether it is the install
/// script's earlier, in-place one; a copy installed otherwise is an error that
/// says how to update it.
fn own_layout() -> Result<(Layout, bool)> {
    let exe = this_exe()?;
    match installed_as(&exe) {
        Installed::Managed { root, .. } => Ok((Layout { root }, false)),
        Installed::InPlace { root } => Ok((Layout { root }, true)),
        Installed::Other { exe, how, .. } => bail!(
            "{} was not installed by the install script, so `basert self` does not change it; {how}",
            exe.display()
        ),
    }
}

/// `self update`.
fn update(
    version: Option<&str>,
    dry_run: bool,
    yes: bool,
    say: &mut dyn FnMut(&str),
) -> Result<()> {
    let exe = this_exe()?;
    if let Installed::Other { .. } = installed_as(&exe) {
        return update_by_manager(&exe, version, dry_run, yes, say);
    }
    let (layout, in_place) = own_layout()?;
    let platform = platform()?;
    let _lock = match dry_run {
        true => None,
        false => Some(Lock::take(&layout)?),
    };
    let found = releases_in(&listing()?, platform);
    let chosen = match version {
        Some(want) => {
            let want = want.trim_start_matches('v');
            found.iter().find(|r| r.version == want).with_context(|| {
                format!(
                    "no release {want} for {platform} (the newest is {})",
                    found.first().map_or("none", |r| r.version.as_str())
                )
            })?
        }
        None => found.first().with_context(|| {
            format!("the release listing names no basert release for {platform}")
        })?,
    };
    let v = chosen.version.as_str();
    let here = match in_place {
        true => layout.stamped(),
        false => layout.current(),
    }
    .unwrap_or_else(|| VERSION.to_string());
    let engine_here = match in_place {
        true => layout.root.clone(),
        false => layout.version_dir(&here),
    };

    if v == here {
        say(&format!(
            "basert {here} is in use, {}",
            if version.is_some() {
                "the release named"
            } else {
                "the newest release"
            }
        ));
        if dry_run {
            return Ok(());
        }
        // superfluid as well: the newest release that serves this engine.
        let pairing =
            pair_superfluid(&engine_here, abi_in(&engine_here).or(chosen.abi), true, say)?;
        return apply(pairing, say);
    }
    if version.is_none() && version_key(v) < version_key(&here) {
        say(&format!(
            "basert {here} is in use, newer than the newest release {v}: nothing is changed (`--version {v}` goes to it)"
        ));
        return Ok(());
    }
    let on_disk = layout.launcher(v).is_file();
    say(&format!(
        "basert {here} -> {v} (published {}{})",
        chosen.published.split('T').next().unwrap_or("?"),
        if on_disk { ", already here" } else { "" }
    ));
    if dry_run {
        say("dry run: nothing fetched or changed");
        return Ok(());
    }

    let kept = match in_place {
        true => {
            flat_unused(&layout.root)?;
            Some(keep_in_place(&layout, say)?)
        }
        false => None,
    };
    let placed = place(&layout, chosen, say)?;
    let engine = layout.version_dir(v);
    let pairing = match pair_superfluid(&engine, abi_in(&engine).or(chosen.abi), true, say) {
        Ok(pairing) => pairing,
        Err(e) => {
            if placed {
                let _ = fs::remove_dir_all(&engine);
            }
            return Err(e.context(format!(
                "basert {v} was not made the one in use; {here} stays"
            )));
        }
    };
    apply(pairing, say)?;
    switch(&layout, v, &chosen.tag, kept.as_deref(), say)
}

/// The layout's `current` turned to `v`, the root's stamp and links after it,
/// older versions pruned; `kept` is the in-place copy just moved under it.
fn switch(
    layout: &Layout,
    v: &str,
    tag: &str,
    kept: Option<&str>,
    say: &mut dyn FnMut(&str),
) -> Result<()> {
    layout.set_current(v)?;
    if let Some(old) = kept.filter(|old| *old != v) {
        layout.set_previous(old)?;
    }
    layout.stamp(tag)?;
    for note in layout.link()? {
        say(&note);
    }
    println!(
        "basert {v} is in use ({})",
        layout.versions().join("current").display()
    );
    if let Some(prev) = layout.previous() {
        say(&format!(
            "{prev} stays installed: `basert self use {prev}` goes back to it"
        ));
    }
    match layout.prune() {
        Ok(gone) if !gone.is_empty() => say(&format!("removed {}", gone.join(", "))),
        Ok(_) => {}
        Err(e) => say(&format!(
            "an older version was not removed ({e:#}); the next update tries again"
        )),
    }
    say("a server already running keeps the engine it started with; starting it again uses this one");
    Ok(())
}

/// `self use <v>`.
fn use_version(v: &str, say: &mut dyn FnMut(&str)) -> Result<()> {
    let (layout, in_place) = own_layout()?;
    if in_place {
        bail!(
            "basert {} is installed in place in {}, with no other version beside it \
             (`basert self update` moves it under versions/)",
            layout.stamped().unwrap_or_else(|| VERSION.to_string()),
            layout.root.display()
        );
    }
    let v = v.trim_start_matches('v');
    let _lock = Lock::take(&layout)?;
    if !layout.launcher(v).is_file() {
        bail!("{v} is not installed here (`basert self list` shows what is)");
    }
    if layout.current().as_deref() == Some(v) {
        println!("basert {v} is in use");
        return Ok(());
    }
    let engine = layout.version_dir(v);
    let pairing = pair_superfluid(&engine, abi_in(&engine), false, say)
        .with_context(|| format!("basert {v} was not made the one in use"))?;
    apply(pairing, say)?;
    let tag = fs::read_to_string(engine.join(STAMP))
        .map(|t| t.trim().to_string())
        .unwrap_or_else(|_| format!("v{v}"));
    switch(&layout, v, &tag, None, say)
}

/// `self install --from <dir>`: the install script's last step.
fn install_from(from: &Path, root: Option<PathBuf>, say: &mut dyn FnMut(&str)) -> Result<()> {
    let root = match root {
        Some(root) => root,
        None => match std::env::var_os("BASERT_INSTALL_DIR").filter(|d| !d.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => dirs::home_dir()
                .context("no home directory")?
                .join(".basert"),
        },
    };
    let layout = Layout { root };
    let v = launcher_version(&from.join("basert"))?;
    let tag = format!("v{v}");
    let _lock = Lock::take(&layout)?;
    let in_place = fs::symlink_metadata(layout.root.join("basert")).is_ok_and(|m| m.is_file())
        && layout.stamped().is_some();
    let kept = match in_place {
        true => {
            flat_unused(&layout.root)?;
            Some(keep_in_place(&layout, say)?)
        }
        false => None,
    };
    let dest = layout.version_dir(&v);
    // A server on that version reads its engine by its path, and its record
    // is in it: it stays, and is made the one in use.
    let pids = servers_on_version(&dest);
    if !pids.is_empty() && superfluid::is_executable(&dest.join("basert")) {
        say(&format!(
            "basert {v} in {} stays as it is: a server runs on it (pid {}); {} is not placed",
            dest.display(),
            pid_list(&pids),
            from.display()
        ));
        return switch(&layout, &v, &tag, kept.as_deref(), say);
    }
    let aside = layout
        .versions()
        .join(".staging")
        .join(format!("replaced-{v}"));
    let _ = fs::remove_dir_all(&aside);
    let had = dest.exists();
    if had {
        fs::create_dir_all(aside.parent().unwrap_or(&layout.root))?;
        fs::rename(&dest, &aside).with_context(|| format!("moving {} aside", dest.display()))?;
    }
    let moved = fs::rename(from, &dest).or_else(|_| {
        // Another filesystem: copied, then the bundle removed.
        copy_entry(from, &dest).and_then(|()| {
            fs::remove_dir_all(from).with_context(|| format!("removing {}", from.display()))
        })
    });
    if let Err(e) = moved {
        let _ = fs::remove_dir_all(&dest);
        if had {
            let _ = fs::rename(&aside, &dest);
        }
        return Err(e).with_context(|| format!("placing {} as {}", from.display(), dest.display()));
    }
    let _ = fs::remove_dir_all(&aside);
    fs::write(dest.join(STAMP), format!("{tag}\n"))?;
    switch(&layout, &v, &tag, kept.as_deref(), say)
}

/// `self list`.
fn list() -> Result<()> {
    match installed_as(&this_exe()?) {
        Installed::Managed { root, .. } => {
            let layout = Layout { root };
            let (current, previous) = (layout.current(), layout.previous());
            println!("versions in {}:", layout.versions().display());
            for v in layout.on_disk() {
                let dir = layout.version_dir(&v);
                let mut notes = Vec::new();
                if Some(&v) == current.as_ref() {
                    notes.push("in use".to_string());
                }
                if Some(&v) == previous.as_ref() {
                    notes.push(format!(
                        "the one before: `basert self use {v}` goes back to it"
                    ));
                }
                if dir.join(KEEP).exists() {
                    notes.push("kept".to_string());
                }
                if serving(&dir) {
                    notes.push("a server runs on it".to_string());
                }
                let mark = if Some(&v) == current.as_ref() {
                    "*"
                } else {
                    " "
                };
                match notes.is_empty() {
                    true => println!("{mark} {v}"),
                    false => println!("{mark} {v} ({})", notes.join("; ")),
                }
            }
        }
        Installed::InPlace { root } => println!(
            "basert {} is installed in place in {} (the install script's earlier layout; \
             `basert self update` moves it under versions/)",
            Layout { root: root.clone() }
                .stamped()
                .unwrap_or_else(|| VERSION.to_string()),
            root.display()
        ),
        Installed::Other { exe, how, .. } => {
            println!("basert {VERSION} at {}: {how}", exe.display())
        }
    }
    match superfluid::Install::locate()? {
        Some(i) => println!(
            "superfluid {} at {}",
            i.installed_release()
                .unwrap_or_else(|| "of a release it does not say".to_string()),
            i.root.display()
        ),
        None => {
            println!("superfluid: not installed (`basert serve` installs it at its first start)")
        }
    }
    Ok(())
}

// --- the check, and the notice at a launch ------------------------------------

/// What the last look at the release listings found, kept in the state
/// directory for the notice.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Check {
    checked_at: u64,
    /// The newest basert release past the hold-off whose engine a superfluid
    /// release serves.
    newest: Option<String>,
    /// The newest superfluid release past the hold-off that serves the engine
    /// beside this basert.
    superfluid: Option<String>,
    notified_at: u64,
}

impl Check {
    fn file() -> Option<PathBuf> {
        crate::serve::state_dir()
            .ok()
            .map(|d| d.join("update-check.json"))
    }

    fn read(path: &Path) -> Option<Check> {
        let v: Value = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
        Some(Check {
            checked_at: v["checked_at"].as_u64().unwrap_or(0),
            newest: v["newest"].as_str().map(str::to_string),
            superfluid: v["superfluid"].as_str().map(str::to_string),
            notified_at: v["notified_at"].as_u64().unwrap_or(0),
        })
    }

    fn write(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let v = json!({
            "checked_at": self.checked_at,
            "newest": self.newest,
            "superfluid": self.superfluid,
            "notified_at": self.notified_at,
        });
        fs::write(path, format!("{v:#}\n")).with_context(|| format!("writing {}", path.display()))
    }

    /// The line a launch prints for what this found, if any: `installed` is
    /// how this basert was installed, `superfluid_have` the release of the
    /// superfluid beside it.
    fn line(&self, installed: &Installed, superfluid_have: Option<&str>) -> Option<String> {
        let ours = matches!(
            installed,
            Installed::Managed { .. } | Installed::InPlace { .. }
        );
        if let Some(n) = self
            .newest
            .as_deref()
            .filter(|n| version_key(n) > version_key(VERSION))
        {
            let command = match installed {
                Installed::Other { command, .. } => command.as_deref()?,
                _ => "basert self update",
            };
            return Some(format!(
                "basert {n} is available (this is {VERSION}): `{command}` installs it; {NO_CHECK_ENV}=1 silences this"
            ));
        }
        let sf = self.superfluid.as_deref()?;
        let have = superfluid_have?;
        if !superfluid::newer(Some(sf), have) {
            return None;
        }
        let command = if ours {
            "basert self update"
        } else {
            "basert superfluid install"
        };
        Some(format!(
            "superfluid {sf}, the server `basert serve` runs, is available (installed: {have}): \
             `{command}` installs it; {NO_CHECK_ENV}=1 silences this"
        ))
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Seconds since the epoch of a timestamp of GitHub's (`2026-10-27T06:42:10Z`).
fn epoch_of(stamp: &str) -> Option<u64> {
    let (date, time) = stamp.trim_end_matches('Z').split_once('T')?;
    let mut d = date.split('-').map(|p| p.parse::<i64>().ok());
    let (y, m, day) = (d.next()??, d.next()??, d.next()??);
    let mut t = time.split(':').map(|p| p.parse::<i64>().ok());
    let (h, min, s) = (t.next()??, t.next()??, t.next()??);
    // Days from 1970-01-01 to the date (Howard Hinnant's days_from_civil).
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let days = era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468;
    u64::try_from(days * 86_400 + h * 3_600 + min * 60 + s).ok()
}

/// Whether a release published at `stamp` is past the hold-off at `now`.
fn settled(stamp: &str, now: u64) -> bool {
    epoch_of(stamp).is_none_or(|at| now.saturating_sub(at) >= HOLD_OFF.as_secs())
}

/// What the listings say now: basert's releases, newest first, each with
/// whether a superfluid release serves its engine, and superfluid's listing
/// when it could be read.
struct Found {
    releases: Vec<(Release, bool)>,
    superfluid: Option<Value>,
}

fn look() -> Result<Found> {
    let releases = releases_in(&listing()?, platform()?);
    let sf_listing = superfluid::releases().ok();
    let target = superfluid::target().ok();
    let releases = releases
        .into_iter()
        .map(|r| {
            let served = match (r.abi, &sf_listing, target) {
                (Some(abi), Some(l), Some(t)) => superfluid::newest_in(l, t, abi).is_some(),
                _ => true,
            };
            (r, served)
        })
        .collect();
    Ok(Found {
        releases,
        superfluid: sf_listing,
    })
}

impl Found {
    /// The newest release a `self update` would go to, past the hold-off when
    /// asked.
    fn newest(&self, hold_off: Option<u64>) -> Option<&Release> {
        self.releases
            .iter()
            .filter(|(r, served)| *served && hold_off.is_none_or(|now| settled(&r.published, now)))
            .map(|(r, _)| r)
            .next()
    }

    /// The newest superfluid release that serves this basert's engine.
    fn newest_superfluid(&self, hold_off: Option<u64>) -> Option<String> {
        let listing = self.superfluid.as_ref()?;
        let keep = |r: &Value| {
            hold_off.is_none_or(|now| settled(r["published_at"].as_str().unwrap_or_default(), now))
        };
        superfluid::newest_where(
            listing,
            superfluid::target().ok()?,
            superfluid::ENGINE_ABI,
            &keep,
        )
        .map(|r| r.tag)
    }

    fn record(&self, path: &Path) {
        let now = now();
        let mut check = Check::read(path).unwrap_or_default();
        check.checked_at = now;
        check.newest = self.newest(Some(now)).map(|r| r.version.clone());
        check.superfluid = self.newest_superfluid(Some(now));
        let _ = check.write(path);
    }
}

/// `self check`.
fn check() -> Result<()> {
    let found = look()?;
    if let Some(path) = Check::file() {
        found.record(&path);
    }
    let installed = installed_as(&this_exe()?);
    let command = match &installed {
        Installed::Other { command, .. } => command.clone(),
        _ => Some("basert self update".to_string()),
    };
    match found.releases.first() {
        Some((r, served)) => {
            let after = match (version_key(&r.version) > version_key(VERSION), served) {
                (true, true) => command
                    .map(|c| format!(": `{c}` installs it"))
                    .unwrap_or_default(),
                (_, false) => format!(
                    "; it states engine ABI {}, which no superfluid release serves yet",
                    r.abi.unwrap_or_default()
                ),
                (false, true) => String::new(),
            };
            println!(
                "basert {VERSION} is this one; the newest release is {} (published {}){after}",
                r.version,
                r.published.split('T').next().unwrap_or("?"),
            );
        }
        None => {
            println!("basert {VERSION} is this one; the listing names no release for this machine")
        }
    }
    let newest_sf = found.newest_superfluid(None);
    match superfluid::Install::locate()? {
        Some(i) => println!(
            "superfluid {} at {}; the newest release that serves this engine is {}",
            i.installed_release()
                .unwrap_or_else(|| "of a release it does not say".to_string()),
            i.root.display(),
            newest_sf.as_deref().unwrap_or("not known")
        ),
        None => println!(
            "superfluid: not installed; the newest release that serves this engine is {}",
            newest_sf.as_deref().unwrap_or("not known")
        ),
    }
    Ok(())
}

/// `self check --record`: the look a launch started, recorded for the notice.
fn record() {
    let Some(path) = Check::file() else { return };
    // A failed look is tried again a day later: the launch that started this
    // claimed the day.
    if let Ok(found) = look() {
        found.record(&path);
    }
}

/// `self check --background`: `--record` in a process of its own, which
/// outlives this one (and the launch that ran this, which waits only for this).
fn background() -> Result<()> {
    use std::os::unix::process::CommandExt;
    Command::new(this_exe()?)
        .args(["self", "check", "--record"])
        .env(NO_CHECK_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        // Its own process group: a Ctrl-C at the terminal is not for it.
        .process_group(0)
        .spawn()
        .context("starting the update check")?;
    Ok(())
}

fn check_disabled() -> bool {
    std::env::var_os(NO_CHECK_ENV).is_some_and(|v| !v.is_empty() && v != "0")
}

/// What a launch does before its command, on a terminal: start the day's look
/// for a newer release when it is due, and print the line for one found
/// before, once a day at most. `first` is the command's first argument.
pub fn at_launch(first: Option<&str>) {
    if matches!(
        first,
        None | Some("self" | "-V" | "--version" | "-h" | "--help" | "help")
    ) {
        return;
    }
    if check_disabled() || !std::io::stderr().is_terminal() {
        return;
    }
    let Ok(exe) = this_exe() else { return };
    let installed = installed_as(&exe);
    if matches!(installed, Installed::Other { command: None, .. }) {
        return;
    }
    let Some(path) = Check::file() else { return };
    let mut check = Check::read(&path).unwrap_or_default();
    let now = now();
    if now.saturating_sub(check.checked_at) >= CHECK_EVERY.as_secs() {
        // Claimed before it starts, so the commands that follow start none.
        check.checked_at = now;
        let _ = check.write(&path);
        let _ = Command::new(&exe)
            .args(["self", "check", "--background"])
            .env(NO_CHECK_ENV, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    if now.saturating_sub(check.notified_at) < CHECK_EVERY.as_secs() {
        return;
    }
    let have = superfluid::Install::locate()
        .ok()
        .flatten()
        .and_then(|i| i.installed_release());
    if let Some(line) = check.line(&installed, have.as_deref()) {
        eprintln!("{line}");
        // Read again: the look may have recorded since.
        let mut latest = Check::read(&path).unwrap_or(check);
        latest.notified_at = now;
        let _ = latest.write(&path);
    }
}

// --- processes ------------------------------------------------------------------

/// The other processes running an executable under `dir`, by pid.
pub(crate) fn running_from(dir: &Path) -> Vec<i32> {
    processes_running(dir, &|_| true)
}

/// [`running_from`], among the executables `keep` accepts.
fn processes_running(dir: &Path, keep: &dyn Fn(&Path) -> bool) -> Vec<i32> {
    let Ok(dir) = fs::canonicalize(dir) else {
        return Vec::new();
    };
    other_processes()
        .into_iter()
        .filter(|(_, exe)| exe.starts_with(&dir) && keep(exe))
        .map(|(p, _)| p)
        .collect()
}

/// The other processes on this machine, each with its executable as a real path.
#[cfg(target_os = "linux")]
fn other_processes() -> Vec<(i32, PathBuf)> {
    let me = std::process::id() as i32;
    fs::read_dir("/proc")
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let pid = e.file_name().to_str()?.parse::<i32>().ok()?;
                    (pid != me).then(|| (pid, e.path()))
                })
                .filter_map(|(p, dir)| Some((p, fs::read_link(dir.join("exe")).ok()?)))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(target_os = "macos")]
fn other_processes() -> Vec<(i32, PathBuf)> {
    let me = std::process::id() as libc::c_int;
    // SAFETY: with a null buffer, proc_listallpids only reports how many pids there are.
    let n = unsafe { libc::proc_listallpids(std::ptr::null_mut(), 0) };
    if n <= 0 {
        return Vec::new();
    }
    let mut pids = vec![0 as libc::c_int; n as usize * 2 + 16];
    let bytes = (pids.len() * std::mem::size_of::<libc::c_int>()) as libc::c_int;
    // SAFETY: the buffer is this function's, and its size in bytes is what is passed.
    let got = unsafe { libc::proc_listallpids(pids.as_mut_ptr() as *mut libc::c_void, bytes) };
    pids.truncate(got.max(0) as usize);
    pids.into_iter()
        .filter(|&p| p > 0 && p != me)
        .filter_map(|p| {
            let mut buf = vec![0u8; 4096];
            // SAFETY: proc_pidpath writes at most the buffer's length, which it is told.
            let len = unsafe {
                libc::proc_pidpath(p, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as u32)
            };
            (len > 0).then(|| {
                (
                    p,
                    PathBuf::from(String::from_utf8_lossy(&buf[..len as usize]).to_string()),
                )
            })
        })
        .collect()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn other_processes() -> Vec<(i32, PathBuf)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn executable(path: &Path, text: &str) {
        use std::os::unix::fs::PermissionsExt;
        fs::write(path, text).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// A release directory: a launcher script that says `version`, a library
    /// and its link, a tool, the header stating `abi`.
    fn release_dir(dir: &Path, version: &str, abi: u32) {
        fs::create_dir_all(dir.join("include").join("baseRT")).unwrap();
        executable(
            &dir.join("basert"),
            &format!("#!/bin/sh\n[ \"$1\" = --version ] && echo \"basert {version}\"\n"),
        );
        executable(&dir.join("basert-chat"), "#!/bin/sh\n");
        fs::write(dir.join(format!("libbaseRT.{version}.dylib")), version).unwrap();
        std::os::unix::fs::symlink(
            format!("libbaseRT.{version}.dylib"),
            dir.join("libbaseRT.dylib"),
        )
        .unwrap();
        fs::write(
            dir.join("include").join("baseRT").join("baseRT.h"),
            format!("#pragma once\n#define BASERT_ABI_VERSION {abi}\n"),
        )
        .unwrap();
        fs::write(dir.join(STAMP), format!("v{version}\n")).unwrap();
    }

    fn listed(tag: &str, published: &str, names: &[&str]) -> Value {
        json!({
            "tag_name": tag,
            "published_at": published,
            "draft": false,
            "prerelease": false,
            "assets": names.iter().map(|n| json!({"name": n, "browser_download_url": format!("https://x/{tag}/{n}")})).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn the_listing_gives_this_platforms_releases_newest_first_with_their_abi() {
        let listing = json!([
            listed(
                "v0.4.2",
                "2026-10-27T06:42:10Z",
                &[
                    "basert-engine-abi-1.txt",
                    "basert-engine-macos-arm64-0.4.2.tar.gz",
                    "basert-engine-macos-arm64-0.4.2.tar.gz.sha256",
                ]
            ),
            listed(
                "v0.4.10",
                "2026-12-01T00:00:00Z",
                &[
                    "basert-engine-abi-2.txt",
                    "basert-engine-macos-arm64-0.4.10.tar.gz",
                    "basert-engine-macos-arm64-0.4.10.tar.gz.sha256",
                ]
            ),
            // No checksum: not a release to install.
            listed(
                "v0.5.0",
                "2026-12-02T00:00:00Z",
                &["basert-engine-macos-arm64-0.5.0.tar.gz"]
            ),
            // Another platform only.
            listed(
                "v0.4.11",
                "2026-12-03T00:00:00Z",
                &[
                    "basert-engine-linux-arm64-cuda-0.4.11.tar.gz",
                    "basert-engine-linux-arm64-cuda-0.4.11.tar.gz.sha256",
                ]
            ),
            // A staging tag.
            listed(
                "v0.4.12-staging",
                "2026-12-04T00:00:00Z",
                &[
                    "basert-engine-macos-arm64-v0.4.12-staging.tar.gz",
                    "basert-engine-macos-arm64-v0.4.12-staging.tar.gz.sha256",
                ]
            ),
        ]);
        let found = releases_in(&listing, "macos-arm64");
        let versions: Vec<(&str, Option<u32>)> =
            found.iter().map(|r| (r.version.as_str(), r.abi)).collect();
        assert_eq!(versions, [("0.4.10", Some(2)), ("0.4.2", Some(1))]);
        assert_eq!(found[1].tag, "v0.4.2");
        assert_eq!(
            found[1].tarball_url,
            "https://x/v0.4.2/basert-engine-macos-arm64-0.4.2.tar.gz"
        );
        let mut pre = listed(
            "v0.4.13",
            "2026-12-05T00:00:00Z",
            &[
                "basert-engine-macos-arm64-0.4.13.tar.gz",
                "basert-engine-macos-arm64-0.4.13.tar.gz.sha256",
            ],
        );
        pre["prerelease"] = json!(true);
        assert!(releases_in(&json!([pre]), "macos-arm64").is_empty());
    }

    #[test]
    fn how_a_launcher_was_installed_is_read_from_where_it_is() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join(".basert");
        release_dir(&root.join("versions").join("0.4.3"), "0.4.3", 1);
        assert_eq!(
            installed_as(&root.join("versions").join("0.4.3").join("basert")),
            Installed::Managed {
                root: fs::canonicalize(&root).unwrap(),
                version: "0.4.3".into()
            }
        );
        // Through the root's link, as PATH finds it.
        fs::create_dir_all(root.join("versions")).unwrap();
        std::os::unix::fs::symlink("0.4.3", root.join("versions").join("current")).unwrap();
        std::os::unix::fs::symlink("versions/current/basert", root.join("basert")).unwrap();
        assert!(
            matches!(installed_as(&root.join("basert")), Installed::Managed { version, .. } if version == "0.4.3")
        );

        let flat = tmp.path().join("flat");
        release_dir(&flat, "0.4.2", 1);
        assert_eq!(
            installed_as(&flat.join("basert")),
            Installed::InPlace {
                root: fs::canonicalize(&flat).unwrap()
            }
        );

        let pip = tmp
            .path()
            .join("site-packages")
            .join("baseRT")
            .join("_engine");
        release_dir(&pip, "0.4.2", 1);
        fs::remove_file(pip.join(STAMP)).unwrap();
        match installed_as(&pip.join("basert")) {
            Installed::Other { command, .. } => {
                assert_eq!(command.as_deref(), Some("pip install --upgrade basert"))
            }
            other => panic!("{other:?}"),
        }
        // The package's own file decides, whatever the path says.
        fs::write(pip.join(INSTALLED_BY), "uv: uv tool upgrade basert\n").unwrap();
        fs::write(pip.join(STAMP), "v0.4.2\n").unwrap();
        match installed_as(&pip.join("basert")) {
            Installed::Other { command, how, .. } => {
                assert_eq!(command.as_deref(), Some("uv tool upgrade basert"));
                assert!(how.starts_with("uv installed it"), "{how}");
            }
            other => panic!("{other:?}"),
        }

        let build = tmp.path().join("tools").join("target").join("release");
        fs::create_dir_all(&build).unwrap();
        executable(&build.join("basert"), "#!/bin/sh\n");
        assert!(matches!(
            installed_as(&build.join("basert")),
            Installed::Other { command: None, .. }
        ));
    }

    #[test]
    fn the_files_of_a_release_are_told_from_the_users_own() {
        for name in [
            "basert",
            "basert-chat",
            "basert-benchmark-harness",
            "baseRT_cuda_bench",
            "libbaseRT.dylib",
            "libbaseRT.so.0",
            "baseRT.metallib",
            "baseRT_tensor.metallib",
            "baseRT_cuda_kernels.fatbin",
            "include",
            "LICENSE",
            "NOTICE",
        ] {
            assert!(is_bundle_file(name), "{name}");
        }
        for name in [
            "superfluid",
            "versions",
            ".release",
            "models",
            "notes.txt",
            "baseRT.h",
        ] {
            assert!(!is_bundle_file(name), "{name}");
        }
    }

    #[test]
    fn an_in_place_copy_moves_under_the_layout_and_the_root_links_through_current() {
        let tmp = TempDir::new().unwrap();
        let layout = Layout {
            root: tmp.path().to_path_buf(),
        };
        release_dir(&layout.root, "0.4.2", 1);
        fs::create_dir_all(layout.root.join("superfluid").join("bin")).unwrap();
        fs::write(layout.root.join("notes.txt"), "mine").unwrap();
        let mut said = Vec::new();
        let old = keep_in_place(&layout, &mut |l| said.push(l.to_string())).unwrap();
        assert_eq!(old, "0.4.2");
        let kept = layout.version_dir("0.4.2");
        assert_eq!(launcher_version(&kept.join("basert")).unwrap(), "0.4.2");
        assert_eq!(
            fs::read_link(kept.join("libbaseRT.dylib")).unwrap(),
            Path::new("libbaseRT.0.4.2.dylib")
        );
        assert!(!kept.join("superfluid").exists() && !kept.join("notes.txt").exists());
        // Nothing in the root has changed yet.
        assert!(fs::symlink_metadata(layout.root.join("basert"))
            .unwrap()
            .is_file());

        release_dir(&layout.version_dir("0.4.3"), "0.4.3", 1);
        fs::remove_file(layout.version_dir("0.4.3").join("basert-chat")).unwrap();
        layout.set_current("0.4.3").unwrap();
        layout.set_previous(&old).unwrap();
        let notes = layout.link().unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(
            fs::read_link(layout.root.join("basert")).unwrap(),
            Path::new("versions/current/basert")
        );
        assert_eq!(
            launcher_version(&layout.root.join("basert")).unwrap(),
            "0.4.3"
        );
        assert!(fs::read_link(layout.root.join("include")).is_ok());
        // A file the in-place copy had and the version in use does not: kept
        // under versions/0.4.2, gone from the root.
        assert!(fs::symlink_metadata(layout.root.join("basert-chat")).is_err());
        assert!(kept.join("basert-chat").is_file());
        assert!(fs::symlink_metadata(layout.root.join("libbaseRT.0.4.2.dylib")).is_err());
        assert_eq!(
            fs::read_to_string(layout.root.join("notes.txt")).unwrap(),
            "mine"
        );
        assert!(layout.root.join("superfluid").join("bin").is_dir());

        // Back, then forward: the links follow, a dropped file's link goes.
        layout.set_current("0.4.2").unwrap();
        layout.link().unwrap();
        assert!(fs::read_link(layout.root.join("basert-chat")).is_ok());
        assert_eq!(layout.previous().as_deref(), Some("0.4.3"));
        layout.set_current("0.4.3").unwrap();
        layout.link().unwrap();
        assert!(fs::symlink_metadata(layout.root.join("basert-chat")).is_err());
        assert_eq!(layout.on_disk(), ["0.4.3", "0.4.2"]);
    }

    #[test]
    fn pruning_keeps_the_current_the_previous_a_kept_and_a_served_version() {
        let tmp = TempDir::new().unwrap();
        let layout = Layout {
            root: tmp.path().to_path_buf(),
        };
        for v in ["0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.4.4", "0.4.5"] {
            release_dir(&layout.version_dir(v), v, 1);
        }
        fs::write(layout.version_dir("0.4.1").join(KEEP), "").unwrap();
        // A server on 0.4.2 (this test's own pid stands in for it); one on 0.4.0 that stopped.
        fs::create_dir_all(layout.version_dir("0.4.2").join(SERVING)).unwrap();
        fs::write(
            layout
                .version_dir("0.4.2")
                .join(SERVING)
                .join(std::process::id().to_string()),
            "",
        )
        .unwrap();
        fs::create_dir_all(layout.version_dir("0.4.0").join(SERVING)).unwrap();
        fs::write(
            layout.version_dir("0.4.0").join(SERVING).join("2147483646"),
            "",
        )
        .unwrap();
        layout.set_current("0.4.4").unwrap();
        layout.set_current("0.4.5").unwrap();
        let gone = layout.prune().unwrap();
        assert_eq!(gone, ["0.4.3", "0.4.0"]);
        assert_eq!(layout.on_disk(), ["0.4.5", "0.4.4", "0.4.2", "0.4.1"]);
    }

    #[test]
    fn a_server_on_a_version_is_noted_in_it_and_nowhere_else() {
        let tmp = TempDir::new().unwrap();
        let version = tmp.path().join("versions").join("0.4.3");
        fs::create_dir_all(&version).unwrap();
        note_serving(Some(&version));
        assert!(version
            .join(SERVING)
            .join(std::process::id().to_string())
            .is_file());
        assert!(serving(&version));
        let flat = tmp.path().join("flat");
        fs::create_dir_all(&flat).unwrap();
        note_serving(Some(&flat));
        assert!(!flat.join(SERVING).exists());
    }

    #[test]
    fn a_server_on_a_library_named_by_its_file_is_noted_in_its_version() {
        let tmp = TempDir::new().unwrap();
        let version = tmp.path().join("versions").join("0.4.4");
        fs::create_dir_all(&version).unwrap();
        fs::write(version.join("libbaseRT.0.4.4.dylib"), "").unwrap();
        // What `$BASERT_LIB` may name, through the root's link as well.
        std::os::unix::fs::symlink(
            Path::new("versions/0.4.4/libbaseRT.0.4.4.dylib"),
            tmp.path().join("libbaseRT.dylib"),
        )
        .unwrap();
        note_serving(Some(&tmp.path().join("libbaseRT.dylib")));
        assert!(serving(&version));
    }

    #[test]
    fn a_second_changer_waits_its_turn_and_a_dead_ones_lock_is_taken() {
        let tmp = TempDir::new().unwrap();
        let layout = Layout {
            root: tmp.path().to_path_buf(),
        };
        let held = Lock::take(&layout).unwrap();
        let err = Lock::take(&layout).err().unwrap().to_string();
        assert!(
            err.contains(&format!("pid {}", std::process::id())),
            "{err}"
        );
        drop(held);
        fs::write(layout.versions().join(".lock"), "2147483646\n").unwrap();
        assert!(Lock::take(&layout).is_ok());
    }

    #[test]
    fn github_timestamps_read_as_the_epoch() {
        assert_eq!(epoch_of("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(epoch_of("2026-10-27T06:42:10Z"), Some(1_793_083_330));
        assert_eq!(epoch_of("2000-03-01T00:00:00Z"), Some(951_868_800));
        assert_eq!(epoch_of("soon"), None);
        let at = epoch_of("2026-10-27T06:42:10Z").unwrap();
        assert!(!settled("2026-10-27T06:42:10Z", at + 2 * 86_400));
        assert!(settled("2026-10-27T06:42:10Z", at + 3 * 86_400));
        assert!(settled("", at));
    }

    #[test]
    fn the_notice_names_the_command_that_updates_this_copy() {
        let ahead = format!("{}.{}", VERSION, 1);
        let managed = Installed::Managed {
            root: "/x".into(),
            version: VERSION.into(),
        };
        let brew = Installed::Other {
            exe: "/opt/homebrew/Cellar/basert/1/libexec/basert".into(),
            how: String::new(),
            command: Some("brew upgrade basert".into()),
        };
        let unknown = Installed::Other {
            exe: "/x".into(),
            how: String::new(),
            command: None,
        };
        let newer = Check {
            newest: Some(ahead.clone()),
            ..Check::default()
        };
        let line = newer.line(&managed, None).unwrap();
        assert!(
            line.starts_with(&format!(
                "basert {ahead} is available (this is {VERSION}): `basert self update`"
            )),
            "{line}"
        );
        assert!(newer
            .line(&brew, None)
            .unwrap()
            .contains("`brew upgrade basert`"));
        assert_eq!(newer.line(&unknown, None), None);
        let same = Check {
            newest: Some(VERSION.into()),
            ..Check::default()
        };
        assert_eq!(same.line(&managed, None), None);

        let sf = Check {
            newest: Some(VERSION.into()),
            superfluid: Some("v0.1.9".into()),
            ..Check::default()
        };
        let line = sf.line(&managed, Some("v0.1.5")).unwrap();
        assert!(line.starts_with("superfluid v0.1.9, the server `basert serve` runs, is available (installed: v0.1.5): `basert self update`"), "{line}");
        assert!(sf
            .line(&brew, Some("v0.1.5"))
            .unwrap()
            .contains("`basert superfluid install`"));
        assert_eq!(sf.line(&managed, Some("v0.1.9")), None);
        assert_eq!(sf.line(&managed, None), None);
    }

    #[test]
    fn a_release_states_its_abi_in_its_header() {
        let tmp = TempDir::new().unwrap();
        release_dir(tmp.path(), "0.4.3", 7);
        assert_eq!(abi_in(tmp.path()), Some(7));
        assert_eq!(abi_in(&tmp.path().join("nowhere")), None);
    }

    /// A launcher (a script) at `rel` under `root`, its real path.
    fn launcher_at(root: &Path, rel: &str) -> PathBuf {
        let dir = root.join(rel);
        fs::create_dir_all(&dir).unwrap();
        executable(&dir.join("basert"), "#!/bin/sh\n");
        fs::canonicalize(dir.join("basert")).unwrap()
    }

    #[test]
    fn homebrew_is_run_by_the_brew_whose_cellar_holds_the_copy_unless_pinned() {
        let tmp = TempDir::new().unwrap();
        let exe = launcher_at(tmp.path(), "brew/Cellar/basert/0.4.3/libexec");
        let named = Some(("Homebrew".to_string(), "brew upgrade basert".to_string()));
        match delegate(&exe, named.clone()).unwrap() {
            Delegate::Name { why, .. } => assert!(why.starts_with("there is no "), "{why}"),
            other => panic!("{other:?}"),
        }
        fs::create_dir_all(tmp.path().join("brew/bin")).unwrap();
        executable(&tmp.path().join("brew/bin/brew"), "#!/bin/sh\n");
        let brew = fs::canonicalize(tmp.path().join("brew/bin/brew")).unwrap();
        assert_eq!(
            delegate(&exe, named.clone()),
            Some(Delegate::Run {
                manager: "Homebrew".into(),
                command: "brew upgrade basert".into(),
                argv: vec![
                    brew.display().to_string(),
                    "upgrade".into(),
                    "basert".into()
                ],
                path_first: None,
            })
        );
        // Without the file: the Cellar in its path says Homebrew.
        assert!(matches!(delegate(&exe, None), Some(Delegate::Run { .. })));
        fs::create_dir_all(tmp.path().join("brew/var/homebrew/pinned")).unwrap();
        fs::write(tmp.path().join("brew/var/homebrew/pinned/basert"), "").unwrap();
        match delegate(&exe, named).unwrap() {
            Delegate::Name { why, .. } => {
                assert!(why.contains("`brew unpin basert` first"), "{why}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn npm_is_run_for_a_global_copy_only_and_puts_its_own_bin_first() {
        let tmp = TempDir::new().unwrap();
        let named = Some((
            "npm".to_string(),
            "npm install --global @basecompute/basert@latest".to_string(),
        ));
        let project = launcher_at(
            tmp.path(),
            "app/node_modules/@basecompute/basert-darwin-arm64/engine",
        );
        match delegate(&project, named.clone()).unwrap() {
            Delegate::Name { why, command, .. } => {
                assert!(why.contains("update it in that project"), "{why}");
                assert_eq!(command, "npm install @basecompute/basert@latest");
            }
            other => panic!("{other:?}"),
        }
        let global = launcher_at(
            tmp.path(),
            "node/lib/node_modules/@basecompute/basert-darwin-arm64/engine",
        );
        fs::create_dir_all(tmp.path().join("node/bin")).unwrap();
        executable(&tmp.path().join("node/bin/npm"), "#!/bin/sh\n");
        let bin = fs::canonicalize(tmp.path().join("node/bin")).unwrap();
        match delegate(&global, named).unwrap() {
            Delegate::Run {
                argv, path_first, ..
            } => {
                assert_eq!(argv[0], bin.join("npm").display().to_string());
                assert_eq!(
                    argv[1..],
                    ["install", "--global", "@basecompute/basert@latest"]
                );
                assert_eq!(path_first, Some(bin));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pip_is_named_with_its_environments_python_and_pipx_and_uv_are_told_apart() {
        let tmp = TempDir::new().unwrap();
        let named = Some((
            "pip".to_string(),
            "pip install --upgrade basert".to_string(),
        ));
        let env = tmp.path().join("venv");
        let exe = launcher_at(&env, "lib/python3.12/site-packages/baseRT/_engine");
        fs::create_dir_all(env.join("bin")).unwrap();
        executable(&env.join("bin/python"), "#!/bin/sh\n");
        let python = fs::canonicalize(env.join("bin/python")).unwrap();
        match delegate(&exe, named.clone()).unwrap() {
            Delegate::Name { command, .. } => {
                assert_eq!(
                    command,
                    format!("{} -m pip install --upgrade basert", python.display())
                )
            }
            other => panic!("{other:?}"),
        }
        let pipx = launcher_at(
            tmp.path(),
            ".local/pipx/venvs/basert/lib/python3.12/site-packages/baseRT/_engine",
        );
        let d = delegate(&pipx, named.clone()).unwrap();
        assert_eq!((d.manager(), d.command()), ("pipx", "pipx upgrade basert"));
        let uv = launcher_at(
            tmp.path(),
            ".local/share/uv/tools/basert/lib/python3.12/site-packages/baseRT/_engine",
        );
        let d = delegate(&uv, named).unwrap();
        assert_eq!((d.manager(), d.command()), ("uv", "uv tool upgrade basert"));
    }

    #[test]
    fn cargo_is_run_by_the_cargo_beside_the_copy() {
        let tmp = TempDir::new().unwrap();
        let exe = launcher_at(tmp.path(), ".cargo/bin");
        executable(&tmp.path().join(".cargo/bin/cargo"), "#!/bin/sh\n");
        let bin = fs::canonicalize(tmp.path().join(".cargo/bin")).unwrap();
        assert_eq!(
            delegate(&exe, None),
            Some(Delegate::Run {
                manager: "cargo".into(),
                command: "cargo install basert-cli".into(),
                argv: vec![
                    bin.join("cargo").display().to_string(),
                    "install".into(),
                    "basert-cli".into()
                ],
                path_first: Some(bin),
            })
        );
        // A manager basert does not run is named with the file's own command.
        let d = delegate(&exe, Some(("conda".into(), "conda update basert".into()))).unwrap();
        assert!(
            matches!(&d, Delegate::Name { command, .. } if command == "conda update basert"),
            "{d:?}"
        );
    }

    #[test]
    fn a_server_recorded_on_a_package_engine_is_found_and_a_stopped_ones_record_goes() {
        let tmp = TempDir::new().unwrap();
        let records = tmp.path().join("serving");
        let engine = tmp.path().join("libexec");
        fs::create_dir_all(&records).unwrap();
        fs::create_dir_all(&engine).unwrap();
        let me = std::process::id() as i32;
        fs::write(
            records.join(me.to_string()),
            format!("{}\n", canonical(&engine).display()),
        )
        .unwrap();
        fs::write(
            records.join("2147483646"),
            format!("{}\n", canonical(&engine).display()),
        )
        .unwrap();
        assert_eq!(servers_recorded(&records, &engine), [me]);
        assert!(!records.join("2147483646").exists());
        assert!(servers_recorded(&records, tmp.path()).is_empty());
    }
}
