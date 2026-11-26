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
//! process of its own so that nothing waits on it. A later command updates to
//! the one found (once it is out, and only one a superfluid release that is
//! out serves; `basert self auto-update wait <days>` has a release count as
//! out only that many days after it is published, so that early reports land
//! first): in the foreground for the install script's copy, with a progress
//! bar, then runs itself again on the new release; by its manager's command,
//! pinned to that release, in the background once the command is done, for a
//! copy Homebrew, npm, pipx, uv or cargo installed; and with one line naming
//! the command for a copy basert cannot update. basert's own superfluid is
//! brought to a newer release that is out and serves the engine the same way,
//! in the foreground, when basert itself is the newest (never under a server).
//! `basert self auto-update off` leaves it to a line, and `$BASERT_NO_UPDATE`,
//! `--no-update` (one run) and `$BASERT_NO_UPDATE_CHECK` turn the look, the
//! update and the line off.

use std::ffi::OsString;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
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
/// Set (and not `0`), no command looks for a newer release by itself (nor
/// updates to one, nor says there is one).
pub const NO_CHECK_ENV: &str = "BASERT_NO_UPDATE_CHECK";
/// Set (and not `0`), what [`NO_CHECK_ENV`] does: no look, no update, no
/// line (as `SUPERFLUID_NO_UPDATE` for superfluid).
pub const NO_UPDATE_ENV: &str = "BASERT_NO_UPDATE";
/// The flag that does what [`NO_UPDATE_ENV`] does, for one run, on any command.
pub const NO_UPDATE_FLAG: &str = "--no-update";
/// A number of days: the wait ([`WAIT_KEY`]) for a shell or a service instead
/// of the setting's. One that is not a whole number from 0 to
/// [`MAX_WAIT_DAYS`] is ignored.
pub const WAIT_ENV: &str = "BASERT_UPDATE_WAIT_DAYS";
/// Set by an automatic update on the command it runs again on the new
/// release: the version it came from. That run updates nothing.
pub const UPDATED_FROM_ENV: &str = "BASERT_UPDATED_FROM";
/// The directory `settings.json` is in, instead of `<config>/baseRT`.
pub const CONFIG_DIR_ENV: &str = "BASERT_CONFIG_DIR";
/// The setting in `settings.json` (missing: on).
const AUTO_UPDATE_KEY: &str = "autoUpdate";
/// The setting in `settings.json` for how many days after its publication a
/// release counts as out, for automatic updates and the line saying one is
/// (missing: 0, as soon as it is published).
const WAIT_KEY: &str = "updateWaitDays";
/// The longest wait.
const MAX_WAIT_DAYS: u32 = 365;
/// What turns automatic updates off, as the notices name it.
const OFF_COMMAND: &str = "basert self auto-update off";
/// A release's notes, by its tag, when the listing gives no page for it.
const NOTES: &str = "https://github.com/basecompute/baseRT/releases/tag/";
const SUPERFLUID_NOTES: &str = "https://github.com/basecompute/superfluid/releases/tag/";
/// In `versions/`: the version in use before going back to an older one.
/// Automatic updates leave the older one in use until a newer release is out.
const HELD_BACK: &str = ".held-back";
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
const DAY: u64 = 24 * 60 * 60;
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
    /// Say whether basert updates itself when a newer release is out, and how
    /// long it waits; or turn that on or off, or set the wait.
    ///
    /// The settings are `autoUpdate` and `updateWaitDays` in `settings.json`
    /// in basert's config directory (`$BASERT_CONFIG_DIR` instead). For a
    /// shell: BASERT_UPDATE_WAIT_DAYS=<days> waits that long instead, and
    /// BASERT_NO_UPDATE=1 turns the daily look, updates and the line saying a
    /// release is out off; `--no-update` on any command does so for one run.
    #[command(disable_help_subcommand = true)]
    AutoUpdate {
        /// None says whether automatic updates are on, why not, and the wait.
        #[command(subcommand)]
        change: Option<AutoUpdateChange>,
    },
    /// A package manager's update a launch started in the background: run
    /// once that launch (`--after`) is done and no server runs on the copy.
    #[command(hide = true)]
    RunManager {
        #[arg(long)]
        after: i32,
        /// The copy's directory.
        #[arg(long)]
        dir: PathBuf,
        /// The release it updates to.
        #[arg(long)]
        release: String,
        #[arg(long)]
        manager: String,
        #[arg(long)]
        path_first: Option<PathBuf>,
        /// The manager's command.
        #[arg(last = true, required = true)]
        argv: Vec<String>,
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

/// `basert self auto-update <change>`.
#[derive(Subcommand, Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoUpdateChange {
    /// Update basert by itself when a newer release is out.
    On,
    /// Leave updates to `basert self update`; a newer release is still announced.
    Off,
    /// Wait this many days after a release before it counts as out, for
    /// automatic updates and the line saying one is (0, the default: as soon
    /// as it is published). `basert self update` takes the newest at once.
    Wait {
        /// 0 to 365.
        #[arg(value_parser = clap::value_parser!(u32).range(0..=365))]
        days: u32,
    },
}

pub fn cmd(args: SelfArgs) -> Result<()> {
    let mut say = |line: &str| eprintln!("{line}");
    // A download on a terminal shows its progress.
    superfluid::show_progress(true);
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
        Action::AutoUpdate { change } => auto_update_cmd(change),
        Action::RunManager {
            after,
            dir,
            release,
            manager,
            path_first,
            argv,
        } => run_manager(after, &dir, &release, &manager, path_first, &argv),
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

/// Whether `path` is this user's own to change: they may write it and it is
/// theirs (as root, every path is writable, another user's too).
fn ours(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    // SAFETY: geteuid cannot fail.
    let me = unsafe { libc::geteuid() };
    writable(path) && fs::metadata(path).is_ok_and(|m| m.uid() == me)
}

/// `tool` in the directories of the PATH value `path`, those named in full
/// only: a relative entry (`.`, an empty one, `node_modules/.bin`) is the
/// directory a command happens to run in.
fn find_on(path: Option<&std::ffi::OsStr>, tool: &str) -> Option<PathBuf> {
    std::env::split_paths(path?)
        .filter(|d| d.is_absolute())
        .map(|d| d.join(tool))
        .find(|p| superfluid::is_executable(p))
}

/// `tool args…` with `tool` from PATH, or named when PATH has none.
fn on_path(tool: &str, args: &[&str]) -> Delegate {
    let command = std::iter::once(tool)
        .chain(args.iter().copied())
        .collect::<Vec<_>>()
        .join(" ");
    let found = find_on(std::env::var_os("PATH").as_deref(), tool);
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

/// Whether the path under a global `node_modules` is basert's own package:
/// `@basecompute/basert/…`, its platform package `@basecompute/basert-<p>/…`
/// (beside it, or in its own `node_modules`), and no other package's
/// dependency.
fn global_basert(under: &str) -> bool {
    let parts: Vec<&str> = under.split('/').collect();
    let basert = |i: usize, platform_only: bool| {
        parts.get(i) == Some(&"@basecompute")
            && parts
                .get(i + 1)
                .is_some_and(|n| n.starts_with("basert-") || (!platform_only && *n == "basert"))
    };
    if !basert(0, false) {
        return false;
    }
    let rest = match parts[1] == "basert" && parts.get(2) == Some(&"node_modules") {
        true if basert(3, true) => &parts[5..],
        true => return false,
        false => &parts[2..],
    };
    !rest.contains(&"node_modules")
}

/// `npm install --global` by the npm whose global folder holds the copy; a
/// project's dependency (or another global package's, or pnpm's, yarn's,
/// bun's) is that project's to update.
fn npm(p: &str) -> Delegate {
    let manager = "npm".to_string();
    let command = "npm install --global @basecompute/basert@latest".to_string();
    let global = p
        .find("/lib/node_modules/")
        .filter(|&i| global_basert(&p[i + "/lib/node_modules/".len()..]));
    let Some(i) = global else {
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
    platform_for(std::env::consts::OS, std::env::consts::ARCH)
}

/// One engine per OS and CPU: Metal on Apple silicon, CUDA on Linux arm64
/// (GB10), ROCm on Linux x86-64 (AMD Strix Halo and Strix Point; one bundle
/// carries both chips' kernels).
fn platform_for(os: &str, arch: &str) -> Result<&'static str> {
    match (os, arch) {
        ("macos", "aarch64") => Ok("macos-arm64"),
        ("linux", "aarch64") => Ok("linux-arm64-cuda"),
        ("linux", "x86_64") => Ok("linux-x64-rocm"),
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
    /// The release's page: the listing's `html_url`, else GitHub's by its tag.
    pub notes: String,
    tarball: String,
    tarball_url: String,
    /// The tarball's size in bytes, as the listing gives it.
    tarball_size: Option<u64>,
    checksum_url: String,
}

/// The page of basert release `v`'s notes, when no listing gives it.
fn notes_for(v: &str) -> String {
    format!("{NOTES}v{}", v.trim_start_matches('v'))
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
                notes: r["html_url"]
                    .as_str()
                    .filter(|u| u.starts_with("https://") || u.starts_with("http://"))
                    .map_or_else(|| notes_for(version), str::to_string),
                tarball_url: url(&tarball)?,
                tarball_size: assets.iter().find(|a| a["name"] == tarball.as_str())?["size"]
                    .as_u64(),
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

    /// The version in use before an older one was made the one in use, while
    /// that older one is: automatic updates go no further than it.
    fn held_back(&self) -> Option<String> {
        let v = fs::read_to_string(self.versions().join(HELD_BACK)).ok()?;
        let v = v.trim();
        version_key(v).map(|_| v.to_string())
    }

    /// After `now` was made the one in use in place of `was`: going back
    /// holds automatic updates back to `was` (or what held them before);
    /// going to or past what held them lets them go on.
    fn note_held_back(&self, was: Option<&str>, now: &str) {
        let at = self.versions().join(HELD_BACK);
        let held = self.held_back();
        let newest = [was, held.as_deref()]
            .into_iter()
            .flatten()
            .max_by_key(|v| version_key(v));
        match newest {
            Some(h) if version_key(now) < version_key(h) => {
                let _ = write_atomic(&at, format!("{h}\n").as_bytes());
            }
            _ => {
                let _ = fs::remove_file(&at);
            }
        }
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
fn flat_unused(root: &Path, run: Run) -> Result<()> {
    let pids = servers_on_flat(root);
    if pids.is_empty() {
        return Ok(());
    }
    bail!(
        "a server runs on the engine in {} (pid {}), whose files this would replace: {}",
        root.display(),
        pid_list(&pids),
        run.again()
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
    // On a terminal a progress bar names it; elsewhere, a line.
    let bar = format!("basert {}", release.version);
    let bar = superfluid::progress_shown().then_some(bar.as_str());
    if bar.is_none() {
        say(&format!(
            "downloading basert {} ({}) from {}",
            release.version, release.tarball, release.tarball_url
        ));
    }
    superfluid::download_as(&release.tarball_url, &tarball, bar, release.tarball_size)?;
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

/// Who asked for a change: a command the user ran (`self update`, `self
/// use`), or an automatic update at a launch, which waits for nothing and
/// takes only superfluid releases that count as `out`, chosen from
/// `listing`, the one its look read (`None`: it could not be read).
#[derive(Clone, Copy)]
enum Run<'a> {
    Asked,
    Automatic {
        out: Out,
        listing: Option<&'a Value>,
    },
}

impl Run<'_> {
    /// What a server in the way leaves the user to do.
    fn again(self) -> &'static str {
        match self {
            Run::Asked => "stop it, then run this again",
            Run::Automatic { .. } => {
                "stop it; `basert self update` updates now, and automatic updates try again \
                 after the next check"
            }
        }
    }

    /// The newest superfluid release that serves engine ABI `abi` here, as
    /// this run chooses: an automatic update's from its look, among those out.
    fn newest_superfluid(self, abi: u32) -> Result<Option<superfluid::Release>, String> {
        match self {
            Run::Asked => superfluid::newest_for(abi).map_err(|e| format!("{e:#}")),
            Run::Automatic { out, listing } => {
                let listing = listing.ok_or("the superfluid release listing could not be read")?;
                let target = superfluid::target().map_err(|e| format!("{e:#}"))?;
                Ok(superfluid::newest_where(listing, target, abi, &|r| {
                    out.entry(r)
                }))
            }
        }
    }
}

/// What superfluid needs for an engine, found before anything changes.
enum Pairing {
    /// None is installed: `basert serve` installs one at its first start.
    NotInstalled,
    /// The installed one serves the engine (and stays).
    Serves,
    /// A newer release serves the engine too, but a server uses the
    /// installed one, which serves it and stays until no server does (said).
    Busy,
    /// A release, downloaded and asked, to take the installed one's place.
    Replace {
        installed: superfluid::Install,
        staged: superfluid::Staged,
        /// This install's turn among the installs into that directory,
        /// kept until the swap.
        turn: Option<superfluid::Held>,
        /// The release installed, and whether it serves the engine (then a
        /// server that takes it before the swap keeps it).
        have: String,
        serves: bool,
    },
}

/// What [`apply`] did.
#[derive(Debug, PartialEq, Eq)]
enum Applied {
    /// superfluid stays as it is.
    Unchanged,
    /// This release took the installed one's place.
    Swapped(String),
    /// A server uses the installed one, which stays until none does (said).
    Busy,
}

/// Who uses `installed`, for a message: the servers run from it, else one
/// `basert serve` started (it holds the install wherever its programs run
/// from); `None` when nothing does.
fn users_of(installed: &superfluid::Install) -> Option<String> {
    let pids = running_from(&installed.root);
    if !pids.is_empty() {
        return Some(format!("pid {}", pid_list(&pids)));
    }
    installed
        .in_use()
        .then(|| "one basert serve started".to_string())
}

/// The line for a newer superfluid `tag` that a server using `installed`
/// (`who`) keeps out.
fn stays_while_used(installed: &superfluid::Install, tag: &str, have: &str, who: &str) -> String {
    format!(
        "superfluid {tag} is out, but a server runs from {} ({who}): {have} stays until no \
         server uses it",
        installed.root.display()
    )
}

/// What superfluid needs to serve the engine in `engine`, whose ABI number is
/// `abi`. When the installed one cannot, the newest release that states the
/// number is downloaded and asked, whatever its order (going back to an older
/// basert may take an older superfluid); that no release can is an error. With
/// `forward`, a newer release that serves it is taken even when the installed
/// one can. Nothing is replaced under a running server. A `dry_run` names the
/// release that would come, or what keeps the installed one, and fetches,
/// marks and changes nothing. `run` says who asked, which decides the
/// releases chosen from and whether another install into the directory is
/// waited for.
fn pair_superfluid(
    engine: &Path,
    abi: Option<u32>,
    forward: bool,
    dry_run: bool,
    run: Run,
    say: &mut dyn FnMut(&str),
) -> Result<Pairing> {
    let Some(installed) = superfluid::Install::locate()? else {
        return Ok(Pairing::NotInstalled);
    };
    let turn = match dry_run {
        true => Ok(None),
        false => {
            installed.claim();
            // Installs into it take turns; what is there is asked once this
            // one's turn has come (an automatic update does not wait for it).
            match run {
                Run::Asked => installed.take_turn(say),
                Run::Automatic { .. } => installed.take_turn_now(),
            }
        }
    };
    let home = crate::serve::superfluid_home().ok();
    let have = installed
        .installed_release()
        .unwrap_or_else(|| "of a release it does not say".to_string());
    let turn = match turn {
        Ok(turn) => turn,
        // Another install under way, which an automatic update does not
        // wait for, and the installed one serves the engine: it stays, and
        // this goes on.
        Err(e)
            if matches!(run, Run::Automatic { .. })
                && installed.probe(Some(engine), home.as_deref()).ready.is_ok() =>
        {
            say(&format!("{e:#}: superfluid {have} stays"));
            return Ok(Pairing::Busy);
        }
        Err(e) => return Err(e),
    };
    let ready = installed.probe(Some(engine), home.as_deref()).ready;
    if ready.is_ok() && !forward {
        return Ok(Pairing::Serves);
    }
    let found = match abi {
        Some(abi) => run.newest_superfluid(abi),
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
    match free_to_replace(&installed, &have, &tag, &ready, run, say)? {
        Free::Yes => {}
        Free::Busy => return Ok(Pairing::Busy),
        Free::No => return Ok(Pairing::Serves),
    }
    if dry_run {
        // Not fetched, the release is not asked whether it serves the engine.
        if goes_whole(&installed, None, &have, &tag, &ready, say)? {
            say(&match &ready {
                Ok(()) => format!("superfluid {have} -> {tag} as well"),
                Err(why) => {
                    format!("superfluid {have} -> {tag}: {have} cannot serve this engine ({why})")
                }
            });
        }
        return Ok(Pairing::Serves);
    }
    let staged = installed.stage(&superfluid::Source::GitHub, &release, say)?;
    match staged.fresh.probe(Some(engine), home.as_deref()).ready {
        Ok(()) => {
            let fresh = Some(staged.fresh.root.as_path());
            match goes_whole(&installed, fresh, &have, &tag, &ready, say)? {
                true => Ok(Pairing::Replace {
                    installed,
                    staged,
                    turn,
                    have,
                    serves: ready.is_ok(),
                }),
                false => Ok(Pairing::Serves),
            }
        }
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

/// The pairing carried out: a release staged takes the installed one's
/// place, unless a server took the installed one meanwhile and it serves
/// the engine, which then stays (said).
fn apply(pairing: Pairing, say: &mut dyn FnMut(&str)) -> Result<Applied> {
    match pairing {
        Pairing::Replace {
            installed,
            staged,
            turn,
            have,
            serves,
        } => {
            let tag = staged.tag.clone();
            let swapped = installed.swap_in(staged, say);
            drop(turn);
            match swapped {
                Ok(()) => Ok(Applied::Swapped(tag)),
                Err(e) => match users_of(&installed).filter(|_| serves) {
                    Some(who) => {
                        say(&stays_while_used(&installed, &tag, &have, &who));
                        Ok(Applied::Busy)
                    }
                    None => Err(e),
                },
            }
        }
        Pairing::Busy => Ok(Applied::Busy),
        Pairing::NotInstalled | Pairing::Serves => Ok(Applied::Unchanged),
    }
}

/// Whether an install may give way to a release, as far as is known before
/// it is fetched.
#[derive(Debug, PartialEq, Eq)]
enum Free {
    Yes,
    /// A server uses it, and it serves the engine: it stays (said).
    Busy,
    /// Its directory cannot be written, and it serves the engine: it stays.
    No,
}

/// Whether superfluid `have` at `installed` may give way to release `tag`, as
/// far as is known before it is fetched: no server runs from it (nor holds
/// it, as one `basert serve` started does), and its directory can be
/// written. `ready` is whether `have` serves the engine, or why not: what
/// keeps one that cannot is an error; one that can stays.
fn free_to_replace(
    installed: &superfluid::Install,
    have: &str,
    tag: &str,
    ready: &Result<(), String>,
    run: Run,
    say: &mut dyn FnMut(&str),
) -> Result<Free> {
    if let Some(who) = users_of(installed) {
        if let Err(why) = ready {
            bail!(
                "superfluid {have} at {} cannot serve this engine ({why}) and {tag} can, but a \
                 server runs from it ({who}): {}",
                installed.root.display(),
                run.again()
            );
        }
        say(&stays_while_used(installed, tag, have, &who));
        return Ok(Free::Busy);
    }
    if !installed.writable() {
        if let Err(why) = ready {
            bail!(
                "superfluid {have} at {} cannot serve this engine ({why}), and that directory \
                 cannot be written to put {tag} in its place",
                installed.root.display()
            );
        }
        return Ok(Free::No);
    }
    Ok(Free::Yes)
}

/// Whether `installed`'s directory may go whole for release `tag`: `fresh` is
/// the release staged, `None` before it is fetched. One that holds more than
/// superfluid (a prefix `$BASERT_SUPERFLUID_DIR` names) keeps `have`: said
/// when `have` serves the engine (`ready`), an error when it cannot.
fn goes_whole(
    installed: &superfluid::Install,
    fresh: Option<&Path>,
    have: &str,
    tag: &str,
    ready: &Result<(), String>,
    say: &mut dyn FnMut(&str),
) -> Result<bool> {
    match (installed.replaceable(fresh), ready) {
        (Ok(()), _) => Ok(true),
        (Err(e), Ok(())) => {
            say(&format!("superfluid {tag} is out, but {have} stays: {e:#}"));
            Ok(false)
        }
        (Err(e), Err(why)) => bail!(
            "superfluid {have} at {} cannot serve this engine ({why}) and {tag} can, but {e:#}",
            installed.root.display()
        ),
    }
}

/// For a dry run to basert `v`, not fetched yet, whose engine superfluid
/// cannot be asked about: the newest superfluid release that states its ABI
/// number `abi`, when that is newer than the installed one, or what keeps
/// the installed one in its place.
fn name_newer_superfluid(v: &str, abi: Option<u32>, say: &mut dyn FnMut(&str)) -> Result<()> {
    let (Some(installed), Some(abi)) = (superfluid::Install::locate()?, abi) else {
        return Ok(());
    };
    let have = installed
        .installed_release()
        .unwrap_or_else(|| "of a release it does not say".to_string());
    match superfluid::newest_for(abi) {
        Ok(Some(release)) if superfluid::newer(Some(&release.tag), &have) => {
            // What keeps it is said as for an engine it serves; one it
            // cannot serve makes the update fail instead.
            let (tag, serves) = (&release.tag, Ok(()));
            match free_to_replace(&installed, &have, tag, &serves, Run::Asked, say)? == Free::Yes
                && goes_whole(&installed, None, &have, tag, &serves, say)?
            {
                true => say(&format!(
                    "superfluid {have} -> {tag} as well, the newest that states engine ABI {abi}"
                )),
                false => say(&format!(
                    "basert {v} is made the one in use only if superfluid {have} serves it"
                )),
            }
        }
        Ok(Some(_)) => {}
        Ok(None) => say(&format!(
            "no superfluid release states engine ABI {abi} yet: basert {v} is made the one in \
             use only if superfluid {have} serves it"
        )),
        Err(e) => say(&format!(
            "the superfluid releases were not looked up: {e:#}"
        )),
    }
    Ok(())
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
        // superfluid as well: the newest release that serves this engine.
        let abi = abi_in(&engine_here).or(chosen.abi);
        let pairing = pair_superfluid(&engine_here, abi, true, dry_run, Run::Asked, say)?;
        if dry_run {
            say("dry run: nothing fetched or changed");
            return Ok(());
        }
        return apply(pairing, say).map(|_| ());
    }
    if version.is_none() && version_key(v) < version_key(&here) {
        say(&format!(
            "basert {here} is in use, newer than the newest release {v}: nothing is changed (`--version {v}` goes to it)"
        ));
        return Ok(());
    }
    let on_disk = layout.launcher(v).is_file();
    say(&format!(
        "basert {here} -> {v} (published {}{}) · release notes: {}",
        chosen.published.split('T').next().unwrap_or("?"),
        if on_disk { ", already here" } else { "" },
        chosen.notes
    ));
    if dry_run {
        // superfluid is asked about an engine only once it is on disk.
        match on_disk {
            true => {
                let engine = layout.version_dir(v);
                pair_superfluid(
                    &engine,
                    abi_in(&engine).or(chosen.abi),
                    true,
                    true,
                    Run::Asked,
                    say,
                )?;
            }
            false => name_newer_superfluid(v, chosen.abi, say)?,
        }
        say("dry run: nothing fetched or changed");
        return Ok(());
    }
    update_to(&layout, in_place, chosen, &here, Run::Asked, say)?;
    announce(&layout, v, say);
    Ok(())
}

/// The update of `layout` from `here` to `chosen`, once the layout is
/// locked: the in-place copy kept under it (`in_place`), the release placed,
/// superfluid brought to one that serves its engine (as `run` chooses and
/// waits), then the switch. Until the switch, a Ctrl-C an automatic update
/// caught stops it with nothing changed.
fn update_to(
    layout: &Layout,
    in_place: bool,
    chosen: &Release,
    here: &str,
    run: Run,
    say: &mut dyn FnMut(&str),
) -> Result<()> {
    let v = chosen.version.as_str();
    let kept = match in_place {
        true => {
            flat_unused(&layout.root, run)?;
            Some(keep_in_place(layout, say)?)
        }
        false => None,
    };
    let placed = place(layout, chosen, say)?;
    let engine = layout.version_dir(v);
    let unplace = || {
        if placed {
            let _ = fs::remove_dir_all(&engine);
        }
    };
    // An automatic update's failure line says what stays itself.
    let stays = |e: anyhow::Error| match run {
        Run::Asked => e.context(format!(
            "basert {v} was not made the one in use; {here} stays"
        )),
        Run::Automatic { .. } => e,
    };
    let paired = match not_interrupted() {
        Ok(()) => pair_superfluid(
            &engine,
            abi_in(&engine).or(chosen.abi),
            true,
            false,
            run,
            say,
        ),
        Err(e) => Err(e),
    };
    let applied = paired
        .and_then(|p| not_interrupted().map(|()| p))
        .and_then(|p| apply(p, say));
    if let Err(e) = applied {
        unplace();
        return Err(stays(e));
    }
    switch(layout, v, &chosen.tag, kept.as_deref(), say)
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
    let was = kept.map(str::to_string).or_else(|| layout.current());
    layout.set_current(v)?;
    if let Some(old) = kept.filter(|old| *old != v) {
        layout.set_previous(old)?;
    }
    layout.note_held_back(was.as_deref(), v);
    layout.stamp(tag)?;
    for note in layout.link()? {
        say(&note);
    }
    match layout.prune() {
        Ok(gone) if !gone.is_empty() => say(&format!("removed {}", gone.join(", "))),
        Ok(_) => {}
        Err(e) => say(&format!(
            "an older version was not removed ({e:#}); the next update tries again"
        )),
    }
    Ok(())
}

/// What a command that switched to `v` says after: which is in use (on
/// stdout), the way back, and what automatic updates do now.
fn announce(layout: &Layout, v: &str, say: &mut dyn FnMut(&str)) {
    println!(
        "basert {v} is in use ({})",
        layout.versions().join("current").display()
    );
    if let Some(prev) = layout.previous() {
        say(&format!(
            "{prev} stays installed: `basert self use {prev}` goes back to it"
        ));
    }
    if let Some(held) = layout.held_back() {
        say(&format!(
            "automatic updates leave {v} in use until a release newer than {held} is out \
             (`basert self update` goes forward now)"
        ));
    }
    say("a server already running keeps the engine it started with; starting it again uses this one");
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
    pair_superfluid(&engine, abi_in(&engine), false, false, Run::Asked, say)
        .and_then(|p| apply(p, say))
        .with_context(|| format!("basert {v} was not made the one in use"))?;
    let tag = fs::read_to_string(engine.join(STAMP))
        .map(|t| t.trim().to_string())
        .unwrap_or_else(|_| format!("v{v}"));
    switch(&layout, v, &tag, None, say)?;
    announce(&layout, v, say);
    Ok(())
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
            flat_unused(&layout.root, Run::Asked)?;
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
        switch(&layout, &v, &tag, kept.as_deref(), say)?;
        announce(&layout, &v, say);
        return Ok(());
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
    switch(&layout, &v, &tag, kept.as_deref(), say)?;
    announce(&layout, &v, say);
    Ok(())
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

/// A release a look found, and from when it is out: what the wait is counted
/// from.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Listed {
    version: String,
    /// Its release notes' page, when the listing gave one.
    notes: Option<String>,
    /// Seconds since the epoch: for basert, the later of its publication and
    /// the first of a superfluid release that serves its engine; for
    /// superfluid, its publication.
    out_at: u64,
}

impl Listed {
    fn to_json(&self) -> Value {
        json!({ "version": self.version, "notes": self.notes, "out_at": self.out_at })
    }

    fn from_json(v: &Value) -> Option<Listed> {
        Some(Listed {
            version: v["version"].as_str()?.to_string(),
            notes: v["notes"].as_str().map(str::to_string),
            out_at: v["out_at"].as_u64().unwrap_or(0),
        })
    }
}

/// What the last look at the release listings found, kept in the state
/// directory for a later launch, and what launches did with it. A launch
/// applies the wait it has to what was found, so that a wait changed since
/// the look counts at once.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Check {
    checked_at: u64,
    /// basert's releases whose engine a superfluid release serves, newest first.
    releases: Vec<Listed>,
    /// superfluid's releases that serve the engine beside this basert, newest first.
    superfluids: Vec<Listed>,
    notified_at: u64,
    /// The release an automatic update failed to go to, and when: not tried
    /// again until the next look.
    failed: Option<String>,
    failed_at: u64,
    /// The same for superfluid, when basert was the newest.
    superfluid_failed: Option<String>,
    superfluid_failed_at: u64,
    /// The release a package manager's update was started for in the background.
    started: Option<String>,
}

impl Check {
    fn file() -> Option<PathBuf> {
        crate::serve::state_dir()
            .ok()
            .map(|d| d.join("update-check.json"))
    }

    fn read(path: &Path) -> Option<Check> {
        let v: Value = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
        let text = |k: &str| v[k].as_str().map(str::to_string);
        let listed = |k: &str| -> Vec<Listed> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(Listed::from_json).collect())
                .unwrap_or_default()
        };
        Some(Check {
            checked_at: v["checked_at"].as_u64().unwrap_or(0),
            releases: listed("releases"),
            superfluids: listed("superfluid_releases"),
            notified_at: v["notified_at"].as_u64().unwrap_or(0),
            failed: text("failed"),
            failed_at: v["failed_at"].as_u64().unwrap_or(0),
            superfluid_failed: text("superfluid_failed"),
            superfluid_failed_at: v["superfluid_failed_at"].as_u64().unwrap_or(0),
            started: text("started"),
        })
    }

    fn write(&self, path: &Path) -> Result<()> {
        let listed = |l: &[Listed]| l.iter().map(Listed::to_json).collect::<Vec<_>>();
        let v = json!({
            "checked_at": self.checked_at,
            "releases": listed(&self.releases),
            "superfluid_releases": listed(&self.superfluids),
            "notified_at": self.notified_at,
            "failed": self.failed,
            "failed_at": self.failed_at,
            "superfluid_failed": self.superfluid_failed,
            "superfluid_failed_at": self.superfluid_failed_at,
            "started": self.started,
        });
        write_atomic(path, format!("{v:#}\n").as_bytes())
    }

    /// The newest basert this found that is `out`, when newer than this one.
    fn newer(&self, out: Out) -> Option<&str> {
        self.releases
            .iter()
            .find(|r| out.since(r.out_at))
            .map(|r| r.version.as_str())
            .filter(|n| version_key(n) > version_key(VERSION))
    }

    /// The newest superfluid this found that is `out`, when newer than
    /// `have`, the one installed.
    fn newer_superfluid(&self, have: &str, out: Out) -> Option<&str> {
        self.superfluids
            .iter()
            .find(|r| out.since(r.out_at))
            .map(|r| r.version.as_str())
            .filter(|sf| superfluid::newer(Some(sf), have))
    }

    /// The newer basert this found that is `out`, unless an automatic update
    /// to it failed since the last look.
    fn due(&self, out: Out) -> Option<&str> {
        let n = self.newer(out)?;
        let failed = self.failed.as_deref() == Some(n) && self.failed_at >= self.checked_at;
        (!failed).then_some(n)
    }

    /// The newer superfluid this found than `have` that is `out`, unless an
    /// automatic update to it failed since the last look.
    fn superfluid_due(&self, have: &str, out: Out) -> Option<&str> {
        let sf = self.newer_superfluid(have, out)?;
        let failed = self.superfluid_failed.as_deref() == Some(sf)
            && self.superfluid_failed_at >= self.checked_at;
        (!failed).then_some(sf)
    }

    /// When basert `n` is out, as this found it.
    fn out_at(&self, n: &str) -> Option<u64> {
        self.releases
            .iter()
            .find(|r| r.version == n)
            .map(|r| r.out_at)
    }

    /// The page of the release's notes.
    fn notes_of(&self, n: &str) -> String {
        self.releases
            .iter()
            .find(|r| r.version == n)
            .and_then(|r| r.notes.clone())
            .unwrap_or_else(|| notes_for(n))
    }

    /// The line a launch prints for what this found that is `out`, if
    /// anything: `installed` is how this basert was installed,
    /// `superfluid_have` the release of the superfluid beside it, `auto`
    /// whether automatic updates are on (then the line says how to turn them
    /// off). Under a wait, the release named is the one the wait let through,
    /// and the command installs the newest, which may be a younger one.
    fn line(
        &self,
        installed: &Installed,
        superfluid_have: Option<&str>,
        auto: bool,
        out: Out,
    ) -> Option<String> {
        let ours = matches!(
            installed,
            Installed::Managed { .. } | Installed::InPlace { .. }
        );
        let off = match auto {
            true => format!(" · turn off: {OFF_COMMAND}"),
            false => String::new(),
        };
        let installs = match out.wait {
            0 => "installs it",
            _ => "installs the newest",
        };
        if let Some(n) = self.newer(out) {
            let command = match installed {
                Installed::Other { command, .. } => command.as_deref()?,
                _ => "basert self update",
            };
            return Some(format!(
                "basert {n} is available (this is {VERSION}): `{command}` {installs} · \
                 release notes: {}{off}",
                self.notes_of(n)
            ));
        }
        let have = superfluid_have?;
        let sf = self.newer_superfluid(have, out)?;
        let command = if ours {
            "basert self update"
        } else {
            "basert superfluid install"
        };
        Some(format!(
            "superfluid {sf}, the server `basert serve` runs, is available (installed: {have}): \
             `{command}` {installs} · release notes: {SUPERFLUID_NOTES}v{}{off}",
            sf.trim_start_matches('v')
        ))
    }
}

/// `bytes` written to `path` in one rename, so that a reader finds the old
/// file or the new one, never part of one.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let written = fs::write(&tmp, bytes).and_then(|()| fs::rename(&tmp, path));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written.with_context(|| format!("writing {}", path.display()))
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

/// Which releases count as out at `now`: those published at least `wait`
/// seconds before (one whose time cannot be read, at once; one published
/// after `now`, not yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Out {
    now: u64,
    wait: u64,
}

impl Out {
    /// Whether a release published at `at` (seconds since the epoch) is out.
    fn since(self, at: u64) -> bool {
        at.saturating_add(self.wait) <= self.now
    }

    /// [`Out::since`] for a timestamp of GitHub's.
    fn stamp(self, stamp: &str) -> bool {
        self.since(published_at(stamp))
    }

    /// [`Out::since`] for a release listing's entry.
    fn entry(self, r: &Value) -> bool {
        self.stamp(r["published_at"].as_str().unwrap_or_default())
    }
}

/// When a release was published, from its timestamp: one that cannot be
/// read counts as long ago.
fn published_at(stamp: &str) -> u64 {
    epoch_of(stamp).unwrap_or(0)
}

/// What the listings say now: basert's releases, newest first, and
/// superfluid's listing when it could be read.
struct Found {
    releases: Vec<Release>,
    superfluid: Option<Value>,
}

fn look() -> Result<Found> {
    let releases = releases_in(&listing()?, platform()?);
    Ok(Found {
        releases,
        superfluid: superfluid::releases().ok(),
    })
}

impl Found {
    /// From when `r` counts as out: the later of its publication and the
    /// first of a superfluid release that serves its engine; `None` when no
    /// superfluid release serves it. Unknown (superfluid's listing could not
    /// be read, or `r` states no ABI number) is served.
    fn out_at(&self, r: &Release) -> Option<u64> {
        let published = published_at(&r.published);
        let (Some(abi), Some(listing), Ok(target)) =
            (r.abi, &self.superfluid, superfluid::target())
        else {
            return Some(published);
        };
        let first = superfluid::serving(listing, target, abi)
            .iter()
            .map(|(_, at)| published_at(at))
            .min()?;
        Some(published.max(first))
    }

    /// The newest release a `self update` would go to that a superfluid
    /// release serves; with `out`, the newest that is out by then.
    fn newest(&self, out: Option<Out>) -> Option<&Release> {
        self.releases.iter().find(|r| {
            self.out_at(r)
                .is_some_and(|at| out.is_none_or(|out| out.since(at)))
        })
    }

    /// The superfluid releases that serve this basert's engine, newest first.
    fn superfluids(&self) -> Vec<Listed> {
        let (Some(listing), Ok(target)) = (&self.superfluid, superfluid::target()) else {
            return Vec::new();
        };
        superfluid::serving(listing, target, superfluid::ENGINE_ABI)
            .into_iter()
            .map(|(r, at)| Listed {
                version: r.tag,
                notes: None,
                out_at: published_at(&at),
            })
            .collect()
    }

    /// The newest superfluid release that serves this basert's engine; with
    /// `out`, the newest that is out by then.
    fn newest_superfluid(&self, out: Option<Out>) -> Option<String> {
        self.superfluids()
            .into_iter()
            .find(|r| out.is_none_or(|out| out.since(r.out_at)))
            .map(|r| r.version)
    }

    /// This recorded in the check file at `path`. A look a launch started
    /// (`claimed`) keeps the time the launch claimed it at, so that an
    /// automatic update that failed after the claim is not tried again
    /// before the next one.
    fn record(&self, path: &Path, claimed: bool) {
        let now = now();
        let mut check = Check::read(path).unwrap_or_default();
        let fresh_claim =
            check.checked_at <= now && now.saturating_sub(check.checked_at) < CHECK_EVERY.as_secs();
        if !(claimed && fresh_claim) {
            check.checked_at = now;
        }
        check.releases = self
            .releases
            .iter()
            .filter_map(|r| {
                Some(Listed {
                    version: r.version.clone(),
                    notes: Some(r.notes.clone()),
                    out_at: self.out_at(r)?,
                })
            })
            .collect();
        check.superfluids = self.superfluids();
        let _ = check.write(path);
    }
}

/// `self check`.
fn check() -> Result<()> {
    let found = look()?;
    if let Some(path) = Check::file() {
        found.record(&path, false);
    }
    let env = |k: &str| std::env::var_os(k);
    let wait = update_wait(&env, settings_file().as_deref());
    let out = wait.out(now());
    let installed = installed_as(&this_exe()?);
    let command = match &installed {
        Installed::Other { command, .. } => command.clone(),
        _ => Some("basert self update".to_string()),
    };
    match found.releases.first() {
        Some(r) => {
            let newer = version_key(&r.version) > version_key(VERSION);
            let served = found.out_at(r).is_some();
            let after = match (newer, served) {
                (true, true) => match command {
                    Some(c) => format!(": `{c}` installs it · release notes: {}", r.notes),
                    None => format!(" · release notes: {}", r.notes),
                },
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
            // What an automatic update would take instead, under the wait.
            let auto = found
                .newest(Some(out))
                .map(|a| a.version.as_str())
                .filter(|a| version_key(a) > version_key(VERSION));
            if newer && served && auto != Some(r.version.as_str()) {
                println!(
                    "an automatic update takes {} ({})",
                    auto.unwrap_or("nothing yet"),
                    wait.rule()
                );
            }
        }
        None => {
            println!("basert {VERSION} is this one; the listing names no release for this machine")
        }
    }
    let newest_sf = found.newest_superfluid(None);
    let auto_sf = found.newest_superfluid(Some(out));
    let where_sf = match superfluid::Install::locate()? {
        Some(i) => format!(
            "superfluid {} at {}",
            i.installed_release()
                .unwrap_or_else(|| "of a release it does not say".to_string()),
            i.root.display()
        ),
        None => "superfluid: not installed".to_string(),
    };
    let differs = match newest_sf.is_some() && auto_sf != newest_sf {
        true => format!(
            "; an automatic update takes {} ({})",
            auto_sf.as_deref().unwrap_or("none yet"),
            wait.rule()
        ),
        false => String::new(),
    };
    println!(
        "{where_sf}; the newest release that serves this engine is {}{differs}",
        newest_sf.as_deref().unwrap_or("not known")
    );
    Ok(())
}

/// `self check --record`: the look a launch started, recorded for the notice.
fn record() {
    let Some(path) = Check::file() else { return };
    // A failed look is tried again a day later: the launch that started this
    // claimed the day.
    if let Ok(found) = look() {
        found.record(&path, true);
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

/// Whether an environment variable's value turns its switch on: set, not
/// empty and not `0` (nor `false`).
fn set_on(value: Option<OsString>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0" && v != "false")
}

// --- automatic updates: the setting and the switches -------------------------

/// `<config>/baseRT/settings.json`, or in `$BASERT_CONFIG_DIR`.
fn settings_file() -> Option<PathBuf> {
    let dir = match std::env::var_os(CONFIG_DIR_ENV).filter(|d| !d.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => dirs::config_dir()?.join("baseRT"),
    };
    Some(dir.join("settings.json"))
}

/// The automatic-update setting.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Setting {
    On,
    Off,
    /// The settings file is there but does not say on or off, or holds a
    /// wait that is not one: off, an opt-out written by hand is never taken
    /// for on. Why, for a message.
    Unreadable(String),
}

/// The settings file at `path`, as JSON: `None` when there is none.
fn read_settings(path: &Path) -> Result<Option<Value>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{} could not be read: {e}", path.display())),
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(v) if v.is_object() => Ok(Some(v)),
        Ok(_) => Err(format!("{} is not a JSON object", path.display())),
        Err(e) => Err(format!("{} is not JSON: {e}", path.display())),
    }
}

/// `days` as a wait: a whole number from 0 to [`MAX_WAIT_DAYS`].
fn wait_days(days: u64) -> Option<u32> {
    u32::try_from(days).ok().filter(|d| *d <= MAX_WAIT_DAYS)
}

/// The wait in the settings `v`, read from `path`: 0 when it is missing;
/// why, when it is not a wait.
fn wait_in(v: &Value, path: &Path) -> Result<u32, String> {
    match &v[WAIT_KEY] {
        Value::Null => Ok(0),
        other => other.as_u64().and_then(wait_days).ok_or_else(|| {
            format!(
                "{WAIT_KEY} in {} is {other}, not a whole number of days from 0 to \
                 {MAX_WAIT_DAYS}",
                path.display()
            )
        }),
    }
}

/// The automatic-update setting in the settings file at `path`: on when the
/// file or the setting is missing, else as it says.
fn auto_update_setting(path: Option<&Path>) -> Setting {
    let Some(path) = path else {
        return Setting::On;
    };
    match read_settings(path) {
        Ok(None) => Setting::On,
        Ok(Some(v)) => {
            if let Err(why) = wait_in(&v, path) {
                return Setting::Unreadable(why);
            }
            match &v[AUTO_UPDATE_KEY] {
                Value::Null => Setting::On,
                Value::Bool(true) => Setting::On,
                Value::Bool(false) => Setting::Off,
                other => Setting::Unreadable(format!(
                    "{AUTO_UPDATE_KEY} in {} is {other}, not true or false",
                    path.display()
                )),
            }
        }
        Err(why) => Setting::Unreadable(why),
    }
}

/// The wait the settings file at `path` sets: 0 when the file or the
/// setting is missing; why, when it cannot be read.
fn wait_setting(path: Option<&Path>) -> Result<u32, String> {
    let Some(path) = path else {
        return Ok(0);
    };
    match read_settings(path)? {
        None => Ok(0),
        Some(v) => wait_in(&v, path),
    }
}

/// The wait `$BASERT_UPDATE_WAIT_DAYS` holds, when it holds one.
fn wait_from_env(value: Option<OsString>) -> Option<u32> {
    let text = value?.into_string().ok()?;
    text.trim().parse::<u64>().ok().and_then(wait_days)
}

/// How many days after its publication a release counts as out, for
/// automatic updates and the line saying one is, and whether
/// `$BASERT_UPDATE_WAIT_DAYS` says so rather than the setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Wait {
    days: u32,
    from_env: bool,
}

impl Wait {
    /// What counts as out at `now`.
    fn out(self, now: u64) -> Out {
        Out {
            now,
            wait: u64::from(self.days) * DAY,
        }
    }

    /// `as soon as it is out`, `once it has been out 3 days`.
    fn when(self) -> String {
        match self.days {
            0 => "as soon as it is out".to_string(),
            1 => "once it has been out 1 day".to_string(),
            d => format!("once it has been out {d} days"),
        }
    }

    /// The rule as one sentence: what `self auto-update wait` says.
    fn rule(self) -> String {
        match self.days {
            0 => "automatic updates install a release as soon as it is out".to_string(),
            1 => "automatic updates wait 1 day after a release".to_string(),
            d => format!("automatic updates wait {d} days after a release"),
        }
    }

    /// ` out 3 days`, after "release" in a message; empty for no wait.
    fn out_for(self) -> String {
        match self.days {
            0 => String::new(),
            1 => " out 1 day".to_string(),
            d => format!(" out {d} days"),
        }
    }
}

/// The wait: `$BASERT_UPDATE_WAIT_DAYS`'s when it holds one, else the
/// setting's in the settings file at `path` (none when the file cannot be
/// read: automatic updates are off then, and a newer release is said as
/// soon as it is out).
fn update_wait(env: &dyn Fn(&str) -> Option<OsString>, path: Option<&Path>) -> Wait {
    match wait_from_env(env(WAIT_ENV)) {
        Some(days) => Wait {
            days,
            from_env: true,
        },
        None => Wait {
            days: wait_setting(path).unwrap_or(0),
            from_env: false,
        },
    }
}

/// The settings file at `path` with `key` set to `value`, the rest of it as
/// it was. One that is there but cannot be read as JSON is left alone:
/// writing it over would lose what else it holds.
fn set_setting(path: &Path, key: &str, value: Value) -> Result<()> {
    let mut v = match read_settings(path) {
        Ok(v) => v.unwrap_or_else(|| json!({})),
        Err(why) => bail!("{why}: fix or remove it, then run this again"),
    };
    v[key] = value;
    write_atomic(path, format!("{v:#}\n").as_bytes())
}

/// The settings file at `path` with the automatic-update setting `on`.
fn set_auto_update_setting(path: &Path, on: bool) -> Result<()> {
    set_setting(path, AUTO_UPDATE_KEY, Value::Bool(on))
}

/// The settings file at `path` with the wait `days`.
fn set_wait_setting(path: &Path, days: u32) -> Result<()> {
    set_setting(path, WAIT_KEY, json!(days))
}

/// The variable that turns the look, automatic updates and the line saying
/// a release is out all off, when one does.
fn silenced(env: &dyn Fn(&str) -> Option<OsString>) -> Option<String> {
    if set_on(env(NO_UPDATE_ENV)) {
        return Some(format!("{NO_UPDATE_ENV} is set"));
    }
    set_on(env(NO_CHECK_ENV)).then(|| format!("{NO_CHECK_ENV} is set"))
}

/// Why automatic updates are off for every command (a variable, the
/// setting), or `None` when they are on: what `self auto-update` says.
fn auto_update_off(env: &dyn Fn(&str) -> Option<OsString>, setting: &Setting) -> Option<String> {
    if let Some(why) = silenced(env) {
        return Some(why);
    }
    if set_on(env("CI")) {
        return Some("CI is set".to_string());
    }
    match setting {
        Setting::On => None,
        Setting::Off => Some(OFF_COMMAND.to_string()),
        Setting::Unreadable(why) => Some(why.clone()),
    }
}

/// What `self auto-update` says: whether automatic updates are on, or why
/// not, then the wait, with `$BASERT_UPDATE_WAIT_DAYS` when it sets it or
/// holds no wait (as `superfluid self auto-update` says it).
fn state_line(env: &dyn Fn(&str) -> Option<OsString>, setting: &Setting, wait: Wait) -> String {
    let on = match auto_update_off(env, setting) {
        None => "automatic updates: on".to_string(),
        Some(why) => format!("automatic updates: off ({why})"),
    };
    let given = env(WAIT_ENV).filter(|v| !v.is_empty());
    let by = match (wait.from_env, given) {
        (true, _) => format!(" ({WAIT_ENV} is set)"),
        (false, Some(v)) => format!(
            " ({WAIT_ENV}={} is not a whole number of days from 0 to {MAX_WAIT_DAYS}, so \
             the setting is used)",
            v.to_string_lossy()
        ),
        (false, None) => String::new(),
    };
    format!("{on} · a release installs {}{by}", wait.when())
}

/// `self auto-update [on|off|wait <days>]`.
fn auto_update_cmd(change: Option<AutoUpdateChange>) -> Result<()> {
    let path = settings_file();
    let env = |k: &str| std::env::var_os(k);
    if let Some(change) = change {
        let file = path
            .as_deref()
            .context("no config directory to keep the setting in (set BASERT_CONFIG_DIR)")?;
        match change {
            AutoUpdateChange::On => set_auto_update_setting(file, true)?,
            AutoUpdateChange::Off => set_auto_update_setting(file, false)?,
            AutoUpdateChange::Wait { days } => {
                set_wait_setting(file, days)?;
                let set = Wait {
                    days,
                    from_env: false,
                };
                println!("{}", set.rule());
                let wait = update_wait(&env, Some(file));
                if wait.from_env {
                    eprintln!(
                        "{WAIT_ENV} is set, which this shell goes by instead: {}",
                        wait.rule()
                    );
                }
                return Ok(());
            }
        }
    }
    let setting = auto_update_setting(path.as_deref());
    let wait = update_wait(&env, path.as_deref());
    println!("{}", state_line(&env, &setting, wait));
    Ok(())
}

/// `argv` without `--no-update` (before a `--`), and whether it was there:
/// the flag is any command's, so it is taken out before the command sees it.
pub fn strip_no_update(argv: Vec<OsString>) -> (Vec<OsString>, bool) {
    let mut found = false;
    let mut ended = false;
    let mut kept = Vec::with_capacity(argv.len());
    for (i, arg) in argv.into_iter().enumerate() {
        if i > 0 && !ended {
            if arg == "--" {
                ended = true;
            } else if arg == NO_UPDATE_FLAG {
                found = true;
                continue;
            }
        }
        kept.push(arg);
    }
    (kept, found)
}

/// Whether the command `args` (without the program) only tells something,
/// so that updating first would only slow it down or change what it tells:
/// help, versions, `self` (its own updates), `superfluid`, listings,
/// inspections and anything asked for as JSON.
fn informational(args: &[OsString]) -> bool {
    let first = args.first().map(|a| a.to_string_lossy());
    let tells = match first.as_deref() {
        None => true,
        Some(f) => {
            f.starts_with('-')
                || matches!(
                    f,
                    "self"
                        | "help"
                        | "superfluid"
                        | "list"
                        | "inspect"
                        | "verify"
                        | "fingerprint"
                        | "inventory"
                        | "catalog-manifest"
                        | "catalog-scan"
                        | "status"
                )
        }
    };
    tells
        || args
            .iter()
            .take_while(|a| *a != "--")
            .map(|a| a.to_string_lossy())
            .any(|a| {
                matches!(a.as_ref(), "--json" | "-h" | "--help" | "-V" | "--version")
                    || a.starts_with("--json=")
            })
}

/// Why this launch updates nothing by itself, or `None` when it may: `args`
/// is the command, `terminal` whether stderr is one, `as_root` whether it
/// runs as root, `flag` whether `--no-update` was given, `env` the
/// environment, `setting` the setting.
fn auto_update_blocked(
    args: &[OsString],
    terminal: bool,
    as_root: bool,
    flag: bool,
    env: &dyn Fn(&str) -> Option<OsString>,
    setting: &Setting,
) -> Option<String> {
    if !terminal {
        // An app or a service started it: nobody sees an update, or wants one.
        return Some("stderr is not a terminal".to_string());
    }
    if as_root {
        // `sudo basert …` over a user's install would leave it root's.
        return Some("it runs as root: the install's owner updates it".to_string());
    }
    if env(UPDATED_FROM_ENV).is_some() {
        return Some(format!(
            "{UPDATED_FROM_ENV} is set: this run follows an update"
        ));
    }
    if flag {
        return Some(format!("{NO_UPDATE_FLAG} was given"));
    }
    if let Some(why) = auto_update_off(env, setting) {
        return Some(why);
    }
    if informational(args) {
        return Some("the command only tells something".to_string());
    }
    None
}

// --- automatic updates: at a launch ------------------------------------------

/// What a launch does before its command, on a terminal: start the day's look
/// for a newer release when it is due; update to one found before, when
/// automatic updates are on (then run the command again on it, which does not
/// return), or, basert being the newest, bring its superfluid to a newer
/// release found; else print the line for it, once a day at most. What counts
/// as found is a release out by the wait. `args` is the command without the
/// program, `no_update` whether `--no-update` was given, which (as
/// `BASERT_NO_UPDATE`) leaves out the look, the update and the line.
pub fn at_launch(args: &[OsString], no_update: bool) {
    let first = args.first().and_then(|a| a.to_str());
    if matches!(
        first,
        None | Some("self" | "-V" | "--version" | "-h" | "--help" | "help")
    ) {
        return;
    }
    let env = |k: &str| std::env::var_os(k);
    let terminal = std::io::stderr().is_terminal();
    if no_update || silenced(&env).is_some() || !terminal || env(UPDATED_FROM_ENV).is_some() {
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
    let settings = settings_file();
    let setting = auto_update_setting(settings.as_deref());
    let auto = auto_update_off(&env, &setting).is_none();
    let wait = update_wait(&env, settings.as_deref());
    let out = wait.out(now);
    // SAFETY: geteuid cannot fail.
    let as_root = unsafe { libc::geteuid() } == 0;
    let blocked = auto_update_blocked(args, terminal, as_root, no_update, &env, &setting);
    let have = superfluid::Install::locate()
        .ok()
        .flatten()
        .and_then(|i| i.installed_release());
    if blocked.is_none() {
        if let Some(n) = check.due(out) {
            let n = n.to_string();
            let acted = match &installed {
                Installed::Managed { .. } | Installed::InPlace { .. } => {
                    update_at_launch(&installed, &n, &check, &path, args, wait)
                }
                Installed::Other { exe, .. } => update_in_background(exe, &n, &check, &path),
            };
            if acted {
                return;
            }
        }
        if let Some(sf) = have.as_deref().and_then(|h| check.superfluid_due(h, out)) {
            let sf = sf.to_string();
            if update_superfluid_at_launch(&installed, &sf, &check, &path, wait) {
                return;
            }
        }
    }
    if now.saturating_sub(check.notified_at) < CHECK_EVERY.as_secs() {
        return;
    }
    if let Some(line) = check.line(&installed, have.as_deref(), auto, out) {
        eprintln!("{line}");
        note_in_check(&path, &check, |c| c.notified_at = now);
    }
}

/// The check file changed by `change`: read again first, since the look may
/// have recorded since `check` was read.
fn note_in_check(path: &Path, check: &Check, change: impl FnOnce(&mut Check)) {
    let mut latest = Check::read(path).unwrap_or_else(|| check.clone());
    change(&mut latest);
    let _ = latest.write(path);
}

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_interrupt(_: libc::c_int) {
    // A second Ctrl-C is not waited on: it stops basert, as it would have.
    if INTERRUPTED.swap(true, Ordering::SeqCst) {
        // SAFETY: signal and raise are async-signal-safe.
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::raise(libc::SIGINT);
        }
    }
}

/// While held, a Ctrl-C stops the automatic update in its next step rather
/// than the command it came before (a second one stops basert). Dropped, the
/// command goes on as if none had come.
struct CatchInterrupt(libc::sighandler_t);

impl CatchInterrupt {
    fn new() -> CatchInterrupt {
        INTERRUPTED.store(false, Ordering::SeqCst);
        let handler = on_interrupt as extern "C" fn(libc::c_int);
        // SAFETY: the handler only stores to an atomic and, the second time,
        // calls async-signal-safe functions.
        CatchInterrupt(unsafe { libc::signal(libc::SIGINT, handler as libc::sighandler_t) })
    }
}

impl Drop for CatchInterrupt {
    fn drop(&mut self) {
        // SAFETY: puts back the disposition `new` replaced.
        unsafe { libc::signal(libc::SIGINT, self.0) };
        // The update is over: what the command downloads next is not stopped.
        INTERRUPTED.store(false, Ordering::SeqCst);
    }
}

/// Whether a Ctrl-C came while an automatic update was under way.
pub(crate) fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

fn not_interrupted() -> Result<()> {
    match interrupted() {
        true => bail!("interrupted"),
        false => Ok(()),
    }
}

/// `work` (a look at a release listing, which a Ctrl-C cannot cut short)
/// in a thread of its own, waited for until it is done or a Ctrl-C comes:
/// then it is left to finish by itself, and this is "interrupted".
pub(crate) fn interruptible<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(done) => return done,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => not_interrupted()?,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("the look at the release listings stopped")
            }
        }
    }
}

/// The automatic update of the install script's copy to the newest release
/// out by the `wait` that a superfluid release out as long serves (`cached`
/// is the one the last look found), in the foreground: the release
/// downloaded and placed, superfluid brought along, `current` turned, then
/// `args` run again on the new release (which does not return). Whether it
/// acted: `false` when it is not this launch's to do (a version run by its
/// own path, one gone back to, a root this user cannot change); when it
/// failed, it said so, and the command runs on this basert.
fn update_at_launch(
    installed: &Installed,
    cached: &str,
    check: &Check,
    path: &Path,
    args: &[OsString],
    wait: Wait,
) -> bool {
    let (layout, in_place) = match installed {
        Installed::Managed { root, version } => {
            let layout = Layout { root: root.clone() };
            // A version run by its own path, not the one in use, is the user's choice.
            if layout.current().as_deref() != Some(version.as_str()) {
                return false;
            }
            (layout, false)
        }
        Installed::InPlace { root } => (Layout { root: root.clone() }, true),
        Installed::Other { .. } => return false,
    };
    // Another user's install (a shared root), or one this user cannot write:
    // the line names the command for its owner.
    let versions = layout.versions();
    if !ours(&layout.root) || (versions.exists() && !ours(&versions)) {
        return false;
    }
    let held = layout.held_back();
    let past_held = |v: &str| {
        held.as_deref()
            .is_none_or(|h| version_key(v) > version_key(h))
    };
    if !past_held(cached) {
        return false;
    }
    // Said before the look, so that the wait for it is the update's.
    eprintln!(
        "Updating basert {VERSION} → {cached} · release notes: {}",
        check.notes_of(cached)
    );
    superfluid::show_progress(true);
    let mut target = cached.to_string();
    let mut say = |line: &str| eprintln!("{line}");
    let done = {
        let _catch = CatchInterrupt::new();
        let target = &mut target;
        (|| -> Result<String> {
            let _lock = Lock::take(&layout)?;
            let found = interruptible(look)?;
            let out = wait.out(now());
            let chosen = found
                .newest(Some(out))
                .filter(|c| version_key(&c.version) > version_key(VERSION) && past_held(&c.version))
                .with_context(|| {
                    format!(
                        "the release listing names no newer release{} that a superfluid release \
                         serves",
                        wait.out_for()
                    )
                })?;
            if chosen.version != *target {
                *target = chosen.version.clone();
                eprintln!(
                    "Updating basert {VERSION} → {} · release notes: {}",
                    chosen.version, chosen.notes
                );
            }
            let run = Run::Automatic {
                out,
                listing: found.superfluid.as_ref(),
            };
            update_to(&layout, in_place, chosen, VERSION, run, &mut say)?;
            Ok(chosen.version.clone())
        })()
        .map_err(|e| match interrupted() {
            true => anyhow::anyhow!("interrupted"),
            false => e,
        })
    };
    // What the command downloads next says a line, as it would have.
    superfluid::show_progress(false);
    let v = match done {
        Ok(v) => v,
        Err(e) => {
            eprintln!("couldn't update basert to {target}: {e:#}; staying on {VERSION}");
            let now = now();
            note_in_check(path, check, |c| {
                c.failed = Some(target.clone());
                c.failed_at = now;
                c.notified_at = now;
            });
            return true;
        }
    };
    eprintln!(
        "basert {v} is in use · `basert self use {VERSION}` goes back · `{OFF_COMMAND}` turns \
         automatic updates off"
    );
    use std::os::unix::process::CommandExt;
    let launcher = layout.launcher(&v);
    let mut again = Command::new(&launcher);
    if let Some(arg0) = std::env::args_os().next() {
        again.arg0(arg0);
    }
    let err = again.args(args).env(UPDATED_FROM_ENV, VERSION).exec();
    eprintln!(
        "basert {v} did not start ({err}): this command runs on {VERSION}; the next one runs on {v}"
    );
    true
}

/// The engine this launch's basert runs on, as `basert serve` finds it:
/// for the versions layout, the one in use (a version run by its own path is
/// the user's choice, `None`).
fn engine_of(installed: &Installed) -> Option<PathBuf> {
    if let Installed::Managed { root, version } = installed {
        let layout = Layout { root: root.clone() };
        if layout.current().as_deref() != Some(version.as_str()) {
            return None;
        }
    }
    match crate::serve::engine_dir() {
        Ok(Some(dir)) => Some(dir),
        Ok(None) => std::env::var_os("BASERT_LIB").map(PathBuf::from),
        Err(_) => None,
    }
}

/// Whether basert may bring `sf` to a newer release by itself: one basert
/// installed (not one `$BASERT_SUPERFLUID_DIR` names that the user put there,
/// nor one in a package's folder), in a directory of this user's that holds
/// superfluid alone, where an install can be made.
fn superfluid_ours(sf: &superfluid::Install) -> bool {
    let place = sf.root.parent().unwrap_or(Path::new("."));
    sf.takes_installs()
        && (sf.origin != superfluid::Origin::Env || sf.owned_by_basert())
        && ours(place)
        && (!sf.root.exists() || ours(&sf.root))
        && sf.writable()
        && sf.replaceable(None).is_ok()
}

/// The automatic update of basert's superfluid to `cached`, the newer
/// release the last look found that serves this engine, when basert itself
/// is the newest: in the foreground, with a progress bar, as basert's own;
/// the command then runs as it is (superfluid is a program of its own).
/// Whether it acted: `false` when it is not this launch's to do (see
/// [`superfluid_ours`], a release gone back from); a server using it leaves
/// it in place, said once a day, until none does; a failure is said, and not
/// tried again before the next look.
fn update_superfluid_at_launch(
    installed: &Installed,
    cached: &str,
    check: &Check,
    path: &Path,
    wait: Wait,
) -> bool {
    let Some(engine) = engine_of(installed) else {
        return false;
    };
    let Ok(Some(sf)) = superfluid::Install::locate() else {
        return false;
    };
    if !superfluid_ours(&sf) {
        return false;
    }
    let Some(have) = sf.installed_release() else {
        return false;
    };
    if !superfluid::newer(Some(cached), &have)
        || sf
            .held_back()
            .is_some_and(|h| !superfluid::newer(Some(cached), &h))
    {
        return false;
    }
    let now = now();
    // Nothing is fetched under a server: the next launch without one updates it.
    if let Some(who) = users_of(&sf) {
        if now.saturating_sub(check.notified_at) >= CHECK_EVERY.as_secs() {
            eprintln!("{}", stays_while_used(&sf, cached, &have, &who));
            note_in_check(path, check, |c| c.notified_at = now);
        }
        return true;
    }
    let abi = abi_in(&engine).unwrap_or(superfluid::ENGINE_ABI);
    let updating = |tag: &str| {
        eprintln!(
            "Updating superfluid {have} → {tag} · release notes: {SUPERFLUID_NOTES}v{}",
            tag.trim_start_matches('v')
        )
    };
    // Said before the look, so that the wait for it is the update's.
    updating(cached);
    superfluid::show_progress(true);
    let mut target = cached.to_string();
    let mut say = |line: &str| eprintln!("{line}");
    let done = {
        let _catch = CatchInterrupt::new();
        let target = &mut target;
        (|| -> Result<Applied> {
            let listing = interruptible(superfluid::releases)?;
            let out = wait.out(now);
            let chosen =
                superfluid::newest_where(&listing, superfluid::target()?, abi, &|r| out.entry(r))
                    .filter(|r| superfluid::newer(Some(&r.tag), &have))
                    .with_context(|| {
                        format!(
                            "the superfluid listing names no release newer than {have}{} that \
                             serves engine ABI {abi}",
                            wait.out_for()
                        )
                    })?;
            if chosen.tag != *target {
                *target = chosen.tag.clone();
                updating(&chosen.tag);
            }
            let run = Run::Automatic {
                out,
                listing: Some(&listing),
            };
            let pairing = pair_superfluid(&engine, Some(abi), true, false, run, &mut say)?;
            not_interrupted()?;
            apply(pairing, &mut say)
        })()
        .map_err(|e| match interrupted() {
            true => anyhow::anyhow!("interrupted"),
            false => e,
        })
    };
    superfluid::show_progress(false);
    let now = self::now();
    match done {
        Ok(Applied::Swapped(tag)) => {
            eprintln!(
                "superfluid {tag} is in use · `basert superfluid install {have}` goes back · \
                 `{OFF_COMMAND}` turns automatic updates off"
            );
            note_in_check(path, check, |c| c.notified_at = now);
        }
        // A server took it meanwhile (said): tried again at the next launch.
        Ok(Applied::Busy) => note_in_check(path, check, |c| c.notified_at = now),
        // What kept it was said: not tried again before the next look.
        Ok(Applied::Unchanged) => note_in_check(path, check, |c| {
            c.superfluid_failed = Some(target.clone());
            c.superfluid_failed_at = now;
            c.notified_at = now;
        }),
        Err(e) => {
            eprintln!("couldn't update superfluid to {target}: {e:#}; staying on {have}");
            note_in_check(path, check, |c| {
                c.superfluid_failed = Some(target.clone());
                c.superfluid_failed_at = now;
                c.notified_at = now;
            });
        }
    }
    true
}

/// What a launch does with a newer release for a copy a package manager
/// installed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ManagerPlan {
    /// The manager's update, run in the background.
    Background,
    /// The line naming its command.
    Notice,
}

/// For a copy `delegate` updates: its manager's update runs in the
/// background when basert runs that manager, the copy's directory is this
/// user's (`ours`), no server runs on it (`busy`), and it was not started
/// for this release `newest` before (`started`).
fn manager_plan(
    delegate: &Delegate,
    ours: bool,
    busy: bool,
    started: Option<&str>,
    newest: &str,
) -> ManagerPlan {
    match delegate {
        Delegate::Run { .. } if ours && !busy && started != Some(newest) => ManagerPlan::Background,
        _ => ManagerPlan::Notice,
    }
}

/// The manager's command `argv` (as [`delegate`] builds it, for the newest
/// release) for release `newest` exactly: the one the look found out by the
/// wait and served, not whatever the registry has now. Homebrew cannot
/// be told a version: its formula's is checked before it runs.
fn pinned(manager: &str, argv: &[String], newest: &str) -> Vec<String> {
    let Some(bin) = argv.first().cloned() else {
        return Vec::new();
    };
    let args: &[&str] = match manager {
        "npm" => &["install", "--global"],
        "pipx" => &["install", "--force"],
        "uv" => &["tool", "install"],
        "cargo" => &["install", "basert-cli", "--version"],
        _ => return argv.to_vec(),
    };
    let release = match manager {
        "npm" => format!("@basecompute/basert@{newest}"),
        "pipx" | "uv" => format!("basert=={newest}"),
        _ => newest.to_string(),
    };
    std::iter::once(bin)
        .chain(args.iter().map(|a| a.to_string()))
        .chain(std::iter::once(release))
        .collect()
}

/// The automatic update of a package manager's copy at `exe` to `newest`: its
/// manager's command, pinned to that release, started in the background to
/// run once this command is done, its output in a log in the state
/// directory. Whether it was started (or another launch is starting it);
/// when not, the launch prints the line naming the command.
fn update_in_background(exe: &Path, newest: &str, check: &Check, path: &Path) -> bool {
    let dir = exe.parent().unwrap_or(Path::new("/")).to_path_buf();
    let Some(plan) = delegate(exe, package_manager(&dir)) else {
        return false;
    };
    let Some(state) = path.parent() else {
        return false;
    };
    // One launch starts it: others starting together find this held, and
    // leave it to that one.
    if fs::create_dir_all(state).is_err() {
        return false;
    }
    let Ok(lock) = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(state.join("auto-update.lock"))
    else {
        return false;
    };
    match superfluid::flock(&lock, libc::LOCK_EX | libc::LOCK_NB) {
        Ok(true) => {}
        Ok(false) => return true,
        Err(_) => return false,
    }
    // Under the lock, what launches before this one did.
    let check = Check::read(path).unwrap_or_else(|| check.clone());
    let busy = match plan.manager() {
        "cargo" => false,
        _ => !servers_on_package(&dir).is_empty(),
    };
    if manager_plan(&plan, ours(&dir), busy, check.started.as_deref(), newest)
        != ManagerPlan::Background
    {
        return false;
    }
    let Delegate::Run {
        manager,
        argv,
        path_first,
        ..
    } = plan
    else {
        return false;
    };
    let argv = pinned(&manager, &argv, newest);
    let log = state.join("auto-update.log");
    let before = check.started.clone();
    let now = now();
    // Noted before it starts, so that no launch after this starts it again.
    note_in_check(path, &check, |c| {
        c.started = Some(newest.to_string());
        c.notified_at = now;
    });
    let started = start_detached(&manager, &argv, path_first, &dir, &log, newest);
    if let Err(e) = started {
        note_in_check(path, &check, |c| c.started = before);
        eprintln!("couldn't update basert to {newest} with {manager}: {e:#}; staying on {VERSION}");
        return false;
    }
    eprintln!(
        "Updating basert {VERSION} → {newest} with {manager} in the background (log: {}) · \
         release notes: {} · turn off: {OFF_COMMAND}",
        log.display(),
        check.notes_of(newest)
    );
    true
}

/// `basert self run-manager` for `argv`, started in a process group of its
/// own that outlives this one (through a shell that exits at once), in the
/// state directory, its output appended to `log`: it runs `argv` once this
/// process is done.
fn start_detached(
    manager: &str,
    argv: &[String],
    path_first: Option<PathBuf>,
    dir: &Path,
    log: &Path,
    newest: &str,
) -> Result<()> {
    use std::io::Write;
    use std::os::unix::process::CommandExt;
    let state = log.parent().unwrap_or(Path::new("/"));
    fs::create_dir_all(state).with_context(|| format!("creating {}", state.display()))?;
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .with_context(|| format!("opening {}", log.display()))?;
    writeln!(
        out,
        "== {} basert {VERSION} -> {newest}: {}",
        now(),
        argv.join(" ")
    )?;
    let mut helper = vec![
        this_exe()?.display().to_string(),
        "self".into(),
        "run-manager".into(),
        "--after".into(),
        std::process::id().to_string(),
        "--dir".into(),
        dir.display().to_string(),
        "--release".into(),
        newest.to_string(),
        "--manager".into(),
        manager.to_string(),
    ];
    if let Some(first) = path_first {
        helper.extend(["--path-first".into(), first.display().to_string()]);
    }
    helper.push("--".into());
    helper.extend(argv.iter().cloned());
    let mut run = Command::new("/bin/sh");
    run.arg("-c")
        .arg("\"$@\" </dev/null &")
        .arg("sh")
        .args(&helper)
        // Not the directory the command runs in: nothing of it is read.
        .current_dir(state)
        .env(NO_CHECK_ENV, "1")
        .stdin(Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(out)
        // Its own process group: a Ctrl-C at the terminal is not for it.
        .process_group(0);
    let status = run.status().context("starting the update")?;
    if !status.success() {
        bail!("the shell that starts it failed ({status})");
    }
    Ok(())
}

/// `self run-manager`: the manager's command `argv`, once the launch that
/// started this (`after`) is done, since it runs from the copy in `dir` the
/// manager replaces, and while no server runs on that copy (a server's
/// workers load the engine from it again with each model; cargo replaces the
/// launcher alone). Homebrew runs only when its formula is `release`, the
/// one the launch chose. What it does goes to the log it was started with.
fn run_manager(
    after: i32,
    dir: &Path,
    release: &str,
    manager: &str,
    path_first: Option<PathBuf>,
    argv: &[String],
) -> Result<()> {
    use std::os::unix::process::CommandExt;
    while alive(after) {
        std::thread::sleep(Duration::from_millis(200));
    }
    // Turned off while it waited (`self auto-update off` from another
    // terminal, say): the update the launch queued is not run.
    if !still_on(release) {
        return Ok(());
    }
    if manager != "cargo" {
        let busy = servers_on_package(dir);
        if !busy.is_empty() {
            println!(
                "a server runs on the engine in {} (pid {}): basert {release} is not installed; \
                 the next launch without one tries again",
                dir.display(),
                pid_list(&busy)
            );
            unqueue(release);
            return Ok(());
        }
    }
    let path = match path_first {
        Some(first) => {
            let rest = std::env::var_os("PATH").unwrap_or_default();
            Some(std::env::join_paths(
                std::iter::once(first).chain(std::env::split_paths(&rest)),
            )?)
        }
        None => None,
    };
    let command = |argv: &[String]| {
        let mut c = Command::new(&argv[0]);
        c.args(&argv[1..]);
        if let Some(path) = &path {
            c.env("PATH", path);
        }
        // What `brew info` said is what `brew upgrade` installs.
        if manager == "Homebrew" {
            c.env("HOMEBREW_NO_AUTO_UPDATE", "1");
        }
        c
    };
    if manager == "Homebrew" {
        let (brew, formula) = (&argv[0], argv.get(2).map_or("basert", String::as_str));
        let info = command(&[
            brew.clone(),
            "info".into(),
            "--json=v2".into(),
            formula.into(),
        ])
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running {brew} info"))?;
        let offered = serde_json::from_slice::<Value>(&info.stdout)
            .ok()
            .and_then(|v| {
                v["formulae"][0]["versions"]["stable"]
                    .as_str()
                    .map(str::to_string)
            });
        match offered {
            Some(v) if v.trim_start_matches('v') == release => {}
            other => {
                println!(
                    "Homebrew's {formula} is {}, not {release}: `brew upgrade {formula}` is left \
                     to you",
                    other.as_deref().unwrap_or("not known")
                );
                return Ok(());
            }
        }
    }
    // Read again just before the manager runs: `brew info` may have taken a while.
    if !still_on(release) {
        return Ok(());
    }
    let err = command(argv).exec();
    bail!("running {}: {err}", argv[0])
}

/// Whether automatic updates are still on for the package manager's update
/// to `release` a launch queued, and `release` still out by the wait: the
/// settings read again now, and the variables as the launch had them. When
/// not, it says so in the log and is no longer noted as started, so that a
/// launch once they are on again (or it is out) queues it anew.
fn still_on(release: &str) -> bool {
    // The launch set BASERT_NO_UPDATE_CHECK for this process itself; had the
    // user set it, nothing would have been queued.
    let env = |k: &str| match k {
        NO_CHECK_ENV => None,
        _ => std::env::var_os(k),
    };
    let settings = settings_file();
    if let Some(why) = auto_update_off(&env, &auto_update_setting(settings.as_deref())) {
        println!("automatic updates are off ({why}): basert {release} is not installed");
        unqueue(release);
        return false;
    }
    let wait = update_wait(&env, settings.as_deref());
    let out_at = Check::file()
        .and_then(|path| Check::read(&path))
        .and_then(|c| c.out_at(release));
    if out_at.is_some_and(|at| !wait.out(now()).since(at)) {
        println!("{}: basert {release} is not installed yet", wait.rule());
        unqueue(release);
        return false;
    }
    true
}

/// The check file no longer noting a package manager's update to `release`
/// as started.
fn unqueue(release: &str) {
    if let Some(path) = Check::file() {
        note_in_check(&path, &Check::default(), |c| {
            if c.started.as_deref() == Some(release) {
                c.started = None;
            }
        });
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
    use std::sync::Arc;
    use std::time::Instant;
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
    fn each_os_and_cpu_takes_its_own_engine_bundle() {
        assert_eq!(platform_for("macos", "aarch64").unwrap(), "macos-arm64");
        assert_eq!(
            platform_for("linux", "aarch64").unwrap(),
            "linux-arm64-cuda"
        );
        assert_eq!(platform_for("linux", "x86_64").unwrap(), "linux-x64-rocm");
        let err = platform_for("macos", "x86_64").unwrap_err().to_string();
        assert!(
            err.contains("no basert release is published for macos/x86_64"),
            "{err}"
        );
        assert!(platform_for("windows", "x86_64").is_err());

        // A release with all three bundles: Linux x86-64 takes the ROCm one.
        let listing = json!([listed(
            "v0.5.0",
            "2026-12-10T00:00:00Z",
            &[
                "basert-engine-abi-1.txt",
                "basert-engine-macos-arm64-0.5.0.tar.gz",
                "basert-engine-macos-arm64-0.5.0.tar.gz.sha256",
                "basert-engine-linux-arm64-cuda-0.5.0.tar.gz",
                "basert-engine-linux-arm64-cuda-0.5.0.tar.gz.sha256",
                "basert-engine-linux-x64-rocm-0.5.0.tar.gz",
                "basert-engine-linux-x64-rocm-0.5.0.tar.gz.sha256",
            ]
        )]);
        let found = releases_in(&listing, platform_for("linux", "x86_64").unwrap());
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].tarball,
            "basert-engine-linux-x64-rocm-0.5.0.tar.gz"
        );
        assert_eq!(
            found[0].checksum_url,
            "https://x/v0.5.0/basert-engine-linux-x64-rocm-0.5.0.tar.gz.sha256"
        );
        assert_eq!(found[0].abi, Some(1));
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
    }

    #[test]
    fn a_release_is_out_once_it_has_been_published_as_long_as_the_wait() {
        let stamp = "2026-10-27T06:42:10Z";
        let at = epoch_of(stamp).unwrap();
        let three = Wait {
            days: 3,
            from_env: false,
        };
        assert!(!three.out(at + 2 * DAY).stamp(stamp));
        assert!(!three.out(at + 3 * DAY - 1).stamp(stamp));
        assert!(three.out(at + 3 * DAY).stamp(stamp));
        // No wait: out once it is published, not before.
        let none = Wait {
            days: 0,
            from_env: false,
        };
        assert!(none.out(at).stamp(stamp));
        assert!(!none.out(at - 1).stamp(stamp));
        // A time that cannot be read counts as long ago.
        assert!(three.out(at).stamp(""));
        assert!(three.out(at).entry(&json!({ "tag_name": "v1.0.0" })));
        // The longest wait does not overflow.
        let longest = Wait {
            days: MAX_WAIT_DAYS,
            from_env: false,
        };
        assert!(!longest.out(u64::MAX - 1).since(u64::MAX - 1));
        assert!(longest.out(u64::MAX).since(0));
    }

    /// A release a look found, out since `out_at`.
    fn found_at(version: &str, out_at: u64) -> Listed {
        Listed {
            version: version.into(),
            notes: None,
            out_at,
        }
    }

    /// A look that found basert `newest` (and superfluid `superfluid`), out
    /// long ago.
    fn looked(newest: &str, superfluid: Option<&str>) -> Check {
        Check {
            releases: vec![found_at(newest, 0)],
            superfluids: superfluid.map(|s| found_at(s, 0)).into_iter().collect(),
            ..Check::default()
        }
    }

    /// Now, waiting for nothing.
    fn out_now() -> Out {
        Wait {
            days: 0,
            from_env: false,
        }
        .out(now())
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
        let newer = looked(&ahead, None);
        let line = newer.line(&managed, None, true, out_now()).unwrap();
        assert!(
            line.starts_with(&format!(
                "basert {ahead} is available (this is {VERSION}): `basert self update`"
            )),
            "{line}"
        );
        assert!(newer
            .line(&brew, None, true, out_now())
            .unwrap()
            .contains("`brew upgrade basert`"));
        assert_eq!(newer.line(&unknown, None, true, out_now()), None);
        let same = looked(VERSION, None);
        assert_eq!(same.line(&managed, None, true, out_now()), None);

        let sf = looked(VERSION, Some("v0.1.9"));
        let line = sf.line(&managed, Some("v0.1.5"), true, out_now()).unwrap();
        assert!(line.starts_with("superfluid v0.1.9, the server `basert serve` runs, is available (installed: v0.1.5): `basert self update`"), "{line}");
        assert!(sf
            .line(&brew, Some("v0.1.5"), true, out_now())
            .unwrap()
            .contains("`basert superfluid install`"));
        assert_eq!(sf.line(&managed, Some("v0.1.9"), true, out_now()), None);
        assert_eq!(sf.line(&managed, None, true, out_now()), None);
    }

    #[test]
    fn every_notice_names_the_release_notes_and_the_way_to_turn_it_off() {
        let ahead = format!("{VERSION}.1");
        let managed = Installed::Managed {
            root: "/x".into(),
            version: VERSION.into(),
        };
        let newer = looked(&ahead, None);
        let line = newer.line(&managed, None, true, out_now()).unwrap();
        assert!(
            line.ends_with(&format!(
                "· release notes: https://github.com/basecompute/baseRT/releases/tag/v{ahead} \
                 · turn off: basert self auto-update off"
            )),
            "{line}"
        );
        assert!(!line.contains(NO_CHECK_ENV), "{line}");
        // The page the listing gave, when it gave one.
        let mut paged = newer.clone();
        paged.releases[0].notes = Some("https://mirror/notes".into());
        assert!(paged
            .line(&managed, None, true, out_now())
            .unwrap()
            .contains("release notes: https://mirror/notes ·"));
        // With automatic updates off, there is nothing to turn off.
        let line = newer.line(&managed, None, false, out_now()).unwrap();
        assert!(
            line.ends_with(&format!(
                "· release notes: https://github.com/basecompute/baseRT/releases/tag/v{ahead}"
            )),
            "{line}"
        );
        let sf = looked(VERSION, Some("v0.1.9"));
        let line = sf.line(&managed, Some("v0.1.5"), true, out_now()).unwrap();
        assert!(
            line.contains(
                "release notes: https://github.com/basecompute/superfluid/releases/tag/v0.1.9 \
                 · turn off: basert self auto-update off"
            ),
            "{line}"
        );
    }

    #[test]
    fn the_listing_gives_each_release_its_notes_page_and_tarball_size() {
        let mut r = listed(
            "v0.4.2",
            "2026-10-27T06:42:10Z",
            &[
                "basert-engine-macos-arm64-0.4.2.tar.gz",
                "basert-engine-macos-arm64-0.4.2.tar.gz.sha256",
            ],
        );
        let found = releases_in(&json!([r.clone()]), "macos-arm64");
        assert_eq!(
            found[0].notes,
            "https://github.com/basecompute/baseRT/releases/tag/v0.4.2"
        );
        assert_eq!(found[0].tarball_size, None);
        r["html_url"] = json!("https://github.com/basecompute/baseRT/releases/tag/v0.4.2-page");
        r["assets"][0]["size"] = json!(1234);
        let found = releases_in(&json!([r]), "macos-arm64");
        assert_eq!(
            found[0].notes,
            "https://github.com/basecompute/baseRT/releases/tag/v0.4.2-page"
        );
        assert_eq!(found[0].tarball_size, Some(1234));
    }

    #[test]
    fn the_auto_update_setting_round_trips_and_keeps_the_other_settings() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("baseRT").join("settings.json");
        // Missing, or a file without it: on.
        assert_eq!(auto_update_setting(None), Setting::On);
        assert_eq!(auto_update_setting(Some(&path)), Setting::On);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, r#"{"theme": "dark"}"#).unwrap();
        assert_eq!(auto_update_setting(Some(&path)), Setting::On);
        fs::write(&path, r#"{"autoUpdate": "no", "theme": "dark"}"#).unwrap();

        set_auto_update_setting(&path, false).unwrap();
        assert_eq!(auto_update_setting(Some(&path)), Setting::Off);
        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v, json!({"autoUpdate": false, "theme": "dark"}));
        set_auto_update_setting(&path, true).unwrap();
        assert_eq!(auto_update_setting(Some(&path)), Setting::On);
        // Written in one rename: no temporary file is left beside it.
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["settings.json"]);
    }

    #[test]
    fn the_wait_setting_round_trips_and_keeps_the_other_settings() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("baseRT").join("settings.json");
        // Missing: no wait.
        assert_eq!(wait_setting(None), Ok(0));
        assert_eq!(wait_setting(Some(&path)), Ok(0));
        set_wait_setting(&path, 3).unwrap();
        assert_eq!(wait_setting(Some(&path)), Ok(3));
        set_auto_update_setting(&path, false).unwrap();
        set_wait_setting(&path, 0).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v, json!({"autoUpdate": false, "updateWaitDays": 0}));
        set_wait_setting(&path, MAX_WAIT_DAYS).unwrap();
        assert_eq!(wait_setting(Some(&path)), Ok(365));
        assert_eq!(auto_update_setting(Some(&path)), Setting::Off);
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["settings.json"]);
    }

    #[test]
    fn a_wait_that_is_not_a_whole_number_of_days_turns_automatic_updates_off_until_set() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");
        let none = |_: &str| None::<OsString>;
        for bad in [r#""3""#, "-1", "2.5", "366", "true"] {
            fs::write(
                &path,
                format!(r#"{{"updateWaitDays": {bad}, "theme": "dark"}}"#),
            )
            .unwrap();
            let setting = auto_update_setting(Some(&path));
            let why = auto_update_off(&none, &setting).unwrap();
            assert!(
                why.starts_with(&format!("updateWaitDays in {} is ", path.display()))
                    && why.ends_with("not a whole number of days from 0 to 365"),
                "{bad}: {why}"
            );
            assert!(wait_setting(Some(&path)).is_err(), "{bad}");
            // A newer release is still said, as soon as it is out.
            assert_eq!(
                update_wait(&none, Some(&path)),
                Wait {
                    days: 0,
                    from_env: false
                }
            );
        }
        // Setting it mends the file and keeps the rest.
        set_wait_setting(&path, 2).unwrap();
        assert_eq!(auto_update_setting(Some(&path)), Setting::On);
        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v, json!({"updateWaitDays": 2, "theme": "dark"}));
        // A file that is not JSON is not written over.
        let text = "{\"updateWaitDays\": 3,}";
        fs::write(&path, text).unwrap();
        let err = set_wait_setting(&path, 1).unwrap_err().to_string();
        assert!(err.contains("is not JSON"), "{err}");
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
    }

    #[test]
    fn the_wait_variable_overrides_the_setting_and_one_that_is_no_wait_is_ignored() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");
        set_wait_setting(&path, 3).unwrap();
        let with =
            |value: &'static str| move |k: &str| (k == WAIT_ENV).then(|| OsString::from(value));
        let wait = |days, from_env| Wait { days, from_env };
        assert_eq!(update_wait(&with("0"), Some(&path)), wait(0, true));
        assert_eq!(update_wait(&with(" 7 "), Some(&path)), wait(7, true));
        assert_eq!(update_wait(&with("365"), Some(&path)), wait(365, true));
        for ignored in ["", "x", "-1", "1.5", "366", "99999999999"] {
            assert_eq!(
                update_wait(&with(ignored), Some(&path)),
                wait(3, false),
                "{ignored:?}"
            );
        }
        assert_eq!(update_wait(&|_| None, None), wait(0, false));
    }

    #[test]
    fn self_auto_update_says_whether_it_is_on_and_the_wait() {
        let none = |_: &str| None::<OsString>;
        let wait = |days, from_env| Wait { days, from_env };
        assert_eq!(
            state_line(&none, &Setting::On, wait(0, false)),
            "automatic updates: on · a release installs as soon as it is out"
        );
        assert_eq!(
            state_line(&none, &Setting::On, wait(3, false)),
            "automatic updates: on · a release installs once it has been out 3 days"
        );
        assert_eq!(
            state_line(&none, &Setting::On, wait(1, true)),
            "automatic updates: on · a release installs once it has been out 1 day \
             (BASERT_UPDATE_WAIT_DAYS is set)"
        );
        // Off, the wait still: as `superfluid self auto-update` says it.
        assert_eq!(
            state_line(&none, &Setting::Off, wait(3, false)),
            "automatic updates: off (basert self auto-update off) · a release installs \
             once it has been out 3 days"
        );
        let no_update = |k: &str| (k == NO_UPDATE_ENV).then(|| OsString::from("1"));
        assert_eq!(
            state_line(&no_update, &Setting::On, wait(0, false)),
            "automatic updates: off (BASERT_NO_UPDATE is set) · a release installs as \
             soon as it is out"
        );
        let no_wait = |k: &str| (k == WAIT_ENV).then(|| OsString::from("soon"));
        assert_eq!(
            state_line(&no_wait, &Setting::On, wait(3, false)),
            "automatic updates: on · a release installs once it has been out 3 days \
             (BASERT_UPDATE_WAIT_DAYS=soon is not a whole number of days from 0 to 365, so \
             the setting is used)"
        );
        let empty = |k: &str| (k == WAIT_ENV).then(|| OsString::from(""));
        assert_eq!(
            state_line(&empty, &Setting::On, wait(3, false)),
            "automatic updates: on · a release installs once it has been out 3 days"
        );
        assert_eq!(
            wait(0, false).rule(),
            "automatic updates install a release as soon as it is out"
        );
        assert_eq!(
            wait(1, false).rule(),
            "automatic updates wait 1 day after a release"
        );
        assert_eq!(
            wait(3, false).rule(),
            "automatic updates wait 3 days after a release"
        );
    }

    #[test]
    fn the_notice_and_the_update_agree_on_which_release_is_out() {
        let now = epoch_of("2026-12-20T00:00:00Z").unwrap();
        let (older, newer) = (format!("{VERSION}.1"), format!("{VERSION}.2"));
        let managed = Installed::Managed {
            root: "/x".into(),
            version: VERSION.into(),
        };
        let check = Check {
            releases: vec![found_at(&newer, now - DAY), found_at(&older, now - 5 * DAY)],
            superfluids: vec![
                found_at("v0.1.9", now - DAY),
                found_at("v0.1.8", now - 9 * DAY),
            ],
            ..Check::default()
        };
        let wait = |days| Wait {
            days,
            from_env: false,
        };
        for (days, basert) in [(0, &newer), (1, &newer), (3, &older), (5, &older)] {
            let out = wait(days).out(now);
            assert_eq!(check.due(out), Some(basert.as_str()), "{days}");
            let line = check.line(&managed, None, true, out).unwrap();
            assert!(
                line.starts_with(&format!("basert {basert} is available")),
                "{line}"
            );
            // Under a wait, `basert self update` installs the newest, which
            // may be younger than the release named.
            let installs = match days {
                0 => "`basert self update` installs it ·",
                _ => "`basert self update` installs the newest ·",
            };
            assert!(line.contains(installs), "{days}: {line}");
        }
        // Neither is out yet: no update, and no line.
        let out = wait(6).out(now);
        assert_eq!(check.due(out), None);
        assert_eq!(check.line(&managed, None, true, out), None);
        // superfluid the same way, when basert is the newest.
        let sf = Check {
            releases: vec![found_at(VERSION, 0)],
            ..check
        };
        let line = |days| sf.line(&managed, Some("v0.1.5"), true, wait(days).out(now));
        assert_eq!(
            sf.superfluid_due("v0.1.5", wait(0).out(now)),
            Some("v0.1.9")
        );
        assert!(line(0).unwrap().starts_with("superfluid v0.1.9,"));
        assert_eq!(
            sf.superfluid_due("v0.1.5", wait(3).out(now)),
            Some("v0.1.8")
        );
        assert!(line(0)
            .unwrap()
            .contains("`basert self update` installs it ·"));
        assert!(line(3).unwrap().starts_with("superfluid v0.1.8,"));
        assert!(line(3)
            .unwrap()
            .contains("`basert self update` installs the newest ·"));
        assert_eq!(sf.superfluid_due("v0.1.8", wait(3).out(now)), None);
        assert_eq!(line(10), None);
    }

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn no_update_is_taken_out_of_any_command_before_a_double_dash() {
        assert_eq!(
            strip_no_update(os(&["basert", "--no-update", "serve", "m"])),
            (os(&["basert", "serve", "m"]), true)
        );
        assert_eq!(
            strip_no_update(os(&["basert", "pull", "m", "--no-update"])),
            (os(&["basert", "pull", "m"]), true)
        );
        // After `--` it is the command's own argument.
        assert_eq!(
            strip_no_update(os(&["basert", "chat", "m", "--", "--no-update"])),
            (os(&["basert", "chat", "m", "--", "--no-update"]), false)
        );
        assert_eq!(
            strip_no_update(os(&["basert", "list"])),
            (os(&["basert", "list"]), false)
        );
        // The program's own name is never taken.
        assert_eq!(
            strip_no_update(os(&["--no-update"])),
            (os(&["--no-update"]), false)
        );
    }

    #[test]
    fn a_settings_file_that_says_neither_on_nor_off_turns_automatic_updates_off_and_is_kept() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");
        let none = |_: &str| None::<OsString>;
        // An opt-out with a trailing comma, a value that is not a boolean,
        // a root that is not an object: off, saying why.
        for (text, says) in [
            ("{\"autoUpdate\": false,}", "is not JSON"),
            (r#"{"autoUpdate": "false"}"#, "autoUpdate in "),
            (r#"{"autoUpdate": 0}"#, "is 0, not true or false"),
            ("[]", "is not a JSON object"),
        ] {
            fs::write(&path, text).unwrap();
            let setting = auto_update_setting(Some(&path));
            assert!(matches!(setting, Setting::Unreadable(_)), "{text}");
            let why = auto_update_off(&none, &setting).unwrap();
            assert!(why.contains(&path.display().to_string()), "{why}");
            assert!(why.contains(says), "{text}: {why}");
        }
        // A file that is there and not JSON is not written over: what else it
        // held would be lost.
        let text = "{\"autoUpdate\": false, \"theme\": \"dark\",}";
        fs::write(&path, text).unwrap();
        let err = set_auto_update_setting(&path, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not JSON"), "{err}");
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        // One that cannot be read: off as well.
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
        if fs::read_to_string(&path).is_err() {
            let setting = auto_update_setting(Some(&path));
            assert!(
                matches!(&setting, Setting::Unreadable(why) if why.contains("could not be read")),
                "{setting:?}"
            );
        }
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    }

    #[test]
    fn automatic_updates_wait_for_a_terminal_and_a_command_that_does_something() {
        let none = |_: &str| None::<OsString>;
        let serve = os(&["serve", "m"]);
        let on = Setting::On;
        assert_eq!(
            auto_update_blocked(&serve, true, false, false, &none, &on),
            None
        );
        let why = |args: &[OsString],
                   terminal,
                   flag,
                   env: &dyn Fn(&str) -> Option<OsString>,
                   setting: bool| {
            let setting = match setting {
                true => Setting::On,
                false => Setting::Off,
            };
            auto_update_blocked(args, terminal, false, flag, env, &setting).unwrap()
        };
        assert_eq!(
            why(&serve, false, false, &none, true),
            "stderr is not a terminal"
        );
        assert_eq!(
            why(&serve, true, true, &none, true),
            "--no-update was given"
        );
        assert_eq!(
            why(&serve, true, false, &none, false),
            "basert self auto-update off"
        );
        // `sudo basert …` over a user's install: never by itself.
        assert_eq!(
            auto_update_blocked(&serve, true, true, false, &none, &on).as_deref(),
            Some("it runs as root: the install's owner updates it")
        );
        let with = |key: &'static str, value: &'static str| {
            move |k: &str| (k == key).then(|| OsString::from(value))
        };
        assert_eq!(
            why(&serve, true, false, &with("BASERT_NO_UPDATE", "1"), true),
            "BASERT_NO_UPDATE is set"
        );
        assert_eq!(
            why(
                &serve,
                true,
                false,
                &with("BASERT_NO_UPDATE_CHECK", "1"),
                true
            ),
            "BASERT_NO_UPDATE_CHECK is set"
        );
        assert_eq!(
            why(&serve, true, false, &with("CI", "true"), true),
            "CI is set"
        );
        assert!(why(
            &serve,
            true,
            false,
            &with("BASERT_UPDATED_FROM", "0.5.4"),
            true
        )
        .starts_with("BASERT_UPDATED_FROM is set"));
        // `0` and empty leave a switch off.
        for value in ["0", ""] {
            assert_eq!(
                auto_update_blocked(
                    &serve,
                    true,
                    false,
                    false,
                    &with("BASERT_NO_UPDATE", value),
                    &on
                ),
                None
            );
        }
        assert_eq!(
            auto_update_blocked(&serve, true, false, false, &with("CI", "false"), &on),
            None
        );
        for args in [
            &["self", "update"][..],
            &["list"],
            &["inspect", "m.base"],
            &["verify", "m.base"],
            &["superfluid", "status"],
            &["pull", "m", "--json"],
            &["serve", "--help"],
            &["--version"],
            &[],
        ] {
            assert_eq!(
                why(&os(args), true, false, &none, true),
                "the command only tells something",
                "{args:?}"
            );
        }
        for args in [
            &["pull", "m"][..],
            &["chat", "m", "--", "--json"],
            &["keygen"],
        ] {
            assert_eq!(
                auto_update_blocked(&os(args), true, false, false, &none, &on),
                None,
                "{args:?}"
            );
        }
    }

    #[test]
    fn self_auto_update_says_why_updates_are_off() {
        let none = |_: &str| None::<OsString>;
        assert_eq!(auto_update_off(&none, &Setting::On), None);
        assert_eq!(
            auto_update_off(&none, &Setting::Off).as_deref(),
            Some("basert self auto-update off")
        );
        let no_update = |k: &str| (k == NO_UPDATE_ENV).then(|| OsString::from("1"));
        // The variable is said before the setting: it is what turned them off.
        assert_eq!(
            auto_update_off(&no_update, &Setting::Off).as_deref(),
            Some("BASERT_NO_UPDATE is set")
        );
    }

    #[test]
    fn a_failed_automatic_update_waits_for_the_next_look() {
        let ahead = format!("{VERSION}.1");
        let out = out_now();
        let mut check = Check {
            checked_at: 100,
            ..looked(&ahead, None)
        };
        assert_eq!(check.due(out), Some(ahead.as_str()));
        check.failed = Some(ahead.clone());
        check.failed_at = 150;
        assert_eq!(check.due(out), None);
        // The next look (claimed at a later launch) lets it try again.
        check.checked_at = 200;
        assert_eq!(check.due(out), Some(ahead.as_str()));
        // A failure for another release does not hold this one.
        check.failed = Some("0.0.1".into());
        check.checked_at = 100;
        assert_eq!(check.due(out), Some(ahead.as_str()));
        check.releases = vec![found_at(VERSION, 0)];
        assert_eq!(check.due(out), None);
    }

    #[test]
    fn the_check_file_round_trips_what_launches_did() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("state").join("update-check.json");
        let check = Check {
            checked_at: 1,
            releases: vec![
                Listed {
                    version: "0.9.0".into(),
                    notes: Some("https://x/notes".into()),
                    out_at: 1_800_000_000,
                },
                found_at("0.8.0", 7),
            ],
            superfluids: vec![found_at("v0.2.0", 5)],
            notified_at: 2,
            failed: Some("0.9.0".into()),
            failed_at: 3,
            superfluid_failed: Some("v0.2.0".into()),
            superfluid_failed_at: 4,
            started: Some("0.9.0".into()),
        };
        check.write(&path).unwrap();
        assert_eq!(Check::read(&path), Some(check));
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["update-check.json"]);
    }

    #[test]
    fn a_failed_automatic_superfluid_update_waits_for_the_next_look() {
        let out = out_now();
        let mut check = Check {
            checked_at: 100,
            ..looked(VERSION, Some("v0.1.9"))
        };
        assert_eq!(check.superfluid_due("v0.1.5", out), Some("v0.1.9"));
        // Installed already, or newer.
        assert_eq!(check.superfluid_due("v0.1.9", out), None);
        assert_eq!(check.superfluid_due("v0.2.0", out), None);
        check.superfluid_failed = Some("v0.1.9".into());
        check.superfluid_failed_at = 150;
        assert_eq!(check.superfluid_due("v0.1.5", out), None);
        // basert's failure is basert's.
        assert_eq!(check.due(out), None);
        check.checked_at = 200;
        assert_eq!(check.superfluid_due("v0.1.5", out), Some("v0.1.9"));
    }

    #[test]
    fn the_look_a_launch_claimed_keeps_the_claim_so_a_failure_after_it_holds() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("update-check.json");
        let ahead = format!("{VERSION}.1");
        let found = Found {
            releases: Vec::new(),
            superfluid: None,
        };
        // Claimed at a launch, which then failed to update, before the look
        // it started was done.
        let claimed = now() - 5;
        let check = Check {
            checked_at: claimed,
            failed: Some(ahead.clone()),
            failed_at: claimed + 1,
            ..looked(&ahead, None)
        };
        check.write(&path).unwrap();
        found.record(&path, true);
        let after = Check::read(&path).unwrap();
        assert_eq!(after.checked_at, claimed);
        assert_eq!(after.failed_at, claimed + 1);
        // A day-old claim, or `self check` asked for: a look of its own.
        let check = Check {
            checked_at: claimed - CHECK_EVERY.as_secs(),
            ..check
        };
        check.write(&path).unwrap();
        found.record(&path, true);
        assert!(Check::read(&path).unwrap().checked_at >= claimed + 5);
        Check {
            checked_at: claimed,
            ..check
        }
        .write(&path)
        .unwrap();
        found.record(&path, false);
        assert!(Check::read(&path).unwrap().checked_at >= claimed + 5);
    }

    /// A superfluid listing's entry for this machine serving `abi`,
    /// published at `published`.
    fn superfluid_listed(tag: &str, abi: u32, published: &str) -> Value {
        let tarball = format!("superfluid-{}.tar.gz", superfluid::target().unwrap());
        listed(
            tag,
            published,
            &[
                &tarball,
                &format!("{tarball}.sha256"),
                &format!("superfluid-basert-abi-{abi}.txt"),
            ],
        )
    }

    #[test]
    fn an_automatic_update_takes_a_basert_only_a_superfluid_out_as_long_serves() {
        let now = epoch_of("2026-12-20T00:00:00Z").unwrap();
        let platform = platform().unwrap();
        let tarball = |v: &str| format!("basert-engine-{platform}-{v}.tar.gz");
        let basert = |v: &str, abi: u32| {
            listed(
                &format!("v{v}"),
                "2026-12-01T00:00:00Z",
                &[
                    &tarball(v),
                    &format!("{}.sha256", tarball(v)),
                    &format!("basert-engine-abi-{abi}.txt"),
                ],
            )
        };
        let found = Found {
            releases: releases_in(&json!([basert("0.9.1", 2), basert("0.9.0", 1)]), platform),
            superfluid: Some(json!([
                // Serves ABI 2, published the day before.
                superfluid_listed("v0.3.0", 2, "2026-12-19T00:00:00Z"),
                superfluid_listed("v0.2.0", 1, "2026-11-01T00:00:00Z"),
            ])),
        };
        let wait = |days| Wait {
            days,
            from_env: false,
        };
        // By hand, 0.9.1; automatically with a wait of 3 days, 0.9.0 until
        // its superfluid has been out as long; with none, 0.9.1 at once.
        assert_eq!(found.newest(None).unwrap().version, "0.9.1");
        assert_eq!(
            found.newest(Some(wait(3).out(now))).unwrap().version,
            "0.9.0"
        );
        assert_eq!(
            found
                .newest(Some(wait(3).out(now + 3 * DAY)))
                .unwrap()
                .version,
            "0.9.1"
        );
        assert_eq!(
            found.newest(Some(wait(0).out(now))).unwrap().version,
            "0.9.1"
        );
        // Each counts as out from the later of its own day and its
        // superfluid's: what a launch's wait is applied to.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("update-check.json");
        found.record(&path, false);
        let check = Check::read(&path).unwrap();
        assert_eq!(
            check.releases,
            [
                Listed {
                    version: "0.9.1".into(),
                    notes: Some(notes_for("0.9.1")),
                    out_at: epoch_of("2026-12-19T00:00:00Z").unwrap(),
                },
                Listed {
                    version: "0.9.0".into(),
                    notes: Some(notes_for("0.9.0")),
                    out_at: epoch_of("2026-12-01T00:00:00Z").unwrap(),
                },
            ]
        );
        // The superfluid an automatic update brings is out by the wait too.
        let run = Run::Automatic {
            out: wait(3).out(now),
            listing: found.superfluid.as_ref(),
        };
        assert_eq!(run.newest_superfluid(2).unwrap(), None);
        assert_eq!(run.newest_superfluid(1).unwrap().unwrap().tag, "v0.2.0");
        let run = Run::Automatic {
            out: wait(0).out(now),
            listing: found.superfluid.as_ref(),
        };
        assert_eq!(run.newest_superfluid(2).unwrap().unwrap().tag, "v0.3.0");
        let unread = Run::Automatic {
            out: wait(0).out(now),
            listing: None,
        };
        assert!(unread.newest_superfluid(1).is_err());
    }

    /// The tests that set the Ctrl-C flag, one at a time.
    static CTRL_C: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A server on 127.0.0.1 that reads a request and then, with `headers`,
    /// answers with the headers of a 1000-byte body and one byte of it, or
    /// without, nothing; then it goes quiet until the flag it gives is set.
    fn stalling_server(headers: bool) -> (String, Arc<AtomicBool>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let at = format!(
            "http://{}/superfluid.tar.gz",
            listener.local_addr().unwrap()
        );
        let released = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&released);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    let mut asked = Vec::new();
                    let mut byte = [0u8; 1];
                    while !asked.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap_or(0) == 1
                    {
                        asked.push(byte[0]);
                    }
                    if headers {
                        let _ = stream
                            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nx")
                            .and_then(|()| stream.flush());
                    }
                    let until = Instant::now() + Duration::from_secs(20);
                    while !done.load(Ordering::SeqCst) && Instant::now() < until {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                });
            }
        });
        (at, released)
    }

    /// `work` with a Ctrl-C 200 ms in, as an automatic update catches it:
    /// what it returned, and how long it took.
    fn ctrl_c_during<T>(work: impl FnOnce() -> Result<T>) -> (Result<T>, Duration) {
        let _one = CTRL_C.lock().unwrap_or_else(|e| e.into_inner());
        let _catch = CatchInterrupt::new();
        let started = Instant::now();
        let stop = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(200));
            INTERRUPTED.store(true, Ordering::SeqCst);
        });
        let done = work();
        let took = started.elapsed();
        stop.join().unwrap();
        (done, took)
    }

    #[test]
    fn a_ctrl_c_stops_a_download_the_server_stalls_after_its_first_byte() {
        let (url, released) = stalling_server(true);
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("superfluid.tar.gz");
        let part = tmp.path().join("superfluid.tar.gz.part");
        let (done, took) =
            ctrl_c_during(|| superfluid::download_as(&url, &dest, Some("superfluid"), None));
        assert_eq!(done.unwrap_err().to_string(), "interrupted");
        assert!(took < Duration::from_millis(1200), "{took:?}");
        assert!(!part.exists() && !dest.exists());
        // The thread left behind, once the server lets go, leaves nothing.
        released.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(300));
        assert!(!part.exists() && !dest.exists());
        assert!(!interrupted());
    }

    #[test]
    fn a_ctrl_c_stops_a_download_and_a_listing_the_server_never_answers() {
        let (url, released) = stalling_server(false);
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("superfluid.tar.gz.sha256");
        let part = tmp.path().join("superfluid.tar.gz.sha256.part");
        let (done, took) = ctrl_c_during(|| superfluid::download(&url, &dest));
        assert_eq!(done.unwrap_err().to_string(), "interrupted");
        assert!(took < Duration::from_millis(1200), "{took:?}");
        assert!(!part.exists() && !dest.exists());
        let (done, took) = ctrl_c_during(|| superfluid::read_listing(&url));
        assert_eq!(done.unwrap_err().to_string(), "interrupted");
        assert!(took < Duration::from_millis(1200), "{took:?}");
        released.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(300));
        assert!(!part.exists() && !dest.exists());
    }

    #[test]
    fn a_ctrl_c_stops_the_look_and_not_the_command_after_the_update() {
        let _one = CTRL_C.lock().unwrap_or_else(|e| e.into_inner());
        fn slow() -> Result<u32> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(1)
        }
        fn quick() -> Result<u32> {
            Ok(7)
        }
        let started = std::time::Instant::now();
        let err = {
            let _catch = CatchInterrupt::new();
            assert_eq!(interruptible(quick).unwrap(), 7);
            let stop = std::thread::spawn(|| {
                std::thread::sleep(Duration::from_millis(100));
                INTERRUPTED.store(true, Ordering::SeqCst);
            });
            let err = interruptible(slow).unwrap_err().to_string();
            stop.join().unwrap();
            assert!(interrupted());
            err
        };
        assert_eq!(err, "interrupted");
        assert!(started.elapsed() < Duration::from_secs(10));
        // Over: what the command downloads next is not stopped.
        assert!(!interrupted());
    }

    #[test]
    fn going_back_holds_automatic_updates_until_a_newer_release() {
        let tmp = TempDir::new().unwrap();
        let layout = Layout {
            root: tmp.path().to_path_buf(),
        };
        fs::create_dir_all(layout.versions()).unwrap();
        layout.note_held_back(Some("0.5.4"), "0.5.5");
        assert_eq!(layout.held_back(), None);
        layout.note_held_back(Some("0.5.5"), "0.5.4");
        assert_eq!(layout.held_back().as_deref(), Some("0.5.5"));
        // Further back: still held to the newest it left.
        layout.note_held_back(Some("0.5.4"), "0.5.3");
        assert_eq!(layout.held_back().as_deref(), Some("0.5.5"));
        // Forward to it, or past it: let go.
        layout.note_held_back(Some("0.5.3"), "0.5.5");
        assert_eq!(layout.held_back(), None);
    }

    #[test]
    fn a_package_managers_update_runs_in_the_background_once_and_only_when_it_may() {
        let run = Delegate::Run {
            manager: "Homebrew".into(),
            command: "brew upgrade basert".into(),
            argv: vec![
                "/opt/homebrew/bin/brew".into(),
                "upgrade".into(),
                "basert".into(),
            ],
            path_first: None,
        };
        let named = Delegate::Name {
            manager: "pip".into(),
            command: "pip install --upgrade basert".into(),
            why: String::new(),
        };
        assert_eq!(
            manager_plan(&run, true, false, None, "0.5.5"),
            ManagerPlan::Background
        );
        assert_eq!(
            manager_plan(&run, true, false, Some("0.5.4"), "0.5.5"),
            ManagerPlan::Background
        );
        // Once per release.
        assert_eq!(
            manager_plan(&run, true, false, Some("0.5.5"), "0.5.5"),
            ManagerPlan::Notice
        );
        // Another user's directory, or a server on its engine.
        assert_eq!(
            manager_plan(&run, false, false, None, "0.5.5"),
            ManagerPlan::Notice
        );
        assert_eq!(
            manager_plan(&run, true, true, None, "0.5.5"),
            ManagerPlan::Notice
        );
        // pip, a project's npm, a pinned formula: only named.
        assert_eq!(
            manager_plan(&named, true, false, None, "0.5.5"),
            ManagerPlan::Notice
        );
    }

    #[test]
    fn a_package_managers_update_is_pinned_to_the_release_the_look_chose() {
        let argv = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            pinned(
                "npm",
                &argv(&[
                    "/n/bin/npm",
                    "install",
                    "--global",
                    "@basecompute/basert@latest"
                ]),
                "0.5.6"
            ),
            argv(&[
                "/n/bin/npm",
                "install",
                "--global",
                "@basecompute/basert@0.5.6"
            ])
        );
        assert_eq!(
            pinned("pipx", &argv(&["/b/pipx", "upgrade", "basert"]), "0.5.6"),
            argv(&["/b/pipx", "install", "--force", "basert==0.5.6"])
        );
        assert_eq!(
            pinned(
                "uv",
                &argv(&["/b/uv", "tool", "upgrade", "basert"]),
                "0.5.6"
            ),
            argv(&["/b/uv", "tool", "install", "basert==0.5.6"])
        );
        assert_eq!(
            pinned(
                "cargo",
                &argv(&["/c/cargo", "install", "basert-cli"]),
                "0.5.6"
            ),
            argv(&["/c/cargo", "install", "basert-cli", "--version", "0.5.6"])
        );
        // Homebrew's formula is checked before it runs instead.
        let brew = argv(&["/h/bin/brew", "upgrade", "basert"]);
        assert_eq!(pinned("Homebrew", &brew, "0.5.6"), brew);
    }

    #[test]
    fn a_tool_is_found_only_in_the_directories_path_names_in_full() {
        let tmp = TempDir::new().unwrap();
        let bin = tmp.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        executable(&bin.join("uv"), "#!/bin/sh\n");
        // A relative entry, an empty one, `.`: where the command happens to
        // run, never looked in.
        let path =
            std::env::join_paths(["node_modules/.bin", "", ".", bin.to_str().unwrap()]).unwrap();
        assert_eq!(find_on(Some(&path), "uv"), Some(bin.join("uv")));
        let relative = std::env::join_paths(["node_modules/.bin", "", "."]).unwrap();
        assert_eq!(find_on(Some(&relative), "sh"), None);
        assert_eq!(find_on(None, "uv"), None);
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
        match delegate(&global, named.clone()).unwrap() {
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
        // basert's package with its platform package inside: global too.
        let nested = launcher_at(
            tmp.path(),
            "node/lib/node_modules/@basecompute/basert/node_modules/@basecompute/basert-darwin-arm64/engine",
        );
        assert!(matches!(
            delegate(&nested, named.clone()),
            Some(Delegate::Run { .. })
        ));
        // A dependency of another global package is that package's.
        for rel in [
            "node/lib/node_modules/some-tool/node_modules/@basecompute/basert/node_modules/@basecompute/basert-darwin-arm64/engine",
            "node/lib/node_modules/some-tool/node_modules/@basecompute/basert-darwin-arm64/engine",
            "node/lib/node_modules/@basecompute/basert/node_modules/other/node_modules/@basecompute/basert-darwin-arm64/engine",
        ] {
            let exe = launcher_at(tmp.path(), rel);
            match delegate(&exe, named.clone()).unwrap() {
                Delegate::Name { why, .. } => {
                    assert!(why.contains("update it in that project"), "{rel}: {why}")
                }
                other => panic!("{rel}: {other:?}"),
            }
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
