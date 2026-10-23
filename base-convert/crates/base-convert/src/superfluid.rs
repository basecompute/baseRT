//! superfluid, the server `basert serve` runs.
//!
//! basert ships no server of its own. `basert serve` runs a superfluid
//! release: the one the one-shot installer put beside basert, or one basert
//! installs for itself at a start. Which superfluid may serve this engine is
//! superfluid's own call: before a start the launcher asks each install it
//! finds (`superfluid runtimes --json`, the engine library named) whether its
//! basert runtime is ready, and runs the first that is, whatever its release:
//! nothing is ever downgraded. When none is installed, or none can serve, the
//! newest release that states this engine's ABI number beside its tarballs
//! (`superfluid-basert-abi-<N>.txt`) is installed, or the release this basert
//! was tested with ([`RELEASE`]) when the release listing cannot be read or
//! names none.
//!
//! Where superfluid is looked for, in order: `$BASERT_SUPERFLUID_DIR`;
//! `superfluid/` beside basert, the one-shot installer's layout; a directory
//! of the user's own, where basert installs when the one beside it cannot be
//! written (a package manager's, an app bundle's). Each holds superfluid's
//! own layout (`bin/superfluid`, the runtime workers in `libexec/superfluid/`)
//! off PATH: a superfluid the user installed for themselves is neither
//! shadowed nor run in its place.
//!
//! A download at a start is asked for on a terminal and made without asking
//! otherwise, since an older Local calls `basert serve` without knowing
//! superfluid exists; `$BASERT_SUPERFLUID_AUTO_INSTALL=0` forbids it, and the
//! command to run is named instead. `basert superfluid install` installs
//! ahead of time; `$BASERT_SUPERFLUID_RELEASES` names another release listing
//! (a mirror's, a file), and `$BASERT_SUPERFLUID_SOURCE` a directory holding
//! the tested release's tarball and checksum, read instead of GitHub.

use std::fs;
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use sha2::{Digest, Sha256};

/// The superfluid release this basert was tested with, a tag of
/// <https://github.com/basecompute/superfluid/releases>: installed instead
/// of the newest when the release listing cannot be read or names no release
/// for this engine, and the release a `$BASERT_SUPERFLUID_SOURCE` holds.
pub const RELEASE: &str = "v0.1.3";

/// `BASERT_ABI_VERSION` of the engine shipped beside this basert, which a
/// superfluid release names when it serves it (`superfluid-basert-abi-<N>.txt`).
pub const ENGINE_ABI: u32 = 1;

const REPO: &str = "basecompute/superfluid";
/// The releases the newest superfluid is chosen from, newest first.
const RELEASES: &str = "https://api.github.com/repos/basecompute/superfluid/releases?per_page=30";
/// Another release listing in GitHub's form, a URL or a file: a mirror's, or the tests'.
pub const RELEASES_ENV: &str = "BASERT_SUPERFLUID_RELEASES";
/// Where superfluid is installed, instead of the places looked in by default.
pub const DIR_ENV: &str = "BASERT_SUPERFLUID_DIR";
/// A directory holding the release's tarball and its `.sha256`, read instead
/// of GitHub: an install on a machine without internet access, and the tests.
pub const SOURCE_ENV: &str = "BASERT_SUPERFLUID_SOURCE";
/// `0` forbids `basert serve` any download; `1` lets it download without
/// asking, even on a terminal.
pub const AUTO_INSTALL_ENV: &str = "BASERT_SUPERFLUID_AUTO_INSTALL";
/// Written into an install with the release it holds.
const STAMP: &str = ".release";
/// The name superfluid's basert runtime answers to in `runtimes --json`.
const RUNTIME_ID: &str = "basert";

/// `basert superfluid <action>`.
#[derive(Args, Debug)]
pub struct SuperfluidArgs {
    #[command(subcommand)]
    pub action: Action,
}

#[derive(Subcommand, Debug)]
pub enum Action {
    /// Install the newest superfluid release that serves this engine where
    /// `basert serve` looks for it (nothing to do when it is already there).
    Install {
        /// Install again even when that release, or a newer one, is there.
        #[arg(long)]
        force: bool,
    },
    /// Show the release this basert was tested with, what is installed where,
    /// and whether it can serve the engine beside basert. Exits 0 when an
    /// installed superfluid can, 1 when none is installed, 2 when the
    /// installed one cannot.
    Status,
}

pub fn cmd(args: SuperfluidArgs) -> Result<()> {
    match args.action {
        Action::Install { force } => {
            let source = Source::locate();
            let release = wanted(&source, &mut |line| eprintln!("{line}"));
            let tag = &release.tag;
            let dest = match Install::locate()? {
                Some(i) if !force && i.installed_release().as_deref() == Some(tag.as_str()) => {
                    println!("superfluid {tag} is installed at {}", i.root.display());
                    return Ok(());
                }
                Some(i) if !force && newer(i.installed_release().as_deref(), tag) => {
                    println!(
                        "superfluid {} is installed at {}, newer than {tag}: nothing is downgraded (--force installs {tag})",
                        i.installed_release().unwrap_or_default(),
                        i.root.display()
                    );
                    return Ok(());
                }
                Some(i) if i.writable() => i,
                _ => Install::for_install()?,
            };
            dest.install(&source, &release, &mut |line| eprintln!("{line}"))?;
            println!("superfluid {tag} is installed at {}", dest.root.display());
            Ok(())
        }
        Action::Status => {
            println!("tested:    {RELEASE}");
            let code = match Install::locate()? {
                None => {
                    let dest = Install::for_install()?;
                    println!(
                        "installed: nothing (`basert superfluid install` puts the newest release at {})",
                        dest.root.display()
                    );
                    1
                }
                Some(i) => {
                    let have = i
                        .installed_release()
                        .unwrap_or_else(|| "a release that does not say".to_string());
                    println!(
                        "installed: {have} ({}, {})",
                        i.root.display(),
                        i.origin.said()
                    );
                    let engine = crate::serve::engine_dir().ok().flatten();
                    let home = crate::serve::superfluid_home().ok();
                    match i.probe(engine.as_deref(), home.as_deref()).ready {
                        Ok(()) => {
                            println!("engine:    ready to serve");
                            0
                        }
                        Err(why) => {
                            println!("engine:    cannot serve: {why}");
                            2
                        }
                    }
                }
            };
            std::process::exit(code)
        }
    }
}

/// The release tarball's target triple for this machine.
pub fn target() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        ("linux", "aarch64") => Ok("aarch64-unknown-linux-gnu"),
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-gnu"),
        (os, arch) => bail!(
            "superfluid, which basert serve runs, publishes no release for {os}/{arch}; build it \
             from https://github.com/{REPO} and set {DIR_ENV} to a directory holding bin/superfluid"
        ),
    }
}

/// basert's own place for a superfluid it installs when the directory beside
/// it cannot be written: under the user's local data directory
/// (`~/Library/Application Support/baseRT` on macOS, `$XDG_DATA_HOME/baseRT`
/// or `~/.local/share/baseRT` on Linux).
pub fn user_data_dir() -> Result<PathBuf> {
    let data = dirs::data_local_dir()
        .context("no data directory to install superfluid in (set BASERT_SUPERFLUID_DIR)")?;
    Ok(data.join("baseRT").join("superfluid"))
}

/// Where an install was looked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// `$BASERT_SUPERFLUID_DIR`.
    Env,
    /// `superfluid/` beside basert: the one-shot installer's layout.
    BesideBasert,
    /// [`user_data_dir`].
    UserData,
}

impl Origin {
    /// For a message: where this is.
    fn said(self) -> &'static str {
        match self {
            Origin::Env => "named by BASERT_SUPERFLUID_DIR",
            Origin::BesideBasert => "beside basert",
            Origin::UserData => "basert's data directory",
        }
    }
}

/// A superfluid install: `root/bin/superfluid`, with the runtime workers in
/// `root/libexec/superfluid/`, where superfluid looks for them.
#[derive(Debug, Clone)]
pub struct Install {
    pub root: PathBuf,
    pub origin: Origin,
}

impl Install {
    /// Where superfluid is looked for, in order: `$BASERT_SUPERFLUID_DIR`
    /// alone when set; else beside basert, then the user's data directory.
    pub fn candidates() -> Result<Vec<Install>> {
        if let Some(dir) = std::env::var_os(DIR_ENV).filter(|d| !d.is_empty()) {
            return Ok(vec![Install {
                root: PathBuf::from(dir),
                origin: Origin::Env,
            }]);
        }
        let exe = std::env::current_exe().context("this program's own path")?;
        let dir = exe.parent().context("this program's own directory")?;
        Ok(vec![
            Install {
                root: dir.join("superfluid"),
                origin: Origin::BesideBasert,
            },
            Install {
                root: user_data_dir()?,
                origin: Origin::UserData,
            },
        ])
    }

    /// The first place with a superfluid installed.
    pub fn locate() -> Result<Option<Install>> {
        Ok(Self::candidates()?.into_iter().find(Install::is_installed))
    }

    /// Where an install goes: the first place that can be written, else the
    /// last, whose own error then says why not.
    pub fn for_install() -> Result<Install> {
        let candidates = Self::candidates()?;
        let last = candidates.len() - 1;
        Ok(candidates
            .into_iter()
            .enumerate()
            .find(|(k, c)| *k == last || c.writable())
            .map(|(_, c)| c)
            .expect("at least one candidate"))
    }

    /// Whether an install can be made or replaced here: its parent takes a
    /// new directory.
    pub fn writable(&self) -> bool {
        let parent = self.parent();
        if !parent.is_dir() && fs::create_dir_all(&parent).is_err() {
            return false;
        }
        tempfile::Builder::new()
            .prefix(".superfluid-probe-")
            .tempdir_in(&parent)
            .is_ok()
    }

    fn parent(&self) -> PathBuf {
        match self.root.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    }

    pub fn binary(&self) -> PathBuf {
        self.root.join("bin").join("superfluid")
    }

    pub fn is_installed(&self) -> bool {
        is_executable(&self.binary())
    }

    /// The release installed here: the stamp an install writes, else what the
    /// binary says of itself (a copy put here by hand). `None` when nothing is
    /// installed.
    pub fn installed_release(&self) -> Option<String> {
        if !self.is_installed() {
            return None;
        }
        if let Ok(stamp) = fs::read_to_string(self.root.join(STAMP)) {
            let stamp = stamp.trim();
            if !stamp.is_empty() {
                return Some(stamp.to_string());
            }
        }
        let out = Command::new(self.binary()).arg("--version").output().ok()?;
        // `superfluid --version` prints "superfluid 0.1.3".
        let text = String::from_utf8_lossy(&out.stdout);
        let version = text.split_whitespace().nth(1)?;
        Some(match version.strip_prefix('v') {
            Some(_) => version.to_string(),
            None => format!("v{version}"),
        })
    }

    /// Ask this superfluid whether its basert runtime can serve the engine
    /// library in `engine` (`None`: the one `BASERT_LIB` or superfluid's own
    /// search finds), with `home` as its `SUPERFLUID_HOME`. The answer is
    /// superfluid's: `runtimes --json` reports the runtime ready, or why not.
    pub fn probe(&self, engine: Option<&Path>, home: Option<&Path>) -> Probe {
        let mut command = Command::new(self.binary());
        command
            .args(["runtimes", "--json"])
            .env("SUPERFLUID_NO_UPDATE_CHECK", "1")
            .stdin(Stdio::null());
        if let Some(dir) = engine {
            command.env("BASERT_LIB", dir);
        }
        if let Some(dir) = home {
            command.env("SUPERFLUID_HOME", dir);
        }
        match command.output() {
            Ok(out) => Probe::from_report(&out.stdout, &out.stderr, out.status.success()),
            Err(e) => Probe::failed(format!("{} could not be run: {e}", self.binary().display())),
        }
    }

    /// Download `release`, check its SHA-256 against the one the release
    /// publishes, unpack it beside the install and swap it in. An install
    /// that is there stays until the new one is complete, and comes back if
    /// the swap fails.
    pub fn install(
        &self,
        source: &Source,
        release: &Release,
        say: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let tag = &release.tag;
        let name = format!("superfluid-{}", target()?);
        let parent = self.parent();
        fs::create_dir_all(&parent).with_context(|| format!("creating {}", parent.display()))?;
        // Beside the install, so the final rename stays on one filesystem.
        let stage = tempfile::Builder::new()
            .prefix(".superfluid-new-")
            .tempdir_in(&parent)
            .with_context(|| format!("creating a staging directory in {}", parent.display()))?;
        let tarball = stage.path().join(format!("{name}.tar.gz"));
        let checksum = stage.path().join(format!("{name}.tar.gz.sha256"));

        say(&format!(
            "downloading superfluid {tag} ({name}.tar.gz) from {}",
            source.describe(release, &format!("{name}.tar.gz"))
        ));
        source.fetch(release, &format!("{name}.tar.gz"), &tarball)?;
        source.fetch(release, &format!("{name}.tar.gz.sha256"), &checksum)?;
        let want = fs::read_to_string(&checksum)
            .with_context(|| format!("reading {}", checksum.display()))?
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let have = sha256_of(&tarball)?;
        if want.len() != 64 || want != have {
            bail!(
                "{name}.tar.gz: SHA-256 is {have}, the release's checksum file says {want:?}: \
                 the download is not the published file"
            );
        }

        let unpacked = stage.path().join("unpacked");
        fs::create_dir(&unpacked)?;
        let status = Command::new("tar")
            .arg("-xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(&unpacked)
            .status()
            .context("running tar")?;
        if !status.success() {
            bail!("tar -xzf {} failed ({status})", tarball.display());
        }
        let fresh = unpacked.join(&name);
        if !is_executable(&fresh.join("bin").join("superfluid")) {
            bail!("{name}.tar.gz holds no {name}/bin/superfluid");
        }
        fs::write(fresh.join(STAMP), format!("{tag}\n"))?;

        let old = parent.join(format!(".superfluid-old-{}", std::process::id()));
        let had_old = self.root.exists();
        if had_old {
            fs::rename(&self.root, &old).with_context(|| {
                format!(
                    "moving the installed superfluid aside ({})",
                    self.root.display()
                )
            })?;
        }
        if let Err(e) = fs::rename(&fresh, &self.root) {
            if had_old {
                let _ = fs::rename(&old, &self.root);
            }
            return Err(e)
                .with_context(|| format!("installing superfluid into {}", self.root.display()));
        }
        if had_old {
            let _ = fs::remove_dir_all(&old);
        }
        say(&format!(
            "installed superfluid {tag} at {}",
            self.root.display()
        ));
        Ok(())
    }
}

/// What a superfluid says of its basert runtime against an engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// Ready, or why not, in superfluid's words.
    pub ready: Result<(), String>,
    /// The engine ABI number in use, when superfluid reports one (0.1.3 on).
    pub abi: Option<u64>,
}

impl Probe {
    fn failed(why: String) -> Probe {
        Probe {
            ready: Err(why),
            abi: None,
        }
    }

    /// The basert row of `superfluid runtimes --json`: its `ready` is `true`
    /// or the reason it is not, from 0.1.2 on; `abi` arrives with 0.1.3.
    pub fn from_report(stdout: &[u8], stderr: &[u8], ok: bool) -> Probe {
        let text = String::from_utf8_lossy(stdout);
        let report: serde_json::Value = match serde_json::from_str(text.trim()) {
            Ok(v) => v,
            Err(_) => {
                let err = String::from_utf8_lossy(stderr);
                let said = [err.trim(), text.trim()]
                    .into_iter()
                    .find(|s| !s.is_empty())
                    .and_then(|s| s.lines().last())
                    .unwrap_or("nothing");
                let what = if ok { "answered no report" } else { "failed" };
                return Probe::failed(format!("`superfluid runtimes --json` {what}: {said}"));
            }
        };
        let Some(row) = report
            .as_array()
            .into_iter()
            .flatten()
            .find(|r| r["id"] == RUNTIME_ID)
        else {
            return Probe::failed("this superfluid ships no basert runtime".to_string());
        };
        let ready = match &row["ready"] {
            serde_json::Value::Bool(true) => Ok(()),
            serde_json::Value::String(why) => Err(why.clone()),
            _ => Err(row["state"].as_str().unwrap_or("not ready").to_string()),
        };
        Probe {
            ready,
            abi: row["abi"].as_u64(),
        }
    }
}

/// Where a release's files come from.
pub enum Source {
    /// The release's assets on GitHub, or where the listing that named it
    /// says they are.
    GitHub,
    /// A directory holding the tarball and its `.sha256` ([`SOURCE_ENV`]).
    Dir(PathBuf),
}

/// A superfluid release to install: its tag, and the URL of each asset its
/// listing gave, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub tag: String,
    urls: Vec<(String, String)>,
}

impl Release {
    /// [`RELEASE`], on GitHub.
    pub fn tested() -> Release {
        Release {
            tag: RELEASE.to_string(),
            urls: Vec::new(),
        }
    }

    /// Where the asset `file` is: the URL the listing gave for it, else
    /// GitHub's download URL for this tag. A mirror may keep a checksum
    /// elsewhere than its tarball.
    fn url(&self, file: &str) -> String {
        match self.urls.iter().find(|(name, _)| name == file) {
            Some((_, url)) => url.clone(),
            None => format!(
                "https://github.com/{REPO}/releases/download/{}/{file}",
                self.tag
            ),
        }
    }
}

impl Source {
    pub fn locate() -> Source {
        match std::env::var_os(SOURCE_ENV).filter(|d| !d.is_empty()) {
            Some(dir) => Source::Dir(PathBuf::from(dir)),
            None => Source::GitHub,
        }
    }

    fn fetch(&self, release: &Release, file: &str, dest: &Path) -> Result<()> {
        match self {
            Source::Dir(dir) => {
                let from = dir.join(file);
                fs::copy(&from, dest)
                    .with_context(|| format!("copying {}", from.display()))
                    .map(|_| ())
            }
            Source::GitHub => {
                let url = release.url(file);
                // GitHub answers an asset with a redirect to its storage, which
                // the agent follows.
                let response = agent(Duration::from_secs(600))
                    .get(&url)
                    .call()
                    .with_context(|| format!("downloading {url}"))?;
                let mut body = response.into_body().into_reader();
                let mut out = fs::File::create(dest)
                    .with_context(|| format!("creating {}", dest.display()))?;
                std::io::copy(&mut body, &mut out).with_context(|| format!("reading {url}"))?;
                Ok(())
            }
        }
    }
}

impl Source {
    /// Where `file` of `release` comes from, for a message.
    fn describe(&self, release: &Release, file: &str) -> String {
        match self {
            Source::GitHub => release.url(file),
            Source::Dir(dir) => dir.display().to_string(),
        }
    }
}

/// An HTTP agent whose server, gone quiet, fails the request rather than
/// hanging a `basert serve` start; a body may take up to `body`.
fn agent(body: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_recv_body(Some(body))
        .build()
        .into()
}

/// The release a start, or `basert superfluid install`, installs from
/// `source`: from GitHub the newest that serves this engine, else the one
/// this basert was tested with, saying why; from a directory, the one it holds.
pub fn wanted(source: &Source, say: &mut dyn FnMut(&str)) -> Release {
    if let Source::Dir(_) = source {
        return Release::tested();
    }
    let found = target().and_then(|target| {
        let listing = releases()?;
        newest_in(&listing, target, ENGINE_ABI).with_context(|| {
            format!("no superfluid release there serves engine ABI {ENGINE_ABI} on {target}")
        })
    });
    found.unwrap_or_else(|why| {
        say(&format!(
            "the newest superfluid release could not be chosen ({why:#}); {RELEASE}, the release this basert was tested with, is installed"
        ));
        Release::tested()
    })
}

/// superfluid's release listing, or the one `$BASERT_SUPERFLUID_RELEASES` names.
fn releases() -> Result<serde_json::Value> {
    let named = std::env::var(RELEASES_ENV).ok().filter(|v| !v.is_empty());
    let at = named.as_deref().unwrap_or(RELEASES);
    let text = if at.starts_with("https://") || at.starts_with("http://") {
        agent(Duration::from_secs(60))
            .get(at)
            .header("Accept", "application/vnd.github+json")
            .call()
            .and_then(|r| r.into_body().read_to_string())
            .with_context(|| format!("reading the release listing {at}"))?
    } else {
        fs::read_to_string(at).with_context(|| format!("reading the release listing {at}"))?
    };
    serde_json::from_str(&text).with_context(|| format!("the release listing {at}"))
}

/// The newest release in a listing of GitHub's form that is neither a draft
/// nor a pre-release and publishes, for `target`, the tarball, its checksum
/// and the marker of engine ABI `abi`.
pub fn newest_in(listing: &serde_json::Value, target: &str, abi: u32) -> Option<Release> {
    let tarball = format!("superfluid-{target}.tar.gz");
    let checksum = format!("{tarball}.sha256");
    let marker = format!("superfluid-basert-abi-{abi}.txt");
    listing
        .as_array()?
        .iter()
        .filter(|r| !r["draft"].as_bool().unwrap_or(false))
        .filter(|r| !r["prerelease"].as_bool().unwrap_or(false))
        .filter_map(|r| {
            let tag = r["tag_name"].as_str()?;
            let assets = r["assets"].as_array()?;
            let named = |name: &str| assets.iter().find(|a| a["name"] == name);
            let url = |name: &str| -> Option<(String, String)> {
                let at = named(name)?["browser_download_url"].as_str()?;
                Some((name.to_string(), at.to_string()))
            };
            named(&marker)?;
            Some((
                release_key(tag)?,
                Release {
                    tag: tag.to_string(),
                    urls: vec![url(&tarball)?, url(&checksum)?],
                },
            ))
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, release)| release)
}

/// `v0.1.3` as `[0, 1, 3]`; `None` for a tag that is not a release number.
fn release_key(tag: &str) -> Option<Vec<u64>> {
    let digits = tag.strip_prefix('v').unwrap_or(tag);
    digits.split('.').map(|part| part.parse().ok()).collect()
}

/// Whether the release `have` is newer than `than`.
fn newer(have: Option<&str>, than: &str) -> bool {
    match (have.and_then(release_key), release_key(than)) {
        (Some(have), Some(than)) => have > than,
        _ => false,
    }
}

/// Whether a start may download superfluid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Download {
    /// Ask first: a terminal is there to answer.
    Ask,
    /// Download without asking, saying so.
    Auto,
    /// Never: name the command instead.
    Never,
}

/// The policy for this start: `$BASERT_SUPERFLUID_AUTO_INSTALL` decides when
/// set (`0` never, `1` without asking); otherwise a terminal is asked, and a
/// start with none (Local's, a service's) downloads and says so.
pub fn policy() -> Download {
    match std::env::var(AUTO_INSTALL_ENV)
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("0") | Some("no") | Some("false") => Download::Never,
        Some("1") | Some("yes") | Some("true") => Download::Auto,
        _ if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() => Download::Ask,
        _ => Download::Auto,
    }
}

/// The superfluid a `basert serve` start runs, as a path to its binary: the
/// first install whose basert runtime superfluid itself reports ready for
/// the engine in `engine`, under `home` as its `SUPERFLUID_HOME`. `say` gets
/// a line for each step taken on the user's behalf.
pub fn ensure(engine: Option<&Path>, home: &Path, say: &mut dyn FnMut(&str)) -> Result<PathBuf> {
    ensure_with(policy(), &Source::locate(), engine, home, say)
}

pub fn ensure_with(
    policy: Download,
    source: &Source,
    engine: Option<&Path>,
    home: &Path,
    say: &mut dyn FnMut(&str),
) -> Result<PathBuf> {
    // Every install found is asked; the first that can serve runs.
    let mut unready: Vec<(Install, Option<String>, String)> = Vec::new();
    for install in Install::candidates()?
        .into_iter()
        .filter(Install::is_installed)
    {
        match install.probe(engine, Some(home)).ready {
            Ok(()) => return Ok(install.binary()),
            Err(why) => {
                let have = install.installed_release();
                unready.push((install, have, why));
            }
        }
    }
    // Looked up only when it may be downloaded: a start forbidden to download
    // does not wait on GitHub.
    let release = match policy {
        Download::Never => None,
        Download::Ask | Download::Auto => Some(wanted(source, say)),
    };
    let tag = release
        .as_ref()
        .map_or(RELEASE, |r| r.tag.as_str())
        .to_string();
    for (install, have, why) in &unready {
        // An install that cannot serve and is the release itself makes the
        // engine the problem, which no download mends.
        if have.as_deref() == Some(tag.as_str()) {
            bail!(
                "superfluid {tag} at {} cannot serve this engine: {why}",
                install.root.display()
            );
        }
        say(&format!(
            "superfluid {} at {} cannot serve this engine ({why}){}",
            have.as_deref().unwrap_or("of a release it does not say"),
            install.root.display(),
            if release.is_some() {
                format!("; {tag} can")
            } else {
                String::new()
            }
        ));
    }
    let dest = match unready.iter().map(|(i, _, _)| i).find(|i| i.writable()) {
        Some(found) => found.clone(),
        None => Install::for_install()?,
    };
    let hint = format!(
        "install it with\n  basert superfluid install\n(the newest superfluid release that serves \
         this engine, from https://github.com/{REPO}/releases, into {})",
        dest.root.display()
    );
    let release = match (policy, release) {
        (Download::Never, _) | (_, None) => {
            if unready.is_empty() {
                bail!("superfluid is not installed, and {AUTO_INSTALL_ENV}=0 forbids a download at a start; {hint}");
            }
            bail!("{AUTO_INSTALL_ENV}=0 forbids a download at a start; {hint}");
        }
        (Download::Ask, Some(release)) => {
            if unready.is_empty() {
                eprintln!("basert serve runs superfluid, which is not installed.");
            }
            if !confirm(&format!(
                "Download and install superfluid {tag} into {} now? [Y/n] ",
                dest.root.display()
            ))? {
                bail!("superfluid {tag} is not installed; {hint}");
            }
            release
        }
        (Download::Auto, Some(release)) => {
            say(&format!(
                "installing superfluid {tag} into {} ({AUTO_INSTALL_ENV}=0 forbids this)",
                dest.root.display()
            ));
            release
        }
    };
    dest.install(source, &release, say)?;
    match dest.probe(engine, Some(home)).ready {
        Ok(()) => Ok(dest.binary()),
        Err(why) => bail!(
            "superfluid {tag}, installed at {}, cannot serve this engine: {why}",
            dest.root.display()
        ),
    }
}

/// A yes/no question on stderr, answered on stdin; an empty answer is yes.
fn confirm(question: &str) -> Result<bool> {
    eprint!("{question}");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "" | "y" | "yes"
    ))
}

fn sha256_of(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A release directory holding `superfluid-<target>.tar.gz` and its
    /// checksum, the tarball unpacking to superfluid's layout with a shell
    /// script for a binary: `--version` answers `version`, `runtimes --json`
    /// reports the basert runtime ready unless `FAKE_SUPERFLUID_UNREADY`
    /// names a reason for the version `FAKE_SUPERFLUID_UNREADY_VERSION`.
    fn release(dir: &Path, version: &str, checksum_ok: bool) -> PathBuf {
        let name = format!("superfluid-{}", target().unwrap());
        let tree = dir.join("tree");
        let bin = tree.join(&name).join("bin");
        let libexec = tree.join(&name).join("libexec").join("superfluid");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(&libexec).unwrap();
        fs::write(bin.join("superfluid"), fake_superfluid(version)).unwrap();
        fs::write(libexec.join("superfluid-worker-basert"), "#!/bin/sh\n").unwrap();
        for p in [
            bin.join("superfluid"),
            libexec.join("superfluid-worker-basert"),
        ] {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let assets = dir.join("assets");
        fs::create_dir_all(&assets).unwrap();
        let tarball = assets.join(format!("{name}.tar.gz"));
        assert!(Command::new("tar")
            .arg("-czf")
            .arg(&tarball)
            .arg("-C")
            .arg(&tree)
            .arg(&name)
            .status()
            .unwrap()
            .success());
        let sum = if checksum_ok {
            sha256_of(&tarball).unwrap()
        } else {
            "0".repeat(64)
        };
        fs::write(
            assets.join(format!("{name}.tar.gz.sha256")),
            format!("{sum}  {name}.tar.gz\n"),
        )
        .unwrap();
        assets
    }

    pub(super) fn fake_superfluid(version: &str) -> String {
        format!(
            "#!/bin/sh\n\
             if [ \"$1\" = \"--version\" ]; then echo \"superfluid {version}\"; exit 0; fi\n\
             if [ \"$1\" = \"runtimes\" ]; then\n\
               if [ -n \"${{FAKE_SUPERFLUID_UNREADY:-}}\" ] && [ \"${{FAKE_SUPERFLUID_UNREADY_VERSION:-}}\" = \"{version}\" ]; then\n\
                 echo \"[{{\\\"id\\\":\\\"basert\\\",\\\"state\\\":\\\"not ready\\\",\\\"ready\\\":\\\"$FAKE_SUPERFLUID_UNREADY\\\",\\\"version\\\":\\\"{version}\\\"}}]\"\n\
               else\n\
                 echo \"[{{\\\"id\\\":\\\"basert\\\",\\\"state\\\":\\\"ready\\\",\\\"ready\\\":true,\\\"version\\\":\\\"{version}\\\",\\\"abi\\\":1}}]\"\n\
               fi\n\
               exit 0\n\
             fi\n\
             printf 'ARGS:'; for a in \"$@\"; do printf ' %s' \"$a\"; done; printf '\\n'\n"
        )
    }

    fn install_at(root: PathBuf) -> Install {
        Install {
            root,
            origin: Origin::Env,
        }
    }

    #[test]
    fn an_install_unpacks_the_release_and_stamps_it() {
        let tmp = tempfile::tempdir().unwrap();
        let assets = release(tmp.path(), "0.1.3", true);
        let install = install_at(tmp.path().join("bin").join("superfluid"));
        assert!(!install.is_installed());
        assert_eq!(install.installed_release(), None);
        let mut said = Vec::new();
        install
            .install(&Source::Dir(assets), &Release::tested(), &mut |l| {
                said.push(l.to_string())
            })
            .unwrap();
        assert!(install.is_installed());
        assert!(install
            .root
            .join("libexec")
            .join("superfluid")
            .join("superfluid-worker-basert")
            .is_file());
        assert_eq!(install.installed_release().as_deref(), Some(RELEASE));
        assert_eq!(said.len(), 2, "{said:?}");
        assert!(said[0].starts_with(&format!("downloading superfluid {RELEASE} (")));
        // Nothing of the staging is left behind.
        let left: Vec<_> = fs::read_dir(tmp.path().join("bin"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(left, ["superfluid"]);
    }

    #[test]
    fn a_wrong_checksum_refuses_the_download_and_keeps_what_is_installed() {
        let tmp = tempfile::tempdir().unwrap();
        let good = release(&tmp.path().join("good"), "0.1.3", true);
        let bad = release(&tmp.path().join("bad"), "0.1.3", false);
        let install = install_at(tmp.path().join("superfluid"));
        install
            .install(&Source::Dir(good), &Release::tested(), &mut |_| {})
            .unwrap();
        fs::write(install.root.join("marker"), "kept").unwrap();
        let err = install
            .install(&Source::Dir(bad), &Release::tested(), &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("the download is not the published file"),
            "{err}"
        );
        assert_eq!(
            fs::read_to_string(install.root.join("marker")).unwrap(),
            "kept"
        );
        assert_eq!(install.installed_release().as_deref(), Some(RELEASE));
    }

    #[test]
    fn a_missing_asset_is_an_error_that_names_it() {
        let tmp = tempfile::tempdir().unwrap();
        let install = install_at(tmp.path().join("superfluid"));
        let err = install
            .install(
                &Source::Dir(tmp.path().join("nowhere")),
                &Release::tested(),
                &mut |_| {},
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("copying") && err.contains(".tar.gz"), "{err}");
        assert!(!install.root.exists());
    }

    #[test]
    fn the_release_installed_is_the_stamp_else_what_the_binary_says() {
        let tmp = tempfile::tempdir().unwrap();
        let assets = release(tmp.path(), "0.0.9", true);
        let install = install_at(tmp.path().join("superfluid"));
        install
            .install(&Source::Dir(assets), &Release::tested(), &mut |_| {})
            .unwrap();
        assert_eq!(install.installed_release().as_deref(), Some(RELEASE));
        // A copy put here by hand has no stamp: the binary is asked.
        fs::remove_file(install.root.join(STAMP)).unwrap();
        assert_eq!(install.installed_release().as_deref(), Some("v0.0.9"));
        // A second install replaces it, stamp included.
        let newer = release(&tmp.path().join("again"), "0.1.3", true);
        install
            .install(&Source::Dir(newer), &Release::tested(), &mut |_| {})
            .unwrap();
        assert_eq!(install.installed_release().as_deref(), Some(RELEASE));
        assert!(!tmp
            .path()
            .join(format!(".superfluid-old-{}", std::process::id()))
            .exists());
    }

    #[test]
    fn the_target_is_one_superfluid_publishes() {
        let t = target().unwrap();
        assert!(
            [
                "aarch64-apple-darwin",
                "aarch64-unknown-linux-gnu",
                "x86_64-unknown-linux-gnu"
            ]
            .contains(&t),
            "{t}"
        );
    }

    #[test]
    fn the_probe_reads_the_basert_row_of_either_generation() {
        // 0.1.2: ready is true or the reason; no abi.
        let p = Probe::from_report(
            br#"[{"id":"llamacpp","state":"not installed","ready":"no worker"},{"id":"basert","state":"ready","ready":true,"version":"0.1.2"}]"#,
            b"",
            true,
        );
        assert_eq!(
            p,
            Probe {
                ready: Ok(()),
                abi: None
            }
        );
        // 0.1.3: the abi comes along.
        let p = Probe::from_report(
            br#"[{"id":"basert","state":"ready","ready":true,"version":"0.1.3","abi":1,"requires":{"abi":1}}]"#,
            b"",
            true,
        );
        assert_eq!(
            p,
            Probe {
                ready: Ok(()),
                abi: Some(1)
            }
        );
        let p = Probe::from_report(
            br#"[{"id":"basert","state":"not ready","ready":"libbaseRT 0.4.0 states no ABI number and is not a 0.3 library"}]"#,
            b"",
            true,
        );
        assert_eq!(
            p.ready.unwrap_err(),
            "libbaseRT 0.4.0 states no ABI number and is not a 0.3 library"
        );
        // No basert row, no report, a failure: each says what it saw.
        assert!(
            Probe::from_report(br#"[{"id":"mlx","ready":true}]"#, b"", true)
                .ready
                .unwrap_err()
                .contains("no basert runtime")
        );
        let why = Probe::from_report(b"", b"error: unknown flag --json\n", false)
            .ready
            .unwrap_err();
        assert!(
            why.contains("failed") && why.contains("unknown flag --json"),
            "{why}"
        );
        let why = Probe::from_report(b"not json", b"", true)
            .ready
            .unwrap_err();
        assert!(why.contains("answered no report"), "{why}");
    }

    #[test]
    fn the_probe_runs_the_binary_with_the_engine_and_home_named() {
        let tmp = tempfile::tempdir().unwrap();
        let assets = release(tmp.path(), "0.1.3", true);
        let install = install_at(tmp.path().join("superfluid"));
        install
            .install(&Source::Dir(assets), &Release::tested(), &mut |_| {})
            .unwrap();
        let p = install.probe(Some(Path::new("/lib")), Some(Path::new("/home")));
        assert_eq!(
            p,
            Probe {
                ready: Ok(()),
                abi: Some(1)
            }
        );
        let p = install_at(tmp.path().join("nowhere")).probe(None, None);
        assert!(p.ready.unwrap_err().contains("could not be run"));
    }

    /// A release as GitHub's listing has it, with assets of these names.
    fn listed(tag: &str, names: &[String]) -> serde_json::Value {
        let assets: Vec<serde_json::Value> = names
            .iter()
            .map(|name| {
                serde_json::json!({
                    "name": name,
                    "browser_download_url": format!("https://github.com/{REPO}/releases/download/{tag}/{name}"),
                })
            })
            .collect();
        serde_json::json!({"tag_name": tag, "draft": false, "prerelease": false, "assets": assets})
    }

    #[test]
    fn the_newest_release_is_the_highest_that_serves_the_engine_here() {
        let t = "aarch64-apple-darwin";
        let tarball = format!("superfluid-{t}.tar.gz");
        let checksum = format!("{tarball}.sha256");
        let marker = |abi: u32| format!("superfluid-basert-abi-{abi}.txt");
        let whole =
            |tag: &str, abi: u32| listed(tag, &[tarball.clone(), checksum.clone(), marker(abi)]);
        let mut draft = whole("v0.9.0", 1);
        draft["draft"] = true.into();
        let mut pre = whole("v0.8.0", 1);
        pre["prerelease"] = true.into();
        let listing = serde_json::json!([
            draft,
            pre,
            whole("v0.7.0", 2),
            listed("v0.6.0", &[tarball.clone(), marker(1)]),
            listed("v0.5.0", &[tarball.clone(), checksum.clone()]),
            whole("nightly", 1),
            whole("v0.1.9", 1),
            whole("v0.1.10", 1),
        ]);
        let newest = newest_in(&listing, t, 1).unwrap();
        assert_eq!(newest.tag, "v0.1.10", "by number, not as text");
        assert_eq!(
            newest.url(&tarball),
            format!("https://github.com/{REPO}/releases/download/v0.1.10/{tarball}")
        );
        // Each asset where the listing says it is, not beside the tarball.
        let mut mirrored = whole("v0.2.0", 1);
        mirrored["assets"][1]["browser_download_url"] =
            format!("https://mirror/checksums/{checksum}").into();
        let release = newest_in(&serde_json::json!([mirrored]), t, 1).unwrap();
        assert_eq!(
            release.url(&checksum),
            format!("https://mirror/checksums/{checksum}")
        );
        assert_eq!(
            Release::tested().url(&checksum),
            format!("https://github.com/{REPO}/releases/download/{RELEASE}/{checksum}")
        );
        assert_eq!(newest_in(&listing, t, 2).unwrap().tag, "v0.7.0");
        assert_eq!(newest_in(&listing, "x86_64-unknown-linux-gnu", 1), None);
        let refused = serde_json::json!({"message": "API rate limit exceeded"});
        assert_eq!(newest_in(&refused, t, 1), None);
    }

    #[test]
    fn nothing_installed_is_downgraded() {
        assert!(newer(Some("v0.1.10"), "v0.1.9"));
        assert!(!newer(Some("v0.1.3"), "v0.1.3"));
        assert!(!newer(Some("v0.1.2"), "v0.1.3"));
        assert!(!newer(None, "v0.1.3"));
        assert!(!newer(Some("a release that does not say"), "v0.1.3"));
    }

    #[test]
    fn the_engine_abi_is_the_one_the_header_states() {
        // The headers are at the repository's root: four directories up here,
        // three in the public repository, where this crate is `base-convert/`.
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let header = [
            "../../../../include/baseRT/baseRT.h",
            "../../../include/baseRT/baseRT.h",
        ]
        .iter()
        .map(|rel| crate_dir.join(rel))
        .find(|p| p.is_file())
        .expect("include/baseRT/baseRT.h at the repository's root");
        let text = fs::read_to_string(&header).unwrap();
        let stated = text
            .lines()
            .find_map(|l| l.strip_prefix("#define BASERT_ABI_VERSION "))
            .map(|n| n.trim().parse::<u32>().unwrap());
        assert_eq!(stated, Some(ENGINE_ABI), "{}", header.display());
    }

    /// `ensure_with` against installs named by `BASERT_SUPERFLUID_DIR` is
    /// covered end to end in tests/serve_launcher.rs, where the environment
    /// is the launched process's own; here the pieces that need none.
    #[test]
    fn the_policy_words_are_read_as_set() {
        // The environment is process-wide: this test alone touches the variable,
        // and only to values, never to unset it mid-flight.
        for (value, want) in [
            ("0", Download::Never),
            ("no", Download::Never),
            ("1", Download::Auto),
            ("yes", Download::Auto),
        ] {
            std::env::set_var(AUTO_INSTALL_ENV, value);
            assert_eq!(policy(), want, "{value}");
        }
        std::env::remove_var(AUTO_INSTALL_ENV);
        // Unset: a terminal is asked, and a start without one downloads.
        let terminal = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
        assert_eq!(
            policy(),
            if terminal {
                Download::Ask
            } else {
                Download::Auto
            }
        );
    }
}
