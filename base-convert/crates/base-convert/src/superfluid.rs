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
//! `superfluid/` beside basert, the one-shot installer's layout (and a
//! release bundle's, or a build's); basert's data directory
//! ([`user_data_dir`]), where basert installs when the one beside it cannot be
//! written (an app bundle's). A basert a package manager placed (Homebrew,
//! npm, pip, pipx, uv, cargo) installs only into its data directory, looked
//! in first: a package's folder is its manager's, replaced by an upgrade and
//! left behind by an uninstall. A superfluid already beside such a basert is
//! looked for after it and runs while it can serve, but nothing is installed
//! there. Each holds superfluid's own layout (`bin/superfluid`, the runtime
//! workers in `libexec/superfluid/`) off PATH: a superfluid the user
//! installed for themselves is neither shadowed nor run in its place.
//! An install replaces the directory whole, so one that holds anything but
//! superfluid (a prefix such as `~/.local`, which superfluid's own install
//! script fills) is never replaced.
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
pub const RELEASE: &str = "v0.1.6";

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
/// Written into an install basert made, as a package manager's copy carries
/// one: superfluid's own `self update` then leaves the copy to basert, which
/// picks the release that serves its engine, instead of moving it under
/// `~/.superfluid` and checking it against an engine that is not basert's.
const INSTALLED_BY: &str = "installed-by";
const OWNED_BY: &str = "basert: basert superfluid install";
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
    /// and whether it can serve the engine beside basert: the install a start
    /// would run, the first that can. Exits 0 when an installed superfluid
    /// can, 1 when none is installed, 2 when none installed can.
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
                Some(i) if i.takes_installs() && i.writable() => i,
                _ => Install::for_install()?,
            };
            // A server's workers start from this directory: replaced under
            // it, the next one would be another release's. Looked at before
            // the download, and again at the swap.
            drop(dest.unused()?);
            dest.install(&source, &release, &mut |line| eprintln!("{line}"))?;
            println!("superfluid {tag} is installed at {}", dest.root.display());
            Ok(())
        }
        Action::Status => {
            println!("tested:    {RELEASE}");
            let engine = crate::serve::engine_dir().ok().flatten();
            let home = crate::serve::superfluid_home().ok();
            let code = match Install::choose(engine.as_deref(), home.as_deref())? {
                None => {
                    let dest = Install::for_install()?;
                    println!(
                        "installed: nothing (`basert superfluid install` puts the newest release at {})",
                        dest.root.display()
                    );
                    1
                }
                Some((i, ready)) => {
                    i.claim();
                    let have = i
                        .installed_release()
                        .unwrap_or_else(|| "a release that does not say".to_string());
                    println!(
                        "installed: {have} ({}, {})",
                        i.root.display(),
                        i.origin.said()
                    );
                    match ready {
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

/// basert's own place for a superfluid it installs for a basert a package
/// manager placed, or when the directory beside basert cannot be written:
/// under the user's local data directory (`~/Library/Application
/// Support/baseRT` on macOS, `$XDG_DATA_HOME/baseRT` or `~/.local/share/baseRT`
/// on Linux).
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
    /// `superfluid/` beside a basert a package manager placed: run when it
    /// is there, never installed into.
    BesidePackage,
    /// [`user_data_dir`].
    UserData,
}

impl Origin {
    /// For a message: where this is.
    fn said(self) -> &'static str {
        match self {
            Origin::Env => "named by BASERT_SUPERFLUID_DIR",
            Origin::BesideBasert | Origin::BesidePackage => "beside basert",
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
    /// alone when set; else beside basert, then the user's data directory;
    /// for a basert a package manager placed, the other way round.
    pub fn candidates() -> Result<Vec<Install>> {
        if let Some(dir) = std::env::var_os(DIR_ENV).filter(|d| !d.is_empty()) {
            return Ok(vec![Install {
                root: PathBuf::from(dir),
                origin: Origin::Env,
            }]);
        }
        // The install directory, not the version a versions-layout basert
        // runs from: superfluid stays put while basert's versions change.
        let beside = crate::selfupdate::install_dir()?.join("superfluid");
        let data = Install {
            root: user_data_dir()?,
            origin: Origin::UserData,
        };
        let exe = std::env::current_exe().context("this program's own path")?;
        if crate::selfupdate::installed_as(&exe).by_package_manager() {
            // A package's folder is its manager's: Homebrew's keg and npm's
            // package are replaced by an upgrade, pip's is left behind by an
            // uninstall, cargo's is ~/.cargo/bin. One installed there before
            // still runs, after the data directory's.
            return Ok(vec![
                data,
                Install {
                    root: beside,
                    origin: Origin::BesidePackage,
                },
            ]);
        }
        Ok(vec![
            Install {
                root: beside,
                origin: Origin::BesideBasert,
            },
            data,
        ])
    }

    /// The first place with a superfluid installed.
    pub fn locate() -> Result<Option<Install>> {
        Ok(Self::candidates()?.into_iter().find(Install::is_installed))
    }

    /// Where an install goes: the first place that takes one and can be
    /// written, else the last that takes one, whose own error then says why not.
    pub fn for_install() -> Result<Install> {
        let places: Vec<Install> = Self::candidates()?
            .into_iter()
            .filter(Install::takes_installs)
            .collect();
        let last = places.len() - 1;
        Ok(places
            .into_iter()
            .enumerate()
            .find(|(k, c)| *k == last || c.writable())
            .map(|(_, c)| c)
            .expect("at least one candidate"))
    }

    /// Whether basert installs here: everywhere but a package's folder.
    pub fn takes_installs(&self) -> bool {
        self.origin != Origin::BesidePackage
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

    /// Mark an install basert made (beside it or in its data directory, not
    /// one `$BASERT_SUPERFLUID_DIR` names) as basert's, for a copy installed
    /// before installs carried the file. Best effort: a directory that cannot
    /// be written stays as it is.
    pub fn claim(&self) {
        if self.origin != Origin::Env
            && self.is_installed()
            && !self.root.join(INSTALLED_BY).exists()
        {
            let _ = fs::write(self.root.join(INSTALLED_BY), format!("{OWNED_BY}\n"));
        }
    }

    /// That this install's directory may be replaced whole, as an install
    /// replaces it: it holds superfluid alone (see [`not_superfluids`]), or
    /// nothing yet. A prefix such as `~/.local`, which superfluid's own install
    /// script fills and `$BASERT_SUPERFLUID_DIR` may name, holds the user's
    /// other files, which would be removed with it. `fresh` is the release
    /// about to take its place.
    pub fn replaceable(&self, fresh: &Path) -> Result<()> {
        if fs::symlink_metadata(&self.root).is_err() {
            return Ok(());
        }
        // A program of superfluid's name (a `cargo install superfluid`) is
        // not a directory to look in.
        let found = match fs::metadata(&self.root).is_ok_and(|m| m.is_dir()) {
            true => match not_superfluids(&self.root, fresh, Level::Top) {
                Some(found) => found,
                None => return Ok(()),
            },
            false => self.root.clone(),
        };
        let what = match found == self.root {
            true => format!(
                "{} is a file, not superfluid's directory",
                self.root.display()
            ),
            false => format!(
                "{} holds {}, which is not superfluid's",
                self.root.display(),
                found.display()
            ),
        };
        let then = match self.origin {
            Origin::Env => format!(
                "name a directory of superfluid's own in {DIR_ENV} (a new one: basert installs superfluid there)"
            ),
            _ => format!("move {} aside", found.display()),
        };
        bail!("{what}, and replacing superfluid there would remove it: nothing is changed; {then}")
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
        let _turn = self.take_turn(say)?;
        let staged = self.stage(source, release, say)?;
        self.swap_in(staged, say)
    }

    /// The lock file `kind` beside this install, made when missing; `None`
    /// where it cannot be (a directory basert cannot write, where it
    /// installs nothing either).
    fn lock(&self, kind: &str) -> Option<fs::File> {
        let name = self.root.file_name()?.to_string_lossy();
        let parent = self.parent();
        // Not there before a first install.
        let _ = fs::create_dir_all(&parent);
        let path = parent.join(format!(".{}.{kind}.lock", name.trim_start_matches('.')));
        fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .ok()
    }

    /// Installs into this directory take turns: one that finds another under
    /// way says so and waits for it, then looks again at what is there.
    pub fn take_turn(&self, say: &mut dyn FnMut(&str)) -> Result<Option<Held>> {
        let Some(file) = self.lock("install") else {
            return Ok(None);
        };
        if !flock(&file, libc::LOCK_EX | libc::LOCK_NB)? {
            say(&format!(
                "waiting for another basert installing superfluid into {}",
                self.root.display()
            ));
            flock(&file, libc::LOCK_EX)?;
        }
        Ok(Some(Held(file)))
    }

    /// Held by a start from asking this install until its server stops: no
    /// install swaps it in between. Waits only while one swaps.
    pub fn hold(&self) -> Option<Held> {
        let file = self.lock("use")?;
        flock(&file, libc::LOCK_SH).ok()?.then_some(Held(file))
    }

    /// That no server runs from this install (one `basert serve` started
    /// holds it; others are found by what runs from the directory) and none
    /// starts while the returned lock is held.
    pub fn unused(&self) -> Result<Option<Held>> {
        let file = self.lock("use");
        let free = match &file {
            Some(file) => flock(file, libc::LOCK_EX | libc::LOCK_NB)?,
            None => true,
        };
        let pids = crate::selfupdate::running_from(&self.root);
        if free && pids.is_empty() {
            return Ok(file.map(Held));
        }
        let who = match pids.is_empty() {
            true => "one basert serve started".to_string(),
            false => format!(
                "pid {}",
                pids.iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        bail!(
            "a server runs from {} ({who}): stop it, then run this again",
            self.root.display()
        )
    }

    /// The install a start would run for the engine in `engine`, asked as a
    /// start asks but nothing installed: the first found that can serve,
    /// else the first found, with why it cannot.
    pub fn choose(
        engine: Option<&Path>,
        home: Option<&Path>,
    ) -> Result<Option<(Install, Result<(), String>)>> {
        let mut first = None;
        for install in Self::candidates()?
            .into_iter()
            .filter(Install::is_installed)
        {
            let ready = install.probe(engine, home).ready;
            if ready.is_ok() {
                return Ok(Some((install, ready)));
            }
            first.get_or_insert((install, ready));
        }
        Ok(first)
    }

    /// The first half of [`Install::install`]: `release` downloaded, checked
    /// and unpacked beside this install, where it can be asked whether it
    /// serves an engine before it takes this one's place.
    pub fn stage(
        &self,
        source: &Source,
        release: &Release,
        say: &mut dyn FnMut(&str),
    ) -> Result<Staged> {
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
        check_sha256(&tarball, &checksum)?;

        let unpacked = stage.path().join("unpacked");
        fs::create_dir(&unpacked)?;
        untar(&tarball, &unpacked)?;
        let fresh = unpacked.join(&name);
        if !is_executable(&fresh.join("bin").join("superfluid")) {
            bail!("{name}.tar.gz holds no {name}/bin/superfluid");
        }
        fs::write(fresh.join(STAMP), format!("{tag}\n"))?;
        fs::write(fresh.join(INSTALLED_BY), format!("{OWNED_BY}\n"))?;
        Ok(Staged {
            _dir: stage,
            fresh: Install {
                root: fresh,
                origin: self.origin,
            },
            tag: tag.clone(),
            replacing: self.installed_release(),
        })
    }

    /// The second half of [`Install::install`]: the staged release takes this
    /// install's place.
    pub fn swap_in(&self, staged: Staged, say: &mut dyn FnMut(&str)) -> Result<()> {
        let tag = &staged.tag;
        let fresh = &staged.fresh.root;
        self.replaceable(fresh)?;
        // The last look, held through the swap: no server runs from it, and
        // none starts until it is done; and it holds what it held when the
        // release was staged.
        let _swapping = self.unused()?;
        let now = self.installed_release();
        if now != staged.replacing {
            bail!(
                "{} changed while superfluid {tag} was fetched (it holds {}, not {}): nothing \
                 is changed; run this again",
                self.root.display(),
                now.as_deref().unwrap_or("no superfluid"),
                staged.replacing.as_deref().unwrap_or("none")
            );
        }
        let parent = self.parent();
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
        if let Err(e) = fs::rename(fresh, &self.root) {
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

/// A release [`Install::stage`] unpacked beside an install: `fresh` can be
/// probed as an install of its own until [`Install::swap_in`] moves it into
/// place. Dropped instead, it is removed.
pub struct Staged {
    _dir: tempfile::TempDir,
    pub fresh: Install,
    pub tag: String,
    /// The release installed where it goes when it was staged.
    replacing: Option<String>,
}

/// A lock on an install, in a file beside its directory (which an install
/// replaces whole). Installs into the directory take turns on one
/// ([`Install::take_turn`]); a start holds the other shared from asking the
/// install until its server stops ([`Install::hold`],
/// [`Serving::hold_through_exec`]), and an install holds it exclusive while
/// it swaps ([`Install::unused`]).
pub struct Held(fs::File);

/// `flock(op)` on `file`: false when `LOCK_NB` is asked and another holds it.
fn flock(file: &fs::File, op: libc::c_int) -> std::io::Result<bool> {
    use std::os::fd::AsRawFd;
    loop {
        // SAFETY: flock on a descriptor this process holds open.
        if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
            return Ok(true);
        }
        let e = std::io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EWOULDBLOCK) => return Ok(false),
            _ => return Err(e),
        }
    }
}

/// The superfluid a start runs, and its hold on it.
pub struct Serving {
    pub binary: PathBuf,
    /// The release the install says it is (`vX.Y.Z`), when it says.
    pub release: Option<String>,
    held: Option<Held>,
}

impl Serving {
    /// The hold kept open across the `exec` that makes this process the
    /// server, and by each process the server starts: no install swaps
    /// superfluid under it while it runs.
    pub fn hold_through_exec(self) {
        use std::os::fd::IntoRawFd;
        if let Some(Held(file)) = self.held {
            let fd = file.into_raw_fd();
            // SAFETY: fcntl on a descriptor this process holds open.
            unsafe {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags >= 0 {
                    libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC);
                }
            }
        }
    }
}

/// Where in a release [`not_superfluids`] looks.
#[derive(Debug, Clone, Copy)]
enum Level {
    Top,
    Bin,
    Libexec,
    Other,
}

/// The first entry under `dir` that is not superfluid's, `at` that level of a
/// release: superfluid's is a name `fresh` (the same level of the release
/// about to replace it) has too, or one of superfluid's layout (`bin/` holding
/// `superfluid*`, `libexec/superfluid/`, the release's `LICENSE`, `NOTICE`,
/// `README.md` and `compat.json`, the `.release` and `installed-by` basert
/// writes). A link is superfluid's only where `fresh` has one (a release holds
/// none; superfluid's own install script links `bin/superfluid` to its
/// versions); a directory that cannot be read is not.
fn not_superfluids(dir: &Path, fresh: &Path, at: Level) -> Option<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Some(dir.to_path_buf());
    };
    for entry in entries {
        let Ok(entry) = entry else {
            return Some(dir.to_path_buf());
        };
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            return Some(path);
        };
        let theirs = fs::symlink_metadata(fresh.join(&name));
        if meta.file_type().is_symlink() {
            if !theirs.is_ok_and(|m| m.file_type().is_symlink()) {
                return Some(path);
            }
            continue;
        }
        let layout = match at {
            Level::Top => {
                matches!(
                    name.as_str(),
                    "bin" | "libexec" | "LICENSE" | "NOTICE" | "README.md" | "compat.json"
                ) || name == STAMP
                    || name == INSTALLED_BY
            }
            Level::Bin => name.starts_with("superfluid"),
            Level::Libexec => name == "superfluid",
            Level::Other => false,
        };
        if !layout && theirs.is_err() {
            return Some(path);
        }
        if meta.is_dir() {
            let inner = match (at, name.as_str()) {
                // superfluid's own directory: its workers and libraries.
                (Level::Libexec, "superfluid") => continue,
                (Level::Top, "bin") => Level::Bin,
                (Level::Top, "libexec") => Level::Libexec,
                _ => Level::Other,
            };
            if let Some(found) = not_superfluids(&path, &fresh.join(&name), inner) {
                return Some(found);
            }
        }
    }
    None
}

/// That `tarball` is the file whose SHA-256 the release's `checksum` file
/// (`<hex>  <name>`) states.
pub(crate) fn check_sha256(tarball: &Path, checksum: &Path) -> Result<()> {
    let name = tarball.file_name().unwrap_or_default().to_string_lossy();
    let want = fs::read_to_string(checksum)
        .with_context(|| format!("reading {}", checksum.display()))?
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let have = sha256_of(tarball)?;
    if want.len() != 64 || want != have {
        bail!(
            "{name}: SHA-256 is {have}, the release's checksum file says {want:?}: \
             the download is not the published file"
        );
    }
    Ok(())
}

pub(crate) fn untar(tarball: &Path, into: &Path) -> Result<()> {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(tarball)
        .arg("-C")
        .arg(into)
        .status()
        .context("running tar")?;
    if !status.success() {
        bail!("tar -xzf {} failed ({status})", tarball.display());
    }
    Ok(())
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
            Source::GitHub => download(&release.url(file), dest),
        }
    }
}

/// `url` into the file `dest`. A listing in a file (a mirror's, the tests')
/// may name its assets by path instead.
pub(crate) fn download(url: &str, dest: &Path) -> Result<()> {
    if !url.starts_with("https://") && !url.starts_with("http://") {
        return fs::copy(url, dest)
            .with_context(|| format!("copying {url}"))
            .map(|_| ());
    }
    // GitHub answers an asset with a redirect to its storage, which the
    // agent follows.
    let response = agent(Duration::from_secs(600))
        .get(url)
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let mut body = response.into_body().into_reader();
    let mut out = fs::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    std::io::copy(&mut body, &mut out).with_context(|| format!("reading {url}"))?;
    Ok(())
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
pub(crate) fn releases() -> Result<serde_json::Value> {
    let named = std::env::var(RELEASES_ENV).ok().filter(|v| !v.is_empty());
    read_listing(named.as_deref().unwrap_or(RELEASES))
}

/// The newest superfluid release that serves engine ABI `abi` here, from
/// [`releases`]: `None` when the listing names none.
pub fn newest_for(abi: u32) -> Result<Option<Release>> {
    let target = target()?;
    Ok(newest_in(&releases()?, target, abi))
}

/// A release listing in GitHub's form, from a URL or a file.
pub(crate) fn read_listing(at: &str) -> Result<serde_json::Value> {
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
    newest_where(listing, target, abi, &|_| true)
}

/// [`newest_in`] among the releases `keep` accepts (by their listing entry).
pub fn newest_where(
    listing: &serde_json::Value,
    target: &str,
    abi: u32,
    keep: &dyn Fn(&serde_json::Value) -> bool,
) -> Option<Release> {
    let tarball = format!("superfluid-{target}.tar.gz");
    let checksum = format!("{tarball}.sha256");
    let marker = format!("superfluid-basert-abi-{abi}.txt");
    listing
        .as_array()?
        .iter()
        .filter(|r| !r["draft"].as_bool().unwrap_or(false))
        .filter(|r| !r["prerelease"].as_bool().unwrap_or(false))
        .filter(|r| keep(r))
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
pub(crate) fn newer(have: Option<&str>, than: &str) -> bool {
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

/// The superfluid a `basert serve` start runs, with its hold on it: the
/// first install whose basert runtime superfluid itself reports ready for
/// the engine in `engine`, under `home` as its `SUPERFLUID_HOME`. `say` gets
/// a line for each step taken on the user's behalf.
pub fn ensure(engine: Option<&Path>, home: &Path, say: &mut dyn FnMut(&str)) -> Result<Serving> {
    ensure_with(policy(), &Source::locate(), engine, home, say)
}

pub fn ensure_with(
    policy: Download,
    source: &Source,
    engine: Option<&Path>,
    home: &Path,
    say: &mut dyn FnMut(&str),
) -> Result<Serving> {
    // Every install found is asked; the first that can serve runs. Each is
    // held while asked, and the one that runs until its server stops.
    let mut unready: Vec<(Install, Option<String>, String)> = Vec::new();
    for install in Install::candidates()?
        .into_iter()
        .filter(Install::is_installed)
    {
        install.claim();
        let held = install.hold();
        match install.probe(engine, Some(home)).ready {
            Ok(()) => {
                return Ok(Serving {
                    binary: install.binary(),
                    release: install.installed_release(),
                    held,
                })
            }
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
    let dest = match unready
        .iter()
        .map(|(i, _, _)| i)
        .find(|i| i.takes_installs() && i.writable())
    {
        Some(found) => found.clone(),
        None => Install::for_install()?,
    };
    // A server's workers start from that directory: replaced under it, its
    // next one would be another release's. Two baserts on engines that need
    // different releases share one (a package's basert installs in its data
    // directory), so the other's server may run from it. Looked at before
    // the download, and again at the swap.
    if let Err(e) = dest.unused() {
        bail!(
            "superfluid at {} cannot serve this engine and {tag} can, but {e:#}",
            dest.root.display()
        );
    }
    // Installs into it take turns: one that finished while this waited may
    // serve this engine, and then runs.
    let turn = match (policy, &release) {
        (Download::Never, _) | (_, None) => None,
        _ => {
            let turn = dest.take_turn(say)?;
            if dest.is_installed() {
                let held = dest.hold();
                if dest.probe(engine, Some(home)).ready.is_ok() {
                    return Ok(Serving {
                        binary: dest.binary(),
                        release: dest.installed_release(),
                        held,
                    });
                }
            }
            turn
        }
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
    let staged = dest.stage(source, &release, say)?;
    dest.swap_in(staged, say)?;
    let held = dest.hold();
    drop(turn);
    match dest.probe(engine, Some(home)).ready {
        Ok(()) => Ok(Serving {
            binary: dest.binary(),
            release: dest.installed_release(),
            held,
        }),
        Err(why) => bail!(
            "superfluid {tag}, installed at {}, cannot serve this engine: {why}",
            dest.root.display()
        ),
    }
}

/// A yes/no question on stderr, answered on stdin; an empty answer is yes.
pub(crate) fn confirm(question: &str) -> Result<bool> {
    eprint!("{question}");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "" | "y" | "yes"
    ))
}

pub(crate) fn sha256_of(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn is_executable(path: &Path) -> bool {
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
    fn a_copy_basert_made_before_is_claimed_and_one_the_user_named_is_not() {
        let tmp = tempfile::tempdir().unwrap();
        for (origin, claimed) in [
            (Origin::BesideBasert, true),
            (Origin::BesidePackage, true),
            (Origin::UserData, true),
            (Origin::Env, false),
        ] {
            let root = tmp.path().join(format!("{origin:?}"));
            fs::create_dir_all(root.join("bin")).unwrap();
            fs::write(
                root.join("bin").join("superfluid"),
                fake_superfluid("0.1.3"),
            )
            .unwrap();
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                root.join("bin").join("superfluid"),
                fs::Permissions::from_mode(0o755),
            )
            .unwrap();
            let install = Install {
                root: root.clone(),
                origin,
            };
            install.claim();
            assert_eq!(root.join(INSTALLED_BY).is_file(), claimed, "{origin:?}");
        }
        // A file already there is left as it is.
        let root = tmp.path().join("BesideBasert");
        fs::write(root.join(INSTALLED_BY), "someone: else\n").unwrap();
        Install {
            root: root.clone(),
            origin: Origin::BesideBasert,
        }
        .claim();
        assert_eq!(
            fs::read_to_string(root.join(INSTALLED_BY)).unwrap(),
            "someone: else\n"
        );
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
        // basert's own copy, which superfluid's `self update` leaves to basert.
        assert_eq!(
            fs::read_to_string(install.root.join(INSTALLED_BY)).unwrap(),
            format!("{OWNED_BY}\n")
        );
        assert_eq!(said.len(), 2, "{said:?}");
        assert!(said[0].starts_with(&format!("downloading superfluid {RELEASE} (")));
        // Nothing of the staging is left behind; the locks beside it stay.
        let mut left: Vec<_> = fs::read_dir(tmp.path().join("bin"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            [
                ".superfluid.install.lock",
                ".superfluid.use.lock",
                "superfluid"
            ]
        );
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

    /// What `bin/superfluid` in `root` says it is.
    fn says(root: &Path) -> String {
        fs::read_to_string(root.join("bin").join("superfluid")).unwrap()
    }

    #[test]
    fn a_directory_holding_more_than_superfluid_is_not_replaced() {
        let tmp = tempfile::tempdir().unwrap();
        let first = release(&tmp.path().join("first"), "0.1.3", true);
        let next = release(&tmp.path().join("next"), "0.1.4", true);
        let elsewhere = tmp.path().join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        // What each directory holds beside (or instead of) superfluid: a
        // prefix such as ~/.local, which superfluid's own install script
        // fills and `$BASERT_SUPERFLUID_DIR` may name, and a directory of
        // the user's own named before anything was installed in it.
        type Extra = fn(&Path, &Path);
        let cases: [(&str, bool, Extra, &str); 4] = [
            (
                "share",
                true,
                |root, _| {
                    fs::create_dir_all(root.join("share").join("app")).unwrap();
                    fs::write(root.join("share").join("app").join("data.txt"), "mine").unwrap();
                },
                "share",
            ),
            (
                "tool",
                true,
                |root, _| {
                    fs::write(root.join("bin").join("other-tool"), "mine").unwrap();
                },
                "other-tool",
            ),
            (
                "link",
                true,
                |root, elsewhere| {
                    // superfluid's install script links bin/superfluid to its
                    // own versions.
                    fs::rename(
                        root.join("bin").join("superfluid"),
                        elsewhere.join("superfluid"),
                    )
                    .unwrap();
                    std::os::unix::fs::symlink(
                        elsewhere.join("superfluid"),
                        root.join("bin").join("superfluid"),
                    )
                    .unwrap();
                },
                "superfluid",
            ),
            (
                "mine",
                false,
                |root, _| {
                    fs::create_dir_all(root).unwrap();
                    fs::write(root.join("notes.txt"), "mine").unwrap();
                },
                "notes.txt",
            ),
        ];
        for (name, with_superfluid, extra, named) in cases {
            let install = install_at(tmp.path().join(name));
            if with_superfluid {
                install
                    .install(&Source::Dir(first.clone()), &Release::tested(), &mut |_| {})
                    .unwrap();
            }
            extra(&install.root, &elsewhere);
            let before = says_all(&install.root);
            let err = install
                .install(&Source::Dir(next.clone()), &Release::tested(), &mut |_| {})
                .unwrap_err();
            let err = format!("{err:#}");
            assert!(err.contains("which is not superfluid's"), "{name}: {err}");
            assert!(err.contains(named), "{name}: {err}");
            assert!(err.contains("BASERT_SUPERFLUID_DIR"), "{name}: {err}");
            assert_eq!(says_all(&install.root), before, "{name}");
            // Nothing staged is left beside it (the locks stay).
            let left: Vec<_> = fs::read_dir(tmp.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .filter(|n| n.starts_with('.') && !n.ends_with(".lock"))
                .collect();
            assert!(left.is_empty(), "{name}: {left:?}");
        }
    }

    /// Every file under `root` with what it holds, links as their targets.
    fn says_all(root: &Path) -> Vec<(PathBuf, String)> {
        let mut all = Vec::new();
        let mut dirs = vec![root.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            for entry in fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                let meta = fs::symlink_metadata(&path).unwrap();
                if meta.file_type().is_symlink() {
                    all.push((
                        path.clone(),
                        format!("-> {}", fs::read_link(&path).unwrap().display()),
                    ));
                } else if meta.is_dir() {
                    dirs.push(path);
                } else {
                    all.push((path.clone(), fs::read_to_string(&path).unwrap()));
                }
            }
        }
        all.sort();
        all
    }

    #[test]
    fn a_file_where_superfluid_would_go_is_left_alone_and_said_to_be_one() {
        let tmp = tempfile::tempdir().unwrap();
        let assets = release(tmp.path(), "0.1.4", true);
        // A program of superfluid's name (`cargo install superfluid` puts one
        // beside a basert cargo installed).
        let install = Install {
            root: tmp.path().join("superfluid"),
            origin: Origin::BesideBasert,
        };
        fs::write(&install.root, "#!/bin/sh\n").unwrap();
        let err = install
            .install(&Source::Dir(assets), &Release::tested(), &mut |_| {})
            .unwrap_err();
        let err = format!("{err:#}");
        assert!(
            err.contains(&format!(
                "{} is a file, not superfluid's directory",
                install.root.display()
            )),
            "{err}"
        );
        assert!(
            err.contains(&format!("move {} aside", install.root.display())),
            "{err}"
        );
        assert_eq!(fs::read_to_string(&install.root).unwrap(), "#!/bin/sh\n");
    }

    #[test]
    fn what_the_next_release_also_holds_is_superfluids() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, fresh) = (tmp.path().join("root"), tmp.path().join("fresh"));
        for dir in [&root, &fresh] {
            fs::create_dir_all(dir.join("share").join("superfluid")).unwrap();
            fs::write(dir.join("share").join("superfluid").join("t.json"), "").unwrap();
            fs::create_dir_all(dir.join("bin")).unwrap();
            fs::write(dir.join("bin").join("sf-helper"), "").unwrap();
        }
        assert_eq!(not_superfluids(&root, &fresh, Level::Top), None);
        fs::write(root.join("share").join("other.txt"), "").unwrap();
        assert_eq!(
            not_superfluids(&root, &fresh, Level::Top),
            Some(root.join("share").join("other.txt"))
        );
    }

    #[test]
    fn a_directory_holding_superfluid_alone_is_replaced() {
        let tmp = tempfile::tempdir().unwrap();
        let first = release(&tmp.path().join("first"), "0.1.3", true);
        let next = release(&tmp.path().join("next"), "0.1.4", true);
        let install = install_at(tmp.path().join("superfluid"));
        install
            .install(&Source::Dir(first), &Release::tested(), &mut |_| {})
            .unwrap();
        // All a release holds, the release's own files included, and what
        // basert writes beside them.
        for file in ["LICENSE", "NOTICE", "README.md", "compat.json"] {
            fs::write(install.root.join(file), "release").unwrap();
        }
        for tool in ["superfluid-noded", "superfluid-workerd"] {
            fs::write(install.root.join("bin").join(tool), "release").unwrap();
        }
        assert!(install.root.join(STAMP).is_file());
        assert!(install.root.join(INSTALLED_BY).is_file());
        install
            .install(&Source::Dir(next), &Release::tested(), &mut |_| {})
            .unwrap();
        assert!(says(&install.root).contains("superfluid 0.1.4"));
        assert!(!install.root.join("LICENSE").exists());
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
        // The crate from crates.io carries its own copy.
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let header = [
            "../../../../include/baseRT/baseRT.h",
            "../../../include/baseRT/baseRT.h",
            "include/baseRT/baseRT.h",
        ]
        .iter()
        .map(|rel| crate_dir.join(rel))
        .find(|p| p.is_file())
        .expect("include/baseRT/baseRT.h at the repository's root, or in the crate");
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
