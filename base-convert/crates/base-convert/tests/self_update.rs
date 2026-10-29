//! `basert self`, end to end: the launcher, placed in a directory the way the
//! install script placed it before the versions layout, updates itself from a
//! release listing in a file, goes back and forward, and brings a superfluid
//! beside it to one that serves the engine. The releases are fake (a launcher
//! that is a shell script, a library of a few bytes); the superfluid is a
//! script that serves one engine ABI number and reads the engine's from its
//! header.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// The version of the launcher under test, and the one a fake release is.
const HERE: &str = env!("CARGO_PKG_VERSION");

fn next_version() -> String {
    let mut parts: Vec<u64> = HERE.split('.').map(|p| p.parse().unwrap()).collect();
    *parts.last_mut().unwrap() += 1;
    parts
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

fn platform() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "macos-arm64",
        ("linux", "aarch64") => "linux-arm64-cuda",
        other => panic!("no basert release for {other:?}"),
    }
}

fn superfluid_target() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        other => panic!("no superfluid release for {other:?}"),
    }
}

fn executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn sha256(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

/// The engine's files: a library with its link, a tool, the header stating `abi`.
fn engine_files(dir: &Path, version: &str, abi: u32) {
    fs::create_dir_all(dir.join("include").join("baseRT")).unwrap();
    fs::write(dir.join(format!("libbaseRT.{version}.dylib")), version).unwrap();
    symlink(
        format!("libbaseRT.{version}.dylib"),
        dir.join("libbaseRT.dylib"),
    )
    .unwrap();
    executable(
        &dir.join("basert-chat"),
        &format!("#!/bin/sh\necho chat {version}\n"),
    );
    fs::write(
        dir.join("include").join("baseRT").join("baseRT.h"),
        format!("#pragma once\n#define BASERT_ABI_VERSION {abi}\n"),
    )
    .unwrap();
    fs::write(dir.join("LICENSE"), "engine").unwrap();
}

/// A superfluid at `root` (its layout: bin/, libexec/superfluid/) that is
/// `version` and serves engine ABI `serves`: `runtimes --json` reports the
/// basert runtime ready when the header beside `$BASERT_LIB` states it.
fn superfluid_at(root: &Path, version: &str, serves: u32) {
    fs::create_dir_all(root.join("bin")).unwrap();
    fs::create_dir_all(root.join("libexec").join("superfluid")).unwrap();
    executable(
        &root.join("bin").join("superfluid"),
        &format!(
            "#!/bin/sh\n\
             if [ \"$1\" = --version ]; then echo \"superfluid {version}\"; exit 0; fi\n\
             abi=$(sed -n 's/^#define BASERT_ABI_VERSION //p' \"$BASERT_LIB/include/baseRT/baseRT.h\" 2>/dev/null)\n\
             if [ \"$abi\" = \"{serves}\" ]; then\n\
               echo '[{{\"id\":\"basert\",\"state\":\"ready\",\"ready\":true,\"abi\":{serves}}}]'\n\
             else\n\
               echo \"[{{\\\"id\\\":\\\"basert\\\",\\\"state\\\":\\\"not ready\\\",\\\"ready\\\":\\\"drives engine ABI {serves}, the engine states $abi\\\"}}]\"\n\
             fi\n"
        ),
    );
    executable(
        &root
            .join("libexec")
            .join("superfluid")
            .join("superfluid-worker-basert"),
        "#!/bin/sh\n",
    );
}

/// A listing's assets, each named by its path.
fn listed_assets(paths: &[&Path]) -> serde_json::Value {
    paths
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.file_name().unwrap().to_str().unwrap(),
                "browser_download_url": p.to_str().unwrap(),
            })
        })
        .collect()
}

struct Fixture {
    tmp: TempDir,
    root: PathBuf,
    releases: Vec<serde_json::Value>,
    superfluid_releases: Vec<serde_json::Value>,
}

impl Fixture {
    /// The launcher under test placed in `<tmp>/.basert` as the install
    /// script did before versions (flat, with a `.release` stamp), and a
    /// superfluid beside it that is v0.1.3 and serves ABI 1.
    fn in_place() -> Fixture {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join(".basert");
        fs::create_dir_all(&root).unwrap();
        let launcher = root.join("basert");
        let built = Path::new(env!("CARGO_BIN_EXE_basert"));
        // Copied, as package_copy explains.
        fs::copy(built, &launcher).unwrap();
        engine_files(&root, HERE, 1);
        fs::write(root.join(".release"), format!("v{HERE}\n")).unwrap();
        superfluid_at(&root.join("superfluid"), "0.1.3", 1);
        fs::write(root.join("superfluid").join(".release"), "v0.1.3\n").unwrap();
        fs::write(root.join("notes.txt"), "mine").unwrap();
        let mut f = Fixture {
            tmp,
            root,
            releases: Vec::new(),
            superfluid_releases: Vec::new(),
        };
        f.superfluid_release("v0.1.3", 1);
        f
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.tmp.path().join(rel)
    }

    /// A basert release `version` for this platform whose engine states `abi`.
    fn release(&mut self, version: &str, abi: u32) {
        let tree = self.path(&format!("build-{version}"));
        fs::create_dir_all(&tree).unwrap();
        engine_files(&tree, version, abi);
        executable(
            &tree.join("basert"),
            &format!("#!/bin/sh\n[ \"$1\" = --version ] && echo \"basert {version}\" && exit 0\necho \"basert {version}: $*\"\n"),
        );
        let assets = self.path(&format!("assets-{version}"));
        fs::create_dir_all(&assets).unwrap();
        let name = format!("basert-engine-{}-{version}.tar.gz", platform());
        let tarball = assets.join(&name);
        assert!(Command::new("tar")
            .arg("-czf")
            .arg(&tarball)
            .arg("-C")
            .arg(&tree)
            .arg(".")
            .status()
            .unwrap()
            .success());
        let checksum = assets.join(format!("{name}.sha256"));
        fs::write(&checksum, format!("{}  {name}\n", sha256(&tarball))).unwrap();
        let marker = assets.join(format!("basert-engine-abi-{abi}.txt"));
        fs::write(&marker, format!("{abi}\n")).unwrap();
        self.releases.push(serde_json::json!({
            "tag_name": format!("v{version}"),
            "published_at": "2026-01-01T00:00:00Z",
            "draft": false,
            "prerelease": false,
            "assets": listed_assets(&[&tarball, &checksum, &marker]),
        }));
    }

    /// A superfluid release `tag` that serves engine ABI `serves`.
    fn superfluid_release(&mut self, tag: &str, serves: u32) {
        let name = format!("superfluid-{}", superfluid_target());
        let tree = self.path(&format!("sf-build-{tag}"));
        superfluid_at(&tree.join(&name), tag.trim_start_matches('v'), serves);
        let assets = self.path(&format!("sf-assets-{tag}"));
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
        let checksum = assets.join(format!("{name}.tar.gz.sha256"));
        fs::write(&checksum, format!("{}  {name}.tar.gz\n", sha256(&tarball))).unwrap();
        let marker = assets.join(format!("superfluid-basert-abi-{serves}.txt"));
        fs::write(&marker, format!("{serves}\n")).unwrap();
        self.superfluid_releases.push(serde_json::json!({
            "tag_name": tag,
            "published_at": "2026-01-01T00:00:00Z",
            "draft": false,
            "prerelease": false,
            "assets": listed_assets(&[&tarball, &checksum, &marker]),
        }));
    }

    /// `launcher self <args>` against this fixture's listings, home and state.
    fn run(&self, launcher: &Path, args: &[&str]) -> (i32, String, String) {
        let mut words = vec!["self"];
        words.extend_from_slice(args);
        self.launch(launcher, &words)
    }

    /// `launcher <args>` against this fixture's listings, home and state.
    fn launch(&self, launcher: &Path, args: &[&str]) -> (i32, String, String) {
        self.launch_with(launcher, args, &[])
    }

    /// [`Fixture::launch`] with `env` set as well.
    fn launch_with(
        &self,
        launcher: &Path,
        args: &[&str],
        env: &[(&str, &Path)],
    ) -> (i32, String, String) {
        let out = self.command(launcher, args, env).output().unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// `launcher <args>` with this fixture's listings, home and state, and
    /// `env`, to run or to start.
    fn command(&self, launcher: &Path, args: &[&str], env: &[(&str, &Path)]) -> Command {
        let listing = self.path("releases.json");
        fs::write(&listing, serde_json::to_string(&self.releases).unwrap()).unwrap();
        let sf_listing = self.path("superfluid-releases.json");
        fs::write(
            &sf_listing,
            serde_json::to_string(&self.superfluid_releases).unwrap(),
        )
        .unwrap();
        let mut command = Command::new(launcher);
        command
            .args(args)
            .env("HOME", self.path("home"))
            .env("XDG_DATA_HOME", self.path("home/.local/share"))
            .env("BASERT_STATE_DIR", self.path("state"))
            .env("BASERT_RELEASES", &listing)
            .env("BASERT_SUPERFLUID_RELEASES", &sf_listing)
            .env("BASERT_NO_UPDATE_CHECK", "1")
            .env_remove("BASERT_LIB")
            .env_remove("BASERT_SUPERFLUID_DIR")
            .env_remove("BASERT_SUPERFLUID_SOURCE")
            .env_remove("BASERT_SUPERFLUID_AUTO_INSTALL")
            .env_remove("BASERT_INSTALL_DIR")
            .envs(env.iter().copied());
        command
    }

    fn version_dir(&self, v: &str) -> PathBuf {
        self.root.join("versions").join(v)
    }

    fn current(&self) -> Option<String> {
        fs::read_link(self.root.join("versions").join("current"))
            .ok()
            .map(|t| t.to_string_lossy().into_owned())
    }

    fn says(&self, program: &str) -> String {
        let out = Command::new(self.root.join(program))
            .arg("--version")
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn superfluid_release_installed(&self) -> String {
        fs::read_to_string(self.root.join("superfluid").join(".release"))
            .unwrap()
            .trim()
            .to_string()
    }

    /// The superfluid in basert's data directory under this fixture's home.
    fn data_superfluid(&self) -> PathBuf {
        let data = match std::env::consts::OS {
            "macos" => self.path("home/Library/Application Support"),
            _ => self.path("home/.local/share"),
        };
        data.join("baseRT").join("superfluid")
    }
}

#[test]
fn an_in_place_install_updates_into_the_versions_layout_and_goes_back_and_forward() {
    let mut f = Fixture::in_place();
    let next = next_version();
    f.release(HERE, 1);
    f.release(&next, 1);

    let (code, out, err) = f.run(&f.root.join("basert"), &["update", "--dry-run"]);
    assert_eq!(code, 0, "{err}");
    assert!(err.contains(&format!("basert {HERE} -> {next}")), "{err}");
    assert!(err.contains("dry run: nothing fetched or changed"), "{err}");
    assert!(f.current().is_none() && !f.version_dir(&next).exists());

    let (code, out2, err) = f.run(&f.root.join("basert"), &["update"]);
    assert_eq!(code, 0, "{out}{out2}{err}");
    assert!(out2.contains(&format!("basert {next} is in use")), "{out2}");
    assert!(
        err.contains(&format!(
            "{HERE} stays installed: `basert self use {HERE}` goes back to it"
        )),
        "{err}"
    );
    assert_eq!(f.current().as_deref(), Some(next.as_str()));
    // The in-place copy is the version to go back to, the launcher under test among it.
    assert!(fs::symlink_metadata(f.version_dir(HERE).join("basert"))
        .unwrap()
        .is_file());
    assert_eq!(
        fs::read_link(f.version_dir(HERE).join("libbaseRT.dylib")).unwrap(),
        Path::new(&format!("libbaseRT.{HERE}.dylib"))
    );
    // The root is links through `current`: what PATH runs is the new release.
    assert_eq!(
        fs::read_link(f.root.join("basert")).unwrap(),
        Path::new("versions/current/basert")
    );
    assert_eq!(f.says("basert"), format!("basert {next}"));
    assert_eq!(
        fs::read_link(f.root.join("libbaseRT.dylib")).unwrap(),
        Path::new("versions/current/libbaseRT.dylib")
    );
    assert!(fs::symlink_metadata(f.root.join(format!("libbaseRT.{HERE}.dylib"))).is_err());
    assert_eq!(
        fs::read_to_string(f.root.join(".release")).unwrap().trim(),
        format!("v{next}")
    );
    // Not the release's: left alone. superfluid already served the engine.
    assert_eq!(
        fs::read_to_string(f.root.join("notes.txt")).unwrap(),
        "mine"
    );
    assert_eq!(f.superfluid_release_installed(), "v0.1.3");

    // The launcher under test, now a version of the layout.
    let old = f.version_dir(HERE).join("basert");
    let (code, out, err) = f.run(&old, &["list"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains(&format!("* {next} (in use)")), "{out}");
    assert!(
        out.contains(&format!(
            "  {HERE} (the one before: `basert self use {HERE}` goes back to it)"
        )),
        "{out}"
    );
    assert!(out.contains("superfluid v0.1.3 at "), "{out}");

    let (code, out, err) = f.run(&old, &["use", HERE]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains(&format!("basert {HERE} is in use")), "{out}");
    assert_eq!(f.current().as_deref(), Some(HERE));
    assert_eq!(f.says("basert"), format!("basert {HERE}"));

    // Forward again from the root: the release on disk is used as it is.
    let (code, out, err) = f.run(&f.root.join("basert"), &["update"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(err.contains(", already here)"), "{err}");
    assert!(!err.contains("downloading"), "{err}");
    assert_eq!(f.current().as_deref(), Some(next.as_str()));

    let (code, _, err) = f.run(&old, &["update"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        err.contains(&format!("basert {next} is in use, the newest release")),
        "{err}"
    );
}

#[test]
fn a_superfluid_that_cannot_serve_the_new_engine_is_replaced_by_the_newest_that_can() {
    let mut f = Fixture::in_place();
    let next = next_version();
    f.release(&next, 2);
    f.superfluid_release("v0.1.4", 2);
    f.superfluid_release("v0.1.5", 2);

    let (code, out, err) = f.run(&f.root.join("basert"), &["update"]);
    assert_eq!(code, 0, "{out}{err}");
    assert_eq!(f.current().as_deref(), Some(next.as_str()));
    assert_eq!(f.superfluid_release_installed(), "v0.1.5");
    assert!(err.contains("installed superfluid v0.1.5"), "{err}");

    // Back to the ABI 1 engine: the newest superfluid for it is older, and is what it takes.
    let (code, out, err) = f.run(&f.version_dir(HERE).join("basert"), &["use", HERE]);
    assert_eq!(code, 0, "{out}{err}");
    assert_eq!(f.current().as_deref(), Some(HERE));
    assert_eq!(f.superfluid_release_installed(), "v0.1.3");
}

#[test]
fn with_no_superfluid_for_the_new_engine_nothing_changes() {
    let mut f = Fixture::in_place();
    let next = next_version();
    f.release(&next, 2);

    let (code, _, err) = f.run(&f.root.join("basert"), &["update"]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains("no superfluid release states its ABI number 2 yet"),
        "{err}"
    );
    assert!(
        err.contains(&format!(
            "basert {next} was not made the one in use; {HERE} stays"
        )),
        "{err}"
    );
    assert!(fs::symlink_metadata(f.root.join("basert"))
        .unwrap()
        .is_file());
    assert!(f.current().is_none());
    assert!(!f.version_dir(&next).exists());
    assert_eq!(f.superfluid_release_installed(), "v0.1.3");
}

/// `<home>/.local` as superfluid's own install script fills it, which
/// `BASERT_SUPERFLUID_DIR` may name: superfluid `version` serving `serves` in
/// `bin/` and `libexec/superfluid/`, beside the user's other tools and files.
fn shared_prefix(f: &Fixture, version: &str, serves: u32) -> PathBuf {
    let prefix = f.path("home/.local");
    superfluid_at(&prefix, version, serves);
    executable(&prefix.join("bin").join("other-tool"), "#!/bin/sh\n");
    fs::create_dir_all(prefix.join("share").join("app")).unwrap();
    fs::write(prefix.join("share").join("app").join("data.txt"), "mine").unwrap();
    prefix
}

fn prefix_untouched(prefix: &Path, version: &str) {
    assert!(prefix.join("bin").join("other-tool").is_file());
    assert_eq!(
        fs::read_to_string(prefix.join("share").join("app").join("data.txt")).unwrap(),
        "mine"
    );
    let out = Command::new(prefix.join("bin").join("superfluid"))
        .arg("--version")
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        format!("superfluid {version}")
    );
}

#[test]
fn a_superfluid_dir_that_holds_other_files_keeps_its_superfluid_when_a_newer_one_is_out() {
    let mut f = Fixture::in_place();
    f.release(HERE, 1);
    f.superfluid_release("v0.1.4", 1);
    let prefix = shared_prefix(&f, "0.1.3", 1);
    let (code, out, err) = f.launch_with(
        &f.root.join("basert"),
        &["self", "update"],
        &[("BASERT_SUPERFLUID_DIR", prefix.as_path())],
    );
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains("superfluid v0.1.4 is out, but v0.1.3 stays"),
        "{err}"
    );
    assert!(err.contains("which is not superfluid's"), "{err}");
    prefix_untouched(&prefix, "0.1.3");
}

#[test]
fn a_superfluid_dir_that_holds_other_files_is_not_replaced_for_a_new_engine() {
    let mut f = Fixture::in_place();
    let next = next_version();
    f.release(&next, 2);
    f.superfluid_release("v0.1.4", 2);
    let prefix = shared_prefix(&f, "0.1.3", 1);
    let (code, out, err) = f.launch_with(
        &f.root.join("basert"),
        &["self", "update"],
        &[("BASERT_SUPERFLUID_DIR", prefix.as_path())],
    );
    assert_eq!(code, 1, "{out}{err}");
    assert!(err.contains("which is not superfluid's"), "{err}");
    assert!(
        err.contains(&format!(
            "basert {next} was not made the one in use; {HERE} stays"
        )),
        "{err}"
    );
    prefix_untouched(&prefix, "0.1.3");
    assert!(fs::symlink_metadata(f.root.join("basert"))
        .unwrap()
        .is_file());
    assert!(!f.version_dir(&next).exists());
}

#[test]
fn a_newer_superfluid_comes_as_well_when_basert_is_up_to_date() {
    let mut f = Fixture::in_place();
    f.release(HERE, 1);
    f.superfluid_release("v0.1.4", 1);
    let (code, out, err) = f.run(&f.root.join("basert"), &["update"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains(&format!("basert {HERE} is in use, the newest release")),
        "{err}"
    );
    assert_eq!(f.superfluid_release_installed(), "v0.1.4");
    // Still in place: nothing of basert changed.
    assert!(fs::symlink_metadata(f.root.join("basert"))
        .unwrap()
        .is_file());
}

#[test]
fn the_install_script_places_a_bundle_over_an_in_place_copy() {
    let f = Fixture::in_place();
    let next = next_version();
    let bundle = f.path("stage");
    fs::create_dir_all(&bundle).unwrap();
    engine_files(&bundle, &next, 1);
    executable(
        &bundle.join("basert"),
        &format!("#!/bin/sh\n[ \"$1\" = --version ] && echo \"basert {next}\"\n"),
    );
    let launcher = Path::new(env!("CARGO_BIN_EXE_basert"));
    let root = f.root.to_str().unwrap().to_string();
    let (code, out, err) = f.run(
        launcher,
        &[
            "install",
            "--from",
            bundle.to_str().unwrap(),
            "--root",
            &root,
        ],
    );
    assert_eq!(code, 0, "{out}{err}");
    assert!(!bundle.exists());
    assert_eq!(f.current().as_deref(), Some(next.as_str()));
    assert_eq!(f.says("basert"), format!("basert {next}"));
    assert!(fs::symlink_metadata(f.version_dir(HERE).join("basert"))
        .unwrap()
        .is_file());
    assert_eq!(
        fs::read_to_string(f.root.join(".release")).unwrap().trim(),
        format!("v{next}")
    );
    assert_eq!(
        fs::read_to_string(f.root.join("versions").join("previous"))
            .unwrap()
            .trim(),
        HERE
    );
}

#[test]
fn serve_from_the_layout_runs_the_superfluid_in_the_root_on_its_own_versions_engine() {
    let mut f = Fixture::in_place();
    f.release(&next_version(), 1);
    assert_eq!(f.run(&f.root.join("basert"), &["update"]).0, 0);
    // Back to the launcher under test, which PATH then finds through the root's links.
    let (code, _, err) = f.run(&f.version_dir(HERE).join("basert"), &["use", HERE]);
    assert_eq!(code, 0, "{err}");
    let model = f.path("m.base");
    fs::write(&model, "").unwrap();
    let (code, out, err) = f.launch(
        &f.root.join("basert"),
        &["serve", model.to_str().unwrap(), "--print-command"],
    );
    assert_eq!(code, 0, "{out}{err}");
    let root = fs::canonicalize(&f.root).unwrap();
    let superfluid = root.join("superfluid").join("bin").join("superfluid");
    assert!(
        out.contains(&format!(" {} serve ", superfluid.display())),
        "{out}"
    );
    // The version's own directory by its real path: an update turns the
    // root's links, never this.
    let engine = root.join("versions").join(HERE);
    assert!(
        out.trim_end()
            .ends_with(&format!("--basert-lib {}", engine.display())),
        "{out}"
    );
}

/// The launcher under test placed at `rel` under the fixture, as a package
/// manager places it (the release's files beside it, an `installed-by` file
/// when `named`), and a manager script at `manager` (relative too) that logs
/// its arguments and the first PATH entry to `<tmp>/manager.log`.
fn package_copy(f: &Fixture, rel: &str, named: Option<&str>, manager: &str) -> PathBuf {
    let dir = f.path(rel);
    fs::create_dir_all(&dir).unwrap();
    let launcher = dir.join("basert");
    let built = Path::new(env!("CARGO_BIN_EXE_basert"));
    // A copy, not a hard link: macOS may give a hard-linked program's path as
    // any of its links, so one test's launcher would seem to run from another's.
    fs::copy(built, &launcher).unwrap();
    engine_files(&dir, HERE, 1);
    if let Some(line) = named {
        fs::write(dir.join("installed-by"), format!("{line}\n")).unwrap();
    }
    let script = f.path(manager);
    fs::create_dir_all(script.parent().unwrap()).unwrap();
    executable(
        &script,
        &format!(
            "#!/bin/sh\necho \"$*|${{PATH%%:*}}\" >> {}\n",
            f.path("manager.log").display()
        ),
    );
    launcher
}

fn manager_log(f: &Fixture) -> String {
    fs::read_to_string(f.path("manager.log")).unwrap_or_default()
}

#[test]
fn a_homebrew_copy_is_upgraded_by_its_own_brew_when_asked() {
    let f = Fixture::in_place();
    let exe = package_copy(
        &f,
        "brew/Cellar/basert/0.4.3/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    // No terminal and no --yes: named, not run.
    let (code, _, err) = f.run(&exe, &["update"]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains("installed by Homebrew: run `brew upgrade basert`, or `basert self update --yes` to let basert run it"),
        "{err}"
    );
    assert_eq!(manager_log(&f), "");

    let (code, _, err) = f.run(&exe, &["update", "--dry-run"]);
    assert_eq!(code, 0, "{err}");
    assert!(err.contains("`self update` would run "), "{err}");
    assert!(err.contains("brew/bin/brew upgrade basert"), "{err}");

    let (code, _, err) = f.run(&exe, &["update", "--yes"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        manager_log(&f).starts_with("upgrade basert|"),
        "{}",
        manager_log(&f)
    );

    // A version is the manager's to choose.
    let (code, _, err) = f.run(&exe, &["update", "--yes", "--version", "0.4.2"]);
    assert_eq!(code, 1);
    assert!(err.contains("which chooses its version"), "{err}");

    // A server recorded on this engine: nothing is run.
    let records = f.path("state").join("serving");
    fs::create_dir_all(&records).unwrap();
    let engine = fs::canonicalize(exe.parent().unwrap()).unwrap();
    fs::write(
        records.join(std::process::id().to_string()),
        format!("{}\n", engine.display()),
    )
    .unwrap();
    let (code, _, err) = f.run(&exe, &["update", "--yes"]);
    assert_eq!(code, 1);
    assert!(
        err.contains(&format!(
            "a server runs on the engine in {} (pid {})",
            engine.display(),
            std::process::id()
        )),
        "{err}"
    );
    fs::remove_file(records.join(std::process::id().to_string())).unwrap();

    // Pinned: named, with the way out.
    fs::create_dir_all(f.path("brew/var/homebrew/pinned")).unwrap();
    fs::write(f.path("brew/var/homebrew/pinned/basert"), "").unwrap();
    let (code, _, err) = f.run(&exe, &["update", "--yes"]);
    assert_eq!(code, 1);
    assert!(
        err.contains("it is pinned in Homebrew: `brew unpin basert` first"),
        "{err}"
    );
    assert_eq!(manager_log(&f).lines().count(), 1);
}

#[test]
fn a_global_npm_copy_is_updated_by_that_npm_with_its_bin_first_on_path() {
    let f = Fixture::in_place();
    let exe = package_copy(
        &f,
        "node/lib/node_modules/@basecompute/basert-darwin-arm64/engine",
        Some("npm: npm install --global @basecompute/basert@latest"),
        "node/bin/npm",
    );
    let (code, _, err) = f.run(&exe, &["update", "--yes"]);
    assert_eq!(code, 0, "{err}");
    let bin = fs::canonicalize(f.path("node/bin")).unwrap();
    assert_eq!(
        manager_log(&f).trim(),
        format!(
            "install --global @basecompute/basert@latest|{}",
            bin.display()
        )
    );
}

#[test]
fn a_projects_npm_copy_and_a_pip_copy_are_named_not_run() {
    let f = Fixture::in_place();
    let exe = package_copy(
        &f,
        "app/node_modules/@basecompute/basert-darwin-arm64/engine",
        Some("npm: npm install --global @basecompute/basert@latest"),
        "app/node_modules/.bin/npm",
    );
    let (code, _, err) = f.run(&exe, &["update", "--yes"]);
    assert_eq!(code, 1);
    assert!(err.contains("it is a project's dependency"), "{err}");
    let exe = package_copy(
        &f,
        "venv/lib/python3.12/site-packages/baseRT/_engine",
        Some("pip: pip install --upgrade basert"),
        "venv/bin/python",
    );
    let (code, _, err) = f.run(&exe, &["update", "--yes"]);
    assert_eq!(code, 1);
    let python = fs::canonicalize(f.path("venv/bin/python")).unwrap();
    assert!(
        err.contains(&format!(
            "run `{} -m pip install --upgrade basert`",
            python.display()
        )),
        "{err}"
    );
    assert_eq!(manager_log(&f), "");
}

#[test]
fn superfluid_is_not_replaced_under_a_server_running_from_it() {
    let f = Fixture::in_place();
    // A process run from basert's superfluid stands in for its server: a copy
    // of the launcher, blocked reading a FIFO named as its release listing. (A
    // copy of /bin/sleep is killed at its start on macOS.)
    let program = f.root.join("superfluid/libexec/superfluid/sleeper");
    fs::copy(env!("CARGO_BIN_EXE_basert"), &program).unwrap();
    let fifo = f.path("never");
    assert!(Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let mut server = Command::new(&program)
        .args(["self", "check", "--record"])
        .env("BASERT_RELEASES", &fifo)
        .env("BASERT_STATE_DIR", f.path("state"))
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        server.try_wait().unwrap().is_none(),
        "the stand-in server is up"
    );
    let (code, _, err) = f.launch(
        &f.root.join("basert"),
        &["superfluid", "install", "--force"],
    );
    let _ = server.kill();
    let _ = server.wait();
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains(&format!(
            "(pid {}): stop it, then run this again",
            server.id()
        )),
        "{err}"
    );
    assert_eq!(f.superfluid_release_installed(), "v0.1.3");
}

#[test]
fn a_package_copy_installs_superfluid_in_the_data_directory_not_its_folder() {
    let f = Fixture::in_place();
    let data = f.data_superfluid();
    for (rel, named, manager) in [
        (
            "brew/Cellar/basert/0.4.3/libexec",
            Some("Homebrew: brew upgrade basert"),
            "brew/bin/brew",
        ),
        (
            "node/lib/node_modules/@basecompute/basert-darwin-arm64/engine",
            Some("npm: npm install --global @basecompute/basert@latest"),
            "node/bin/npm",
        ),
        (
            "venv/lib/python3.12/site-packages/baseRT/_engine",
            Some("pip: pip install --upgrade basert"),
            "venv/bin/python",
        ),
        ("cargo/.cargo/bin", None, "cargo/.cargo/bin/cargo"),
    ] {
        let _ = fs::remove_dir_all(&data);
        let exe = package_copy(&f, rel, named, manager);
        let (code, out, err) = f.launch(&exe, &["superfluid", "install"]);
        assert_eq!(code, 0, "{rel}: {out}{err}");
        assert!(
            out.contains(&format!(
                "superfluid v0.1.3 is installed at {}",
                data.display()
            )),
            "{rel}: {out}"
        );
        assert!(data.join("bin").join("superfluid").is_file(), "{rel}");
        assert_eq!(
            fs::read_to_string(data.join(".release")).unwrap(),
            "v0.1.3\n"
        );
        assert!(
            !exe.parent().unwrap().join("superfluid").exists(),
            "{rel}: nothing in the package's folder"
        );
        // Found there the next time: nothing is downloaded again.
        let (code, out, err) = f.launch(&exe, &["superfluid", "install"]);
        assert_eq!(code, 0, "{rel}: {out}{err}");
        assert!(!err.contains("downloading"), "{rel}: {err}");
        let (code, out, err) = f.launch(&exe, &["superfluid", "status"]);
        assert_eq!(code, 0, "{rel}: {out}{err}");
        assert!(
            out.contains(&format!(
                "installed: v0.1.3 ({}, basert's data directory)",
                data.display()
            )),
            "{rel}: {out}"
        );
    }
}

#[test]
fn a_superfluid_beside_a_package_copy_still_runs_and_its_successor_goes_to_the_data_directory() {
    let f = Fixture::in_place();
    let exe = package_copy(
        &f,
        "brew/Cellar/basert/0.4.2/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    // Where a basert of 0.4.2 put it.
    let beside = exe.parent().unwrap().join("superfluid");
    superfluid_at(&beside, "0.1.3", 1);
    fs::write(beside.join(".release"), "v0.1.3\n").unwrap();
    let data = f.data_superfluid();
    let model = f.path("m.base");
    fs::write(&model, "").unwrap();
    let model = model.to_str().unwrap();
    // The path as the launcher gives it: macOS keeps the one it was started
    // by, Linux the real one.
    let in_keg = "/brew/Cellar/basert/0.4.2/libexec/superfluid";

    // Found and run: nothing downloaded, nothing in the data directory.
    let (code, out, err) = f.launch(&exe, &["superfluid", "install"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("superfluid v0.1.3 is installed at "), "{out}");
    assert!(out.trim_end().ends_with(in_keg), "{out}");
    let (code, out, err) = f.launch(&exe, &["serve", model, "--print-command"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        out.contains(&format!("{in_keg}/bin/superfluid serve ")),
        "{out}"
    );
    let (code, out, err) = f.launch(&exe, &["serve", model]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("installing superfluid"), "{err}");
    assert!(!data.exists(), "{}", data.display());

    // One that cannot serve the engine is left where it is; the release that
    // can goes to the data directory, which is run from then on.
    fs::remove_dir_all(&beside).unwrap();
    superfluid_at(&beside, "0.1.2", 2);
    fs::write(beside.join(".release"), "v0.1.2\n").unwrap();
    let (code, out, err) = f.launch(&exe, &["serve", model]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains(&format!(
            "basert serve: installing superfluid v0.1.3 into {} (",
            data.display()
        )),
        "{err}"
    );
    assert_eq!(
        fs::read_to_string(data.join(".release")).unwrap(),
        "v0.1.3\n"
    );
    assert_eq!(
        fs::read_to_string(beside.join(".release")).unwrap(),
        "v0.1.2\n"
    );
    let (code, out, err) = f.launch(&exe, &["serve", model, "--print-command"]);
    assert_eq!(code, 0, "{out}{err}");
    // Quoted when it holds a space, as macOS's Application Support does.
    let binary = data.join("bin").join("superfluid").display().to_string();
    assert!(
        out.contains(&format!(" {binary} serve ")) || out.contains(&format!(" '{binary}' serve ")),
        "{out}"
    );
    let (code, out, err) = f.launch(&exe, &["serve", model]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("installing superfluid"), "{err}");
}

#[test]
fn a_start_does_not_replace_the_superfluid_another_server_runs_from() {
    let f = Fixture::in_place();
    // Two package copies share basert's data directory: this one's engine
    // states ABI 1, and the superfluid there serves ABI 2, for another
    // copy's engine, whose server runs from it.
    let exe = package_copy(
        &f,
        "brew/Cellar/basert/0.4.3/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    let data = f.data_superfluid();
    superfluid_at(&data, "0.1.4", 2);
    fs::write(data.join(".release"), "v0.1.4\n").unwrap();
    // A stand-in for that server, as in
    // superfluid_is_not_replaced_under_a_server_running_from_it.
    let program = data.join("libexec/superfluid/sleeper");
    fs::copy(env!("CARGO_BIN_EXE_basert"), &program).unwrap();
    let fifo = f.path("never");
    assert!(Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let mut server = Command::new(&program)
        .args(["self", "check", "--record"])
        .env("BASERT_RELEASES", &fifo)
        .env("BASERT_STATE_DIR", f.path("state"))
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        server.try_wait().unwrap().is_none(),
        "the stand-in server is up"
    );
    let model = f.path("m.base");
    fs::write(&model, "").unwrap();
    let (code, out, err) = f.launch(&exe, &["serve", model.to_str().unwrap()]);
    let _ = server.kill();
    let _ = server.wait();
    assert_eq!(code, 1, "{out}{err}");
    assert!(
        err.contains(&format!(
            "a server runs from {} (pid {}): stop it, then run this again",
            data.display(),
            server.id()
        )),
        "{err}"
    );
    assert_eq!(
        fs::read_to_string(data.join(".release")).unwrap(),
        "v0.1.4\n"
    );
}

#[test]
fn a_cargo_copy_leaves_a_superfluid_cargo_installed_beside_it_alone() {
    let f = Fixture::in_place();
    let exe = package_copy(&f, "cargo/.cargo/bin", None, "cargo/.cargo/bin/cargo");
    // `cargo install superfluid` puts its binary where basert would once have
    // made a superfluid/ directory.
    let theirs = exe.parent().unwrap().join("superfluid");
    executable(&theirs, "#!/bin/sh\necho superfluid 0.1.5\n");
    let (code, out, err) = f.launch(&exe, &["superfluid", "install"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        out.contains(&format!(
            "superfluid v0.1.3 is installed at {}",
            f.data_superfluid().display()
        )),
        "{out}"
    );
    assert_eq!(
        fs::read_to_string(&theirs).unwrap(),
        "#!/bin/sh\necho superfluid 0.1.5\n"
    );
}

/// A loopback HTTP server that answers every request with `body` once `go`
/// is sent: downloads held in flight.
fn held_download(body: Vec<u8>) -> (String, mpsc::Sender<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://{}/superfluid.tar.gz",
        listener.local_addr().unwrap()
    );
    let (go, wait) = mpsc::channel::<()>();
    let open = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let opened = open.clone();
    std::thread::spawn(move || {
        let _ = wait.recv();
        opened.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let (body, open) = (body.clone(), open.clone());
            std::thread::spawn(move || {
                let mut request = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut request);
                while !open.load(std::sync::atomic::Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(10));
                }
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(&body);
            });
        }
    });
    (url, go)
}

/// The lines `child` writes to stderr, as they come.
fn stderr_lines(child: &mut Child) -> mpsc::Receiver<String> {
    let err = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(err).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

/// The first line holding one of `texts`, unless the stream ends (or 30 s
/// pass) first; every line read is kept in `seen`.
fn wait_for(
    lines: &mpsc::Receiver<String>,
    texts: &[&str],
    seen: &mut Vec<String>,
) -> Option<String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while let Ok(line) = lines.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        seen.push(line.clone());
        if texts.iter().any(|t| line.contains(t)) {
            return Some(line);
        }
    }
    None
}

/// `command` started with no stdin (a start then downloads without asking)
/// and no proxy, its output piped.
fn start(mut command: Command) -> Child {
    command
        .env_remove("HTTP_PROXY")
        .env_remove("http_proxy")
        .env_remove("ALL_PROXY")
        .env_remove("all_proxy")
        .env("NO_PROXY", "127.0.0.1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// A stand-in server run from `dir`: a copy of the launcher there, blocked
/// reading a FIFO named as its release listing (a copy of /bin/sleep is
/// killed at its start on macOS).
fn stand_in_server(f: &Fixture, dir: &Path) -> Child {
    let program = dir.join("sleeper");
    fs::copy(env!("CARGO_BIN_EXE_basert"), &program).unwrap();
    let fifo = f.path("never");
    if !fifo.exists() {
        assert!(Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
    }
    let mut server = Command::new(&program)
        .args(["self", "check", "--record"])
        .env("BASERT_RELEASES", &fifo)
        .env("BASERT_STATE_DIR", f.path("state"))
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        server.try_wait().unwrap().is_none(),
        "the stand-in server is up"
    );
    server
}

#[test]
fn a_start_waits_for_an_install_into_the_shared_directory_and_runs_it() {
    let mut f = Fixture::in_place();
    // The release comes over connections the test holds open.
    let tarball = f.superfluid_releases[0]["assets"][0]["browser_download_url"]
        .as_str()
        .unwrap()
        .to_string();
    let (url, go) = held_download(fs::read(&tarball).unwrap());
    f.superfluid_releases[0]["assets"][0]["browser_download_url"] = url.into();
    // Two package copies, which share basert's data directory.
    let a = package_copy(
        &f,
        "brew/Cellar/basert/0.4.3/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    let b = package_copy(
        &f,
        "node/lib/node_modules/@basecompute/basert-darwin-arm64/engine",
        Some("npm: npm install --global @basecompute/basert@latest"),
        "node/bin/npm",
    );
    let model = f.path("m.base");
    fs::write(&model, "").unwrap();
    let data = f.data_superfluid();

    let mut first = start(f.command(&a, &["superfluid", "install"], &[]));
    let first_lines = stderr_lines(&mut first);
    let mut seen = Vec::new();
    assert!(
        wait_for(&first_lines, &["downloading superfluid v0.1.3"], &mut seen).is_some(),
        "{seen:?}"
    );
    // B starts while A's download is in flight: it waits for A's install
    // instead of making one of its own beside it.
    let mut second = start(f.command(&b, &["serve", model.to_str().unwrap()], &[]));
    let second_lines = stderr_lines(&mut second);
    let mut second_seen = Vec::new();
    let said = wait_for(
        &second_lines,
        &[
            "waiting for another basert installing superfluid into",
            "downloading",
        ],
        &mut second_seen,
    );
    go.send(()).unwrap();
    let first = first.wait_with_output().unwrap();
    let second_out = second.wait_with_output().unwrap();
    seen.extend(first_lines.try_iter());
    second_seen.extend(second_lines.try_iter());
    assert!(
        said.is_some_and(|l| l.contains("waiting for another basert")),
        "{second_seen:?}"
    );
    assert!(first.status.success(), "{seen:?}");
    assert!(second_out.status.success(), "{second_seen:?}");
    // A's install serves B's engine too: B runs it, downloading nothing.
    assert!(
        !second_seen.iter().any(|l| l.contains("downloading")),
        "{second_seen:?}"
    );
    assert_eq!(
        fs::read_to_string(data.join(".release")).unwrap(),
        "v0.1.3\n"
    );
}

#[test]
fn an_install_is_not_swapped_in_under_a_server_started_while_it_downloaded() {
    let mut f = Fixture::in_place();
    let tarball = f.superfluid_releases[0]["assets"][0]["browser_download_url"]
        .as_str()
        .unwrap()
        .to_string();
    let (url, go) = held_download(fs::read(&tarball).unwrap());
    f.superfluid_releases[0]["assets"][0]["browser_download_url"] = url.into();
    let a = package_copy(
        &f,
        "brew/Cellar/basert/0.4.3/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    let data = f.data_superfluid();

    let mut install = start(f.command(&a, &["superfluid", "install"], &[]));
    let lines = stderr_lines(&mut install);
    let mut seen = Vec::new();
    assert!(
        wait_for(&lines, &["downloading superfluid v0.1.3"], &mut seen).is_some(),
        "{seen:?}"
    );
    // Meanwhile a superfluid lands there (another basert's, or one copied by
    // hand) and a server starts from it.
    superfluid_at(&data, "0.1.4", 2);
    fs::write(data.join(".release"), "v0.1.4\n").unwrap();
    let mut server = stand_in_server(&f, &data.join("libexec").join("superfluid"));
    go.send(()).unwrap();
    let out = install.wait_with_output().unwrap();
    seen.extend(lines.try_iter());
    let alive = server.try_wait().unwrap().is_none();
    let _ = server.kill();
    let _ = server.wait();
    assert_eq!(out.status.code(), Some(1), "{seen:?}");
    assert!(
        seen.iter().any(|l| l.contains(&format!(
            "a server runs from {} (pid {})",
            data.display(),
            server.id()
        ))),
        "{seen:?}"
    );
    assert!(alive, "the server is still running");
    assert_eq!(
        fs::read_to_string(data.join(".release")).unwrap(),
        "v0.1.4\n"
    );
}

/// superfluid 0.1.3 at `root` (over what is there) whose server is a
/// program run from elsewhere, cat blocked on the FIFO `$HOLD`, which no look
/// at what runs from the directory finds.
fn holding_superfluid(root: &Path) {
    fs::create_dir_all(root.join("bin")).unwrap();
    executable(
        &root.join("bin").join("superfluid"),
        "#!/bin/sh\n\
         case \"$1\" in\n\
           --version) echo \"superfluid 0.1.3\" ;;\n\
           runtimes) echo '[{\"id\":\"basert\",\"state\":\"ready\",\"ready\":true,\"abi\":1}]' ;;\n\
           serve) exec cat \"$HOLD\" ;;\n\
         esac\n",
    );
}

/// `launcher serve` of a model, with `env`, once it has become the server (the
/// same process: the cat of [`holding_superfluid`]).
fn serve_held(f: &Fixture, launcher: &Path, env: &[(&str, &Path)]) -> Child {
    let hold = f.path("hold");
    if !hold.exists() {
        assert!(Command::new("mkfifo")
            .arg(&hold)
            .status()
            .unwrap()
            .success());
    }
    let model = f.path("m.base");
    fs::write(&model, "").unwrap();
    let mut env = env.to_vec();
    env.push(("HOLD", hold.as_path()));
    let mut server = start(f.command(launcher, &["serve", model.to_str().unwrap()], &env));
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let comm = Command::new("ps")
            .args(["-o", "comm=", "-p", &server.id().to_string()])
            .output()
            .unwrap();
        if String::from_utf8_lossy(&comm.stdout)
            .trim()
            .ends_with("cat")
        {
            return server;
        }
        if let Some(status) = server.try_wait().unwrap() {
            let out = server.wait_with_output().unwrap();
            panic!(
                "the server ended ({status}): {}",
                String::from_utf8_lossy(&out.stdout)
            );
        }
        assert!(Instant::now() < deadline, "the server never came up");
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn a_server_basert_serve_started_keeps_its_superfluid_in_place() {
    let f = Fixture::in_place();
    let a = package_copy(
        &f,
        "brew/Cellar/basert/0.4.3/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    let data = f.data_superfluid();
    superfluid_at(&data, "0.1.3", 1);
    fs::write(data.join(".release"), "v0.1.3\n").unwrap();
    holding_superfluid(&data);
    let mut server = serve_held(&f, &a, &[]);

    let (code, out, err) = f.launch(&a, &["superfluid", "install", "--force"]);
    let _ = server.kill();
    let _ = server.wait();
    assert_eq!(code, 1, "{out}{err}");
    assert!(
        err.contains(&format!("a server runs from {}", data.display())),
        "{err}"
    );
    assert!(err.contains("one basert serve started"), "{err}");
    let said = Command::new(data.join("bin").join("superfluid"))
        .arg("--version")
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&said.stdout).trim(),
        "superfluid 0.1.3"
    );
}

#[test]
fn status_and_the_printed_command_name_the_superfluid_a_start_runs() {
    let f = Fixture::in_place();
    let exe = package_copy(
        &f,
        "brew/Cellar/basert/0.4.2/libexec",
        Some("Homebrew: brew upgrade basert"),
        "brew/bin/brew",
    );
    // The data directory's serves ABI 2, for another basert; the one a
    // basert of 0.4.2 put beside this one serves its ABI 1 engine.
    let data = f.data_superfluid();
    superfluid_at(&data, "0.1.4", 2);
    fs::write(data.join(".release"), "v0.1.4\n").unwrap();
    let beside = exe.parent().unwrap().join("superfluid");
    superfluid_at(&beside, "0.1.3", 1);
    fs::write(beside.join(".release"), "v0.1.3\n").unwrap();
    let in_keg = "/brew/Cellar/basert/0.4.2/libexec/superfluid";
    let model = f.path("m.base");
    fs::write(&model, "").unwrap();
    let model = model.to_str().unwrap();

    let (code, out, err) = f.launch(&exe, &["superfluid", "status"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("installed: v0.1.3 ("), "{out}");
    assert!(out.contains(in_keg), "{out}");
    assert!(out.contains("engine:    ready to serve"), "{out}");
    let (code, out, err) = f.launch(&exe, &["serve", model, "--print-command"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        out.contains(&format!("{in_keg}/bin/superfluid serve ")),
        "{out}"
    );
    let (code, out, err) = f.launch(&exe, &["serve", "--version"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("superfluid v0.1.3 ("), "{out}");
    assert!(out.contains(in_keg), "{out}");
    // And a start runs it, as before.
    let (code, out, err) = f.launch(&exe, &["serve", model]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("installing superfluid"), "{err}");
}

/// A release bundle of this version holding the launcher under test, for
/// `self install --from`: `<tmp>/<name>`, whose engine states ABI 1.
fn real_bundle(f: &Fixture, name: &str) -> PathBuf {
    let dir = f.path(name);
    fs::create_dir_all(&dir).unwrap();
    fs::copy(env!("CARGO_BIN_EXE_basert"), dir.join("basert")).unwrap();
    engine_files(&dir, HERE, 1);
    dir
}

/// The fixture's install moved into the versions layout by `self install`,
/// with this version (the launcher under test) in use.
fn managed(f: &Fixture) {
    let bundle = real_bundle(f, "bundle-here");
    let (code, out, err) = f.run(
        &f.root.join("basert"),
        &[
            "install",
            "--from",
            bundle.to_str().unwrap(),
            "--root",
            f.root.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "{out}{err}");
    assert_eq!(f.current().as_deref(), Some(HERE));
}

#[test]
fn a_flat_install_is_not_moved_under_a_server_running_on_it() {
    let mut f = Fixture::in_place();
    let next = next_version();
    f.release(&next, 1);
    holding_superfluid(&f.root.join("superfluid"));
    let mut server = serve_held(&f, &f.root.join("basert"), &[]);
    // `self update`, and the install script's `self install`.
    let (update, _, update_err) = f.run(&f.root.join("basert"), &["update"]);
    let bundle = real_bundle(&f, "bundle");
    let (install, _, install_err) = f.run(
        &f.root.join("basert"),
        &[
            "install",
            "--from",
            bundle.to_str().unwrap(),
            "--root",
            f.root.to_str().unwrap(),
        ],
    );
    let alive = server.try_wait().unwrap().is_none();
    let _ = server.kill();
    let _ = server.wait();
    // `self update` names the root by its real path, `self install` as given.
    let said = |root: &Path| {
        format!(
            "a server runs on the engine in {} (pid {}), whose files this would replace: stop it, then run this again",
            root.display(),
            server.id()
        )
    };
    assert_eq!(update, 1, "{update_err}");
    assert!(
        update_err.contains(&said(&fs::canonicalize(&f.root).unwrap())),
        "{update_err}"
    );
    assert_eq!(install, 1, "{install_err}");
    assert!(install_err.contains(&said(&f.root)), "{install_err}");
    assert!(alive);
    // Still the flat copy, the engine's files in place.
    for name in ["basert", &format!("libbaseRT.{HERE}.dylib")] {
        assert!(
            fs::symlink_metadata(f.root.join(name)).unwrap().is_file(),
            "{name}"
        );
    }
    assert!(f.current().is_none());
    assert!(!f.version_dir(&next).exists());
}

#[test]
fn reinstalling_the_version_a_server_runs_on_keeps_it_and_its_record() {
    let f = Fixture::in_place();
    managed(&f);
    holding_superfluid(&f.root.join("superfluid"));
    let marker = f.version_dir(HERE).join("marker");
    fs::write(&marker, "the copy the server runs on").unwrap();
    let mut server = serve_held(&f, &f.root.join("basert"), &[]);
    let record = f
        .version_dir(HERE)
        .join(".serving")
        .join(server.id().to_string());
    assert!(record.is_file(), "{}", record.display());

    let again = real_bundle(&f, "bundle-again");
    let (code, out, err) = f.run(
        &f.root.join("basert"),
        &[
            "install",
            "--from",
            again.to_str().unwrap(),
            "--root",
            f.root.to_str().unwrap(),
        ],
    );
    let kept = (marker.is_file(), record.is_file());
    let _ = server.kill();
    let _ = server.wait();
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains(&format!(
            "basert {HERE} in {} stays as it is: a server runs on it (pid {})",
            f.version_dir(HERE).display(),
            server.id()
        )),
        "{err}"
    );
    assert_eq!(kept, (true, true));
    assert_eq!(f.current().as_deref(), Some(HERE));
}

#[test]
fn a_server_on_the_engine_basert_lib_names_keeps_its_version() {
    let mut f = Fixture::in_place();
    managed(&f);
    let next = next_version();
    f.release(&next, 1);
    f.release("0.0.1", 1);
    holding_superfluid(&f.root.join("superfluid"));
    let engine = f.version_dir(HERE);
    let mut server = serve_held(
        &f,
        &f.root.join("basert"),
        &[("BASERT_LIB", engine.as_path())],
    );
    // Two switches away: it is neither the version in use nor the one before.
    let launcher = engine.join("basert");
    let (first, out1, err1) = f.run(&launcher, &["update", "--version", &next]);
    let (second, out2, err2) = f.run(&launcher, &["update", "--version", "0.0.1"]);
    let kept = engine.join("basert").is_file();
    let _ = server.kill();
    let _ = server.wait();
    assert_eq!(first, 0, "{out1}{err1}");
    assert_eq!(second, 0, "{out2}{err2}");
    assert_eq!(f.current().as_deref(), Some("0.0.1"));
    assert!(kept, "{}", engine.display());
}
