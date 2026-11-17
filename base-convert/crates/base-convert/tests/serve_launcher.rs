//! `basert serve` and `basert superfluid`, end to end against a fake superfluid
//! release: the launcher installs it from a directory of release assets, or
//! the newest a release listing names, asks it whether it can serve the
//! engine, replaces one that cannot, and execs it with the C++ server's
//! command line translated to superfluid's.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn target() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        other => panic!("no superfluid release for {other:?}"),
    }
}

/// A release directory, an install directory for it, a model file, a state
/// directory and an engine directory, all under one temporary directory.
struct Fixture {
    tmp: TempDir,
    assets: PathBuf,
    install: PathBuf,
    model: PathBuf,
    state: PathBuf,
    lib: PathBuf,
}

fn executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// Release assets whose `superfluid` is a shell script: `--version` answers
/// `version`; `runtimes --json` reports the basert runtime ready, unless
/// `FAKE_SUPERFLUID_UNREADY` names a reason and `FAKE_SUPERFLUID_UNREADY_VERSION`
/// is this version, and notes the environment it was probed under in the
/// file `FAKE_SUPERFLUID_PROBE_LOG` names; anything else echoes its arguments
/// and the environment the launcher sets.
fn release_assets(dir: &Path, version: &str) -> PathBuf {
    let name = format!("superfluid-{}", target());
    let tree = dir.join("tree");
    let bin = tree.join(&name).join("bin");
    let libexec = tree.join(&name).join("libexec").join("superfluid");
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir_all(&libexec).unwrap();
    executable(
        &bin.join("superfluid"),
        &format!(
            "#!/bin/sh\n\
             if [ \"$1\" = \"--version\" ]; then echo \"superfluid {version}\"; exit 0; fi\n\
             if [ \"$1\" = \"runtimes\" ]; then\n\
               if [ -n \"${{FAKE_SUPERFLUID_PROBE_LOG:-}}\" ]; then echo \"PROBE: BASERT_LIB=${{BASERT_LIB:-unset}} SUPERFLUID_HOME=${{SUPERFLUID_HOME:-unset}} SUPERFLUID_NO_UPDATE_CHECK=${{SUPERFLUID_NO_UPDATE_CHECK:-unset}}\" >> \"$FAKE_SUPERFLUID_PROBE_LOG\"; fi\n\
               if [ -n \"${{FAKE_SUPERFLUID_UNREADY:-}}\" ] && [ \"${{FAKE_SUPERFLUID_UNREADY_VERSION:-}}\" = \"{version}\" ]; then\n\
                 echo \"[{{\\\"id\\\":\\\"basert\\\",\\\"state\\\":\\\"not ready\\\",\\\"ready\\\":\\\"$FAKE_SUPERFLUID_UNREADY\\\",\\\"version\\\":\\\"{version}\\\"}}]\"\n\
               else\n\
                 echo \"[{{\\\"id\\\":\\\"basert\\\",\\\"state\\\":\\\"ready\\\",\\\"ready\\\":true,\\\"version\\\":\\\"{version}\\\",\\\"abi\\\":1}}]\"\n\
               fi\n\
               exit 0\n\
             fi\n\
             printf 'ARGS:'; for a in \"$@\"; do printf ' %s' \"$a\"; done; printf '\\n'\n\
             echo \"SUPERFLUID_HOME=$SUPERFLUID_HOME\"\n\
             echo \"SUPERFLUID_NO_UPDATE_CHECK=${{SUPERFLUID_NO_UPDATE_CHECK:-unset}}\"\n\
             echo \"SUPERFLUID_ENGINE_VERBOSE=${{SUPERFLUID_ENGINE_VERBOSE:-unset}}\"\n"
        ),
    );
    executable(&libexec.join("superfluid-worker-basert"), "#!/bin/sh\n");
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
    let sum = format!("{:x}", Sha256::digest(fs::read(&tarball).unwrap()));
    fs::write(
        assets.join(format!("{name}.tar.gz.sha256")),
        format!("{sum}  {name}.tar.gz\n"),
    )
    .unwrap();
    assets
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let assets = release_assets(&tmp.path().join("release"), "9.9.9");
    let model = tmp.path().join("m.base");
    fs::write(&model, b"").unwrap();
    let lib = tmp.path().join("lib");
    fs::create_dir(&lib).unwrap();
    Fixture {
        install: tmp.path().join("superfluid"),
        state: tmp.path().join("state"),
        assets,
        model,
        lib,
        tmp,
    }
}

impl Fixture {
    /// `basert` with its superfluid, its release source, its state and the
    /// engine directory all under the fixture, stdin closed (no terminal),
    /// and no policy set: a start without a terminal downloads.
    fn basert(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_basert"));
        c.env("BASERT_SUPERFLUID_DIR", &self.install)
            .env("BASERT_SUPERFLUID_SOURCE", &self.assets)
            .env("BASERT_STATE_DIR", &self.state)
            .env("BASERT_LIB", &self.lib)
            // `basert serve` speculates by default, which reads the model
            // cache and the catalog: the fixture's, and the bundled catalog.
            .env("BASERT_MODELS_DIR", self.tmp.path().join("models-cache"))
            .env("BASERT_CATALOG_OFFLINE", "1")
            .env("FAKE_SUPERFLUID_PROBE_LOG", self.probe_log())
            .env_remove("BASERT_SUPERFLUID_AUTO_INSTALL")
            .env_remove("FAKE_SUPERFLUID_UNREADY")
            .env_remove("FAKE_SUPERFLUID_UNREADY_VERSION")
            .stdin(Stdio::null());
        c
    }

    fn model(&self) -> &str {
        self.model.to_str().unwrap()
    }

    fn probe_log(&self) -> PathBuf {
        self.tmp.path().join("probes.log")
    }

    /// What the fake superfluid was probed under, one line per probe.
    fn probes(&self) -> String {
        fs::read_to_string(self.probe_log()).unwrap_or_default()
    }

    fn stamp(&self) -> String {
        fs::read_to_string(self.install.join(".release"))
            .unwrap()
            .trim()
            .to_string()
    }

    /// Install, from assets of `version`, stamped as that version: a copy of
    /// another release than the pin.
    fn install_old(&self, version: &str) -> PathBuf {
        let assets = release_assets(&self.tmp.path().join(format!("release-{version}")), version);
        let (code, out, err) = run(self
            .basert()
            .env("BASERT_SUPERFLUID_SOURCE", &assets)
            .args(["superfluid", "install"]));
        assert_eq!(code, 0, "{out}{err}");
        fs::write(self.install.join(".release"), format!("v{version}\n")).unwrap();
        assets
    }
}

fn run(command: &mut Command) -> (i32, String, String) {
    let Output {
        status,
        stdout,
        stderr,
    } = command.output().unwrap();
    (
        status.code().unwrap_or(-1),
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

/// The release basert was tested with, as `basert superfluid status` prints
/// it: what a directory of release assets holds.
fn pinned(f: &Fixture) -> String {
    let (_, out, _) = run(f.basert().args(["superfluid", "status"]));
    out.lines()
        .find_map(|l| l.strip_prefix("tested:"))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| panic!("no tested line in {out:?}"))
}

/// Serves the files under `root` over HTTP on 127.0.0.1 while the test runs,
/// as GitHub serves a release's assets; returns the base URL.
fn serve_files(root: PathBuf) -> String {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            if reader.read_line(&mut request).is_err() {
                continue;
            }
            let mut line = String::new();
            while reader.read_line(&mut line).is_ok_and(|n| n > 0) && line != "\r\n" {
                line.clear();
            }
            let path = request.split_whitespace().nth(1).unwrap_or("/");
            let _ = match fs::read(root.join(path.trim_start_matches('/'))) {
                Ok(body) => write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .and_then(|_| stream.write_all(&body)),
                Err(_) => write!(
                    stream,
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                ),
            };
        }
    });
    base
}

#[test]
fn without_superfluid_the_newest_release_that_serves_the_engine_is_installed() {
    let f = fixture();
    // A mirror's layout: each checksum in another directory than its tarball.
    let www = f.tmp.path().join("www");
    for version in ["1.2.0", "2.0.0"] {
        let assets = release_assets(&f.tmp.path().join(format!("release-{version}")), version);
        for entry in fs::read_dir(&assets).unwrap() {
            let entry = entry.unwrap();
            let kind = match entry.path().extension().and_then(|e| e.to_str()) {
                Some("sha256") => "checksums",
                _ => "files",
            };
            let dir = www.join(kind).join(format!("v{version}"));
            fs::create_dir_all(&dir).unwrap();
            fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
    }
    let base = serve_files(www);
    let name = format!("superfluid-{}", target());
    let release = |tag: &str, abi: u32| {
        serde_json::json!({"tag_name": tag, "draft": false, "prerelease": false, "assets": [
            {"name": format!("{name}.tar.gz"), "browser_download_url": format!("{base}/files/{tag}/{name}.tar.gz")},
            {"name": format!("{name}.tar.gz.sha256"), "browser_download_url": format!("{base}/checksums/{tag}/{name}.tar.gz.sha256")},
            {"name": format!("superfluid-basert-abi-{abi}.txt"), "browser_download_url": format!("{base}/{tag}/abi")},
        ]})
    };
    // 2.0.0 serves another engine ABI: 1.2.0 is the newest that serves this one.
    let listing = f.tmp.path().join("releases.json");
    let list = |releases: Vec<serde_json::Value>| {
        fs::write(&listing, serde_json::Value::from(releases).to_string()).unwrap()
    };
    list(vec![
        release("v2.0.0", 2),
        release("v1.2.0", 1),
        release("v1.1.0", 1),
    ]);
    let basert = || {
        let mut c = f.basert();
        c.env_remove("BASERT_SUPERFLUID_SOURCE")
            .env("BASERT_SUPERFLUID_RELEASES", &listing);
        c
    };

    let (code, out, err) = run(basert()
        .env("BASERT_SUPERFLUID_AUTO_INSTALL", "0")
        .args(["serve", f.model()]));
    assert_eq!(code, 1, "{out}{err}");
    assert!(err.contains("superfluid is not installed"), "{err}");
    assert!(
        !err.contains("v1.2.0"),
        "a start forbidden to download looks nothing up: {err}"
    );

    let (code, out, err) = run(basert().args(["serve", f.model()]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains("basert serve: installing superfluid v1.2.0 into"),
        "{err}"
    );
    assert!(
        err.contains(&format!(
            "downloading superfluid v1.2.0 ({name}.tar.gz) from {base}/files/v1.2.0/{name}.tar.gz"
        )),
        "{err}"
    );
    assert_eq!(f.stamp(), "v1.2.0");
    assert!(out.contains("ARGS: serve"), "{out}");

    let (code, out, _) = run(basert().args(["superfluid", "install"]));
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("superfluid v1.2.0 is installed at"), "{out}");

    // A listing whose newest is older than the install downgrades nothing.
    list(vec![release("v1.1.0", 1)]);
    let (code, out, err) = run(basert().args(["superfluid", "install"]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        out.contains("newer than v1.1.0: nothing is downgraded"),
        "{out}"
    );
    assert_eq!(f.stamp(), "v1.2.0");
}

#[test]
fn superfluid_install_puts_the_pinned_release_in_place_and_status_says_so() {
    let f = fixture();
    let (code, out, _) = run(f.basert().args(["superfluid", "status"]));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("installed: nothing"), "{out}");
    let pin = pinned(&f);
    assert!(pin.starts_with('v'), "{pin}");

    let (code, out, err) = run(f.basert().args(["superfluid", "install"]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains(&format!("downloading superfluid {pin} (")),
        "{err}"
    );
    assert!(
        out.contains(&format!("superfluid {pin} is installed at")),
        "{out}"
    );
    assert!(f.install.join("bin").join("superfluid").is_file());
    assert!(f
        .install
        .join("libexec")
        .join("superfluid")
        .join("superfluid-worker-basert")
        .is_file());
    assert_eq!(f.stamp(), pin);

    let (code, out, _) = run(f.basert().args(["superfluid", "status"]));
    assert_eq!(code, 0, "{out}");
    assert!(out.contains(&format!("installed: {pin} (")), "{out}");
    assert!(out.contains("engine:    ready to serve"), "{out}");

    // Installed at the pin: nothing to download.
    let (code, out, err) = run(f.basert().args(["superfluid", "install"]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(err.is_empty(), "{err}");
    assert!(out.contains("is installed at"), "{out}");

    // One that cannot serve the engine: status says so and exits 2.
    let (code, out, _) = run(f
        .basert()
        .env("FAKE_SUPERFLUID_UNREADY", "libbaseRT states no ABI number")
        .env("FAKE_SUPERFLUID_UNREADY_VERSION", "9.9.9")
        .args(["superfluid", "status"]));
    assert_eq!(code, 2, "{out}");
    assert!(
        out.contains("engine:    cannot serve: libbaseRT states no ABI number"),
        "{out}"
    );
}

#[test]
fn serve_without_superfluid_and_without_a_terminal_installs_it_and_says_so() {
    let f = fixture();
    let pin = pinned(&f);
    let (code, out, err) = run(f.basert().args(["serve", f.model()]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains(&format!("basert serve: installing superfluid {pin} into")),
        "{err}"
    );
    assert!(
        err.contains("BASERT_SUPERFLUID_AUTO_INSTALL=0 forbids this"),
        "{err}"
    );
    assert!(
        err.contains(&format!("downloading superfluid {pin} (")),
        "{err}"
    );
    assert_eq!(f.stamp(), pin);
    assert!(out.contains("ARGS: serve"), "{out}");
    // The probe ran against the engine, under the server's own home, with
    // superfluid's update check off.
    let probes = f.probes();
    assert!(
        probes.contains(&format!(
            "PROBE: BASERT_LIB={} SUPERFLUID_HOME={} SUPERFLUID_NO_UPDATE_CHECK=1",
            f.lib.display(),
            f.state.join("superfluid").display()
        )),
        "{probes}"
    );
}

#[test]
fn a_start_forbidden_to_download_names_the_install_command() {
    let f = fixture();
    let (code, out, err) = run(f
        .basert()
        .env("BASERT_SUPERFLUID_AUTO_INSTALL", "0")
        .args(["serve", f.model()]));
    assert_eq!(code, 1, "{out}{err}");
    assert!(err.contains("superfluid is not installed"), "{err}");
    assert!(err.contains("basert superfluid install"), "{err}");
    assert!(!f.install.exists(), "nothing is downloaded when forbidden");
}

#[test]
fn serve_execs_superfluid_with_the_cpp_servers_command_line_translated() {
    let f = fixture();
    assert_eq!(run(f.basert().args(["superfluid", "install"])).0, 0);
    let sessions = f.state.join("local");
    let (code, out, err) = run(f.basert().args([
        "serve",
        f.model(),
        "--port",
        "9000",
        "--api-key",
        "k",
        "--max-batch",
        "2",
        "--sessions",
        sessions.to_str().unwrap(),
        "--paged-kv",
        "--verbose",
    ]));
    assert_eq!(code, 0, "{out}{err}");
    let want = format!(
        "ARGS: serve --model {} --runtime basert --sessions {} --host 127.0.0.1 --port 9000 \
         --max-batch 2 --api-key k --offline --no-park --sessions-cache --no-tui --no-http-qos-header \
         --speculate auto",
        f.model(),
        sessions.display()
    );
    assert!(out.lines().any(|l| l == want), "{out}\nwant {want}");
    assert!(
        out.contains(&format!(
            "SUPERFLUID_HOME={}",
            f.state.join("superfluid").display()
        )),
        "{out}"
    );
    assert!(out.contains("SUPERFLUID_NO_UPDATE_CHECK=1"), "{out}");
    assert!(out.contains("SUPERFLUID_ENGINE_VERBOSE=1"), "{out}");
    assert!(err.contains("--paged-kv is no longer used"), "{err}");
    // BASERT_LIB names the engine: superfluid reads it itself.
    assert!(!out.contains("--basert-lib"), "{out}");
    // No download was needed: nothing said of one.
    assert!(!err.contains("installing superfluid"), "{err}");
}

#[test]
fn superfluids_own_flags_pass_through_and_print_command_shows_the_start() {
    let f = fixture();
    assert_eq!(run(f.basert().args(["superfluid", "install"])).0, 0);
    let (code, out, err) = run(f.basert().args([
        "serve",
        f.model(),
        "--speculate",
        "auto",
        "--prefix-blob-budget-mb",
        "512",
        "--",
        "--anything",
    ]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        out.lines().any(|l| l.ends_with(
            "--no-http-qos-header --speculate auto --prefix-blob-budget-mb 512 --anything"
        )),
        "{out}"
    );
    // `--speculate off` is handed over alone: no default `auto` beside it.
    let (code, out, err) = run(f.basert().args(["serve", f.model(), "--speculate", "off"]));
    assert_eq!(code, 0, "{out}{err}");
    let args = out
        .lines()
        .find(|l| l.starts_with("ARGS: serve"))
        .unwrap_or_else(|| panic!("{out}"));
    assert!(
        args.ends_with("--no-http-qos-header --speculate off"),
        "{args}"
    );
    assert_eq!(args.matches("--speculate").count(), 1, "{args}");

    // --print-command shows the start and runs nothing, installed or not.
    let (code, out, err) =
        run(f
            .basert()
            .args(["serve", f.model(), "--print-command", "--port", "9001"]));
    assert_eq!(code, 0, "{out}{err}");
    let line = out
        .lines()
        .find(|l| l.starts_with("SUPERFLUID_HOME="))
        .unwrap_or_else(|| panic!("{out}"));
    assert!(line.contains("SUPERFLUID_NO_UPDATE_CHECK=1"), "{line}");
    assert!(
        line.contains(&format!(
            "{} serve --model {}",
            f.install.join("bin").join("superfluid").display(),
            f.model()
        )),
        "{line}"
    );
    assert!(line.contains("--port 9001"), "{line}");
    assert!(line.ends_with("--speculate auto"), "the default: {line}");
    assert!(!out.contains("ARGS:"), "{out}");
    let g = fixture();
    let (code, out, _) = run(g.basert().args(["serve", g.model(), "--print-command"]));
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("superfluid (not installed;"), "{out}");
    assert!(!g.install.exists(), "print-command installs nothing");
}

#[test]
fn serve_hands_superfluid_the_engine_library_beside_the_launcher() {
    let f = fixture();
    // A bundle: basert and libbaseRT in one directory, superfluid installed
    // beside them by default.
    let bundle = f.install.parent().unwrap().join("bundle");
    fs::create_dir(&bundle).unwrap();
    fs::copy(env!("CARGO_BIN_EXE_basert"), bundle.join("basert")).unwrap();
    for lib in ["libbaseRT.dylib", "libbaseRT.so"] {
        fs::write(bundle.join(lib), b"").unwrap();
    }
    let basert = || {
        let mut c = Command::new(bundle.join("basert"));
        c.env("BASERT_SUPERFLUID_SOURCE", &f.assets)
            .env("BASERT_STATE_DIR", &f.state)
            .env("FAKE_SUPERFLUID_PROBE_LOG", f.probe_log())
            .env("HOME", f.tmp.path())
            .env("XDG_DATA_HOME", f.tmp.path().join("xdg"))
            .env_remove("BASERT_SUPERFLUID_DIR")
            .env_remove("BASERT_LIB")
            .env_remove("BASERT_SUPERFLUID_AUTO_INSTALL")
            .stdin(Stdio::null());
        c
    };
    let (code, out, err) = run(basert().args(["superfluid", "install"]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(bundle
        .join("superfluid")
        .join("bin")
        .join("superfluid")
        .is_file());

    let (code, out, err) = run(basert().args(["serve", f.model()]));
    assert_eq!(code, 0, "{out}{err}");
    let canonical = fs::canonicalize(&bundle).unwrap();
    assert!(
        out.contains(&format!("--basert-lib {}", canonical.display())),
        "{out}"
    );
    assert!(out.contains("--port 8453 --max-batch 1"), "{out}");
    // The probe was told the same engine directory.
    let probes = f.probes();
    assert!(
        probes.contains(&format!("PROBE: BASERT_LIB={}", canonical.display())),
        "{probes}"
    );
    // The user's own --basert-lib is superfluid's flag, given once, and the
    // probe's engine.
    let (code, out, err) =
        run(basert().args(["serve", f.model(), "--basert-lib", f.lib.to_str().unwrap()]));
    assert_eq!(code, 0, "{out}{err}");
    assert_eq!(out.matches("--basert-lib").count(), 1, "{out}");
    assert!(
        out.contains(&format!("--basert-lib {}", f.lib.display())),
        "{out}"
    );
    let probes = f.probes();
    assert!(
        probes.contains(&format!("PROBE: BASERT_LIB={}", f.lib.display())),
        "{probes}"
    );
}

#[test]
fn a_bundle_that_cannot_be_written_gets_superfluid_in_the_users_data_directory() {
    use std::os::unix::fs::PermissionsExt;
    let f = fixture();
    let bundle = f.tmp.path().join("app");
    fs::create_dir(&bundle).unwrap();
    fs::copy(env!("CARGO_BIN_EXE_basert"), bundle.join("basert")).unwrap();
    fs::write(bundle.join("libbaseRT.dylib"), b"").unwrap();
    fs::write(bundle.join("libbaseRT.so"), b"").unwrap();
    let home = f.tmp.path().join("home");
    fs::create_dir(&home).unwrap();
    fs::set_permissions(&bundle, fs::Permissions::from_mode(0o555)).unwrap();
    let basert = || {
        let mut c = Command::new(bundle.join("basert"));
        c.env("BASERT_SUPERFLUID_SOURCE", &f.assets)
            .env("BASERT_STATE_DIR", &f.state)
            .env("HOME", &home)
            .env("XDG_DATA_HOME", home.join(".local").join("share"))
            .env_remove("BASERT_SUPERFLUID_DIR")
            .env_remove("BASERT_LIB")
            .env_remove("BASERT_SUPERFLUID_AUTO_INSTALL")
            .stdin(Stdio::null());
        c
    };
    let (code, out, err) = run(basert().args(["serve", f.model()]));
    fs::set_permissions(&bundle, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        !bundle.join("superfluid").exists(),
        "nothing written into the bundle"
    );
    // Under the user's data directory, wherever the platform keeps it.
    let installed = err
        .lines()
        .find_map(|l| l.strip_prefix("basert serve: installed superfluid "))
        .unwrap_or_else(|| panic!("{err}"));
    let (_, at) = installed.split_once(" at ").unwrap();
    assert!(at.starts_with(home.to_str().unwrap()), "{at}");
    assert!(at.ends_with("baseRT/superfluid"), "{at}");
    assert!(Path::new(at).join("bin").join("superfluid").is_file());
    assert!(out.contains("ARGS: serve"), "{out}");
    // The next start finds it there and installs nothing.
    let (code, out, err) = run(basert().args(["serve", f.model()]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("installing superfluid"), "{err}");
    assert!(out.contains("ARGS: serve"), "{out}");
}

#[test]
fn an_installed_copy_that_cannot_serve_the_engine_is_brought_to_the_pin() {
    let f = fixture();
    f.install_old("0.0.1");
    let pin = pinned(&f);
    let unready = |c: &mut Command| {
        c.env(
            "FAKE_SUPERFLUID_UNREADY",
            "libbaseRT 0.4.0 states no ABI number and is not a 0.3 library",
        )
        .env("FAKE_SUPERFLUID_UNREADY_VERSION", "0.0.1");
    };
    let mut status = f.basert();
    unready(&mut status);
    let (code, out, _) = run(status.args(["superfluid", "status"]));
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("installed: v0.0.1 ("), "{out}");

    let mut serve = f.basert();
    unready(&mut serve);
    let (code, out, err) = run(serve.args(["serve", f.model()]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        err.contains(&format!(
            "superfluid v0.0.1 at {} cannot serve this engine (libbaseRT 0.4.0 states no ABI number and is not a 0.3 library); {pin} can",
            f.install.display()
        )),
        "{err}"
    );
    assert!(
        err.contains(&format!("installing superfluid {pin} into")),
        "{err}"
    );
    assert_eq!(f.stamp(), pin);
    assert!(out.contains("ARGS: serve"), "{out}");

    // A release the probe accepts is kept as it is, pinned or not.
    let g = fixture();
    g.install_old("0.0.1");
    let (code, out, err) = run(g.basert().args(["serve", g.model()]));
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("installing superfluid"), "{err}");
    assert_eq!(g.stamp(), "v0.0.1");
    assert!(out.contains("ARGS: serve"), "{out}");
}

#[test]
fn the_pin_that_cannot_serve_the_engine_is_an_error_and_no_download() {
    let f = fixture();
    assert_eq!(run(f.basert().args(["superfluid", "install"])).0, 0);
    let pin = pinned(&f);
    let (code, out, err) = run(f
        .basert()
        .env(
            "FAKE_SUPERFLUID_UNREADY",
            "libbaseRT states ABI 2, this build serves ABI 1",
        )
        .env("FAKE_SUPERFLUID_UNREADY_VERSION", "9.9.9")
        .args(["serve", f.model()]));
    assert_eq!(code, 1, "{out}{err}");
    assert!(
        err.contains(&format!(
            "superfluid {pin} at {} cannot serve this engine: libbaseRT states ABI 2, this build serves ABI 1",
            f.install.display()
        )),
        "{err}"
    );
    assert!(!err.contains("downloading"), "{err}");
    assert!(!out.contains("ARGS:"), "{out}");
}

#[test]
fn a_failed_install_of_the_pin_is_an_error_that_names_the_cause() {
    let f = fixture();
    f.install_old("0.0.1");
    let nowhere = f.state.join("no-release");
    fs::create_dir_all(&nowhere).unwrap();
    let (code, out, err) = run(f
        .basert()
        .env("FAKE_SUPERFLUID_UNREADY", "an engine of another minor")
        .env("FAKE_SUPERFLUID_UNREADY_VERSION", "0.0.1")
        .env("BASERT_SUPERFLUID_SOURCE", &nowhere)
        .args(["serve", f.model()]));
    assert_eq!(code, 1, "{out}{err}");
    assert!(
        err.contains("cannot serve this engine (an engine of another minor)"),
        "{err}"
    );
    assert!(err.contains("copying") && err.contains(".tar.gz"), "{err}");
    assert!(!out.contains("ARGS:"), "{out}");
    assert_eq!(f.stamp(), "v0.0.1", "the copy that was there is kept");
}

#[test]
fn serve_help_version_and_a_missing_model() {
    let f = fixture();
    let (code, out, _) = run(f.basert().args(["serve", "--help"]));
    assert_eq!(code, 0);
    assert!(out.starts_with("Usage: basert serve"), "{out}");
    assert!(out.contains("basert superfluid install"), "{out}");
    assert!(out.contains("BASERT_SUPERFLUID_AUTO_INSTALL"), "{out}");

    let (code, out, _) = run(f.basert().arg("serve"));
    assert_eq!(code, 1);
    assert!(out.contains("Usage: basert serve"), "{out}");

    let (code, out, _) = run(f.basert().args(["serve", "--version"]));
    assert_eq!(code, 0);
    assert!(out.starts_with("basert serve "), "{out}");
    assert!(out.contains("superfluid"), "{out}");

    let (code, _, err) = run(f
        .basert()
        .args(["serve", "m.base", "--runtime", "llamacpp"]));
    assert_eq!(code, 1);
    assert!(err.contains("--runtime is not accepted"), "{err}");

    assert_eq!(run(f.basert().args(["superfluid", "install"])).0, 0);
    let missing = f.state.join("nowhere.base");
    let (code, _, err) = run(f.basert().args(["serve", missing.to_str().unwrap()]));
    assert_eq!(code, 1);
    assert!(err.contains("Failed to load model"), "{err}");
}

/// A header-only bundle (the reader never reaches a blob) of `arch`, with no
/// speculator head.
fn headless_bundle(path: &Path, arch: &str) {
    use base_format::{
        AlignmentConfig, Header, HeaderFlags, ModelConfig, QuantScheme, SourceInfo, TargetBackend,
        TokenizerBlob, FORMAT_VERSION, MAGIC,
    };
    let h = Header {
        schema: 1,
        arch: arch.to_string(),
        quant_scheme: QuantScheme::F16,
        min_hw: "apple_m1".to_string(),
        created: "2026-11-16T00:00:00Z".to_string(),
        base_rt_version: "0.1.0-test".to_string(),
        source: SourceInfo {
            format: "test".to_string(),
            sha256: "0".repeat(64),
            filename: "synthetic".to_string(),
        },
        tokenizer: TokenizerBlob {
            fields: Default::default(),
        },
        config: ModelConfig {
            fields: [("hidden_size".to_string(), serde_json::json!(8))]
                .into_iter()
                .collect(),
        },
        metadata: Default::default(),
        target_backend: TargetBackend::Metal,
        quant_profile: String::new(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::empty(),
        layers: vec![],
        tensors: vec![],
        mmproj: None,
        speculator: None,
        calibration: None,
        provenance: None,
        sig: None,
    };
    let json = h.to_canonical_json().unwrap();
    let mut bytes = MAGIC.to_vec();
    bytes.extend(FORMAT_VERSION.to_le_bytes());
    bytes.extend((json.len() as u64).to_le_bytes());
    bytes.extend(json);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

/// The default `auto` pulls a catalog model again when its installed file
/// has no MTP head and the catalog's current file has one, as `--speculate
/// auto` does — but never under `--offline`: the start serves the installed
/// file. Through the whole `basert serve <id>` path, with the Hub at a dead
/// address, so a pull is seen to start and fail, never to download.
#[test]
fn the_default_auto_pulls_a_model_again_for_its_head_except_offline() {
    let f = fixture();
    let cache = f.tmp.path().join("models-cache");
    let id = "basecompute/gemma-4-E4B-it";
    let vdir = cache.join(id).join("default-q8");
    let installed = vdir.join("model.base");
    headless_bundle(&installed, "gemma4");
    fs::write(
        vdir.join("hub.json"),
        serde_json::json!({
            "id": id, "source_kind": "catalog", "hf_repo": id, "revision": "main",
            "variant": "default-q8", "pulled_at": "0", "base_sha256": "b".repeat(64),
            "file": "gemma-4-E4B-it-Q8.base"
        })
        .to_string(),
    )
    .unwrap();
    // The catalog (read as cached: BASERT_CATALOG_OFFLINE) publishes another
    // file of this variant, with Google's head in it.
    fs::write(
        cache.join(".catalog-cache.json"),
        serde_json::json!({"schema": 1, "models": [{
            "id": id, "hf_repo": id, "file": "gemma-4-E4B-it-Q8.base", "quant": "default-q8",
            "sha256": "a".repeat(64), "speculator": "gemma4-mtp"
        }]})
        .to_string(),
    )
    .unwrap();
    let start = |extra: &[&str]| {
        run(f
            .basert()
            .env("HF_ENDPOINT", "http://127.0.0.1:9")
            .args(["serve", &format!("{id}:default-q8"), "--print-command"])
            .args(extra))
    };
    let pulled_again = "pulling it again for --speculate auto";

    // The default: pulled again, like a hand-given `auto` (the pull fails here).
    let (code, out, err) = start(&[]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(err.contains(pulled_again), "{err}");
    let (code, _, err) = start(&["--speculate", "auto"]);
    assert_eq!(code, 1);
    assert!(err.contains(pulled_again), "{err}");

    // --offline: nothing is pulled, the installed file is served under `auto`.
    let (code, out, err) = start(&["--offline"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("pulling"), "{err}");
    assert!(out.contains(&installed.display().to_string()), "{out}");
    assert!(out.trim_end().ends_with("--speculate auto"), "{out}");

    // --speculate off: nothing asked of the head, nothing pulled.
    let (code, out, err) = start(&["--speculate", "off"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!err.contains("pulling"), "{err}");
}

/// An HTTP responder on 127.0.0.1 that answers every request 404 and counts
/// them; returns its base URL and the count.
fn count_requests() -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use std::io::{BufRead, BufReader, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let hits = std::sync::Arc::new(AtomicUsize::new(0));
    let seen = std::sync::Arc::clone(&hits);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while reader.read_line(&mut line).is_ok_and(|n| n > 0) && line != "\r\n" {
                line.clear();
            }
            seen.fetch_add(1, Ordering::SeqCst);
            let _ = write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
        }
    });
    (base, hits)
}

/// `--offline` keeps the default `auto` off the network: a start given a
/// model by path reads the catalog only as cached, never from the Hub. Run
/// without the fixture's `BASERT_CATALOG_OFFLINE`, the catalog URL at a local
/// responder that counts what reaches it, the model cache empty (no cached
/// catalog, so an online read would go out).
#[test]
fn the_default_auto_reads_no_catalog_online_under_offline() {
    use std::sync::atomic::Ordering;
    let f = fixture();
    assert_eq!(run(f.basert().args(["superfluid", "install"])).0, 0);
    let (url, hits) = count_requests();
    let start = |extra: &[&str]| {
        run(f
            .basert()
            .env_remove("BASERT_CATALOG_OFFLINE")
            .env("BASERT_CATALOG_URL", format!("{url}/catalog.json"))
            .args(["serve", f.model(), "--print-command"])
            .args(extra))
    };

    // Online, the default `auto` reads the catalog (answered 404 here; the
    // start goes on with the bundled one): the responder sees it.
    let (code, out, err) = start(&[]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(hits.load(Ordering::SeqCst) >= 1, "the online read is seen");

    for extra in [
        &["--offline"][..],
        &["--speculate", "auto", "--offline"],
        &["--speculate", "off"],
    ] {
        let before = hits.load(Ordering::SeqCst);
        let (code, out, err) = start(extra);
        assert_eq!(code, 0, "{extra:?}: {out}{err}");
        assert_eq!(
            hits.load(Ordering::SeqCst),
            before,
            "{extra:?} read the catalog online"
        );
    }
    let (_, out, _) = start(&["--offline"]);
    assert!(
        out.trim_end().ends_with("--speculate auto"),
        "still the default: {out}"
    );
}

/// `basert serve <hub id> --offline` resolves the id from what is installed
/// and the catalog as cached, never from the Hub: an installed model starts
/// with no request, and one that is not installed is refused with the pull
/// to run instead of being downloaded. Without the fixture's
/// `BASERT_CATALOG_OFFLINE`, the catalog URL and the Hub at a local responder
/// that counts requests.
#[test]
fn a_hub_id_start_under_offline_reads_no_catalog_and_pulls_nothing() {
    use std::sync::atomic::Ordering;
    let f = fixture();
    assert_eq!(run(f.basert().args(["superfluid", "install"])).0, 0);
    let id = "basecompute/gemma-4-E4B-it";
    let installed = f
        .tmp
        .path()
        .join("models-cache")
        .join(id)
        .join("default-q8")
        .join("model.base");
    headless_bundle(&installed, "gemma4");
    let (url, hits) = count_requests();
    let start = |model: &str, extra: &[&str]| {
        run(f
            .basert()
            .env_remove("BASERT_CATALOG_OFFLINE")
            .env("BASERT_CATALOG_URL", format!("{url}/catalog.json"))
            .env("HF_ENDPOINT", &url)
            .args(["serve", model, "--print-command"])
            .args(extra))
    };

    // Online, resolving the id reads the catalog: the responder sees it.
    let (code, out, err) = start(&format!("{id}:default-q8"), &["--speculate", "off"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(hits.load(Ordering::SeqCst) >= 1, "the online read is seen");

    // --offline, installed: the installed file, no request.
    for extra in [&["--offline"][..], &["--offline", "--speculate", "off"]] {
        let before = hits.load(Ordering::SeqCst);
        let (code, out, err) = start(&format!("{id}:default-q8"), extra);
        assert_eq!(code, 0, "{extra:?}: {out}{err}");
        assert!(
            out.contains(&installed.display().to_string()),
            "{extra:?}: {out}"
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            before,
            "{extra:?} reached the network"
        );
    }

    // --offline, not installed: refused with the pull to run, no request.
    let before = hits.load(Ordering::SeqCst);
    let (code, out, err) = start("basecompute/Qwen3-4B:default-q4", &["--offline"]);
    assert_ne!(code, 0, "{out}{err}");
    assert!(
        err.contains("--offline") && err.contains("basert pull basecompute/Qwen3-4B:default-q4"),
        "{err}"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        before,
        "a model not installed reached the network"
    );
}
