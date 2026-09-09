//! End-to-end check of `basert list`: generate a synthetic `.base` into the
//! cache layout via the real binary, then confirm `list` discovers it.
//! Also pins the binary name to `basert` (CARGO_BIN_EXE_basert).

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_basert")
}

#[test]
fn list_discovers_synthetic_model() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let vdir = root.join("basecompute").join("demo").join("default-q4");
    std::fs::create_dir_all(&vdir).unwrap();
    let artifact = vdir.join("model.base");

    // Synthetic convert writes a real .base header (arch = "synthetic").
    let status = Command::new(bin())
        .args(["convert", "--synthetic", "demo", "-o"])
        .arg(&artifact)
        .status()
        .expect("run basert convert");
    assert!(status.success(), "synthetic convert failed");
    assert!(artifact.exists());

    // JSON form: exactly one installed entry with the derived id/variant.
    let out = Command::new(bin())
        .args(["list", "--json"])
        .env("BASERT_MODELS_DIR", root)
        .output()
        .expect("run basert list --json");
    assert!(out.status.success());
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("list --json should emit JSON");
    let arr = v.as_array().expect("array");
    assert_eq!(arr.len(), 1, "got {arr:?}");
    assert_eq!(arr[0]["id"], "basecompute/demo");
    assert_eq!(arr[0]["variant"], "default-q4");
    assert_eq!(arr[0]["installed"], true);
    assert_eq!(arr[0]["arch"], "synthetic");

    // Table form mentions the id and an installed status.
    let out2 = Command::new(bin())
        .args(["list"])
        .env("BASERT_MODELS_DIR", root)
        .output()
        .expect("run basert list");
    let table = String::from_utf8(out2.stdout).unwrap();
    assert!(table.contains("basecompute/demo"), "table: {table}");
    assert!(table.contains("installed"), "table: {table}");
}

#[cfg(unix)]
#[test]
fn computearena_dispatches_to_standalone_cli_with_basert_adapter() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let executable = tmp.path().join("computearena");
    std::fs::write(&executable, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n").unwrap();
    let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&executable, permissions).unwrap();

    let output = Command::new(bin())
        .args(["computearena", "--harness", "/tmp/harness", "run"])
        .env("PATH", tmp.path())
        .output()
        .expect("run basert computearena");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "basert\n--harness\n/tmp/harness\nrun\n"
    );
}

#[cfg(unix)]
#[test]
fn computearena_dispatch_exposes_a_bundled_harness_without_overriding_the_user() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let launcher = tmp.path().join("basert");
    std::fs::copy(bin(), &launcher).unwrap();

    let computearena = tmp.path().join("computearena");
    std::fs::write(
        &computearena,
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\nprintf 'harness=%s\\n' \"$COMPUTEARENA_BASERT_HARNESS\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&computearena).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&computearena, permissions).unwrap();

    let harness = tmp.path().join("basert-benchmark-harness");
    std::fs::write(&harness, b"fixture").unwrap();

    let output = Command::new(&launcher)
        .args(["computearena", "list"])
        .env("PATH", tmp.path())
        .env_remove("COMPUTEARENA_BASERT_HARNESS")
        .env_remove("BASERT_COMPUTEARENA_HARNESS")
        .output()
        .expect("run bundled basert computearena");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("basert\nlist\nharness={}\n", harness.display())
    );

    let output = Command::new(&launcher)
        .args(["computearena", "list"])
        .env("PATH", tmp.path())
        .env("COMPUTEARENA_BASERT_HARNESS", "/user/selected/harness")
        .output()
        .expect("run basert computearena with an explicit harness environment");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "basert\nlist\nharness=/user/selected/harness\n"
    );
}

#[cfg(unix)]
#[test]
fn computearena_dispatch_exposes_the_source_build_harness() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path().join("baseRT");
    let launcher_dir = repo.join("tools/base-convert/target/release");
    std::fs::create_dir_all(&launcher_dir).unwrap();
    std::fs::write(repo.join("tools/base-convert/Cargo.toml"), b"[workspace]\n").unwrap();

    let launcher = launcher_dir.join("basert");
    std::fs::copy(bin(), &launcher).unwrap();

    let cli_dir = tmp.path().join("cli");
    std::fs::create_dir_all(&cli_dir).unwrap();
    let computearena = cli_dir.join("computearena");
    std::fs::write(
        &computearena,
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\nprintf 'harness=%s\\n' \"$COMPUTEARENA_BASERT_HARNESS\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&computearena).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&computearena, permissions).unwrap();

    let harness = repo.join("build/basert-benchmark-harness");
    std::fs::create_dir_all(harness.parent().unwrap()).unwrap();
    std::fs::write(&harness, b"fixture").unwrap();

    let output = Command::new(&launcher)
        .args(["computearena", "list"])
        .env("PATH", &cli_dir)
        .env_remove("COMPUTEARENA_BASERT_HARNESS")
        .env_remove("BASERT_COMPUTEARENA_HARNESS")
        .output()
        .expect("run source-built basert computearena");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("basert\nlist\nharness={}\n", harness.display())
    );
}

#[test]
fn missing_computearena_points_to_the_public_quickstart() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(bin())
        .arg("computearena")
        .env("PATH", tmp.path())
        .output()
        .expect("run basert without computearena installed");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("ComputeArena is not installed."),
        "{stderr}"
    );
    assert!(
        stderr.contains("https://computearena.ai/quickstart"),
        "{stderr}"
    );
    assert!(stderr.contains("`basert computearena` again"), "{stderr}");
    assert!(!stderr.contains("--harness"), "{stderr}");
}
