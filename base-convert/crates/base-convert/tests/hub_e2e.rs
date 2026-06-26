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
