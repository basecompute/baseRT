//! Inventory test: walks every .gguf in a configurable models directory
//! and converts to base_q4. Designed as a CI canary against regressions
//! in arch mapping or dequant coverage.
//!
//! Default ignored so CI doesn't need the model zoo. Run manually:
//!   BASE_MODELS_DIR=$HOME/Projects/baseRT/models \
//!     cargo test -p base-convert --release --test inventory -- --ignored
//!
//! The test fails if:
//!   - A file in `EXPECTED_SKIP` converts successfully (list is stale)
//!   - A file NOT in `EXPECTED_SKIP` fails to convert (regression)
//!
//! Update `EXPECTED_SKIP` when adding coverage for a new arch/quant.

use std::path::PathBuf;
use std::process::Command;

/// Known-not-yet-supported GGUF variants. Each entry is a substring
/// match against the filename. Maintaining this list documents the
/// coverage frontier.
const EXPECTED_SKIP: &[(&str, &str)] = &[
    ("mmproj", "standalone mmproj bundles handled as sub-bundles, not primary"),
];

fn models_dir() -> Option<PathBuf> {
    // BASE_MODELS_DIR is the explicit override; otherwise look for a
    // `models/` directory at the repo root (relative to this crate).
    std::env::var_os("BASE_MODELS_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            p.pop(); // crates/base-convert -> crates
            p.pop(); // crates -> base-convert (tool dir)
            p.pop(); // base-convert -> tools
            p.pop(); // tools -> repo root
            p.push("models");
            if p.is_dir() {
                Some(p)
            } else {
                None
            }
        })
}

fn base_convert_bin() -> PathBuf {
    let mut target = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    target.pop();
    target.pop();
    target.push("target");
    target.push("release");
    target.push("base-convert");
    if target.exists() {
        return target;
    }
    target.pop();
    target.push("debug");
    target.push("base-convert");
    target
}

fn expected_to_skip(filename: &str) -> Option<&'static str> {
    EXPECTED_SKIP
        .iter()
        .find(|(pat, _)| filename.contains(pat))
        .map(|(_, reason)| *reason)
}

#[test]
#[ignore]
fn converts_every_local_gguf() {
    let Some(dir) = models_dir() else {
        eprintln!("no models dir; set BASE_MODELS_DIR");
        return;
    };
    let bin = base_convert_bin();
    assert!(
        bin.exists(),
        "base-convert binary not found at {:?} — run `cargo build --release` first",
        bin
    );

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out.base");

    let mut unexpected_failures: Vec<String> = Vec::new();
    let mut unexpected_successes: Vec<String> = Vec::new();
    let mut succeeded: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        let fname = path.file_name().unwrap().to_str().unwrap().to_string();

        let result = Command::new(&bin)
            .args([
                "convert",
                path.to_str().unwrap(),
                "--target",
                "base-q4",
                "-o",
                out.to_str().unwrap(),
            ])
            .output()
            .expect("spawn base-convert");

        let expected_skip = expected_to_skip(&fname);
        match (result.status.success(), expected_skip) {
            (true, None) => succeeded.push(fname),
            (true, Some(_reason)) => unexpected_successes.push(fname),
            (false, None) => {
                let stderr = String::from_utf8_lossy(&result.stderr);
                let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
                let mut tail_rev = tail;
                tail_rev.reverse();
                unexpected_failures.push(format!("{}: {}", fname, tail_rev.join(" | ")));
            }
            (false, Some(reason)) => skipped.push(format!("{} ({})", fname, reason)),
        }
    }

    eprintln!("\n=== inventory ===");
    eprintln!("succeeded: {}", succeeded.len());
    for s in &succeeded {
        eprintln!("    {}", s);
    }
    eprintln!("skipped:   {}", skipped.len());
    for s in &skipped {
        eprintln!("    {}", s);
    }

    let mut panics = Vec::new();
    if !unexpected_successes.is_empty() {
        panics.push(format!(
            "\n{} files listed in EXPECTED_SKIP but converted OK (update the list):\n  {}",
            unexpected_successes.len(),
            unexpected_successes.join("\n  ")
        ));
    }
    if !unexpected_failures.is_empty() {
        panics.push(format!(
            "\n{} unexpected failures (arch/dequant regression):\n  {}",
            unexpected_failures.len(),
            unexpected_failures.join("\n  ")
        ));
    }
    if !panics.is_empty() {
        panic!("{}", panics.join("\n"));
    }
}
