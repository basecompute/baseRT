//! Model removal through the real CLI, using only isolated caches and no network.

use base_hub::cache::{self, HubSidecar};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_basert"))
        .args(args)
        .env("BASERT_MODELS_DIR", root)
        .env("BASERT_CATALOG_OFFLINE", "1")
        .output()
        .expect("run basert")
}

fn install(root: &Path, id: &str, variant: &str) -> PathBuf {
    let dir = cache::variant_dir(root, id, variant).unwrap();
    cache::write_sidecar(
        &dir,
        &HubSidecar {
            id: id.into(),
            source_kind: "huggingface".into(),
            hf_repo: id.into(),
            source_repo: None,
            revision: "main".into(),
            variant: variant.into(),
            profile: None,
            pulled_at: "2026-06-24T00:00:00Z".into(),
            base_sha256: None,
        },
    )
    .unwrap();
    // LocalRegistry recognizes even an unreadable header as installed.
    std::fs::write(cache::base_artifact_path(&dir), b"model payload").unwrap();
    dir
}

fn assert_error(out: &Output, message: &str) {
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains(message),
        "{out:?}"
    );
    assert!(!String::from_utf8_lossy(&out.stderr).contains("Removed"));
}

#[test]
fn rm_removes_all_variants_in_custom_cache_and_preserves_other_models_and_sources() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("custom-models");
    let id = "Qwen/Qwen3-4B";
    install(&root, id, "default-q4");
    install(&root, id, "default-q8");
    let others = [
        install(&root, "Qwen/Other", "default-q4"),
        install(&root, "meta-llama/Llama-3.2-1B", "default-q4"),
    ];
    let before: Vec<_> = others
        .iter()
        .map(|dir| std::fs::read(dir.join(cache::SIDECAR_NAME)).unwrap())
        .collect();
    let staging = cache::hf_staging_dir(&root).join("models--Qwen--Qwen3-4B/blobs");
    std::fs::create_dir_all(&staging).unwrap();
    std::fs::write(staging.join("source"), b"retained source").unwrap();

    let out = run(&root, &["rm", id]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        "Removed Qwen/Qwen3-4B\n"
    );
    assert!(!root.join(id).exists());
    assert!(root.is_dir());
    for (dir, sidecar) in others.iter().zip(before) {
        assert_eq!(
            std::fs::read(cache::base_artifact_path(dir)).unwrap(),
            b"model payload"
        );
        assert_eq!(
            std::fs::read(dir.join(cache::SIDECAR_NAME)).unwrap(),
            sidecar
        );
    }
    assert_eq!(
        std::fs::read(staging.join("source")).unwrap(),
        b"retained source"
    );

    let listed = run(&root, &["list", "--json"]);
    assert!(listed.status.success(), "{listed:?}");
    let rows: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    let ids: Vec<_> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["Qwen/Other", "meta-llama/Llama-3.2-1B"]);
}

#[test]
fn rm_missing_model_and_uninstalled_directories_leave_data_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("models");
    assert_error(&run(&root, &["rm", "Qwen/DoesNotExist"]), "not installed");
    assert!(!root.exists(), "removal must not create the cache");
    let dir = install(&root, "Qwen/Qwen3-4B", "default-q4");
    let note = root.join("Qwen/DoesNotExist/notes.txt");
    std::fs::create_dir_all(note.parent().unwrap()).unwrap();
    std::fs::write(&note, b"user data").unwrap();
    for id in ["Qwen/DoesNotExist", "Qwen", "Qwen/Qwen3-4B/default-q4"] {
        assert_error(&run(&root, &["rm", id]), "not installed");
    }
    assert_eq!(std::fs::read(note).unwrap(), b"user data");
    assert_eq!(
        std::fs::read(cache::base_artifact_path(&dir)).unwrap(),
        b"model payload"
    );
    assert!(cache::read_sidecar(&dir).unwrap().is_some());
}

#[test]
fn rm_preserves_unrelated_files_and_nested_models() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let dir = install(root, "Qwen/Qwen3-4B", "default-q4");
    let nested = install(root, "Qwen/Qwen3-4B/nested", "default-q8");
    std::fs::write(dir.join("notes.txt"), b"user notes").unwrap();
    let out = run(root, &["rm", "Qwen/Qwen3-4B"]);
    assert!(out.status.success(), "{out:?}");
    assert!(!cache::base_artifact_path(&dir).exists());
    assert!(!dir.join(cache::SIDECAR_NAME).exists());
    assert_eq!(std::fs::read(dir.join("notes.txt")).unwrap(), b"user notes");
    assert_eq!(
        std::fs::read(cache::base_artifact_path(&nested)).unwrap(),
        b"model payload"
    );
    assert!(cache::read_sidecar(&nested).unwrap().is_some());
    let out = run(root, &["rm", "Qwen/Qwen3-4B/nested"]);
    assert!(out.status.success(), "{out:?}");
    assert!(!nested.parent().unwrap().exists());
}

#[test]
fn rm_rejects_unsafe_ids_without_deleting_anything() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("models");
    let dir = install(&root, "Qwen/Qwen3-4B", "default-q4");
    let outside = install(tmp.path(), "outside/model", "default-q4");
    let staging = install(&root, ".src/hf", "snapshot");
    for id in [
        "",
        " ",
        "../",
        "../../something",
        "/",
        "/absolute/path",
        "../outside/model",
        "Qwen/../../outside/model",
        "Qwen/..",
        "Qwen/./Qwen3-4B",
        "Qwen//Qwen3-4B",
        "Qwen/Qwen3-4B/",
        "\\Qwen\\Qwen3-4B",
        "C:/models",
        "Qwen/Qwen3-4B:default-q4",
        ".src",
        ".src/hf",
        "Qwen/\nQwen3-4B",
        outside.parent().unwrap().to_str().unwrap(),
    ] {
        let out = run(&root, &["rm", id]);
        assert_error(&out, "model id");
    }
    for dir in [dir, outside, staging] {
        assert_eq!(
            std::fs::read(cache::base_artifact_path(&dir)).unwrap(),
            b"model payload"
        );
        assert!(cache::read_sidecar(&dir).unwrap().is_some());
    }
}

#[cfg(unix)]
#[test]
fn rm_never_follows_model_variant_or_artifact_symlinks() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("models");
    let outside = install(tmp.path(), "outside/model", "default-q4");
    let inside = install(&root, "Qwen/Other", "default-q4");
    symlink(tmp.path().join("outside"), root.join("External")).unwrap();
    symlink(inside.parent().unwrap(), root.join("Qwen/Alias")).unwrap();
    symlink(outside.parent().unwrap(), root.join("Qwen/External")).unwrap();
    for id in ["External/model", "Qwen/Alias", "Qwen/External"] {
        assert_error(&run(&root, &["rm", id]), "without symlinks");
    }
    let dir = root.join("Qwen/Linked");
    std::fs::create_dir_all(dir.join("artifact-link")).unwrap();
    symlink(&outside, dir.join("variant-link")).unwrap();
    symlink(
        cache::base_artifact_path(&outside),
        dir.join("artifact-link/model.base"),
    )
    .unwrap();
    assert_error(&run(&root, &["rm", "Qwen/Linked"]), "not installed");
    for dir in [inside, outside] {
        assert_eq!(
            std::fs::read(cache::base_artifact_path(&dir)).unwrap(),
            b"model payload"
        );
        assert!(cache::read_sidecar(&dir).unwrap().is_some());
    }
    assert!(dir.join("variant-link").is_symlink());
    assert!(dir.join("artifact-link/model.base").is_symlink());
}

#[test]
fn rm_handles_missing_sidecars_and_checks_sidecar_types_before_deletion() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let q4 = install(root, "Qwen/Qwen3-4B", "default-q4");
    let q8 = install(root, "Qwen/Qwen3-4B", "default-q8");
    std::fs::remove_file(q4.join(cache::SIDECAR_NAME)).unwrap();
    std::fs::remove_file(q8.join(cache::SIDECAR_NAME)).unwrap();
    // A directory at hub.json is not model metadata; preserve it and both
    // artifacts even if the other variant was discovered first.
    std::fs::create_dir(q8.join(cache::SIDECAR_NAME)).unwrap();
    assert_error(
        &run(root, &["rm", "Qwen/Qwen3-4B"]),
        "expected a sidecar file",
    );
    for dir in [&q4, &q8] {
        assert_eq!(
            std::fs::read(cache::base_artifact_path(dir)).unwrap(),
            b"model payload"
        );
    }
    std::fs::remove_dir(q8.join(cache::SIDECAR_NAME)).unwrap();
    let out = run(root, &["rm", "Qwen/Qwen3-4B"]);
    assert!(out.status.success(), "{out:?}");
    assert!(!root.join("Qwen/Qwen3-4B").exists());
}

#[cfg(unix)]
#[test]
fn rm_unlinks_sidecar_symlink_without_deleting_its_target() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("models");
    let dir = install(&root, "Qwen/Qwen3-4B", "default-q4");
    let outside = tmp.path().join("user-data");
    std::fs::write(&outside, b"user data").unwrap();
    std::fs::remove_file(dir.join(cache::SIDECAR_NAME)).unwrap();
    std::os::unix::fs::symlink(&outside, dir.join(cache::SIDECAR_NAME)).unwrap();
    let out = run(&root, &["rm", "Qwen/Qwen3-4B"]);
    assert!(out.status.success(), "{out:?}");
    assert!(!root.join("Qwen/Qwen3-4B").exists());
    assert_eq!(std::fs::read(outside).unwrap(), b"user data");
}

#[test]
fn rm_help_and_required_argument() {
    let tmp = tempfile::tempdir().unwrap();
    for args in [&["--help"][..], &["rm", "--help"][..]] {
        let out = run(tmp.path(), args);
        assert!(out.status.success(), "{out:?}");
        assert!(String::from_utf8_lossy(&out.stdout).contains("rm"));
        if args.len() == 2 {
            assert!(String::from_utf8_lossy(&out.stdout).contains("rm <MODEL>"));
        }
    }
    let out = run(tmp.path(), &["rm"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("<MODEL>"));
}
