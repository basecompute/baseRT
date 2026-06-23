use std::env;
use std::path::PathBuf;

// baseRT-sys emits the link-search + link-lib (those propagate to us), but
// Cargo does NOT propagate a dependency's `rustc-link-arg`, so the rpath it
// sets never reaches this crate's examples/tests/binaries. Re-emit the rpath
// here so `cargo run --example` / `cargo test` find libbaseRT.dylib at runtime
// without DYLD_LIBRARY_PATH. (Downstream user crates that build executables
// should do the same, or set an rpath / DYLD_LIBRARY_PATH — see README.)
fn main() {
    let lib_dir = env::var("BASERT_LIB_DIR")
        .or_else(|_| env::var("BASERT_LIB_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest_dir.join("..").join("..").join("..").join("build")
        });

    // Windows resolves DLLs via PATH / the exe directory, not rpath.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        if let Ok(dir) = lib_dir.canonicalize() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
    println!("cargo:rerun-if-env-changed=BASERT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BASERT_LIB_PATH");
}
