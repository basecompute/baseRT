use std::env;

// baseRT-sys emits the link-search + link-lib (those propagate to us), but
// Cargo does NOT propagate a dependency's `rustc-link-arg`, so the rpath it
// sets never reaches this crate's examples/tests/binaries. Re-emit the rpath
// here so `cargo run --example` / `cargo test` find libbaseRT.dylib at runtime
// without DYLD_LIBRARY_PATH. (Downstream user crates that build executables
// should do the same, or set an rpath / DYLD_LIBRARY_PATH — see README.)
//
// The directory comes from baseRT-sys, which resolves it (BASERT_LIB_DIR, a
// source checkout's build/, ~/.basert or Homebrew) and passes it on through its
// `links` metadata as DEP_BASERT_LIB_DIR. It is unset on docs.rs, where
// baseRT-sys links nothing.
fn main() {
    println!("cargo:rerun-if-env-changed=DEP_BASERT_LIB_DIR");

    // Windows resolves DLLs via PATH / the exe directory, not rpath.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        if let Ok(dir) = env::var("DEP_BASERT_LIB_DIR") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
