use std::env;
use std::path::{Component, PathBuf};

// By default this links the single prebuilt shared library, libbaseRT.dylib,
// which already carries the Metal kernels (embedded metallib) and its own
// framework/C++ dependencies. That keeps the engine a single redistributable
// artifact and matches the Python / Node / Swift bindings.
//
// Opt into linking the static engine archive (libbaseRT_engine.a) plus the
// frameworks manually with `--features static` if you need a fully static link.
//
// `--features no-link` links nothing, for a dependent that loads the library
// itself at run time.
fn main() {
    println!("cargo:rerun-if-env-changed=BASERT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BASERT_LIB_PATH");
    println!("cargo:rerun-if-env-changed=BASERT_INSTALL_DIR");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let checkout = manifest_dir.join("..").join("..").join("..");

    // The engine's C headers, for dependents' build scripts: through `links`
    // this arrives as DEP_BASERT_INCLUDE, holding baseRT/baseRT.h. The crate
    // as published carries them in its own include/; in a source checkout
    // they are the checkout's. Exported before anything can return early, so
    // dependents get it with no-link and on docs.rs too.
    let include = [manifest_dir.join("include"), checkout.join("include")]
        .into_iter()
        .find(|d| d.join("baseRT").join("baseRT.h").is_file())
        .and_then(|d| d.canonicalize().ok())
        .unwrap_or_else(|| manifest_dir.join("include"));
    println!("cargo:include={}", include.display());

    // docs.rs builds without the engine. The crate is only FFI declarations,
    // so documenting it needs nothing to link against.
    if env::var_os("DOCS_RS").is_some() {
        return;
    }
    // The dependent loads the library at run time: nothing to find or link.
    if cfg!(feature = "no-link") {
        return;
    }

    // Branch on the TARGET os, not the host: in a build script cfg!(target_os)
    // reflects the build script's own (host) target, so read CARGO_CFG_TARGET_OS.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let dylib_file = match target_os.as_str() {
        "macos" => "libbaseRT.dylib",
        "windows" => "baseRT.dll",
        _ => "libbaseRT.so", // linux / other unix
    };
    let wanted = if cfg!(feature = "static") {
        "libbaseRT_engine.a"
    } else {
        dylib_file
    };

    // Search path: BASERT_LIB_DIR (or legacy BASERT_LIB_PATH) if set, else the
    // first of these that holds the library: a source checkout's
    // ../../../build, then a prebuilt engine installed by install.sh
    // ($BASERT_INSTALL_DIR, default ~/.basert) or Homebrew. The checkout comes
    // first so a fresh local build is never shadowed by an older release.
    let lib_dir = match env::var("BASERT_LIB_DIR").or_else(|_| env::var("BASERT_LIB_PATH")) {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            let mut candidates = vec![checkout.join("build")];
            match env::var_os("BASERT_INSTALL_DIR") {
                Some(dir) => candidates.push(PathBuf::from(dir)),
                None => {
                    if let Some(home) = env::var_os("HOME") {
                        candidates.push(PathBuf::from(home).join(".basert"));
                    }
                }
            }
            candidates.push(PathBuf::from("/opt/homebrew/opt/basert/libexec"));
            candidates.push(PathBuf::from(
                "/home/linuxbrew/.linuxbrew/opt/basert/libexec",
            ));
            let found = candidates
                .iter()
                .find(|d| d.join(wanted).is_file())
                .cloned()
                .unwrap_or_else(|| {
                    panic!(
                        "BaseRT engine library ({}) not found. Install the engine \
                         (`brew install basecompute/tap/basert` or install.sh), set BASERT_LIB_DIR, \
                         or build with `make shared`. Looked in: {:?}",
                        wanted, candidates
                    )
                });
            // Built inside a source checkout whose build/ has no library, this
            // links an installed engine, which may be an older release than the
            // checkout's headers. Say so rather than do it silently.
            if found != candidates[0] && checkout.join("include/baseRT/baseRT.h").is_file() {
                let build = checkout
                    .canonicalize()
                    .map_or(candidates[0].clone(), |c| c.join("build"));
                println!(
                    "cargo:warning=no {} in {}, so the installed engine in {} is linked, \
                     which may be an older release than this checkout. Build this checkout's \
                     (`make shared`) or set BASERT_LIB_DIR to choose.",
                    wanted,
                    build.display(),
                    found.display()
                );
            }
            found
        }
    };

    // An absolute path is kept as given, not resolved through symlinks: Homebrew's
    // opt/basert/libexec is a stable link to the versioned Cellar folder, and an
    // rpath into that folder would break on the next `brew upgrade`. Only a
    // relative path, or one with `..` (the checkout's ../../../build), is resolved.
    let keep = lib_dir.is_absolute() && !lib_dir.components().any(|c| c == Component::ParentDir);
    let lib_dir = if keep { Ok(lib_dir.clone()) } else { lib_dir.canonicalize() }
        .ok()
        .filter(|d| d.is_dir())
        .unwrap_or_else(|| {
            panic!(
                "BaseRT library directory not found: {:?}. Set BASERT_LIB_DIR or build with `make shared`.",
                lib_dir
            )
        });

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // Hand the resolved directory to crates that depend on this one (via
    // `links = "baseRT"` it arrives as DEP_BASERT_LIB_DIR), so baseRT embeds
    // an rpath to the same place without repeating this search.
    println!("cargo:lib_dir={}", lib_dir.display());

    if cfg!(feature = "static") {
        // Static engine + the platform deps it pulls in.
        println!("cargo:rustc-link-lib=static=baseRT_engine");
        if target_os == "macos" {
            // Apple frameworks the Metal backend needs (mac-only).
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=IOKit");
            println!("cargo:rustc-link-lib=c++");
        }
        // (A future CUDA/ROCm static build links its own runtime libs here.)
        println!(
            "cargo:rerun-if-changed={}",
            lib_dir.join("libbaseRT_engine.a").display()
        );
    } else {
        // Dynamic: one self-contained shared library.
        println!("cargo:rustc-link-lib=dylib=baseRT");
        // Embed an rpath so the built binary / test harness finds it without
        // DYLD_LIBRARY_PATH / LD_LIBRARY_PATH. Windows has no rpath (DLLs are
        // resolved via PATH / the exe directory), so skip it there.
        if target_os != "windows" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        println!(
            "cargo:rerun-if-changed={}",
            lib_dir.join(dylib_file).display()
        );
    }
}
