use std::env;
use std::path::PathBuf;

// By default this links the single prebuilt shared library, libbaseRT.dylib,
// which already carries the Metal kernels (embedded metallib) and its own
// framework/C++ dependencies. That keeps the engine a single redistributable
// artifact and matches the Python / Node / Swift bindings.
//
// Opt into linking the static engine archive (libbaseRT_engine.a) plus the
// frameworks manually with `--features static` if you need a fully static link.
fn main() {
    // Search path: BASERT_LIB_DIR (or legacy BASERT_LIB_PATH), else ../../../build.
    let lib_dir = env::var("BASERT_LIB_DIR")
        .or_else(|_| env::var("BASERT_LIB_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest_dir.join("..").join("..").join("..").join("build")
        });

    let lib_dir = lib_dir.canonicalize().unwrap_or_else(|_| {
        panic!(
            "BaseRT library directory not found: {:?}. Set BASERT_LIB_DIR or build with `make shared`.",
            lib_dir
        )
    });

    // Branch on the TARGET os, not the host: in a build script cfg!(target_os)
    // reflects the build script's own (host) target, so read CARGO_CFG_TARGET_OS.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let dylib_file = match target_os.as_str() {
        "macos" => "libbaseRT.dylib",
        "windows" => "baseRT.dll",
        _ => "libbaseRT.so", // linux / other unix
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

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

    println!("cargo:rerun-if-env-changed=BASERT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BASERT_LIB_PATH");
}
