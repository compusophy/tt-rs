// On Windows + CUDA 11.x, the NVRTC DLL ships under its ABI name (e.g.
// `nvrtc64_112_0.dll` in CUDA Toolkit 11.7), but cudarc 0.13 searches for
// names derived from the toolkit version (`nvrtc64_117_0.dll`). We bridge
// that by copying the on-disk NVRTC DLL into the target deps dir under the
// expected name, so `cargo test`/`cargo run` just work.
//
// On Linux/macOS the dynamic loader naming matches and this script does
// nothing.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" { return; }

    // Locate the CUDA toolkit.
    let cuda_root = match env::var("CUDA_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            // Look for any installed version under the default path.
            let default = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
            if let Ok(entries) = std::fs::read_dir(&default) {
                let mut found: Option<PathBuf> = None;
                for e in entries.flatten() {
                    let name = e.file_name();
                    let n = name.to_string_lossy();
                    if n.starts_with('v') && e.path().is_dir() {
                        found = Some(e.path());
                    }
                }
                match found {
                    Some(p) => p,
                    None => {
                        println!("cargo:warning=CUDA toolkit not found; smoke test won't run");
                        return;
                    }
                }
            } else {
                println!("cargo:warning=CUDA toolkit not found; smoke test won't run");
                return;
            }
        }
    };

    let bin = cuda_root.join("bin");
    if !bin.exists() { return; }

    println!("cargo:rustc-link-search=native={}", bin.display());

    // Find the NVRTC DLL on disk (matches `nvrtc64_*_*.dll`, excluding `builtins`).
    let actual_nvrtc = match std::fs::read_dir(&bin) {
        Ok(entries) => entries.flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .find(|n| n.starts_with("nvrtc64_")
                   && n.ends_with(".dll")
                   && !n.contains("builtins")),
        Err(_) => None,
    };
    let Some(actual) = actual_nvrtc else {
        println!("cargo:warning=no nvrtc64_*.dll found in {}", bin.display());
        return;
    };
    let actual_path = bin.join(&actual);

    // The names cudarc 0.13 + cuda-11070 feature will accept.
    let expected_names = [
        "nvrtc64_117_0.dll",
        "nvrtc64_117.dll",
        "nvrtc64_11.dll",
        "nvrtc.dll",
    ];

    // Resolve `target/{profile}/deps`. OUT_DIR looks like
    // `…/target/{profile}/build/<pkg>-<hash>/out` — walk up 3 levels.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        let deps_dir = profile_dir.join("deps");
        if let Err(e) = std::fs::create_dir_all(&deps_dir) {
            println!("cargo:warning=could not create {}: {}", deps_dir.display(), e);
        }
        // Place the NVRTC DLL (and its companion builtins DLL) under all the
        // names cudarc will look for. Copy not symlink — Windows symlinks
        // need admin.
        for name in &expected_names {
            let dst = deps_dir.join(name);
            if !dst.exists() {
                let _ = std::fs::copy(&actual_path, &dst);
            }
        }
        // Builtins DLL is loaded transparently from the same dir.
        if let Ok(entries) = std::fs::read_dir(&bin) {
            for e in entries.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("nvrtc-builtins64_") && n.ends_with(".dll") {
                    let dst = deps_dir.join(&n);
                    if !dst.exists() {
                        let _ = std::fs::copy(e.path(), dst);
                    }
                }
            }
        }
        println!("cargo:warning=NVRTC ({}) staged into {}", actual, deps_dir.display());
    }
}
