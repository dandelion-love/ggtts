//! Without the `ggml` feature this does nothing: the crate is then an empty shell, so the workspace builds with no
//! submodule and no C toolchain.
//!
//! With `ggml`, it builds ggml from source with CMake (static, CPU only, ggml's own defaults otherwise), links it,
//! and selects the bindings: the committed `src/bindings/x86_64-unknown-linux-gnu.rs`, or, with the `bindgen`
//! feature, bindings generated now from `wrapper.h` (needs libclang).
//!
//! Environment:
//! * `GGML_SRC`: a ggml source tree to build instead of the `ggml` submodule. It must be the pinned commit.
//! * `GGML_LIB_DIR`: link prebuilt static archives (`libggml.a`, `libggml-cpu.a`, `libggml-base.a`) from this
//!   directory instead of building. A parity check sets it to the reference's own ggml build, so both sides link
//!   the same compiled kernels.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "ggml")]
    with_ggml::main();
}

#[cfg(feature = "ggml")]
mod with_ggml {
    use std::env;
    use std::path::{Path, PathBuf};

    /// The target the committed bindings were generated for.
    const COMMITTED_TARGET: &str = "x86_64-unknown-linux-gnu";

    pub fn main() {
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let src = env::var_os("GGML_SRC")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest.join("ggml"));
        println!("cargo:rerun-if-env-changed=GGML_SRC");
        println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");
        println!("cargo:rerun-if-changed=wrapper.h");
        if !src.join("include/ggml.h").exists() {
            panic!(
                "ggml source not found at {}: run `git submodule update --init crates/ggtts-ggml-sys/ggml`, \
                 or set GGML_SRC to a checkout of ggml at the pinned commit",
                src.display()
            );
        }
        println!("cargo:include={}", src.join("include").display());

        if let Some(dir) = env::var_os("GGML_LIB_DIR") {
            let dir = PathBuf::from(dir);
            for lib in [
                "libggml.a",
                "libggml-cpu.a",
                "libggml-base.a",
                "ggml-cuda/libggml-cuda.a",
            ] {
                println!("cargo:rerun-if-changed={}", dir.join(lib).display());
            }
            println!("cargo:rustc-link-search=native={}", dir.display());
            // A CMake build of ggml puts the CUDA archive in a directory of its own.
            println!("cargo:rustc-link-search=native={}", dir.join("ggml-cuda").display());
        } else {
            build_ggml(&src);
        }
        link();
        bindings(&manifest, &src);
    }

    fn build_ggml(src: &Path) {
        let dst = cmake::Config::new(src)
            .profile("Release")
            // Leave out the cc crate's flags (-O level, -fPIC, sections), so ggml compiles with exactly the flags
            // its own CMake gives it, as the reference does.
            .no_default_flags(true)
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("GGML_BUILD_TESTS", "OFF")
            .define("GGML_BUILD_EXAMPLES", "OFF")
            .define("GGML_CUDA", if cfg!(feature = "cuda") { "ON" } else { "OFF" })
            .define(
                "CMAKE_CUDA_ARCHITECTURES",
                env::var("GGML_CUDA_ARCH").unwrap_or_else(|_| "86".into()),
            )
            // Rust links position-independent executables.
            .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
            .build();
        for dir in ["lib", "lib64"] {
            println!("cargo:rustc-link-search=native={}", dst.join(dir).display());
        }
    }

    fn link() {
        // Order matters for static linking: ggml (the backend registry), then the backends, then ggml-base.
        println!("cargo:rustc-link-lib=static=ggml");
        #[cfg(feature = "cuda")]
        {
            // CUDA_HOME (default /usr/local/cuda) for the runtime and cuBLAS; /usr/lib/wsl/lib holds libcuda under
            // WSL.
            let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
            println!("cargo:rerun-if-env-changed=CUDA_HOME");
            println!("cargo:rustc-link-lib=static=ggml-cuda");
            println!("cargo:rustc-link-search=native={cuda}/lib64");
            println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
            println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");
            for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
        let target = env::var("TARGET").unwrap();
        if target.contains("linux") {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=dylib=m");
        } else if target.contains("windows") {
            // ggml-cpu reads the CPU's name from the registry (RegOpenKeyExA and friends).
            println!("cargo:rustc-link-lib=dylib=advapi32");
        } else if target.contains("apple") {
            println!("cargo:rustc-link-lib=dylib=c++");
        }
    }

    #[cfg(not(feature = "bindgen"))]
    fn bindings(manifest: &Path, _src: &Path) {
        let target = env::var("TARGET").unwrap();
        if target != COMMITTED_TARGET {
            panic!(
                "no committed ggml bindings for {target} (only {COMMITTED_TARGET}); build with the `bindgen` \
                 feature, which needs libclang"
            );
        }
        let file = manifest.join("src/bindings").join(format!("{COMMITTED_TARGET}.rs"));
        println!("cargo:rerun-if-changed={}", file.display());
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::copy(&file, out.join("bindings.rs")).expect("copy the committed bindings");
    }

    #[cfg(feature = "bindgen")]
    fn bindings(manifest: &Path, src: &Path) {
        let _ = COMMITTED_TARGET;
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        generate(manifest, src)
            .write_to_file(out.join("bindings.rs"))
            .expect("write the bindings");
    }

    /// The one bindgen configuration. `scripts/regenerate-bindings.sh` writes its output to `src/bindings/`.
    #[cfg(feature = "bindgen")]
    fn generate(manifest: &Path, src: &Path) -> bindgen::Bindings {
        bindgen::Builder::default()
            .header(manifest.join("wrapper.h").to_string_lossy())
            .clang_arg(format!("-I{}", src.join("include").display()))
            .allowlist_function("ggml_.*")
            .allowlist_function("gguf_.*")
            .allowlist_type("ggml_.*")
            .allowlist_type("gguf_.*")
            .allowlist_var("GGML_.*")
            .allowlist_var("GGUF_.*")
            .prepend_enum_name(false)
            .layout_tests(false)
            .generate()
            .expect("bindgen failed")
    }
}
