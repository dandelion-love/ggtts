//! Builds ggml from the submodule with CMake (static, CPU only, ggml's own defaults otherwise) and generates the
//! bindings with bindgen. The CMake options are the ones the reference harness builds with, so both sides run the
//! same kernels: Release, static, GGML_NATIVE and GGML_OPENMP at ggml's defaults.
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = env::var_os("GGML_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("ggml"));
    if !src.join("include/ggml.h").exists() {
        panic!(
            "ggml source not found at {}; run `git submodule update --init spike/layer0/ggml-sys/ggml` or set GGML_SRC",
            src.display()
        );
    }
    println!("cargo:rerun-if-env-changed=GGML_SRC");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");

    // GGML_LIB_DIR: link prebuilt static libraries (libggml.a, libggml-cpu.a, libggml-base.a) instead of building
    // them. The parity check sets it to the reference harness's own ggml build, so both sides link the same
    // archives and a difference cannot come from build flags.
    if let Some(dir) = env::var_os("GGML_LIB_DIR") {
        let dir = PathBuf::from(dir);
        for lib in ["libggml.a", "libggml-cpu.a", "libggml-base.a"] {
            println!("cargo:rerun-if-changed={}", dir.join(lib).display());
        }
        println!("cargo:rustc-link-search=native={}", dir.display());
    } else {
        build_ggml(&src);
    }
    link_and_bind(&manifest, &src);
}

fn build_ggml(src: &std::path::Path) {
    let dst = cmake::Config::new(src)
        .profile("Release")
        // Leave out the cc crate's flags (-O level, -fPIC, sections), so ggml compiles with exactly the flags its
        // own CMake gives it, as on the reference side.
        .no_default_flags(true)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        .define("GGML_CUDA", "OFF")
        // Rust links position-independent executables; the reference harness is built with the same setting.
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();

    for dir in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={}", dst.join(dir).display());
    }
}

fn link_and_bind(manifest: &std::path::Path, src: &std::path::Path) {
    // Order matters for static linking: ggml (registry) -> ggml-cpu -> ggml-base.
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    let target = env::var("TARGET").unwrap();
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=m");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }

    let bindings = bindgen::Builder::default()
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
        .expect("bindgen failed");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("bindings.rs")).unwrap();
}
