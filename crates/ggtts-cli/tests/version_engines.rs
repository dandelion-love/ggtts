//! `--version` lists the engines compiled in, and `scripts/buildinfo_engines.sh` refuses a build that lists none or
//! misses one it was asked to carry. The default test build has no `ggml` feature, so it is the planted
//! engine-less build: it must say so and fail the release check. A `ggml` build must list both engines and pass.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_ggtts-render");

fn version() -> String {
    let o = Command::new(BIN).arg("--version").output().unwrap();
    assert!(o.status.success());
    String::from_utf8(o.stdout).unwrap()
}

#[test]
fn version_lists_what_was_compiled_in() {
    let v = version();
    if cfg!(feature = "ggml") {
        assert!(v.contains("engine qwen3-tts backends cpu"), "{v}");
        assert!(v.contains("engine kitten-tts-2 backends cpu"), "{v}");
        assert!(!v.contains("engines: none"), "{v}");
    } else {
        assert!(v.contains("engines: none"), "{v}");
        assert!(!v.contains("engine qwen3-tts"), "{v}");
    }
}

#[cfg(unix)]
fn check(required: &[&str]) -> (i32, String) {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/buildinfo_engines.sh");
    let o = Command::new("bash")
        .arg(script)
        .arg(BIN)
        .args(required)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).trim().to_string(),
    )
}

#[cfg(unix)]
#[test]
fn the_release_check_refuses_a_build_without_engines() {
    let (code, out) = check(&["qwen3-tts", "kitten-tts-2"]);
    if cfg!(feature = "ggml") {
        assert_eq!((code, out.as_str()), (0, r#"["qwen3-tts","kitten-tts-2"]"#));
        let (code, _) = check(&["qwen3-tts", "an-engine-it-does-not-have"]);
        assert_eq!(code, 1, "a missing required engine must fail");
    } else {
        assert_eq!(
            code, 1,
            "the planted build without ggml must fail the release check, printed {out}"
        );
    }
}
