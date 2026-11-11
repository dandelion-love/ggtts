//! A refused Kitten input is reported alone: the usage follows an argument error, not a refused model.
#![cfg(feature = "ggml")]

use std::process::Command;

#[test]
fn a_refused_model_is_reported_without_the_usage() {
    let dir = std::env::temp_dir().join(format!("ggtts-render-kitten-refusal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let not_gguf = dir.join("model.gguf");
    std::fs::write(&not_gguf, b"not a gguf").unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "kitten-tts-2", "--model"])
        .arg(&not_gguf)
        .arg("--codec")
        .arg(&not_gguf)
        .arg("--voice")
        .arg(dir.join("voice.json"))
        .args(["--text", "Hello.", "--out"])
        .arg(dir.join("out.flac"))
        .output()
        .unwrap();
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success());
    assert!(e.contains("(--model): not a GGUF file"), "{e}");
    assert!(!e.contains("usage:"), "the usage buried the refusal: {e}");
    let o = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .arg("--no-such-option")
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&o.stderr).contains("usage:"));
    std::fs::remove_dir_all(&dir).unwrap();
}
