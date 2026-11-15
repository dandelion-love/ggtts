//! `ggtts-render --steer URL`: offline, a URL not in the cache is refused before any model is read, naming the URL,
//! and nothing is written. The download itself is `ggtts_models::Cache::fetch_url`'s, tested there.

use std::process::Command;

#[test]
fn offline_an_uncached_steering_url_is_refused_naming_it() {
    let dir = std::env::temp_dir().join(format!("ggtts-steer-url-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let url = "https://huggingface.co/someone/presets/resolve/main/presets/angry.safetensors";
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "qwen3-tts", "--offline", "--log-level", "error"])
        .args(["--steer", url, "--text", "你好。", "--out", "o.flac"])
        .env("GGTTS_CACHE_DIR", dir.join("cache"))
        .current_dir(&dir)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "rendered with an uncached URL offline: {stderr}");
    assert!(
        stderr.contains("--steer") && stderr.contains(url) && stderr.contains("offline mode does not download it"),
        "{stderr}"
    );
    assert!(!dir.join("o.flac").exists(), "something was rendered");
    let _ = std::fs::remove_dir_all(dir);
}
