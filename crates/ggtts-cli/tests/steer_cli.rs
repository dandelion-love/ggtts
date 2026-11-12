//! `ggtts-render --steer` on a real talker: steering the talker cannot take (a vector of the wrong length, a layer
//! out of range) is refused once the talker is loaded, and a scale that overflows f32 before it is loaded, each with
//! a message naming the problem, and nothing is rendered.
//!
//! Needs the tiny fixture (`scripts/tiny_fixture.sh`): `GGTTS_TINY_TALKER` and `GGTTS_CODEC`.
//!
//! ```text
//! eval "$(scripts/tiny_fixture.sh)"
//! cargo test -p ggtts-cli --features ggml --test steer_cli -- --ignored --nocapture
//! ```
#![cfg(feature = "ggml")]

use std::path::PathBuf;
use std::process::Command;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name} (see scripts/tiny_fixture.sh)")))
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn steering_the_talker_cannot_take_is_refused_and_nothing_is_rendered() {
    let (talker, codec) = (env_path("GGTTS_TINY_TALKER"), env_path("GGTTS_CODEC"));
    let dir = std::env::temp_dir().join(format!("ggtts-steer-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // (case, tensor name, vector length, --steer-scale, what the refusal says)
    let cases = [
        (
            "wrong-length",
            "talker.layer.0",
            2047,
            "1",
            "has 2047 values; the talker's hidden size is 2048",
        ),
        (
            "out-of-range",
            "talker.layer.1",
            2048,
            "1",
            "steering layer 1 is out of range: the talker has layers 0 to 0",
        ),
        // A finite f64 that is infinite in f32: refused before the talker is even loaded.
        (
            "scale-overflows-f32",
            "talker.layer.0",
            2048,
            "1e39",
            "not a finite number in f32",
        ),
    ];
    for (tag, name, len, scale, want) in cases {
        let file = dir.join(format!("{tag}.safetensors"));
        let v = vec![0.5f32; len];
        std::fs::write(&file, ggtts_core::steer::write_safetensors(&[(name, &v)])).unwrap();
        let out = dir.join(format!("{tag}.flac"));
        let o = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
            .arg("--model")
            .arg(&talker)
            .arg("--codec")
            .arg(&codec)
            .args(["--text", "你好。", "--out"])
            .arg(&out)
            .arg("--steer")
            .arg(&file)
            .args(["--steer-scale", scale])
            .output()
            .unwrap();
        let e = String::from_utf8_lossy(&o.stderr);
        println!("{tag}: {}", e.trim());
        assert!(!o.status.success(), "{tag}: accepted");
        assert!(e.contains(want), "{tag}: {e}");
        assert!(!out.exists(), "{tag}: a refused render wrote {}", out.display());
    }
    std::fs::remove_dir_all(&dir).ok();
}
