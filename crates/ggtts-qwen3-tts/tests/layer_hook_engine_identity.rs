#![cfg(feature = "ggml")]

//! The exported layer taps must leave a complete engine render unchanged.

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::hooks::{Hooks, LayerStep, Step, TensorHook, Wants};
use ggtts_core::trace::NullEvents;
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

struct LayerCollector;
impl TensorHook for LayerCollector {
    fn name(&self) -> &str {
        "layer-collector"
    }
    fn wants(&self) -> Wants {
        Wants::NONE
    }
    fn layers(&self) -> Vec<usize> {
        vec![0]
    }
    fn layer(&mut self, _: &LayerStep<'_>) {}
    fn step(&mut self, _: &Step<'_>) {}
}

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn fixture_sha(key: &str) -> String {
    std::fs::read_to_string(here().join("fixtures/tiny/SOURCE"))
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap()
        .trim()
        .to_string()
}
fn pinned(var: &str, key: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} (see scripts/tiny_fixture.sh)")));
    let bytes = std::fs::read(&path).unwrap();
    let hash = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(
        hash,
        fixture_sha(key),
        "{var}={} is not the pinned fixture",
        path.display()
    );
    path
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn layer_capture_preserves_codes_and_decoded_samples() {
    let talker = pinned("GGTTS_TINY_TALKER", "CUT_SHA256");
    let codec = pinned("GGTTS_CODEC", "CODEC_SHA256");
    let voice = Voice {
        max_frames: 8,
        ..Default::default()
    };
    let mut plain = Qwen3Tts::load(&talker, &codec, 4, voice.clone()).unwrap();
    let mut hooked = Qwen3Tts::load(&talker, &codec, 4, voice).unwrap();
    let request = ChunkRequest {
        text: "今日天氣好好。",
        seed: 20260905,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let a = plain.generate(&request).unwrap();
    let audio_a = plain.decode(&a.codes).unwrap();
    let mut hooks = Hooks::default();
    hooks.add(Box::new(LayerCollector));
    let b = hooked.generate_hooked(&request, &mut NullEvents, &mut hooks).unwrap();
    let audio_b = hooked.decode(&b.codes).unwrap();
    assert_eq!(a.codes, b.codes);
    assert_eq!(audio_a.samples, audio_b.samples);
    assert_eq!(audio_a.sample_rate, audio_b.sample_rate);
    assert!(hooks.failures().is_empty());
}
