//! The no-change check on a real engine, in CI: the tiny fixture (the published v0.0.1 talker cut to one talker layer
//! and one code-predictor layer, `fixtures/tiny/`) with the published codec.
//!
//! **Identity only.** A model cut to one layer cannot speak, so nothing here judges audio. What it checks is that an
//! instrument changes nothing on a real graph at a fixed seed:
//! * a quiet frame hook leaves every codec token as it was, and sees exactly the frames returned;
//! * the stop-probability trace (`Voice::trace`) leaves every token and every decoded sample as it was;
//! * the planted `FrameHookDraw` (a hook that draws once from the sampler's stream) turns the first check red, and
//!   is quiet without a hook, so the check is shown to see what it claims to.
//!
//! Both files are checked against `fixtures/tiny/SOURCE` by sha256 first: a different file fails, never passes.
//! `scripts/tiny_fixture.sh` fetches the published files, cuts the fixture and prints the two variables:
//!
//! ```text
//! eval "$(scripts/tiny_fixture.sh)"
//! cargo test --release -p ggtts-qwen3-tts --features ggml,plant --test tiny_model -- --ignored --nocapture
//! ```
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};
#[cfg(feature = "plant")]
use ggtts_qwen3_tts::talker::{language_id, Plant, Sampling, Talker};
use sha2::{Digest, Sha256};

/// Frames per request: enough for the hook, the trace and the sampler to run many steps, and quick on one layer.
const FRAMES: i32 = 24;
const SEED: u64 = 20260905;

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `KEY=VALUE` lines from `fixtures/tiny/SOURCE`.
fn source(key: &str) -> String {
    let text = std::fs::read_to_string(here().join("fixtures/tiny/SOURCE")).unwrap();
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("fixtures/tiny/SOURCE has no {key}"))
        .trim()
        .to_string()
}

fn sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A file named by `var`, refused unless its sha256 is `want`: a check on the wrong file must fail, not pass.
fn pinned(var: &str, want: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} (see scripts/tiny_fixture.sh)")));
    let got = sha256(&p);
    assert_eq!(got, want, "{var}={} is not the pinned file", p.display());
    p
}

fn talker_path() -> PathBuf {
    pinned("GGTTS_TINY_TALKER", &source("CUT_SHA256"))
}

fn codec_path() -> PathBuf {
    pinned("GGTTS_CODEC", &source("CODEC_SHA256"))
}

/// The CPU fixtures' texts, written for the purpose.
fn texts() -> Vec<(String, String)> {
    std::fs::read_to_string(here().join("fixtures/cpu/texts.tsv"))
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(n, t)| (n.to_string(), t.to_string()))
        .collect()
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn the_trace_changes_no_token_and_no_sample_on_the_tiny_model() {
    let (talker, codec) = (talker_path(), codec_path());
    let scratch = std::env::temp_dir().join(format!("ggtts-tiny-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let voice = |trace: Option<PathBuf>| Voice {
        max_frames: FRAMES,
        trace,
        ..Default::default()
    };
    let mut plain = Qwen3Tts::load(&talker, &codec, 4, voice(None)).unwrap();
    let mut traced = Qwen3Tts::load(&talker, &codec, 4, voice(Some(scratch.join("trace.jsonl")))).unwrap();
    let mut failures = Vec::new();
    for (name, text) in texts() {
        let req = ChunkRequest {
            text: &text,
            seed: SEED,
            chunk: 0,
            draw: 0,
            instruction: None,
        };
        let a = plain.generate(&req).unwrap();
        let pa = plain.decode(&a.codes).unwrap();
        let b = traced.generate(&req).unwrap();
        let pb = traced.decode(&b.codes).unwrap();
        let tokens = a.codes == b.codes;
        let samples = pa.samples == pb.samples && pa.sample_rate == pb.sample_rate;
        println!(
            "{name:<14} {} frames, {} samples: tokens {}, samples {}",
            a.codes.n_frames(),
            pa.samples.len(),
            if tokens { "identical" } else { "DIFFER" },
            if samples { "identical" } else { "DIFFER" },
        );
        if !(tokens && samples && a.codes.n_frames() > 0) {
            failures.push(name);
        }
    }
    let lines = std::fs::read_to_string(scratch.join("trace.jsonl")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&scratch);
    assert!(
        !lines.trim().is_empty(),
        "the trace was not written, so it was not exercised"
    );
    assert!(failures.is_empty(), "the trace changed the output on {failures:?}");
}

/// Generate one text with a frame hook (or none): the codes and the frames the hook saw.
#[cfg(feature = "plant")]
fn run(talker: &mut Talker, text: &str, hook: bool) -> (Vec<i32>, Vec<i32>) {
    let mut seen = Vec::new();
    let mut f = |codes: &[i32]| seen.extend_from_slice(codes);
    let g = talker
        .generate_streamed(
            text,
            None,
            "hk_cantonese_speaker",
            language_id("zh").unwrap(),
            SEED,
            FRAMES,
            Sampling::default(),
            None,
            if hook { Some(&mut f) } else { None },
        )
        .unwrap();
    (g.codes, seen)
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TINY_TALKER (scripts/tiny_fixture.sh)"]
fn a_quiet_hook_changes_no_token_and_a_drawing_hook_is_red_on_the_tiny_model() {
    let path = talker_path();
    let mut talker = Talker::load(&path, 4).unwrap();
    let mut planted = Talker::load_planted(&path, 4, Default::default(), Plant::FrameHookDraw).unwrap();
    let mut failures = Vec::new();
    for (name, text) in texts() {
        let (bare, _) = run(&mut talker, &text, false);
        let (hooked, seen) = run(&mut talker, &text, true);
        let quiet = hooked == bare && seen == hooked;
        let (planted_hooked, _) = run(&mut planted, &text, true);
        let red = planted_hooked != bare;
        let (planted_bare, _) = run(&mut planted, &text, false);
        let off = planted_bare == bare;
        println!(
            "{name:<14} {} codes: quiet hook {}, hook saw the returned frames {}; planted hook {}; plant without a \
             hook {}",
            bare.len(),
            if hooked == bare { "identical" } else { "DIFFERS" },
            seen == hooked,
            if red { "red" } else { "NOT SEEN" },
            if off { "identical" } else { "DIFFERS" },
        );
        if !(quiet && red && off) {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "the no-change check failed on {failures:?}");
}
