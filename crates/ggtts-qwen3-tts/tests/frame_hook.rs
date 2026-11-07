//! The talker's frame hook and frame-level streaming end to end, on every CPU golden fixture.
//!
//! * With a no-op hook the tokens are the reference's, every frame, and the hook sees exactly the returned frames,
//!   in order.
//! * The planted `FrameHookDraw` (the hook draws once from the sampler's stream) is red with a hook, and quiet
//!   without one: the check sees a hook that is not quiet.
//! * `Qwen3Tts::render_streamed` at `every = 12, hold = 32`: the concatenated samples are bit-identical to the batch
//!   decode of the same codes, the first samples arrive before the talker has finished, and the stop-probability
//!   trace it writes is byte-identical to the one `generate` writes for the same request.
//!
//! The hook's own check needs the test-only `plant` feature.
//!
//! Needs both published GGUFs: `GGTTS_TALKER=<talker.gguf> GGTTS_CODEC=<codec.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test frame_hook -- --ignored --nocapture`.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::stream::first_sample_diff;
use ggtts_qwen3_tts::codec::Decoder;
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};
use ggtts_qwen3_tts::fixtures::Golden;
use ggtts_qwen3_tts::stream::{DEFAULT_EVERY, DEFAULT_HOLD};
#[cfg(feature = "plant")]
use ggtts_qwen3_tts::talker::{language_id, Plant, Sampling, Talker};

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn goldens() -> Vec<Golden> {
    let g = Golden::load_all(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu")).unwrap();
    assert!(!g.is_empty());
    g
}

#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC, the published GGUFs"]
fn event_trace_emits_steps_without_changing_tokens_or_audio() {
    let (talker, codec) = (env_path("GGTTS_TALKER"), env_path("GGTTS_CODEC"));
    let gold = goldens().remove(0);
    let voice = Voice {
        speaker: gold.speaker.clone(),
        language: gold.language.clone(),
        max_frames: gold.max_tokens as i32,
        ..Default::default()
    };
    let req = ChunkRequest {
        text: &gold.text,
        seed: gold.seed,
        chunk: 0,
        draw: 0,
    };
    let mut engine = Qwen3Tts::load(&talker, &codec, 4, voice).unwrap();
    let plain = engine.generate(&req).unwrap();
    let mut events = ggtts_core::trace::Recorder::new(1);
    let traced = engine.generate_traced(&req, &mut events).unwrap();
    assert!(plain.codes.frames == traced.codes.frames, "trace changed tokens");
    let a = engine.decode(&plain.codes).unwrap();
    let b = engine.decode(&traced.codes).unwrap();
    assert!(a.samples == b.samples, "trace changed audio");
    // The talker also reports the EOS step and any non-audio thinking steps.
    assert!(events.lines.len() >= traced.codes.n_frames());
    assert!(events
        .lines
        .iter()
        .all(|line| line["ev"] == "step" && line["p_stop"].is_number()));
}

/// Generate a fixture with a hook (or none); the returned codes and the frames the hook saw.
#[cfg(feature = "plant")]
fn run(talker: &mut Talker, gold: &Golden, hook: bool) -> (Vec<i32>, Vec<i32>) {
    let lang = language_id(&gold.language).unwrap();
    let mut seen = Vec::new();
    let mut f = |codes: &[i32]| seen.extend_from_slice(codes);
    let g = talker
        .generate_streamed(
            &gold.text,
            &gold.speaker,
            lang,
            gold.seed,
            gold.max_tokens as i32,
            Sampling::default(),
            None,
            if hook { Some(&mut f) } else { None },
        )
        .unwrap();
    (g.codes, seen)
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn a_quiet_hook_changes_no_token_and_a_drawing_hook_is_red() {
    let path = env_path("GGTTS_TALKER");
    let mut talker = Talker::load(&path, 4).unwrap();
    let mut planted = Talker::load_planted(&path, 4, Default::default(), Plant::FrameHookDraw).unwrap();
    let mut failures = Vec::new();
    for gold in &goldens() {
        let (codes, seen) = run(&mut talker, gold, true);
        let quiet = codes == gold.codes && seen == codes;
        let (planted_codes, _) = run(&mut planted, gold, true);
        let red = planted_codes != gold.codes;
        let (unhooked, _) = run(&mut planted, gold, false);
        let off = unhooked == gold.codes;
        println!(
            "{:<14} {} frames: no-op hook {}, hook sees the returned frames {}; planted hook {}; plant without a hook {}",
            gold.name,
            gold.frames(),
            if codes == gold.codes { "IDENTICAL" } else { "DIFFERS" },
            seen == codes,
            if red { "red" } else { "NOT SEEN" },
            if off { "identical" } else { "DIFFERS" },
        );
        if !(quiet && red && off) {
            failures.push(gold.name.clone());
        }
    }
    assert!(failures.is_empty(), "frame hook check failed on {failures:?}");
}

#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC, the published GGUFs"]
fn frame_level_streaming_end_to_end_is_the_batch_decode() {
    let (talker, codec) = (env_path("GGTTS_TALKER"), env_path("GGTTS_CODEC"));
    let mut batch = Decoder::load(&codec, 4).unwrap();
    let mut failures = Vec::new();
    for gold in &goldens() {
        let scratch = std::env::temp_dir().join(format!("ggtts-frame-hook-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let (streamed_trace, batch_trace) = (scratch.join("streamed.jsonl"), scratch.join("batch.jsonl"));
        let voice = |trace: &Path| Voice {
            speaker: gold.speaker.clone(),
            language: gold.language.clone(),
            max_frames: gold.max_tokens as i32,
            trace: Some(trace.to_path_buf()),
            ..Default::default()
        };
        let mut engine = Qwen3Tts::load(&talker, &codec, 4, voice(&streamed_trace)).unwrap();
        let req = ChunkRequest {
            text: &gold.text,
            seed: gold.seed,
            chunk: 0,
            draw: 0,
        };
        let mut stream = Vec::new();
        let mut first_at = None;
        let g = engine
            .render_streamed(&req, DEFAULT_EVERY, DEFAULT_HOLD, &mut |s, frames| {
                first_at.get_or_insert(frames);
                stream.extend_from_slice(s);
            })
            .unwrap();
        let codes: Vec<i32> = g.codes.frames.iter().map(|&c| i32::from(c)).collect();
        // The trace: render_streamed writes the same lines generate does.
        drop(engine);
        let mut batch_engine = Qwen3Tts::load(&talker, &codec, 4, voice(&batch_trace)).unwrap();
        batch_engine.generate(&req).unwrap();
        drop(batch_engine);
        let (st, bt) = (
            std::fs::read(&streamed_trace).unwrap(),
            std::fs::read(&batch_trace).unwrap(),
        );
        let traced = !st.is_empty() && st == bt;
        let want = batch.decode(&codes).unwrap().samples;
        let diff = first_sample_diff(&stream, &want);
        let n = g.codes.n_frames();
        let early = first_at.is_some_and(|f| f < n);
        println!(
            "{:<14} {n} frames: stream {} the batch decode; first audio after {} of {n} frames; trace {}",
            gold.name,
            match diff {
                None => "IDENTICAL to".to_string(),
                Some(i) => format!("DIFFERS at sample {i} from"),
            },
            first_at.map_or("-".to_string(), |f| f.to_string()),
            if traced {
                "identical to generate's"
            } else {
                "DIFFERS from generate's (or is empty)"
            },
        );
        if codes != gold.codes || diff.is_some() || (n > 44 && !early) || !traced {
            failures.push(gold.name.clone());
        }
    }
    assert!(failures.is_empty(), "frame-level streaming failed on {failures:?}");
}
