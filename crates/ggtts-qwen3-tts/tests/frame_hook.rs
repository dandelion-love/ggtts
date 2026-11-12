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
#[cfg(feature = "plant")]
use ggtts_core::hooks::Wants;
use ggtts_core::hooks::{Hooks, StopTrace};
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

/// The frame cap for the promotion gate's two cases: the fixture's own unless `GGTTS_NO_CHANGE_MAX_FRAMES` lowers it.
/// The hosted CI job lowers it on the tiny fixture, which never stops and so always runs to the cap; the gate leaves
/// it unset, so on the full model nothing changes.
fn frame_cap(gold: &Golden) -> i32 {
    let own = gold.max_tokens as i32;
    match std::env::var("GGTTS_NO_CHANGE_MAX_FRAMES") {
        Ok(v) => {
            let cap: i32 = v.parse().expect("GGTTS_NO_CHANGE_MAX_FRAMES must be a whole number");
            assert!(cap > 0, "GGTTS_NO_CHANGE_MAX_FRAMES must be above zero");
            cap.min(own)
        }
        Err(_) => own,
    }
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
        instruction: None,
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

#[test]
#[ignore = "needs pinned GGTTS_TALKER and GGTTS_CODEC; GGTTS_DEVICE selects cpu or cuda"]
fn stop_trace_tensor_hook_preserves_real_tokens_and_samples() {
    let (talker, codec) = (env_path("GGTTS_TALKER"), env_path("GGTTS_CODEC"));
    let backend = std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = ggtts_qwen3_tts::cpu::Device::parse(&backend).expect("GGTTS_DEVICE must be cpu or cuda");
    let gold = goldens().remove(0);
    let voice = Voice {
        speaker: gold.speaker.clone(),
        language: gold.language.clone(),
        max_frames: frame_cap(&gold),
        device,
        ..Default::default()
    };
    let req = ChunkRequest {
        text: &gold.text,
        seed: gold.seed,
        chunk: 0,
        draw: 0,
        instruction: None,
    };
    let mut engine = Qwen3Tts::load(&talker, &codec, 4, voice).unwrap();
    let plain = engine.generate(&req).unwrap();
    let plain_audio = engine.decode(&plain.codes).unwrap();

    let path = std::env::temp_dir().join(format!("ggtts-real-stop-hook-{}-{backend}.jsonl", std::process::id()));
    let file = std::fs::File::create(&path).unwrap();
    let mut hooks = Hooks::new();
    hooks.add(Box::new(StopTrace::new(file)));
    let mut events = ggtts_core::trace::Recorder::new(1);
    let observed = engine.generate_hooked(&req, &mut events, &mut hooks).unwrap();
    hooks.finish();
    assert!(hooks.failures().is_empty(), "{:?}", hooks.failures());
    let observed_audio = engine.decode(&observed.codes).unwrap();
    assert!(!plain.codes.frames.is_empty(), "an empty generation proves nothing");
    assert_eq!(
        plain.codes, observed.codes,
        "event tracing or stop-trace hook changed codec tokens"
    );
    assert_eq!(
        plain_audio.samples, observed_audio.samples,
        "tracing or stop-trace hook changed audio"
    );
    assert_eq!(plain_audio.sample_rate, observed_audio.sample_rate);
    let hex = |bytes: &[u8]| {
        ggtts_core::seed::sha256(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let token_bytes: Vec<u8> = plain.codes.frames.iter().flat_map(|x| x.to_le_bytes()).collect();
    let sample_bytes: Vec<u8> = plain_audio
        .samples
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let lines = std::fs::read_to_string(&path).unwrap();
    assert!(!lines.trim().is_empty(), "the hook produced no samples");
    assert!(!events.lines.is_empty(), "the event trace was empty");
    let _ = std::fs::remove_file(path);
    println!(
        "backend={backend}; seed={}; frames={}; samples={}; tokens_sha256={}; samples_sha256={}; \
              full tokens and samples identical; \
              limits: one fixture, no timing or external-artifact identity",
        gold.seed,
        plain.codes.n_frames(),
        plain_audio.samples.len(),
        hex(&token_bytes),
        hex(&sample_bytes)
    );
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs pinned GGTTS_TALKER; GGTTS_DEVICE selects cpu or cuda"]
fn a_hook_boundary_logit_mutation_is_detected_on_real_qwen() {
    let talker = env_path("GGTTS_TALKER");
    let backend = std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = ggtts_qwen3_tts::cpu::Device::parse(&backend).expect("GGTTS_DEVICE must be cpu or cuda");
    let gold = goldens().remove(0);
    let run = |talker: &mut Talker, observe: bool| {
        let mut seen = 0usize;
        let mut callback = |_: i32, _: i32, _: Option<f64>, logits: Option<&[f32]>| {
            assert!(logits.is_some());
            seen += 1;
        };
        let wants = Wants {
            logits: true,
            p_stop: false,
        };
        let result = talker
            .generate_streamed(
                &gold.text,
                None,
                &gold.speaker,
                language_id(&gold.language).unwrap(),
                gold.seed,
                frame_cap(&gold),
                Sampling::default(),
                if observe { Some((wants, &mut callback)) } else { None },
                None,
            )
            .unwrap();
        (result.codes, seen)
    };
    let mut normal = Talker::load_on(&talker, 4, device).unwrap();
    let (base, _) = run(&mut normal, false);
    let (observed, seen) = run(&mut normal, true);
    assert!(!base.is_empty() && seen > 0);
    assert_eq!(base, observed, "a read-only logit observer changed tokens");
    drop(normal);
    let mut planted = Talker::load_planted(&talker, 4, device, Plant::HookLogitMutation).unwrap();
    let (plant_off, _) = run(&mut planted, false);
    let (plant_on, plant_seen) = run(&mut planted, true);
    assert_eq!(base, plant_off, "the plant fired without an observer");
    assert!(plant_seen > 0);
    assert_ne!(
        base, plant_on,
        "the identity comparison missed a hook-boundary logit mutation"
    );
    println!(
        "backend={backend}; seed={}; observer_steps={seen}; quiet=identical; \
              logit_mutation=detected; plant_without_observer=identical; \
              limit: simulated corruption at the hook boundary, not safe Rust mutating its immutable view",
        gold.seed
    );
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
            None,
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
            instruction: None,
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
