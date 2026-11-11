//! The style instruction (`Voice::instruction`, the reference CLI's `--instruction`) end to end, on real checkpoints.
//!
//! * No instruction and an empty one give the codes of the CPU golden fixtures, recorded before instructions
//!   existed: an absent instruction adds nothing to the prompt. (`token_identity`'s planted `InstructionLeak` shows
//!   the same fixtures turn red when a prefix leaks in.)
//! * On a CustomVoice checkpoint an instruction changes the codes for the same text and seed, while rendering the
//!   same request twice does not: the difference is the instruction's, not the sampler's.
//! * On a Base checkpoint an instruction is refused with an error naming the checkpoint type, at load and per
//!   request.
//!
//! Parity of the instructed codes with the reference is `token_identity`'s `instructed_codec_tokens_are_identical_to_the_reference`.
//!
//! Needs the published GGUFs, `GGTTS_TALKER=<talker.gguf> GGTTS_CODEC=<codec.gguf>`, and for the refusal a Base
//! talker converted from `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `GGTTS_BASE_TALKER=<base talker.gguf>`:
//! `cargo test -p ggtts-qwen3-tts --features ggml --release --test instruction -- --ignored --nocapture`.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};
use ggtts_qwen3_tts::fixtures::{recorded_threads, Golden};
use ggtts_qwen3_tts::talker::{language_id, Sampling, Talker};

const INSTRUCTION: &str = "Speak slowly and sadly.";

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn goldens() -> Vec<Golden> {
    let g = Golden::load_all(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu")).unwrap();
    assert!(!g.is_empty());
    g
}

fn voice(gold: &Golden, instruction: Option<&str>) -> Voice {
    Voice {
        speaker: gold.speaker.clone(),
        instruction: instruction.map(str::to_string),
        language: gold.language.clone(),
        max_frames: gold.max_tokens as i32,
        ..Default::default()
    }
}

fn codes(engine: &mut Qwen3Tts, gold: &Golden) -> Vec<i32> {
    let req = ChunkRequest {
        text: &gold.text,
        seed: gold.seed,
        chunk: 0,
        draw: 0,
    };
    let g = engine.generate(&req).unwrap();
    g.codes.frames.iter().map(|&c| i32::from(c)).collect()
}

#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC, the published GGUFs"]
fn no_instruction_and_an_empty_one_are_the_golden_codes() {
    let (talker, codec) = (env_path("GGTTS_TALKER"), env_path("GGTTS_CODEC"));
    let goldens = goldens();
    let threads = recorded_threads(&goldens).unwrap();
    let mut failures = Vec::new();
    for gold in &goldens {
        assert!(
            gold.instruction.is_empty(),
            "{}: a CPU fixture with an instruction",
            gold.name
        );
        let mut none = Qwen3Tts::load(&talker, &codec, threads, voice(gold, None)).unwrap();
        let a = codes(&mut none, gold);
        drop(none);
        let mut empty = Qwen3Tts::load(&talker, &codec, threads, voice(gold, Some(""))).unwrap();
        let b = codes(&mut empty, gold);
        let ok = a == gold.codes && b == gold.codes;
        println!(
            "{:<14} {} frames: no instruction {}, empty instruction {}",
            gold.name,
            gold.frames(),
            if a == gold.codes { "IDENTICAL" } else { "DIFFERS" },
            if b == gold.codes { "IDENTICAL" } else { "DIFFERS" },
        );
        if !ok {
            failures.push(gold.name.clone());
        }
    }
    assert!(
        failures.is_empty(),
        "an absent or empty instruction changed {failures:?}"
    );
}

#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC, the published GGUFs"]
fn an_instruction_changes_the_codes_on_a_custom_voice_checkpoint() {
    let (talker, codec) = (env_path("GGTTS_TALKER"), env_path("GGTTS_CODEC"));
    let goldens = goldens();
    let threads = recorded_threads(&goldens).unwrap();
    let gold = goldens.iter().find(|g| g.name == "park").expect("the park fixture");
    let mut plain = Qwen3Tts::load(&talker, &codec, threads, voice(gold, None)).unwrap();
    let first = codes(&mut plain, gold);
    let again = codes(&mut plain, gold);
    drop(plain);
    let mut styled = Qwen3Tts::load(&talker, &codec, threads, voice(gold, Some(INSTRUCTION))).unwrap();
    let instructed = codes(&mut styled, gold);
    let instructed_again = codes(&mut styled, gold);
    let first_diff = first.iter().zip(&instructed).position(|(a, b)| a != b);
    println!(
        "{}: seed {}; no instruction {} frames, repeated {}; instruction {INSTRUCTION:?} {} frames, repeated {}; \
         first difference at code {:?}",
        gold.name,
        gold.seed,
        first.len() / 16,
        if again == first { "identical" } else { "DIFFERS" },
        instructed.len() / 16,
        if instructed_again == instructed {
            "identical"
        } else {
            "DIFFERS"
        },
        first_diff,
    );
    // The no-op reads as no change: the same request twice gives the same codes, with or without the instruction.
    assert_eq!(
        first, again,
        "the same request without an instruction rendered twice differs"
    );
    assert_eq!(
        instructed, instructed_again,
        "the same instructed request rendered twice differs"
    );
    assert_ne!(first, instructed, "the instruction did not change the codes");
}

#[test]
#[ignore = "needs GGTTS_BASE_TALKER, a talker converted from Qwen3-TTS-12Hz-1.7B-Base"]
fn an_instruction_on_a_base_checkpoint_is_refused() {
    let base = env_path("GGTTS_BASE_TALKER");
    let mut talker = Talker::load(&base, 4).unwrap();
    assert_eq!(talker.config.tts_model_type, "base");
    assert!(!talker.config.supports_instruction);
    // Per request: refused before anything is generated, whatever the speaker.
    let e = talker
        .generate(
            "Hello there.",
            "any",
            INSTRUCTION,
            language_id("en").unwrap(),
            1,
            8,
            Sampling::default(),
        )
        .unwrap_err();
    println!("Talker::generate: {}", e.0);
    assert!(e.0.contains("does not take a style instruction"), "{}", e.0);
    assert!(e.0.contains("\"base\""), "{}", e.0);
    assert!(talker.check_instruction("").is_ok(), "no instruction is refused");
    drop(talker);
    // At load: refused before the codec is read, so no codec file is needed to see it.
    let voice = Voice {
        instruction: Some(INSTRUCTION.into()),
        ..Default::default()
    };
    let e = Qwen3Tts::load(&base, Path::new("no-codec.gguf"), 4, voice)
        .err()
        .expect("an instruction on a Base checkpoint must be refused");
    println!("Qwen3Tts::load: {}", e.0);
    assert!(e.0.contains("does not take a style instruction"), "{}", e.0);
    assert!(e.0.contains("\"base\""), "{}", e.0);
}
