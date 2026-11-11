//! The talker and code predictor against the reference: for every CPU golden fixture, the same text, speaker,
//! language and seed must give the same codec tokens, every frame and every codebook, and the same frame count.
//!
//! Needs the published talker GGUF: `GGTTS_TALKER=<qwen-talker-...-q8_0.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test token_identity -- --ignored --nocapture`. Without `GGTTS_TALKER` the test fails
//! rather than passing quietly. The talker runs with the thread count the fixtures were recorded with.
//! `GGTTS_ONLY=<name>` runs one fixture.
//!
//! The same holds for the style-instruction fixtures in `fixtures/instruct/`, recorded with the reference's
//! `--instruction` (`instructed_codec_tokens_are_identical_to_the_reference`), and for the plain fixtures rendered with
//! an empty instruction, which must be no instruction (`an_empty_instruction_is_no_instruction`).
//! `a_base_checkpoint_refuses_an_instruction` needs a Base talker in `GGTTS_BASE_TALKER`.
//!
//! With the `plant` feature, the `planted::*` tests are the expected failures: each plants one defect and passes only
//! if every fixture it runs then differs.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_core::engine::EngineError;
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::fixtures::{recorded_threads, Golden};
use ggtts_qwen3_tts::talker::{language_id, Sampling, Talker};
use sha2::{Digest, Sha256};

/// `GGTTS_DEVICE`: `cpu` (the default) or `cuda`.
fn device() -> Device {
    let d = std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    Device::parse(&d).unwrap_or_else(|| panic!("GGTTS_DEVICE must be cpu or cuda, not {d}"))
}

/// `GGTTS_FIXTURES`: a fixture directory, by default the committed CPU fixtures. A CUDA run points it at fixtures
/// recorded with the reference on CUDA (`scripts/cuda_probe.sh`).
fn fixtures_dir() -> PathBuf {
    std::env::var_os("GGTTS_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu"))
}

/// `GGTTS_INSTRUCT_FIXTURES`: the style-instruction fixtures, by default the committed CPU ones.
fn instruct_fixtures_dir() -> PathBuf {
    std::env::var_os("GGTTS_INSTRUCT_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/instruct"))
}

/// What each fixture is rendered with besides its own request.
#[derive(Clone, Copy)]
enum Instruction {
    /// The fixture's own instruction: none for a fixture recorded without one.
    AsRecorded,
    /// An empty instruction, for fixtures recorded without one.
    Empty,
}

/// What one pass over the fixtures found.
struct Outcome {
    same: usize,
    differ: Vec<String>,
}

/// Generates every fixture's request with the talker `load` gives, prints one line per fixture and a summary line,
/// and returns the fixtures that differ. Refuses a talker the fixtures were not recorded with.
fn check(label: &str, load: impl FnOnce(&Path, i32, Device) -> Result<Talker, EngineError>) -> Outcome {
    check_in(&fixtures_dir(), Instruction::AsRecorded, label, load)
}

fn check_in(
    dir: &Path,
    instruction: Instruction,
    label: &str,
    load: impl FnOnce(&Path, i32, Device) -> Result<Talker, EngineError>,
) -> Outcome {
    let path = PathBuf::from(std::env::var("GGTTS_TALKER").expect("set GGTTS_TALKER to the talker GGUF"));
    let only = std::env::var("GGTTS_ONLY").ok();
    let goldens = Golden::load_all(dir).unwrap();
    assert!(!goldens.is_empty());
    let threads = recorded_threads(&goldens).unwrap();
    let mut h = Sha256::new();
    std::io::copy(&mut std::fs::File::open(&path).unwrap(), &mut h).unwrap();
    let talker_sha = format!("{:x}", h.finalize());
    let mut talker = load(&path, threads, device()).unwrap();
    let (mut same, mut differ) = (0, Vec::new());
    for gold in goldens.iter().filter(|g| only.as_deref().is_none_or(|o| o == g.name)) {
        assert_eq!(
            talker_sha, gold.talker_sha256,
            "{}: not the talker the fixtures were recorded with",
            gold.name
        );
        let lang = language_id(&gold.language).expect("a language the reference CLI knows");
        let instruct = match instruction {
            Instruction::AsRecorded => gold.instruction.as_deref(),
            Instruction::Empty => {
                assert!(
                    gold.instruction.is_none(),
                    "{}: recorded with an instruction",
                    gold.name
                );
                Some("")
            }
        };
        let got = talker
            .generate(
                &gold.text,
                instruct,
                &gold.speaker,
                lang,
                gold.seed,
                gold.max_tokens as i32,
                Sampling::default(),
            )
            .unwrap();
        let first = got.codes.iter().zip(&gold.codes).position(|(a, b)| a != b);
        let identical = got.codes == gold.codes;
        match (identical, first) {
            (true, _) => println!("IDENTICAL {}: {} frames", gold.name, gold.frames()),
            (false, Some(i)) => println!(
                "DIFFERS {}: first at frame {} codebook {} (got {}, reference {}); {} frames against {}",
                gold.name,
                i / 16,
                i % 16,
                got.codes[i],
                gold.codes[i],
                got.codes.len() / 16,
                gold.frames()
            ),
            (false, None) => println!(
                "DIFFERS {}: a common prefix, then {} frames against {}",
                gold.name,
                got.codes.len() / 16,
                gold.frames()
            ),
        }
        if identical {
            same += 1;
        } else {
            differ.push(gold.name.clone());
        }
    }
    println!(
        "token identity ({label}, {threads} threads): {same} identical, {} differ",
        differ.len()
    );
    assert!(same + differ.len() > 0, "no fixture ran");
    Outcome { same, differ }
}

#[test]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn codec_tokens_are_identical_to_the_reference() {
    let o = check("no plant", Talker::load_on);
    assert!(o.same > 0);
    assert!(
        o.differ.is_empty(),
        "tokens differ from the reference on {:?}",
        o.differ
    );
}

/// The style instruction placed before the prompt, as the reference places it: every frame and codebook of every
/// instruction fixture.
#[test]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn instructed_codec_tokens_are_identical_to_the_reference() {
    let o = check_in(
        &instruct_fixtures_dir(),
        Instruction::AsRecorded,
        "instructions",
        Talker::load_on,
    );
    assert!(o.same > 0);
    assert!(
        o.differ.is_empty(),
        "instructed tokens differ from the reference on {:?}",
        o.differ
    );
}

/// An empty instruction is no instruction: the plain fixtures, rendered with `Some("")`, give the reference's tokens
/// for no instruction, which are the tokens of a render without the field.
#[test]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn an_empty_instruction_is_no_instruction() {
    let o = check_in(
        &fixtures_dir(),
        Instruction::Empty,
        "empty instruction",
        Talker::load_on,
    );
    assert!(o.same > 0);
    assert!(
        o.differ.is_empty(),
        "an empty instruction changed the tokens of {:?}",
        o.differ
    );
}

/// On a checkpoint that takes instructions, an instruction changes the codes for a fixed text and seed: each
/// instruction fixture's text, rendered with and without its instruction.
#[test]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn an_instruction_changes_the_codes() {
    let path = PathBuf::from(std::env::var("GGTTS_TALKER").expect("set GGTTS_TALKER to the talker GGUF"));
    let goldens = Golden::load_all(&instruct_fixtures_dir()).unwrap();
    let threads = recorded_threads(&goldens).unwrap();
    let mut talker = Talker::load_on(&path, threads, device()).unwrap();
    assert!(
        talker.supports_instruction(),
        "a {} checkpoint takes no instruction",
        talker.model_type()
    );
    for gold in &goldens {
        let lang = language_id(&gold.language).unwrap();
        let mut render = |instruction| {
            talker
                .generate(
                    &gold.text,
                    instruction,
                    &gold.speaker,
                    lang,
                    gold.seed,
                    gold.max_tokens as i32,
                    Sampling::default(),
                )
                .unwrap()
                .codes
        };
        let with = render(gold.instruction.as_deref());
        let without = render(None);
        let first = with.iter().zip(&without).position(|(a, b)| a != b);
        println!(
            "{}: {} frames with the instruction, {} without; first difference (frame, codebook) {:?}",
            gold.name,
            with.len() / 16,
            without.len() / 16,
            first.map(|i| (i / 16, i % 16))
        );
        assert_ne!(with, without, "{}: the instruction left the codes unchanged", gold.name);
    }
}

/// A Base checkpoint takes no instruction, and says so, rather than rendering without it.
///
/// Needs `GGTTS_BASE_TALKER`, a Qwen3-TTS Base talker GGUF, for example `Qwen/Qwen3-TTS-12Hz-1.7B-Base` converted
/// with the reference's `scripts/convert_tts_to_gguf.py -t q8_0`. Without it the test fails rather than passing.
#[test]
#[ignore = "needs GGTTS_BASE_TALKER, a Qwen3-TTS Base talker GGUF"]
fn a_base_checkpoint_refuses_an_instruction() {
    let path = PathBuf::from(std::env::var("GGTTS_BASE_TALKER").expect("set GGTTS_BASE_TALKER to a Base talker GGUF"));
    let mut talker = Talker::load_on(&path, 4, device()).unwrap();
    assert_eq!(talker.model_type(), "base");
    assert!(!talker.supports_instruction());
    let e = talker
        .generate(
            "今日天氣好好",
            Some("Speak slowly."),
            "hk_cantonese_speaker",
            language_id("zh").unwrap(),
            20260905,
            8,
            Sampling::default(),
        )
        .expect_err("a Base checkpoint must refuse an instruction");
    println!("refused: {e}");
    assert!(
        e.0.contains("base model") && e.0.contains("no style instruction"),
        "{e}"
    );
    // No instruction and an empty one are not refused for being instructions.
    assert!(talker.check_instruction(None).is_ok());
    assert!(talker.check_instruction(Some("")).is_ok());
}

/// The expected failures: each test passes only if its planted defect turns every fixture red.
#[cfg(feature = "plant")]
mod planted {
    use super::*;
    use ggtts_qwen3_tts::talker::Plant;

    fn with(plant: Plant) -> Outcome {
        check(&format!("planted {plant:?}"), |p, t, d| {
            Talker::load_planted(p, t, d, plant)
        })
    }

    #[test]
    #[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
    fn rope_normal_is_seen() {
        let o = with(Plant::RopeNormal);
        assert_eq!(o.same, 0, "the planted rope left {} fixtures identical", o.same);
    }

    #[test]
    #[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
    fn sampler_off_by_one_is_seen() {
        let o = with(Plant::SamplerOffByOne);
        assert_eq!(o.same, 0, "the planted sampler left {} fixtures identical", o.same);
    }

    /// The instruction's rows after the role rows instead of before them: every instruction fixture must differ.
    #[test]
    #[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
    fn instruction_after_role_is_seen() {
        let o = check_in(
            &instruct_fixtures_dir(),
            Instruction::AsRecorded,
            "planted InstructionAfterRole",
            |p, t, d| Talker::load_planted(p, t, d, Plant::InstructionAfterRole),
        );
        assert_eq!(
            o.same, 0,
            "the misplaced instruction left {} fixtures identical",
            o.same
        );
    }
}
