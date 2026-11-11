//! The talker and code predictor against the reference: for every CPU golden fixture, the same text, speaker,
//! language and seed must give the same codec tokens, every frame and every codebook, and the same frame count.
//!
//! Needs the published talker GGUF: `GGTTS_TALKER=<qwen-talker-...-q8_0.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test token_identity -- --ignored --nocapture`. Without `GGTTS_TALKER` the test fails
//! rather than passing quietly. The talker runs with the thread count the fixtures were recorded with.
//! `GGTTS_ONLY=<name>` runs one fixture. The style-instruction fixtures in `fixtures/instruct/` (recorded with the
//! reference CLI's `--instruction`) are checked the same way, against the same talker.
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

/// What one pass over the fixtures found.
struct Outcome {
    same: usize,
    differ: Vec<String>,
}

/// Generates every fixture's request with the talker `load` gives, prints one line per fixture and a summary line,
/// and returns the fixtures that differ. Refuses a talker the fixtures were not recorded with.
fn check(label: &str, dir: &Path, load: impl FnOnce(&Path, i32, Device) -> Result<Talker, EngineError>) -> Outcome {
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
        let got = talker
            .generate(
                &gold.text,
                &gold.speaker,
                &gold.instruction,
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
    let o = check("no plant", &fixtures_dir(), Talker::load_on);
    assert!(o.same > 0);
    assert!(
        o.differ.is_empty(),
        "tokens differ from the reference on {:?}",
        o.differ
    );
}

/// The style-instruction fixtures (`fixtures/instruct/`): the same talker and speaker, each request with an
/// instruction the reference CLI was given as `--instruction`.
fn instruct_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/instruct")
}

#[test]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn instructed_codec_tokens_are_identical_to_the_reference() {
    let o = check("instruction, no plant", &instruct_dir(), Talker::load_on);
    assert!(o.same > 0);
    assert!(
        o.differ.is_empty(),
        "tokens differ from the reference on {:?}",
        o.differ
    );
}

/// The expected failures: each test passes only if its planted defect turns every fixture red.
#[cfg(feature = "plant")]
mod planted {
    use super::*;
    use ggtts_qwen3_tts::talker::Plant;

    fn with(plant: Plant) -> Outcome {
        with_in(plant, &fixtures_dir())
    }

    fn with_in(plant: Plant, dir: &Path) -> Outcome {
        check(&format!("planted {plant:?}"), dir, |p, t, d| {
            Talker::load_planted(p, t, d, plant)
        })
    }

    /// The no-instruction fixtures are the guard that an absent instruction adds nothing: a prefix leaking into
    /// them must turn every one red.
    #[test]
    #[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
    fn instruction_leak_is_seen() {
        let o = with(Plant::InstructionLeak);
        assert_eq!(o.same, 0, "the leaked instruction left {} fixtures identical", o.same);
    }

    /// The instruction fixtures must see an instruction that never reaches the talker.
    #[test]
    #[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
    fn instruction_dropped_is_seen() {
        let o = with_in(Plant::InstructionDropped, &instruct_dir());
        assert_eq!(o.same, 0, "the dropped instruction left {} fixtures identical", o.same);
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
}
