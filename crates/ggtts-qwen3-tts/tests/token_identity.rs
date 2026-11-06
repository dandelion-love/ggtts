//! The talker and code predictor against the reference: for every CPU golden fixture, the same text, speaker,
//! language and seed must give the same codec tokens, every frame and every codebook, and the same frame count.
//!
//! Needs the published talker GGUF: `GGTTS_TALKER=<qwen-talker-...-q8_0.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test token_identity -- --ignored --nocapture`. Without `GGTTS_TALKER` the test fails
//! rather than passing quietly. `GGTTS_PLANT` (`rope-normal`, `sampler-off-by-one`) plants a defect, and the test
//! must then fail. `GGTTS_ONLY=<name>` runs one fixture.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::fixtures::Golden;
use ggtts_qwen3_tts::talker::{language_id, Plant, Sampling, Talker};
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

#[test]
#[ignore = "needs GGTTS_TALKER, the published talker GGUF"]
fn codec_tokens_are_identical_to_the_reference() {
    let path = PathBuf::from(std::env::var("GGTTS_TALKER").expect("set GGTTS_TALKER to the talker GGUF"));
    let plant = match std::env::var("GGTTS_PLANT").as_deref() {
        Err(_) | Ok("") | Ok("none") => Plant::None,
        Ok("rope-normal") => Plant::RopeNormal,
        Ok("sampler-off-by-one") => Plant::SamplerOffByOne,
        Ok(p) => panic!("unknown plant {p}"),
    };
    let only = std::env::var("GGTTS_ONLY").ok();
    let goldens = Golden::load_all(&fixtures_dir()).unwrap();
    assert!(!goldens.is_empty());
    let mut h = Sha256::new();
    std::io::copy(&mut std::fs::File::open(&path).unwrap(), &mut h).unwrap();
    let talker_sha = format!("{:x}", h.finalize());
    let mut talker = Talker::load_on(&path, 4, device(), plant).unwrap();
    let (mut same, mut failed) = (0, Vec::new());
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
            failed.push(gold.name.clone());
        }
    }
    println!("token identity: {same} identical, {} differ", failed.len());
    assert!(same + failed.len() > 0, "no fixture ran");
    assert!(failed.is_empty(), "tokens differ from the reference on {failed:?}");
}
