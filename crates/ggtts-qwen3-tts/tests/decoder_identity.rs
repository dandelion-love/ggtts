//! The codec decoder against the reference: every CPU golden fixture's tokens, decoded here, must give samples whose
//! little-endian f32 bytes hash to the sha256 the reference decoder recorded. Bit identity, no tolerance.
//!
//! Needs the published codec GGUF: `GGTTS_CODEC=<qwen-tokenizer-12hz-q8_0.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test decoder_identity -- --ignored`. Without `GGTTS_CODEC` the test fails rather than
//! passing quietly. `GGTTS_PLANT` (`rope-normal`, `snake-no-epsilon`, `window-off-by-one`) plants a defect, and the
//! test must then fail on every fixture.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_qwen3_tts::codec::{Decoder, Plant};
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::fixtures::Golden;
use sha2::{Digest, Sha256};

fn sha256_file(path: &Path) -> String {
    let mut h = Sha256::new();
    let mut f = std::fs::File::open(path).unwrap();
    std::io::copy(&mut f, &mut h).unwrap();
    format!("{:x}", h.finalize())
}

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
#[ignore = "needs GGTTS_CODEC, the published codec GGUF"]
fn decoder_samples_are_bit_identical_to_the_reference() {
    let codec = PathBuf::from(std::env::var("GGTTS_CODEC").expect("set GGTTS_CODEC to the codec GGUF"));
    let plant = match std::env::var("GGTTS_PLANT").as_deref() {
        Err(_) | Ok("") | Ok("none") => Plant::None,
        Ok("rope-normal") => Plant::RopeNormal,
        Ok("snake-no-epsilon") => Plant::SnakeNoEpsilon,
        Ok("window-off-by-one") => Plant::WindowOffByOne,
        Ok(p) => panic!("unknown plant {p}"),
    };
    let goldens = Golden::load_all(&fixtures_dir()).unwrap();
    assert!(!goldens.is_empty());
    let codec_sha = sha256_file(&codec);
    let mut dec = Decoder::load_on(&codec, 4, device(), plant).unwrap();
    let mut failed = Vec::new();
    for gold in &goldens {
        assert_eq!(
            codec_sha, gold.codec_sha256,
            "{}: not the codec the fixtures were recorded with",
            gold.name
        );
        let pcm = dec.decode(&gold.codes).unwrap();
        let bytes: Vec<u8> = pcm.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let got = format!("{:x}", Sha256::digest(&bytes));
        let same = pcm.samples.len() == gold.sample_count && got == gold.samples_sha256;
        println!(
            "{} {}: {} frames, {} samples, sha256 {}",
            if same { "IDENTICAL" } else { "DIFFERS" },
            gold.name,
            gold.frames(),
            pcm.samples.len(),
            &got[..16]
        );
        if !same {
            failed.push(gold.name.clone());
        }
    }
    println!(
        "decoder identity: {} identical, {} differ",
        goldens.len() - failed.len(),
        failed.len()
    );
    assert!(failed.is_empty(), "decoder differs from the reference on {failed:?}");
}
