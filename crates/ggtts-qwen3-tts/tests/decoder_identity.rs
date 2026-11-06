//! The codec decoder against the reference: every CPU golden fixture's tokens, decoded here, must give samples whose
//! little-endian f32 bytes hash to the sha256 the reference decoder recorded. Bit identity, no tolerance.
//!
//! Needs the published codec GGUF: `GGTTS_CODEC=<qwen-tokenizer-12hz-q8_0.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test decoder_identity -- --ignored`. Without `GGTTS_CODEC` the test fails rather than
//! passing quietly. The decoder runs with the thread count the fixtures were recorded with.
//!
//! With the `plant` feature, the `planted::*` tests are the expected failures: each plants one defect and passes only
//! if the check then finds fixtures that differ. `rope-normal` and `snake-no-epsilon` must change every fixture;
//! `window-off-by-one` must change at least one, since a fixture shorter than the window cannot see it.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_core::engine::EngineError;
use ggtts_qwen3_tts::codec::Decoder;
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::fixtures::{recorded_threads, Golden};
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

/// What one pass over the fixtures found.
struct Outcome {
    total: usize,
    differ: Vec<String>,
}

/// Decodes every fixture with the decoder `load` gives, prints one line per fixture and a summary line, and returns
/// the fixtures that differ. Refuses a codec the fixtures were not recorded with.
fn check(label: &str, load: impl FnOnce(&Path, i32, Device) -> Result<Decoder, EngineError>) -> Outcome {
    let codec = PathBuf::from(std::env::var("GGTTS_CODEC").expect("set GGTTS_CODEC to the codec GGUF"));
    let goldens = Golden::load_all(&fixtures_dir()).unwrap();
    assert!(!goldens.is_empty());
    let threads = recorded_threads(&goldens).unwrap();
    let codec_sha = sha256_file(&codec);
    let mut dec = load(&codec, threads, device()).unwrap();
    let mut differ = Vec::new();
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
            differ.push(gold.name.clone());
        }
    }
    println!(
        "decoder identity ({label}, {threads} threads): {} identical, {} differ",
        goldens.len() - differ.len(),
        differ.len()
    );
    Outcome {
        total: goldens.len(),
        differ,
    }
}

#[test]
#[ignore = "needs GGTTS_CODEC, the published codec GGUF"]
fn decoder_samples_are_bit_identical_to_the_reference() {
    let o = check("no plant", Decoder::load_on);
    assert!(o.total > 0);
    assert!(
        o.differ.is_empty(),
        "decoder differs from the reference on {:?}",
        o.differ
    );
}

/// The expected failures: each test passes only if its planted defect turns the check red.
#[cfg(feature = "plant")]
mod planted {
    use super::*;
    use ggtts_qwen3_tts::codec::Plant;

    fn with(plant: Plant) -> Outcome {
        check(&format!("planted {plant:?}"), |p, t, d| {
            Decoder::load_planted(p, t, d, plant)
        })
    }

    #[test]
    #[ignore = "needs GGTTS_CODEC, the published codec GGUF"]
    fn rope_normal_is_seen() {
        let o = with(Plant::RopeNormal);
        assert_eq!(o.differ.len(), o.total, "the planted rope must change every fixture");
    }

    #[test]
    #[ignore = "needs GGTTS_CODEC, the published codec GGUF"]
    fn snake_no_epsilon_is_seen() {
        let o = with(Plant::SnakeNoEpsilon);
        assert_eq!(o.differ.len(), o.total, "the planted snake must change every fixture");
    }

    #[test]
    #[ignore = "needs GGTTS_CODEC, the published codec GGUF"]
    fn window_off_by_one_is_seen() {
        let o = with(Plant::WindowOffByOne);
        assert!(!o.differ.is_empty(), "the planted window changed no fixture");
    }
}
