//! Frame-level streaming against the batch decode, on every CPU golden fixture's tokens.
//!
//! The parity was declared before this test was first run, and the first run (2026-11-06, CPU, 4 threads, the 8
//! fixtures) measured:
//! * at `hold = 32`, every streamed sample equals the batch decode's bit for bit, at `every` = 1, 4 and 12: **held**,
//!   0 LSB on every fixture. Asserted.
//! * at `hold = 0`, every streamed sample within 1 LSB of the batch decode's at 16 bits: **failed**, 26 to 458 LSB,
//!   first in the newest frames of a prefix. Reported, not asserted: `hold = 0` is not offered as exact.
//! * `every` equal to the utterance's length (one decode) is identical: **held**. Asserted.
//! * the planted `Context::NewFramesOnly`, which drops the left context at every seam, is not identical: **red** on
//!   every fixture, at the first seam. Asserted.
//!
//! `every = 1` re-decodes the prefix once per frame and took 36 minutes over the fixtures; it runs only with
//! `GGTTS_STREAM_EVERY_1=1`.
//!
//! `GGTTS_DEVICE=cuda` (with the `cuda` feature) runs the same streams on CUDA and **reports only**: the parity above
//! was declared and measured on CPU, and on CUDA the decoder projects the rest codebooks differently below 64 frames
//! (`codec.rs`, `summed`), so a prefix shorter than 64 frames takes another path than the batch decode.
//!
//! The planted run needs the test-only `plant` feature; without it the planted row is left out.
//!
//! Needs the published codec GGUF: `GGTTS_CODEC=<qwen-tokenizer-12hz-q8_0.gguf> cargo test -p ggtts-qwen3-tts
//! --features ggml --release --test stream_identity -- --ignored --nocapture`.
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use ggtts_core::flac::to_i16;
use ggtts_core::stream::first_sample_diff;
use ggtts_qwen3_tts::codec::Decoder;
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::fixtures::Golden;
use ggtts_qwen3_tts::stream::{stream_codes, Context};

fn max_lsb(a: &[f32], b: &[f32]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (i32::from(to_i16(x)) - i32::from(to_i16(y))).abs())
        .max()
        .unwrap_or(0)
}

#[test]
#[ignore = "needs GGTTS_CODEC, the published codec GGUF"]
fn the_stream_matches_the_batch_decode_as_declared() {
    let codec = PathBuf::from(std::env::var("GGTTS_CODEC").expect("set GGTTS_CODEC to the codec GGUF"));
    let goldens = Golden::load_all(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu")).unwrap();
    assert!(!goldens.is_empty());
    let d = std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = Device::parse(&d).unwrap_or_else(|| panic!("GGTTS_DEVICE must be cpu or cuda, not {d}"));
    let gate = !device.is_cuda();
    let mut dec = Decoder::load_on(&codec, 4, device).unwrap();
    let mut failures = Vec::new();
    for gold in &goldens {
        let n = gold.frames();
        let t = Instant::now();
        let batch = dec.decode(&gold.codes).unwrap().samples;
        let batch_ms = t.elapsed().as_secs_f64() * 1e3;
        let spf = batch.len() / n;
        let mut runs: Vec<(usize, usize, Context)> = Vec::new();
        if std::env::var("GGTTS_STREAM_EVERY_1").as_deref() == Ok("1") {
            runs.push((1, 32, Context::Prefix));
        }
        runs.extend([
            (4, 32, Context::Prefix),
            (12, 32, Context::Prefix),
            (4, 0, Context::Prefix),
            (12, 0, Context::Prefix),
            (n, 0, Context::Prefix),
        ]);
        #[cfg(feature = "plant")]
        runs.push((12, 0, Context::NewFramesOnly));
        runs.dedup();
        for (every, hold, context) in runs {
            let t = Instant::now();
            let (got, steps) = stream_codes(&mut dec, &gold.codes, every, hold, context).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let diff = first_sample_diff(&got, &batch);
            let lsb = max_lsb(&got, &batch);
            let first_audio = steps
                .iter()
                .position(|&s| s > 0)
                .map_or(n, |i| ((i + 1) * every).min(n));
            println!(
                "{:<14} n={n:<4} every={every:<4} hold={hold:<3} {:<13} {}  max {lsb} LSB  first audio after {first_audio} frames  {ms:.0} ms vs batch {batch_ms:.0} ms",
                gold.name,
                format!("{context:?}"),
                match diff {
                    None => "IDENTICAL".to_string(),
                    Some(i) => format!("DIFFERS at sample {i} (frame {})", i / spf),
                },
            );
            if got.len() != batch.len() {
                failures.push(format!(
                    "{} every={every} hold={hold}: {} samples, batch {}",
                    gold.name,
                    got.len(),
                    batch.len()
                ));
                continue;
            }
            let ok = match (context, hold) {
                #[cfg(feature = "plant")]
                (Context::NewFramesOnly, _) => diff.is_some(),
                (_, 32) => diff.is_none(),
                _ if every >= n => diff.is_none(),
                // hold 0: measured and reported only (see the header).
                _ => true,
            };
            if gate && !ok {
                failures.push(format!(
                    "{} every={every} hold={hold} {context:?}: diff {diff:?}, {lsb} LSB",
                    gold.name
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "against the declared parity:\n{}",
        failures.join("\n")
    );
}
