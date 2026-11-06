//! The single-token forward, which generation runs once per token, against the batched forward on the same positions.
//!
//! The parity check runs one batched forward, so it never reaches the path every generated token takes: one
//! column, the KV cache read back, and on CUDA different kernels and fusions. A failure there was first seen as
//! non-finite logits in a CUDA render. This holds the two paths to the bar in DESIGN.md (level 2, "single-token
//! path"), and also requires the logits to be finite, which the forward itself now refuses otherwise.
//!
//! Needs a converted Kitten-TTS 2 LM GGUF, which is never in this repository: set `GGTTS_KITTEN_LM`. Skipped when it
//! is not set. `GGTTS_DEVICE=cuda` runs it on CUDA (with the `cuda` feature).
#![cfg(feature = "ggml")]

use ggtts_kitten::backend::{Backend, Device};
use ggtts_kitten::gguf::Gguf;
use ggtts_kitten::model::{Model, Session, Weights};
use ggtts_kitten::parity;

#[test]
fn single_token_steps_match_the_batched_forward() {
    let Some(path) = std::env::var_os("GGTTS_KITTEN_LM") else {
        eprintln!("skipped: set GGTTS_KITTEN_LM to a converted Kitten-TTS 2 LM GGUF");
        return;
    };
    let device = Device::parse(&std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into())).expect("GGTTS_DEVICE");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(4));
    let be = Backend::new(device, threads).expect("backend");
    let gguf = Gguf::open(std::path::Path::new(&path)).expect("open the GGUF");
    for weights in [Weights::Ternary, Weights::F32] {
        let model = Model::load(&gguf, weights, be.clone()).expect("load");
        let ids = parity::pseudo_ids(48, model.hp.n_vocab, 1234);
        let r = Session::step_check(&model, &ids, 40).expect("both paths run and give finite logits");
        eprintln!("{weights:?} on {device:?}: {r:?}");
        assert!(
            r.passes(),
            "{weights:?}: {r:?} misses the bar (cosine >= {})",
            parity::STEP_MIN_COSINE
        );
    }
}
