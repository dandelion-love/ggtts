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

/// An expected failure, pinned: on the CPU, the ternary arm's one-column product does not match the many-column one.
///
/// Measured on 2026-11-06 with the recorded prompt (`tools/kitten_ref.py logits`: 328 positions, a prefill of
/// 312 and then 16 single-token steps, 4 threads). The f32 arm agrees to rel L2 `1.3e-7` with every argmax equal. The
/// ternary arm (`Q4_0`, bf16 head) reaches cosine `0.9999995`, rel L2 `1.4e-3`, and argmax **15 of 16**, which
/// misses the bar. DESIGN.md records it; the cause is not yet found and is tracked in ggtts#21.
///
/// This test passes while that stays true. It fails, on purpose, when the behaviour changes either way: the gap
/// closes (unpin it and let the check above cover the recorded prompt), or it widens. Both deserve a look.
///
/// Needs `GGTTS_KITTEN_LM` (the GGUF converted from Kitten-TTS 2 at `c93cb7c`) and `GGTTS_KITTEN_REF` (the
/// `kitten_ref.py logits` directory, with `meta.json`). Skipped when either is unset. It runs on the CPU only: the
/// pinned numbers are the CPU's kernels.
#[test]
fn cpu_one_column_ternary_product_differs_pinned() {
    let (Some(path), Some(refdir)) = (
        std::env::var_os("GGTTS_KITTEN_LM"),
        std::env::var_os("GGTTS_KITTEN_REF"),
    ) else {
        eprintln!("skipped: set GGTTS_KITTEN_LM and GGTTS_KITTEN_REF");
        return;
    };
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&refdir).join("meta.json")).expect("meta.json"),
    )
    .expect("meta.json parses");
    let ids: Vec<u32> = meta["ids"]
        .as_array()
        .expect("meta.ids")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(
        ids.len(),
        328,
        "the pinned numbers are for the 328-position recorded prompt"
    );
    let be = Backend::new(Device::Cpu, 4).expect("backend");
    let gguf = Gguf::open(std::path::Path::new(&path)).expect("open the GGUF");

    let f32m = Model::load(&gguf, Weights::F32, be.clone()).expect("load f32");
    let r = Session::step_check(&f32m, &ids, 312).expect("f32 arm");
    eprintln!("f32: {r:?}");
    assert!(
        r.passes() && r.max_rel_l2 < 1e-5,
        "the f32 arm should agree to f32 rounding: {r:?}"
    );
    drop(f32m);

    let tern = Model::load(&gguf, Weights::Ternary, be).expect("load ternary");
    let r = Session::step_check(&tern, &ids, 312).expect("ternary arm");
    eprintln!("ternary: {r:?}");
    assert_eq!(r.positions, 16);
    assert_eq!(r.argmax_equal, 15, "pinned 15/16 changed: {r:?}");
    assert!(!r.passes(), "the ternary arm now passes; unpin this test: {r:?}");
    assert!(r.min_cosine >= 0.999999, "pinned cosine 0.9999995 got worse: {r:?}");
    assert!(
        (1.0e-3..2.0e-3).contains(&r.max_rel_l2),
        "pinned rel L2 1.4e-3 moved: {r:?}"
    );
}
