//! The restricted head (`Model::head_rows`) against the full head, on the same forward.
//!
//! Generation computes the head for the ids a step can sample only. Inside that range its logits must be the full
//! head's, bit for bit (the same rows through the same kernel), and every id outside it must read `f32::MIN`. Whether
//! that changes a draw is a separate question, measured by `examples/head_rows_check.rs` on a test set.
//!
//! Needs a converted Kitten-TTS 2 LM GGUF, which is never in this repository: set `GGTTS_KITTEN_LM`. Skipped when it
//! is not set.
#![cfg(feature = "ggml")]

use ggtts_kitten::backend::{Backend, Device};
use ggtts_kitten::gguf::Gguf;
use ggtts_kitten::model::{Model, Session, Weights};
use ggtts_kitten::parity;
use ggtts_kitten::prompt::TokenMap;

#[test]
fn the_restricted_head_is_the_full_head_inside_its_rows() {
    let Some(path) = std::env::var_os("GGTTS_KITTEN_LM") else {
        eprintln!("skipped: set GGTTS_KITTEN_LM to a converted Kitten-TTS 2 LM GGUF");
        return;
    };
    let be = Backend::new(Device::Cpu, 4).expect("backend");
    let gguf = Gguf::open(std::path::Path::new(&path)).expect("open the GGUF");
    let mut model = Model::load(&gguf, Weights::Ternary, be).expect("load");
    let tm = TokenMap::from_meta(&model.meta).expect("token map");
    let rows = tm.speech_end.min(tm.stop) as usize..(tm.audio_base + tm.audio_count) as usize;
    let n = model.hp.n_embd;
    let ids = parity::pseudo_ids(24, model.hp.n_vocab, 99);
    let mut embd = Vec::new();
    model.embed_tokens(&ids, &mut embd);

    // The same prefill, then three one-token steps, with each head.
    let run = |model: &Model| -> Vec<Vec<f32>> {
        let mut s = Session::new(model, 64).expect("session");
        let mut out = vec![s.forward(&embd[..20 * n], 1, false).expect("prefill").logits];
        for i in 20..23 {
            out.push(s.forward(&embd[i * n..(i + 1) * n], 1, false).expect("step").logits);
        }
        out
    };
    model.head_rows = None;
    let full = run(&model);
    model.head_rows = Some(rows.clone());
    let restricted = run(&model);

    for (step, (f, r)) in full.iter().zip(&restricted).enumerate() {
        assert_eq!(
            f.len(),
            r.len(),
            "step {step}: the restricted logits keep the vocabulary's length"
        );
        for (i, (a, b)) in f.iter().zip(r).enumerate() {
            if rows.contains(&i) {
                assert_eq!(a.to_bits(), b.to_bits(), "step {step}, id {i}: {a} against {b}");
            } else {
                assert_eq!(*b, f32::MIN, "step {step}, id {i} is outside the rows");
            }
        }
    }
}
