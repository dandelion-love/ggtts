//! The template hook's self-test: a planted early stop is seen, the no-op reads as no change, and the fake engine
//! renders the same tokens and audio with and without the hook.

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::hooks::{Hooks, Step, TensorHook};
use ggtts_core::pipeline::{plan_story, render_story, render_story_hooked, RenderOptions};
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;
use ggtts_instrument_tensor_template::EarlyStop;

fn paragraphs() -> Vec<String> {
    vec![
        "今日天氣好好，我哋去公園行下。".to_string(),
        "聽日再傾，記得早啲瞓。".to_string(),
    ]
}

/// Feed one draw of `n` frames whose `p_stop` is `p(frame)`; the summary.
fn feed(n: u32, p: impl Fn(u32) -> f64) -> ggtts_instrument_tensor_template::Summary {
    let (mut hook, out) = EarlyStop::new(0.5);
    for f in 0..n {
        hook.step(&Step {
            chunk: 0,
            draw: 0,
            seed: 1,
            frame: f,
            token: 0,
            p_stop: Some(p(f)),
            p_stop_semantics: Some("fake-synthetic/1"),
            logits: None,
        });
    }
    hook.finish().unwrap();
    let s = out.borrow().clone();
    s
}

fn rising(n: u32) -> impl Fn(u32) -> f64 {
    move |f| 0.001 + 0.989 * ((f + 1) as f64 / n as f64).powi(8)
}

#[test]
fn a_planted_early_stop_is_seen() {
    let clean = feed(20, rising(20));
    let planted = feed(20, |f| if f == 1 { 0.95 } else { rising(20)(f) });
    assert_eq!(planted.early_stop_frames, clean.early_stop_frames + 1);
    assert_eq!(planted.p_stop_early_max, Some(0.95));
}

#[test]
fn a_clean_draw_reads_no_early_stop() {
    let s = feed(20, rising(20));
    assert_eq!(s.early_stop_frames, 0);
    assert!(s.p_stop_early_max.unwrap() < 0.01);
}

/// Render the story with the hook; the read and the summary.
fn hooked() -> (Vec<f32>, ggtts_instrument_tensor_template::Summary) {
    let mut e = FakeEngine::default();
    let chunks = plan_story(e.profile(), &paragraphs()).unwrap();
    let opts = RenderOptions::for_profile(e.profile());
    let (hook, out) = EarlyStop::new(0.5);
    let mut h = Hooks::new();
    h.add(Box::new(hook));
    let read = render_story_hooked(&mut e, "template", &chunks, &opts, &mut NullEvents, &mut h).unwrap();
    h.finish();
    assert!(h.failures().is_empty(), "{:?}", h.failures());
    let s = out.borrow().clone();
    (read.samples, s)
}

#[test]
fn the_same_seed_renders_identically_with_and_without_the_hook() {
    let mut e = FakeEngine::default();
    let chunks = plan_story(e.profile(), &paragraphs()).unwrap();
    let opts = RenderOptions::for_profile(e.profile());
    let plain = render_story(&mut e, "template", &chunks, &opts).unwrap();
    let (samples, summary) = hooked();
    assert_eq!(plain.samples, samples, "the hook changed the audio");
    assert!(summary.draws > 0, "the hook saw no draw");

    // Tokens, request by request.
    for (i, c) in chunks.iter().enumerate() {
        let req = ChunkRequest {
            text: &c.text,
            seed: 1000 + i as u64,
            chunk: i as u32,
            draw: 0,
        };
        let (mut a, mut b) = (FakeEngine::default(), FakeEngine::default());
        let (hook, _) = EarlyStop::new(0.5);
        let mut h = Hooks::new();
        h.add(Box::new(hook));
        assert_eq!(
            a.generate(&req).unwrap(),
            b.generate_hooked(&req, &mut NullEvents, &mut h).unwrap(),
            "chunk {i}: the hook changed a token"
        );
    }
}

#[test]
fn two_renders_read_the_same() {
    assert_eq!(hooked().1, hooked().1);
}
