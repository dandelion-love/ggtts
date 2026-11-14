//! The options value's render section gives Kitten-TTS 2's profile defaults, as `RenderOptions::for_profile` does
//! (`ggtts_core::options`; the other engines' profiles are checked in ggtts-core's own tests).
#![cfg(feature = "ggml")]

use ggtts_core::options::Options;
use ggtts_core::pipeline::RenderOptions;

#[test]
fn the_default_render_options_are_kittens_profile() {
    let p = ggtts_kitten::engine::PROFILE;
    let got = Options::default().render_options(&p).unwrap();
    let want = RenderOptions::for_profile(&p);
    assert_eq!(got.redraw, want.redraw);
    assert_eq!(got.seed_salt, want.seed_salt);
    assert_eq!(got.edge_s.to_bits(), want.edge_s.to_bits());
    assert_eq!(got.measure_timings, want.measure_timings);
    assert_eq!(got.instruction, want.instruction);
    assert_eq!(got.steer, want.steer);
    let set = Options::from_json(r#"{"render": {"redraw": 7, "edge_s": 0.5}}"#)
        .unwrap()
        .render_options(&p)
        .unwrap();
    assert_eq!((set.redraw, set.edge_s), (7, 0.5));
}
