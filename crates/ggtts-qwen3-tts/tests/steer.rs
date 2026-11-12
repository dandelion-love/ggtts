//! Steering on a real engine: a vector added to the talker's hidden state at the output of chosen layers
//! (`docs/steering.md`).
//!
//! Most checks run on the tiny fixture (`fixtures/tiny/`: the published v0.0.1 talker cut to one talker layer), so
//! the only layer to steer is layer 0. A model that small cannot speak; these checks are about tokens and hidden
//! states, never audio quality. The checks named `full_model` run on the published talker itself, at a middle layer.
//!
//! * Absent is untouched: an unsteered render gives the tokens and samples recorded before steering existed, and its
//!   graph has no steering node.
//! * Scale zero gives the unsteered tokens.
//! * A non-zero vector changes the tokens, the same request twice gives the same tokens, and the planted missing add
//!   node (`Plant::SteerAddMissing`) turns that check red.
//! * The steering's effect on the final hidden state agrees with a PyTorch forward-hook reference
//!   (`scripts/steer_reference.py`) within twice the engine's own error on the unsteered prefill: on the tiny talker
//!   at layer 0 (`fixtures/steer/`), and on the full talker at layer 14 of 28 (`fixtures/steer/full/`), where the
//!   planted vector one layer late, the planted vector at an extra layer and the planted missing add each fail it.
//! * Steering the talker cannot take is refused before anything is generated.
//! * The trace-identity check passes on steered requests, and its planted instrument still turns it red.
//! * A layer subset of a file gives the tokens of a file holding only those layers.
//!
//! ```text
//! eval "$(scripts/tiny_fixture.sh)"
//! cargo test --release -p ggtts-qwen3-tts --features ggml,plant --test steer -- --ignored --nocapture
//! GGTTS_TALKER=<published v0.0.1 talker> cargo test --release -p ggtts-qwen3-tts --features ggml,plant --test steer \
//!     full_model -- --ignored --nocapture
//! ```
#![cfg(feature = "ggml")]

use std::path::{Path, PathBuf};

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::steer::{read_f32_tensors, write_safetensors, SteerFile, Steering};
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};
use ggtts_qwen3_tts::talker::Talker;
use sha2::{Digest, Sha256};

/// Frames per request, as the tiny model's other checks use.
const FRAMES: i32 = 24;
const SEED: u64 = 20260905;

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `KEY=VALUE` lines from `fixtures/tiny/SOURCE`.
fn source(key: &str) -> String {
    let text = std::fs::read_to_string(here().join("fixtures/tiny/SOURCE")).unwrap();
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("fixtures/tiny/SOURCE has no {key}"))
        .trim()
        .to_string()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A file named by `var`, refused unless its sha256 is `want`: a check on the wrong file must fail, not pass.
fn pinned(var: &str, want: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} (see scripts/tiny_fixture.sh)")));
    let got = hex(&std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())));
    assert_eq!(got, want, "{var}={} is not the pinned file", p.display());
    p
}

fn tiny() -> PathBuf {
    pinned("GGTTS_TINY_TALKER", &source("CUT_SHA256"))
}

fn codec() -> PathBuf {
    pinned("GGTTS_CODEC", &source("CODEC_SHA256"))
}

/// The published v0.0.1 talker the tiny fixture is cut from.
fn full() -> PathBuf {
    pinned("GGTTS_TALKER", &source("TALKER_SHA256"))
}

/// The CPU fixtures' texts.
fn texts() -> Vec<(String, String)> {
    std::fs::read_to_string(here().join("fixtures/cpu/texts.tsv"))
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(n, t)| (n.to_string(), t.to_string()))
        .collect()
}

/// A repeatable vector of `n` values in `[-amp, amp]`.
fn vector(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// A steering file holding `layers`, read back as the CLI reads one.
fn file(layers: &[(u32, Vec<f32>)]) -> SteerFile {
    let names: Vec<String> = layers.iter().map(|(l, _)| format!("talker.layer.{l}")).collect();
    let tensors: Vec<(&str, &[f32])> = names
        .iter()
        .zip(layers)
        .map(|(n, (_, v))| (n.as_str(), v.as_slice()))
        .collect();
    SteerFile::parse(&write_safetensors(&tensors)).unwrap()
}

fn engine(talker: &Path) -> Qwen3Tts {
    let voice = Voice {
        max_frames: FRAMES,
        ..Default::default()
    };
    Qwen3Tts::load(talker, &codec(), 4, voice).unwrap()
}

fn request<'a>(text: &'a str, steer: Option<&'a Steering>) -> ChunkRequest<'a> {
    ChunkRequest {
        text,
        seed: SEED,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer,
    }
}

fn tokens(e: &mut Qwen3Tts, text: &str, steer: Option<&Steering>) -> Vec<u16> {
    e.generate(&request(text, steer)).unwrap().codes.frames
}

/// The tiny fixture's unsteered tokens and samples for each text, as the build before steering rendered them
/// (`FRAMES` frames, seed `SEED`, CPU, 4 threads, x86_64): the sha256 of the little-endian u16 tokens and of the
/// little-endian f32 samples.
const BEFORE: [(&str, &str, &str); 8] = [
    (
        "greeting",
        "a5743d9b5be5540c6212966b775692283859c7951922dc3b203043b728a348f7",
        "7aa295fa860714c7f6934b014272fce3b063cfafbd8d1b73998f0d2633cc436d",
    ),
    (
        "park",
        "0c61badd9a8b5ab6644707a08af561a72bb91d9510ca9d709093a2019213977f",
        "6bbf2822f19876cbafdd31ff05e0366a2853f2193080d74a72b6b180a321fb19",
    ),
    (
        "cafe",
        "58bdc56195dd2999aac0f1e0838bd562cfdd95e789eeb9c5d7a7f0452e4eb07c",
        "d3bd900cd27019872830a3eab86f4530cc4cbc8dc959f824f7f0162d2875d13b",
    ),
    (
        "dinner",
        "8e23ddd35cfad4c7f8f672e14357fbd708f7c9d77673fcccc640b424a695c7d7",
        "12f3d9c389d08f4a8125a9c2fd2c3ce7cc2f5b07d5a5156f0f12bdec6913fa53",
    ),
    (
        "beach",
        "c4c8731e23da8f805df518636428115cd369dbefd67812a667dffadf2d6407ae",
        "2361f720368d27ec770dea6521496a206e54527211f48db93ffac2c393149f95",
    ),
    (
        "first-meeting",
        "fe14f565407b5808561e8dcf4dbfe139b39941cf5374912ba001668476aa7949",
        "f847d1c32b215e532b3786f6da33857995123cc5071d20e50766a4c2e6f8601a",
    ),
    (
        "rainy-day",
        "854509b8ec5ec54a43e3df1afa8b57240734623326d956c760e7f2b817f40991",
        "14e60026890adaa26349971306c6785535aedabc08c148a50294dee0c6ac0747",
    ),
    (
        "bus",
        "8fe0abccaac2a4ca0557e060f0d60560ab61fbb5d794c22b49b1b810d97d3ba2",
        "b7e7c0a9dbed47b1cc3874c3c65e46586edc54f4ea7d57adc7cc3a522301f842",
    ),
];

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn an_unsteered_render_is_the_render_before_steering() {
    let mut e = engine(&tiny());
    let mut failures = Vec::new();
    for (name, text) in texts() {
        let g = e.generate(&request(&text, None)).unwrap();
        let pcm = e.decode(&g.codes).unwrap();
        let t: Vec<u8> = g.codes.frames.iter().flat_map(|c| c.to_le_bytes()).collect();
        let s: Vec<u8> = pcm.samples.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (t, s) = (hex(&t), hex(&s));
        let want = BEFORE.iter().find(|(n, _, _)| *n == name).expect("a recorded text");
        let same = t == want.1 && s == want.2;
        println!(
            "{name:<14} tokens {}, samples {}",
            if t == want.1 { "as before" } else { "DIFFER" },
            if s == want.2 { "as before" } else { "DIFFER" }
        );
        if !same {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "the unsteered render changed on {failures:?}");
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TINY_TALKER (scripts/tiny_fixture.sh)"]
fn an_unsteered_graph_has_no_steering_node() {
    let mut t = Talker::load(&tiny(), 4).unwrap();
    let h = t.config.hidden_size as usize;
    let s = file(&[(0, vector(1, h, 1.0))]).select(None, 1.0).unwrap();
    let zero = file(&[(0, vector(1, h, 1.0))]).select(None, 0.0).unwrap();
    let plain = t.prefill_graph_nodes(12, None).unwrap();
    let steered = t.prefill_graph_nodes(12, Some(&s)).unwrap();
    let zeroed = t.prefill_graph_nodes(12, Some(&zero)).unwrap();
    println!("prefill graph nodes: unsteered {plain}, one steered layer {steered}, at scale 0 {zeroed}");
    assert_eq!(steered, plain + 1, "one steered layer must add exactly one node");
    assert_eq!(zeroed, steered, "scale 0 is the same graph as any other scale");
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn scale_zero_is_no_steering() {
    let mut e = engine(&tiny());
    let h = 2048;
    let s = file(&[(0, vector(7, h, 2.0))]).select(None, 0.0).unwrap();
    for (name, text) in texts() {
        let plain = tokens(&mut e, &text, None);
        let zero = tokens(&mut e, &text, Some(&s));
        println!(
            "{name:<14} scale 0: {}",
            if zero == plain { "identical" } else { "DIFFERS" }
        );
        assert_eq!(zero, plain, "{name}: scale 0 changed the tokens");
    }
}

/// Steering changes the tokens of every text, and the same request twice gives the same tokens. `changes` is false
/// for a text whose tokens steering left as they were.
fn steering_changes_tokens(e: &mut Qwen3Tts, s: &Steering, n_texts: usize) -> Vec<(String, bool, bool)> {
    texts()
        .into_iter()
        .take(n_texts)
        .map(|(name, text)| {
            let plain = tokens(e, &text, None);
            let a = tokens(e, &text, Some(s));
            let b = tokens(e, &text, Some(s));
            (name, a != plain, a == b)
        })
        .collect()
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn steering_changes_the_tokens_and_repeats() {
    let mut e = engine(&tiny());
    let s = file(&[(0, vector(3, 2048, 1.0))]).select(None, 1.0).unwrap();
    let rows = steering_changes_tokens(&mut e, &s, 8);
    for (name, changes, repeats) in &rows {
        println!("{name:<14} steered: changes tokens {changes}, same twice {repeats}");
    }
    assert!(
        rows.iter().all(|r| r.1),
        "steering left some tokens as they were: {rows:?}"
    );
    assert!(rows.iter().all(|r| r.2), "steering is not repeatable: {rows:?}");
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn a_missing_add_node_turns_the_steering_check_red() {
    use ggtts_qwen3_tts::talker::Plant;
    let voice = Voice {
        max_frames: FRAMES,
        ..Default::default()
    };
    let mut planted = Qwen3Tts::load_planted(&tiny(), &codec(), 4, voice, Plant::SteerAddMissing).unwrap();
    let s = file(&[(0, vector(3, 2048, 1.0))]).select(None, 1.0).unwrap();
    let rows = steering_changes_tokens(&mut planted, &s, 3);
    for (name, changes, _) in &rows {
        println!("{name:<14} planted missing add: steering changes tokens {changes}");
    }
    assert!(
        rows.iter().all(|r| !r.1),
        "with the add node left out, steering must change nothing; the check could not see a missing add: {rows:?}"
    );
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn steering_the_talker_cannot_take_is_refused_before_generating() {
    let mut e = engine(&tiny());
    let cases = [
        (
            "a vector of the wrong length",
            file(&[(0, vector(1, 2047, 1.0))]),
            "hidden size is 2048",
        ),
        (
            "a layer out of range",
            file(&[(1, vector(1, 2048, 1.0))]),
            "out of range",
        ),
    ];
    for (what, f, want) in cases {
        let s = f.select(None, 1.0).unwrap();
        let e1 = e.check_steer(&s).unwrap_err();
        let e2 = e.generate(&request("你好。", Some(&s))).unwrap_err();
        println!("{what}: {e2}");
        assert!(e1.0.contains(want) && e2.0.contains(want), "{what}: {e1} / {e2}");
    }
    // The engine still renders afterwards: a refusal leaves nothing half-built.
    assert!(!tokens(&mut e, "你好。", None).is_empty());
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn trace_identity_holds_under_steering_and_its_plant_is_red() {
    use ggtts_core::trace::check_trace_identity;
    use ggtts_qwen3_tts::talker::Plant;
    let (talker, codec) = (tiny(), codec());
    let s = file(&[(0, vector(3, 2048, 1.0))]).select(None, 1.0).unwrap();
    let all = texts();
    let reqs: Vec<ChunkRequest<'_>> = all
        .iter()
        .take(3)
        .enumerate()
        .map(|(i, (_, t))| ChunkRequest {
            chunk: i as u32,
            ..request(t, Some(&s))
        })
        .collect();
    let voice = || Voice {
        max_frames: FRAMES,
        ..Default::default()
    };
    let mut make = || Box::new(Qwen3Tts::load(&talker, &codec, 4, voice()).unwrap()) as Box<dyn Engine>;
    let quiet = check_trace_identity(&mut make, &reqs, 1).unwrap();
    println!("steered, traced against untraced: {quiet:?}");
    assert!(quiet.is_none(), "the trace changed a steered render: {quiet:?}");
    let mut planted = || {
        Box::new(Qwen3Tts::load_planted(&talker, &codec, 4, voice(), Plant::StepHookDraw).unwrap()) as Box<dyn Engine>
    };
    let red = check_trace_identity(&mut planted, &reqs, 1).unwrap();
    println!("steered, with the planted drawing instrument: {red:?}");
    assert!(red.is_some(), "the planted instrument was not seen on a steered render");
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER and GGTTS_CODEC (scripts/tiny_fixture.sh)"]
fn a_layer_subset_is_the_file_of_those_layers() {
    let mut e = engine(&tiny());
    let v = vector(3, 2048, 1.0);
    // Layer 9 is past the tiny talker's one layer: selecting it would be refused, so the subset must drop it.
    let both = file(&[(0, v.clone()), (9, vector(4, 2048, 1.0))]);
    let subset = both.select(Some(&[0]), 1.0).unwrap();
    let only = file(&[(0, v)]).select(None, 1.0).unwrap();
    assert!(e.check_steer(&both.select(None, 1.0).unwrap()).is_err());
    for (name, text) in texts().into_iter().take(4) {
        let a = tokens(&mut e, &text, Some(&subset));
        let b = tokens(&mut e, &text, Some(&only));
        println!(
            "{name:<14} layer subset against a file of that layer: {}",
            if a == b { "identical" } else { "DIFFER" }
        );
        assert_eq!(a, b, "{name}");
    }
}

/// A reference fixture (`scripts/steer_reference.py`) in `fixtures/<dir>`.
struct Reference {
    meta: serde_json::Value,
    input: Vec<f32>,
    plain: Vec<f32>,
    steered: Vec<f32>,
    steering: Steering,
}

fn reference(dir: &str) -> Reference {
    let dir = here().join("fixtures").join(dir);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let mut t = read_f32_tensors(&std::fs::read(dir.join("reference.safetensors")).unwrap(), |_| Ok(())).unwrap();
    let mut take = |k: &str| t.remove(k).unwrap_or_else(|| panic!("the reference lacks {k}")).1;
    let (input, plain, steered) = (take("input"), take("hidden.plain"), take("hidden.steered"));
    let steering = SteerFile::read(&dir.join("steer.safetensors"))
        .unwrap()
        .select(None, meta["scale"].as_f64().unwrap())
        .unwrap();
    Reference {
        meta,
        input,
        plain,
        steered,
        steering,
    }
}

/// The root-mean-square difference over the root-mean-square of `b`.
fn relative_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    let n: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (d / n).sqrt()
}

fn minus(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The reference check, on talker `t` and the fixture in `fixtures/<dir>`, whose talker sha256 must be `sha`.
///
/// What it compares is the steering's effect: the engine's steered hidden state minus its unsteered one, against the
/// reference's steered minus unsteered, as a relative root-mean-square difference. Taking the difference cancels most
/// of the error the engine already has unsteered (the q8_0 activations and the f16 KV cache), which grows with depth:
/// about 1% relative on the one-layer talker, about 16% after 28 layers. The tolerance is twice that unsteered error.
/// A missing add gives an effect error of 1 (no effect at all), so the tolerance must stay under 0.5 for the check
/// to see one. `Err` names what failed; the tests assert on it.
fn reference_check(t: &mut Talker, dir: &str, sha: &str) -> Result<String, String> {
    let r = reference(dir);
    assert_eq!(
        r.meta["talker_sha256"].as_str(),
        Some(sha),
        "the reference in {dir} was recorded on another talker"
    );
    let n = r.meta["tokens"].as_u64().unwrap() as usize;
    let got_plain = t.prefill_hidden(&r.input, n, None).unwrap();
    let got_steered = t.prefill_hidden(&r.input, n, Some(&r.steering)).unwrap();
    let unsteered = relative_rms(&got_plain, &r.plain);
    let steered = relative_rms(&got_steered, &r.steered);
    let effect = relative_rms(&minus(&got_steered, &got_plain), &minus(&r.steered, &r.plain));
    let tolerance = 2.0 * unsteered;
    let summary = format!(
        "{dir} (layers {:?}): relative rms |engine - reference| unsteered {unsteered:.5}, steered {steered:.5}; \
         steering effect error {effect:.5} (tolerance {tolerance:.5})",
        r.steering.layer_indices(),
    );
    if tolerance >= 0.5 {
        return Err(format!(
            "the unsteered error is too large to check steering against: {summary}"
        ));
    }
    if effect > tolerance {
        return Err(format!("the steering effect is off the reference: {summary}"));
    }
    Ok(summary)
}

#[test]
#[ignore = "needs GGTTS_TINY_TALKER (scripts/tiny_fixture.sh)"]
fn the_steered_hidden_state_matches_the_pytorch_reference() {
    let mut t = Talker::load(&tiny(), 4).unwrap();
    let summary = reference_check(&mut t, "steer", &source("CUT_SHA256")).unwrap_or_else(|e| panic!("{e}"));
    println!("{summary}");
}

#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TINY_TALKER (scripts/tiny_fixture.sh)"]
fn a_missing_add_node_fails_the_reference_check() {
    use ggtts_qwen3_tts::talker::Plant;
    let mut t = Talker::load_planted(&tiny(), 4, Default::default(), Plant::SteerAddMissing).unwrap();
    let e = reference_check(&mut t, "steer", &source("CUT_SHA256"))
        .expect_err("the reference check passed a talker with no add node");
    println!("planted missing add: red: {e}");
}

/// The reference check on the full talker, steering layer 14 of 28: a layer that is neither the first nor the last,
/// so a vector one layer early, one layer late or at an extra layer changes the hidden state.
#[test]
#[ignore = "needs GGTTS_TALKER (the published v0.0.1 talker)"]
fn full_model_the_steered_hidden_state_at_a_middle_layer_matches_the_pytorch_reference() {
    let mut t = Talker::load(&full(), 4).unwrap();
    let summary = reference_check(&mut t, "steer/full", &source("TALKER_SHA256")).unwrap_or_else(|e| panic!("{e}"));
    println!("{summary}");
}

/// Each planted placement defect fails the full talker's reference check: the vector one layer late, the vector at
/// its layer and the next, and no add at all.
#[test]
#[cfg(feature = "plant")]
#[ignore = "needs GGTTS_TALKER (the published v0.0.1 talker)"]
fn full_model_a_vector_at_a_wrong_or_extra_layer_fails_the_reference_check() {
    use ggtts_qwen3_tts::cpu::Device;
    use ggtts_qwen3_tts::talker::Plant;
    let path = full();
    for plant in [Plant::SteerLayerShift, Plant::SteerNextLayerToo, Plant::SteerAddMissing] {
        let mut t = Talker::load_planted(&path, 4, Device::Cpu, plant).unwrap();
        let e = reference_check(&mut t, "steer/full", &source("TALKER_SHA256"))
            .expect_err(&format!("the reference check passed the planted {plant:?}"));
        println!("planted {plant:?}: red: {e}");
    }
}

#[test]
#[ignore = "needs GGTTS_TALKER (the published v0.0.1 talker) and GGTTS_CODEC"]
fn full_model_steering_a_middle_layer_changes_the_tokens_and_repeats() {
    let mut e = engine(&full());
    let s = file(&[(14, vector(3, 2048, 1.0))]).select(None, 1.0).unwrap();
    let rows = steering_changes_tokens(&mut e, &s, 3);
    for (name, changes, repeats) in &rows {
        println!("{name:<14} layer 14 steered: changes tokens {changes}, same twice {repeats}");
    }
    assert!(rows.iter().all(|r| r.1 && r.2), "{rows:?}");
}

#[test]
#[ignore = "needs GGTTS_TALKER (the published v0.0.1 talker) and GGTTS_CODEC"]
fn full_model_a_layer_subset_is_the_file_of_those_layers() {
    let mut e = engine(&full());
    let (a, b, c) = (vector(1, 2048, 1.0), vector(2, 2048, 1.0), vector(3, 2048, 1.0));
    let three = file(&[(10, a), (14, b.clone()), (18, c.clone())]);
    let subset = three.select(Some(&[14, 18]), 1.0).unwrap();
    let only = file(&[(14, b), (18, c)]).select(None, 1.0).unwrap();
    let all = three.select(None, 1.0).unwrap();
    for (name, text) in texts().into_iter().take(2) {
        let x = tokens(&mut e, &text, Some(&subset));
        let y = tokens(&mut e, &text, Some(&only));
        let z = tokens(&mut e, &text, Some(&all));
        println!(
            "{name:<14} layers 14,18 of three against a file of 14,18: {}; all three differ: {}",
            if x == y { "identical" } else { "DIFFER" },
            x != z
        );
        assert_eq!(x, y, "{name}");
        assert_ne!(
            x, z,
            "{name}: the third layer changed nothing, so the subset check cannot fail"
        );
    }
}

/// CUDA against CPU, on the same steered request. CPU and CUDA already differ unsteered on this stack (see
/// `docs/instruments/stop-trace-identity-2026-11-07.md`), so this states both differences and their size rather
/// than asserting identity; it asserts that steering takes effect on CUDA and repeats there.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs a CUDA device, GGTTS_TINY_TALKER and GGTTS_CODEC"]
fn cuda_and_cpu_steered_tokens_are_stated() {
    use ggtts_qwen3_tts::cpu::Device;
    let load = |device| {
        let voice = Voice {
            max_frames: FRAMES,
            device,
            ..Default::default()
        };
        Qwen3Tts::load(&tiny(), &codec(), 4, voice).unwrap()
    };
    let (mut cpu, mut cuda) = (load(Device::Cpu), load(Device::Cuda));
    let s = file(&[(0, vector(3, 2048, 1.0))]).select(None, 1.0).unwrap();
    let differing = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    for (name, text) in texts().into_iter().take(3) {
        let (cp, cs) = (tokens(&mut cpu, &text, None), tokens(&mut cpu, &text, Some(&s)));
        let (gp, gs) = (tokens(&mut cuda, &text, None), tokens(&mut cuda, &text, Some(&s)));
        let gs2 = tokens(&mut cuda, &text, Some(&s));
        println!(
            "{name:<14} CUDA against CPU: unsteered {} of {} codes differ, steered {} of {} differ",
            differing(&cp, &gp),
            cp.len(),
            differing(&cs, &gs),
            cs.len()
        );
        assert_ne!(gs, gp, "{name}: steering changed nothing on CUDA");
        assert_eq!(gs, gs2, "{name}: steering is not repeatable on CUDA");
    }
}
