//! #350: one chunk at an exact seed from a loaded engine. A render document that gives `render.seed` renders its text
//! as one chunk at that seed, drawn once and kept whole, as `ggtts-render --text TEXT --seed S` does; the samples are
//! the same, so the FLAC `ggtts-render` writes is the FLAC of the binding's samples, byte for byte. Every front end
//! builds it with `ggtts_core::pipeline::exact`.
//!
//! The fake engine runs in CI. On a real model: `GGTTS_TINY_TALKER` and `GGTTS_CODEC` (`scripts/tiny_fixture.sh`) and
//! `cargo test -p ggtts-cli --features ggml --release --test exact_seed -- --ignored --nocapture`. With
//! `GGTTS_TALKER` instead of `GGTTS_TINY_TALKER`, the full model; `GGTTS_DEVICE=cuda` runs both on CUDA.

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ggtts_ffi::{ggtts_free, ggtts_last_error_key, ggtts_load_opts, ggtts_pcm_free, ggtts_render_opts, GGTTS_OK};

const BIN: &str = env!("CARGO_BIN_EXE_ggtts-render");

/// Short Cantonese lines from the committed fixtures (`crates/ggtts-qwen3-tts/fixtures/cpu/texts.tsv`).
const TEXTS: [&str; 3] = [
    "多謝晒",
    "今日天氣好好，我哋去公園行下",
    "呢間茶餐廳嘅奶茶好滑，菠蘿包又脆",
];

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-exact-seed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(ggtts_ffi::ggtts_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn last_key() -> Option<String> {
    let p = ggtts_last_error_key();
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// A handle loaded from the load document `load`.
fn load(load: &str) -> u64 {
    let doc = CString::new(load).unwrap();
    let mut h = 0u64;
    let code = unsafe { ggtts_load_opts(doc.as_ptr(), &mut h) };
    assert_eq!(code, GGTTS_OK, "{load}: {}", last_error());
    h
}

/// `text` rendered on `handle` with the render document `doc`: the samples and rate, or the code and refused key.
fn render(handle: u64, text: &str, doc: &str) -> Result<(Vec<f32>, u32), (i32, Option<String>)> {
    let (text, doc) = (CString::new(text).unwrap(), CString::new(doc).unwrap());
    let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
    let code = unsafe {
        ggtts_render_opts(
            handle,
            text.as_ptr(),
            doc.as_ptr(),
            None,
            std::ptr::null_mut(),
            &mut p,
            &mut n,
            &mut r,
        )
    };
    if code != GGTTS_OK {
        return Err((code, last_key()));
    }
    let samples = unsafe { std::slice::from_raw_parts(p, n) }.to_vec();
    unsafe { ggtts_pcm_free(p, n) };
    Ok((samples, r))
}

/// `ggtts-render --text text --seed seed --out` with the extra arguments: the FLAC's bytes.
fn cli(dir: &Path, extra: &[&str], text: &str, seed: u64) -> Vec<u8> {
    let out = dir.join(format!("cli-{seed}.flac"));
    let o = Command::new(BIN)
        .args(extra)
        .args([
            "--text",
            text,
            "--seed",
            &seed.to_string(),
            "--log-level",
            "error",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "ggtts-render: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    std::fs::read(&out).unwrap()
}

/// The FLAC ggtts-render would write for `samples`.
fn flac(dir: &Path, samples: &[f32], rate: u32) -> Vec<u8> {
    let path = dir.join("binding.flac");
    ggtts_core::flac::write_flac(&path, samples, rate).unwrap();
    std::fs::read(&path).unwrap()
}

/// AC2 on the fake engine: for each text and seed, the binding's samples are `ggtts-render --text`'s, byte for byte
/// once written as FLAC, and the trace records one chunk drawn once, at exactly that seed. (The fake engine's audio is
/// a tone per character whatever the seed; its tokens, which the trace records, depend on it. Its output also depends on
/// what it rendered before, so each render is on a fresh handle, as each `ggtts-render` is a fresh process.)
#[test]
fn an_exact_seed_renders_what_ggtts_render_text_renders() {
    let dir = scratch("fake");
    for text in TEXTS {
        for seed in [0u64, 7, 20260905, u64::MAX] {
            let trace = dir.join("t.jsonl");
            let doc = format!(
                r#"{{"render": {{"seed": {seed}}}, "observe": {{"trace": {}}}}}"#,
                serde_json::to_string(&trace.to_string_lossy()).unwrap()
            );
            let h = load(r#"{"load": {"engine": "fake"}}"#);
            let (samples, rate) =
                render(h, text, &doc).unwrap_or_else(|e| panic!("{text} {seed}: {e:?} {}", last_error()));
            ggtts_free(h);
            assert!(
                flac(&dir, &samples, rate) == cli(&dir, &["--engine", "fake"], text, seed),
                "{text} at {seed}: the FLACs differ"
            );
            let begins: Vec<serde_json::Value> = std::fs::read_to_string(&trace)
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
                .filter(|l| l["ev"] == "chunk_begin")
                .collect();
            assert_eq!(begins.len(), 1, "{text} at {seed}: one chunk, one draw");
            assert_eq!(begins[0]["seed"].as_u64(), Some(seed), "{text}");
            assert_eq!(begins[0]["text"], text, "the text is the chunk, unplanned");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// AC4 and the options an exact seed cannot take: each refused under its key, before anything is rendered.
#[test]
fn what_an_exact_seed_cannot_render_is_refused_by_key() {
    let h = load(r#"{"load": {"engine": "fake"}}"#);
    for (text, doc, key) in [
        ("   ", r#"{"render": {"seed": 3}}"#, "render.text"),
        ("", r#"{"render": {"seed": 3}}"#, "render.text"),
        ("多謝晒", r#"{"render": {"seed": 18446744073709551616}}"#, "render.seed"),
        ("多謝晒", r#"{"render": {"seed": -1}}"#, "render.seed"),
        ("多謝晒", r#"{"render": {"seed": 3, "seed_salt": "x"}}"#, "render.seed"),
        ("多謝晒", r#"{"render": {"seed": 3, "redraw": 1}}"#, "render.seed"),
        ("多謝晒", r#"{"render": {"seed": 3, "edge_s": 0.2}}"#, "render.seed"),
    ] {
        match render(h, text, doc) {
            Err((ggtts_ffi::GGTTS_ERR_ARG, Some(k))) => assert_eq!(k, key, "{text:?} {doc}: {}", last_error()),
            other => panic!("{text:?} {doc}: {:?} {}", other.map(|(s, _)| s.len()), last_error()),
        }
    }
    ggtts_free(h);
    // A seed in a load document is refused there: a seed is per render.
    let doc = CString::new(r#"{"load": {"engine": "fake"}, "render": {"seed": 3}}"#).unwrap();
    let mut refused = 0u64;
    assert_eq!(
        unsafe { ggtts_load_opts(doc.as_ptr(), &mut refused) },
        ggtts_ffi::GGTTS_ERR_ARG
    );
    assert_eq!(last_key().as_deref(), Some("render.seed"), "{}", last_error());
}

/// The model files of a real-model run: the tiny cut, or the full talker, with the codec.
fn real_model() -> (String, String) {
    let talker = std::env::var("GGTTS_TINY_TALKER")
        .or_else(|_| std::env::var("GGTTS_TALKER"))
        .expect("set GGTTS_TINY_TALKER (scripts/tiny_fixture.sh) or GGTTS_TALKER");
    let codec = std::env::var("GGTTS_CODEC").expect("set GGTTS_CODEC");
    (talker, codec)
}

fn device() -> String {
    std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into())
}

/// AC2 on a real Qwen3-TTS model: the same samples as `ggtts-render --text`, at 4 threads on `GGTTS_DEVICE`.
#[test]
#[ignore = "needs GGTTS_TINY_TALKER (or GGTTS_TALKER) and GGTTS_CODEC"]
fn an_exact_seed_on_a_real_model_renders_what_ggtts_render_text_renders() {
    let (talker, codec) = real_model();
    let device = device();
    let dir = scratch("real");
    let cases = [(TEXTS[0], 20260905u64), (TEXTS[1], 7)];
    // The binding's renders first, on one handle, freed before any ggtts-render process loads the model: one model
    // loaded at a time.
    let h = load(&format!(
        r#"{{"load": {{"engine": "qwen3-tts", "model": {}, "codec": {}, "threads": 4, "device": "{device}"}}}}"#,
        serde_json::to_string(&talker).unwrap(),
        serde_json::to_string(&codec).unwrap()
    ));
    let ours: Vec<Vec<u8>> = cases
        .iter()
        .map(|(text, seed)| {
            let (samples, rate) = render(h, text, &format!(r#"{{"render": {{"seed": {seed}}}}}"#))
                .unwrap_or_else(|e| panic!("{e:?} {}", last_error()));
            flac(&dir, &samples, rate)
        })
        .collect();
    ggtts_free(h);
    let args = [
        "--engine",
        "qwen3-tts",
        "--model",
        &talker,
        "--codec",
        &codec,
        "--threads",
        "4",
        "--device",
        &device,
        "--offline",
    ];
    for ((text, seed), ours) in cases.iter().zip(ours) {
        assert!(
            ours == cli(&dir, &args, text, *seed),
            "{text} at {seed}: the FLACs differ"
        );
        eprintln!(
            "exact_seed: {text} at {seed}: {} FLAC bytes, the same as ggtts-render --text",
            ours.len()
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// AC3: loading once and rendering N chunks pays the load once. Times N chunks on one handle against N
/// `ggtts-render --text` processes, on `GGTTS_DEVICE` at 4 threads, and prints both.
#[test]
#[ignore = "a timing note: needs GGTTS_TINY_TALKER (or GGTTS_TALKER) and GGTTS_CODEC"]
fn loading_once_pays_the_load_once() {
    let (talker, codec) = real_model();
    let device = device();
    let dir = scratch("timing");
    let t = Instant::now();
    let h = load(&format!(
        r#"{{"load": {{"engine": "qwen3-tts", "model": {}, "codec": {}, "threads": 4, "device": "{device}"}}}}"#,
        serde_json::to_string(&talker).unwrap(),
        serde_json::to_string(&codec).unwrap()
    ));
    let load_ms = t.elapsed().as_millis();
    let mut binding_ms = Vec::new();
    for (i, text) in TEXTS.iter().enumerate() {
        let t = Instant::now();
        render(h, text, &format!(r#"{{"render": {{"seed": {i}}}}}"#)).unwrap();
        binding_ms.push(t.elapsed().as_millis());
    }
    ggtts_free(h);
    let args = [
        "--engine",
        "qwen3-tts",
        "--model",
        &talker,
        "--codec",
        &codec,
        "--threads",
        "4",
        "--device",
        &device,
        "--offline",
    ];
    let mut cli_ms = Vec::new();
    for (i, text) in TEXTS.iter().enumerate() {
        let t = Instant::now();
        cli(&dir, &args, text, i as u64);
        cli_ms.push(t.elapsed().as_millis());
    }
    eprintln!(
        "exact_seed timing ({device}, 4 threads): load once {load_ms} ms, then per chunk {binding_ms:?} ms; \
         ggtts-render --text per chunk (load, hash and render each) {cli_ms:?} ms"
    );
    std::fs::remove_dir_all(&dir).ok();
}
