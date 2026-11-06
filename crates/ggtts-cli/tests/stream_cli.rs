//! `ggtts-render --stream`, end to end on the fake engine: the stream on stdout is a valid container holding the
//! story's pieces before the level step, the `--out` FLAC is the levelled batch read, unchanged, and stderr
//! reports the level gain that turns one into the other.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ggtts_core::engine::Engine;
use ggtts_core::flac::to_i16;
use ggtts_core::pipeline::{plan_story, render_story_streamed, RenderOptions};
use ggtts_core::stream::{raw_header, UNKNOWN_LENGTH};
use ggtts_core::text::read_text;
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;

const BIN: &str = env!("CARGO_BIN_EXE_ggtts-render");

/// A story file built from the recorded render cases' paragraphs, so no new text enters the tests.
fn story(dir: &Path) -> PathBuf {
    let cases = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/pipeline/render.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cases).unwrap()).unwrap();
    let mut paras: Vec<String> = Vec::new();
    for c in v["cases"].as_array().unwrap() {
        for p in c["paragraphs"].as_array().unwrap() {
            paras.push(p.as_str().unwrap().to_string());
        }
    }
    let path = dir.join("story.txt");
    std::fs::write(&path, format!("streamed\n{}\n", paras.join("\n"))).unwrap();
    path
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-stream-cli-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The story as the core streams it on the fake engine: (pre-level pieces, levelled batch samples, rate, gain).
fn reference(path: &Path) -> (Vec<f32>, Vec<f32>, u32, Option<f64>) {
    let (title, paragraphs) = read_text(&std::fs::read_to_string(path).unwrap());
    let mut engine = FakeEngine::default();
    let profile = *engine.profile();
    let chunks = plan_story(&profile, &paragraphs);
    let mut pieces = Vec::new();
    let read = render_story_streamed(
        &mut engine,
        &title,
        &chunks,
        &RenderOptions::for_profile(&profile),
        &mut NullEvents,
        &mut |p| pieces.extend_from_slice(p.samples),
    )
    .unwrap();
    (pieces, read.samples, read.sample_rate, read.gain)
}

fn run(args: &[&str]) -> (Vec<u8>, String) {
    let mut child = Command::new(BIN)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut out = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
    let mut err = String::new();
    child.stderr.take().unwrap().read_to_string(&mut err).unwrap();
    assert!(child.wait().unwrap().success(), "ggtts-render failed: {err}");
    (out, err)
}

fn pcm(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn flac_samples(path: &Path) -> Vec<i16> {
    let mut r = claxon::FlacReader::open(path).unwrap();
    r.samples().map(|s| s.unwrap() as i16).collect()
}

#[test]
fn a_wav_stream_on_stdout_and_the_flac_beside_it() {
    let dir = scratch("wav");
    let story = story(&dir);
    let flac = dir.join("out.flac");
    let (out, err) = run(&[
        "--engine",
        "fake",
        "--story",
        story.to_str().unwrap(),
        "--out",
        flac.to_str().unwrap(),
        "--stream",
        "-",
    ]);
    let (pieces, batch, rate, gain) = reference(&story);

    assert_eq!(&out[0..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(out[4..8].try_into().unwrap()), UNKNOWN_LENGTH);
    assert_eq!(u32::from_le_bytes(out[24..28].try_into().unwrap()), rate);
    assert_eq!(&out[36..40], b"data");
    let streamed = pcm(&out[44..]);
    let want: Vec<i16> = pieces.iter().map(|&x| to_i16(x)).collect();
    assert_eq!(streamed.len(), want.len());
    assert!(streamed == want, "the stream is the read before its level step");

    let flac = flac_samples(&flac);
    let want_flac: Vec<i16> = batch.iter().map(|&x| to_i16(x)).collect();
    assert!(flac == want_flac, "the FLAC is the levelled batch read");
    assert_ne!(flac, streamed, "this story is levelled, so the two differ by the gain");

    let gain_db = 20.0 * gain.unwrap().log10();
    assert!(
        err.contains(&format!("{gain_db:+.4} dB")),
        "stderr reports the gain: {err}"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_raw_stream_to_a_file_without_a_flac() {
    let dir = scratch("raw");
    let story = story(&dir);
    let raw = dir.join("out.s16");
    let (out, _) = run(&[
        "--engine",
        "fake",
        "--story",
        story.to_str().unwrap(),
        "--stream",
        raw.to_str().unwrap(),
        "--stream-format",
        "s16le",
    ]);
    assert!(out.is_empty(), "nothing on stdout when the stream goes to a file");
    let (pieces, _, rate, _) = reference(&story);
    let bytes = std::fs::read(&raw).unwrap();
    let head = raw_header(rate);
    assert!(bytes.starts_with(head.as_bytes()));
    let want: Vec<i16> = pieces.iter().map(|&x| to_i16(x)).collect();
    assert!(pcm(&bytes[head.len()..]) == want);
    assert!(!dir.join("out.flac").exists());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_text_stream_holds_the_flac_samples() {
    let dir = scratch("text");
    let flac = dir.join("one.flac");
    let (out, _) = run(&[
        "--engine",
        "fake",
        "--text",
        "一二三四五六",
        "--out",
        flac.to_str().unwrap(),
        "--stream",
        "-",
    ]);
    assert!(
        pcm(&out[44..]) == flac_samples(&flac),
        "no level step on --text, so stream and FLAC match"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn stream_flags_are_checked() {
    for args in [
        vec![
            "--engine",
            "fake",
            "--text",
            "一二三四",
            "--stream",
            "-",
            "--stream-format",
            "mp3",
        ],
        vec![
            "--engine",
            "fake",
            "--text",
            "一二三四",
            "--stream-format",
            "wav",
            "--out",
            "x.flac",
        ],
        vec!["--engine", "fake", "--text", "一二三四"],
    ] {
        let o = Command::new(BIN).args(&args).output().unwrap();
        assert!(!o.status.success(), "{args:?} should be refused");
    }
}
