//! The Python instrument SDK and templates (`instruments/`), run against a fake-engine render.
//!
//! This renders the recorded pipeline case with the most chunks on the fake engine, traced at level 2. It writes the
//! trace and the read's FLAC as the runner hands them to an instrument, plus the 16-bit samples the FLAC must decode
//! to. Then it runs, with `GGTTS_FIXTURE_DIR` pointing at them:
//! - the SDK's tests;
//! - each template's and the worked example's self-test: a planted defect it must detect, and a no-op that must
//!   read as no change.
//!
//! Each suite must report `OK`, run at least the tests it is known to have, and skip none. A discovery that finds
//! nothing, or a test that quietly skips, fails here.
//!
//! It needs Python 3.8 or later as `python3` or `python`. Without one it fails, unless `GGTTS_SKIP_PYTHON=1` says to
//! skip it on purpose.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use ggtts_core::flac::{to_i16, write_flac};
use ggtts_core::pipeline::render_story_traced;
use ggtts_core::trace::{emit_header, Header, JsonlEvents};

fn repo() -> PathBuf {
    // Not canonicalized: on Windows that gives a verbatim path, a needless thing to hand Python as its cwd.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// A Python 3.8+ interpreter, or `None` when skipping was asked for.
fn python() -> Option<String> {
    for exe in ["python3", "python"] {
        let ok = Command::new(exe)
            .args(["-c", "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"])
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return Some(exe.into());
        }
    }
    if std::env::var("GGTTS_SKIP_PYTHON").as_deref() == Ok("1") {
        eprintln!("GGTTS_SKIP_PYTHON=1: the Python instrument suites were not run");
        return None;
    }
    panic!("no Python 3.8+ found as python3 or python; install one, or set GGTTS_SKIP_PYTHON=1 to skip on purpose");
}

/// The fake engine's render of the case with the most chunks, written as the runner would hand it over.
fn fixture(dir: &Path) {
    let case = common::cases()
        .into_iter()
        .max_by_key(|c| {
            let e = c.engine();
            c.chunks(&e).len()
        })
        .unwrap();
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    let mut w = JsonlEvents::new(std::fs::File::create(dir.join("trace.jsonl")).unwrap(), 2);
    emit_header(&mut w, &Header::for_engine(&engine, 2));
    let read = render_story_traced(&mut engine, &case.story_id, &chunks, &opts, &mut w).expect("renders");
    w.finish().expect("the trace is written");
    write_flac(&dir.join("audio.flac"), &read.samples, read.sample_rate).unwrap();
    let raw: Vec<u8> = read.samples.iter().flat_map(|&x| to_i16(x).to_le_bytes()).collect();
    std::fs::write(dir.join("audio.s16le"), raw).unwrap();
    std::fs::write(
        dir.join("meta.json"),
        format!("{{\"rate\": {}, \"case\": \"{}\"}}\n", read.sample_rate, case.id),
    )
    .unwrap();
}

/// Run one unittest invocation; `(tests run, skipped)`, after requiring `OK`.
fn unittest(py: &str, cwd: &Path, args: &[&str], fixture: &Path) -> (usize, usize) {
    let out = Command::new(py)
        .args(["-m", "unittest"])
        .args(args)
        .arg("-v")
        .current_dir(cwd)
        .env("GGTTS_FIXTURE_DIR", fixture)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("python runs");
    let text = String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{} {args:?} failed:\n{text}", cwd.display());
    let ran = text
        .lines()
        .find_map(|l| {
            l.strip_prefix("Ran ")
                .and_then(|r| r.split_whitespace().next()?.parse().ok())
        })
        .unwrap_or_else(|| panic!("no test count in:\n{text}"));
    let skipped = text.matches("... skipped").count();
    assert!(text.lines().any(|l| l.starts_with("OK")), "not OK:\n{text}");
    (ran, skipped)
}

#[test]
fn the_sdk_templates_and_example_pass_their_self_tests_on_a_fake_engine_render() {
    let Some(py) = python() else { return };
    let dir = std::env::temp_dir().join(format!("ggtts-python-instruments-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    fixture(&dir);

    let root = repo().join("instruments");
    let (ran, skipped) = unittest(&py, &root.join("python"), &["discover", "-s", "tests"], &dir);
    assert!(ran >= 15, "the SDK suite ran only {ran} tests");
    assert_eq!(skipped, 0, "an SDK test skipped although the fixture was given");

    // Each template's self-test: at least the planted defect and the no-op.
    for t in [
        "templates/python/step",
        "templates/python/chunk",
        "templates/python/trim",
        "templates/python/render",
        "templates/python/audio",
        "examples/python/internal-pauses",
        "examples/python/pitch-range",
        "examples/python/asr-sensevoice",
        "examples/python/asr-qwen",
    ] {
        let (ran, skipped) = unittest(&py, &root.join(t), &["test_instrument"], &dir);
        assert!(ran >= 2, "{t}: only {ran} tests ran");
        assert_eq!(skipped, 0, "{t}: a test skipped");
        eprintln!("{t}: {ran} tests OK");
    }
    // The pYIN plant also runs in a separate, dependency-equipped environment. The default
    // suite still exercises the arithmetic plant, core boundaries and fail-closed runtime checks.
    let (ran, skipped) = unittest(
        &py,
        &root.join("examples/python/loudness-ends"),
        &[
            "test_instrument.EndsTest.test_loudness_arithmetic_noop_and_compression",
            "test_instrument.EndsTest.test_speech_core_discards_a_short_separated_edge",
            "test_instrument.EndsTest.test_refuses_missing_audio",
            "test_instrument.EndsTest.test_missing_dependencies_are_incomplete",
            "test_instrument.EndsTest.test_unverified_librosa_version_is_incomplete",
        ],
        &dir,
    );
    assert_eq!((ran, skipped), (5, 0), "loudness dependency-free controls");
    std::fs::remove_dir_all(&dir).unwrap();
}
