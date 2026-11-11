use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ggtts_core::pipeline::gap_samples;
use ggtts_fake::{LEAD_S, SAMPLE_RATE, TAIL_S};
use ggtts_qwen3_tts::PROFILE;

/// The fake engine's tone for a chunk with no CJK character, in seconds.
const SPEECH_S: f64 = 0.25;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-duration-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn samples_of(seconds: f64) -> u64 {
    (seconds * f64::from(SAMPLE_RATE)).round() as u64
}

fn flac_samples(path: &Path) -> u64 {
    let mut reader = claxon::FlacReader::open(path).unwrap();
    reader.samples().fold(0, |n, s| {
        n + u64::from(s.is_ok() || panic!("a FLAC sample failed to decode"))
    })
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn instrumented(out: &Path, extra: &[&std::ffi::OsStr]) -> Output {
    let instruments = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments");
    Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .env("GGTTS_INSTRUMENTS_DIR", instruments)
        .args(["--engine", "fake", "--text", "Hello.", "--out"])
        .arg(out)
        .args(["--instrument", "audio-levels"])
        .args(extra)
        .output()
        .unwrap()
}

fn story_report(dir: &Path) -> serde_json::Value {
    let story = dir.join("story.txt");
    std::fs::write(&story, "Title\nFirst sentence.\nSecond sentence.\nThird sentence.").unwrap();
    let render = |audio: &str, report: Option<&str>| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ggtts-render"));
        cmd.args(["--engine", "fake", "--story"])
            .arg(&story)
            .arg("--out")
            .arg(dir.join(audio));
        if let Some(report) = report {
            cmd.arg("--duration-report").arg(dir.join(report));
        }
        let result = cmd.output().unwrap();
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    };
    render("plain.flac", None);
    render("measured.flac", Some("duration.json"));
    assert_eq!(
        std::fs::read(dir.join("plain.flac")).unwrap(),
        std::fs::read(dir.join("measured.flac")).unwrap()
    );
    json(&dir.join("duration.json"))
}

fn text_report(dir: &Path) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(["--engine", "fake", "--text", "Hello.", "--out"])
        .arg(dir.join("out.flac"))
        .arg("--duration-report")
        .arg(dir.join("duration.json"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    json(&dir.join("duration.json"))
}

/// Every expected count comes from the fake engine's constants, the profile's pause table and the FLAC on disk,
/// never from another field of the report.
#[test]
fn duration_report_is_trace_free_and_preserves_audio() {
    let dir = scratch("story");
    let d = story_report(&dir);
    let decoded = samples_of(LEAD_S + SPEECH_S + TAIL_S);
    let trimmed = samples_of(SPEECH_S + 2.0 * PROFILE.edge_s);
    // Each paragraph is one chunk ending in an ASCII full stop, so each join is the default pause plus the
    // paragraph pause; there is none after the last chunk.
    let join = gap_samples(&PROFILE, ".", true, SAMPLE_RATE) as u64;
    assert_eq!((decoded, trimmed, join), (25_200, 11_760, 15_120));
    let chunks = d["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), 3);
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c["chunk"], i as u64);
        assert_eq!(c["kept_draw"], 0);
        assert_eq!(c["sample_rate"], SAMPLE_RATE);
        assert_eq!(c["decoded_samples"], decoded);
        assert_eq!(c["trimmed_samples"], trimmed);
    }
    assert_eq!(d["join_samples"], 2 * join);
    assert_eq!(d["output_samples"], 3 * trimmed + 2 * join);
    assert_eq!(d["output_samples"], flac_samples(&dir.join("measured.flac")));
    assert_eq!(d["output_sample_rate"], SAMPLE_RATE);
}

#[test]
fn text_reports_decode_without_story_trimming() {
    let dir = scratch("text");
    let d = text_report(&dir);
    assert_eq!(d["path"], "text");
    // A text render is the untrimmed decode: lead silence, tone and tail silence.
    let decoded = samples_of(LEAD_S + SPEECH_S + TAIL_S);
    assert_eq!(decoded, 25_200);
    assert_eq!(d["text_decoded_samples"], decoded);
    assert_eq!(d["output_samples"], decoded);
    assert_eq!(flac_samples(&dir.join("out.flac")), decoded);
    assert!(d["chunks"].is_null() && d["join_samples"].is_null());
}

#[test]
fn published_schema_accepts_real_reports_and_rejects_planted_defects() {
    let schema = json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/ggtts-duration-1.schema.json"));
    let validator = jsonschema::validator_for(&schema).expect("the published duration schema compiles");
    let story = story_report(&scratch("schema-story"));
    let text = text_report(&scratch("schema-text"));
    for report in [&story, &text] {
        let errors: Vec<String> = validator.iter_errors(report).map(|e| e.to_string()).collect();
        assert!(errors.is_empty(), "{errors:?}");
    }
    let plant = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut bad = story.clone();
        edit(&mut bad);
        bad
    };
    let defects = [
        ("another format", plant(&|v| v["format"] = "ggtts-duration/2".into())),
        ("an unknown path", plant(&|v| v["path"] = "stream".into())),
        (
            "discarded draws claimed",
            plant(&|v| v["discarded_draws_included"] = true.into()),
        ),
        ("a negative count", plant(&|v| v["output_samples"] = (-1).into())),
        ("a fractional count", plant(&|v| v["join_samples"] = 1.5.into())),
        ("an unknown key", plant(&|v| v["extra"] = 1.into())),
        (
            "a missing key",
            plant(&|v| {
                v.as_object_mut().unwrap().remove("rate_note");
            }),
        ),
        (
            "a chunk missing a count",
            plant(&|v| {
                v["chunks"][0].as_object_mut().unwrap().remove("trimmed_samples");
            }),
        ),
        ("an unknown chunk key", plant(&|v| v["chunks"][0]["extra"] = 1.into())),
    ];
    for (what, bad) in &defects {
        assert!(!validator.is_valid(bad), "the published schema must reject {what}");
    }
}

/// A rerun to the same output succeeds, as it does without duration reports: the report directory's files are
/// replaced, like the run log and instrument reports beside them.
#[test]
fn instrumented_rerun_to_the_same_output_replaces_duration_sidecars() {
    let dir = scratch("rerun");
    let out = dir.join("o.flac");
    let report = dir.join("o.report");
    let first = instrumented(&out, &[]);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let fresh = std::fs::read(report.join("duration.json")).unwrap();
    assert_eq!(json(&report.join("duration.json"))["format"], "ggtts-duration/1");
    std::fs::write(report.join("duration.json"), b"stale").unwrap();
    std::fs::write(report.join("duration.md"), b"stale").unwrap();
    let second = instrumented(&out, &[]);
    assert!(second.status.success(), "{}", String::from_utf8_lossy(&second.stderr));
    assert_eq!(std::fs::read(report.join("duration.json")).unwrap(), fresh);
    assert!(std::fs::read_to_string(report.join("duration.md"))
        .unwrap()
        .contains("Audio durations"));
}

#[test]
fn explicit_duration_can_share_new_instrument_report_destination() {
    let dir = scratch("shared");
    let report = dir.join("report");
    let explicit = report.join("duration.json");
    let out = instrumented(
        &dir.join("out.flac"),
        &[
            "--require".as_ref(),
            "audio-levels".as_ref(),
            "--report".as_ref(),
            report.as_os_str(),
            "--duration-report".as_ref(),
            explicit.as_os_str(),
        ],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(json(&explicit)["format"], "ggtts-duration/1");
    assert!(report.join("duration.md").exists());
}

/// An explicit report is never overwritten, and a path that cannot be written is refused before any audio is.
#[test]
fn explicit_duration_report_is_checked_before_rendering() {
    let dir = scratch("explicit");
    let run = |audio: &str, report: &Path| {
        Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
            .args(["--engine", "fake", "--text", "Hello.", "--out"])
            .arg(dir.join(audio))
            .arg("--duration-report")
            .arg(report)
            .output()
            .unwrap()
    };
    let missing = run("missing.flac", &dir.join("missing-dir/d.json"));
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("is not a directory"));
    assert!(
        !dir.join("missing.flac").exists(),
        "nothing is rendered for an unwritable report"
    );
    std::fs::write(dir.join("kept.json"), b"keep").unwrap();
    assert!(!run("existing.flac", &dir.join("kept.json")).status.success());
    assert_eq!(std::fs::read(dir.join("kept.json")).unwrap(), b"keep");
    assert!(!dir.join("existing.flac").exists());
}

/// The explicit report may name the not-yet-created instrument report directory by any spelling of it.
fn shared_fresh_report_dir(name: &str, spell: &dyn Fn(&Path) -> PathBuf) {
    let dir = scratch(name);
    let instruments = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments");
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .current_dir(&dir)
        .env("GGTTS_INSTRUMENTS_DIR", instruments)
        .args(["--engine", "fake", "--text", "Hello.", "--out", "e.flac"])
        .args(["--instrument", "audio-levels", "--duration-report"])
        .arg(spell(&dir))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(json(&dir.join("e.report/duration.json"))["format"], "ggtts-duration/1");
    assert!(dir.join("e.report/duration.md").exists());
}

#[test]
fn explicit_duration_shares_fresh_report_dir_spelled_with_dot_slash() {
    shared_fresh_report_dir("dot-slash", &|_| PathBuf::from("./e.report/duration.json"));
}

#[test]
fn explicit_duration_shares_fresh_report_dir_spelled_absolute() {
    shared_fresh_report_dir("absolute", &|dir| dir.join("e.report/duration.json"));
}

#[test]
fn explicit_duration_shares_fresh_report_dir_spelled_with_dot_dot() {
    shared_fresh_report_dir("dot-dot", &|_| PathBuf::from("e.report/../e.report/duration.json"));
}
