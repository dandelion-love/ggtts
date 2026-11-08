//! The runner keeps every instrument from breaking a run: a crashing, hanging, invalid or audio-writing instrument
//! each get the right status in the merged `ggtts-report/1` report, and the render's output is byte for byte what it
//! was. Completeness is judged on the required instruments alone, and the runner's cost is logged against its budget.
//!
//! The instruments are `ggtts-instrument-selftest` in its planted modes. Each planted failure is shown to produce its
//! status, and the same instrument in mode `ok` (the no-op) is shown to produce `ok` with the audio unchanged.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ggtts_core::flac::write_flac;
use ggtts_core::pipeline::{plan_story, render_story, render_story_traced, RenderOptions};
use ggtts_core::report::{Direction, Metric, Status};
use ggtts_core::trace::{emit_header, Header, Recorder};
use ggtts_fake::FakeEngine;
use ggtts_instruments::config::{Config, OnError};
use ggtts_instruments::runner::{
    check, in_process, run, Inputs, BUDGET_PER_INSTRUMENT_MS, BUDGET_RUN_MS, EXIT_INCOMPLETE, EXIT_OK,
    EXIT_RENDER_FAILED,
};

const SELFTEST: &str = env!("CARGO_BIN_EXE_ggtts-instrument-selftest");
const RUN: &str = "run-test";

fn short() -> Vec<String> {
    vec!["今日天氣好好，我哋去海邊行下。".into(), "行完之後，去食飯。".into()]
}

/// The longest prose case in the pipeline fixtures.
fn long_prose() -> Vec<String> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/pipeline/render.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let paras = |c: &serde_json::Value| -> Vec<String> {
        c["paragraphs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_string())
            .collect()
    };
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(paras)
        .max_by_key(|p| p.iter().map(|s| s.chars().count()).sum::<usize>())
        .unwrap()
}

/// A fresh fake render's FLAC bytes, untraced.
fn render_bytes(dir: &Path, paragraphs: &[String]) -> Vec<u8> {
    let mut engine = FakeEngine::new(HashMap::new(), HashMap::new());
    let profile = *ggtts_core::engine::Engine::profile(&engine);
    let read = render_story(
        &mut engine,
        "story",
        &plan_story(&profile, paragraphs),
        &RenderOptions::for_profile(&profile),
    )
    .unwrap();
    let p = write_flac(&dir.join("fresh"), &read.samples, read.sample_rate).unwrap();
    std::fs::read(p).unwrap()
}

/// A run directory with a traced fake render: the FLAC and its event stream.
fn setup_with(tag: &str, paragraphs: &[String]) -> (PathBuf, Inputs) {
    let dir = std::env::temp_dir().join(format!("ggtts-instruments-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut engine = FakeEngine::new(HashMap::new(), HashMap::new());
    let profile = *ggtts_core::engine::Engine::profile(&engine);
    let mut rec = Recorder::new(1);
    // A real trace starts with its header; the Python SDK refuses one that does not.
    emit_header(&mut rec, &Header::for_engine(&engine, 1));
    let read = render_story_traced(
        &mut engine,
        "story",
        &plan_story(&profile, paragraphs),
        &RenderOptions::for_profile(&profile),
        &mut rec,
    )
    .unwrap();
    let audio = write_flac(&dir.join("out"), &read.samples, read.sample_rate).unwrap();
    let events = dir.join("events.jsonl");
    let lines: Vec<String> = rec
        .lines
        .iter()
        .map(|m| serde_json::Value::Object(m.clone()).to_string())
        .collect();
    std::fs::write(&events, lines.join("\n") + "\n").unwrap();
    let inputs = Inputs {
        run: RUN.into(),
        render: "story".into(),
        report_dir: dir.join("report"),
        audio,
        audio_state: ggtts_core::report::AudioState::Trimmed,
        audio_seconds: Some(read.seconds()),
        events: Some(events),
        bin_dir: Path::new(SELFTEST).parent().map(Path::to_path_buf),
        instruments_dir: Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments")),
    };
    (dir, inputs)
}

fn setup(tag: &str) -> (PathBuf, Inputs) {
    setup_with(tag, &short())
}

/// An entry running the selftest program under its own name, in `mode`.
fn planted(name: &str, mode: &str, extra: &str) -> String {
    format!("{name},command={SELFTEST},version=t,arg.mode={mode}{extra}")
}

fn read(p: PathBuf) -> String {
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn an_ok_instrument_reports_ok_with_identity_and_the_report_is_merged() {
    let (dir, inputs) = setup("ok");
    let mut c = Config::default();
    c.apply_instrument_flag("selftest").unwrap();
    check(&c).unwrap();
    let s = run(&c, &inputs).unwrap();
    let e = &s.envelopes[0];
    assert_eq!(e.status, Status::Ok, "{e:?}");
    assert_eq!(e.log.as_deref(), Some("instruments/selftest/selftest.log"));
    assert_eq!(
        (e.identity.run.as_str(), e.identity.instrument.as_str()),
        (RUN, "selftest")
    );
    let events = e
        .metrics
        .iter()
        .find(|m| m.name == "events")
        .expect("the events were read");
    assert!(events.value.unwrap() > 4.0, "{events:?}");
    assert_eq!(
        events.render.as_deref(),
        Some("story"),
        "the render identity reached the instrument"
    );
    // It read the whole audio file: the instrument started after the render had written it.
    let bytes = e.metrics.iter().find(|m| m.name == "audio_bytes").unwrap();
    assert_eq!(
        bytes.value,
        Some(std::fs::metadata(&inputs.audio).unwrap().len() as f64)
    );
    assert_eq!(e.provenance.inputs.len(), 2, "the audio and the events, by hash");

    // The merged report: written as json, md and html, and agreeing with the envelopes.
    assert_eq!(s.report.run, RUN);
    assert_eq!(s.report.entries.len(), 1);
    assert!(
        s.report.entries[0].invalid.is_empty(),
        "{:?}",
        s.report.entries[0].invalid
    );
    for f in ["report.json", "report.md", "report.html"] {
        assert!(inputs.report_dir.join(f).exists(), "{f}");
    }
    let log = read(inputs.report_dir.join("run.log"));
    assert!(
        log.contains(&format!(
            "instrument selftest {}: ok (exited with code 0",
            env!("CARGO_PKG_VERSION")
        )),
        "{log}"
    );
    assert!(log.contains("time to first audio +0 ms"), "{log}");
    assert!(log.contains("measurement: complete"), "{log}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn crash_hang_invalid_and_tamper_each_get_their_status_and_the_render_is_untouched() {
    let (dir, inputs) = setup("contain");
    let rendered = std::fs::read(&inputs.audio).unwrap();
    let mut c = Config::default();
    for spec in [
        planted("crash", "crash", ""),
        planted("hang", "hang", ",timeout_s=1"),
        planted("wrong", "bad-envelope", ""),
        planted("tamper", "tamper", ""),
        planted("after", "ok", ""),
    ] {
        c.apply_instrument_flag(&spec).unwrap();
    }
    assert_eq!(c.on_error, OnError::Continue, "continue is the default");
    let s = run(&c, &inputs).unwrap();
    let got: Vec<(&str, Status)> = s.envelopes.iter().map(|e| (e.name.as_str(), e.status)).collect();
    assert_eq!(
        got,
        vec![
            ("crash", Status::Failed),
            ("hang", Status::Timeout),
            ("wrong", Status::Invalid),
            ("tamper", Status::Failed),
            ("after", Status::Ok),
        ]
    );
    let why = |n: &str| {
        s.envelopes
            .iter()
            .find(|e| e.name == n)
            .unwrap()
            .error
            .clone()
            .unwrap_or_default()
    };
    assert!(why("crash").starts_with("it exited with code"), "{}", why("crash"));
    assert!(why("hang").starts_with("no exit within 1 s"), "{}", why("hang"));
    assert!(why("wrong").starts_with("invalid envelope:"), "{}", why("wrong"));
    assert!(why("tamper").contains("changed the run's audio"), "{}", why("tamper"));
    assert!(!s.aborted);
    assert_eq!(s.failures(), 4);

    // The crash's panic message is in its own log, not anyone else's.
    let crash_log = read(inputs.report_dir.join("instruments/crash/crash.log"));
    assert!(crash_log.contains("planted crash"), "{crash_log}");
    assert!(!read(inputs.report_dir.join("instruments/after/after.log")).contains("planted crash"));

    // A broken instrument is reported, never dropped: every envelope the runner kept validates, carries the run's
    // identity and the configured version, and the merged report has one valid entry per instrument.
    for e in &s.envelopes {
        let v: serde_json::Value = serde_json::from_str(&read(
            inputs.report_dir.join(format!("instruments/{}/envelope.json", e.name)),
        ))
        .unwrap();
        ggtts_core::report::validate(&v).unwrap_or_else(|w| panic!("{}: {w:?}", e.name));
        assert_eq!(v["identity"]["run"], RUN);
        let want = if e.status == Status::Ok {
            env!("CARGO_PKG_VERSION")
        } else {
            "t"
        };
        assert_eq!(v["version"], want, "{}", e.name);
    }
    assert_eq!(s.report.entries.len(), 5);
    assert!(s.report.entries.iter().all(|e| e.invalid.is_empty()));

    // The render is unaffected: its bytes are as written, and as a fresh render without instruments.
    assert_eq!(
        std::fs::read(&inputs.audio).unwrap(),
        rendered,
        "the run's audio changed"
    );
    assert_eq!(
        render_bytes(&dir, &short()),
        rendered,
        "a render without instruments differs"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn abort_skips_the_rest_and_keeps_the_render() {
    let (dir, inputs) = setup("abort");
    let rendered = std::fs::read(&inputs.audio).unwrap();
    let mut c = Config::default();
    c.apply_on_error_flag("abort").unwrap();
    c.apply_instrument_flag(&planted("first", "crash", "")).unwrap();
    c.apply_instrument_flag("selftest").unwrap();
    let s = run(&c, &inputs).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Failed);
    assert_eq!(s.envelopes[1].status, Status::Skipped);
    assert!(s.aborted);
    assert!(s.complete(), "abort governs the run; nothing was required");
    let log = read(inputs.report_dir.join("run.log"));
    assert!(
        log.contains("instrument selftest") && log.contains("skipped (not started"),
        "{log}"
    );
    assert!(log.contains("1 skipped, 0 truncated; aborted"), "{log}");
    assert_eq!(std::fs::read(&inputs.audio).unwrap(), rendered);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_failing_required_instrument_makes_the_measurement_incomplete_and_an_optional_one_does_not() {
    let (dir, inputs) = setup("require");
    let rendered = std::fs::read(&inputs.audio).unwrap();

    // An optional instrument fails: the report shows it, and the measurement is still complete.
    let mut c = Config::default();
    c.apply_instrument_flag(&planted("optional", "crash", "")).unwrap();
    c.apply_instrument_flag("selftest").unwrap();
    c.apply_require_flag("selftest").unwrap();
    let s = run(&c, &inputs).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Failed);
    assert!(s.complete(), "{:?}", s.incomplete);
    assert!(s.report.complete());
    assert_eq!(s.exit_code(), EXIT_OK);

    // The same failure, required: INCOMPLETE with the reason, a distinct exit code, and the partial result kept.
    c.apply_require_flag("optional").unwrap();
    let s = run(&c, &inputs).unwrap();
    assert!(!s.complete());
    assert!(!s.report.complete(), "the merged report agrees");
    assert_eq!(s.exit_code(), EXIT_INCOMPLETE);
    assert_ne!(EXIT_INCOMPLETE, EXIT_OK);
    assert_ne!(EXIT_INCOMPLETE, EXIT_RENDER_FAILED);
    assert_eq!(s.incomplete.len(), 1, "{:?}", s.incomplete);
    assert!(
        s.incomplete[0].starts_with("required instrument optional is failed: it exited"),
        "{:?}",
        s.incomplete
    );
    let ok = s.envelopes.iter().find(|e| e.name == "selftest").unwrap();
    assert_eq!(ok.status, Status::Ok, "the other instrument's result is kept");
    assert!(!ok.metrics.is_empty());
    let log = read(inputs.report_dir.join("run.log"));
    assert!(
        log.contains("measurement: INCOMPLETE: required instrument optional is failed"),
        "{log}"
    );

    // A required instrument nobody configured is missing, which is INCOMPLETE too, never a pass.
    let mut missing = Config::default();
    missing.apply_instrument_flag("selftest").unwrap();
    missing.apply_require_flag("absent").unwrap();
    let s = run(&missing, &inputs).unwrap();
    assert_eq!(
        s.incomplete,
        vec!["required instrument absent was not configured to run".to_string()]
    );
    assert_eq!(s.exit_code(), EXIT_INCOMPLETE);

    assert_eq!(std::fs::read(&inputs.audio).unwrap(), rendered);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_required_input_that_is_absent_or_truncated_makes_the_measurement_incomplete() {
    let (dir, inputs) = setup("inputs");
    let mut c = Config::default();
    c.apply_instrument_flag(&planted("reader", "ok", ",needs_events=true"))
        .unwrap();
    c.apply_require_flag("reader").unwrap();

    // With its whole event stream, it runs.
    let s = run(&c, &inputs).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Ok);
    assert!(s.complete());

    // Cut short: the last line is not the end event.
    let events = inputs.events.clone().unwrap();
    let text = read(events);
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines.last().unwrap().contains("\"ev\":\"end\""),
        "the fixture ends with its end event"
    );
    let cut = dir.join("cut.jsonl");
    std::fs::write(&cut, lines[..lines.len() - 3].join("\n") + "\n").unwrap();
    let truncated = Inputs {
        events: Some(cut),
        ..inputs.clone()
    };
    let s = run(&c, &truncated).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Truncated);
    assert!(
        s.envelopes[0].error.as_deref().unwrap().contains("truncated"),
        "{:?}",
        s.envelopes[0]
    );
    assert_eq!(s.exit_code(), EXIT_INCOMPLETE);

    // No event stream at all.
    let none = Inputs {
        events: None,
        ..inputs.clone()
    };
    let s = run(&c, &none).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Skipped);
    assert!(s.envelopes[0].error.as_deref().unwrap().contains("no event stream"));
    assert_eq!(s.exit_code(), EXIT_INCOMPLETE);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_overhead_is_logged_against_the_budget_stated_before_measuring() {
    let paragraphs = long_prose();
    let (dir, inputs) = setup_with("overhead", &paragraphs);
    let mut c = Config::default();
    // Every shipped instrument: the selftest binary and the Python templates and example, unless Python is skipped on
    // purpose (GGTTS_SKIP_PYTHON=1, as the Python instrument suites honour).
    let skip_python = std::env::var_os("GGTTS_SKIP_PYTHON").is_some();
    for k in ggtts_instruments::catalogue::CATALOGUE {
        if k.interpreter.is_none() || !skip_python {
            c.apply_instrument_flag(k.name).unwrap();
        }
    }
    let s = run(&c, &inputs).unwrap();
    assert!(s.envelopes.iter().all(|e| e.status == Status::Ok), "{:?}", s.envelopes);
    let o = s.overhead;
    assert_eq!(o.own_ms + o.instruments_ms, o.total_ms);
    assert_eq!(
        o.budget_ms,
        BUDGET_PER_INSTRUMENT_MS * c.instruments.len() as u128 + BUDGET_RUN_MS
    );
    let log = read(inputs.report_dir.join("run.log"));
    let line = log
        .lines()
        .find(|l| l.starts_with("overhead: time to first audio +0 ms"))
        .unwrap();
    assert!(
        line.contains(&format!("runner {} ms (budget {} ms", o.own_ms, o.budget_ms)),
        "{line}"
    );
    assert!(log.contains("overhead: RTF added"), "{log}");
    eprintln!("{line}");
    for e in &s.envelopes {
        eprintln!("  {} {}: {}", e.name, e.version, e.status.as_str());
    }
    eprintln!("{}", log.lines().find(|l| l.starts_with("overhead: RTF")).unwrap_or(""));
    // The budget is a claim about the shipped (optimised) runner, so a release build checks it. Most of the runner's
    // own time is the audio's SHA-256 for provenance, which an unoptimised build runs about 50 times slower (86 ms
    // for 3 MB, measured), so a debug build only refuses a cost ten times over budget.
    let limit = if cfg!(debug_assertions) {
        o.budget_ms * 10
    } else {
        o.budget_ms
    };
    assert!(o.own_ms <= limit, "the runner's own cost is over budget: {line}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_unknown_instrument_is_refused_before_the_render() {
    let mut c = Config::default();
    c.apply_instrument_flag("selftset").unwrap();
    let e = check(&c).unwrap_err();
    assert!(
        e.contains("\"selftset\" is not shipped") && e.contains("selftest"),
        "{e}"
    );
}

#[test]
fn an_in_process_hook_panic_is_caught_and_reported_but_is_not_a_full_boundary() {
    let ok = in_process(RUN, "tensor-ok", "1", &["per-step"], || {
        Ok(vec![Metric {
            name: "x".into(),
            value: Some(1.0),
            unit: "n".into(),
            direction: Direction::Neutral,
            interval: None,
            render: None,
            chunk: None,
        }])
    });
    assert_eq!(ok.status, Status::Ok);
    assert_eq!(ok.metrics.len(), 1);
    ggtts_core::report::validate(&ok.to_json()).unwrap();

    let err = in_process(RUN, "tensor-err", "1", &["per-step"], || Err("no logits".into()));
    assert_eq!((err.status, err.error.as_deref()), (Status::Failed, Some("no logits")));

    // The planted panic: caught, reported with its message, and the test (standing in for the run) goes on.
    let panicked = in_process(RUN, "tensor-panic", "1", &["per-step"], || panic!("planted hook panic"));
    assert_eq!(panicked.status, Status::Failed);
    assert_eq!(panicked.error.as_deref(), Some("panicked: planted hook panic"));
    assert!(
        panicked.metrics.is_empty(),
        "a failed hook reports no value, never a zero"
    );
    ggtts_core::report::validate(&panicked.to_json()).unwrap();
    // A hang or a process abort is not contained here; runner.rs's docs say so, and nothing tests a containment
    // that does not exist.
}

#[test]
fn a_script_instrument_runs_under_its_interpreter_and_a_missing_one_is_reported() {
    if std::env::var_os("GGTTS_SKIP_PYTHON").is_some() {
        return;
    }
    let (dir, inputs) = setup("interpreter");
    // The shipped Python instrument runs under the default interpreter, never as a bare `.py` (Windows cannot).
    let mut c = Config::default();
    c.apply_instrument_flag("audio-levels").unwrap();
    let s = run(&c, &inputs).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Ok, "{:?}", s.envelopes[0]);
    // The planted case: an interpreter that does not exist is a failure that names it, not a silent skip.
    let mut c = Config::default();
    c.apply_instrument_flag("audio-levels,interpreter=no-such-python")
        .unwrap();
    let s = run(&c, &inputs).unwrap();
    assert_eq!(s.envelopes[0].status, Status::Failed);
    let why = s.envelopes[0].error.clone().unwrap_or_default();
    assert!(why.starts_with("could not start no-such-python "), "{why}");
    let _ = std::fs::remove_dir_all(dir);
}
