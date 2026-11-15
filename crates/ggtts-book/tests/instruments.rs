//! `ggtts-book render` runs instruments through the same library call as `ggtts-render`
//! (`ggtts_instruments::options`): on each chapter FLAC not yet measured with the same settings, with one report per
//! chapter beside its FLAC, each chapter's own text as an ASR's reference.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::nested;

fn ggtts_book() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ggtts-book"));
    c.env(
        "GGTTS_INSTRUMENTS_DIR",
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments"),
    );
    c
}

fn render(book: &Path, out: &Path, args: &[&str]) -> Output {
    ggtts_book()
        .arg("render")
        .arg(book)
        .arg("--out")
        .arg(out)
        .args(["--engine", "fake"])
        .args(args)
        .output()
        .unwrap()
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

fn stdout_json(o: &Output) -> serde_json::Value {
    serde_json::from_slice(&o.stdout).unwrap_or_else(|e| {
        panic!(
            "{e}: {}\n{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        )
    })
}

fn ok(o: &Output, code: i32) -> serde_json::Value {
    assert_eq!(o.status.code(), Some(code), "{}", String::from_utf8_lossy(&o.stderr));
    stdout_json(o)
}

/// The written chapters of a render report: each chapter's entry and its FLAC under `out`.
fn written(report: &serde_json::Value, out: &Path) -> Vec<(serde_json::Value, PathBuf)> {
    report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .map(|c| (c.clone(), out.join(c["file"].as_str().unwrap())))
        .collect()
}

fn measured(report: &serde_json::Value, out: &Path) -> Vec<String> {
    written(report, out)
        .iter()
        .map(|(c, _)| c["instruments"]["measured"].as_str().unwrap_or("absent").to_string())
        .collect()
}

/// AC2: one shipped instrument runs on every chapter written, with one report per chapter beside the chapter's FLAC.
/// A second run with the same settings measures nothing again and reports each chapter as measured by the earlier
/// run; a run with other instruments measures every chapter again.
#[test]
fn an_instrument_runs_on_every_chapter_written_and_reports_beside_it() {
    let book = nested().write_temp("instr-every-chapter");
    let out = book.parent().unwrap().join("out");
    let report = ok(
        &render(&book, &out, &["--instrument", "audio-levels", "--run", "r1"]),
        0,
    );
    assert_eq!(report["measurement"], "complete");
    let chapters = written(&report, &out);
    assert!(
        chapters.len() >= 2,
        "the fixture writes more than one chapter: {report}"
    );
    let mut logs = Vec::new();
    for (chapter, flac) in &chapters {
        let dir = flac.with_extension("report");
        assert_eq!(
            chapter["instruments"]["report"],
            dir.strip_prefix(&out).unwrap().to_str().unwrap()
        );
        assert_eq!(chapter["instruments"]["measured"], "this run", "{chapter}");
        assert_eq!(chapter["instruments"]["status"]["audio-levels"], "ok", "{chapter}");
        assert_eq!(chapter["instruments"]["complete"], true);
        for f in ["report.json", "report.md", "report.html", "run.log", "measured.json"] {
            assert!(dir.join(f).is_file(), "{} has no {f}", dir.display());
        }
        let envelope = json(&dir.join("instruments/audio-levels/envelope.json"));
        assert_eq!(envelope["status"], "ok");
        assert_eq!(envelope["identity"]["run"], "r1");
        // The envelope measured this chapter's FLAC: its content hash is among the inputs.
        let sha: String = ggtts_core::seed::sha256(&std::fs::read(flac).unwrap())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let inputs = envelope["provenance"]["inputs"].as_array().unwrap();
        assert!(
            inputs.iter().any(|i| i["sha256"] == sha.as_str()),
            "{} ({sha}) not among the inputs: {inputs:?}",
            flac.display()
        );
        logs.push(std::fs::read(dir.join("run.log")).unwrap());
    }
    // The same settings again: nothing is measured again, every chapter is reported from its stamp.
    let report = ok(
        &render(&book, &out, &["--instrument", "audio-levels", "--run", "r2"]),
        0,
    );
    assert!(measured(&report, &out).iter().all(|m| m == "earlier run"), "{report}");
    assert_eq!(report["measurement"], "complete");
    for ((_, flac), log) in written(&report, &out).iter().zip(&logs) {
        assert_eq!(
            &std::fs::read(flac.with_extension("report").join("run.log")).unwrap(),
            log
        );
    }
    // Other instruments: every chapter is measured again.
    let report = ok(
        &render(
            &book,
            &out,
            &["--instrument", "audio-levels", "--instrument", "internal-pauses"],
        ),
        0,
    );
    assert!(measured(&report, &out).iter().all(|m| m == "this run"), "{report}");
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// Review of #295: a book rendered without instruments (or by a run that stopped before measuring) is measured by the
/// next run that asks, though no chunk is rendered again; "complete" never means nothing was measured.
#[test]
fn chapters_finished_by_an_earlier_run_are_measured_when_never_measured() {
    let book = nested().write_temp("instr-measure-later");
    let out = book.parent().unwrap().join("out");
    let report = ok(&render(&book, &out, &[]), 0);
    assert!(report.get("measurement").is_none());
    let report = ok(
        &render(
            &book,
            &out,
            &["--instrument", "audio-levels", "--require", "audio-levels"],
        ),
        0,
    );
    for (c, _) in written(&report, &out) {
        assert_eq!(c["rendered"], 0, "{c}");
        assert_eq!(c["instruments"]["measured"], "this run", "{c}");
        assert_eq!(c["instruments"]["status"]["audio-levels"], "ok", "{c}");
    }
    assert_eq!(report["measurement"], "complete");
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// Review of #295, point 1: an instrument that takes a reference is given each chapter's own text, never one file
/// for every chapter.
#[test]
fn a_reference_taking_instrument_gets_each_chapters_own_text() {
    let book = nested().write_temp("instr-reference");
    let root = book.parent().unwrap();
    let out = root.join("out");
    // asr-qwen with a program that does not exist: it fails on every chapter, but its envelope records the arguments
    // it was given, and the book has written each chapter's reference.
    let spec = format!("asr-qwen,command={}", root.join("no-asr").display());
    let report = ok(&render(&book, &out, &["--instrument", &spec]), 0);
    let inspect = ggtts_book()
        .args(["inspect"])
        .arg(&book)
        .args(["--engine", "fake"])
        .output()
        .unwrap();
    let inspected = stdout_json(&inspect);
    let mut references = Vec::new();
    for (c, flac) in written(&report, &out) {
        let dir = flac.with_extension("report");
        let reference = dir.join("reference.txt");
        let envelope = json(&dir.join("instruments/asr-qwen/envelope.json"));
        assert_eq!(
            envelope["provenance"]["config"]["args"]["reference_file"],
            reference.to_str().unwrap(),
            "{envelope}"
        );
        let text = std::fs::read_to_string(&reference).unwrap();
        let chapter = inspected["chapters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|ch| ch["id"] == c["id"])
            .unwrap();
        let mut at = 0;
        for chunk in chapter["chunks"].as_array().unwrap() {
            let t = chunk["text"].as_str().unwrap();
            let found = text[at..]
                .find(t)
                .unwrap_or_else(|| panic!("{t:?} not in order in {text:?}"));
            at += found + t.len();
        }
        references.push(text);
    }
    references.dedup();
    assert!(references.len() >= 2, "every chapter got the same reference");
    // A failed instrument is not stamped: the next run measures it again.
    let report = ok(&render(&book, &out, &["--instrument", &spec]), 0);
    assert!(measured(&report, &out).iter().all(|m| m == "this run"), "{report}");
    std::fs::remove_dir_all(root).ok();
}

/// Review of #295, point 1: a reference file given for a book is refused before anything is rendered.
#[test]
fn a_reference_file_given_for_a_book_is_refused() {
    let book = nested().write_temp("instr-given-reference");
    let out = book.parent().unwrap().join("out");
    let o = render(&book, &out, &["--instrument", "asr-qwen,arg.reference_file=ref.txt"]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        stderr.contains("arg.reference_file is refused by ggtts-book render"),
        "{stderr}"
    );
    assert!(!out.exists());
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// Review of #295, point 3: under `abort`, a failure on one chapter stops the chapters after it. They are reported
/// unmeasured, which makes the measurement incomplete when an instrument is required.
#[test]
fn abort_stops_the_chapters_after_the_failure() {
    let book = nested().write_temp("instr-abort");
    let root = book.parent().unwrap();
    let spec = format!("broken,command={},version=t", root.join("no-such-program").display());
    for (tag, extra, code, measurement) in [
        ("plain", &[][..], 0, "complete"),
        ("required", &["--require", "broken"][..], 3, "incomplete"),
    ] {
        let out = root.join(tag);
        let mut args = vec!["--instrument", &spec, "--on-instrument-error", "abort"];
        args.extend(extra);
        let report = ok(&render(&book, &out, &args), code);
        assert_eq!(report["measurement"], measurement, "{report}");
        let m = measured(&report, &out);
        assert_eq!(m[0], "this run", "{m:?}");
        assert!(m[1..].iter().all(|x| x == "no"), "{m:?}");
        for (c, _) in written(&report, &out).iter().skip(1) {
            assert!(c["instruments"]["reason"].as_str().unwrap().contains("abort"), "{c}");
        }
    }
    // The required instrument is ok on the first chapter; another one's failure aborts the rest. The chapters left
    // unmeasured make the measurement incomplete, though every chapter that was measured is complete.
    let out = root.join("required-ok");
    let args = [
        "--instrument",
        "audio-levels",
        "--require",
        "audio-levels",
        "--instrument",
        spec.as_str(),
        "--on-instrument-error",
        "abort",
    ];
    let report = ok(&render(&book, &out, &args), 3);
    let chapters = written(&report, &out);
    assert_eq!(chapters[0].0["instruments"]["complete"], true, "{report}");
    assert!(measured(&report, &out)[1..].iter().all(|x| x == "no"));
    assert_eq!(report["measurement"], "incomplete");
    std::fs::remove_dir_all(root).ok();
}

/// `--report DIR` puts each chapter's report in `DIR/NNN-<chapter id>`.
#[test]
fn report_puts_each_chapters_report_under_it() {
    let book = nested().write_temp("instr-report-dir");
    let root = book.parent().unwrap();
    let out = root.join("out");
    let reports = root.join("reports");
    let report = ok(
        &render(
            &book,
            &out,
            &["--instrument", "audio-levels", "--report", reports.to_str().unwrap()],
        ),
        0,
    );
    for (_, flac) in written(&report, &out) {
        let stem = flac.file_stem().unwrap();
        assert!(reports
            .join(stem)
            .join("instruments/audio-levels/envelope.json")
            .is_file());
        assert!(!flac.with_extension("report").exists());
    }
    std::fs::remove_dir_all(root).ok();
}

/// AC3: an unknown instrument is refused before any chapter is rendered, or the engine loaded.
#[test]
fn an_unknown_instrument_is_refused_before_any_chapter_is_rendered() {
    let book = nested().write_temp("instr-unknown");
    let out = book.parent().unwrap().join("out");
    let o = render(&book, &out, &["--instrument", "no-such-instrument"]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("\"no-such-instrument\" is not shipped"), "{stderr}");
    assert!(!out.exists(), "a chapter was rendered before the refusal");
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC5: a required instrument that fails gives the book exit 3, as for ggtts-render, with every chapter's FLAC
/// written; the same instrument not required exits 0.
#[test]
fn a_required_instrument_that_fails_exits_3_with_every_chapter_written() {
    let book = nested().write_temp("instr-required");
    let root = book.parent().unwrap();
    let spec = format!("broken,command={},version=t", root.join("no-such-program").display());

    let out = root.join("required");
    let o = render(&book, &out, &["--instrument", &spec, "--require", "broken"]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("measurement INCOMPLETE"));
    let report = ok(&o, 3);
    assert_eq!(report["measurement"], "incomplete");
    let chapters = written(&report, &out);
    assert!(chapters.len() >= 2);
    for (c, flac) in &chapters {
        assert!(flac.is_file(), "{} was not written", flac.display());
        assert_eq!(c["instruments"]["complete"], false, "{c}");
        assert_ne!(c["instruments"]["status"]["broken"], "ok", "{c}");
    }

    let out = root.join("not-required");
    let report = ok(&render(&book, &out, &["--instrument", &spec]), 0);
    assert_eq!(report["measurement"], "complete");
    std::fs::remove_dir_all(root).ok();
}

/// The book's trace covers every chapter, so a chapter is handed no event stream: an instrument that needs one is
/// skipped, saying why, and the measurement goes on.
#[test]
fn an_instrument_that_needs_the_event_stream_is_skipped_with_the_reason() {
    let book = nested().write_temp("instr-events");
    let out = book.parent().unwrap().join("out");
    let report = ok(&render(&book, &out, &["--instrument", "render-totals"]), 0);
    let (_, flac) = &written(&report, &out)[0];
    let envelope = json(
        &flac
            .with_extension("report")
            .join("instruments/render-totals/envelope.json"),
    );
    assert_eq!(envelope["status"], "skipped");
    assert!(
        envelope["error"].as_str().unwrap().contains("no event stream"),
        "{envelope}"
    );
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// Without an instrument option the render report carries no instrument fields, as before.
#[test]
fn without_instruments_the_report_has_no_measurement() {
    let book = nested().write_temp("instr-none");
    let out = book.parent().unwrap().join("out");
    let report = ok(&render(&book, &out, &[]), 0);
    assert!(report.get("measurement").is_none());
    for (c, flac) in written(&report, &out) {
        assert!(c.get("instruments").is_none(), "{c}");
        assert!(!flac.with_extension("report").exists());
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}
