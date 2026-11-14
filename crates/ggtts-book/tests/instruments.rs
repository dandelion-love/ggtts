//! `ggtts-book render` runs instruments through the same library call as `ggtts-render`
//! (`ggtts_instruments::options`): once per chapter FLAC the run wrote, with one report per chapter beside its FLAC.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::nested;

fn render(book: &Path, out: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ggtts-book"))
        .env(
            "GGTTS_INSTRUMENTS_DIR",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments"),
        )
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

/// The chapter FLACs a render report names, as paths under `out`.
fn written(report: &serde_json::Value, out: &Path) -> Vec<PathBuf> {
    report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .map(|c| out.join(c["file"].as_str().unwrap()))
        .collect()
}

/// AC2: one shipped instrument runs on every chapter written in the run, with one report per chapter beside the
/// chapter's FLAC; a second run that resumes every chapter as finished measures nothing.
#[test]
fn an_instrument_runs_on_every_chapter_written_and_reports_beside_it() {
    let book = nested().write_temp("instr-every-chapter");
    let out = book.parent().unwrap().join("out");
    let o = render(&book, &out, &["--instrument", "audio-levels", "--run", "r1"]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let report = stdout_json(&o);
    assert_eq!(report["measurement"], "complete");
    let files = written(&report, &out);
    assert!(files.len() >= 2, "the fixture writes more than one chapter: {report}");
    for (chapter, flac) in report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .zip(&files)
    {
        let dir = flac.with_extension("report");
        assert_eq!(
            chapter["instruments"]["report"],
            dir.strip_prefix(&out).unwrap().to_str().unwrap()
        );
        assert_eq!(chapter["instruments"]["status"]["audio-levels"], "ok", "{chapter}");
        assert_eq!(chapter["instruments"]["complete"], true);
        for f in ["report.json", "report.md", "report.html", "run.log"] {
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
        std::fs::remove_dir_all(&dir).unwrap();
    }
    // Every chapter is finished: a second run renders nothing and so measures nothing.
    let o = render(&book, &out, &["--instrument", "audio-levels"]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let report = stdout_json(&o);
    for (c, flac) in report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .zip(written(&report, &out))
    {
        assert!(c.get("instruments").is_none(), "a resumed chapter was measured: {c}");
        assert!(!flac.with_extension("report").exists());
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// `--report DIR` puts each chapter's report in `DIR/NNN-<chapter id>`.
#[test]
fn report_puts_each_chapters_report_under_it() {
    let book = nested().write_temp("instr-report-dir");
    let root = book.parent().unwrap();
    let out = root.join("out");
    let reports = root.join("reports");
    let o = render(
        &book,
        &out,
        &["--instrument", "audio-levels", "--report", reports.to_str().unwrap()],
    );
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let report = stdout_json(&o);
    for flac in written(&report, &out) {
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
    let missing = root.join("no-such-program");
    let spec = format!("broken,command={},version=t", missing.display());

    let out = root.join("required");
    let o = render(&book, &out, &["--instrument", &spec, "--require", "broken"]);
    assert_eq!(o.status.code(), Some(3), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stderr).contains("measurement INCOMPLETE"));
    let report = stdout_json(&o);
    assert_eq!(report["measurement"], "incomplete");
    let files = written(&report, &out);
    assert!(files.len() >= 2);
    for flac in &files {
        assert!(flac.is_file(), "{} was not written", flac.display());
    }
    for c in report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
    {
        assert_eq!(c["instruments"]["complete"], false, "{c}");
        assert_ne!(c["instruments"]["status"]["broken"], "ok", "{c}");
    }

    let out = root.join("not-required");
    let o = render(&book, &out, &["--instrument", &spec]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(stdout_json(&o)["measurement"], "complete");
    std::fs::remove_dir_all(root).ok();
}

/// The book's trace covers every chapter, so a chapter is handed no event stream: an instrument that needs one is
/// skipped, saying why, and the measurement goes on.
#[test]
fn an_instrument_that_needs_the_event_stream_is_skipped_with_the_reason() {
    let book = nested().write_temp("instr-events");
    let out = book.parent().unwrap().join("out");
    let o = render(&book, &out, &["--instrument", "render-totals"]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let report = stdout_json(&o);
    let flac = &written(&report, &out)[0];
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
    let o = render(&book, &out, &[]);
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    let report = stdout_json(&o);
    assert!(report.get("measurement").is_none());
    for c in report["chapters"].as_array().unwrap() {
        assert!(c.get("instruments").is_none(), "{c}");
    }
    for flac in written(&report, &out) {
        assert!(!flac.with_extension("report").exists());
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}
