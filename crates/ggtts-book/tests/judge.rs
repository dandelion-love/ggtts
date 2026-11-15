//! `ggtts-book render --judge`: the ASR judge (`ggtts_instruments::judge`) reads every draw back through the core
//! chunk step, a chunk whose read-back lost text is redrawn at its next seed, and each chunk's reads are recorded in
//! its chapter manifest. The ASR is the stand-in worker `fixtures/judge/fake_judge_worker.py`.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::nested;

fn worker() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/judge/fake_judge_worker.py")
}

/// A stand-in ASR's model directory under `root`, holding `answers`.
fn asr(root: &Path, answers: &str) -> (PathBuf, String) {
    let model = root.join("asr-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("answers.json"), answers).unwrap();
    let spec = format!(
        "asr-qwen,model={},command={},interpreter=python",
        model.display(),
        worker().display()
    );
    (model, spec)
}

fn render(book: &Path, out: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ggtts-book"))
        .arg("render")
        .arg(book)
        .arg("--out")
        .arg(out)
        .args(["--engine", "fake"])
        .args(args)
        .output()
        .unwrap()
}

fn ok(o: &Output) -> serde_json::Value {
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).unwrap()
}

/// Every chunk entry of every chapter manifest under `out`.
fn chunks(out: &Path) -> Vec<serde_json::Value> {
    let mut found = Vec::new();
    for e in std::fs::read_dir(out).unwrap().flatten() {
        let m = e.path().join("manifest.json");
        if let Ok(text) = std::fs::read_to_string(&m) {
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            found.extend(v["manifest"]["chunks"].as_array().unwrap().iter().cloned());
        }
    }
    found
}

/// AC (#306): a chunk whose read-back lost a run of text is rejected and redrawn at the next draw's seed; the reason
/// and the transcript of each draw are in the chunk's manifest entry. A chunk with no CJK characters is read back and
/// never rejected (the judge compares CJK characters only).
#[test]
fn a_chunk_whose_read_back_lost_text_is_redrawn_and_its_reads_recorded() {
    let book = nested().write_temp("judge-redraw");
    let root = book.parent().unwrap();
    let (model, spec) = asr(root, r#"{"first_read_loses": 4}"#);
    let out = root.join("out");
    ok(&render(&book, &out, &["--judge", &spec, "--redraw", "2"]));
    let all = chunks(&out);
    let redrawn: Vec<_> = all.iter().filter(|c| c["kept_draw"] == 1).collect();
    assert_eq!(redrawn.len(), 1, "only the Cantonese chunk loses text: {all:?}");
    let c = redrawn[0];
    assert_eq!(
        (c["draws"].as_u64(), c["verdict"].clone()),
        (Some(2), serde_json::Value::Null),
        "{c}"
    );
    let reads = c["reads"].as_array().unwrap();
    assert_eq!(reads.len(), 2, "{c}");
    assert_eq!(
        (reads[0]["draw"].as_u64(), reads[0]["read_verdict"].as_str()),
        (Some(0), Some("lost-text"))
    );
    assert_eq!(reads[0]["longest_lost"], 4);
    assert!(reads[0]["transcript"].as_str().unwrap().starts_with("今日"));
    assert_eq!(
        (reads[1]["draw"].as_u64(), reads[1]["read_verdict"].clone()),
        (Some(1), serde_json::Value::Null)
    );
    // Every other chunk was read back once and kept.
    for c in all.iter().filter(|c| c["kept_draw"] == 0) {
        let reads = c["reads"].as_array().unwrap();
        assert_eq!(reads.len(), 1, "{c}");
        assert_eq!(reads[0]["lost"], 0, "{c}");
    }
    // The ASR was asked about every draw the duration judge kept (one recorded read each), with its audio on disk.
    // (The fake engine's English chunks are too short for their letters, so the duration judge rejects them as
    // silent and they are never read back.)
    let reads: usize = all.iter().filter_map(|c| c["reads"].as_array()).map(Vec::len).sum();
    let requests = std::fs::read_to_string(model.join("requests.jsonl")).unwrap();
    assert_eq!(requests.lines().count(), reads);
    assert!(reads >= 2);
    assert!(
        requests.lines().all(|l| l.contains("\"audio_exists\": true")),
        "{requests}"
    );
    std::fs::remove_dir_all(root).ok();
}

/// AC (#306): with the judge off, no manifest carries a read and nothing is redrawn for text.
#[test]
fn without_the_judge_nothing_is_read_back() {
    let book = nested().write_temp("judge-off");
    let out = book.parent().unwrap().join("out");
    ok(&render(&book, &out, &["--redraw", "2"]));
    assert!(chunks(&out).iter().all(|c| c.get("reads").is_none()));
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// The judge rejects to redraw, so it is refused without `--redraw`; an ASR that cannot start is refused before
/// anything is rendered, with its reason.
#[test]
fn a_judge_that_cannot_work_is_refused_before_rendering() {
    let book = nested().write_temp("judge-refused");
    let root = book.parent().unwrap();
    let (_, spec) = asr(root, r#"{"never_ready": "no model here"}"#);
    for (args, says) in [
        (vec!["--judge", spec.as_str()], "give --redraw N"),
        (
            vec!["--judge", spec.as_str(), "--redraw", "1"],
            "asr-qwen did not start: no model here",
        ),
        (vec!["--judge", "asr-qwen", "--redraw", "1"], "needs model=DIR"),
    ] {
        let out = root.join("out");
        let o = render(&book, &out, &args);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(
            stderr.contains("--judge (render.judge): ") && stderr.contains(says),
            "{args:?}: {stderr}"
        );
        assert!(!out.exists(), "{args:?} rendered before refusing");
    }
    std::fs::remove_dir_all(root).ok();
}

/// The judge decides which draw is kept, so a resume under another judge (or none) starts the chapter again, naming
/// the difference; the ASR's model path is where it runs, not what it decides, and does not.
#[test]
fn a_resume_under_another_judge_starts_the_chapter_again() {
    let book = nested().write_temp("judge-resume");
    let root = book.parent().unwrap();
    let (_, spec) = asr(root, "{}");
    let out = root.join("out");
    ok(&render(&book, &out, &["--judge", &spec, "--redraw", "1"]));
    let report = ok(&render(&book, &out, &["--redraw", "1"]));
    let written: Vec<_> = report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .collect();
    assert!(!written.is_empty());
    for c in &written {
        let refused = c["refused_manifest"].as_str().unwrap_or_default();
        assert!(refused.contains("judge"), "{c}");
    }
    // The same judge, its model elsewhere: nothing is rendered again.
    let moved = root.join("asr-model-2");
    std::fs::create_dir_all(&moved).unwrap();
    std::fs::write(moved.join("answers.json"), "{}").unwrap();
    let spec2 = spec.replace(
        &root.join("asr-model").display().to_string(),
        &moved.display().to_string(),
    );
    ok(&render(&book, &out, &["--judge", &spec, "--redraw", "1"]));
    let report = ok(&render(&book, &out, &["--judge", &spec2, "--redraw", "1"]));
    for c in report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
    {
        assert_eq!(c["rendered"], 0, "{c}");
    }
    std::fs::remove_dir_all(root).ok();
}

/// Review of #306: a worker that dies mid-render stops the render (exit 2, naming the judge), so no chunk is kept as
/// passed without a read; the next run, with a working worker, resumes and reads every chunk it renders. No chunk's
/// recorded read carries the dead worker's error.
#[test]
fn a_worker_that_dies_mid_render_stops_it_and_the_resume_reads_what_it_renders() {
    let book = nested().write_temp("judge-dies");
    let root = book.parent().unwrap();
    let (model, spec) = asr(root, r#"{"exit_after": 2}"#);
    let out = root.join("out");
    let o = render(&book, &out, &["--judge", &spec, "--redraw", "1"]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        stderr.contains("judge: the ASR exited") && stderr.contains("resume"),
        "{stderr}"
    );
    let before = chunks(&out).iter().filter(|c| c["state"] == "done").count();
    // The worker works again: the same command resumes and finishes.
    std::fs::write(model.join("answers.json"), "{}").unwrap();
    let report = ok(&render(&book, &out, &["--judge", &spec, "--redraw", "1"]));
    let all = chunks(&out);
    assert!(all.iter().all(|c| c["state"] == "done"));
    assert!(
        all.len() > before,
        "the run stopped before the end: {before} of {}",
        all.len()
    );
    for c in &all {
        for r in c["reads"].as_array().into_iter().flatten() {
            assert_eq!(r["error"], serde_json::Value::Null, "{c}");
        }
    }
    assert!(report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["resumed"].as_u64() > Some(0)));
    std::fs::remove_dir_all(root).ok();
}

/// Review of #306: a worker that stops answering is stopped at its deadline (`timeout=S`), and the render with it;
/// one that never says it is ready is refused at `start_timeout`, before anything is rendered.
#[test]
fn a_worker_that_hangs_is_stopped_at_its_deadline() {
    let book = nested().write_temp("judge-hangs");
    let root = book.parent().unwrap();
    let (_, spec) = asr(root, r#"{"hang_after": 1}"#);
    let out = root.join("out");
    let t = std::time::Instant::now();
    let o = render(&book, &out, &["--judge", &format!("{spec},timeout=1"), "--redraw", "1"]);
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("judge: the ASR did not answer within 1 s"), "{stderr}");
    assert!(t.elapsed().as_secs() < 60, "took {:?}", t.elapsed());

    let starts = root.join("starts");
    std::fs::create_dir_all(&starts).unwrap();
    let (_, spec) = asr(&starts, r#"{"hang_at_start": true}"#);
    let out = root.join("out-start");
    let o = render(
        &book,
        &out,
        &["--judge", &format!("{spec},start_timeout=1"), "--redraw", "1"],
    );
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        stderr.contains("asr-qwen did not start: the ASR did not answer within 1 s"),
        "{stderr}"
    );
    assert!(!out.exists());
    std::fs::remove_dir_all(root).ok();
}
