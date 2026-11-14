//! `ggtts-render --story --judge`: the ASR judge reads every draw back, a chunk whose read-back lost text is redrawn,
//! the trace records each reading (`read`), and the trace still replays. The ASR is the stand-in worker
//! `fixtures/judge/fake_judge_worker.py`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Map, Value};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-judge-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A stand-in ASR under `dir` with `answers`, and its `--judge` spec.
fn asr(dir: &Path, answers: &str) -> String {
    let model = dir.join("asr-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("answers.json"), answers).unwrap();
    format!(
        "asr-qwen,model={},command={},interpreter=python",
        model.display(),
        root().join("fixtures/judge/fake_judge_worker.py").display()
    )
}

const STORY: &str = "今日天氣好好，我哋一齊去公園散步。\n\n佢話聽日會落雨，所以我哋要帶遮。\n";

fn render(dir: &Path, args: &[&str]) -> Output {
    std::fs::write(dir.join("s.txt"), STORY).unwrap();
    Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .current_dir(dir)
        .args(["--engine", "fake", "--log-level", "error"])
        .args(args)
        .output()
        .unwrap()
}

fn trace(path: &Path) -> Vec<Map<String, Value>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap().as_object().unwrap().clone())
        .collect()
}

/// AC (#306): with the judge on, each chunk's first draw loses a clause in its read-back, is rejected as lost text
/// and redrawn at its next seed; the trace records the verdict on each `chunk_end` and what was read in a `read`
/// event, it validates, and it replays (replay takes the recorded text verdicts as given).
#[test]
fn a_story_chunk_that_lost_text_is_redrawn_and_the_trace_records_and_replays_it() {
    let dir = scratch("redraw");
    let spec = asr(&dir, r#"{"first_read_loses": 4}"#);
    let o = render(
        &dir,
        &[
            "--story", "s.txt", "--out", "o.flac", "--judge", &spec, "--redraw", "2", "--trace", "t.jsonl",
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let lines = trace(&dir.join("t.jsonl"));
    let ends: Vec<(u64, u64, Value)> = lines
        .iter()
        .filter(|l| l["ev"] == "chunk_end")
        .map(|l| {
            (
                l["chunk"].as_u64().unwrap(),
                l["draw"].as_u64().unwrap(),
                l["verdict"].clone(),
            )
        })
        .collect();
    let chunks: std::collections::BTreeSet<u64> = ends.iter().map(|e| e.0).collect();
    assert!(!chunks.is_empty());
    for c in &chunks {
        let mine: Vec<_> = ends.iter().filter(|e| e.0 == *c).collect();
        assert_eq!(mine.len(), 2, "chunk {c}: {ends:?}");
        assert_eq!((mine[0].1, mine[0].2.as_str()), (0, Some("lost-text")), "chunk {c}");
        assert_eq!((mine[1].1, &mine[1].2), (1, &Value::Null), "chunk {c}");
    }
    let reads: Vec<_> = lines.iter().filter(|l| l["ev"] == "read").collect();
    assert_eq!(reads.len(), ends.len(), "one read per draw");
    for r in &reads {
        let lost = r["draw"] == 0;
        assert_eq!(r["read_verdict"].as_str(), lost.then_some("lost-text"), "{r:?}");
        assert!(r["transcript"].as_str().is_some_and(|t| !t.is_empty()));
    }
    // The trace validates against ggtts-trace/2.
    let schema: Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("schemas/ggtts-trace-2.schema.json")).unwrap())
            .unwrap();
    let v = jsonschema::validator_for(&schema).unwrap();
    for l in &lines {
        let doc = Value::Object(l.clone());
        assert!(v.is_valid(&doc), "{doc}");
    }
    // It replays on a fresh fake engine, the recorded text verdicts standing in for the ASR.
    let report = ggtts_core::trace::replay(&mut ggtts_fake::FakeEngine::default(), &lines).unwrap();
    assert_eq!(report.mismatch, None, "{report:?}");
    assert_eq!(report.draws, ends.len());
    std::fs::remove_dir_all(&dir).ok();
}

/// With the judge off, the trace has no `read` event and nothing is redrawn for text.
#[test]
fn without_the_judge_the_trace_has_no_reads() {
    let dir = scratch("off");
    let o = render(
        &dir,
        &[
            "--story", "s.txt", "--out", "o.flac", "--redraw", "2", "--trace", "t.jsonl",
        ],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(trace(&dir.join("t.jsonl")).iter().all(|l| l["ev"] != "read"));
    std::fs::remove_dir_all(&dir).ok();
}

/// The judge is refused by name, before anything is written: on `--text` (one draw, nothing to redraw), without
/// `--redraw`, and when its ASR does not start.
#[test]
fn the_judge_is_refused_where_it_cannot_work() {
    let dir = scratch("refused");
    let spec = asr(&dir, "{}");
    let never = {
        let d = dir.join("never");
        std::fs::create_dir_all(&d).unwrap();
        asr(&d, r#"{"never_ready": "no model here"}"#)
    };
    for (args, says) in [
        (
            vec!["--text", "你好。", "--out", "o.flac", "--judge", &spec, "--redraw", "1"],
            "--text renders one draw",
        ),
        (
            vec!["--story", "s.txt", "--out", "o.flac", "--judge", &spec],
            "give --redraw N",
        ),
        (
            vec![
                "--story", "s.txt", "--out", "o.flac", "--judge", &never, "--redraw", "1",
            ],
            "asr-qwen did not start: no model here",
        ),
    ] {
        let o = render(&dir, &args);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(
            stderr.contains("--judge (render.judge): ") && stderr.contains(says),
            "{args:?}: {stderr}"
        );
        assert!(!dir.join("o.flac").exists(), "{args:?} wrote output");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Review of #306: a worker that dies mid-story (on its first read: the story has one chunk, its first paragraph
/// being the title) stops the render with the judge's reason, and nothing is written.
#[test]
fn a_worker_that_dies_mid_story_stops_the_render() {
    let dir = scratch("dies");
    let spec = asr(&dir, r#"{"exit_after": 0}"#);
    let o = render(
        &dir,
        &["--story", "s.txt", "--out", "o.flac", "--judge", &spec, "--redraw", "1"],
    );
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("judge: the ASR exited"), "{stderr}");
    assert!(!dir.join("o.flac").exists());
    std::fs::remove_dir_all(&dir).ok();
}

/// Review of #306: a worker that closes its output but stays alive is stopped, not waited on, so the render ends
/// promptly with the judge's reason.
#[test]
fn a_worker_that_closes_its_output_and_lives_on_is_stopped() {
    let dir = scratch("closes");
    let spec = asr(&dir, r#"{"close_after": 0}"#);
    let t = std::time::Instant::now();
    let o = render(
        &dir,
        &["--story", "s.txt", "--out", "o.flac", "--judge", &spec, "--redraw", "1"],
    );
    assert_eq!(o.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        stderr.contains("judge: the ASR closed its output, and was stopped"),
        "{stderr}"
    );
    assert!(t.elapsed().as_secs() < 60, "took {:?}", t.elapsed());
    std::fs::remove_dir_all(&dir).ok();
}
