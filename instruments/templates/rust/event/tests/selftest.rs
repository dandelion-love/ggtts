//! The template's self-test, on a fake-engine render: the planted defect is seen, the no-op reads as no change, and
//! a truncated stream is reported as `truncated`.

use std::path::{Path, PathBuf};
use std::process::Command;

use ggtts_core::engine::Engine;
use ggtts_core::pipeline::{plan_story, render_story_traced, RenderOptions};
use ggtts_core::trace::{emit_header, Header, JsonlEvents};
use ggtts_fake::FakeEngine;
use serde_json::{Map, Value};

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-event-template-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A fake-engine render's trace, as JSON lines.
fn trace() -> Vec<Map<String, Value>> {
    let mut e = FakeEngine::default();
    let paragraphs = vec!["今日天氣好好，我哋去公園行下。".to_string(), "聽日再傾。".to_string()];
    let chunks = plan_story(e.profile(), &paragraphs);
    let opts = RenderOptions::for_profile(e.profile());
    let mut w = JsonlEvents::new(Vec::new(), 1);
    emit_header(&mut w, &Header::for_engine(&e, 1));
    render_story_traced(&mut e, "template", &chunks, &opts, &mut w).unwrap();
    let text = String::from_utf8(w.finish().unwrap()).unwrap();
    text.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn write(path: &Path, lines: &[Map<String, Value>]) {
    let text: String = lines
        .iter()
        .map(|l| Value::Object(l.clone()).to_string() + "\n")
        .collect();
    std::fs::write(path, text).unwrap();
}

/// Run the instrument as the runner does; its envelope.
fn run(d: &Path, lines: &[Map<String, Value>]) -> Value {
    let (events, out) = (d.join("trace.jsonl"), d.join("env.json"));
    write(&events, lines);
    let _ = std::fs::remove_file(&out);
    let st = Command::new(env!("CARGO_BIN_EXE_ggtts-event-template"))
        .args([
            "--events",
            events.to_str().unwrap(),
            "--envelope",
            out.to_str().unwrap(),
        ])
        .args(["--run", "selftest", "--render", "story"])
        .status()
        .unwrap();
    let env: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(st.success(), env["status"] == "ok", "the exit code follows the status");
    env
}

fn metric(env: &Value, name: &str) -> Value {
    env["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == name)
        .unwrap_or_else(|| panic!("no {name}"))["value"]
        .clone()
}

/// The envelope fields every `ggtts-report/1` envelope carries.
fn assert_shape(env: &Value) {
    for k in [
        "format",
        "name",
        "version",
        "identity",
        "provenance",
        "hooks",
        "status",
        "metrics",
        "artifacts",
        "log",
    ] {
        assert!(env.get(k).is_some(), "the envelope has no {k}");
    }
    assert_eq!(env["format"], "ggtts-report/1");
    assert_eq!(env["identity"]["run"], "selftest");
    let input = &env["provenance"]["inputs"][0]["sha256"];
    assert!(
        input.as_str().is_some_and(|h| h.len() == 64),
        "the input is recorded by hash"
    );
    for m in env["metrics"].as_array().unwrap() {
        assert!(["higher", "lower", "neutral", "n/a"].contains(&m["direction"].as_str().unwrap()));
    }
}

#[test]
fn the_no_op_reads_as_no_change() {
    let d = dir("noop");
    let lines = trace();
    let a = run(&d, &lines);
    assert_eq!(a["status"], "ok");
    assert_shape(&a);
    let moved: Vec<Map<String, Value>> = lines
        .iter()
        .map(|l| {
            let mut l = l.clone();
            let t = l["t_ms"].as_u64().unwrap();
            l.insert("t_ms".into(), (t + 1000).into());
            if let Some(ms) = l.get("ms").and_then(Value::as_u64) {
                l.insert("ms".into(), (ms + 1000).into());
            }
            l
        })
        .collect();
    let b = run(&d, &moved);
    assert_eq!(a["metrics"], b["metrics"]);
    assert!(metric(&a, "draws").as_u64().unwrap() > 0);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_planted_runaway_draw_is_seen() {
    let d = dir("plant");
    let mut lines = trace();
    let before = metric(&run(&d, &lines), "longest_draw_frames").as_u64().unwrap();
    assert!(before < 5000);
    let end = lines.iter_mut().find(|l| l["ev"] == "chunk_end").unwrap();
    end.insert("frames".into(), 5000.into());
    let env = run(&d, &lines);
    assert_eq!(metric(&env, "longest_draw_frames"), 5000);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_truncated_stream_is_reported_not_read_as_a_short_render() {
    let d = dir("cut");
    let mut lines = trace();
    assert_eq!(lines.last().unwrap()["ev"], "end");
    lines.pop();
    let env = run(&d, &lines);
    assert_eq!(env["status"], "truncated");
    assert_shape(&env);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn what_is_not_a_trace_is_a_failed_envelope() {
    let d = dir("bad");
    let out = d.join("env.json");
    std::fs::write(d.join("x.jsonl"), "{\"ev\":\"step\",\"t_ms\":0}\n").unwrap();
    let st = Command::new(env!("CARGO_BIN_EXE_ggtts-event-template"))
        .args([
            "--events",
            d.join("x.jsonl").to_str().unwrap(),
            "--envelope",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!st.success());
    let env: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(env["status"], "failed");
    assert!(env["error"].as_str().unwrap().contains("not a trace"));
    std::fs::remove_dir_all(&d).unwrap();
}
