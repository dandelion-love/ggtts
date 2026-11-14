//! A book takes the shared options (`ggtts_core::options`) through `ggtts-book render`'s parser: the style
//! instruction, the steering and the trace reach every chunk the way they reach a story's, and a chapter's manifest
//! records the instruction and steering so a resume under different ones starts the chapter again, naming the
//! difference.

mod common;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common::nested;
use ggtts_book::render::{render_command_options, LoadedEngine};
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::profile::Profile;
use ggtts_fake::FakeEngine;
use serde_json::Value;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-options-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// What one request carried: its chunk, instruction and steering record.
type Seen = (u32, Option<String>, Option<Value>);

/// The fake engine, recording every request's instruction and steering, then rendering it without them (the fake
/// engine refuses both, as an engine without them must).
struct Recording {
    inner: FakeEngine,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Engine for Recording {
    fn info(&self) -> &EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &Profile {
        self.inner.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.seen.lock().unwrap().push((
            req.chunk,
            req.instruction().map(str::to_string),
            req.steer.map(|s| s.record()),
        ));
        let plain = ChunkRequest {
            text: req.text,
            seed: req.seed,
            chunk: req.chunk,
            draw: req.draw,
            instruction: None,
            steer: None,
        };
        self.inner.generate(&plain)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.inner.decode(codes)
    }
}

/// `ggtts-book render BOOK <args>` through the command line's parser and `render_command_options`, on an engine that
/// records what each request carried. The report and the requests.
fn render(book: &Path, args: &[&str]) -> Result<(Value, Vec<Seen>), String> {
    let mut all = vec![book.to_string_lossy().into_owned()];
    all.extend(args.iter().map(|s| s.to_string()));
    let (path, options) = ggtts_book::options::parse("render", &all, &|_| None)?;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let engine = Recording {
        inner: FakeEngine::default(),
        seen: seen.clone(),
    };
    let report = render_command_options(&path, &options, || {
        Ok(LoadedEngine {
            engine: Box::new(engine),
            files: Vec::new(),
            speaker: None,
        })
    })?;
    let seen = seen.lock().unwrap().clone();
    Ok((report, seen))
}

fn steering_file(d: &Path, name: &str, value: f32) -> String {
    let path = d.join(name);
    let v = [value; 8];
    std::fs::write(
        &path,
        ggtts_core::steer::write_safetensors(&[("talker.layer.0", &v), ("talker.layer.1", &v)]),
    )
    .unwrap();
    path.to_string_lossy().into_owned()
}

/// Each written chapter's (id, chunks rendered in this run, why its manifest was refused).
fn chapters(report: &Value) -> Vec<(String, u64, Option<String>)> {
    report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["state"] == "written")
        .map(|c| {
            (
                c["id"].as_str().unwrap().to_string(),
                c["rendered"].as_u64().unwrap(),
                c["refused_manifest"].as_str().map(String::from),
            )
        })
        .collect()
}

/// AC2: `--instruction`, `--steer`, `--steer-scale` and `--steer-layers` reach every chunk's request, as they reach a
/// story's: the same instruction, and the steering record of the file with its scale and chosen layers.
#[test]
fn the_instruction_and_steering_reach_every_chunk_request() {
    let d = dir("requests");
    let book = nested().write_temp("options-requests");
    let steer = steering_file(&d, "s.safetensors", 0.25);
    let out = d.join("out");
    let (report, seen) = render(
        &book,
        &[
            "--out",
            out.to_str().unwrap(),
            "--engine",
            "fake",
            "--instruction",
            "slowly",
            "--steer",
            &steer,
            "--steer-scale",
            "0.5",
            "--steer-layers",
            "1",
        ],
    )
    .unwrap();
    let rendered: u64 = chapters(&report).iter().map(|c| c.1).sum();
    assert!(rendered > 3, "the fixture book renders chunks");
    assert_eq!(seen.len() as u64, rendered, "one request per chunk at redraw 0");
    for (chunk, instruction, steer) in &seen {
        assert_eq!(instruction.as_deref(), Some("slowly"), "chunk {chunk}");
        let steer = steer
            .as_ref()
            .unwrap_or_else(|| panic!("chunk {chunk} was not steered"));
        assert_eq!(steer["scale"], 0.5, "chunk {chunk}");
        assert_eq!(steer["layers"], serde_json::json!([1]), "chunk {chunk}");
    }
    // The report and every manifest record them.
    assert_eq!(report["render"]["instruction"], "slowly");
    assert_eq!(report["render"]["steer"], seen[0].2.clone().unwrap());
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC2: an unsteered book with no instruction sends none: no field is invented.
#[test]
fn without_the_options_no_request_carries_them() {
    let d = dir("plain");
    let book = nested().write_temp("options-plain");
    let out = d.join("out");
    let (report, seen) = render(&book, &["--out", out.to_str().unwrap(), "--engine", "fake"]).unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|(_, i, s)| i.is_none() && s.is_none()));
    assert!(report.get("render").is_none(), "a render with neither records neither");
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC2: `--trace` writes one ggtts-trace/2 stream for the chunks the run renders: a header naming the engine, the
/// instruction and the steering, one `chunk_begin` per rendered chunk, and a trace that replays on the same engine.
#[test]
fn the_trace_is_written_and_replays() {
    let d = dir("trace");
    let book = nested().write_temp("options-trace");
    let out = d.join("out");
    let trace = d.join("book.jsonl");
    let (path, options) = ggtts_book::options::parse(
        "render",
        &[
            book.to_string_lossy().into_owned(),
            "--out".into(),
            out.to_string_lossy().into_owned(),
            "--engine".into(),
            "fake".into(),
            "--trace".into(),
            trace.to_string_lossy().into_owned(),
            "--trace-level".into(),
            "2".into(),
        ],
        &|_| None,
    )
    .unwrap();
    let report = render_command_options(&path, &options, || {
        Ok(LoadedEngine {
            engine: Box::new(FakeEngine::default()),
            files: Vec::new(),
            speaker: None,
        })
    })
    .unwrap();
    let rendered: u64 = chapters(&report).iter().map(|c| c.1).sum();
    let lines = ggtts_core::trace::parse(&std::fs::read_to_string(&trace).unwrap()).unwrap();
    assert_eq!(lines[0]["ev"], "trace");
    assert_eq!(lines[0]["engine"], "fake");
    assert_eq!(lines[0]["trace_level"], 2);
    let begins = lines.iter().filter(|l| l["ev"] == "chunk_begin").count() as u64;
    assert_eq!(begins, rendered, "one chunk_begin per rendered chunk");
    let replay = ggtts_core::trace::replay(&mut FakeEngine::default(), &lines).unwrap();
    assert_eq!(replay.mismatch, None, "the book's trace replays");
    assert_eq!(replay.draws as u64, rendered);
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC4: a chapter resumed under another instruction, or with an instruction where there was none, starts again, its
/// manifest refused naming the instruction; the same render options resume.
#[test]
fn a_resume_under_another_instruction_starts_again_naming_it() {
    let d = dir("resume-instruction");
    let book = nested().write_temp("options-resume-instruction");
    let out = d.join("out");
    let o = out.to_str().unwrap();
    let (first, _) = render(&book, &["--out", o, "--engine", "fake", "--instruction", "calm"]).unwrap();
    let total: u64 = chapters(&first).iter().map(|c| c.1).sum();
    let (same, seen) = render(&book, &["--out", o, "--engine", "fake", "--instruction", "calm"]).unwrap();
    assert!(seen.is_empty(), "the same options resume everything");
    assert!(chapters(&same).iter().all(|c| c.1 == 0 && c.2.is_none()));
    for (args, manifest, plan) in [
        (vec!["--instruction", "brisk"], "calm", "brisk"),
        (vec![], "brisk", "none"),
        (vec!["--instruction", "calm"], "none", "calm"),
    ] {
        let mut all = vec!["--out", o, "--engine", "fake"];
        all.extend(args);
        let (report, _) = render(&book, &all).unwrap();
        let cs = chapters(&report);
        assert_eq!(cs.iter().map(|c| c.1).sum::<u64>(), total, "{plan}: every chunk again");
        for (id, _, refused) in cs {
            let refused = refused.unwrap_or_else(|| panic!("{id}: the manifest was trusted under {plan}"));
            assert!(
                refused.contains("instruction") && refused.contains(manifest) && refused.contains(plan),
                "{id}: {refused}"
            );
        }
    }
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC4: a chapter resumed with another steering file starts again, its manifest refused naming the steering; the
/// same file at the same scale resumes.
#[test]
fn a_resume_with_another_steering_file_starts_again_naming_it() {
    let d = dir("resume-steer");
    let book = nested().write_temp("options-resume-steer");
    let (a, b) = (
        steering_file(&d, "a.safetensors", 0.25),
        steering_file(&d, "b.safetensors", 0.75),
    );
    let out = d.join("out");
    let o = out.to_str().unwrap();
    render(&book, &["--out", o, "--engine", "fake", "--steer", &a]).unwrap();
    let (same, seen) = render(&book, &["--out", o, "--engine", "fake", "--steer", &a]).unwrap();
    assert!(seen.is_empty(), "the same steering resumes everything");
    assert!(chapters(&same).iter().all(|c| c.2.is_none()));
    for args in [vec!["--steer", &b], vec!["--steer", &b, "--steer-scale", "2"], vec![]] {
        let mut all = vec!["--out", o, "--engine", "fake"];
        all.extend(args.iter().copied());
        let (report, _) = render(&book, &all).unwrap();
        for (id, rendered, refused) in chapters(&report) {
            let refused = refused.unwrap_or_else(|| panic!("{id}: trusted under {args:?}"));
            assert!(refused.contains("steering"), "{id}: {refused}");
            assert!(rendered > 0);
        }
    }
    std::fs::remove_dir_all(&d).ok();
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC4: a manifest written before the book took render options (no `render` field; committed from a render by the
/// book before this change) is a render with no instruction and no steering: trusted when the run has neither, and
/// refused naming the difference when it has either.
#[test]
fn a_manifest_written_before_render_options_resumes_as_none() {
    use ggtts_book::manifest::{Manifest, ManifestError};
    use ggtts_book::plan::plan_book;
    let text = include_str!("fixtures/manifest-before-render-options.json");
    let body: Value = serde_json::from_str::<Value>(text).unwrap()["manifest"].clone();
    assert!(body.get("render").is_none(), "the fixture predates the field");
    let book = ggtts_book::Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&ggtts_qwen3_tts::PROFILE, &book, body["seed_root"].as_str().unwrap()).unwrap();
    let mut plan = plans
        .into_iter()
        .find(|p| p.chapter_id == body["chapter_id"])
        .expect("the fixture's chapter is in the fixture book");
    plan.engine = body["engine"].clone();
    let m = Manifest::parse(text, &plan).expect("no instruction and no steering resumes");
    assert!(m.done.iter().all(Option::is_some));
    let refused = |render: Value| {
        let mut p = plan.clone();
        p.render = render;
        match Manifest::parse(text, &p) {
            Err(ManifestError::Mismatch { field, manifest, plan }) => (field, manifest, plan),
            other => panic!("expected a mismatch, got {other:?}"),
        }
    };
    assert_eq!(
        refused(serde_json::json!({"instruction": "calm"})),
        ("instruction", "none".to_string(), "calm".to_string())
    );
    let steer = serde_json::json!({"sha256": "ab", "layers": [0], "scale": 1.0});
    let (field, manifest, _) = refused(serde_json::json!({ "steer": steer }));
    assert_eq!((field, manifest.as_str()), ("steering", "none"));
}
