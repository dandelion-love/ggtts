//! A book judges every chunk and redraws one the judge rejects, up to the render option `redraw` (`--redraw`, unset:
//! the engine profile's), through the same core chunk step and judge as a story. Each redraw is at the chunk's seed for
//! that draw. The manifest records each chunk's kept draw, its draw count and the kept draw's verdict, and a resume
//! keeps a done chunk only when a render at the new setting would end at the same draw.

mod common;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common::nested;
use ggtts_book::plan::{plan_book, ChapterPlan, DEFAULT_SEED_ROOT};
use ggtts_book::render::{render_command_options, LoadedEngine};
use ggtts_book::Book;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::pipeline::{DurationJudge, Judge};
use ggtts_core::profile::Profile;
use ggtts_fake::FakeEngine;
use ggtts_qwen3_tts::PROFILE;
use serde_json::Value;

/// Which draws come back twenty times too long, which the render spec's judge rejects as a runaway.
#[derive(Clone, Copy)]
enum Plant {
    /// Every chunk's first draw; redraws are clean.
    FirstDraw,
    /// Every draw.
    Always,
}

/// The fake engine with planted runaways, recording each request's (chunk, draw, seed). Each request gets a fresh
/// fake engine, so a chunk's audio does not depend on what was rendered before it.
///
/// The fake's own tone lasts 0.25 s per CJK character and has quiet edges, so on this English fixture its draws land on
/// either side of the judge's bars by text. Here each draw is a steady tone, with no edge to trim, whose length is set
/// by the judge's own character count (`judge_chars`): 0.3 s per character for a clean draw (inside the profile's
/// bars, 0.07 to 0.60), 1.2 s per character for a runaway.
struct Planted {
    shown: FakeEngine,
    pending: Option<FakeEngine>,
    plant: Plant,
    runaway: bool,
    chars: usize,
    seen: Arc<Mutex<Vec<(u32, u32, u64)>>>,
}

/// A steady 220 Hz tone of `seconds` at `rate`.
fn tone(seconds: f64, rate: u32) -> Vec<f32> {
    let n = (seconds * f64::from(rate)) as usize;
    (0..n)
        .map(|i| (0.3 * (std::f64::consts::TAU * 220.0 * i as f64 / f64::from(rate)).sin()) as f32)
        .collect()
}

impl Engine for Planted {
    fn info(&self) -> &EngineInfo {
        self.shown.info()
    }
    fn profile(&self) -> &Profile {
        self.shown.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.seen.lock().unwrap().push((req.chunk, req.draw, req.seed));
        self.chars = ggtts_core::audio::judge_chars(req.text);
        self.runaway = match self.plant {
            Plant::FirstDraw => req.draw == 0,
            Plant::Always => true,
        };
        let mut fresh = FakeEngine::default();
        let generated = fresh.generate(req)?;
        self.pending = Some(fresh);
        Ok(generated)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        let mut pcm = self.pending.take().expect("decode follows generate").decode(codes)?;
        let per_char = if self.runaway { 1.2 } else { 0.3 };
        pcm.samples = tone(per_char * self.chars as f64, pcm.sample_rate);
        Ok(pcm)
    }
}

struct Run {
    report: Value,
    /// Every request: (chunk, draw, seed).
    seen: Vec<(u32, u32, u64)>,
}

fn render(book: &Path, out: &Path, plant: Plant, args: &[&str]) -> Run {
    let mut all = vec![
        book.to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--engine".into(),
        "fake".into(),
    ];
    all.extend(args.iter().map(|s| s.to_string()));
    let (path, options) = ggtts_book::options::parse("render", &all, &|_| None).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let engine = Planted {
        shown: FakeEngine::default(),
        pending: None,
        plant,
        runaway: false,
        chars: 1,
        seen: seen.clone(),
    };
    let report = render_command_options(&path, &options, || {
        Ok(LoadedEngine {
            engine: Box::new(engine),
            files: Vec::new(),
            speaker: None,
        })
    })
    .unwrap();
    let seen = seen.lock().unwrap().clone();
    Run { report, seen }
}

fn temp(name: &str) -> (PathBuf, PathBuf) {
    let book = nested().write_temp(&format!("redraw-{name}"));
    let out = book.parent().unwrap().join("out");
    (book, out)
}

/// The planned chapters that are written, with their directories under `out`.
fn written(out: &Path) -> Vec<(ChapterPlan, PathBuf)> {
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    book.chapters
        .iter()
        .zip(plans)
        .enumerate()
        .filter(|(_, (c, p))| !c.skippable && !p.chunks.is_empty())
        .map(|(i, (c, p))| (p, out.join(ggtts_book::render::chapter_stem(i, c))))
        .collect()
}

/// Each chunk's manifest entry, chapter by chapter.
fn manifest_chunks(dir: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(dir.join(ggtts_book::manifest::MANIFEST_FILE)).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    v["manifest"]["chunks"].as_array().unwrap().clone()
}

fn total(run: &Run, key: &str) -> u64 {
    run.report["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c[key].as_u64())
        .sum()
}

/// The planted runaway is one the judge rejects, so the tests below can see a redraw.
#[test]
fn the_planted_draw_is_one_the_judge_rejects() {
    let mut e = Planted {
        shown: FakeEngine::default(),
        pending: None,
        plant: Plant::FirstDraw,
        runaway: false,
        chars: 1,
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let text = "The lamp was lit at dusk.";
    let req = ChunkRequest {
        text,
        seed: 1,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let g = e.generate(&req).unwrap();
    let pcm = e.decode(&g.codes).unwrap();
    assert!(DurationJudge
        .judge(&PROFILE, &pcm.samples, pcm.sample_rate, text)
        .verdict
        .is_some());
    let g = e.generate(&ChunkRequest { draw: 1, ..req }).unwrap();
    let pcm = e.decode(&g.codes).unwrap();
    assert!(DurationJudge
        .judge(&PROFILE, &pcm.samples, pcm.sample_rate, text)
        .verdict
        .is_none());
}

/// AC1 and AC4: a chunk whose first draw the judge rejects is redrawn, at the seed of draw 1, and the redraw is kept;
/// the manifest records the kept draw, the draw count and the kept draw's verdict for every chunk.
#[test]
fn a_chunk_the_judge_rejects_is_redrawn_at_its_next_seed() {
    let (book, out) = temp("first");
    let run = render(&book, &out, Plant::FirstDraw, &["--redraw", "2"]);
    let chapters = written(&out);
    let n: usize = chapters.iter().map(|(p, _)| p.chunks.len()).sum();
    assert_eq!(run.seen.len(), 2 * n, "each chunk is drawn twice: rejected, then kept");
    // The requests, chapter by chapter, are draw 0 then draw 1 of each chunk at the plan's seeds for those draws.
    let mut want = Vec::new();
    for (plan, _) in &chapters {
        for i in 0..plan.chunks.len() {
            for draw in 0..2 {
                want.push((i as u32, draw, u64::from(plan.seed_at(i, draw))));
            }
        }
    }
    assert_eq!(run.seen, want);
    for (plan, dir) in &chapters {
        let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m["manifest"]["redraw"], 2, "{}", plan.chapter_id);
        for c in manifest_chunks(dir) {
            assert_eq!(
                (c["kept_draw"].as_u64(), c["draws"].as_u64()),
                (Some(1), Some(2)),
                "{c}"
            );
            assert!(c["verdict"].is_null(), "{c}");
        }
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC1: a chunk the judge rejects on every draw gets exactly `redraw` redraws, and the last is kept with its verdict.
#[test]
fn a_chunk_rejected_on_every_draw_keeps_its_last_draw_with_the_verdict() {
    let (book, out) = temp("always");
    let run = render(&book, &out, Plant::Always, &["--redraw", "3"]);
    let n: usize = written(&out).iter().map(|(p, _)| p.chunks.len()).sum();
    assert_eq!(run.seen.len(), 4 * n);
    for (_, dir) in written(&out) {
        for c in manifest_chunks(&dir) {
            assert_eq!(
                (c["kept_draw"].as_u64(), c["draws"].as_u64()),
                (Some(3), Some(4)),
                "{c}"
            );
            assert_eq!(c["verdict"], "runaway", "{c}");
        }
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC3: at `--redraw 0` a rejected first draw is kept, one request per chunk, and the manifest carries no draw fields:
/// the same bytes the book wrote before it redrew (see the AC1 capture for the byte comparison).
#[test]
fn at_redraw_0_the_first_draw_is_kept_and_no_draw_is_recorded() {
    let (book, out) = temp("zero");
    let run = render(&book, &out, Plant::FirstDraw, &["--redraw", "0"]);
    assert!(run.seen.iter().all(|&(_, d, _)| d == 0));
    for (_, dir) in written(&out) {
        let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert!(m["manifest"].get("redraw").is_none());
        for c in manifest_chunks(&dir) {
            assert!(c.get("kept_draw").is_none() && c.get("draws").is_none(), "{c}");
        }
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC2: a resume keeps a done chunk's kept draw; a chunk recorded at a lower redraw setting is judged again under the
/// new one, not silently kept; a setting at which the recorded draws still hold renders nothing.
#[test]
fn a_resume_keeps_kept_draws_and_rejudges_chunks_recorded_at_a_lower_setting() {
    let (book, out) = temp("resume");
    let n = written(&out).iter().map(|(p, _)| p.chunks.len()).sum::<usize>() as u64;
    // Redraw 0: every first draw is kept, never judged into a redraw.
    let zero = render(&book, &out, Plant::FirstDraw, &["--redraw", "0"]);
    assert_eq!(zero.seen.len() as u64, n);
    // Redraw 1: every chunk was kept unjudged at a lower setting, so each is judged again and redrawn.
    let one = render(&book, &out, Plant::FirstDraw, &["--redraw", "1"]);
    assert_eq!(total(&one, "rejudged"), n, "{}", one.report);
    assert_eq!(total(&one, "rendered"), n);
    assert_eq!(one.seen.len() as u64, 2 * n);
    // Redraw 1 again, and redraw 3: each chunk's kept draw 1 passed the judge, so it holds; nothing is rendered.
    for setting in ["1", "3"] {
        let again = render(&book, &out, Plant::FirstDraw, &["--redraw", setting]);
        assert!(again.seen.is_empty(), "--redraw {setting} rendered {:?}", again.seen);
        assert_eq!(total(&again, "resumed"), n);
    }
    // Redraw 0 again: a fresh render would keep draw 0, so the chunks kept at draw 1 are rendered again.
    let back = render(&book, &out, Plant::FirstDraw, &["--redraw", "0"]);
    assert_eq!(total(&back, "rejudged"), n);
    assert!(back.seen.iter().all(|&(_, d, _)| d == 0));
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}

/// AC2: a chunk the judge rejected on every draw at a lower setting gets the further draws a higher one allows; at
/// the same setting it is kept with its verdict.
#[test]
fn a_chunk_rejected_at_every_draw_is_redrawn_further_when_the_setting_rises() {
    let (book, out) = temp("rise");
    let n = written(&out).iter().map(|(p, _)| p.chunks.len()).sum::<usize>() as u64;
    render(&book, &out, Plant::Always, &["--redraw", "1"]);
    let same = render(&book, &out, Plant::Always, &["--redraw", "1"]);
    assert!(same.seen.is_empty(), "a chunk kept at its last allowed draw holds");
    let higher = render(&book, &out, Plant::Always, &["--redraw", "2"]);
    assert_eq!(total(&higher, "rejudged"), n);
    assert_eq!(higher.seen.len() as u64, 3 * n);
    for (_, dir) in written(&out) {
        for c in manifest_chunks(&dir) {
            assert_eq!(c["kept_draw"].as_u64(), Some(2), "{c}");
        }
    }
    std::fs::remove_dir_all(book.parent().unwrap()).ok();
}
