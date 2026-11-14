//! Story 3, AC1: the walking skeleton. A fixture book rendered end to end through the fake engine: chapters,
//! seeds, manifests, resume, one FLAC per chapter. No model is needed.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use ggtts_book::manifest::{Manifest, MANIFEST_FILE};
use ggtts_book::plan::{plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::render::{render_book, ChapterOutcome, ChapterReport};
use ggtts_book::Book;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::profile::Profile;
use ggtts_fake::FakeEngine;
use ggtts_qwen3_tts::PROFILE;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ggtts-book-render-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// The fake engine, counting generate calls and failing at the `fail_at`-th one (counted from 0), as a killed
/// process would stop.
///
/// Each chunk is rendered by a fresh fake engine, so a chunk's audio does not depend on what was rendered before it in
/// the same process. `ggtts-fake` alone keeps history (its output alternates with the number of texts it has seen),
/// which made a resume compare equal only at even kill points (review finding 212-A). Resuming to identical chapters
/// holds for an engine without such history; the docs say so.
struct Counting {
    shown: FakeEngine,
    pending: Option<FakeEngine>,
    calls: usize,
    fail_at: Option<usize>,
    /// Every request's (text, seed), in order.
    requests: Vec<(String, u64)>,
}

impl Counting {
    fn new(fail_at: Option<usize>) -> Self {
        Counting {
            shown: FakeEngine::default(),
            pending: None,
            calls: 0,
            fail_at,
            requests: Vec::new(),
        }
    }
}

impl Engine for Counting {
    fn info(&self) -> &EngineInfo {
        self.shown.info()
    }
    fn profile(&self) -> &Profile {
        self.shown.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        let n = self.calls;
        self.calls += 1;
        self.requests.push((req.text.to_string(), req.seed));
        if self.fail_at == Some(n) {
            return Err(EngineError("killed".into()));
        }
        let mut fresh = FakeEngine::default();
        let generated = fresh.generate(req)?;
        self.pending = Some(fresh);
        Ok(generated)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.pending.take().expect("decode follows generate").decode(codes)
    }
}

fn render(engine: &mut dyn Engine, out: &Path) -> Result<Vec<ChapterReport>, String> {
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    render_book(engine, &book, &plans, out)
}

/// Each written chapter's id and the SHA-256 of its FLAC's bytes.
fn chapter_files(reports: &[ChapterReport]) -> Vec<(String, String)> {
    use ggtts_core::seed::{hex, sha256};
    reports
        .iter()
        .filter_map(|r| match &r.outcome {
            ChapterOutcome::Written { file, .. } => Some((r.id.clone(), hex(&sha256(&std::fs::read(file).unwrap())))),
            ChapterOutcome::Skipped { .. } => None,
        })
        .collect()
}

fn total_chunks() -> usize {
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    book.chapters
        .iter()
        .zip(&plans)
        .filter(|(c, _)| !c.skippable)
        .map(|(_, p)| p.chunks.len())
        .sum()
}

/// The skeleton: every chapter that is not front or back matter becomes one FLAC that decodes, with a manifest
/// whose every chunk is done; front and back matter is left out and says why.
#[test]
fn a_book_renders_end_to_end_through_the_fake_engine() {
    let out = dir("skeleton");
    let mut engine = Counting::new(None);
    let reports = render(&mut engine, &out).unwrap();
    let written: Vec<&str> = reports
        .iter()
        .filter(|r| matches!(r.outcome, ChapterOutcome::Written { .. }))
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(written, ["part1", "ch1", "ch2a", "part2", "part2#c3", "part2#c4"]);
    let skipped: Vec<(&str, &str)> = reports
        .iter()
        .filter_map(|r| match &r.outcome {
            ChapterOutcome::Skipped { reason } => Some((r.id.as_str(), reason.as_str())),
            ChapterOutcome::Written { .. } => None,
        })
        .collect();
    assert_eq!(
        skipped,
        [
            ("titlepage", "epub:type titlepage"),
            ("copyright", "epub:type copyright-page"),
            ("nav", "epub:type toc"),
            ("colophon", "epub:type colophon")
        ]
    );
    // One engine for the whole book, one generate per chunk.
    assert_eq!(engine.calls, total_chunks());
    for r in &reports {
        if let ChapterOutcome::Written {
            file,
            rendered,
            resumed,
            refused,
        } = &r.outcome
        {
            assert_eq!(*resumed, 0);
            assert!(refused.is_none());
            assert!(!rendered.is_empty());
            let mut flac = claxon::FlacReader::open(file).unwrap();
            assert_eq!(flac.streaminfo().channels, 1);
            let n = flac.samples().count();
            assert!(n > 0, "{} is empty", file.display());
            let stem = file.file_stem().unwrap().to_string_lossy().into_owned();
            let book = Book::read(nested().bytes()).unwrap();
            let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
            let plan = plans.iter().find(|p| p.chapter_id == r.id).unwrap();
            let m = Manifest::load(&out.join(&stem), plan).unwrap();
            assert!(m.done.iter().all(Option::is_some), "{} not finished", r.id);
            assert!(out.join(&stem).join(MANIFEST_FILE).exists());
        }
    }
    std::fs::remove_dir_all(&out).ok();
}

/// The same book and seed root give byte-identical chapters.
#[test]
fn the_same_seed_root_gives_identical_chapters() {
    let (a, b) = (dir("same-a"), dir("same-b"));
    let first = chapter_files(&render(&mut Counting::new(None), &a).unwrap());
    let second = chapter_files(&render(&mut Counting::new(None), &b).unwrap());
    assert_eq!(first.len(), 6);
    assert_eq!(first, second);
    std::fs::remove_dir_all(&a).ok();
    std::fs::remove_dir_all(&b).ok();
}

/// The engine is asked for every spoken chunk, in reading order, at exactly its planned first-draw seed; another seed
/// root changes every seed it is asked for. (The fake engine's tone does not depend on the seed, so this is checked
/// at the request, not in the audio.)
#[test]
fn the_planned_seed_reaches_the_engine() {
    let book = Book::read(nested().bytes()).unwrap();
    for (root, name) in [(DEFAULT_SEED_ROOT, "seed-a"), ("another root", "seed-b")] {
        let out = dir(name);
        let plans = plan_book(&PROFILE, &book, root).unwrap();
        let mut engine = Counting::new(None);
        render_book(&mut engine, &book, &plans, &out).unwrap();
        let expected: Vec<(String, u64)> = book
            .chapters
            .iter()
            .zip(&plans)
            .filter(|(c, _)| !c.skippable)
            .flat_map(|(_, p)| p.chunks.iter().map(|c| (c.text.clone(), u64::from(c.seed))))
            .collect();
        assert_eq!(engine.requests, expected, "seed root {root}");
        std::fs::remove_dir_all(&out).ok();
    }
    let a = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let b = plan_book(&PROFILE, &book, "another root").unwrap();
    let seeds = |plans: &[ggtts_book::plan::ChapterPlan]| -> Vec<u32> {
        plans.iter().flat_map(|p| p.chunks.iter().map(|c| c.seed)).collect()
    };
    assert!(seeds(&a).iter().zip(seeds(&b)).all(|(x, y)| *x != y));
}

/// A render killed partway resumes, renders only what was left, and ends byte-identical to an uninterrupted one, at
/// every kill point: the first chunk, odd and even points, either side of the middle, and the last chunk.
#[test]
fn a_killed_render_resumes_to_the_same_chapters() {
    let clean = dir("clean");
    let expected = chapter_files(&render(&mut Counting::new(None), &clean).unwrap());
    let total = total_chunks();
    assert!(total >= 6, "the fixture should have several chunks, has {total}");

    for kill_at in [0, 1, 2, 3, total / 2 - 1, total / 2, total / 2 + 1, total - 1] {
        let out = dir(&format!("killed-{kill_at}"));
        let err = render(&mut Counting::new(Some(kill_at)), &out).unwrap_err();
        assert!(err.contains("killed"), "{err}");

        let mut engine = Counting::new(None);
        let reports = render(&mut engine, &out).unwrap();
        // Exactly the chunks the killed run did not finish: a resume that renders a finished chunk again fails here.
        assert_eq!(engine.calls, total - kill_at, "kill at {kill_at}");
        let resumed: usize = reports
            .iter()
            .map(|r| match &r.outcome {
                ChapterOutcome::Written { resumed, .. } => *resumed,
                ChapterOutcome::Skipped { .. } => 0,
            })
            .sum();
        assert_eq!(resumed, kill_at, "kill at {kill_at}");
        assert_eq!(chapter_files(&reports), expected, "kill at {kill_at}");
        std::fs::remove_dir_all(&out).ok();
    }
    std::fs::remove_dir_all(&clean).ok();
}

/// A chunk rendered alone (a fresh engine, nothing rendered before it) is the same file as that chunk rendered in a
/// full run, for every chunk of a chapter.
#[test]
fn a_chunk_rendered_alone_equals_it_in_a_full_run() {
    let clean = dir("alone");
    render(&mut Counting::new(None), &clean).unwrap();
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let (i, plan) = book
        .chapters
        .iter()
        .zip(&plans)
        .enumerate()
        .find(|(_, (c, _))| c.id == "ch1")
        .map(|(i, (_, p))| (i, p))
        .unwrap();
    let stem = ggtts_book::render::chapter_stem(i, &book.chapters[i]);
    for index in (0..plan.chunks.len()).rev() {
        let mut engine = Counting::new(None);
        let opts = ggtts_book::render::BookRender::for_profile(engine.profile()).opts;
        let mut hooks = ggtts_core::hooks::Hooks::new();
        let mut events = ggtts_core::trace::NullEvents;
        let mut obs = ggtts_core::pipeline::Observers::new(&mut events, &mut hooks);
        let alone = ggtts_book::render::render_chunk(&mut engine, plan, index, &opts, &mut obs).unwrap();
        let in_run = std::fs::read(clean.join(&stem).join(ggtts_book::manifest::chunk_file(index))).unwrap();
        assert!(
            alone == in_run,
            "chunk {index} rendered alone differs from the full run"
        );
    }
    std::fs::remove_dir_all(&clean).ok();
}
