//! A book keeps each chunk's first draw: the core step runs at redraw 0 for a book, so a chunk the judge rejects is
//! kept as drawn, as before the book moved onto the step, and the output stays byte-identical.

mod common;

use std::sync::{Arc, Mutex};

use common::nested;
use ggtts_book::plan::{plan_book, DEFAULT_SEED_ROOT};
use ggtts_book::render::{book_render_options, render_book_observed, BookRender};
use ggtts_book::Book;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::pipeline::{DurationJudge, Judge, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;
use ggtts_qwen3_tts::PROFILE;

/// The fake engine with every draw twenty times too long, which the render spec's judge rejects as a runaway.
/// Records each request's (chunk, draw).
pub struct Runaway {
    inner: FakeEngine,
    pub draws: Arc<Mutex<Vec<(u32, u32)>>>,
}

impl Engine for Runaway {
    fn info(&self) -> &EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &Profile {
        self.inner.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.draws.lock().unwrap().push((req.chunk, req.draw));
        self.inner.generate(req)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        let mut pcm = self.inner.decode(codes)?;
        pcm.samples = pcm.samples.repeat(20);
        Ok(pcm)
    }
}

#[test]
fn a_book_keeps_the_first_draw_of_a_chunk_the_judge_rejects() {
    let out = std::env::temp_dir().join(format!("ggtts-book-first-draw-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let draws = Arc::new(Mutex::new(Vec::new()));
    let mut engine = Runaway {
        inner: FakeEngine::default(),
        draws: draws.clone(),
    };
    // The judge does reject these draws, so a book that redrew would make more than one per chunk.
    let mut probe = Runaway {
        inner: FakeEngine::default(),
        draws: Arc::new(Mutex::new(Vec::new())),
    };
    let text = &plans.iter().find(|p| !p.chunks.is_empty()).unwrap().chunks[0].text;
    let req = ChunkRequest {
        text,
        seed: 1,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let generated = probe.generate(&req).unwrap();
    let pcm = probe.decode(&generated.codes).unwrap();
    let verdict = DurationJudge
        .judge(&PROFILE, &pcm.samples, pcm.sample_rate, text)
        .verdict;
    assert!(verdict.is_some(), "the judge rejects the runaway engine's draws");

    // Render options that ask for redraws (the qwen3-tts profile's own default is 0) still give a book draw 0 only.
    let settings = BookRender {
        include_skippable: false,
        opts: book_render_options(RenderOptions {
            redraw: 3,
            ..RenderOptions::for_profile(&PROFILE)
        }),
    };
    render_book_observed(&mut engine, &book, &plans, &out, &settings, &mut NullEvents).unwrap();
    let draws = draws.lock().unwrap().clone();
    assert!(draws.len() > 3);
    assert!(
        draws.iter().all(|&(_, d)| d == 0),
        "a book draws each chunk once: {draws:?}"
    );
    std::fs::remove_dir_all(&out).ok();
}
