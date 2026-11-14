//! The core chunk step (`ggtts_core::pipeline::render_chunk`) does what the book's own chunk renderer does, so the
//! book can move onto it with no change: at the chapter plan's seeds and redraw 0, every chunk of a fixture book gives
//! the same samples as a copy of the book's renderer (generate, decode, trim at the profile's `edge_s`), and the same
//! FLAC bytes as `ggtts_book::render::render_chunk` itself.

mod common;

use common::nested;
use ggtts_book::plan::{plan_book, ChapterPlan, DEFAULT_SEED_ROOT};
use ggtts_book::Book;
use ggtts_core::audio::trim_edges_f32;
use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::flac::encode_flac;
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{render_chunk, DurationJudge, Observers, RenderOptions};
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;
use ggtts_qwen3_tts::PROFILE;

/// A copy of the book's chunk renderer before it moved to the core step: generate, decode and edge-trim one
/// planned chunk at its first draw's seed. The trimmed samples, before conversion to 16 bits.
fn book_chunk_samples(engine: &mut dyn Engine, plan: &ChapterPlan, index: usize) -> (Vec<f32>, u32) {
    let chunk = &plan.chunks[index];
    let profile = *engine.profile();
    let req = ChunkRequest {
        text: &chunk.text,
        seed: u64::from(plan.seed_at(index, 0)),
        chunk: index as u32,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let generated = engine.generate(&req).unwrap();
    let pcm = engine.decode(&generated.codes).unwrap();
    let t = trim_edges_f32(&profile, &pcm.samples, pcm.sample_rate, profile.edge_s);
    (pcm.samples[t.start_sample..t.end_sample].to_vec(), pcm.sample_rate)
}

/// The book's conversion to 16 bits (a copy: it is private to the book).
fn to_i16(x: f32) -> i16 {
    (f64::from(x).clamp(-1.0, 1.0) * 32767.0).round() as i16
}

#[test]
fn the_core_step_renders_a_book_chunk_as_the_book_does() {
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    // Three engines fed the same chunks in the same order: the fake engine keeps history, so each sees the same.
    let (mut copy, mut core, mut live) = (FakeEngine::default(), FakeEngine::default(), FakeEngine::default());
    let opts = RenderOptions {
        redraw: 0,
        ..RenderOptions::for_profile(core.profile())
    };
    let mut n = 0;
    for plan in &plans {
        let seeds = |i: usize, _: &str, k: u32| plan.seed_at(i, k);
        for (i, chunk) in plan.chunks.iter().enumerate() {
            let (want, rate) = book_chunk_samples(&mut copy, plan, i);
            let mut hooks = Hooks::new();
            let kept = render_chunk(
                &mut core,
                &chunk.text,
                i,
                &seeds,
                &mut DurationJudge,
                &opts,
                &mut Observers::new(&mut NullEvents, &mut hooks),
            )
            .unwrap();
            let bits = |s: &[f32]| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&kept.samples), bits(&want), "{} chunk {i}", plan.chapter_id);
            assert_eq!(kept.sample_rate, rate);
            assert_eq!(kept.draws.len(), 1, "redraw 0 is one draw");
            assert_eq!(kept.kept().seed, plan.seed_at(i, 0));
            let pcm: Vec<i16> = kept.samples.iter().map(|&x| to_i16(x)).collect();
            let flac = encode_flac(&pcm, kept.sample_rate).unwrap();
            // The book's renderer, on the core step since it moved, with the options a book renders with by default.
            let book_opts = ggtts_book::render::BookRender::for_profile(live.profile()).opts;
            let mut live_hooks = Hooks::new();
            let book_flac = ggtts_book::render::render_chunk(
                &mut live,
                plan,
                i,
                &book_opts,
                &mut Observers::new(&mut NullEvents, &mut live_hooks),
            )
            .unwrap();
            assert_eq!(flac, book_flac, "{} chunk {i}: the book's stored FLAC", plan.chapter_id);
            n += 1;
        }
    }
    assert!(n > 3, "the fixture book has chunks to compare");
}

/// The fake engine with audio that depends on the request's seed and chunk number: the fake's own tone depends only
/// on the text, so a wrong seed or chunk number would pass the comparison above unseen.
struct SeedScaled {
    inner: FakeEngine,
    scale: f32,
}

impl Engine for SeedScaled {
    fn info(&self) -> &ggtts_core::engine::EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &ggtts_core::profile::Profile {
        self.inner.profile()
    }
    fn generate(
        &mut self,
        req: &ChunkRequest<'_>,
    ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
        self.scale = 0.25 + ((req.seed ^ u64::from(req.chunk) << 7) % 97) as f32 / 160.0;
        self.inner.generate(req)
    }
    fn decode(
        &mut self,
        codes: &ggtts_core::engine::Codes,
    ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
        let mut pcm = self.inner.decode(codes)?;
        pcm.samples.iter_mut().for_each(|x| *x *= self.scale);
        Ok(pcm)
    }
}

/// The book's renderer on the core step stores the same FLAC bytes as its renderer before the move, on an engine
/// whose audio moves with the seed and the chunk number.
#[test]
fn the_book_stores_the_same_chunk_bytes_on_a_seed_sensitive_engine() {
    let book = Book::read(nested().bytes()).unwrap();
    let plans = plan_book(&PROFILE, &book, DEFAULT_SEED_ROOT).unwrap();
    let engine = || SeedScaled {
        inner: FakeEngine::default(),
        scale: 1.0,
    };
    let (mut copy, mut live) = (engine(), engine());
    let opts = ggtts_book::render::BookRender::for_profile(live.profile()).opts;
    let mut distinct = std::collections::BTreeSet::new();
    for plan in &plans {
        for i in 0..plan.chunks.len() {
            let (want, rate) = book_chunk_samples(&mut copy, plan, i);
            let pcm: Vec<i16> = want.iter().map(|&x| to_i16(x)).collect();
            let want = encode_flac(&pcm, rate).unwrap();
            let mut hooks = Hooks::new();
            let got = ggtts_book::render::render_chunk(
                &mut live,
                plan,
                i,
                &opts,
                &mut Observers::new(&mut NullEvents, &mut hooks),
            )
            .unwrap();
            assert_eq!(got, want, "{} chunk {i}", plan.chapter_id);
            distinct.insert(want);
        }
    }
    // The engine does tell the chunks apart, so the comparison above can fail.
    assert!(distinct.len() > 4, "only {} distinct chunk files", distinct.len());
}
