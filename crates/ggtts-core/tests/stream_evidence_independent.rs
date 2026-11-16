//! Independent evidence-to-actual-stream and kept-draw checks.
mod common;

use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{
    render_plan, DurationJudge, Observers, Output, RenderError, StorySeeds, StreamEvidencePiece, StreamMode,
    StreamPiece,
};
use ggtts_core::trace::NullEvents;

#[derive(Clone, Debug)]
struct Evidence {
    chunk: usize,
    text: String,
    seed: u64,
    draw: u32,
    stop: ggtts_core::engine::Stop,
    start: usize,
    speech_end: usize,
    samples: Vec<f32>,
    sample_rate: u32,
}

fn redraw_case() -> common::Case {
    let mut case = common::cases()
        .into_iter()
        .find(|c| c.id == "arm-redraw-exhausted")
        .unwrap();
    let engine = case.engine();
    let first = case.chunks(&engine)[0].text.clone();
    case.script.insert(first, vec![0.3, 3.0]);
    case.redraw = 1;
    case
}

fn run(case: &common::Case, mode: StreamMode) -> (Vec<Evidence>, ggtts_core::pipeline::Read, ggtts_fake::FakeEngine) {
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    let seeds = StorySeeds {
        story_id: &case.story_id,
        salt: &opts.seed_salt,
    };
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let mut pieces = Vec::new();
    let mut sink = |p: StreamEvidencePiece<'_>| {
        pieces.push(Evidence {
            chunk: p.chunk,
            text: p.text.to_string(),
            seed: p.seed,
            draw: p.draw,
            stop: p.stop,
            start: p.start,
            speech_end: p.speech_end,
            samples: p.samples.to_vec(),
            sample_rate: p.sample_rate,
        });
        Ok(())
    };
    let read = render_plan(
        &mut engine,
        &chunks,
        &seeds,
        &mut DurationJudge,
        &opts,
        &mut Observers::new(&mut events, &mut hooks),
        Output::StreamEvidence {
            sink: &mut sink,
            level: mode,
        },
    )
    .unwrap();
    (pieces, read, engine)
}

fn legacy(case: &common::Case, mode: StreamMode) -> Vec<(usize, usize, Vec<f32>, u32)> {
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    let seeds = StorySeeds {
        story_id: &case.story_id,
        salt: &opts.seed_salt,
    };
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let mut pieces = Vec::new();
    let mut sink = |p: StreamPiece<'_>| {
        pieces.push((p.chunk, p.start, p.samples.to_vec(), p.sample_rate));
        Ok(())
    };
    render_plan(
        &mut engine,
        &chunks,
        &seeds,
        &mut DurationJudge,
        &opts,
        &mut Observers::new(&mut events, &mut hooks),
        Output::Stream {
            sink: &mut sink,
            level: mode,
        },
    )
    .unwrap();
    pieces
}

#[test]
fn kept_draw_and_emitted_pcm_match_legacy_raw_and_held() {
    let case = redraw_case();
    let mut saw_redraw = false;
    let mut saw_gap = false;
    for mode in [StreamMode::Raw, StreamMode::Levelled { hold_s: 5.0 }] {
        let expected = legacy(&case, mode);
        let (pieces, read, engine) = run(&case, mode);
        assert_eq!(pieces.len(), read.chunks.len());
        assert_eq!(pieces.len(), expected.len());
        let mut next = 0;
        for (i, (p, expected)) in pieces.iter().zip(expected).enumerate() {
            let c = &read.chunks[i];
            assert_eq!((p.chunk, p.start, p.sample_rate), (expected.0, expected.1, expected.3));
            assert_eq!(p.samples, expected.2, "actual emitted PCM");
            assert_eq!(p.start, next);
            assert_eq!(
                (p.chunk, p.text.as_str(), p.seed, p.draw, p.stop, p.start, p.speech_end),
                (i, c.text.as_str(), c.seed, c.redraws, c.stop, c.span.0, c.span.1)
            );
            assert!(p.speech_end >= p.start && p.speech_end <= p.start + p.samples.len());
            assert!(p.samples[p.speech_end - p.start..].iter().all(|v| *v == 0.0));
            saw_gap |= p.speech_end < p.start + p.samples.len();
            let calls: Vec<_> = engine.calls.iter().filter(|x| x.text == c.text).collect();
            assert_eq!(calls.len(), c.redraws as usize + 1);
            assert_eq!(p.seed, calls.last().unwrap().seed);
            if c.redraws > 0 {
                assert_ne!(p.seed, calls.first().unwrap().seed);
                saw_redraw = true;
            }
            next += p.samples.len();
        }
        // A changed seed or shifted speech boundary must fail the same equality oracle.
        let mut planted = pieces[0].clone();
        planted.seed ^= 1;
        assert_ne!(planted.seed, read.chunks[0].seed);
        planted.speech_end += 1;
        assert_ne!(planted.speech_end, read.chunks[0].span.1);
    }
    assert!(saw_redraw && saw_gap, "fixture must exercise a redraw and a gap");
}

#[test]
fn failing_sink_stops_the_evidence_stream() {
    let case = redraw_case();
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    let seeds = StorySeeds {
        story_id: &case.story_id,
        salt: &opts.seed_salt,
    };
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let mut calls = 0;
    let mut sink = |_: StreamEvidencePiece<'_>| {
        calls += 1;
        Err("closed".into())
    };
    let result = render_plan(
        &mut engine,
        &chunks,
        &seeds,
        &mut DurationJudge,
        &opts,
        &mut Observers::new(&mut events, &mut hooks),
        Output::StreamEvidence {
            sink: &mut sink,
            level: StreamMode::Raw,
        },
    );
    assert!(matches!(result, Err(RenderError::Sink(_))));
    assert_eq!(calls, 1);
    assert!(engine.calls.iter().all(|c| c.text == chunks[0].text));
}

struct AbortEngine(ggtts_fake::FakeEngine);
impl ggtts_core::engine::Engine for AbortEngine {
    fn info(&self) -> &ggtts_core::engine::EngineInfo {
        self.0.info()
    }
    fn profile(&self) -> &ggtts_core::profile::Profile {
        self.0.profile()
    }
    fn generate(
        &mut self,
        req: &ggtts_core::engine::ChunkRequest<'_>,
    ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
        let mut generated = self.0.generate(req)?;
        generated.stop = ggtts_core::engine::Stop::Abort;
        Ok(generated)
    }
    fn decode(
        &mut self,
        codes: &ggtts_core::engine::Codes,
    ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
        self.0.decode(codes)
    }
}

#[test]
fn abort_and_mixed_rates_refuse_complete_evidence() {
    let case = redraw_case();
    let probe = case.engine();
    let chunks = case.chunks(&probe);
    assert!(chunks.len() > 1);
    let opts = case.opts(&probe);
    let seeds = StorySeeds {
        story_id: &case.story_id,
        salt: &opts.seed_salt,
    };
    let mut aborted = AbortEngine(case.engine());
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let mut delivered = 0;
    let mut sink = |_: StreamEvidencePiece<'_>| {
        delivered += 1;
        Ok(())
    };
    let result = render_plan(
        &mut aborted,
        &chunks,
        &seeds,
        &mut DurationJudge,
        &opts,
        &mut Observers::new(&mut events, &mut hooks),
        Output::StreamEvidence {
            sink: &mut sink,
            level: StreamMode::Raw,
        },
    );
    assert!(result.is_err(), "an aborted kept draw cannot complete evidence");
    assert_eq!(delivered, 0);

    let mut shapes = case.shape.clone();
    shapes.entry(chunks[1].text.clone()).or_default().sample_rate = Some(16000);
    let mut mixed = ggtts_fake::FakeEngine::new(case.script.clone(), shapes);
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let mut delivered = 0;
    let mut sink = |_: StreamEvidencePiece<'_>| {
        delivered += 1;
        Ok(())
    };
    let result = render_plan(
        &mut mixed,
        &chunks,
        &seeds,
        &mut DurationJudge,
        &opts,
        &mut Observers::new(&mut events, &mut hooks),
        Output::StreamEvidence {
            sink: &mut sink,
            level: StreamMode::Raw,
        },
    );
    assert!(result.is_err(), "a changed sample rate cannot complete evidence");
    assert!(delivered < chunks.len());
}
