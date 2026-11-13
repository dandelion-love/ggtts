//! The single entry point and its chunk step on the fake engine: `render_plan` gives the same read under every
//! combination of trace level, tensor hooks and output mode; the levelled stream is the stream leveller over the
//! raw pieces, labelled with their chunks; `render_chunk` takes its seeds and its judge from the caller; and replay,
//! rebuilt on the step, still reads a trace cut before its last `trim`.

mod common;

use std::cell::Cell;
use std::rc::Rc;

use ggtts_core::audio::{Judgement, Verdict};
use ggtts_core::engine::Engine;
use ggtts_core::hooks::{Hooks, Step, TensorHook, Wants};
use ggtts_core::pipeline::{
    apply_gain, render_chunk, render_plan, render_story, render_story_streamed, DurationJudge, Given, Judge, Observers,
    Output, Read, RenderOptions, StorySeeds, StreamMode, StreamPiece,
};
use ggtts_core::profile::Profile;
use ggtts_core::seed::seed_of;
use ggtts_core::stream::{StreamLevel, DEFAULT_HOLD_S};
use ggtts_core::trace::{emit_header, replay, Header, NullEvents, Recorder};

/// Counts the frames it is handed and the logits it reads.
struct Logits {
    frames: Rc<Cell<usize>>,
}

impl TensorHook for Logits {
    fn name(&self) -> &str {
        "logits"
    }
    fn wants(&self) -> Wants {
        Wants {
            p_stop: false,
            logits: true,
        }
    }
    fn step(&mut self, s: &Step<'_>) {
        assert!(s.logits.is_some(), "a hook that wants logits is given them");
        self.frames.set(self.frames.get() + 1);
    }
}

/// A piece as a sink saw it: its chunk, its first sample in the read, and its samples' bits.
type Piece = (usize, usize, Vec<u32>);

fn bits(s: &[f32]) -> Vec<u32> {
    s.iter().map(|x| x.to_bits()).collect()
}

/// The raw pieces as `render_story_streamed` gives them, the reference every stream is checked against.
fn raw_pieces(case: &common::Case) -> Vec<(usize, usize, Vec<f32>)> {
    let mut e = case.engine();
    let (chunks, opts) = (case.chunks(&e), case.opts(&e));
    let mut out = Vec::new();
    render_story_streamed(&mut e, &case.story_id, &chunks, &opts, &mut NullEvents, &mut |p| {
        out.push((p.chunk, p.start, p.samples.to_vec()));
        Ok(())
    })
    .unwrap();
    out
}

/// The levelled stream the leveller makes of the raw pieces: each released piece keeps its chunk and start.
fn levelled_reference(
    profile: &Profile,
    raw: &[(usize, usize, Vec<f32>)],
    rate: u32,
    hold_s: f64,
) -> (Vec<Piece>, Option<f64>) {
    let mut lv = StreamLevel::new(profile, hold_s);
    let mut held = std::collections::VecDeque::new();
    let mut out = Vec::new();
    for (chunk, start, s) in raw {
        held.push_back((*chunk, *start));
        for p in lv.push(s, rate) {
            let (c, st) = held.pop_front().unwrap();
            out.push((c, st, bits(&p)));
        }
    }
    for p in lv.finish() {
        let (c, st) = held.pop_front().unwrap();
        out.push((c, st, bits(&p)));
    }
    assert!(held.is_empty());
    (out, lv.gain())
}

fn same_read(a: &Read, b: &Read, what: &str) {
    assert_eq!(bits(&a.samples), bits(&b.samples), "{what}: samples");
    assert_eq!(a.chunks, b.chunks, "{what}: chunk records");
    assert_eq!(a.gain, b.gain, "{what}: gain");
    assert_eq!(a.frames_generated, b.frames_generated, "{what}: frames");
}

/// AC1: every combination of {no trace, level 1, level 2} x {no hooks, a logits hook} x {whole, stream raw, stream
/// levelled} renders the same read, including the combinations no `render_story*` function offered (hooks with a
/// stream), and each stream is exactly its reference.
#[test]
fn every_combination_of_observers_and_output_gives_the_same_read() {
    let mut hooked_streams = 0;
    for case in common::cases() {
        let mut e = case.engine();
        let profile = *e.profile();
        let (chunks, opts) = (case.chunks(&e), case.opts(&e));
        let plain = render_story(&mut e, &case.story_id, &chunks, &opts).unwrap();
        let raw = raw_pieces(&case);
        let raw_bits: Vec<Piece> = raw.iter().map(|(c, s, p)| (*c, *s, bits(p))).collect();
        let (levelled, levelled_gain) = levelled_reference(&profile, &raw, plain.sample_rate, DEFAULT_HOLD_S);
        for level in [0u8, 1, 2] {
            for hooked in [false, true] {
                for mode in [
                    None,
                    Some(StreamMode::Raw),
                    Some(StreamMode::Levelled { hold_s: DEFAULT_HOLD_S }),
                ] {
                    let what = format!("{} level {level} hooked {hooked} {mode:?}", case.id);
                    let mut e = case.engine();
                    let mut rec = Recorder::new(level);
                    if level > 0 {
                        emit_header(&mut rec, &Header::for_engine(&e, level));
                    }
                    let frames = Rc::new(Cell::new(0));
                    let mut hooks = Hooks::new();
                    if hooked {
                        hooks.add(Box::new(Logits { frames: frames.clone() }));
                    }
                    let mut pieces: Vec<Piece> = Vec::new();
                    let mut sink = |p: StreamPiece<'_>| {
                        assert_eq!(p.sample_rate, plain.sample_rate);
                        pieces.push((p.chunk, p.start, bits(p.samples)));
                        Ok(())
                    };
                    let output = match mode {
                        None => Output::Whole,
                        Some(level) => Output::Stream { sink: &mut sink, level },
                    };
                    let seeds = StorySeeds {
                        story_id: &case.story_id,
                        salt: &opts.seed_salt,
                    };
                    let read = render_plan(
                        &mut e,
                        &chunks,
                        &seeds,
                        &mut DurationJudge,
                        &opts,
                        &mut Observers::new(&mut rec, &mut hooks),
                        output,
                    )
                    .unwrap();
                    hooks.finish();
                    assert!(hooks.failures().is_empty(), "{what}: {:?}", hooks.failures());
                    same_read(&read, &plain, &what);
                    assert_eq!(hooked, frames.get() > 0, "{what}: the hook saw the frames");
                    if hooked {
                        assert_eq!(frames.get() as u64, read.frames_generated, "{what}: one step per frame");
                    }
                    assert_eq!(level > 0, !rec.lines.is_empty(), "{what}: the trace");
                    match mode {
                        None => {
                            assert!(pieces.is_empty());
                            assert_eq!(read.stream, None);
                        }
                        Some(StreamMode::Raw) => {
                            assert_eq!(pieces, raw_bits, "{what}: the raw stream");
                            assert_eq!(read.stream.unwrap().gain, None);
                        }
                        Some(StreamMode::Levelled { .. }) => {
                            assert_eq!(pieces, levelled, "{what}: the levelled stream");
                            assert_eq!(read.stream.unwrap().gain, levelled_gain, "{what}");
                        }
                    }
                    if hooked && mode.is_some() {
                        hooked_streams += 1;
                    }
                }
            }
        }
    }
    assert!(hooked_streams > 0);
}

/// The levelled stream is levelled by the hold, not by the whole read's gain: on at least one case the two differ,
/// so the comparison above would see a stream levelled with the read's gain.
#[test]
fn the_levelled_stream_is_not_the_read_levelled() {
    let mut differ = 0;
    for case in common::cases() {
        let e = case.engine();
        let profile = *e.profile();
        let raw = raw_pieces(&case);
        let mut e = case.engine();
        let (chunks, opts) = (case.chunks(&e), case.opts(&e));
        let read = render_story(&mut e, &case.story_id, &chunks, &opts).unwrap();
        let (levelled, _) = levelled_reference(&profile, &raw, read.sample_rate, DEFAULT_HOLD_S);
        let whole: Vec<u32> = raw
            .iter()
            .flat_map(|(_, _, s)| {
                let mut s = s.clone();
                if let Some(g) = read.gain {
                    apply_gain(&profile, &mut s, g);
                }
                bits(&s)
            })
            .collect();
        let held: Vec<u32> = levelled.into_iter().flat_map(|(_, _, s)| s).collect();
        if held != whole {
            differ += 1;
        }
    }
    assert!(differ > 0, "some case's hold is shorter than its read");
}

/// Rejects every draw, or none.
struct Fixed(Option<Verdict>);

impl Judge for Fixed {
    fn judge(&mut self, _: &Profile, _: &[f32], _: u32, _: &str) -> Judgement {
        Judgement {
            verdict: self.0,
            s_per_char: 0.0,
        }
    }
}

/// `render_chunk` asks the caller's seed source for every draw and keeps what the caller's judge decides: a judge
/// that rejects every draw takes all `redraw + 1` draws and keeps the last, one that rejects none keeps draw 0.
#[test]
fn the_step_takes_its_seeds_and_its_judge_from_the_caller() {
    let case = &common::cases()[0];
    let e = case.engine();
    let text = case.chunks(&e)[0].text.clone();
    let opts = RenderOptions {
        redraw: 3,
        ..RenderOptions::for_profile(e.profile())
    };
    let seeds = |i: usize, t: &str, k: u32| seed_of("caller", i, t, k, "salt") ^ 0x5a5a;
    for (judge, n) in [(Fixed(Some(Verdict::Runaway)), 4usize), (Fixed(None), 1)] {
        let mut judge = judge;
        let mut e = case.engine();
        let mut rec = Recorder::new(1);
        let mut hooks = Hooks::new();
        let kept = render_chunk(
            &mut e,
            &text,
            7,
            &seeds,
            &mut judge,
            &opts,
            &mut Observers::new(&mut rec, &mut hooks),
        )
        .unwrap();
        assert_eq!(kept.draws.len(), n);
        for (k, d) in kept.draws.iter().enumerate() {
            assert_eq!((d.draw, d.seed), (k as u32, seeds(7, &text, k as u32)));
            assert_eq!(d.judgement.verdict, judge.0);
        }
        let begins: Vec<(u64, u64, u64)> = rec
            .lines
            .iter()
            .filter(|l| l["ev"] == "chunk_begin")
            .map(|l| {
                (
                    l["chunk"].as_u64().unwrap(),
                    l["draw"].as_u64().unwrap(),
                    l["seed"].as_u64().unwrap(),
                )
            })
            .collect();
        let want: Vec<(u64, u64, u64)> = (0..n as u32)
            .map(|k| (7, u64::from(k), u64::from(seeds(7, &text, k))))
            .collect();
        assert_eq!(begins, want, "the requests carry the caller's chunk index and seeds");
        assert_eq!(
            rec.lines.iter().filter(|l| l["ev"] == "trim").count(),
            1,
            "one trim, of the kept draw"
        );
        assert_eq!(kept.kept().draw, n as u32 - 1);
    }
    // `Given` serves every draw one seed.
    let mut e = case.engine();
    let mut hooks = Hooks::new();
    let kept = render_chunk(
        &mut e,
        &text,
        0,
        &Given(42),
        &mut Fixed(Some(Verdict::Silent)),
        &opts,
        &mut Observers::new(&mut NullEvents, &mut hooks),
    )
    .unwrap();
    assert!(kept.draws.iter().all(|d| d.seed == 42));
}

/// A trace cut after a chunk's draws and before its `trim` (a render that failed there) replays as it did before
/// replay was rebuilt on the step: the step's own `trim` is not compared against one the trace does not have.
#[test]
fn a_trace_cut_before_its_last_trim_still_replays() {
    for case in common::cases() {
        let lines = case.trace(2);
        let last_trim = lines.iter().rposition(|l| l["ev"] == "trim").unwrap();
        let cut = &lines[..last_trim];
        let report = replay(&mut case.engine(), cut).unwrap();
        assert_eq!(report.mismatch, None, "{}", case.id);
        let begins = cut.iter().filter(|l| l["ev"] == "chunk_begin").count();
        assert_eq!(report.draws, begins, "{}", case.id);
    }
}

/// Observers at level 0 are given nothing: a recorder at level 0 records no event.
#[test]
fn level_zero_observers_record_nothing() {
    let case = &common::cases()[0];
    let mut e = case.engine();
    let (chunks, opts) = (case.chunks(&e), case.opts(&e));
    let mut rec = Recorder::new(0);
    let mut hooks = Hooks::new();
    let seeds = StorySeeds {
        story_id: &case.story_id,
        salt: &opts.seed_salt,
    };
    let obs = &mut Observers::new(&mut rec, &mut hooks);
    assert_eq!(obs.events.level(), 0);
    render_plan(&mut e, &chunks, &seeds, &mut DurationJudge, &opts, obs, Output::Whole).unwrap();
    assert!(rec.lines.is_empty());
}
