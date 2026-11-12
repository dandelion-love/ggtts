//! The instrument interface: the fake engine and the pipeline emit every `ggtts-trace/2` event
//! with its documented meaning, a recorded trace replays, and tracing changes no output.
//!
//! Each check is shown to see what it claims to rule out: replay goes red on a planted token, depth-codebook or
//! engine change and stays green when only the clock moves; the trace-identity check goes red on a planted
//! instrument that draws from the sampler's stream, and names the chunk, frame and codebook.
//!
//! The stories are the recorded render cases' (`fixtures/pipeline/render.json`), with their scripts and shapes.

mod common;

use std::collections::HashMap;
use std::io;

use common::{cases, Case};

use ggtts_core::engine::{ChunkRequest, Engine};
use ggtts_core::pipeline::{render_story, render_story_traced};
use ggtts_core::seed::seed_of;
use ggtts_core::trace::{
    check_trace_identity, compare, emit_header, parse, replay, Events, Header, JsonlEvents, NullEvents, Recorder,
    FORMAT,
};
use ggtts_fake::{fake_p_stop, fake_token, FakeEngine, CODEBOOKS};
use serde_json::{Map, Value};

fn ev(m: &Map<String, Value>) -> &str {
    m["ev"].as_str().unwrap()
}

fn num(m: &Map<String, Value>, k: &str) -> f64 {
    m[k].as_f64().unwrap_or_else(|| panic!("{k} missing in {m:?}"))
}

#[test]
fn tracing_changes_nothing_in_the_read() {
    for case in cases() {
        let mut plain = case.engine();
        let chunks = case.chunks(&plain);
        let opts = case.opts(&plain);
        let a = render_story(&mut plain, &case.story_id, &chunks, &opts).unwrap();
        for level in [0u8, 1, 2] {
            let mut traced = case.engine();
            let mut rec = Recorder::new(level);
            let b = render_story_traced(&mut traced, &case.story_id, &chunks, &opts, &mut rec).unwrap();
            assert_eq!(a.samples, b.samples, "{}: samples differ at level {level}", case.id);
            assert_eq!(a.chunks, b.chunks, "{}: chunk records differ at level {level}", case.id);
            assert_eq!(
                plain.calls, traced.calls,
                "{}: engine calls differ at level {level}",
                case.id
            );
            if level == 0 {
                assert!(
                    rec.lines.is_empty(),
                    "{}: level 0 emitted {} events",
                    case.id,
                    rec.lines.len()
                );
            }
        }
    }
}

#[test]
fn the_fake_and_the_pipeline_emit_every_event_with_its_meaning() {
    let mut kinds = std::collections::BTreeSet::new();
    let mut redrawn = 0;
    for case in cases() {
        for level in [1u8, 2] {
            let lines = case.trace(level);
            let id = format!("{} level {level}", case.id);
            assert_eq!(ev(&lines[0]), "trace", "{id}: the header is first");
            assert_eq!(lines[0]["format"], FORMAT);
            assert_eq!(lines[0]["engine"], "fake");
            assert_eq!(lines[0]["flags"]["p_stop_semantics"], "fake-synthetic/1");
            assert_eq!(lines[0]["protocol"], "story-v1");
            assert_eq!(lines[0]["trace_level"], level);
            assert!(
                !lines[0].contains_key("model_sha256"),
                "{id}: the fake has no model, so no sha"
            );
            for l in &lines {
                assert!(l["t_ms"].is_u64(), "{id}: every line carries t_ms");
                kinds.insert(ev(l).to_string());
            }

            // Walk each chunk draw: begin, its steps, end.
            let mut seed = 0u64;
            let mut steps: Vec<&Map<String, Value>> = Vec::new();
            let mut total_frames = 0u64;
            let mut trims = 0;
            let mut chunk_draws: HashMap<u64, Vec<(u64, Value)>> = HashMap::new();
            for l in &lines[1..] {
                match ev(l) {
                    "chunk_begin" => {
                        seed = l["seed"].as_u64().unwrap();
                        steps.clear();
                        let (i, k) = (l["chunk"].as_u64().unwrap(), l["draw"].as_u64().unwrap());
                        let text = l["text"].as_str().unwrap();
                        assert_eq!(
                            seed,
                            u64::from(seed_of(&case.story_id, i as usize, text, k as u32, &case.seed_salt)),
                            "{id}: chunk_begin's seed is the chunk's seed"
                        );
                        assert!(!l.contains_key("n_text_tokens"), "{id}: absent, never zero");
                    }
                    "step" => {
                        let f = l["frame"].as_u64().unwrap();
                        assert_eq!(f as usize, steps.len(), "{id}: frames count up from 0");
                        assert_eq!(
                            l["token"].as_u64().unwrap(),
                            u64::from(fake_token(seed, f, 0)),
                            "{id}: token is the frame's codebook-0 token"
                        );
                        let top1 = num(l, "top1_p");
                        assert!((0.25..=0.88).contains(&top1), "{id}: top1_p {top1}");
                        assert!(num(l, "entropy") >= -top1.ln() - 1e-12, "{id}: entropy below its bound");
                        assert!(l["rank"].as_u64().unwrap() < 4);
                        if level >= 2 {
                            let cb = l["cb"].as_array().unwrap();
                            let cb_p = l["cb_p"].as_array().unwrap();
                            assert_eq!(cb.len(), usize::from(CODEBOOKS) - 1, "{id}: cb is codebooks 1 onward");
                            assert_eq!(cb_p.len(), cb.len());
                            for (c, t) in cb.iter().enumerate() {
                                assert_eq!(t.as_u64().unwrap(), u64::from(fake_token(seed, f, c as u64 + 1)));
                            }
                        } else {
                            assert!(
                                !l.contains_key("cb") && !l.contains_key("cb_p"),
                                "{id}: level 1 has no cb"
                            );
                        }
                        steps.push(l);
                    }
                    "chunk_end" => {
                        let n = l["frames"].as_u64().unwrap();
                        assert_eq!(n as usize, steps.len(), "{id}: one step per frame");
                        let p: Vec<f64> = steps.iter().map(|s| num(s, "p_stop")).collect();
                        assert!(p.windows(2).all(|w| w[0] < w[1]), "{id}: p_stop rises");
                        if let Some(last) = p.last() {
                            assert_eq!(*last, fake_p_stop(n - 1, n), "{id}: p_stop peaks at the stop");
                        }
                        assert_eq!(l["stop"], "eos");
                        assert!(l["spc"].is_f64() && l.contains_key("verdict") && l["ms"].is_u64());
                        total_frames += n;
                        chunk_draws
                            .entry(l["chunk"].as_u64().unwrap())
                            .or_default()
                            .push((l["draw"].as_u64().unwrap(), l["verdict"].clone()));
                    }
                    "timing" => {
                        assert!(["generate", "decode", "judge"].contains(&l["stage"].as_str().unwrap()));
                        assert!(l["chunk"].is_u64() && l["draw"].is_u64() && l["ms"].is_u64());
                    }
                    "trim" => {
                        trims += 1;
                        assert!(num(l, "lead_s") >= 0.0 && num(l, "tail_s") >= 0.0);
                        assert!(l["start"].as_u64().unwrap() <= l["end"].as_u64().unwrap());
                    }
                    "emit" => assert_eq!(
                        l["chunk"].as_u64().unwrap() as usize,
                        trims - 1,
                        "{id}: a chunk is emitted right after its trim"
                    ),
                    "join" => assert_eq!(
                        l["chunks"].as_u64().unwrap() as usize,
                        trims,
                        "{id}: join counts chunks"
                    ),
                    "end" => assert_eq!(l["frames"].as_u64().unwrap(), total_frames, "{id}: end sums frames"),
                    other => panic!("{id}: unknown event {other}"),
                }
            }
            assert_eq!(trims, chunk_draws.len(), "{id}: one trim per chunk");
            // A redraw happens only after a rejected draw, and a clean draw ends the chunk.
            for (chunk, draws) in &chunk_draws {
                for (n, (draw, verdict)) in draws.iter().enumerate() {
                    assert_eq!(*draw as usize, n, "{id}: chunk {chunk} draws in order");
                    if n + 1 < draws.len() {
                        assert!(
                            !verdict.is_null(),
                            "{id}: chunk {chunk} draw {n} was redrawn while clean"
                        );
                        redrawn += 1;
                    }
                }
            }
            assert_eq!(ev(lines.last().unwrap()), "end");
        }
    }
    let want: std::collections::BTreeSet<String> = [
        "trace",
        "chunk_begin",
        "step",
        "chunk_end",
        "trim",
        "emit",
        "join",
        "timing",
        "end",
    ]
    .map(String::from)
    .into();
    assert_eq!(kinds, want, "every event kind is emitted");
    assert!(redrawn > 0, "the cases include a redraw, so draw > 0 is exercised");
}

/// Write the trace through the JSON Lines writer, so replay reads what a file would hold.
fn jsonl(case: &Case, level: u8) -> String {
    let mut engine = case.engine();
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    let mut w = JsonlEvents::new(Vec::new(), level);
    emit_header(&mut w, &Header::for_engine(&engine, level));
    render_story_traced(&mut engine, &case.story_id, &chunks, &opts, &mut w).unwrap();
    String::from_utf8(w.finish().unwrap()).unwrap()
}

#[test]
fn a_recorded_trace_replays_exactly() {
    for case in cases() {
        for level in [1u8, 2] {
            let text = jsonl(&case, level);
            let lines = parse(&text).unwrap();
            let report = replay(&mut case.engine(), &lines).unwrap();
            let begins = lines.iter().filter(|l| ev(l) == "chunk_begin").count();
            assert_eq!(report.draws, begins);
            assert!(report.events > begins * 2, "{}: replay compared steps", case.id);
            assert_eq!(report.mismatch, None, "{} level {level}", case.id);
        }
    }
}

#[test]
fn a_slow_trace_sink_is_measured_without_changing_audio_or_tokens() {
    struct SlowOnce {
        data: Vec<u8>,
        delayed: bool,
    }
    impl std::io::Write for SlowOnce {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if !self.delayed {
                self.delayed = true;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            self.data.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let case = &cases()[0];
    let mut baseline_engine = case.engine();
    let baseline = render_story_traced(
        &mut baseline_engine,
        &case.story_id,
        &case.chunks(&case.engine()),
        &case.opts(&case.engine()),
        &mut NullEvents,
    )
    .unwrap();
    let mut engine = case.engine();
    let mut sink = JsonlEvents::new(
        SlowOnce {
            data: Vec::new(),
            delayed: false,
        },
        1,
    );
    let elapsed = std::time::Instant::now();
    emit_header(&mut sink, &Header::for_engine(&engine, 1));
    let read = render_story_traced(
        &mut engine,
        &case.story_id,
        &case.chunks(&case.engine()),
        &case.opts(&case.engine()),
        &mut sink,
    )
    .unwrap();
    let (output, stats) = sink.finish_with_stats().unwrap();
    assert!(elapsed.elapsed() >= std::time::Duration::from_millis(20));
    assert!(
        stats.sink_ns >= 20_000_000,
        "the planted sink delay must be measured: {stats:?}"
    );
    assert!(stats.events_written > 0);
    assert_eq!(baseline.samples, read.samples);
    let lines = parse(std::str::from_utf8(&output.data).unwrap()).unwrap();
    assert_eq!(replay(&mut case.engine(), &lines).unwrap().mismatch, None);
}

/// Change the first line matching `pick` with `edit`, and return the edited trace.
fn tamper(
    lines: &[Map<String, Value>],
    pick: impl Fn(&Map<String, Value>) -> bool,
    edit: impl Fn(&mut Map<String, Value>),
) -> Vec<Map<String, Value>> {
    let mut out = lines.to_vec();
    let l = out.iter_mut().find(|l| pick(l)).expect("a line to tamper");
    edit(l);
    out
}

#[test]
fn replay_sees_a_planted_change_and_not_the_clock() {
    let case = &cases()[0];
    let lines = parse(&jsonl(case, 2)).unwrap();
    let at = |c: u64, f: u64| move |l: &Map<String, Value>| ev(l) == "step" && l["chunk"] == c && l["frame"] == f;

    // A no-op: only the clock moves. Replay must read it as no change.
    let clock = tamper(&lines, at(0, 3), |l| {
        l.insert("t_ms".into(), 999_999.into());
    });
    let clock = tamper(
        &clock,
        |l| ev(l) == "chunk_end",
        |l| {
            l.insert("ms".into(), 123_456.into());
        },
    );
    assert_eq!(replay(&mut case.engine(), &clock).unwrap().mismatch, None);

    // One codebook-0 token.
    let tok = tamper(&lines, at(1, 2), |l| {
        let t = l["token"].as_u64().unwrap();
        l.insert("token".into(), ((t + 1) % 2048).into());
    });
    let m = replay(&mut case.engine(), &tok)
        .unwrap()
        .mismatch
        .expect("a token change is red");
    assert_eq!((m.chunk, m.frame, m.field.as_str()), (Some(1), Some(2), "token"), "{m}");

    // One depth-codebook token, named by its codebook.
    let cb = tamper(&lines, at(0, 1), |l| {
        let mut cb = l["cb"].as_array().unwrap().clone();
        cb[4] = ((cb[4].as_u64().unwrap() + 1) % 2048).into();
        l.insert("cb".into(), cb.into());
    });
    let m = replay(&mut case.engine(), &cb)
        .unwrap()
        .mismatch
        .expect("a depth-token change is red");
    assert_eq!((m.chunk, m.frame, m.field.as_str()), (Some(0), Some(1), "cb[5]"), "{m}");

    // An instrument value.
    let ps = tamper(&lines, at(0, 0), |l| {
        l.insert("p_stop".into(), 0.5.into());
    });
    assert_eq!(
        replay(&mut case.engine(), &ps).unwrap().mismatch.unwrap().field,
        "p_stop"
    );

    // The engine itself changed: a different script lengthens the first chunk.
    let mut changed = case.engine();
    let first = case.chunks(&changed)[0].text.clone();
    changed = FakeEngine::new(HashMap::from([(first, vec![9.0])]), case.shape.clone());
    let m = replay(&mut changed, &lines)
        .unwrap()
        .mismatch
        .expect("a changed engine is red");
    assert_eq!(m.chunk, Some(0), "{m}");

    // The decoder changed and the generator did not: the first chunk now decodes at 16 kHz, with the same tokens.
    // Every step still matches, so this is red only because replay decodes, and it names the trim.
    let first = case.chunks(&case.engine())[0].text.clone();
    let mut shape = case.shape.clone();
    shape.entry(first).or_default().sample_rate = Some(16_000);
    let mut decoder = FakeEngine::new(case.script.clone(), shape);
    let m = replay(&mut decoder, &lines)
        .unwrap()
        .mismatch
        .expect("a changed decoder is red");
    assert_eq!((m.chunk, m.field.as_str()), (Some(0), "end"), "{m}");

    // A recorded judge field or trim bound changed by hand: replay recomputes them, so each is red.
    let spc = tamper(
        &lines,
        |l| ev(l) == "chunk_end",
        |l| {
            let v = l["spc"].as_f64().unwrap();
            l.insert("spc".into(), (v + 0.01).into());
        },
    );
    assert_eq!(replay(&mut case.engine(), &spc).unwrap().mismatch.unwrap().field, "spc");
    let trim = tamper(
        &lines,
        |l| ev(l) == "trim",
        |l| {
            let v = l["start"].as_u64().unwrap();
            l.insert("start".into(), (v + 1).into());
        },
    );
    assert_eq!(
        replay(&mut case.engine(), &trim).unwrap().mismatch.unwrap().field,
        "start"
    );

    // A trace made by another engine is refused, not compared.
    let other = tamper(
        &lines,
        |l| ev(l) == "trace",
        |l| {
            l.insert("engine".into(), "qwen3-tts".into());
        },
    );
    assert!(replay(&mut case.engine(), &other).is_err());
}

fn requests(case: &Case) -> Vec<(String, u64, u32)> {
    let engine = case.engine();
    case.chunks(&engine)
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let seed = u64::from(seed_of(&case.story_id, i, &c.text, 0, &case.seed_salt));
            (c.text.clone(), seed, i as u32)
        })
        .collect()
}

#[test]
fn trace_identity_holds_and_sees_a_planted_draw() {
    for case in cases() {
        let owned = requests(&case);
        let reqs: Vec<ChunkRequest<'_>> = owned
            .iter()
            .map(|(text, seed, chunk)| ChunkRequest {
                text,
                seed: *seed,
                chunk: *chunk,
                draw: 0,
                instruction: None,
            })
            .collect();
        for level in [1u8, 2] {
            let mut make = || Box::new(case.engine()) as Box<dyn Engine>;
            assert_eq!(
                check_trace_identity(&mut make, &reqs, level).unwrap(),
                None,
                "{}",
                case.id
            );
        }
        // The planted instrument draws once from the stream when traced. Pick a chunk with frames past 1.
        let Some(target) = reqs.iter().position(|r| FakeEngine::default_speech_s(r.text) > 0.5) else {
            continue;
        };
        let mut planted = || {
            let mut e = case.engine();
            e.plant_stream_draw = Some((target as u32, 1));
            Box::new(e) as Box<dyn Engine>
        };
        let d = check_trace_identity(&mut planted, &reqs, 1)
            .unwrap()
            .expect("a planted draw from the stream is red");
        assert_eq!(
            (d.chunk, d.frame, d.codebook),
            (target as u32, 1, 0),
            "{}: {d}",
            case.id
        );
        // Untraced, the plant does not fire: the check compares like with like.
        let mut a = planted();
        let mut b = case.engine();
        assert_eq!(
            a.generate_traced(&reqs[target], &mut NullEvents).unwrap(),
            b.generate(&reqs[target]).unwrap()
        );
    }
}

#[test]
fn compare_reports_a_missing_event() {
    let lines = cases()[0].trace(1);
    let short = &lines[..lines.len() - 1];
    let m = compare(&lines, short).expect("a dropped event is red");
    assert_eq!(m.field, "<event>");
}

struct Broken;

impl io::Write for Broken {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("disk full"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_trace_that_could_not_be_written_is_an_error() {
    let case = &cases()[0];
    let mut engine = case.engine();
    let mut w = JsonlEvents::new(Broken, 1);
    emit_header(&mut w, &Header::for_engine(&engine, 1));
    let chunks = case.chunks(&engine);
    let opts = case.opts(&engine);
    render_story_traced(&mut engine, &case.story_id, &chunks, &opts, &mut w).unwrap();
    assert!(w.level() == 1 && w.finish().is_err());
}
