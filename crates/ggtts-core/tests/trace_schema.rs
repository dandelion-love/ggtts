//! The event stream's contract: every line `ggtts-trace/2` can carry validates against the published schema
//! (`schemas/ggtts-trace-2.schema.json`), events arrive in the order `docs/instruments/hooks.md` states, and both
//! checks go red on planted defects.
//!
//! The renders are the recorded pipeline cases (`fixtures/pipeline/render.json`) on the fake engine, written through
//! the real JSON Lines writer and read back with `trace::parse`, at levels 1 and 2.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use ggtts_core::engine::{Engine, Stop};
use ggtts_core::pipeline::{plan_story, render_story_traced, RenderOptions};
use ggtts_core::trace::{emit_header, parse, Event, Header, JsonlEvents, FORMAT};
use ggtts_fake::{FakeEngine, Shape};
use serde_json::{json, Map, Value};

const EVENTS: [&str; 9] = [
    "trace",
    "chunk_begin",
    "step",
    "chunk_end",
    "trim",
    "emit",
    "join",
    "timing",
    "end",
];

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn validator() -> jsonschema::Validator {
    let text = std::fs::read_to_string(root().join("schemas/ggtts-trace-2.schema.json")).expect("schema");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn errors(v: &jsonschema::Validator, line: &Map<String, Value>) -> Vec<String> {
    let inst = Value::Object(line.clone());
    v.iter_errors(&inst).map(|e| e.to_string()).collect()
}

/// Every recorded pipeline case, rendered on a fresh fake engine at `level` through the JSON Lines writer, then
/// parsed back: the stream as an event instrument reading the file would see it.
fn traces(level: u8) -> Vec<(String, Vec<Map<String, Value>>)> {
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(root().join("fixtures/pipeline/render.json")).unwrap()).unwrap();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let script: HashMap<String, Vec<f64>> = c["script"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
                    )
                })
                .collect();
            let shape: HashMap<String, Shape> = c["shape"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        Shape {
                            amplitude_scale: v["amplitude_scale"].as_f64(),
                            spike: v["spike"].as_f64(),
                            sample_rate: v["sample_rate"].as_u64().map(|n| n as u32),
                        },
                    )
                })
                .collect();
            let paragraphs: Vec<String> = c["paragraphs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap().to_string())
                .collect();
            let mut engine = FakeEngine::new(script, shape);
            let chunks = plan_story(engine.profile(), &paragraphs);
            let opts = RenderOptions {
                redraw: c["redraw"].as_u64().unwrap() as u32,
                seed_salt: c["seed_salt"].as_str().unwrap().to_string(),
                ..RenderOptions::for_profile(engine.profile())
            };
            let mut w = JsonlEvents::new(Vec::new(), level);
            emit_header(&mut w, &Header::for_engine(&engine, level));
            render_story_traced(&mut engine, c["story_id"].as_str().unwrap(), &chunks, &opts, &mut w).expect("renders");
            let bytes = w.finish().expect("written");
            let lines = parse(std::str::from_utf8(&bytes).unwrap()).expect("parses");
            (c["id"].as_str().unwrap().to_string(), lines)
        })
        .collect()
}

#[test]
fn every_line_of_every_fake_render_validates() {
    let v = validator();
    let mut seen = BTreeSet::new();
    let mut depth = false;
    for level in [1u8, 2] {
        for (id, lines) in traces(level) {
            for (n, line) in lines.iter().enumerate() {
                let errs = errors(&v, line);
                assert!(errs.is_empty(), "{id} level {level} line {n}: {errs:?}\n{line:?}");
                seen.insert(line["ev"].as_str().unwrap().to_string());
                depth |= line.contains_key("cb");
            }
        }
    }
    // Every branch of the schema met at least one real line, so none of it is untested.
    let want: BTreeSet<String> = EVENTS.iter().map(|s| s.to_string()).collect();
    assert_eq!(seen, want, "event kinds the fake renders produced");
    assert!(
        depth,
        "level 2 produced no depth codebooks, so cb and cb_p were never validated"
    );
}

/// Each event at its fullest (every optional field set) and at its emptiest (none set), as `Event::to_json` writes it.
/// A field added to the writer without a schema change fails here, because the schema names every field it allows.
fn every_variant() -> Vec<Map<String, Value>> {
    let sha = "0".repeat(64);
    let full = Header {
        ggtts: "0.0.1".into(),
        ggml: "af97976".into(),
        engine: "fake".into(),
        backend: "cpu".into(),
        parity: "exact".into(),
        protocol: "story-v1".into(),
        model_sha256: Some(sha.clone()),
        codec_sha256: Some(sha),
        flags: json!({"edge_s": 0.4, "token": {"set": true, "source": "env"}})
            .as_object()
            .unwrap()
            .clone(),
        trace_level: 2,
    };
    let bare = Header {
        model_sha256: None,
        codec_sha256: None,
        flags: Map::new(),
        trace_level: 1,
        ..full.clone()
    };
    let (cb, cb_p) = ([3u16, 4, 5], [0.5, 0.25, 1.0]);
    let evs = [
        Event::Trace(&full),
        Event::Trace(&bare),
        Event::ChunkBegin {
            chunk: 0,
            draw: 1,
            text: "你好。",
            seed: u64::MAX,
            n_text_tokens: Some(4),
        },
        Event::ChunkBegin {
            chunk: 0,
            draw: 0,
            text: "",
            seed: 0,
            n_text_tokens: None,
        },
        Event::Step {
            chunk: 0,
            frame: 7,
            token: 2047,
            p_stop: Some(0.0),
            entropy: Some(3.2),
            top1_p: Some(1.0),
            rank: Some(0),
            cb: Some(&cb),
            cb_p: Some(&cb_p),
        },
        Event::Step {
            chunk: 0,
            frame: 0,
            token: 0,
            p_stop: None,
            entropy: None,
            top1_p: None,
            rank: None,
            cb: None,
            cb_p: None,
        },
        Event::ChunkEnd {
            chunk: 0,
            draw: 0,
            frames: 12,
            stop: Stop::Eos,
            spc: Some(0.31),
            verdict: None,
            ms: 5,
        },
        Event::ChunkEnd {
            chunk: 1,
            draw: 2,
            frames: 400,
            stop: Stop::Cap,
            spc: None,
            verdict: Some("runaway"),
            ms: 0,
        },
        Event::Trim {
            chunk: 0,
            lead_s: 0.0,
            tail_s: 1.25,
            start: 0,
            end: 24000,
        },
        Event::Emit {
            chunk: 0,
            start: 0,
            samples: 24000,
        },
        Event::Timing {
            stage: "generate",
            chunk: Some(0),
            draw: Some(0),
            ms: 5,
        },
        Event::Timing {
            stage: "first_audio",
            chunk: None,
            draw: None,
            ms: 12,
        },
        Event::Timing {
            stage: "render",
            chunk: None,
            draw: None,
            ms: 30,
        },
        Event::Join {
            chunks: 2,
            samples: 48000,
            gain_db: Some(-3.5),
        },
        Event::Join {
            chunks: 1,
            samples: 0,
            gain_db: None,
        },
        Event::End {
            frames: 412,
            seconds: 2.0,
        },
    ];
    evs.iter()
        .map(|e| {
            let mut m = e.to_json();
            m.insert("t_ms".into(), 0.into());
            m
        })
        .collect()
}

#[test]
fn every_event_the_writer_can_produce_validates() {
    let v = validator();
    let all = every_variant();
    for line in &all {
        let errs = errors(&v, line);
        assert!(errs.is_empty(), "{errs:?}\n{line:?}");
    }
    let kinds: BTreeSet<&str> = all.iter().map(|m| m["ev"].as_str().unwrap()).collect();
    assert_eq!(kinds, EVENTS.iter().copied().collect(), "every event kind is exercised");
}

fn line(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn the_schema_refuses_planted_defects_and_ignores_the_clock() {
    let v = validator();
    let all = every_variant();
    let by = |ev: &str| all.iter().find(|m| m["ev"] == ev).unwrap().clone();
    let (head, step, end_ev) = (by("trace"), by("step"), by("chunk_end"));

    // No-op: a different clock is still a valid line.
    let mut later = step.clone();
    later.insert("t_ms".into(), 987_654.into());
    assert!(errors(&v, &later).is_empty(), "the clock is data-free");

    let plants: Vec<(&str, Map<String, Value>)> = vec![
        ("no t_ms", {
            let mut m = step.clone();
            m.remove("t_ms");
            m
        }),
        ("no token", {
            let mut m = step.clone();
            m.remove("token");
            m
        }),
        ("token as a string", {
            let mut m = step.clone();
            m.insert("token".into(), "2047".into());
            m
        }),
        ("token past u16", {
            let mut m = step.clone();
            m.insert("token".into(), 70_000.into());
            m
        }),
        ("p_stop above 1", {
            let mut m = step.clone();
            m.insert("p_stop".into(), 1.5.into());
            m
        }),
        ("negative entropy", {
            let mut m = step.clone();
            m.insert("entropy".into(), (-0.1).into());
            m
        }),
        ("an undeclared field", {
            let mut m = step.clone();
            m.insert("logits".into(), json!([0.1, 0.2]));
            m
        }),
        ("cb without cb_p", {
            let mut m = step.clone();
            m.remove("cb_p");
            m
        }),
        ("an unknown event", {
            let mut m = step.clone();
            m.insert("ev".into(), "stepp".into());
            m
        }),
        ("a stop reason not in the list", {
            let mut m = end_ev.clone();
            m.insert("stop".into(), "done".into());
            m
        }),
        ("a verdict not in the list", {
            let mut m = end_ev.clone();
            m.insert("verdict".into(), "loud".into());
            m
        }),
        ("chunk_end without its verdict", {
            let mut m = end_ev.clone();
            m.remove("verdict");
            m
        }),
        ("the next format", {
            let mut m = head.clone();
            m.insert("format".into(), "ggtts-trace/3".into());
            m
        }),
        ("trace level 0 in a header", {
            let mut m = head.clone();
            m.insert("trace_level".into(), 0.into());
            m
        }),
        ("a short model hash", {
            let mut m = head.clone();
            m.insert("model_sha256".into(), "abc".into());
            m
        }),
        ("not an event at all", line(json!({"t_ms": 0}))),
    ];
    for (name, m) in plants {
        assert!(!errors(&v, &m).is_empty(), "planted defect accepted: {name}\n{m:?}");
    }
}

/// The order `docs/instruments/hooks.md` states. `None` when the stream follows it, else what broke.
fn order_violation(lines: &[Map<String, Value>]) -> Option<String> {
    let ev = |m: &Map<String, Value>| m["ev"].as_str().unwrap_or("").to_string();
    let u = |m: &Map<String, Value>, k: &str| m.get(k).and_then(Value::as_u64);
    if lines.first().map(ev).as_deref() != Some("trace") {
        return Some("the header is not first".into());
    }
    if lines.iter().filter(|m| ev(m) == "trace").count() != 1 {
        return Some("not exactly one header".into());
    }
    let tail: Vec<String> = lines.iter().rev().take(2).map(ev).collect();
    if tail != ["end", "join"] {
        return Some(format!("the stream does not end join, end: {tail:?}"));
    }
    let mut t = 0;
    for (i, m) in lines.iter().enumerate() {
        let now = u(m, "t_ms")?;
        if now < t {
            return Some(format!("t_ms falls at line {i}"));
        }
        t = now;
    }
    // Between the header and join: per chunk, its draws (chunk_begin, steps with frame 0.., chunk_end), then trim,
    // then emit.
    let body = &lines[1..lines.len() - 2];
    let (mut i, mut chunk) = (0, 0u64);
    while i < body.len() {
        let mut draws = 0u64;
        while i < body.len() && ev(&body[i]) == "chunk_begin" {
            if u(&body[i], "chunk") != Some(chunk) || u(&body[i], "draw") != Some(draws) {
                return Some(format!("chunk_begin out of order at body line {i}"));
            }
            i += 1;
            let mut frame = 0;
            while i < body.len() && ev(&body[i]) == "step" {
                if u(&body[i], "chunk") != Some(chunk) || u(&body[i], "frame") != Some(frame) {
                    return Some(format!("step out of order at body line {i}"));
                }
                frame += 1;
                i += 1;
            }
            for stage in ["generate", "decode", "judge"] {
                if i >= body.len()
                    || ev(&body[i]) != "timing"
                    || body[i].get("stage").and_then(Value::as_str) != Some(stage)
                    || u(&body[i], "chunk") != Some(chunk)
                    || u(&body[i], "draw") != Some(draws)
                {
                    return Some(format!("missing {stage} timing for chunk {chunk} draw {draws}"));
                }
                i += 1;
            }
            if i >= body.len() || ev(&body[i]) != "chunk_end" || u(&body[i], "draw") != Some(draws) {
                return Some(format!("draw {draws} of chunk {chunk} does not end with its chunk_end"));
            }
            draws += 1;
            i += 1;
        }
        if draws == 0 {
            return Some(format!("chunk {chunk} has no draw before body line {i}"));
        }
        if i >= body.len() || ev(&body[i]) != "trim" || u(&body[i], "chunk") != Some(chunk) {
            return Some(format!("chunk {chunk}'s draws are not followed by its trim"));
        }
        i += 1;
        if i >= body.len() || ev(&body[i]) != "emit" || u(&body[i], "chunk") != Some(chunk) {
            return Some(format!("chunk {chunk}'s trim is not followed by its emit"));
        }
        i += 1;
        chunk += 1;
    }
    let joined = u(&lines[lines.len() - 2], "chunks");
    (joined != Some(chunk)).then(|| format!("join says {joined:?} chunks, the stream has {chunk}"))
}

#[test]
fn events_arrive_in_the_documented_order_and_a_reorder_is_seen() {
    let mut redrawn = false;
    for level in [1u8, 2] {
        for (id, lines) in traces(level) {
            assert_eq!(order_violation(&lines), None, "{id} at level {level}");
            redrawn |= lines.iter().any(|m| m["ev"] == "chunk_begin" && m["draw"] != 0);
        }
    }
    assert!(
        redrawn,
        "no case redraws a chunk, so the multi-draw order was never checked"
    );

    let (_, lines) = traces(1).into_iter().next().unwrap();
    let step = lines.iter().position(|m| m["ev"] == "step").unwrap();
    let trim = lines.iter().position(|m| m["ev"] == "trim").unwrap();
    let plants: Vec<(&str, Vec<Map<String, Value>>)> = vec![
        ("two steps swapped", {
            let mut l = lines.clone();
            l.swap(step, step + 1);
            l
        }),
        ("a trim before its chunk_end", {
            let mut l = lines.clone();
            l.swap(trim, trim - 1);
            l
        }),
        ("the header second", {
            let mut l = lines.clone();
            l.swap(0, 1);
            l
        }),
        ("end before join", {
            let mut l = lines.clone();
            let n = l.len();
            l.swap(n - 1, n - 2);
            l
        }),
        ("a chunk_end dropped", {
            let mut l = lines.clone();
            let ce = l.iter().position(|m| m["ev"] == "chunk_end").unwrap();
            l.remove(ce);
            l
        }),
        ("an emit before its trim", {
            let mut l = lines.clone();
            l.swap(trim, trim + 1);
            l
        }),
        ("an emit dropped", {
            let mut l = lines.clone();
            let e = l.iter().position(|m| m["ev"] == "emit").unwrap();
            l.remove(e);
            l
        }),
        ("the clock running backwards", {
            let mut l = lines.clone();
            l[step].insert("t_ms".into(), u64::MAX.into());
            l
        }),
    ];
    for (name, l) in plants {
        assert!(order_violation(&l).is_some(), "planted reorder not seen: {name}");
    }
}

#[test]
fn the_stream_is_the_format_its_schema_names() {
    let text = std::fs::read_to_string(root().join("schemas/ggtts-trace-2.schema.json")).unwrap();
    let schema: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(schema["$defs"]["trace"]["properties"]["format"]["const"], FORMAT);
}
