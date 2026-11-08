//! The instrument interface: per-step events, the trace format `ggtts-trace/2`, replay and the trace-identity check.
//!
//! Instruments are opt-in. With tracing off the sink is [`NullEvents`], whose [`level`](Events::level) is 0, and
//! neither the pipeline nor an engine computes an instrument value. With it on the output must not change: an
//! instrument reads values the engine already has and never draws from the sampler's stream.
//! [`check_trace_identity`] tests that rule on any engine, and is itself shown to go red on a planted draw.
//!
//! The format is JSON Lines, one event per line, UTF-8. The first line is the header (`"ev":"trace"`), and every
//! line carries `ev` and `t_ms`. Chunk events carry `chunk`. A field an engine cannot supply is absent, never zero.
//! `t_ms` and `ms` are wall-clock and the only fields that may differ between two runs of the same render.
//!
//! | Field | Meaning |
//! |---|---|
//! | `token` | The sampled codebook-0 token at this frame. |
//! | `p_stop` | Probability of the end-of-speech token at temperature 1, before any filtering. |
//! | `entropy` | Entropy in nats of the main sampler's distribution after temperature, before top-k and top-p. |
//! | `top1_p`, `rank` | The most likely token's probability, and the rank of the token actually drawn. |
//! | `cb`, `cb_p` | Level 2: the depth model's tokens (codebooks 1 onward) and each one's probability. |
//! | `stop` | `eos`, `cap` or `abort`. |
//! | `spc`, `verdict` | The judge's seconds per character and its verdict (`null` when clean). |
//! | `emit` | A chunk's piece of the read (the chunk and its gap, before the level) was handed to the stream: its first sample `start` and its length `samples`. `t_ms` is when. |
//! | `ms` (on `chunk_end`) | Wall-clock for the whole draw: `generate`, `decode` and the judge together, not generation alone. |
//!
//! The schema is what an engine must emit. It is not a claim that any field measures what its name says: an
//! instrument is trusted only once it has been shown to detect a planted defect and to read a no-op as no change.
//!
//! The hook catalogue, the event order and the two instrument tiers are in `docs/instruments/hooks.md`; the JSON
//! schema every line validates against is `schemas/ggtts-trace-2.schema.json`.

use std::io::{self, Write};
use std::time::Instant;

use serde_json::{Map, Value};

use crate::audio::{judge_f32, trim_edges_f32};
use crate::engine::{ChunkRequest, Codes, Engine, EngineError, Pcm, Stop};

/// The trace format this module writes and reads.
pub const FORMAT: &str = "ggtts-trace/2";

/// The formats [`parse`] accepts: the current one first, then at most the one before it. Readers accept the
/// previous version for one version, so a trace written by the last release still reads. Writers write only
/// [`FORMAT`].
pub const READS: &[&str] = &[FORMAT, "ggtts-trace/1"];

/// The fields replay ignores: wall-clock only.
pub const CLOCK_FIELDS: [&str; 2] = ["t_ms", "ms"];

/// Where events go.
pub trait Events {
    /// 0 off, 1 per-frame scalars, 2 adds the per-codebook vectors.
    fn level(&self) -> u8;
    fn emit(&mut self, e: Event<'_>);
}

/// The sink when tracing is off: level 0, and every event is dropped.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullEvents;

impl Events for NullEvents {
    fn level(&self) -> u8 {
        0
    }
    fn emit(&mut self, _e: Event<'_>) {}
}

/// The trace header: what made the trace, so it can be replayed.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    /// `ggtts` version, with the commit when known.
    pub ggtts: String,
    /// The ggml pin, short form.
    pub ggml: String,
    pub engine: String,
    pub backend: String,
    pub parity: String,
    /// The engine's profile protocol, for example `story-v1`.
    pub protocol: String,
    /// The model's sha256, when the engine has a model.
    pub model_sha256: Option<String>,
    pub codec_sha256: Option<String>,
    /// The non-default flags. A secret flag is written as `{"set": true, "source": ...}` by its caller, never as
    /// its value.
    pub flags: Map<String, Value>,
    pub trace_level: u8,
}

impl Header {
    /// A header for `engine` at `trace_level`, with no flags.
    pub fn for_engine(engine: &dyn Engine, trace_level: u8) -> Self {
        let info = engine.info();
        Self {
            ggtts: crate::VERSION.to_string(),
            ggml: crate::ggml::short().to_string(),
            engine: info.family.clone(),
            backend: info.backend.clone(),
            parity: info.parity.clone(),
            protocol: engine.profile().protocol.to_string(),
            model_sha256: None,
            codec_sha256: None,
            flags: Map::new(),
            trace_level,
        }
    }
}

/// One event. Optional fields are absent from the line when `None`.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<'a> {
    Trace(&'a Header),
    ChunkBegin {
        chunk: u32,
        draw: u32,
        text: &'a str,
        seed: u64,
        n_text_tokens: Option<u32>,
    },
    Step {
        chunk: u32,
        frame: u32,
        token: u16,
        p_stop: Option<f64>,
        entropy: Option<f64>,
        top1_p: Option<f64>,
        rank: Option<u32>,
        /// Level 2 only.
        cb: Option<&'a [u16]>,
        /// Level 2 only.
        cb_p: Option<&'a [f64]>,
    },
    ChunkEnd {
        chunk: u32,
        draw: u32,
        frames: u32,
        stop: Stop,
        spc: Option<f64>,
        verdict: Option<&'a str>,
        ms: u64,
    },
    Trim {
        chunk: u32,
        lead_s: f64,
        tail_s: f64,
        start: usize,
        end: usize,
    },
    /// A chunk's piece of the read was handed to the stream.
    Emit {
        chunk: u32,
        start: usize,
        samples: usize,
    },
    /// A measured duration. Values are wall-clock and excluded from trace identity comparisons.
    Timing {
        stage: &'static str,
        chunk: Option<u32>,
        draw: Option<u32>,
        ms: u64,
    },
    Join {
        chunks: u32,
        samples: usize,
        gain_db: Option<f64>,
    },
    End {
        frames: u64,
        seconds: f64,
    },
}

fn put<T: Into<Value>>(m: &mut Map<String, Value>, k: &str, v: Option<T>) {
    if let Some(v) = v {
        m.insert(k.into(), v.into());
    }
}

impl Event<'_> {
    /// The event's line, without `t_ms`.
    pub fn to_json(&self) -> Map<String, Value> {
        let mut m = Map::new();
        match self {
            Event::Trace(h) => {
                m.insert("ev".into(), "trace".into());
                m.insert("format".into(), FORMAT.into());
                m.insert("ggtts".into(), h.ggtts.clone().into());
                m.insert("ggml".into(), h.ggml.clone().into());
                m.insert("backend".into(), h.backend.clone().into());
                m.insert("parity".into(), h.parity.clone().into());
                m.insert("engine".into(), h.engine.clone().into());
                put(&mut m, "model_sha256", h.model_sha256.clone());
                put(&mut m, "codec_sha256", h.codec_sha256.clone());
                m.insert("protocol".into(), h.protocol.clone().into());
                m.insert("flags".into(), Value::Object(h.flags.clone()));
                m.insert("trace_level".into(), h.trace_level.into());
            }
            Event::ChunkBegin {
                chunk,
                draw,
                text,
                seed,
                n_text_tokens,
            } => {
                m.insert("ev".into(), "chunk_begin".into());
                m.insert("chunk".into(), (*chunk).into());
                m.insert("draw".into(), (*draw).into());
                m.insert("text".into(), (*text).into());
                m.insert("seed".into(), (*seed).into());
                put(&mut m, "n_text_tokens", *n_text_tokens);
            }
            Event::Step {
                chunk,
                frame,
                token,
                p_stop,
                entropy,
                top1_p,
                rank,
                cb,
                cb_p,
            } => {
                m.insert("ev".into(), "step".into());
                m.insert("chunk".into(), (*chunk).into());
                m.insert("frame".into(), (*frame).into());
                m.insert("token".into(), (*token).into());
                put(&mut m, "p_stop", *p_stop);
                put(&mut m, "entropy", *entropy);
                put(&mut m, "top1_p", *top1_p);
                put(&mut m, "rank", *rank);
                put(&mut m, "cb", cb.map(|v| v.to_vec()));
                put(&mut m, "cb_p", cb_p.map(|v| v.to_vec()));
            }
            Event::ChunkEnd {
                chunk,
                draw,
                frames,
                stop,
                spc,
                verdict,
                ms,
            } => {
                m.insert("ev".into(), "chunk_end".into());
                m.insert("chunk".into(), (*chunk).into());
                m.insert("draw".into(), (*draw).into());
                m.insert("frames".into(), (*frames).into());
                m.insert("stop".into(), stop.as_str().into());
                put(&mut m, "spc", *spc);
                // The judge's verdict is always known once judged, so a clean draw is an explicit null.
                m.insert("verdict".into(), verdict.map_or(Value::Null, Value::from));
                m.insert("ms".into(), (*ms).into());
            }
            Event::Trim {
                chunk,
                lead_s,
                tail_s,
                start,
                end,
            } => {
                m.insert("ev".into(), "trim".into());
                m.insert("chunk".into(), (*chunk).into());
                m.insert("lead_s".into(), (*lead_s).into());
                m.insert("tail_s".into(), (*tail_s).into());
                m.insert("start".into(), (*start).into());
                m.insert("end".into(), (*end).into());
            }
            Event::Emit { chunk, start, samples } => {
                m.insert("ev".into(), "emit".into());
                m.insert("chunk".into(), (*chunk).into());
                m.insert("start".into(), (*start).into());
                m.insert("samples".into(), (*samples).into());
            }
            Event::Timing { stage, chunk, draw, ms } => {
                m.insert("ev".into(), "timing".into());
                m.insert("stage".into(), (*stage).into());
                put(&mut m, "chunk", *chunk);
                put(&mut m, "draw", *draw);
                m.insert("ms".into(), (*ms).into());
            }
            Event::Join {
                chunks,
                samples,
                gain_db,
            } => {
                m.insert("ev".into(), "join".into());
                m.insert("chunks".into(), (*chunks).into());
                m.insert("samples".into(), (*samples).into());
                put(&mut m, "gain_db", *gain_db);
            }
            Event::End { frames, seconds } => {
                m.insert("ev".into(), "end".into());
                m.insert("frames".into(), (*frames).into());
                m.insert("seconds".into(), (*seconds).into());
            }
        }
        m
    }
}

/// Collects events as JSON lines in memory, each stamped with `t_ms` from its creation.
#[derive(Debug)]
pub struct Recorder {
    level: u8,
    start: Instant,
    pub lines: Vec<Map<String, Value>>,
}

impl Recorder {
    pub fn new(level: u8) -> Self {
        Self {
            level,
            start: Instant::now(),
            lines: Vec::new(),
        }
    }

    /// The trace as JSON Lines text.
    pub fn to_jsonl(&self) -> String {
        let mut s = String::new();
        for l in &self.lines {
            s.push_str(&Value::Object(l.clone()).to_string());
            s.push('\n');
        }
        s
    }
}

impl Events for Recorder {
    fn level(&self) -> u8 {
        self.level
    }
    fn emit(&mut self, e: Event<'_>) {
        let mut m = e.to_json();
        m.insert("t_ms".into(), (self.start.elapsed().as_millis() as u64).into());
        self.lines.push(m);
    }
}

/// Writes events to `w` as JSON Lines as they arrive. Write errors are kept and returned by
/// [`finish`](JsonlEvents::finish), so a trace that could not be written is never mistaken for a complete one.
#[derive(Debug)]
pub struct JsonlEvents<W: Write> {
    level: u8,
    start: Instant,
    w: W,
    error: Option<io::Error>,
    sink_ns: u128,
    events_written: u64,
}

/// Time spent serializing and writing events inside the sink. It excludes event preparation by callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceSinkStats {
    pub events_written: u64,
    pub sink_ns: u128,
    pub flush_ns: u128,
}

impl<W: Write> JsonlEvents<W> {
    pub fn new(w: W, level: u8) -> Self {
        Self {
            level,
            start: Instant::now(),
            w,
            error: None,
            sink_ns: 0,
            events_written: 0,
        }
    }

    /// Flush, and return the first write error if there was one.
    pub fn finish(self) -> io::Result<W> {
        self.finish_with_stats().map(|(writer, _)| writer)
    }

    /// Flush and return sink timings separately from inference and stream handoffs.
    pub fn finish_with_stats(mut self) -> io::Result<(W, TraceSinkStats)> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        let start = Instant::now();
        self.w.flush()?;
        let stats = TraceSinkStats {
            events_written: self.events_written,
            sink_ns: self.sink_ns,
            flush_ns: start.elapsed().as_nanos(),
        };
        Ok((self.w, stats))
    }
}

impl<W: Write> Events for JsonlEvents<W> {
    fn level(&self) -> u8 {
        self.level
    }
    fn emit(&mut self, e: Event<'_>) {
        if self.error.is_some() {
            return;
        }
        let start = Instant::now();
        let mut m = e.to_json();
        m.insert("t_ms".into(), (self.start.elapsed().as_millis() as u64).into());
        if let Err(err) = writeln!(self.w, "{}", Value::Object(m)) {
            self.error = Some(err);
        } else {
            self.events_written += 1;
        }
        self.sink_ns += start.elapsed().as_nanos();
    }
}

/// A trace that could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceError(pub String);

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TraceError {}

/// Parse a trace: the header first, every line an object with `ev` and `t_ms`, and the format this module reads.
pub fn parse(text: &str) -> Result<Vec<Map<String, Value>>, TraceError> {
    let mut lines = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        if raw.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(raw).map_err(|e| TraceError(format!("line {}: {e}", n + 1)))?;
        let Value::Object(m) = v else {
            return Err(TraceError(format!("line {}: not an object", n + 1)));
        };
        if !m.get("ev").is_some_and(Value::is_string) || !m.get("t_ms").is_some_and(Value::is_u64) {
            return Err(TraceError(format!("line {}: an event needs `ev` and `t_ms`", n + 1)));
        }
        lines.push(m);
    }
    let Some(head) = lines.first() else {
        return Err(TraceError("empty trace".into()));
    };
    if head["ev"] != "trace" {
        return Err(TraceError("the first line is not the header".into()));
    }
    if !head
        .get("format")
        .and_then(Value::as_str)
        .is_some_and(|f| READS.contains(&f))
    {
        return Err(TraceError(format!(
            "not a trace this reads ({}): {:?}",
            READS.join(", "),
            head.get("format")
        )));
    }
    Ok(lines)
}

fn strip_clock(m: &Map<String, Value>) -> Map<String, Value> {
    let mut m = m.clone();
    for k in CLOCK_FIELDS {
        m.remove(k);
    }
    m
}

/// Where two event streams first differ.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    /// The index of the event in the recorded stream's engine events.
    pub index: usize,
    pub chunk: Option<u64>,
    pub frame: Option<u64>,
    /// The field that differs, `cb[i]` for a depth codebook, or `<event>` when the events are of different kinds
    /// or one stream ends early.
    pub field: String,
    pub recorded: Value,
    pub replayed: Value,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "event {}", self.index)?;
        if let Some(c) = self.chunk {
            write!(f, ", chunk {c}")?;
        }
        if let Some(fr) = self.frame {
            write!(f, ", frame {fr}")?;
        }
        write!(
            f,
            ", {}: recorded {}, replayed {}",
            self.field, self.recorded, self.replayed
        )
    }
}

/// Compare two event lines field for field, ignoring the clock. `None` when they match.
fn diff_event(index: usize, rec: &Map<String, Value>, rep: &Map<String, Value>) -> Option<Mismatch> {
    let (a, b) = (strip_clock(rec), strip_clock(rep));
    if a == b {
        return None;
    }
    let at = |field: String, r: Value, p: Value| Mismatch {
        index,
        chunk: a.get("chunk").and_then(Value::as_u64),
        frame: a.get("frame").and_then(Value::as_u64),
        field,
        recorded: r,
        replayed: p,
    };
    if a.get("ev") != b.get("ev") {
        return Some(at("<event>".into(), Value::Object(a.clone()), Value::Object(b.clone())));
    }
    let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    keys.sort();
    keys.dedup();
    for k in keys {
        let (x, y) = (
            a.get(k).cloned().unwrap_or(Value::Null),
            b.get(k).cloned().unwrap_or(Value::Null),
        );
        if a.contains_key(k) != b.contains_key(k) || x != y {
            if let (Value::Array(xs), Value::Array(ys)) = (&x, &y) {
                // Name the first codebook that differs, counting codebook 0 as the step's `token`.
                if let Some(i) = (0..xs.len().max(ys.len())).find(|&i| xs.get(i) != ys.get(i)) {
                    let (r, p) = (
                        xs.get(i).cloned().unwrap_or(Value::Null),
                        ys.get(i).cloned().unwrap_or(Value::Null),
                    );
                    return Some(at(format!("{k}[{}]", i + 1), r, p));
                }
            }
            return Some(at(k.clone(), x, y));
        }
    }
    None
}

/// Compare two event streams, ignoring `t_ms` and `ms`. `None` when they match event for event.
pub fn compare(recorded: &[Map<String, Value>], replayed: &[Map<String, Value>]) -> Option<Mismatch> {
    for i in 0..recorded.len().max(replayed.len()) {
        match (recorded.get(i), replayed.get(i)) {
            (Some(a), Some(b)) => {
                if let Some(m) = diff_event(i, a, b) {
                    return Some(m);
                }
            }
            (a, b) => {
                let missing = |v: Option<&Map<String, Value>>| v.map_or(Value::Null, |m| Value::Object(m.clone()));
                return Some(Mismatch {
                    index: i,
                    chunk: a.or(b).and_then(|m| m.get("chunk")).and_then(Value::as_u64),
                    frame: a.or(b).and_then(|m| m.get("frame")).and_then(Value::as_u64),
                    field: "<event>".into(),
                    recorded: missing(a),
                    replayed: missing(b),
                });
            }
        }
    }
    None
}

/// The events replay regenerates, all of each chunk's own: `chunk_begin`, the `step`s, `chunk_end` with the judge's
/// `spc` and `verdict`, and the kept draw's `trim`. Each depends only on that chunk's tokens, its decoded samples and
/// the profile, so replay can blame the decoder as well as the generator. `join` and `end` depend on the whole read
/// (the gain is the read's) and are checked by re-rendering it, not by replay.
fn engine_events(lines: &[Map<String, Value>]) -> Vec<Map<String, Value>> {
    lines
        .iter()
        .filter(|m| matches!(m["ev"].as_str(), Some("chunk_begin" | "step" | "chunk_end" | "trim")))
        .cloned()
        .collect()
}

/// What a replay found.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// Chunk draws re-generated.
    pub draws: usize,
    /// Engine events compared.
    pub events: usize,
    /// The first difference, or `None` when the replay matched.
    pub mismatch: Option<Mismatch>,
}

fn u32_of(m: &Map<String, Value>, k: &str) -> Result<u32, TraceError> {
    m.get(k)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| TraceError(format!("`{k}` missing or not a u32 in {}", Value::Object(m.clone()))))
}

/// Replay a recorded trace on `engine`: each `chunk_begin`'s request (text, seed, chunk, draw) is generated again,
/// in the recorded order, at the header's trace level, then decoded and judged; where the trace records a chunk's
/// `trim`, the last regenerated draw of that chunk is trimmed again. The regenerated events are compared with the
/// recorded ones field for field, ignoring the clock, so a decoder change is red as well as a generator change.
///
/// The trim keeps `edge_s` seconds: the header's `flags.edge_s` when the render set it, else the profile's.
///
/// Pass a fresh engine built as the header names it: an engine with state (the fake's per-text draw count) only
/// replays from the same starting state.
pub fn replay(engine: &mut dyn Engine, lines: &[Map<String, Value>]) -> Result<ReplayReport, TraceError> {
    let head = lines.first().ok_or_else(|| TraceError("empty trace".into()))?;
    let level = head
        .get("trace_level")
        .and_then(Value::as_u64)
        .and_then(|v| u8::try_from(v).ok())
        .ok_or_else(|| TraceError("the header has no trace_level".into()))?;
    let info = engine.info();
    for (k, want) in [("engine", &info.family), ("backend", &info.backend)] {
        if head.get(k).and_then(Value::as_str) != Some(want.as_str()) {
            return Err(TraceError(format!(
                "the trace was made by {k} {:?}, not {want:?}",
                head.get(k)
            )));
        }
    }
    let profile = *engine.profile();
    let edge_s = head
        .get("flags")
        .and_then(|f| f.get("edge_s"))
        .and_then(Value::as_f64)
        .unwrap_or(profile.edge_s);
    let mut rec = Recorder::new(level);
    let mut draws = 0;
    let mut last: Option<(u32, Pcm)> = None;
    for m in lines {
        if m["ev"] == "trim" {
            let chunk = u32_of(m, "chunk")?;
            let pcm = match &last {
                Some((c, pcm)) if *c == chunk => pcm,
                _ => return Err(TraceError(format!("a trim for chunk {chunk} follows no draw of it"))),
            };
            let t = trim_edges_f32(&profile, &pcm.samples, pcm.sample_rate, edge_s);
            rec.emit(Event::Trim {
                chunk,
                lead_s: t.lead_cut_s,
                tail_s: t.tail_cut_s,
                start: t.start_sample,
                end: t.end_sample,
            });
            continue;
        }
        if m["ev"] != "chunk_begin" {
            continue;
        }
        let text = m
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| TraceError("chunk_begin without text".into()))?;
        let seed = m
            .get("seed")
            .and_then(Value::as_u64)
            .ok_or_else(|| TraceError("chunk_begin without seed".into()))?;
        let req = ChunkRequest {
            text,
            seed,
            chunk: u32_of(m, "chunk")?,
            draw: u32_of(m, "draw")?,
        };
        let pcm = generate_chunk(engine, &req, &mut rec).map_err(|e| TraceError(format!("engine: {e}")))?;
        last = Some((req.chunk, pcm));
        draws += 1;
    }
    let recorded = engine_events(lines);
    let replayed = engine_events(&rec.lines);
    Ok(ReplayReport {
        draws,
        events: recorded.len(),
        mismatch: compare(&recorded, &replayed),
    })
}

/// One chunk draw as the pipeline makes it: `chunk_begin`, the engine's `step`s, then decode and judge, and a
/// `chunk_end` with the frames, stop and the judge's fields. The decoded samples, for the trim.
fn generate_chunk(
    engine: &mut dyn Engine,
    req: &ChunkRequest<'_>,
    events: &mut dyn Events,
) -> Result<Pcm, EngineError> {
    if events.level() > 0 {
        events.emit(Event::ChunkBegin {
            chunk: req.chunk,
            draw: req.draw,
            text: req.text,
            seed: req.seed,
            n_text_tokens: None,
        });
    }
    let t0 = Instant::now();
    let g = engine.generate_traced(req, events)?;
    let pcm = engine.decode(&g.codes)?;
    let j = judge_f32(engine.profile(), &pcm.samples, pcm.sample_rate, req.text);
    if events.level() > 0 {
        events.emit(Event::ChunkEnd {
            chunk: req.chunk,
            draw: req.draw,
            frames: g.codes.n_frames() as u32,
            stop: g.stop,
            spc: Some(j.s_per_char),
            verdict: j.verdict.map(|v| v.as_str()),
            ms: t0.elapsed().as_millis() as u64,
        });
    }
    Ok(pcm)
}

/// The first codec token that differs between an untraced and a traced generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDiff {
    pub chunk: u32,
    pub draw: u32,
    pub frame: usize,
    pub codebook: usize,
    pub untraced: Option<u16>,
    pub traced: Option<u16>,
}

impl std::fmt::Display for TokenDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "chunk {} draw {} frame {} codebook {}: untraced {:?}, traced {:?}",
            self.chunk, self.draw, self.frame, self.codebook, self.untraced, self.traced
        )
    }
}

fn first_token_diff(req: &ChunkRequest<'_>, a: &Codes, b: &Codes) -> Option<TokenDiff> {
    let cbs = usize::from(a.codebooks.max(b.codebooks).max(1));
    (0..a.frames.len().max(b.frames.len()))
        .find(|&i| a.codebooks != b.codebooks || a.frames.get(i) != b.frames.get(i))
        .map(|i| TokenDiff {
            chunk: req.chunk,
            draw: req.draw,
            frame: i / cbs,
            codebook: i % cbs,
            untraced: a.frames.get(i).copied(),
            traced: b.frames.get(i).copied(),
        })
}

/// The trace-identity check: the same requests on two fresh engines from `make`, one untraced and one traced at
/// `level`, must give identical tokens. Returns the first token that differs, or `None`.
///
/// An instrument that draws from the sampler's stream shifts every later token, so this names the first chunk,
/// frame and codebook it touched.
pub fn check_trace_identity(
    make: &mut dyn FnMut() -> Box<dyn Engine>,
    requests: &[ChunkRequest<'_>],
    level: u8,
) -> Result<Option<TokenDiff>, EngineError> {
    let (mut plain, mut traced) = (make(), make());
    let mut rec = Recorder::new(level);
    for req in requests {
        let a = plain.generate_traced(req, &mut NullEvents)?;
        let b = traced.generate_traced(req, &mut rec)?;
        if let Some(d) = first_token_diff(req, &a.codes, &b.codes) {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

/// Emit the trace header if tracing is on.
pub fn emit_header(events: &mut dyn Events, header: &Header) {
    if events.level() > 0 {
        events.emit(Event::Trace(header));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn absent_fields_are_absent_not_zero() {
        let m = Event::Step {
            chunk: 0,
            frame: 1,
            token: 7,
            p_stop: None,
            entropy: Some(1.5),
            top1_p: None,
            rank: None,
            cb: None,
            cb_p: None,
        }
        .to_json();
        assert!(!m.contains_key("p_stop") && !m.contains_key("cb") && !m.contains_key("rank"));
        assert_eq!(m["entropy"], json!(1.5));
    }

    #[test]
    fn a_clean_draw_has_an_explicit_null_verdict() {
        let m = Event::ChunkEnd {
            chunk: 0,
            draw: 0,
            frames: 3,
            stop: Stop::Eos,
            spc: Some(0.3),
            verdict: None,
            ms: 1,
        }
        .to_json();
        assert_eq!(m["verdict"], Value::Null);
        assert_eq!(m["stop"], "eos");
    }

    #[test]
    fn compare_ignores_only_the_clock() {
        let a = vec![json!({"ev":"step","chunk":0,"frame":2,"token":5,"t_ms":3})
            .as_object()
            .unwrap()
            .clone()];
        let b = vec![json!({"ev":"step","chunk":0,"frame":2,"token":5,"t_ms":99})
            .as_object()
            .unwrap()
            .clone()];
        assert_eq!(compare(&a, &b), None);
        let c = vec![json!({"ev":"step","chunk":0,"frame":2,"token":6,"t_ms":3})
            .as_object()
            .unwrap()
            .clone()];
        let m = compare(&a, &c).expect("a token change is seen");
        assert_eq!((m.chunk, m.frame, m.field.as_str()), (Some(0), Some(2), "token"));
    }

    #[test]
    fn compare_names_the_depth_codebook() {
        let a = vec![json!({"ev":"step","chunk":1,"frame":0,"token":5,"cb":[1,2,3],"t_ms":0})
            .as_object()
            .unwrap()
            .clone()];
        let b = vec![json!({"ev":"step","chunk":1,"frame":0,"token":5,"cb":[1,9,3],"t_ms":0})
            .as_object()
            .unwrap()
            .clone()];
        assert_eq!(compare(&a, &b).unwrap().field, "cb[2]");
    }

    /// `name/N` for a format string, so the rule can be checked without knowing the version.
    fn version(f: &str) -> (&str, u32) {
        let (name, n) = f.rsplit_once('/').expect("name/N");
        (name, n.parse().expect("a number"))
    }

    #[test]
    fn reads_the_current_format_and_at_most_the_previous_one() {
        assert_eq!(READS[0], FORMAT, "writers write the first format readers accept");
        assert!(
            READS.len() <= 2,
            "a reader keeps one previous version, no more: {READS:?}"
        );
        if let [cur, prev] = READS {
            let ((a, n), (b, m)) = (version(cur), version(prev));
            assert!(
                a == b && m + 1 == n,
                "the second format must be the one before the first: {READS:?}"
            );
        }
        for f in READS {
            let head = format!("{{\"ev\":\"trace\",\"format\":\"{f}\",\"t_ms\":0}}\n");
            assert!(parse(&head).is_ok(), "{f} is listed but not read");
        }
        let (name, n) = version(FORMAT);
        let next = format!("{{\"ev\":\"trace\",\"format\":\"{name}/{}\",\"t_ms\":0}}\n", n + 1);
        assert!(
            parse(&next).is_err(),
            "a reader never accepts a version it does not know"
        );
    }

    #[test]
    fn parse_refuses_a_trace_without_its_header() {
        assert!(parse("{\"ev\":\"step\",\"t_ms\":0}\n").is_err());
        assert!(parse("{\"ev\":\"trace\",\"format\":\"ggtts-trace/0\",\"t_ms\":0}\n").is_err());
        assert!(parse("{\"ev\":\"trace\",\"format\":\"ggtts-trace/1\"}\n").is_err());
    }
}
