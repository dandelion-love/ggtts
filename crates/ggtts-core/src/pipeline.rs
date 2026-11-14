//! The story path (render spec sections 2 to 7): chunk, seed per chunk, render, judge, redraw, trim, join, level.
//!
//! One entry point, [`render_plan`], renders a plan with any observers ([`Observers`]) and any output ([`Output`]),
//! and it is [`render_chunk`] per chunk, then join and level. A caller with its own plan, seeds, store and join (a
//! book) calls [`render_chunk`] alone. The four `render_story*` functions are wrappers over [`render_plan`].
//!
//! Every number here comes from the engine's [`Profile`]. Changing one in a published profile is a new protocol
//! version, with a changed set of pipeline cases.

use crate::audio::{judge_f32, trim_edges_f32, Judgement, Verdict};
use crate::engine::{ChunkRequest, Engine, EngineError, Stop};
use crate::hooks::Hooks;
use crate::numeric::{log10, mean_f32, pow};
use crate::profile::Profile;
use crate::seed::seed_of;
use crate::stream::StreamLevel;
use crate::text::{plan, PlannedChunk};
use crate::trace::{Event, Events, NullEvents};

/// Chunk a story's paragraphs at the profile's [`story_ceiling`](Profile::story_ceiling).
pub fn plan_story(profile: &Profile, paragraphs: &[String]) -> Result<Vec<PlannedChunk>, &'static str> {
    plan(profile, paragraphs, profile.story_ceiling)
}

/// The gap after a chunk, in samples: the profile's pause for its mark, plus
/// [`para_extra_s`](Profile::para_extra_s) at a paragraph end, times the rate, truncated.
pub fn gap_samples(profile: &Profile, punct: &str, last_in_para: bool, sample_rate: u32) -> usize {
    let extra = if last_in_para { profile.para_extra_s } else { 0.0 };
    ((profile.pause_s(punct) + extra) * f64::from(sample_rate)) as usize
}

/// How a story is rendered.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Redraws allowed for a chunk the judge rejects.
    pub redraw: u32,
    /// Appended to every seed key when not empty.
    pub seed_salt: String,
    /// Silence kept at each edge of a chunk.
    pub edge_s: f64,
    /// Collect wall-clock stage durations even when tracing is off. Includes observer work inside generation.
    pub measure_timings: bool,
    /// The style instruction every chunk is rendered with ([`ChunkRequest::instruction`]); `None` for none.
    pub instruction: Option<String>,
    /// The steering every chunk is rendered with ([`ChunkRequest::steer`]); `None` for none.
    pub steer: Option<crate::steer::Steering>,
    /// Trim each kept draw's edge silence to [`edge_s`](Self::edge_s). `false` keeps the decoded samples whole and
    /// emits no `trim` event (`ggtts-render --text`).
    pub trim: bool,
    /// Level the joined read once ([`level`]). `false` returns it unlevelled, with [`Read::gain`] `None`
    /// (`ggtts-render --text`). A levelled stream ([`StreamMode::Levelled`]) is levelled either way.
    pub level: bool,
}

impl RenderOptions {
    /// The profile's defaults: its [`redraw`](Profile::redraw) and [`edge_s`](Profile::edge_s), no salt and no
    /// instruction, trimmed and levelled.
    pub fn for_profile(profile: &Profile) -> Self {
        Self {
            redraw: profile.redraw,
            seed_salt: String::new(),
            edge_s: profile.edge_s,
            measure_timings: false,
            instruction: None,
            steer: None,
            trim: true,
            level: true,
        }
    }
}

/// One chunk of a rendered read.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecord {
    pub i: usize,
    pub text: String,
    /// The seed of the draw that was kept.
    pub seed: u64,
    /// Redraws taken: the kept draw's index.
    pub redraws: u32,
    /// The kept draw only, decoded before edge trimming. Rejected draws are excluded.
    pub decoded_samples: usize,
    /// Decoder-reported rate for this chunk (which may differ from the output rate).
    pub decoded_sample_rate: u32,
    pub first_draw_defect: Option<Verdict>,
    pub final_defect: Option<Verdict>,
    /// The kept draw's seconds of speech per character, unrounded.
    pub s_per_char: f64,
    pub first_s_per_char: f64,
    /// The kept chunk's samples in the read, `[start, end)`.
    pub span: (usize, usize),
    /// Why the kept draw's generation stopped.
    pub stop: Stop,
}

/// A rendered read: the joined, levelled samples and what happened to each chunk.
#[derive(Debug, Clone)]
pub struct Read {
    pub chunks: Vec<ChunkRecord>,
    pub samples: Vec<f32>,
    /// The first chunk's rate. Later chunks are not resampled or checked.
    pub sample_rate: u32,
    /// The linear gain the level step applied, or `None` when no sample was above the floor.
    pub gain: Option<f64>,
    /// Generated codec frames across all draws, including redraws.
    pub frames_generated: u64,
    /// Optional wall-clock stage totals. `None` means timing collection was not requested.
    pub timings: Option<StageTimings>,
    /// What the stream's level step did, when the read was streamed ([`Output::Stream`]).
    pub stream: Option<StreamReport>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTimings {
    pub generate_ns: u128,
    pub decode_ns: u128,
    pub judge_ns: u128,
}

impl Read {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
    pub fn first_draw_defects(&self) -> usize {
        self.chunks.iter().filter(|c| c.first_draw_defect.is_some()).count()
    }
    pub fn final_defects(&self) -> usize {
        self.chunks.iter().filter(|c| c.final_defect.is_some()).count()
    }
}

/// Why a story could not be rendered.
#[derive(Debug)]
pub enum RenderError {
    /// The plan has no chunks.
    Empty,
    Engine(EngineError),
    /// The stream sink refused a piece (for example, its writer failed). The render stops there.
    Sink(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Empty => write!(f, "the story has no chunk to render"),
            RenderError::Engine(e) => write!(f, "engine: {e}"),
            RenderError::Sink(e) => write!(f, "stream: {e}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<EngineError> for RenderError {
    fn from(e: EngineError) -> Self {
        RenderError::Engine(e)
    }
}

/// The seed of a chunk's draw. [`render_chunk`] asks it for each draw it makes.
///
/// [`StorySeeds`] is the story's (render spec section 3), [`Given`] serves a one-chunk plan, and a caller with its
/// own seed key (a book's chapter plan) passes a closure: any `Fn(chunk, text, draw) -> u32` is a seed source. Seeds
/// are `u64`, as [`ChunkRequest::seed`] is; the story's and the book's are `u32` values widened.
pub trait SeedSource {
    /// The seed of chunk `chunk` (its index in the caller's plan), whose text is `text`, at draw `draw`.
    fn seed(&self, chunk: usize, text: &str, draw: u32) -> u64;
}

impl<F: Fn(usize, &str, u32) -> u32> SeedSource for F {
    fn seed(&self, chunk: usize, text: &str, draw: u32) -> u64 {
        u64::from(self(chunk, text, draw))
    }
}

/// A story's seeds: [`seed_of`] with the story's id and the salt ([`RenderOptions::seed_salt`]).
#[derive(Debug, Clone, Copy)]
pub struct StorySeeds<'a> {
    pub story_id: &'a str,
    pub salt: &'a str,
}

impl SeedSource for StorySeeds<'_> {
    fn seed(&self, chunk: usize, text: &str, draw: u32) -> u64 {
        u64::from(seed_of(self.story_id, chunk, text, draw, self.salt))
    }
}

/// One seed for every draw of every chunk: a one-chunk plan whose caller chose the seed. With redraws, every draw
/// gets the same seed.
#[derive(Debug, Clone, Copy)]
pub struct Given(pub u64);

impl SeedSource for Given {
    fn seed(&self, _chunk: usize, _text: &str, _draw: u32) -> u64 {
        self.0
    }
}

/// Decides whether a decoded draw is kept or redrawn. [`DurationJudge`] is the render spec's (section 5): seconds
/// of speech per character against the profile's bars. A judge that also listens (an ASR round trip) implements
/// this and is passed in its place.
pub trait Judge {
    /// Judge one draw of `text`, decoded to `samples` at `sample_rate`. A verdict of `None` keeps the draw.
    fn judge(&mut self, profile: &Profile, samples: &[f32], sample_rate: u32, text: &str) -> Judgement;
}

/// The render spec's judge: [`judge_f32`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DurationJudge;

impl Judge for DurationJudge {
    fn judge(&mut self, profile: &Profile, samples: &[f32], sample_rate: u32, text: &str) -> Judgement {
        judge_f32(profile, samples, sample_rate, text)
    }
}

/// The observers a render feeds: trace events and tensor hooks. Neither changes the audio. They hold references,
/// so they are passed beside the options, never inside them.
pub struct Observers<'a> {
    pub events: &'a mut dyn Events,
    pub hooks: &'a mut Hooks,
}

impl<'a> Observers<'a> {
    pub fn new(events: &'a mut dyn Events, hooks: &'a mut Hooks) -> Self {
        Self { events, hooks }
    }
}

/// How a streamed read is levelled on its way out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamMode {
    /// The pieces before the level step, exactly as the read is joined.
    Raw,
    /// The pieces levelled by [`StreamLevel`] with one gain taken from at least the first `hold_s` seconds.
    Levelled { hold_s: f64 },
}

/// A stream's sink: takes each piece in order, and an error stops the render.
pub type Sink<'a> = &'a mut dyn FnMut(StreamPiece<'_>) -> Result<(), String>;

/// Where a read goes besides the returned [`Read`].
pub enum Output<'a> {
    /// Only the returned read.
    Whole,
    /// Each piece to `sink` as soon as its chunk is kept (and, levelled, as soon as the hold allows).
    Stream { sink: Sink<'a>, level: StreamMode },
}

/// What a streamed read's level step did, in [`Read::stream`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamReport {
    pub mode: StreamMode,
    /// The stream's gain: `None` when raw, or when no held audio was above the floor.
    pub gain: Option<f64>,
    /// Seconds of audio held when the gain was decided (0 when raw).
    pub held_s: f64,
    /// Wall-clock nanoseconds spent preparing and levelling pieces, outside the sink.
    pub level_ns: u128,
}

/// One draw of a chunk, as [`render_chunk`] made it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawRecord {
    pub draw: u32,
    pub seed: u64,
    pub judgement: Judgement,
    /// Codec frames generated.
    pub frames: usize,
    /// Why generation stopped.
    pub stop: Stop,
}

/// A chunk as [`render_chunk`] kept it: the kept draw's samples, trimmed, and every draw it made.
#[derive(Debug, Clone, PartialEq)]
pub struct KeptChunk {
    /// The kept draw, decoded and edge-trimmed to [`RenderOptions::edge_s`].
    pub samples: Vec<f32>,
    /// The decoder's rate for the kept draw.
    pub sample_rate: u32,
    /// The kept draw's samples before trimming.
    pub decoded_samples: usize,
    /// Every draw in order; the last is the kept one.
    pub draws: Vec<DrawRecord>,
    /// Stage totals for this chunk's draws, when [`RenderOptions::measure_timings`] is set.
    pub timings: Option<StageTimings>,
}

impl KeptChunk {
    /// The kept draw.
    pub fn kept(&self) -> &DrawRecord {
        self.draws.last().expect("at least one draw")
    }
    /// The first draw.
    pub fn first(&self) -> &DrawRecord {
        &self.draws[0]
    }
    /// Codec frames across every draw.
    pub fn frames(&self) -> u64 {
        self.draws.iter().map(|d| d.frames as u64).sum()
    }
}

/// Render one story: [`render_plan`] at the story's seeds ([`StorySeeds`]) with the render spec's judge, no observers
/// and the whole read. Each chunk is judged, redrawn up to `opts.redraw` times while the judge rejects it (the last
/// draw is kept either way), trimmed to `opts.edge_s`, joined with the pause table, and levelled once for the whole
/// read. Plan the chunks with the engine's profile ([`plan_story`]).
pub fn render_story(
    engine: &mut dyn Engine,
    story_id: &str,
    chunks: &[PlannedChunk],
    opts: &RenderOptions,
) -> Result<Read, RenderError> {
    render_story_traced(engine, story_id, chunks, opts, &mut NullEvents)
}

/// [`render_story`], emitting the trace's chunk, trim, join and end events into `events`, and the engine's step
/// events through [`Engine::generate_traced`]. Emit the header first ([`crate::trace::emit_header`]). At level 0
/// nothing is emitted or computed for the trace, and at any level the read is the same.
pub fn render_story_traced(
    engine: &mut dyn Engine,
    story_id: &str,
    chunks: &[PlannedChunk],
    opts: &RenderOptions,
    events: &mut dyn Events,
) -> Result<Read, RenderError> {
    render_story_hooked(engine, story_id, chunks, opts, events, &mut Hooks::new())
}

/// [`render_story_traced`], handing every generated frame to `hooks` as well (see [`crate::hooks`]). The read is
/// the same with or without hooks. The caller calls [`Hooks::finish`] and reads [`Hooks::failures`] when the run is
/// done.
pub fn render_story_hooked(
    engine: &mut dyn Engine,
    story_id: &str,
    chunks: &[PlannedChunk],
    opts: &RenderOptions,
    events: &mut dyn Events,
    hooks: &mut Hooks,
) -> Result<Read, RenderError> {
    let seeds = StorySeeds {
        story_id,
        salt: &opts.seed_salt,
    };
    let obs = &mut Observers::new(events, hooks);
    render_plan(engine, chunks, &seeds, &mut DurationJudge, opts, obs, Output::Whole)
}

/// One piece of a streamed read: a kept chunk, trimmed, followed by the gap after it (none after the last).
///
/// The pieces arrive in order, each as soon as its chunk is kept. Under [`StreamMode::Raw`] they are the read
/// **before the level step**: the level's gain is computed over the whole read, so it is known only at the end, as
/// [`Read::gain`]. Concatenated, raw pieces are the batch read's samples before levelling, exactly; [`level`] with
/// the profile over that concatenation gives [`Read::samples`]. Under [`StreamMode::Levelled`] they are the same
/// pieces levelled by [`StreamLevel`], each released once the hold allows.
#[derive(Debug, Clone, Copy)]
pub struct StreamPiece<'a> {
    pub chunk: usize,
    /// The piece's first sample in the read.
    pub start: usize,
    pub samples: &'a [f32],
    /// The read's rate: the first chunk's. Later chunks are not resampled or checked.
    pub sample_rate: u32,
}

/// [`render_story_traced`], handing each raw [`StreamPiece`] to `sink` as soon as its chunk is kept, before the next
/// chunk is generated. The read it returns is the batch read: identical with any sink that accepts every piece.
/// A sink that returns an error stops the render before the next chunk is generated, as
/// [`RenderError::Sink`]: a stream nobody can receive is not worth the rest of the generation.
pub fn render_story_streamed(
    engine: &mut dyn Engine,
    story_id: &str,
    chunks: &[PlannedChunk],
    opts: &RenderOptions,
    events: &mut dyn Events,
    sink: &mut dyn FnMut(StreamPiece<'_>) -> Result<(), String>,
) -> Result<Read, RenderError> {
    let seeds = StorySeeds {
        story_id,
        salt: &opts.seed_salt,
    };
    let mut hooks = Hooks::new();
    let obs = &mut Observers::new(events, &mut hooks);
    let output = Output::Stream {
        sink,
        level: StreamMode::Raw,
    };
    render_plan(engine, chunks, &seeds, &mut DurationJudge, opts, obs, output)
}

/// Render one planned chunk: each draw at `seeds`' seed for (`index`, draw), generated, decoded and judged, redrawn up
/// to `opts.redraw` times while `judge` rejects it (the last draw is kept either way), then trimmed to `opts.edge_s`
/// (unless `opts.trim` is off).
///
/// This is the step every render is made of. [`render_plan`] calls it per chunk and joins; a caller with its own plan,
/// seeds, store and join (a book) calls it alone. `index` is the chunk's number in the caller's plan: the request's
/// [`chunk`](ChunkRequest::chunk) and the trace's. Emits each draw's `chunk_begin`, the engine's steps, the stage
/// timings and `chunk_end`, then the kept draw's `trim`, into `obs.events`, and hands each frame to `obs.hooks`.
pub fn render_chunk(
    engine: &mut dyn Engine,
    text: &str,
    index: usize,
    seeds: &dyn SeedSource,
    judge: &mut dyn Judge,
    opts: &RenderOptions,
    obs: &mut Observers<'_>,
) -> Result<KeptChunk, RenderError> {
    let events = &mut *obs.events;
    let tracing = events.level() > 0;
    let mut timings = opts.measure_timings.then(StageTimings::default);
    let profile = *engine.profile();
    let mut draws = Vec::new();
    let mut kept = None;
    for k in 0..=opts.redraw {
        let seed = seeds.seed(index, text, k);
        let req = ChunkRequest {
            text,
            seed,
            chunk: index as u32,
            draw: k,
            instruction: opts.instruction.as_deref(),
            steer: opts.steer.as_ref(),
        };
        let t0 = tracing.then(std::time::Instant::now);
        if tracing {
            events.emit(Event::ChunkBegin {
                chunk: req.chunk,
                draw: k,
                text,
                seed: req.seed,
                n_text_tokens: None,
                instruction: req.instruction(),
            });
        }
        let stage_start = (tracing || timings.is_some()).then(std::time::Instant::now);
        let generated = engine.generate_hooked(&req, events, obs.hooks)?;
        stage_end(
            stage_start,
            "generate",
            &req,
            events,
            timings.as_mut().map(|t| &mut t.generate_ns),
        );
        let stage_start = (tracing || timings.is_some()).then(std::time::Instant::now);
        let pcm = engine.decode(&generated.codes)?;
        stage_end(
            stage_start,
            "decode",
            &req,
            events,
            timings.as_mut().map(|t| &mut t.decode_ns),
        );
        let stage_start = (tracing || timings.is_some()).then(std::time::Instant::now);
        let j = judge.judge(&profile, &pcm.samples, pcm.sample_rate, text);
        stage_end(
            stage_start,
            "judge",
            &req,
            events,
            timings.as_mut().map(|t| &mut t.judge_ns),
        );
        let frames = generated.codes.n_frames();
        if let Some(t0) = t0 {
            events.emit(Event::ChunkEnd {
                chunk: req.chunk,
                draw: k,
                frames: frames as u32,
                stop: generated.stop,
                spc: Some(j.s_per_char),
                verdict: j.verdict.map(|v| v.as_str()),
                ms: t0.elapsed().as_millis() as u64,
            });
        }
        draws.push(DrawRecord {
            draw: k,
            seed,
            judgement: j,
            frames,
            stop: generated.stop,
        });
        let clean = j.verdict.is_none();
        kept = Some(pcm);
        if clean {
            break;
        }
    }
    let pcm = kept.expect("at least one draw");
    let decoded_samples = pcm.samples.len();
    let mut samples = pcm.samples;
    if opts.trim {
        let t = trim_edges_f32(&profile, &samples, pcm.sample_rate, opts.edge_s);
        if tracing {
            events.emit(Event::Trim {
                chunk: index as u32,
                lead_s: t.lead_cut_s,
                tail_s: t.tail_cut_s,
                start: t.start_sample,
                end: t.end_sample,
            });
        }
        samples.truncate(t.end_sample);
        samples.drain(..t.start_sample);
    }
    Ok(KeptChunk {
        samples,
        sample_rate: pcm.sample_rate,
        decoded_samples,
        draws,
        timings,
    })
}

/// Close one timed stage: add its wall clock to `total` and, when tracing, emit its `timing` event.
fn stage_end(
    start: Option<std::time::Instant>,
    stage: &'static str,
    req: &ChunkRequest<'_>,
    events: &mut dyn Events,
    total: Option<&mut u128>,
) {
    let Some(start) = start else { return };
    let elapsed = start.elapsed();
    if let Some(t) = total {
        *t += elapsed.as_nanos();
    }
    if events.level() > 0 {
        events.emit(Event::Timing {
            stage,
            chunk: Some(req.chunk),
            draw: Some(req.draw),
            ms: elapsed.as_millis() as u64,
        });
    }
}

/// Render a plan: [`render_chunk`] for each chunk in order, joined with the pause table as each is kept, handed to
/// `output`, and levelled once for the whole read. The judge's bars, the pauses and the level read the engine's
/// [`profile`](Engine::profile); plan the chunks with the same profile ([`plan_story`]).
///
/// Every combination of observers and output gives the same read. A streamed output gets each piece before the
/// next chunk is generated ([`StreamMode::Raw`]) or as soon as the hold allows ([`StreamMode::Levelled`]); a sink
/// that returns an error stops the render there, as [`RenderError::Sink`].
pub fn render_plan(
    engine: &mut dyn Engine,
    chunks: &[PlannedChunk],
    seeds: &dyn SeedSource,
    judge: &mut dyn Judge,
    opts: &RenderOptions,
    obs: &mut Observers<'_>,
    output: Output<'_>,
) -> Result<Read, RenderError> {
    let tracing = obs.events.level() > 0;
    let mut total_frames: u64 = 0;
    let mut timings = opts.measure_timings.then(StageTimings::default);
    if chunks.is_empty() {
        return Err(RenderError::Empty);
    }
    let profile = *engine.profile();
    let mut stream = match output {
        Output::Whole => None,
        Output::Stream { sink, level } => Some(Streamer::new(sink, level, &profile)),
    };
    let n = chunks.len();
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate = 0;
    let mut records = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        let kept = render_chunk(engine, &c.text, i, seeds, judge, opts, obs)?;
        total_frames += kept.frames();
        if let (Some(t), Some(k)) = (timings.as_mut(), kept.timings) {
            t.generate_ns += k.generate_ns;
            t.decode_ns += k.decode_ns;
            t.judge_ns += k.judge_ns;
        }
        if i == 0 {
            sample_rate = kept.sample_rate;
        }
        // Join as each chunk is kept: the chunk, then its gap unless it is the last.
        let start = samples.len();
        samples.extend_from_slice(&kept.samples);
        let span = (start, samples.len());
        if i + 1 < n {
            let gap = gap_samples(&profile, &c.punct, c.last_in_para, sample_rate);
            samples.resize(samples.len() + gap, 0.0);
        }
        if let Some(s) = stream.as_mut() {
            s.push(i, start, &samples[start..], sample_rate)?;
        }
        if tracing {
            obs.events.emit(Event::Emit {
                chunk: i as u32,
                start,
                samples: samples.len() - start,
            });
        }
        let (first, last) = (kept.first().judgement, *kept.kept());
        records.push(ChunkRecord {
            i,
            text: c.text.clone(),
            seed: last.seed,
            redraws: last.draw,
            decoded_samples: kept.decoded_samples,
            decoded_sample_rate: kept.sample_rate,
            first_draw_defect: first.verdict,
            final_defect: last.judgement.verdict,
            s_per_char: last.judgement.s_per_char,
            first_s_per_char: first.s_per_char,
            span,
            stop: last.stop,
        });
    }
    // A read shorter than the hold leaves here, levelled over all of it.
    let stream = stream.map(|s| s.finish(sample_rate)).transpose()?;

    let gain = if opts.level {
        level(&profile, &mut samples)
    } else {
        None
    };
    if tracing {
        obs.events.emit(Event::Join {
            chunks: n as u32,
            samples: samples.len(),
            gain_db: gain.map(|g| 20.0 * log10(g)),
        });
        obs.events.emit(Event::End {
            frames: total_frames,
            seconds: samples.len() as f64 / f64::from(sample_rate),
        });
    }
    Ok(Read {
        chunks: records,
        samples,
        sample_rate,
        gain,
        frames_generated: total_frames,
        timings,
        stream,
    })
}

/// The stream side of [`render_plan`]: the sink, and the leveller with the pieces it holds.
struct Streamer<'a> {
    sink: Sink<'a>,
    mode: StreamMode,
    leveller: Option<StreamLevel>,
    /// `(chunk, start)` of each piece the leveller holds, in order: it releases them whole and in the same order.
    held: std::collections::VecDeque<(usize, usize)>,
    level_ns: u128,
}

impl<'a> Streamer<'a> {
    fn new(sink: Sink<'a>, mode: StreamMode, profile: &Profile) -> Self {
        let leveller = match mode {
            StreamMode::Raw => None,
            StreamMode::Levelled { hold_s } => Some(StreamLevel::new(profile, hold_s)),
        };
        Self {
            sink,
            mode,
            leveller,
            held: std::collections::VecDeque::new(),
            level_ns: 0,
        }
    }

    fn push(&mut self, chunk: usize, start: usize, samples: &[f32], sample_rate: u32) -> Result<(), RenderError> {
        let Some(lv) = self.leveller.as_mut() else {
            return (self.sink)(StreamPiece {
                chunk,
                start,
                samples,
                sample_rate,
            })
            .map_err(RenderError::Sink);
        };
        let started = std::time::Instant::now();
        let ready = lv.push(samples, sample_rate);
        self.level_ns += started.elapsed().as_nanos();
        self.held.push_back((chunk, start));
        self.release(ready, sample_rate)
    }

    fn release(&mut self, ready: Vec<Vec<f32>>, sample_rate: u32) -> Result<(), RenderError> {
        for piece in ready {
            let (chunk, start) = self
                .held
                .pop_front()
                .expect("the leveller releases only pieces it holds");
            (self.sink)(StreamPiece {
                chunk,
                start,
                samples: &piece,
                sample_rate,
            })
            .map_err(RenderError::Sink)?;
        }
        Ok(())
    }

    fn finish(mut self, sample_rate: u32) -> Result<StreamReport, RenderError> {
        if let Some(lv) = self.leveller.as_mut() {
            let started = std::time::Instant::now();
            let ready = lv.finish();
            self.level_ns += started.elapsed().as_nanos();
            self.release(ready, sample_rate)?;
        }
        Ok(StreamReport {
            mode: self.mode,
            gain: self.leveller.as_ref().and_then(StreamLevel::gain),
            held_s: self.leveller.as_ref().map_or(0.0, StreamLevel::held_seconds),
            level_ns: self.level_ns,
        })
    }
}

/// Level a read in place: one gain to the profile's [`target_rms_db`](Profile::target_rms_db) over the samples
/// above its [`level_floor`](Profile::level_floor), then a clip to ±[`clip`](Profile::clip), rounded to float32.
/// Returns the gain, or `None` (and leaves the read alone) when no sample is above the floor.
pub fn level(profile: &Profile, read: &mut [f32]) -> Option<f64> {
    let gain = level_gain(profile, read)?;
    apply_gain(profile, read, gain);
    Some(gain)
}

/// The gain [`level`] applies to `read`: to the profile's [`target_rms_db`](Profile::target_rms_db) over the samples
/// above its [`level_floor`](Profile::level_floor). `None` when no sample is above the floor.
pub fn level_gain(profile: &Profile, read: &[f32]) -> Option<f64> {
    let floor = profile.level_floor;
    let squares: Vec<f32> = read.iter().filter(|x| x.abs() > floor).map(|x| x * x).collect();
    if squares.is_empty() {
        return None;
    }
    let rms = f64::from(mean_f32(&squares).sqrt());
    Some(pow(10.0, (profile.target_rms_db - 20.0 * log10(rms + 1e-9)) / 20.0))
}

/// Multiply by `gain`, clip to ±[`clip`](Profile::clip), and round to float32: [`level`]'s arithmetic, per sample.
pub fn apply_gain(profile: &Profile, read: &mut [f32], gain: f64) {
    let clip = profile.clip;
    for x in read.iter_mut() {
        *x = (f64::from(*x) * gain).clamp(-clip, clip) as f32;
    }
}
