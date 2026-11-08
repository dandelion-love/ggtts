//! The story path (render spec sections 2 to 7): chunk, seed per chunk, render, judge, redraw, trim, join, level.
//!
//! Every number here comes from the engine's [`Profile`]. Changing one in a published profile is a new protocol
//! version, with a changed set of pipeline cases.

use crate::audio::{judge_f32, trim_edges_f32, Verdict};
use crate::engine::{ChunkRequest, Engine, EngineError};
use crate::hooks::Hooks;
use crate::numeric::{log10, mean_f32, pow};
use crate::profile::Profile;
use crate::seed::seed_of;
use crate::text::{plan, PlannedChunk};
use crate::trace::{Event, Events, NullEvents};

/// Chunk a story's paragraphs at the profile's [`story_ceiling`](Profile::story_ceiling).
pub fn plan_story(profile: &Profile, paragraphs: &[String]) -> Vec<PlannedChunk> {
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
}

impl RenderOptions {
    /// The profile's defaults: its [`redraw`](Profile::redraw) and [`edge_s`](Profile::edge_s), and no salt.
    pub fn for_profile(profile: &Profile) -> Self {
        Self {
            redraw: profile.redraw,
            seed_salt: String::new(),
            edge_s: profile.edge_s,
            measure_timings: false,
        }
    }
}

/// One chunk of a rendered read.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecord {
    pub i: usize,
    pub text: String,
    /// The seed of the draw that was kept.
    pub seed: u32,
    /// Redraws taken: the kept draw's index.
    pub redraws: u32,
    pub first_draw_defect: Option<Verdict>,
    pub final_defect: Option<Verdict>,
    /// The kept draw's seconds of speech per character, unrounded.
    pub s_per_char: f64,
    pub first_s_per_char: f64,
    /// The kept chunk's samples in the read, `[start, end)`.
    pub span: (usize, usize),
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
}

#[derive(Debug, Clone, Copy, Default)]
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

/// Render one story: each chunk in order at its own seed, judged, redrawn up to `opts.redraw` times while the
/// judge rejects it (the last draw is kept either way), trimmed to `opts.edge_s`, joined with the pause table,
/// and levelled once for the whole read. The judge, the pauses and the level read the engine's
/// [`profile`](Engine::profile); plan the chunks with the same profile ([`plan_story`]).
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
    render(engine, story_id, chunks, opts, events, &mut |_| Ok(()), hooks)
}

/// One piece of a streamed read: a kept chunk, trimmed, followed by the gap after it (none after the last).
///
/// The pieces arrive in order, each as soon as its chunk is kept, and they are the read **before the level
/// step**: the level's gain is computed over the whole read, so it is known only at the end, as [`Read::gain`].
/// Concatenated, the pieces are the batch read's samples before levelling, exactly; [`level`] with the profile
/// over that concatenation gives [`Read::samples`].
#[derive(Debug, Clone, Copy)]
pub struct StreamPiece<'a> {
    pub chunk: usize,
    /// The piece's first sample in the read.
    pub start: usize,
    pub samples: &'a [f32],
    /// The read's rate: the first chunk's. Later chunks are not resampled or checked.
    pub sample_rate: u32,
}

/// [`render_story_traced`], handing each [`StreamPiece`] to `sink` as soon as its chunk is kept, before the next
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
    render(engine, story_id, chunks, opts, events, sink, &mut Hooks::new())
}

/// The story path every public render shares: events, a sink, and tensor hooks.
fn render(
    engine: &mut dyn Engine,
    story_id: &str,
    chunks: &[PlannedChunk],
    opts: &RenderOptions,
    events: &mut dyn Events,
    sink: &mut dyn FnMut(StreamPiece<'_>) -> Result<(), String>,
    hooks: &mut Hooks,
) -> Result<Read, RenderError> {
    let tracing = events.level() > 0;
    let mut total_frames: u64 = 0;
    let mut timings = opts.measure_timings.then(StageTimings::default);
    if chunks.is_empty() {
        return Err(RenderError::Empty);
    }
    let profile = *engine.profile();
    let n = chunks.len();
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate = 0;
    let mut records = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        let mut first = None;
        let mut kept = None;
        for k in 0..=opts.redraw {
            let seed = seed_of(story_id, i, &c.text, k, &opts.seed_salt);
            let req = ChunkRequest {
                text: &c.text,
                seed: u64::from(seed),
                chunk: i as u32,
                draw: k,
            };
            let t0 = tracing.then(std::time::Instant::now);
            if tracing {
                events.emit(Event::ChunkBegin {
                    chunk: req.chunk,
                    draw: k,
                    text: &c.text,
                    seed: req.seed,
                    n_text_tokens: None,
                });
            }
            let stage_start = (tracing || timings.is_some()).then(std::time::Instant::now);
            let generated = engine.generate_hooked(&req, events, hooks)?;
            if let Some(start) = stage_start {
                let elapsed = start.elapsed();
                if let Some(t) = timings.as_mut() {
                    t.generate_ns += elapsed.as_nanos();
                }
                if tracing {
                    events.emit(Event::Timing {
                        stage: "generate",
                        chunk: Some(req.chunk),
                        draw: Some(k),
                        ms: elapsed.as_millis() as u64,
                    });
                }
            }
            let stage_start = (tracing || timings.is_some()).then(std::time::Instant::now);
            let pcm = engine.decode(&generated.codes)?;
            if let Some(start) = stage_start {
                let elapsed = start.elapsed();
                if let Some(t) = timings.as_mut() {
                    t.decode_ns += elapsed.as_nanos();
                }
                if tracing {
                    events.emit(Event::Timing {
                        stage: "decode",
                        chunk: Some(req.chunk),
                        draw: Some(k),
                        ms: elapsed.as_millis() as u64,
                    });
                }
            }
            let stage_start = (tracing || timings.is_some()).then(std::time::Instant::now);
            let j = judge_f32(&profile, &pcm.samples, pcm.sample_rate, &c.text);
            if let Some(start) = stage_start {
                let elapsed = start.elapsed();
                if let Some(t) = timings.as_mut() {
                    t.judge_ns += elapsed.as_nanos();
                }
                if tracing {
                    events.emit(Event::Timing {
                        stage: "judge",
                        chunk: Some(req.chunk),
                        draw: Some(k),
                        ms: elapsed.as_millis() as u64,
                    });
                }
            }
            let frames = generated.codes.n_frames();
            total_frames += frames as u64;
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
            if k == 0 {
                first = Some(j);
            }
            let clean = j.verdict.is_none();
            kept = Some((pcm, j, seed, k));
            if clean {
                break;
            }
        }
        let (pcm, j, seed, k) = kept.expect("at least one draw");
        let first = first.expect("draw 0 ran");
        let t = trim_edges_f32(&profile, &pcm.samples, pcm.sample_rate, opts.edge_s);
        if tracing {
            events.emit(Event::Trim {
                chunk: i as u32,
                lead_s: t.lead_cut_s,
                tail_s: t.tail_cut_s,
                start: t.start_sample,
                end: t.end_sample,
            });
        }
        if i == 0 {
            sample_rate = pcm.sample_rate;
        }
        // Join as each chunk is kept: the chunk, then its gap unless it is the last.
        let start = samples.len();
        samples.extend_from_slice(&pcm.samples[t.start_sample..t.end_sample]);
        let span = (start, samples.len());
        if i + 1 < n {
            let gap = gap_samples(&profile, &c.punct, c.last_in_para, sample_rate);
            samples.resize(samples.len() + gap, 0.0);
        }
        sink(StreamPiece {
            chunk: i,
            start,
            samples: &samples[start..],
            sample_rate,
        })
        .map_err(RenderError::Sink)?;
        if tracing {
            events.emit(Event::Emit {
                chunk: i as u32,
                start,
                samples: samples.len() - start,
            });
        }
        records.push(ChunkRecord {
            i,
            text: c.text.clone(),
            seed,
            redraws: k,
            first_draw_defect: first.verdict,
            final_defect: j.verdict,
            s_per_char: j.s_per_char,
            first_s_per_char: first.s_per_char,
            span,
        });
    }

    let gain = level(&profile, &mut samples);
    if tracing {
        events.emit(Event::Join {
            chunks: n as u32,
            samples: samples.len(),
            gain_db: gain.map(|g| 20.0 * log10(g)),
        });
        events.emit(Event::End {
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
    })
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
