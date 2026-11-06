//! The story path (render spec sections 2 to 7): chunk, seed per chunk, render, judge, redraw, trim, join, level.
//!
//! Every default here is part of the protocol `story-v1`. Changing one is a new protocol version, with a changed
//! set of pipeline cases.

use crate::audio::{judge_f32, trim_edges_f32, Verdict};
use crate::engine::{ChunkRequest, Engine, EngineError};
use crate::numeric::{log10, mean_f32, pow};
use crate::seed::seed_of;
use crate::text::PlannedChunk;

/// The protocol these defaults define.
pub const PROTOCOL: &str = "story-v1";
/// The story path's chunk ceiling, in code points.
pub const STORY_CEILING: usize = 45;
/// The passage path's ceiling. Not a mode: a caller passes it explicitly.
pub const PASSAGE_CEILING: usize = 26;
/// Silence kept at each edge of a chunk after trimming, in seconds.
pub const EDGE_S: f64 = 0.12;
/// Redraws of a chunk the judge rejects. Zero: every chunk is its first draw, defects and all.
pub const DEFAULT_REDRAW: u32 = 0;
/// Extra pause after a paragraph's last chunk, in seconds.
pub const PARA_EXTRA_S: f64 = 0.45;
/// The read's loudness target: RMS over the samples above [`LEVEL_FLOOR`], in dBFS.
pub const TARGET_RMS_DB: f64 = -20.0;
/// Samples at or below this magnitude are not measured by the level step.
pub const LEVEL_FLOOR: f32 = 1e-3;
/// The level step clips to this magnitude.
pub const CLIP: f64 = 0.999;

/// The pause after a chunk, by the mark that ended it, in seconds (render spec section 7).
pub fn pause_s(punct: &str) -> f64 {
    match punct {
        "，" => 0.20,
        "、" => 0.16,
        "：" => 0.30,
        "；" => 0.32,
        "。" | "！" => 0.48,
        "？" => 0.50,
        _ => 0.18,
    }
}

/// The gap after a chunk, in samples: the pause, plus [`PARA_EXTRA_S`] at a paragraph end, times the rate,
/// truncated.
pub fn gap_samples(punct: &str, last_in_para: bool, sample_rate: u32) -> usize {
    let extra = if last_in_para { PARA_EXTRA_S } else { 0.0 };
    ((pause_s(punct) + extra) * f64::from(sample_rate)) as usize
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
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            redraw: DEFAULT_REDRAW,
            seed_salt: String::new(),
            edge_s: EDGE_S,
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
}

impl Read {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
    pub fn first_draw_defects(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.first_draw_defect.is_some())
            .count()
    }
    pub fn final_defects(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.final_defect.is_some())
            .count()
    }
}

/// Why a story could not be rendered.
#[derive(Debug)]
pub enum RenderError {
    /// The plan has no chunks.
    Empty,
    Engine(EngineError),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Empty => write!(f, "the story has no chunk to render"),
            RenderError::Engine(e) => write!(f, "engine: {e}"),
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
/// and levelled once for the whole read.
pub fn render_story(
    engine: &mut dyn Engine,
    story_id: &str,
    chunks: &[PlannedChunk],
    opts: &RenderOptions,
) -> Result<Read, RenderError> {
    if chunks.is_empty() {
        return Err(RenderError::Empty);
    }
    let mut clips: Vec<(Vec<f32>, u32)> = Vec::with_capacity(chunks.len());
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
            let generated = engine.generate(&req)?;
            let pcm = engine.decode(&generated.codes)?;
            let j = judge_f32(&pcm.samples, pcm.sample_rate, &c.text);
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
        let t = trim_edges_f32(&pcm.samples, pcm.sample_rate, opts.edge_s);
        clips.push((
            pcm.samples[t.start_sample..t.end_sample].to_vec(),
            pcm.sample_rate,
        ));
        records.push(ChunkRecord {
            i,
            text: c.text.clone(),
            seed,
            redraws: k,
            first_draw_defect: first.verdict,
            final_defect: j.verdict,
            s_per_char: j.s_per_char,
            first_s_per_char: first.s_per_char,
            span: (0, 0),
        });
    }

    let sample_rate = clips[0].1;
    let mut samples = Vec::new();
    let n = clips.len();
    for (idx, ((y, _), c)) in clips.iter().zip(chunks).enumerate() {
        let start = samples.len();
        samples.extend_from_slice(y);
        records[idx].span = (start, samples.len());
        if idx + 1 < n {
            let gap = gap_samples(&c.punct, c.last_in_para, sample_rate);
            samples.resize(samples.len() + gap, 0.0);
        }
    }
    let gain = level(&mut samples);
    Ok(Read {
        chunks: records,
        samples,
        sample_rate,
        gain,
    })
}

/// Level a read in place: one gain to [`TARGET_RMS_DB`] over the samples above [`LEVEL_FLOOR`], then a clip to
/// ±[`CLIP`], rounded to float32. Returns the gain, or `None` (and leaves the read alone) when no sample is above
/// the floor.
pub fn level(read: &mut [f32]) -> Option<f64> {
    let squares: Vec<f32> = read
        .iter()
        .filter(|x| x.abs() > LEVEL_FLOOR)
        .map(|x| x * x)
        .collect();
    if squares.is_empty() {
        return None;
    }
    let rms = f64::from(mean_f32(&squares).sqrt());
    let gain = pow(10.0, (TARGET_RMS_DB - 20.0 * log10(rms + 1e-9)) / 20.0);
    for x in read.iter_mut() {
        *x = (f64::from(*x) * gain).clamp(-CLIP, CLIP) as f32;
    }
    Some(gain)
}
