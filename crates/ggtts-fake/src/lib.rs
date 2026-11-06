//! The fake engine.
//!
//! It implements the core's [`Engine`] with no model and no compute backend. Each chunk text has a scripted clip
//! length per draw (by default a quarter second per CJK character), the audio is a 220 Hz tone with fixed lead
//! and tail silence, and alternate chunks come back 6 dB apart, so the read's level is not one chunk's level. A
//! chunk may also be given its own amplitude, a click pair or a sample rate. The pipeline's parity cases run
//! against it, so the pipeline is tested before any real engine exists.
//!
//! It stands in for Qwen3-TTS (its default rate is that codec's), so by default it declares that engine's
//! profile, `story-v1/qwen3-tts`. [`FakeEngine::with_profile`] declares another, for a planted change or a second
//! architecture's numbers.
//!
//! Its codec tokens are a deterministic function of the seed, the frame and the codebook, so a seed change is
//! visible in the tokens. They are not the reference sampler's stream: that arrives with the first real engine.
//!
//! Traced, it emits a `step` per frame (see [`ggtts_core::trace`]) whose values are scripted but keep the fields'
//! meanings, so a test can check them:
//!
//! * `token` is the frame's codebook-0 token, the one in the returned codes.
//! * `p_stop` rises with the frame, `0.001 + 0.989 * ((f + 1) / n)^8`, so it peaks at the stopping frame.
//! * `top1_p` is in `[0.25, 0.88]`, and `entropy` is `-ln(top1_p) + 0.5`, never below the bound `-ln(top1_p)`
//!   that any distribution with that top token obeys.
//! * `rank` is in `0..4`. At level 2, `cb` is codebooks 1 to 15 of the frame and `cb_p` each one's probability,
//!   in `[0.5, 0.99]`.
//!
//! No instrument value feeds back into the tokens. With the `plant` feature, [`FakeEngine::plant_stream_draw`]
//! makes one that does: traced, it draws once from the stream at a chosen chunk and frame, which shifts every
//! later token. It exists to show the trace-identity check goes red.

use ggtts_core::hooks::{Hooks, Step as HookStep, Wants};
use std::collections::HashMap;

use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm, Stop};
use ggtts_core::profile::Profile;
use ggtts_core::text::cjk_len;
use ggtts_core::trace::{Event, Events, NullEvents};

pub mod signal;

/// Silence before the tone, in seconds.
pub const LEAD_S: f64 = 0.30;
/// Silence after the tone, in seconds.
pub const TAIL_S: f64 = 0.50;
/// The default sample rate (the Qwen3-TTS codec's).
pub const SAMPLE_RATE: u32 = 24_000;
/// Codec frames per second, for the fake tokens.
pub const FRAME_RATE: f64 = 12.5;
/// Codebooks per frame, for the fake tokens.
pub const CODEBOOKS: u16 = 16;

/// A chunk's own shape, overriding the defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Shape {
    /// The tone's amplitude scale, instead of the alternating 1.0 and 0.5.
    pub amplitude_scale: Option<f64>,
    /// A click pair inside the speech: `+spike` at the middle and `-spike` seven samples later.
    pub spike: Option<f64>,
    /// The clip's sample rate, instead of [`SAMPLE_RATE`].
    pub sample_rate: Option<u32>,
}

/// One `generate` call, as the parity cases record it.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub text: String,
    pub seed: u64,
    pub speech_s: f64,
    pub amplitude_scale: f64,
}

#[derive(Debug, Clone)]
struct Pending {
    text: String,
    speech_s: f64,
    amplitude: f64,
    sample_rate: u32,
    spike: Option<f64>,
}

/// A scripted engine for tests.
#[derive(Debug)]
pub struct FakeEngine {
    info: EngineInfo,
    profile: Profile,
    script: HashMap<String, Vec<f64>>,
    shape: HashMap<String, Shape>,
    draws: HashMap<String, usize>,
    pending: Option<Pending>,
    /// Every `generate` call, in order.
    pub calls: Vec<Call>,
    /// The last decoded clip per chunk text: the draw the pipeline kept.
    pub kept: HashMap<String, Vec<f32>>,
    /// A planted defect: when traced, draw once from the stream at `(chunk, frame)`. Test builds only.
    #[cfg(feature = "plant")]
    pub plant_stream_draw: Option<(u32, u32)>,
    /// A planted defect: whenever a tensor hook wants anything, the export draws once from the stream before the
    /// first frame, as a hook path with a side effect would. Test builds only.
    #[cfg(feature = "plant")]
    pub plant_hook_export_draw: bool,
    /// How many `p_stop` values and logit vectors were computed for tensor hooks: zero when no hook asks.
    pub exported: Exported,
}

/// What the fake engine computed for tensor hooks, so a test can see that nothing is computed unasked.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Exported {
    pub p_stop: u64,
    pub logits: u64,
}

/// The fake's codebook-0 logits at frame `f`: eight values from the seed, never from the stream the tokens are drawn
/// from, so exporting them changes nothing.
pub fn fake_logits(seed: u64, f: u64) -> [f32; 8] {
    std::array::from_fn(|i| f32::from(fake_token(seed, f, 64 + i as u64) % 1000) / 100.0 - 5.0)
}

impl Default for FakeEngine {
    fn default() -> Self {
        Self::new(HashMap::new(), HashMap::new())
    }
}

impl FakeEngine {
    /// `script` maps a chunk text to the speech seconds of its draws (the last repeats); `shape` maps a chunk text
    /// to its overrides. The engine declares `story-v1/qwen3-tts`.
    pub fn new(script: HashMap<String, Vec<f64>>, shape: HashMap<String, Shape>) -> Self {
        Self::with_profile(ggtts_qwen3_tts::PROFILE, script, shape)
    }

    /// As [`FakeEngine::new`], declaring `profile` instead.
    pub fn with_profile(profile: Profile, script: HashMap<String, Vec<f64>>, shape: HashMap<String, Shape>) -> Self {
        Self {
            info: EngineInfo {
                family: "fake".into(),
                backend: "none".into(),
                parity: "none".into(),
            },
            profile,
            script,
            shape,
            draws: HashMap::new(),
            pending: None,
            calls: Vec::new(),
            kept: HashMap::new(),
            #[cfg(feature = "plant")]
            plant_stream_draw: None,
            #[cfg(feature = "plant")]
            plant_hook_export_draw: false,
            exported: Exported::default(),
        }
    }

    /// The default clip length: a quarter second per CJK character, at least one.
    pub fn default_speech_s(text: &str) -> f64 {
        0.25 * std::cmp::max(1, cjk_len(text)) as f64
    }
}

/// A token from `(seed, frame, codebook)`: splitmix64, reduced to a 2048-entry codebook.
pub fn fake_token(seed: u64, frame: u64, codebook: u64) -> u16 {
    let mut z = seed
        .wrapping_add(frame.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(codebook.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) % 2048) as u16
}

/// The fake's scripted stop probability at frame `f` of `n`: rising, highest at the last frame.
pub fn fake_p_stop(f: u64, n: u64) -> f64 {
    0.001 + 0.989 * libm::pow((f + 1) as f64 / n.max(1) as f64, 8.0)
}

impl FakeEngine {
    /// The frame from which the planted instrument shifts the stream, when it fires for this request.
    #[allow(unused_variables)]
    fn planted_frame(&self, req: &ChunkRequest<'_>, events: &dyn Events, wants: Wants) -> Option<u64> {
        #[cfg(feature = "plant")]
        if self.plant_hook_export_draw && wants.any() {
            return Some(0);
        }
        #[cfg(feature = "plant")]
        if events.level() > 0 {
            if let Some((chunk, frame)) = self.plant_stream_draw {
                if chunk == req.chunk {
                    return Some(u64::from(frame));
                }
            }
        }
        None
    }
}

impl Engine for FakeEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn profile(&self) -> &Profile {
        &self.profile
    }

    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.generate_traced(req, &mut NullEvents)
    }

    fn generate_traced(&mut self, req: &ChunkRequest<'_>, events: &mut dyn Events) -> Result<Generated, EngineError> {
        self.generate_hooked(req, events, &mut Hooks::new())
    }

    fn generate_hooked(
        &mut self,
        req: &ChunkRequest<'_>,
        events: &mut dyn Events,
        hooks: &mut Hooks,
    ) -> Result<Generated, EngineError> {
        let wants = hooks.wants();
        let text = req.text.to_string();
        let k = *self.draws.get(&text).unwrap_or(&0);
        self.draws.insert(text.clone(), k + 1);
        let speech_s = match self.script.get(&text) {
            Some(plan) if !plan.is_empty() => plan[std::cmp::min(k, plan.len() - 1)],
            _ => Self::default_speech_s(&text),
        };
        let shape = self.shape.get(&text).cloned().unwrap_or_default();
        // Alternate distinct chunk texts come back 6 dB apart; redraws of one text keep its amplitude.
        let amplitude = shape
            .amplitude_scale
            .unwrap_or(if self.draws.len() % 2 == 1 { 1.0 } else { 0.5 });
        self.calls.push(Call {
            text: text.clone(),
            seed: req.seed,
            speech_s,
            amplitude_scale: amplitude,
        });
        let n_frames = (speech_s * FRAME_RATE).ceil() as u64;
        let planted = self.planted_frame(req, events, wants);
        // The stream: one extra draw from the planted frame on moves every later token.
        let stream = |f: u64| {
            if planted.is_some_and(|p| f >= p) {
                req.seed.wrapping_add(1)
            } else {
                req.seed
            }
        };
        let cbs = u64::from(CODEBOOKS);
        let frames: Vec<u16> = (0..n_frames)
            .flat_map(|f| (0..cbs).map(move |c| fake_token(stream(f), f, c)))
            .collect();
        let level = events.level();
        if level > 0 {
            for f in 0..n_frames {
                let row = &frames[(f * cbs) as usize..((f + 1) * cbs) as usize];
                // Instrument values come from the seed, never from the stream the tokens are drawn from.
                let top1_p = 0.25 + f64::from(fake_token(req.seed, f, cbs) % 64) / 100.0;
                let cb_p: Vec<f64> = row[1..].iter().map(|&t| 0.5 + f64::from(t % 50) / 100.0).collect();
                events.emit(Event::Step {
                    chunk: req.chunk,
                    frame: f as u32,
                    token: row[0],
                    p_stop: Some(fake_p_stop(f, n_frames)),
                    entropy: Some(-libm::log(top1_p) + 0.5),
                    top1_p: Some(top1_p),
                    rank: Some(u32::from(fake_token(req.seed, f, cbs + 1) % 4)),
                    cb: (level >= 2).then_some(&row[1..]),
                    cb_p: (level >= 2).then_some(&cb_p[..]),
                });
            }
        }
        if wants.any() {
            for f in 0..n_frames {
                let p_stop = wants.p_stop.then(|| fake_p_stop(f, n_frames));
                let logits = wants.logits.then(|| fake_logits(req.seed, f));
                self.exported.p_stop += u64::from(p_stop.is_some());
                self.exported.logits += u64::from(logits.is_some());
                hooks.step(&HookStep {
                    chunk: req.chunk,
                    draw: req.draw,
                    seed: req.seed,
                    frame: f as u32,
                    token: u32::from(frames[(f * cbs) as usize]),
                    p_stop,
                    logits: logits.as_ref().map(|l| &l[..]),
                });
            }
        }
        self.pending = Some(Pending {
            text,
            speech_s,
            amplitude,
            sample_rate: shape.sample_rate.unwrap_or(SAMPLE_RATE),
            spike: shape.spike,
        });
        Ok(Generated {
            codes: Codes {
                codebooks: CODEBOOKS,
                frames,
            },
            stop: Stop::Eos,
        })
    }

    fn decode(&mut self, _codes: &Codes) -> Result<Pcm, EngineError> {
        let p = self
            .pending
            .take()
            .ok_or_else(|| EngineError("the fake engine decodes only right after its own generate".into()))?;
        let amp = p.amplitude as f32;
        let mut y: Vec<f32> = signal::tone(p.speech_s, LEAD_S, TAIL_S, p.sample_rate)
            .into_iter()
            .map(|x| x * amp)
            .collect();
        if let Some(spike) = p.spike {
            let mid = ((LEAD_S + p.speech_s / 2.0) * f64::from(p.sample_rate)).round_ties_even() as usize;
            y[mid] = spike as f32;
            y[mid + 7] = -spike as f32;
        }
        self.kept.insert(p.text, y.clone());
        Ok(Pcm {
            samples: y,
            sample_rate: p.sample_rate,
        })
    }
}
