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

use std::collections::HashMap;

use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm, Stop};
use ggtts_core::profile::Profile;
use ggtts_core::text::cjk_len;

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

impl Engine for FakeEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn profile(&self) -> &Profile {
        &self.profile
    }

    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
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
        let frames = (0..n_frames)
            .flat_map(|f| (0..u64::from(CODEBOOKS)).map(move |c| fake_token(req.seed, f, c)))
            .collect();
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
