//! The architecture plug point: one engine per model family.
//!
//! `generate` (text to codec tokens) and `decode` (tokens to audio) are separate on purpose: the parity ladder
//! checks tokens first and the decoder second, and an engine that only returned audio could not say which half
//! drifted. The instrument events (`--trace`) join this interface with the trace format.
//!
//! An engine also declares its [`Profile`]: the protocol numbers the story path renders its output with.

use crate::profile::Profile;

/// What an engine is, for provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// The model family, for example `qwen3-tts`, or `fake`.
    pub family: String,
    /// The compute backend, for example `ggml-cpu`, or `none`.
    pub backend: String,
    /// The backend's parity level: `exact`, `distribution`, or `none` for a test engine.
    pub parity: String,
}

/// One chunk to generate.
#[derive(Debug, Clone, Copy)]
pub struct ChunkRequest<'a> {
    /// The text the engine reads, mark included.
    pub text: &'a str,
    /// The sampler seed for this chunk and draw.
    pub seed: u64,
    /// The chunk's index in the story. For events only; never part of the computation.
    pub chunk: u32,
    /// The draw index. For events only; never part of the computation.
    pub draw: u32,
}

/// Codec tokens, frame-major: `frames[f * codebooks + c]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codes {
    pub codebooks: u16,
    pub frames: Vec<u16>,
}

impl Codes {
    pub fn n_frames(&self) -> usize {
        if self.codebooks == 0 {
            0
        } else {
            self.frames.len() / usize::from(self.codebooks)
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The end-of-speech token.
    Eos,
    /// The frame cap.
    Cap,
    Abort,
}

/// The result of [`Engine::generate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generated {
    pub codes: Codes,
    pub stop: Stop,
}

/// Mono PCM.
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// An engine failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

/// One model family.
pub trait Engine {
    fn info(&self) -> &EngineInfo;
    /// The protocol numbers this engine's output is rendered with: ceilings, pauses, judge bars and level.
    fn profile(&self) -> &Profile;
    /// Text to codec tokens for one chunk.
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError>;
    /// Codec tokens to PCM.
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError>;
}
