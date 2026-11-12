//! The architecture plug point: one engine per model family.
//!
//! `generate` (text to codec tokens) and `decode` (tokens to audio) are separate on purpose: the parity ladder
//! checks tokens first and the decoder second, and an engine that only returned audio could not say which half
//! drifted. The instrument events (`--trace`) reach an engine through [`Engine::generate_traced`]; the format and
//! its rules are in [`trace`](crate::trace).
//!
//! An engine also declares its [`Profile`]: the protocol numbers the story path renders its output with.

use crate::hooks::Hooks;
use crate::profile::Profile;
use crate::trace::Events;

/// What an engine is, for provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// The model family, for example `qwen3-tts`, or `fake`.
    pub family: String,
    /// The compute backend, for example `ggml-cpu`, or `none`.
    pub backend: String,
    /// The backend's parity level: `exact`, `distribution`, `unmeasured` for a backend whose parity has not been
    /// measured yet, or `none` for a test engine.
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
    /// A natural-language style instruction ("speak slowly"), for an engine whose model takes one. `None` and
    /// `Some("")` are the same request: no instruction, rendered exactly as without the field. An engine that cannot
    /// use an instruction refuses a non-empty one ([`refuse_instruction`]) rather than ignoring it.
    pub instruction: Option<&'a str>,
}

impl<'a> ChunkRequest<'a> {
    /// The instruction, when there is one: `None` for no instruction and for an empty one.
    pub fn instruction(&self) -> Option<&'a str> {
        self.instruction.filter(|i| !i.is_empty())
    }
}

/// The error an engine that takes no style instruction returns for a request that carries one. An empty instruction
/// is no instruction, and passes.
pub fn refuse_instruction(req: &ChunkRequest<'_>, family: &str) -> Result<(), EngineError> {
    match req.instruction() {
        None => Ok(()),
        Some(_) => Err(EngineError(format!(
            "the {family} engine takes no style instruction; only qwen3-tts custom_voice (1.7B) and voice_design \
             checkpoints do"
        ))),
    }
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

impl Stop {
    /// The trace's name for it.
    pub fn as_str(self) -> &'static str {
        match self {
            Stop::Eos => "eos",
            Stop::Cap => "cap",
            Stop::Abort => "abort",
        }
    }
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
    /// Meaning of `p_stop` emitted by this engine, when known. Absence means unspecified, not zero.
    fn p_stop_semantics(&self) -> Option<&'static str> {
        None
    }
    /// The protocol numbers this engine's output is rendered with: ceilings, pauses, judge bars and level.
    fn profile(&self) -> &Profile;
    /// Text to codec tokens for one chunk.
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError>;
    /// [`generate`](Engine::generate), emitting a `step` event per frame into `events` when its level is above 0.
    /// The tokens must be identical to `generate`'s at every level: an instrument reads values the engine already
    /// has and never draws from the sampler's stream ([`check_trace_identity`](crate::trace::check_trace_identity)).
    /// The default emits nothing, for an engine with no instruments yet.
    fn generate_traced(&mut self, req: &ChunkRequest<'_>, events: &mut dyn Events) -> Result<Generated, EngineError> {
        let _ = events;
        self.generate(req)
    }
    /// [`generate_traced`](Engine::generate_traced), handing each generated frame to `hooks` with the values they
    /// ask for ([`Hooks::wants`]) and nothing else. The tokens must be identical with and without hooks.
    ///
    /// The default exports nothing: with no hook that wants anything it is `generate_traced`, and otherwise it
    /// refuses, so a hook is never silently given nothing.
    fn generate_hooked(
        &mut self,
        req: &ChunkRequest<'_>,
        events: &mut dyn Events,
        hooks: &mut Hooks,
    ) -> Result<Generated, EngineError> {
        if hooks.wants().any() {
            return Err(EngineError(format!(
                "the {} engine exports nothing to tensor hooks",
                self.info().family
            )));
        }
        self.generate_traced(req, events)
    }
    /// Codec tokens to PCM.
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(instruction: Option<&str>) -> ChunkRequest<'_> {
        ChunkRequest {
            text: "好。",
            seed: 1,
            chunk: 0,
            draw: 0,
            instruction,
        }
    }

    #[test]
    fn an_empty_instruction_is_no_instruction() {
        assert_eq!(req(None).instruction(), None);
        assert_eq!(req(Some("")).instruction(), None);
        assert_eq!(req(Some("speak slowly")).instruction(), Some("speak slowly"));
    }

    #[test]
    fn an_engine_without_instructions_refuses_one_and_names_itself() {
        assert!(refuse_instruction(&req(None), "fake").is_ok());
        assert!(refuse_instruction(&req(Some("")), "fake").is_ok());
        let e = refuse_instruction(&req(Some("speak slowly")), "kitten-tts-2").unwrap_err();
        assert!(
            e.0.contains("kitten-tts-2") && e.0.contains("no style instruction"),
            "{e}"
        );
    }
}
