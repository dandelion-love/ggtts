//! The Qwen3-TTS engine behind the core's [`Engine`] trait: the talker and code predictor for
//! [`generate`](Engine::generate), the codec decoder for [`decode`](Engine::decode), and the
//! `story-v1/qwen3-tts` profile.

use std::path::Path;

use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::profile::Profile;

use crate::codec::Decoder;
use crate::talker::{language_id, Sampling, Talker};
use crate::PROFILE;

/// What a request is rendered with besides its text and seed. The defaults are the reference CLI's, with the
/// published model's one speaker.
#[derive(Debug, Clone)]
pub struct Voice {
    pub speaker: String,
    /// A language the reference CLI's `-l` accepts, for example `zh` or `auto`.
    pub language: String,
    /// The frame cap (`--max-tokens`).
    pub max_frames: i32,
    pub sampling: Sampling,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            speaker: "hk_cantonese_speaker".into(),
            language: "zh".into(),
            max_frames: 1200,
            sampling: Sampling::default(),
        }
    }
}

pub struct Qwen3Tts {
    info: EngineInfo,
    talker: Talker,
    decoder: Decoder,
    voice: Voice,
    language_id: i32,
}

impl Qwen3Tts {
    /// Loads the talker GGUF and the codec GGUF, both on the CPU backend with `threads` threads.
    pub fn load(talker: &Path, codec: &Path, threads: i32, voice: Voice) -> Result<Self, EngineError> {
        let language_id =
            language_id(&voice.language).ok_or_else(|| EngineError(format!("unknown language: {}", voice.language)))?;
        let talker = Talker::load(talker, threads)?;
        if talker.speaker_id(&voice.speaker).is_none() {
            return Err(EngineError(format!("the talker has no speaker {}", voice.speaker)));
        }
        Ok(Self {
            info: EngineInfo {
                family: "qwen3-tts".into(),
                backend: "ggml-cpu".into(),
                parity: "exact".into(),
            },
            talker,
            decoder: Decoder::load(codec, threads)?,
            voice,
            language_id,
        })
    }
}

impl Engine for Qwen3Tts {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn profile(&self) -> &Profile {
        &PROFILE
    }

    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        let g = self.talker.generate(
            req.text,
            &self.voice.speaker,
            self.language_id,
            req.seed,
            self.voice.max_frames,
            self.voice.sampling,
        )?;
        let frames = g
            .codes
            .iter()
            .map(|&c| u16::try_from(c).map_err(|_| EngineError(format!("codec token {c} out of range"))))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Generated {
            codes: Codes { codebooks: 16, frames },
            stop: g.stop,
        })
    }

    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        if codes.codebooks != 16 {
            return Err(EngineError(format!(
                "{} codebooks; the codec needs 16",
                codes.codebooks
            )));
        }
        if codes.frames.is_empty() {
            return Ok(Pcm {
                samples: Vec::new(),
                sample_rate: self.decoder.config.sample_rate as u32,
            });
        }
        let c: Vec<i32> = codes.frames.iter().map(|&x| i32::from(x)).collect();
        self.decoder.decode(&c)
    }
}
