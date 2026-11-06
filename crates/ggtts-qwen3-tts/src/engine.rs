//! The Qwen3-TTS engine behind the core's [`Engine`] trait: the talker and code predictor for
//! [`generate`](Engine::generate), the codec decoder for [`decode`](Engine::decode), and the
//! `story-v1/qwen3-tts` profile.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::profile::Profile;

use crate::codec::Decoder;
use crate::cpu::Device;
use crate::talker::{language_id, FrameStop, Sampling, Talker};
use crate::PROFILE;

/// What a request is rendered with besides its text and seed. The defaults are the reference CLI's, with the
/// published model's one speaker, on CPU, untraced.
#[derive(Debug, Clone)]
pub struct Voice {
    pub speaker: String,
    /// A language the reference CLI's `-l` accepts, for example `zh` or `auto`.
    pub language: String,
    /// The frame cap (`--max-tokens`).
    pub max_frames: i32,
    /// Checked at [`Qwen3Tts::load`]: a temperature at or below zero (greedy decoding) is refused.
    pub sampling: Sampling,
    pub device: Device,
    /// Write the stop-probability trace here, one JSON object per talker step. The trace is read-only: the
    /// rendered output is the same with or without it.
    pub trace: Option<PathBuf>,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            speaker: "hk_cantonese_speaker".into(),
            language: "zh".into(),
            max_frames: 1200,
            sampling: Sampling::default(),
            device: Device::Cpu,
            trace: None,
        }
    }
}

pub struct Qwen3Tts {
    info: EngineInfo,
    talker: Talker,
    decoder: Decoder,
    voice: Voice,
    language_id: i32,
    trace: Option<BufWriter<File>>,
}

impl Qwen3Tts {
    /// Loads the talker GGUF and the codec GGUF on `voice.device` with `threads` CPU threads. Refuses
    /// `voice.sampling` before reading either file if it would decode greedily ([`Sampling::check`]).
    pub fn load(talker: &Path, codec: &Path, threads: i32, voice: Voice) -> Result<Self, EngineError> {
        voice.sampling.check()?;
        let language_id =
            language_id(&voice.language).ok_or_else(|| EngineError(format!("unknown language: {}", voice.language)))?;
        let talker = Talker::load_on(talker, threads, voice.device)?;
        if talker.speaker_id(&voice.speaker).is_none() {
            return Err(EngineError(format!("the talker has no speaker {}", voice.speaker)));
        }
        let trace = match &voice.trace {
            Some(p) => Some(BufWriter::new(
                File::create(p).map_err(|e| EngineError(format!("{}: {e}", p.display())))?,
            )),
            None => None,
        };
        Ok(Self {
            info: EngineInfo {
                family: "qwen3-tts".into(),
                backend: if voice.device.is_cuda() {
                    "ggml-cuda"
                } else {
                    "ggml-cpu"
                }
                .into(),
                // Exact parity is measured on CPU only (the golden fixtures). CUDA stays `unmeasured` until fixtures
                // recorded with the reference on CUDA pass both identity checks.
                parity: if voice.device.is_cuda() { "unmeasured" } else { "exact" }.into(),
            },
            talker,
            decoder: Decoder::load_on(codec, threads, voice.device)?,
            voice,
            language_id,
            trace,
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
        let mut steps: Vec<FrameStop> = Vec::new();
        let g = self.talker.generate_traced(
            req.text,
            &self.voice.speaker,
            self.language_id,
            req.seed,
            self.voice.max_frames,
            self.voice.sampling,
            self.trace.as_ref().map(|_| &mut steps),
        )?;
        if let Some(w) = self.trace.as_mut() {
            for s in &steps {
                writeln!(
                    w,
                    "{{\"chunk\":{},\"draw\":{},\"seed\":{},\"frame\":{},\"token\":{},\"p_stop\":{:e}}}",
                    req.chunk, req.draw, req.seed, s.frame, s.token, s.p_stop
                )
                .and_then(|_| w.flush())
                .map_err(|e| EngineError(format!("trace: {e}")))?;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_is_refused_at_load() {
        for temperature in [0.0, -1.0, f32::NAN] {
            let voice = Voice {
                sampling: Sampling {
                    temperature,
                    ..Sampling::default()
                },
                ..Voice::default()
            };
            // The files do not exist: the refusal comes first, before either is read.
            let e = Qwen3Tts::load(Path::new("no-talker.gguf"), Path::new("no-codec.gguf"), 1, voice)
                .err()
                .expect("greedy sampling must be refused");
            assert!(e.0.contains("greedy decoding"), "{temperature}: {}", e.0);
        }
    }

    #[test]
    fn the_default_sampling_passes() {
        assert!(Sampling::default().check().is_ok());
    }
}
