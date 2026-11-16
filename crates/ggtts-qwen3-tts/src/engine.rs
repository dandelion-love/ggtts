//! The Qwen3-TTS engine behind the core's [`Engine`] trait: the talker and code predictor for
//! [`generate`](Engine::generate), the codec decoder for [`decode`](Engine::decode), and the
//! `story-v1/qwen3-tts` profile.

use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use ggtts_core::capabilities::Capabilities;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::hooks::{Hooks, LayerStep, Step, StopTrace, TensorHook, Wants};
use ggtts_core::profile::Profile;
use ggtts_core::steer::Steering;
use ggtts_core::trace::{Event, Events, NullEvents};

use crate::codec::Decoder;
use crate::cpu::Device;
use crate::stream::IncrementalDecoder;
use crate::talker::{language_id, Sampling, Talker};
use crate::PROFILE;

/// What a request is rendered with besides its text and seed. The defaults are the reference CLI's, with the
/// model's default speaker, on CPU, untraced.
#[derive(Debug, Clone)]
pub struct Voice {
    /// A speaker the model has, case ignored. Empty (the default) is the model's default speaker
    /// ([`Talker::default_speaker`]), resolved at load.
    pub speaker: String,
    /// A language the reference CLI's `-l` accepts, for example `zh` or `auto`.
    pub language: String,
    /// The frame cap (`--max-tokens`).
    pub max_frames: i32,
    /// Checked at [`Qwen3Tts::load`]: a temperature at or below zero (greedy decoding) is refused.
    pub sampling: Sampling,
    pub device: Device,
    /// Write the stop-probability trace here, one JSON object per talker step. It is the core's
    /// [`StopTrace`] tensor hook, which asks for `p_stop` only; the rendered output is the same with or without it.
    pub trace: Option<PathBuf>,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            speaker: String::new(),
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
    trace: Option<StopTrace<BufWriter<File>>>,
}

impl Qwen3Tts {
    /// Loads the talker GGUF and the codec GGUF on `voice.device` with `threads` CPU threads. Refuses
    /// `voice.sampling` before reading either file if it would decode greedily ([`Sampling::check`]).
    pub fn load(talker: &Path, codec: &Path, threads: i32, voice: Voice) -> Result<Self, EngineError> {
        Self::load_for(talker, codec, threads, voice, None)
    }

    /// [`load`](Self::load) for requests that will carry `instruction`: a checkpoint that takes no style instruction
    /// refuses a non-empty one here, before the speaker is looked up or the codec read, so the error a caller sees
    /// is about the instruction ([`check_instruction`](Self::check_instruction)).
    pub fn load_for(
        talker: &Path,
        codec: &Path,
        threads: i32,
        voice: Voice,
        instruction: Option<&str>,
    ) -> Result<Self, EngineError> {
        voice.sampling.check()?;
        let language_id = language_of(&voice)?;
        let talker = Talker::load_on(talker, threads, voice.device)?;
        Self::with_talker(talker, language_id, codec, threads, voice, instruction)
    }

    /// [`load`](Self::load) with a planted defect in the talker, for the identity tests' expected failures.
    #[cfg(feature = "plant")]
    pub fn load_planted(
        talker: &Path,
        codec: &Path,
        threads: i32,
        voice: Voice,
        plant: crate::talker::Plant,
    ) -> Result<Self, EngineError> {
        voice.sampling.check()?;
        let language_id = language_of(&voice)?;
        let talker = Talker::load_planted(talker, threads, voice.device, plant)?;
        Self::with_talker(talker, language_id, codec, threads, voice, None)
    }

    fn with_talker(
        talker: Talker,
        language_id: i32,
        codec: &Path,
        threads: i32,
        mut voice: Voice,
        instruction: Option<&str>,
    ) -> Result<Self, EngineError> {
        // A checkpoint that takes no instruction refuses one here, in the words every front end shows.
        capabilities_of(&talker)
            .check_instruction(instruction)
            .map_err(|r| EngineError(r.to_string()))?;
        // No speaker named: the model's default. Before 2026-11-13 this was always hk_cantonese_speaker, which the
        // released models do not have, so every model given by path failed without --speaker.
        if voice.speaker.is_empty() && !talker.skips_default_speaker() {
            voice.speaker = talker.default_speaker().ok_or_else(|| {
                EngineError("the talker has no named speakers, so there is no default speaker to use".into())
            })?;
        }
        // An unknown speaker fails here, before any text is rendered, with the model's own speaker names.
        talker.speaker(&voice.speaker)?;
        let trace = match &voice.trace {
            Some(p) => Some(StopTrace::new(BufWriter::new(
                File::create(p).map_err(|e| EngineError(format!("{}: {e}", p.display())))?,
            ))),
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

/// What a Qwen3-TTS engine on `talker` supports: everything the family does, except a style instruction on a
/// checkpoint that takes none, which then says why (its type and hidden size).
fn capabilities_of(talker: &Talker) -> Capabilities {
    let mut c = Capabilities::of("qwen3-tts").expect("qwen3-tts is an engine");
    if !talker.supports_instruction() {
        c.instruction = false;
        c.no_instruction_why = talker.check_instruction(Some("an instruction")).err().map(|e| e.0);
    }
    c
}

/// The voice's language as the talker's id, or the error a user sees for one it does not know.
fn language_of(voice: &Voice) -> Result<i32, EngineError> {
    language_id(&voice.language).ok_or_else(|| EngineError(format!("unknown language: {}", voice.language)))
}

/// What the talker must export this request: what the caller's hooks want, plus `p_stop` when the voice has its
/// own trace.
fn wants_of(hooks: &Hooks, own: bool) -> Wants {
    let w = hooks.wants();
    Wants {
        p_stop: w.p_stop || own,
        logits: w.logits,
    }
}

/// Hand one talker step to the voice's own trace and to the caller's hooks.
fn deliver(
    req: &ChunkRequest<'_>,
    own: &mut Option<StopTrace<BufWriter<File>>>,
    hooks: &mut Hooks,
    frame: i32,
    token: i32,
    p_stop: Option<f64>,
    logits: Option<&[f32]>,
) {
    let s = Step {
        chunk: req.chunk,
        draw: req.draw,
        seed: req.seed,
        frame: frame as u32,
        token: token as u32,
        p_stop,
        p_stop_semantics: p_stop.map(|_| "qwen-sampler-weights/1"),
        logits,
    };
    if let Some(t) = own.as_mut() {
        t.step(&s);
    }
    hooks.step(&s);
}

/// The voice's own trace failed to write: the request fails, as it always has.
fn own_trace_error(own: &Option<StopTrace<BufWriter<File>>>) -> Result<(), EngineError> {
    match own.as_ref().and_then(|t| t.error()) {
        Some(e) => Err(EngineError(format!("trace: {e}"))),
        None => Ok(()),
    }
}

impl Qwen3Tts {
    /// The checkpoint type: `base`, `custom_voice` or `voice_design`.
    pub fn model_type(&self) -> &str {
        self.talker.model_type()
    }

    /// Whether the checkpoint takes a style instruction ([`ChunkRequest::instruction`]): custom_voice at 1.7B and
    /// voice_design. A request with an instruction on any other checkpoint is refused.
    pub fn supports_instruction(&self) -> bool {
        self.talker.supports_instruction()
    }

    /// `Ok` for no instruction or a checkpoint that takes one; the error [`generate`](Engine::generate) would
    /// return otherwise, without rendering anything.
    pub fn check_instruction(&self, instruction: Option<&str>) -> Result<(), EngineError> {
        self.talker.check_instruction(instruction)
    }

    /// `Ok` for steering this talker can take ([`ChunkRequest::steer`]): every layer in range and every vector of
    /// the talker's hidden size. The error [`generate`](Engine::generate) would return otherwise, without rendering
    /// anything.
    pub fn check_steer(&self, steer: &Steering) -> Result<(), EngineError> {
        self.talker.check_steer(steer)
    }

    /// The talker, for checks that drive it directly ([`Talker::prefill_hidden`]).
    pub fn talker_mut(&mut self) -> &mut Talker {
        &mut self.talker
    }

    /// Render one request with frame-level streaming: the talker hands each frame to an [`IncrementalDecoder`]
    /// as it is generated, and `sink` gets each batch of ready samples with the number of frames generated so far.
    /// `every` and `hold` are the decoder's (see [`crate::stream`]): at `hold = 32` the concatenated samples were
    /// measured bit-identical to [`decode`](Engine::decode) of the same codes on CPU. Returns the codes, as
    /// [`generate`](Engine::generate) would.
    pub fn render_streamed(
        &mut self,
        req: &ChunkRequest<'_>,
        every: usize,
        hold: usize,
        sink: &mut dyn FnMut(&[f32], usize),
    ) -> Result<Generated, EngineError> {
        let Self {
            talker,
            decoder,
            voice,
            language_id,
            trace,
            ..
        } = self;
        let mut hooks = Hooks::new();
        let wants = wants_of(&hooks, trace.is_some());
        let mut inc = IncrementalDecoder::new(decoder, every, hold);
        let mut failed: Option<EngineError> = None;
        let mut frames = 0usize;
        let mut hook = |codes: &[i32]| {
            frames += 1;
            if failed.is_some() {
                return;
            }
            match inc.push(codes) {
                Ok(Some(samples)) if !samples.is_empty() => sink(&samples, frames),
                Ok(_) => {}
                Err(e) => failed = Some(e),
            }
        };
        let mut on_step = |frame, token, p_stop, logits: Option<&[f32]>| {
            deliver(req, trace, &mut hooks, frame, token, p_stop, logits)
        };
        let g = talker.generate_steered(
            req.text,
            req.instruction(),
            &voice.speaker,
            *language_id,
            req.seed,
            voice.max_frames,
            voice.sampling,
            req.steer,
            wants.any().then_some((wants, &mut on_step as &mut _)),
            Some(&mut hook),
        )?;
        own_trace_error(trace)?;
        if let Some(e) = failed {
            return Err(e);
        }
        let tail = inc.finish()?;
        if !tail.is_empty() {
            sink(&tail, inc.frames());
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
}

impl Qwen3Tts {
    /// The speaker every request renders with: the one the voice named, or the model's default resolved at load.
    pub fn speaker(&self) -> &str {
        &self.voice.speaker
    }
}

impl Engine for Qwen3Tts {
    fn capabilities(&self) -> Capabilities {
        capabilities_of(&self.talker)
    }
    fn speakers(&self) -> Vec<String> {
        self.talker.speaker_names()
    }
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn p_stop_semantics(&self) -> Option<&'static str> {
        Some("qwen-sampler-weights/1")
    }

    fn supports_layer_hooks(&self) -> bool {
        true
    }

    fn profile(&self) -> &Profile {
        &PROFILE
    }

    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.generate_hooked(req, &mut NullEvents, &mut Hooks::new())
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
        let tracing = events.level() > 0;
        let wants = wants_of(hooks, self.trace.is_some() || tracing);
        let layers = hooks.layers();
        let hooks = RefCell::new(hooks);
        let trace = &mut self.trace;
        let mut on_step = |frame, token, p_stop, logits: Option<&[f32]>| {
            if tracing {
                if let Ok(token) = u16::try_from(token) {
                    events.emit(Event::Step {
                        chunk: req.chunk,
                        frame: frame as u32,
                        token,
                        p_stop,
                        entropy: None,
                        top1_p: None,
                        rank: None,
                        cb: None,
                        cb_p: None,
                    });
                }
            }
            deliver(req, trace, *hooks.borrow_mut(), frame, token, p_stop, logits);
        };
        let mut on_layer = |frame: i32, token: i32, layer: usize, is_eos: bool, values: &[f32]| {
            let step = LayerStep {
                chunk: req.chunk,
                draw: req.draw,
                seed: req.seed,
                frame: frame as u32,
                token: token as u32,
                is_eos,
                layer,
                values,
            };
            hooks.borrow_mut().layer(&step);
        };
        let layer_callback = (!layers.is_empty()).then_some(&mut on_layer as &mut _);
        let g = self.talker.generate_steered_with_layers(
            req.text,
            req.instruction(),
            &self.voice.speaker,
            self.language_id,
            req.seed,
            self.voice.max_frames,
            self.voice.sampling,
            req.steer,
            &layers,
            layer_callback,
            wants.any().then_some((wants, &mut on_step as &mut _)),
            None,
        )?;
        own_trace_error(&self.trace)?;
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

    fn supports_cancellation(&self) -> bool {
        self.trace.is_none()
    }

    fn generate_cancellable(
        &mut self,
        req: &ChunkRequest<'_>,
        signal: &ggtts_core::cancel::CancelSignal,
    ) -> Result<ggtts_core::cancel::CancelOutcome<Generated>, EngineError> {
        use ggtts_core::cancel::CancelOutcome;
        if self.trace.is_some() {
            return Ok(CancelOutcome::Unsupported);
        }
        match self.talker.generate_cancellable(
            req.text,
            req.instruction(),
            &self.voice.speaker,
            self.language_id,
            req.seed,
            self.voice.max_frames,
            self.voice.sampling,
            req.steer,
            signal,
        )? {
            CancelOutcome::Stopped => Ok(CancelOutcome::Stopped),
            CancelOutcome::Unsupported => Ok(CancelOutcome::Unsupported),
            CancelOutcome::Completed(g) => {
                let frames = g
                    .codes
                    .iter()
                    .map(|&c| u16::try_from(c).map_err(|_| EngineError(format!("codec token {c} out of range"))))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CancelOutcome::Completed(Generated {
                    codes: Codes { codebooks: 16, frames },
                    stop: g.stop,
                }))
            }
        }
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

    /// #351 AC4: a greedy sub-talker is refused at load too, with the reason.
    #[test]
    fn greedy_subtalker_sampling_is_refused_at_load() {
        for t in [0.0, -1.0, f32::NAN] {
            let voice = Voice {
                sampling: Sampling {
                    subtalker_temperature: Some(t),
                    ..Sampling::default()
                },
                ..Voice::default()
            };
            let e = Qwen3Tts::load(Path::new("no-talker.gguf"), Path::new("no-codec.gguf"), 1, voice)
                .err()
                .expect("a greedy sub-talker must be refused");
            assert!(e.0.contains("greedy decoding of the sub-talker"), "{t}: {}", e.0);
        }
    }

    /// Review of #351: a top-k of 1 is greedy decoding, refused at load for the talker and the sub-talker.
    #[test]
    fn a_top_k_of_1_is_refused_at_load() {
        for (sampling, says) in [
            (
                Sampling {
                    top_k: 1,
                    ..Sampling::default()
                },
                "top-k 1 refused",
            ),
            (
                Sampling {
                    subtalker_top_k: Some(1),
                    ..Sampling::default()
                },
                "sub-talker top-k 1 refused",
            ),
        ] {
            let voice = Voice {
                sampling,
                ..Voice::default()
            };
            let e = Qwen3Tts::load(Path::new("no-talker.gguf"), Path::new("no-codec.gguf"), 1, voice)
                .err()
                .expect("top-k 1 must be refused");
            assert!(e.0.contains(says) && e.0.contains("greedy decoding"), "{}", e.0);
        }
        assert!(Sampling {
            top_k: 2,
            subtalker_top_k: Some(0),
            ..Sampling::default()
        }
        .check()
        .is_ok());
    }

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
