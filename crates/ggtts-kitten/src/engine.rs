//! The Kitten-TTS 2 engine behind ggtts's [`Engine`] trait.
//!
//! `generate`: the prompt (`kittenml/prompt.py`, `cpp/tools/kitten-tts/sampling.h:22-35`), the speaker projection
//! over position 0, prefill, then one token at a time through the sampler until `speech_end` or `stop`. Only ids
//! in the audio range become codes.
//!
//! `decode`: the S3Gen decoder is not ported to ggml yet (DESIGN.md, "The decoder, honestly"). It runs Kitten's own
//! PyTorch S3Gen through `tools/kitten_ref.py decode`, a separate process, and says so in [`EngineInfo`].

use std::path::{Path, PathBuf};
use std::process::Command;

use ggtts_core::engine::{
    refuse_instruction, ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm, Stop,
};
use ggtts_core::profile::Profile;

use crate::backend::{Backend, Device};
use crate::dsp::{self, Noise};
use crate::flow::{Flow, FlowRef};
use crate::gguf::Value;
use crate::hift::Hift;
use crate::model::{Model, Session};
use crate::prompt::TokenMap;
use crate::sampler::{self, Ids, Rng, Settings};
use crate::tokenizer::Tokenizer;

/// A provisional profile: Kitten's own chunking numbers are characters per chunk (`config.json`
/// `generation.chunk_chars` 380), and no story protocol has been recorded for this family yet.
pub const PROFILE: Profile = Profile {
    name: "draft/kitten-tts-2",
    protocol: "draft",
    story_ceiling: 380,
    passage_ceiling: 380,
    min_chunk_chars: 1,
    split_marks: &['.', '!', '?', ';', ':'],
    unmarked_joiner: ',',
    redraw: 0,
    edge_s: 0.12,
    top_db: 40.0,
    runaway_spc: 0.60,
    silent_spc: 0.01,
    pauses: &[(".", 0.30), ("!", 0.30), ("?", 0.30)],
    default_pause_s: 0.2,
    para_extra_s: 0.4,
    target_rms_db: -20.0,
    level_floor: 1e-3,
    clip: 0.999,
};

/// A reference voice, as `tools/kitten_ref.py reference` writes it.
pub struct Reference {
    pub transcript: String,
    pub tokens: Vec<u32>,
    pub embedding: Vec<f32>,
    /// The JSON file itself, which the decoder bridge reads for its conditioning.
    pub path: PathBuf,
    /// The S3Gen conditioning, when the reference carries it (what the ggml decoder needs).
    pub s3: Option<FlowRef>,
}

/// How a reference voice is made, for every refusal of one.
pub const PREPARE_HINT: &str = "make a reference voice with ggtts-kitten, which ships beside ggtts-render: `ggtts-kitten prepare --s3gguf kitten-s3gen.gguf --wav CLIP.flac --transcript TEXT --out voice.json`";

impl Reference {
    pub fn open(path: &Path) -> Result<Self, String> {
        let bad = |why: String| format!("{}: {why}; {PREPARE_HINT}", path.display());
        let s = std::fs::read(path).map_err(|e| bad(e.to_string()))?;
        let j: serde_json::Value =
            serde_json::from_slice(&s).map_err(|e| bad(format!("not a reference voice JSON ({e})")))?;
        Ok(Reference {
            transcript: j["transcript"]
                .as_str()
                .ok_or_else(|| bad("the reference has no transcript".into()))?
                .trim()
                .to_string(),
            tokens: j["reference_tokens"]
                .as_array()
                .ok_or_else(|| bad("the reference has no reference_tokens".into()))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as u32)
                .collect(),
            embedding: j["embedding"]
                .as_array()
                .ok_or_else(|| bad("the reference has no embedding".into()))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect(),
            path: path.to_path_buf(),
            s3: j.get("s3gen").map(|c| {
                let f = |k: &str| -> Vec<f32> {
                    c[k].as_array()
                        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
                        .unwrap_or_default()
                };
                FlowRef {
                    prompt_token: c["prompt_token"]
                        .as_array()
                        .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
                        .unwrap_or_default(),
                    prompt_feat: f("prompt_feat"),
                    embedding: f("embedding"),
                }
            }),
        })
    }
}

/// S3Gen on ggml: the flow, then the HiFT vocoder.
pub struct GgmlDecoder {
    pub flow: Flow,
    pub hift: Hift,
}

impl GgmlDecoder {
    /// `S3Codec.decode`: three silence tokens appended (S3Gen drops its last few as lookahead), the flow from
    /// Gaussian noise, the vocoder with a random harmonic source, and the fade-in. The noise and the source come
    /// from `seed`, so the same codes and seed give the same audio.
    pub fn decode(&self, codes: &[u16], r: &FlowRef, seed: u64) -> Result<Vec<f32>, String> {
        let mut toks: Vec<u32> = codes.iter().filter(|&&c| c < 6561).map(|&c| u32::from(c)).collect();
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        toks.extend([4299u32; 3]);
        let tn = 2 * (r.prompt_token.len() + toks.len());
        let mut noise = Noise::new(seed);
        let z: Vec<f32> = (0..crate::flow::MEL * tn).map(|_| noise.gauss() as f32).collect();
        let (mel, frames) = self.flow.mel(&toks, r, &z)?;
        let f0 = self.hift.f0(&mel, frames)?;
        let src = self.hift.source(&f0, &mut noise)?;
        let (spec, _) = dsp::stft(&src);
        let mut wav = self.hift.decode(&mel, frames, &spec)?;
        crate::hift::trim_fade(&mut wav);
        Ok(wav)
    }
}

/// Which decoder `decode` uses.
pub enum Decoder {
    /// S3Gen ported to ggml. The default when the S3Gen GGUF and the reference's conditioning are given.
    Ggml(Box<GgmlDecoder>),
    /// Kitten's own PyTorch S3Gen in a separate process: the labelled fallback.
    PyTorch(PyDecoder),
}

/// What [`PyDecoder::decode_many`] returns: each chunk's PCM, the S3Gen load time and each chunk's decode time.
pub type Decoded = (Vec<Vec<f32>>, f64, Vec<f64>);

/// The PyTorch S3Gen bridge: a Python with `kittenml`, and `tools/kitten_ref.py`.
pub struct PyDecoder {
    pub python: PathBuf,
    pub script: PathBuf,
    pub threads: usize,
    pub workdir: PathBuf,
}

impl PyDecoder {
    /// A path as the Python process sees it: a Windows Python run from WSL needs a Windows path.
    fn arg(&self, p: &Path) -> String {
        let windows_python = self.python.extension().is_some_and(|e| e.eq_ignore_ascii_case("exe"));
        if windows_python && cfg!(target_os = "linux") {
            if let Ok(o) = Command::new("wslpath").arg("-w").arg(p).output() {
                if o.status.success() {
                    return String::from_utf8_lossy(&o.stdout).trim().to_string();
                }
            }
        }
        p.to_string_lossy().into_owned()
    }

    pub fn decode(&self, reference: &Path, codes: &[u16], seed: u64, tag: &str) -> Result<Vec<f32>, String> {
        let (mut pcm, _) = self.run(reference, &serde_json::to_string(codes).unwrap(), 0, seed, tag)?;
        Ok(pcm.remove(0))
    }

    /// Decode several chunks in one process, so S3Gen loads once. Returns the PCM of each, the load time and each
    /// chunk's decode time, as the bridge measured them.
    pub fn decode_many(&self, reference: &Path, chunks: &[Vec<u16>], seed: u64, tag: &str) -> Result<Decoded, String> {
        let (pcm, report) = self.run(
            reference,
            &serde_json::to_string(chunks).unwrap(),
            chunks.len(),
            seed,
            tag,
        )?;
        let load = report["load_s"].as_f64().unwrap_or(f64::NAN);
        let each = report["decode_s"]
            .as_array()
            .map(|v| v.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();
        Ok((pcm, load, each))
    }

    /// `n == 0`: one chunk, written to `{tag}.pcm.f32`; otherwise `n` chunks, `{tag}.pcm.{i}.f32`.
    fn run(
        &self,
        reference: &Path,
        tokens_json: &str,
        n: usize,
        seed: u64,
        tag: &str,
    ) -> Result<(Vec<Vec<f32>>, serde_json::Value), String> {
        std::fs::create_dir_all(&self.workdir).map_err(|e| e.to_string())?;
        let tokens = self.workdir.join(format!("{tag}.tokens.json"));
        let out = self.workdir.join(format!("{tag}.pcm.f32"));
        std::fs::write(&tokens, tokens_json).map_err(|e| e.to_string())?;
        let status = Command::new(&self.python)
            .env("CUDA_VISIBLE_DEVICES", "")
            .arg("-I")
            .arg(self.arg(&self.script))
            .args(["decode", "--reference"])
            .arg(self.arg(reference))
            .arg("--tokens")
            .arg(self.arg(&tokens))
            .arg("--out")
            .arg(self.arg(&out))
            .args([
                "--seed",
                &(seed % (1 << 31)).to_string(),
                "--threads",
                &self.threads.to_string(),
            ])
            .output()
            .map_err(|e| format!("{}: {e}", self.python.display()))?;
        if !status.status.success() {
            return Err(format!(
                "decoder bridge failed: {}",
                String::from_utf8_lossy(&status.stderr)
            ));
        }
        let stdout = String::from_utf8_lossy(&status.stdout);
        let report: serde_json::Value = stdout
            .lines()
            .rev()
            .find_map(|l| serde_json::from_str(l).ok())
            .unwrap_or(serde_json::Value::Null);
        let paths: Vec<PathBuf> = if n == 0 {
            vec![out]
        } else {
            (0..n)
                .map(|i| self.workdir.join(format!("{tag}.pcm.{i}.f32")))
                .collect()
        };
        let mut pcm = Vec::new();
        for p in paths {
            let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            pcm.push(
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
        Ok((pcm, report))
    }
}

/// The refusal for a language model GGUF converted before the tokenizer was embedded. It names the tool that ships
/// beside ggtts-render and the files it reads, so the fix needs nothing outside the release archive and the model.
pub const NO_TOKENIZER: &str = "this GGUF has no embedded tokenizer (it was converted before ggtts embedded one): re-convert it with ggtts-kitten-convert, which ships beside ggtts-render: `ggtts-kitten-convert --repo DIR --out kitten-lm.gguf`, where DIR is a download of KittenML/kitten-tts-2 (config.json, lm/config.json, lm/model.safetensors, lm/tokenizer.json)";

impl KittenEngine {
    /// Everything from files, for `ggtts-render`: the LM GGUF (with its tokenizer embedded by the converter), the
    /// S3Gen GGUF, and a reference JSON (`ggtts-kitten prepare`). Ternary weights, the ggml decoder.
    pub fn open(lm: &Path, s3gen: &Path, reference: &Path, threads: usize, device: Device) -> Result<Self, String> {
        let be = Backend::new(device, threads)?;
        let g = crate::gguf::Gguf::open(lm).map_err(|e| format!("{} (--model): {e}", lm.display()))?;
        if g.str("general.architecture").ok() != Some(crate::convert::ARCH) {
            return Err(format!(
                "{}: not a Kitten-TTS 2 language model GGUF (--model); ggtts-kitten-convert --repo writes one",
                lm.display()
            ));
        }
        let json =
            g.kv.get("kitten.tokenizer.json")
                .and_then(Value::as_str)
                .ok_or(NO_TOKENIZER)?;
        let tok = Tokenizer::from_json(json)?;
        // Every input is checked for what it is before the slow loads, so a wrong file is named at once. The decoder
        // GGUF is read twice rather than held beside the language model's while that loads.
        let open_s3 = || {
            let s3 = crate::gguf::Gguf::open(s3gen).map_err(|e| format!("{} (--codec): {e}", s3gen.display()))?;
            crate::s3convert::require(&s3, s3gen, "--codec", false)?;
            Ok::<_, String>(s3)
        };
        drop(open_s3()?);
        let reference = Reference::open(reference)?;
        let model = Model::load(&g, crate::model::Weights::Ternary, be.clone())?;
        drop(g);
        let s3 = open_s3()?;
        let decoder = Decoder::Ggml(Box::new(GgmlDecoder {
            flow: Flow::load(&s3, be.clone())?,
            hift: Hift::load(&s3, be.clone())?,
        }));
        KittenEngine::new(model, tok, reference, decoder, threads)
    }
}

/// One generation step, as [`KittenEngine::generate_ids_observed`] shows it.
pub struct Step<'a> {
    /// The logits as the model gave them (with the restricted head, every id outside it is `f32::MIN`).
    pub raw: &'a [f32],
    /// The scores after the sampler's penalties and filters, just before the draw.
    pub filtered: &'a [f32],
    /// The prompt and the tokens drawn so far.
    pub history: &'a [u32],
    pub prompt_len: usize,
    pub ids: sampler::Ids,
    pub settings: Settings,
}

/// Timings of the last `generate`, for the render report.
#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub prefill_s: f64,
    pub step_s: f64,
}

pub struct KittenEngine {
    info: EngineInfo,
    /// The backend string without the head, which each generation appends.
    backend_base: String,
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub tm: TokenMap,
    pub settings: Settings,
    pub threads: usize,
    pub max_new_tokens: usize,
    pub reference: Reference,
    ref_text: Vec<u32>,
    speaker: Vec<f32>,
    pub decoder: Decoder,
    pub last: GenStats,
    seed_for_decode: u64,
    tag: u32,
}

impl KittenEngine {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        reference: Reference,
        decoder: Decoder,
        threads: usize,
    ) -> Result<Self, String> {
        let tm = TokenMap::from_meta(&model.meta)?;
        let ref_text = tokenizer.encode(&reference.transcript)?;
        let speaker = model.project_speaker(&reference.embedding);
        let m = &model.meta;
        let f = |k: &str, d: f32| {
            m.get(&format!("kitten.sampling.{k}"))
                .and_then(Value::as_f32)
                .unwrap_or(d)
        };
        let u = |k: &str, d: u32| {
            m.get(&format!("kitten.sampling.{k}"))
                .and_then(Value::as_u32)
                .unwrap_or(d)
        };
        let settings = Settings {
            temperature: f("temperature", 0.8),
            top_p: f("top_p", 0.8),
            min_p: f("min_p", 0.0),
            top_k: u("top_k", 50) as usize,
            repetition_window: u("repetition_window", 50) as usize,
            ..Settings::default()
        };
        if let Decoder::Ggml(_) = decoder {
            if reference.s3.is_none() {
                return Err(format!(
                    "{}: the reference has no S3Gen conditioning, which the ggml decoder needs; {PREPARE_HINT}",
                    reference.path.display()
                ));
            }
        }
        let dev = match model.be.device {
            Device::Cpu => "ggml-cpu",
            Device::Cuda => "ggml-cuda",
        };
        let lm = match model.weights {
            crate::model::Weights::F32 => format!("{dev} LM (f32)"),
            _ if model.ternary_f16 => format!("{dev} LM (ternary as exact f16)"),
            _ => format!("{dev} LM ({} ternary)", model.layout.name()),
        };
        let dec = match decoder {
            Decoder::Ggml(_) => format!("{dev} S3Gen decoder"),
            Decoder::PyTorch(_) => "PyTorch S3Gen decoder (bridge)".to_string(),
        };
        let backend_base = format!("{lm}; {dec}");
        Ok(KittenEngine {
            info: EngineInfo {
                family: "kitten-tts-2".into(),
                backend: backend_base.clone(),
                parity: "distribution".into(),
            },
            backend_base,
            model,
            tokenizer,
            tm,
            settings,
            threads,
            max_new_tokens: 1000,
            reference,
            ref_text,
            speaker,
            decoder,
            last: GenStats::default(),
            seed_for_decode: 0,
            tag: 0,
        })
    }

    pub fn generate_ids(&mut self, text: &str, seed: u64) -> Result<(Vec<u32>, Stop), String> {
        self.generate_ids_observed(text, seed, None)
    }

    /// The head rows a generation computes: from the first end id to the end of the vocabulary (the end ids, the
    /// audio ids and the special ids after them, 6,571 of 158,241 rows), or `None` for the whole head.
    ///
    /// It is a measured shortcut, not a rule of the model: the sampler draws over the whole vocabulary, as both
    /// references do, so computing only these rows is exact while no other id reaches the top-k set. That held at
    /// every step of a 22-line test set at the default settings (`examples/head_rows_check.rs`), so the subset is
    /// used only inside that envelope: top-k on and at most 50, and top-p below 1. Outside it, or with
    /// `GGTTS_KITTEN_FULL_HEAD=1`, the whole head is computed.
    pub fn head_rows(&self) -> Option<std::ops::Range<usize>> {
        let env = std::env::var("GGTTS_KITTEN_FULL_HEAD").ok();
        sampler::head_rows(&self.settings, &self.tm, self.model.hp.n_vocab, env.as_deref())
    }

    /// [`Self::generate_ids`], also handing `observe` each step: the logits as the model gave them, the scores
    /// after the sampler's filters, and the history so far. For checks that need to see what could have been
    /// sampled; it cannot change the draw.
    pub fn generate_ids_observed(
        &mut self,
        text: &str,
        seed: u64,
        mut observe: Option<&mut dyn FnMut(&Step<'_>)>,
    ) -> Result<(Vec<u32>, Stop), String> {
        self.model.head_rows = self.head_rows();
        // Every report reads the backend string: name the head this generation ran.
        self.info.backend = match &self.model.head_rows {
            Some(r) => format!("{}; output head rows {}..{}", self.backend_base, r.start, r.end),
            None => format!("{}; whole output head", self.backend_base),
        };
        let text_ids = self.tokenizer.encode(text)?;
        let prompt = self
            .tm
            .prompt(&text_ids, &self.ref_text, &self.reference.tokens, &[], true);
        let n = self.model.hp.n_embd;
        let mut embd = Vec::with_capacity(prompt.len() * n);
        self.model.embed_tokens(&prompt, &mut embd);
        embd[..n].copy_from_slice(&self.speaker);
        let mut sess = Session::new(&self.model, prompt.len() + self.max_new_tokens + 1)?;
        let t0 = std::time::Instant::now();
        let mut out = sess
            .forward(&embd, 1, false)
            .map_err(|e| format!("prefill of {} tokens: {e}", prompt.len()))?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let ids = Ids {
            audio_lo: self.tm.audio_base,
            audio_hi: self.tm.audio_base + self.tm.audio_count,
            speech_end: self.tm.speech_end,
            stop: self.tm.stop,
        };
        let mut rng = Rng::new(seed);
        let mut history = prompt.clone();
        let mut generated = Vec::new();
        let mut stop = Stop::Cap;
        let t1 = std::time::Instant::now();
        for step in 0..self.max_new_tokens {
            let mut scores = out.logits;
            let raw = observe.as_ref().map(|_| scores.clone());
            // Step 0 samples the prefill's logits; step k > 0 those of the k-th single-token forward.
            sampler::process(&mut scores, &history, prompt.len(), ids, &self.settings)
                .map_err(|e| format!("generation step {step} (position {}): {e}", history.len() - 1))?;
            if let (Some(f), Some(raw)) = (observe.as_mut(), raw.as_ref()) {
                f(&Step {
                    raw,
                    filtered: &scores,
                    history: &history,
                    prompt_len: prompt.len(),
                    ids,
                    settings: self.settings,
                });
            }
            let tok = sampler::sample(&scores, &mut rng);
            history.push(tok);
            if tok == self.tm.speech_end || tok == self.tm.stop {
                stop = Stop::Eos;
                break;
            }
            generated.push(tok);
            if step + 1 == self.max_new_tokens {
                break;
            }
            let mut e = Vec::with_capacity(n);
            self.model.embed_tokens(&[tok], &mut e);
            out = sess
                .forward(&e, 1, false)
                .map_err(|e| format!("generation step {} (single-token forward): {e}", step + 1))?;
        }
        self.last = GenStats {
            prompt_tokens: prompt.len(),
            generated: generated.len(),
            prefill_s,
            step_s: t1.elapsed().as_secs_f64(),
        };
        Ok((generated, stop))
    }
}

impl Engine for KittenEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn profile(&self) -> &Profile {
        &PROFILE
    }

    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        refuse_instruction(req, &self.info.family)?;
        ggtts_core::steer::refuse_steer(req, &self.info.family)?;
        let (ids, stop) = self.generate_ids(req.text, req.seed).map_err(EngineError)?;
        let (lo, hi) = (self.tm.audio_base, self.tm.audio_base + self.tm.audio_count);
        let frames = ids
            .iter()
            .filter(|&&t| (lo..hi).contains(&t))
            .map(|&t| (t - lo) as u16)
            .collect();
        self.seed_for_decode = req.seed;
        Ok(Generated {
            codes: Codes { codebooks: 1, frames },
            stop,
        })
    }

    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.tag += 1;
        let tag = format!("chunk{}-{}", self.tag, std::process::id());
        let samples = match &self.decoder {
            Decoder::Ggml(d) => {
                let r = self
                    .reference
                    .s3
                    .as_ref()
                    .ok_or_else(|| EngineError("no S3Gen conditioning".into()))?;
                d.decode(&codes.frames, r, self.seed_for_decode).map_err(EngineError)?
            }
            Decoder::PyTorch(p) => p
                .decode(&self.reference.path, &codes.frames, self.seed_for_decode, &tag)
                .map_err(EngineError)?,
        };
        Ok(Pcm {
            samples,
            sample_rate: 24000,
        })
    }
}
