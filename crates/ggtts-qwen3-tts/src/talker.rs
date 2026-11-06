//! The talker and the code predictor: text to codec tokens, one 16-code frame at a time.
//!
//! A port of qwen3-tts.cpp at `16bb5af`, `src/transformer/`, on the path its CLI takes on a CPU backend for a named
//! speaker and no reference audio:
//!
//! * the prompt as `build_prefill_graph` composes it (`transformer_embeddings.cpp:219-702`, the non-ICL branch),
//!   with the text projection of `project_text_tokens` (`:147-217`) called with the same batch sizes;
//! * the talker graphs, `build_prefill_forward_graph` and `build_step_graph` (`transformer_graph_talker.cpp`), with
//!   their inputs filled as `forward_prefill` and `forward_step_internal` fill them (`transformer_runtime.cpp`);
//! * the code predictor's prefill and step graphs (`transformer_graph_code_pred.cpp:165-572`), driven as
//!   `predict_codes_autoregressive` drives them (`transformer_runtime_code_pred.cpp:519-1051`);
//! * the generation loop of `TTSTransformer::generate` (`transformer_generate.cpp:43-441`) and the sampler of
//!   `transformer_sampling.h`: repetition penalty, top-k, top-p, and a Philox draw per sample, one stream shared by
//!   the talker and the predictor.
//!
//! The reference copies the talker's last hidden row into a device tensor the predictor reads; here it goes through
//! the host as the same f32 bits. Its replay graphs and dedicated schedulers are execution strategies over the same
//! graphs and are not reproduced.

use ggtts_core::hooks::Wants;
use std::collections::HashMap;
use std::path::Path;
use std::ptr;

use ggtts_core::engine::{EngineError, Stop};
use ggtts_ggml_sys as g;

use crate::cpu::{Cpu, Device, Meta};
use crate::gguf::{set_name, Gguf, Weights};
use crate::tokenizer::Tokenizer;

type T = *mut g::ggml_tensor;
type Ctx = *mut g::ggml_context;

const MAX_NODES: usize = 16384;
const CODE_PRED_MAX_NODES: usize = 4096;
/// The code predictor's KV cache length (`init_code_pred_kv_cache(16)`).
const CODE_PRED_CTX: i64 = 16;

fn err(msg: impl Into<String>) -> EngineError {
    EngineError(msg.into())
}

/// The sampler settings. The defaults are the reference CLI's (`src/qwen3_tts.h:36-60`): the GGUF's own
/// `generation.*` keys are not read by the reference, so they are not read here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
        }
    }
}

impl Sampling {
    /// Refuses settings this engine will not run. A temperature at or below zero means greedy decoding (`argmax`)
    /// for the talker and the code predictor alike, and on this model greedy decoding does not terminate: it runs
    /// to the frame cap. Neither is a temperature that is not a finite number.
    pub fn check(&self) -> Result<(), EngineError> {
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(err(format!(
                "sampling temperature {} refused: it must be above 0, since greedy decoding does not terminate on \
                 this model",
                self.temperature
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "plant")]
pub use planted::Plant;
#[cfg(not(feature = "plant"))]
use planted::Plant;

mod planted {
    /// A deliberate defect, for showing that the token identity check can fail. Public only with the `plant`
    /// feature, which exists for the identity tests: a build without it cannot plant one.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[cfg_attr(not(feature = "plant"), allow(dead_code))]
    pub enum Plant {
        #[default]
        None,
        /// The talker's rope in normal (adjacent-pair) mode instead of M-RoPE.
        RopeNormal,
        /// The sampler's random stream starts one draw late.
        SamplerOffByOne,
        /// The frame hook of [`Talker::generate_streamed`](super::Talker::generate_streamed) draws once from the
        /// sampler's stream after each frame, so every later token moves. It fires only when a hook is passed: it
        /// exists to show that the hook's identity check can see a hook that is not quiet.
        FrameHookDraw,
    }

    #[cfg(feature = "plant")]
    impl Plant {
        /// `none`, `rope-normal`, `sampler-off-by-one` or `frame-hook-draw`.
        pub fn parse(s: &str) -> Option<Self> {
            Some(match s {
                "" | "none" => Self::None,
                "rope-normal" => Self::RopeNormal,
                "sampler-off-by-one" => Self::SamplerOffByOne,
                "frame-hook-draw" => Self::FrameHookDraw,
                _ => return None,
            })
        }
    }
}

/// The reference CLI's language table (`src/main.cpp:679-689`): `auto` is -1, the rest are codec token ids.
pub fn language_id(lang: &str) -> Option<i32> {
    Some(match lang {
        "auto" => -1,
        "en" | "english" => 2050,
        "ru" | "russian" => 2069,
        "zh" | "chinese" => 2055,
        "ja" | "japanese" => 2058,
        "ko" | "korean" => 2064,
        "de" | "german" => 2053,
        "fr" | "french" => 2061,
        "es" | "spanish" => 2054,
        "it" | "italian" => 2070,
        "pt" | "portuguese" => 2071,
        _ => return None,
    })
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub codec_vocab_size: i32,
    pub n_codebooks: i32,
    pub cp_layers: i32,
    pub cp_vocab_size: i32,
    pub cp_hidden_size: i32,
    pub cp_n_heads: i32,
    pub cp_n_kv_heads: i32,
    pub cp_head_dim: i32,
    pub cp_rms_norm_eps: f32,
    pub cp_rope_theta: f32,
    pub codec_pad_id: i32,
    pub codec_bos_id: i32,
    pub codec_eos_id: i32,
    pub tts_bos_id: i32,
    pub tts_eos_id: i32,
    pub tts_pad_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    /// `None` when the GGUF has no M-RoPE sections; then the talker uses NEOX rope.
    pub mrope_section: Option<[i32; 3]>,
}

struct Layer {
    attn_norm: T,
    attn_q_norm: T,
    attn_k_norm: T,
    attn_q: T,
    attn_k: T,
    attn_v: T,
    attn_output: T,
    ffn_norm: T,
    ffn_gate: T,
    ffn_up: T,
    ffn_down: T,
    /// Q, K and V weights concatenated, created on CUDA only, as the reference creates them in a CUDA build
    /// (`transformer_loader_tensors.cpp:308-370`) and uses them in step graphs on a CUDA backend.
    attn_qkv: T,
}

struct Cache {
    ctx: Ctx,
    buffer: g::ggml_backend_buffer_t,
    k: Vec<T>,
    v: Vec<T>,
    n_ctx: i64,
}

impl Drop for Cache {
    fn drop(&mut self) {
        // SAFETY: owned; freed once.
        unsafe {
            if !self.buffer.is_null() {
                g::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                g::ggml_free(self.ctx);
            }
        }
    }
}

/// Called with each kept frame's 16 codes by [`Talker::generate_streamed`].
pub type FrameHook<'a> = dyn FnMut(&[i32]) + 'a;

/// What one request produced: frame-major codes, why it stopped, and the number of frames generated, thinking
/// frames included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub codes: Vec<i32>,
    pub stop: Stop,
}

/// Called once per talker step, after the codebook-0 token is drawn: the frame, the token, and what the caller's
/// [`Wants`] asked for: the end-of-speech probability under the sampler's distribution, and the logits as the sampler
/// saw them (after suppression and the repetition penalty, before temperature). Nothing unasked is computed.
pub type StepHook<'a> = dyn FnMut(i32, i32, Option<f64>, Option<&[f32]>) + 'a;

/// The end-of-speech probability under the distribution the sampler draws from: the logits after suppression and
/// repetition penalty, scaled by the temperature, cut to the top k (and top p), softmaxed. Computed in f64 on a copy,
/// so it changes nothing the sampler sees and draws nothing from its stream.
fn stop_probability(logits: &[f32], eos: i32, s: Sampling) -> f64 {
    if s.temperature <= 0.0 {
        return if argmax(logits) == eos { 1.0 } else { 0.0 };
    }
    let mut l: Vec<f32> = logits.iter().map(|&x| x * (1.0f32 / s.temperature)).collect();
    if s.top_k > 0 && (s.top_k as usize) < l.len() {
        let mut sorted = l.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[s.top_k as usize - 1];
        for x in l.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    let max = l.iter().fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
    let mut p: Vec<f64> = l.iter().map(|&x| f64::from(x - max).exp()).collect();
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let total: f64 = p.iter().sum();
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0;
        for (j, &i) in order.iter().enumerate() {
            if j > 0 && cum >= f64::from(s.top_p) {
                p[i] = 0.0;
            } else {
                cum += p[i] / total;
            }
        }
    }
    let total: f64 = p.iter().sum();
    if total > 0.0 {
        p[eos as usize] / total
    } else {
        0.0
    }
}

/// The loaded talker, code predictor and text tokenizer.
pub struct Talker {
    pub config: Config,
    pub tokenizer: Tokenizer,
    speakers: HashMap<String, i32>,
    _weights: Weights,
    text_embd: T,
    text_fc1: T,
    text_fc1_b: T,
    text_fc2: T,
    text_fc2_b: T,
    codec_embd: T,
    codec_head: T,
    output_norm: T,
    layers: Vec<Layer>,
    cp_small_to_mtp: T,
    cp_small_to_mtp_b: T,
    cp_layers: Vec<Layer>,
    cp_embd: Vec<T>,
    cp_head: Vec<T>,
    cp_output_norm: T,
    cpu: Cpu,
    meta: Meta,
    cp_meta: Meta,
    cache: Option<Cache>,
    cp_cache: Option<Cache>,
    plant: Plant,
}

/// Typed metadata reads, as `get_u32_any` and `get_f32_any` do (`transformer_loader_metadata.cpp:12-30`).
fn u32_any(gguf: &Gguf, keys: &[&str], default: i32) -> i32 {
    keys.iter()
        .find(|k| gguf.has(k))
        .map(|k| gguf.u32_or(k, default))
        .unwrap_or(default)
}

fn f32_any(gguf: &Gguf, keys: &[&str], default: f32) -> f32 {
    keys.iter()
        .find(|k| gguf.has(k))
        .map(|k| gguf.f32_or(k, default))
        .unwrap_or(default)
}

impl Talker {
    pub fn load(path: &Path, threads: i32) -> Result<Self, EngineError> {
        Self::load_on(path, threads, Device::Cpu)
    }

    /// Loads onto `device`: the weights in its buffer, the caches and graphs on its backend.
    pub fn load_on(path: &Path, threads: i32, device: Device) -> Result<Self, EngineError> {
        Self::load_with(path, threads, device, Plant::None)
    }

    /// [`load_on`](Self::load_on) with a planted defect, for the identity tests' expected failures.
    #[cfg(feature = "plant")]
    pub fn load_planted(path: &Path, threads: i32, device: Device, plant: Plant) -> Result<Self, EngineError> {
        Self::load_with(path, threads, device, plant)
    }

    fn load_with(path: &Path, threads: i32, device: Device, plant: Plant) -> Result<Self, EngineError> {
        let gguf = Gguf::open(path)?;
        let hidden_size = u32_any(
            &gguf,
            &["qwen3-tts.talker.embedding_length", "qwen3-tts.embedding_length"],
            1024,
        );
        let n_layers = u32_any(&gguf, &["qwen3-tts.talker.block_count", "qwen3-tts.block_count"], 28);
        let n_heads = u32_any(
            &gguf,
            &[
                "qwen3-tts.talker.attention.head_count",
                "qwen3-tts.attention.head_count",
            ],
            16,
        );
        let n_kv_heads = u32_any(
            &gguf,
            &[
                "qwen3-tts.talker.attention.head_count_kv",
                "qwen3-tts.attention.head_count_kv",
            ],
            8,
        );
        let head_dim = u32_any(
            &gguf,
            &[
                "qwen3-tts.talker.attention.key_length",
                "qwen3-tts.attention.key_length",
            ],
            128,
        );
        let rms_norm_eps = f32_any(
            &gguf,
            &[
                "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
                "qwen3-tts.attention.layer_norm_rms_epsilon",
            ],
            1e-6,
        );
        let rope_theta = f32_any(
            &gguf,
            &["qwen3-tts.talker.rope.freq_base", "qwen3-tts.rope.freq_base"],
            1e6,
        );
        let mrope_section = ["qwen3-tts.talker.rope.mrope_section", "qwen3-tts.rope.mrope_section"]
            .iter()
            .find_map(|k| gguf.ints(k))
            .filter(|s| s.len() >= 3)
            .map(|s| [s[0], s[1], s[2]]);
        let config = Config {
            hidden_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            codec_vocab_size: u32_any(
                &gguf,
                &[
                    "qwen3-tts.talker.vocab_size",
                    "qwen3-tts.talker.codec_vocab_size",
                    "qwen3-tts.vocab_size",
                ],
                3072,
            ),
            n_codebooks: u32_any(
                &gguf,
                &["qwen3-tts.talker.num_codebooks", "qwen3-tts.num_code_groups"],
                16,
            ),
            cp_layers: u32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.block_count",
                    "qwen3-tts.code_pred.layer_count",
                    "qwen3-tts.code_predictor.layer_count",
                ],
                5,
            ),
            cp_vocab_size: u32_any(
                &gguf,
                &["qwen3-tts.code_pred.vocab_size", "qwen3-tts.code_predictor.vocab_size"],
                2048,
            ),
            cp_hidden_size: u32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.embedding_length",
                    "qwen3-tts.code_predictor.embedding_length",
                ],
                hidden_size,
            ),
            cp_n_heads: u32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.attention.head_count",
                    "qwen3-tts.code_predictor.attention.head_count",
                ],
                n_heads,
            ),
            cp_n_kv_heads: u32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.attention.head_count_kv",
                    "qwen3-tts.code_predictor.attention.head_count_kv",
                ],
                n_kv_heads,
            ),
            cp_head_dim: u32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.attention.key_length",
                    "qwen3-tts.code_predictor.attention.key_length",
                ],
                head_dim,
            ),
            cp_rms_norm_eps: f32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.attention.layer_norm_rms_epsilon",
                    "qwen3-tts.code_predictor.attention.layer_norm_rms_epsilon",
                ],
                rms_norm_eps,
            ),
            cp_rope_theta: f32_any(
                &gguf,
                &[
                    "qwen3-tts.code_pred.rope.freq_base",
                    "qwen3-tts.code_predictor.rope.freq_base",
                ],
                rope_theta,
            ),
            codec_pad_id: u32_any(&gguf, &["qwen3-tts.codec.pad_id"], 2148),
            codec_bos_id: u32_any(&gguf, &["qwen3-tts.codec.bos_id"], 2149),
            codec_eos_id: u32_any(&gguf, &["qwen3-tts.codec.eos_id", "qwen3-tts.codec.eos_token_id"], 2150),
            tts_bos_id: u32_any(
                &gguf,
                &[
                    "qwen3-tts.text.tts_bos_id",
                    "qwen3-tts.tts_bos_token_id",
                    "qwen3-tts.tts.bos_token_id",
                    "qwen3-tts.tts.bos_id",
                ],
                151672,
            ),
            tts_eos_id: u32_any(
                &gguf,
                &[
                    "qwen3-tts.text.tts_eos_id",
                    "qwen3-tts.tts_eos_token_id",
                    "qwen3-tts.tts.eos_token_id",
                    "qwen3-tts.tts.eos_id",
                ],
                151673,
            ),
            tts_pad_id: u32_any(
                &gguf,
                &[
                    "qwen3-tts.text.tts_pad_id",
                    "qwen3-tts.tts_pad_token_id",
                    "qwen3-tts.tts.pad_token_id",
                    "qwen3-tts.tts.pad_id",
                ],
                151671,
            ),
            codec_think_id: u32_any(&gguf, &["qwen3-tts.codec.think_id", "qwen3-tts.codec_think_id"], 2154),
            codec_nothink_id: u32_any(
                &gguf,
                &["qwen3-tts.codec.nothink_id", "qwen3-tts.codec_nothink_id"],
                2155,
            ),
            codec_think_bos_id: u32_any(
                &gguf,
                &["qwen3-tts.codec.think_bos_id", "qwen3-tts.codec_think_bos_id"],
                2156,
            ),
            codec_think_eos_id: u32_any(
                &gguf,
                &["qwen3-tts.codec.think_eos_id", "qwen3-tts.codec_think_eos_id"],
                2157,
            ),
            mrope_section,
        };

        // Speakers: `qwen3-tts.codec.speaker_names` and `speaker_ids`, names lower-cased
        // (`transformer_loader_metadata.cpp:314-350`, `normalize_speaker_name`).
        let mut speakers = HashMap::new();
        if let (Some(names), Some(ids)) = (
            gguf.strings("qwen3-tts.codec.speaker_names"),
            gguf.ints("qwen3-tts.codec.speaker_ids"),
        ) {
            for (n, id) in names.iter().zip(ids) {
                if !n.is_empty() && id >= 0 {
                    speakers.insert(n.to_lowercase(), id);
                }
            }
        }
        let tokenizer = Tokenizer::load(&gguf)?;

        // Every talker.* and code_pred.* tensor, at its file type and shape.
        let names: Vec<String> = (0..gguf.n_tensors())
            .map(|i| gguf.tensor_name(i))
            .filter(|n| n.starts_with("talker.") || n.starts_with("code_pred."))
            .collect();
        // SAFETY: size query; the context is owned by Weights.
        let ctx = unsafe {
            g::ggml_init(g::ggml_init_params {
                mem_size: g::ggml_tensor_overhead() * (2 * names.len() + 8),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            })
        };
        let mut tensors = HashMap::new();
        for name in &names {
            let meta = gguf.meta_tensor(name);
            if meta.is_null() {
                continue;
            }
            // SAFETY: meta is a live metadata tensor; ctx has room.
            let t = unsafe { g::ggml_dup_tensor(ctx, meta) };
            set_name(t, name);
            tensors.insert(name.clone(), t);
        }
        let get = |n: &str| tensors.get(n).copied().unwrap_or(ptr::null_mut());
        let need = |n: &str| {
            tensors
                .get(n)
                .copied()
                .ok_or_else(|| err(format!("talker GGUF lacks {n}")))
        };
        let layer = |p: &str| -> Result<Layer, EngineError> {
            Ok(Layer {
                attn_norm: need(&format!("{p}attn_norm.weight"))?,
                attn_q_norm: get(&format!("{p}attn_q_norm.weight")),
                attn_k_norm: get(&format!("{p}attn_k_norm.weight")),
                attn_q: need(&format!("{p}attn_q.weight"))?,
                attn_k: need(&format!("{p}attn_k.weight"))?,
                attn_v: need(&format!("{p}attn_v.weight"))?,
                attn_output: need(&format!("{p}attn_output.weight"))?,
                ffn_norm: need(&format!("{p}ffn_norm.weight"))?,
                ffn_gate: need(&format!("{p}ffn_gate.weight"))?,
                ffn_up: need(&format!("{p}ffn_up.weight"))?,
                ffn_down: need(&format!("{p}ffn_down.weight"))?,
                attn_qkv: ptr::null_mut(),
            })
        };
        let mut layers = (0..config.n_layers)
            .map(|i| layer(&format!("talker.blk.{i}.")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut cp_layers = (0..config.cp_layers)
            .map(|i| layer(&format!("code_pred.blk.{i}.")))
            .collect::<Result<Vec<_>, _>>()?;
        if device.is_cuda() {
            // Every code predictor layer, and the talker's when its hidden size is over 1024.
            let talker_too = config.hidden_size > 1024;
            for l in cp_layers.iter_mut().chain(layers.iter_mut().filter(|_| talker_too)) {
                // SAFETY: q, k, v are live tensors of ctx, which has room for one more per layer.
                unsafe {
                    let (q, k, v) = (l.attn_q, l.attn_k, l.attn_v);
                    if (*q).type_ != (*k).type_
                        || (*q).type_ != (*v).type_
                        || (*q).ne[0] != (*k).ne[0]
                        || (*q).ne[0] != (*v).ne[0]
                    {
                        return Err(err("incompatible Q/K/V weights for a packed layer"));
                    }
                    let rows = (*q).ne[1] + (*k).ne[1] + (*v).ne[1];
                    l.attn_qkv = g::ggml_new_tensor_2d(ctx, (*q).type_, (*q).ne[0], rows);
                }
            }
        }
        let n_cp = (config.n_codebooks - 1) as usize;
        let cp_embd = (0..n_cp)
            .map(|i| need(&format!("code_pred.codec_embd.{i}.weight")))
            .collect::<Result<Vec<_>, _>>()?;
        let cp_head = (0..n_cp)
            .map(|i| need(&format!("code_pred.lm_head.{i}.weight")))
            .collect::<Result<Vec<_>, _>>()?;
        let small = get("code_pred.small_to_mtp.weight");
        let cp_small_to_mtp = if small.is_null() {
            get("code_pred.mtp_proj.weight")
        } else {
            small
        };
        let small_b = get("code_pred.small_to_mtp.bias");
        let cp_small_to_mtp_b = if small_b.is_null() {
            get("code_pred.mtp_proj.bias")
        } else {
            small_b
        };
        let text_embd = need("talker.text_embd.weight")?;
        let text_fc1 = need("talker.text_proj.fc1.weight")?;
        let text_fc1_b = need("talker.text_proj.fc1.bias")?;
        let text_fc2 = need("talker.text_proj.fc2.weight")?;
        let text_fc2_b = need("talker.text_proj.fc2.bias")?;
        let codec_embd = need("talker.codec_embd.weight")?;
        let codec_head = need("talker.codec_head.weight")?;
        let output_norm = need("talker.output_norm.weight")?;
        let cp_output_norm = need("code_pred.output_norm.weight")?;
        // SAFETY: every tensor in `tensors` was made in ctx above.
        let buffer = unsafe { gguf.load_into(ctx, &tensors, device)? };
        let weights = Weights { ctx, buffer, tensors };
        // The packed tensors get the same bytes, Q then K then V (`transformer_loader_tensors.cpp:436-464`).
        for l in layers.iter().chain(cp_layers.iter()).filter(|l| !l.attn_qkv.is_null()) {
            // SAFETY: all four tensors are allocated in the buffer just filled.
            unsafe {
                let mut off = 0;
                for t in [l.attn_q, l.attn_k, l.attn_v] {
                    let n = g::ggml_nbytes(t);
                    let mut b = vec![0u8; n];
                    g::ggml_backend_tensor_get(t, b.as_mut_ptr().cast(), 0, n);
                    g::ggml_backend_tensor_set(l.attn_qkv, b.as_ptr().cast(), off, n);
                    off += n;
                }
            }
        }
        Ok(Self {
            config,
            tokenizer,
            speakers,
            _weights: weights,
            text_embd,
            text_fc1,
            text_fc1_b,
            text_fc2,
            text_fc2_b,
            codec_embd,
            codec_head,
            output_norm,
            layers,
            cp_small_to_mtp,
            cp_small_to_mtp_b,
            cp_layers,
            cp_embd,
            cp_head,
            cp_output_norm,
            cpu: Cpu::on(device, threads, MAX_NODES)?,
            meta: Meta::new(MAX_NODES),
            cp_meta: Meta::new(CODE_PRED_MAX_NODES),
            cache: None,
            cp_cache: None,
            plant,
        })
    }

    pub fn speaker_id(&self, name: &str) -> Option<i32> {
        self.speakers.get(&name.to_lowercase()).copied()
    }

    /// Text to codec tokens, as the reference CLI renders `-t <text> --speaker <speaker> -l <lang> --seed <seed>
    /// --max-tokens <max_len>`: `Qwen3TTS::synthesize` then `TTSTransformer::generate`.
    pub fn generate(
        &mut self,
        text: &str,
        speaker: &str,
        language_id: i32,
        seed: u64,
        max_len: i32,
        sampling: Sampling,
    ) -> Result<Generation, EngineError> {
        self.generate_streamed(text, speaker, language_id, seed, max_len, sampling, None, None)
    }

    /// [`generate`](Self::generate), with two read-only callbacks.
    /// - `on_step`, with its [`Wants`], is called once per talker step, the step that drew end-of-speech included
    ///   ([`StepHook`]).
    /// - `on_frame` is called once per kept frame, with its 16 codes, as soon as they are final and before the next
    ///   talker step: the frames the returned codes hold, in order.
    ///
    /// Both read values the loop already has, plus what `on_step` asked for, computed on copies; the tokens are the
    /// same with or without them.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streamed(
        &mut self,
        text: &str,
        speaker: &str,
        language_id: i32,
        seed: u64,
        max_len: i32,
        sampling: Sampling,
        mut on_step: Option<(Wants, &mut StepHook<'_>)>,
        mut on_frame: Option<&mut FrameHook<'_>>,
    ) -> Result<Generation, EngineError> {
        sampling.check()?;
        let c = self.config.clone();
        let h = c.hidden_size as usize;
        let spk = self
            .speaker_id(speaker)
            .ok_or_else(|| err(format!("unknown speaker: {speaker}")))?;
        let speaker_embd = self.embedding_rows(self.codec_embd, &[spk])?;
        let text_tokens = self.tokenizer.encode_for_tts(text.as_bytes())?;
        if text_tokens.len() < 4 {
            return Err(err("need at least 4 text tokens"));
        }
        if max_len <= 0 {
            return Ok(Generation {
                codes: Vec::new(),
                stop: Stop::Cap,
            });
        }

        // The prompt (`build_prefill_graph`, non-ICL branch).
        let special = self.project_text(&[c.tts_bos_id, c.tts_eos_id, c.tts_pad_id])?;
        let tts_bos = special[0..h].to_vec();
        let tts_eos = special[h..2 * h].to_vec();
        let tts_pad = special[2 * h..3 * h].to_vec();
        let role = self.project_text(&text_tokens[..3])?;
        let codec_prefill: Vec<i32> = if language_id < 0 {
            vec![c.codec_nothink_id, c.codec_think_bos_id, c.codec_think_eos_id]
        } else {
            vec![
                c.codec_think_id,
                c.codec_think_bos_id,
                language_id,
                c.codec_think_eos_id,
            ]
        };
        let codec_prefill_embed = self.embedding_rows(self.codec_embd, &codec_prefill)?;
        let codec_tail_embed = self.embedding_rows(self.codec_embd, &[c.codec_pad_id, c.codec_bos_id])?;
        let mut codec_input = codec_prefill_embed;
        codec_input.extend_from_slice(&speaker_embd);
        codec_input.extend_from_slice(&codec_tail_embed);
        let codec_input_len = codec_input.len() / h;
        let overlay_len = codec_input_len - 1;
        let mut codec_plus_overlay = vec![0f32; overlay_len * h];
        for t in 0..overlay_len {
            let overlay = if t == overlay_len - 1 { &tts_bos } else { &tts_pad };
            for j in 0..h {
                codec_plus_overlay[t * h + j] = overlay[j] + codec_input[t * h + j];
            }
        }
        let body_count = text_tokens.len() as i64 - 3 - 5;
        if body_count <= 0 {
            return Err(err("the text has no body tokens"));
        }
        let body = self.project_text(&text_tokens[3..3 + body_count as usize])?;
        let codec_bos = &codec_tail_embed[h..2 * h];
        let mut prefill = role;
        prefill.extend_from_slice(&codec_plus_overlay);
        prefill.extend((0..h).map(|j| body[j] + codec_bos[j]));
        let prefill_len = (prefill.len() / h) as i64;
        let mut trailing: Vec<f32> = body[h..].to_vec();
        trailing.extend_from_slice(&tts_eos);
        let trailing_len = trailing.len() / h;

        // Caches: the talker's sized to the request (`generate:140-152`), the predictor's 16 rows.
        let required = prefill_len + max_len as i64 + 8;
        let reuse = self
            .cache
            .as_ref()
            .is_some_and(|k| k.n_ctx >= required && k.n_ctx <= (required * 2).max(512));
        if !reuse {
            self.cache = None;
            self.cache = Some(self.new_cache(
                g::GGML_TYPE_F16,
                [c.head_dim as i64, c.n_kv_heads as i64, required],
                c.n_layers as usize,
                required,
            )?);
        }
        if self.cp_cache.is_none() {
            self.cp_cache = Some(self.new_cache(
                g::GGML_TYPE_F16,
                [c.cp_head_dim as i64, CODE_PRED_CTX, c.cp_n_kv_heads as i64],
                c.cp_layers as usize,
                CODE_PRED_CTX,
            )?);
        }

        let (mut logits, mut hidden) = self.talker_prefill(&prefill, prefill_len)?;

        let mut out = Vec::with_capacity(max_len as usize * c.n_codebooks as usize);
        let mut n_past = prefill_len;
        let mut history: Vec<i32> = Vec::new();
        let mut rng = Philox {
            seed: seed as i64,
            subseq: if self.plant == Plant::SamplerOffByOne { 1 } else { 0 },
        };
        let suppress_start = c.cp_vocab_size.min(c.codec_vocab_size);
        let mut frame_codes = vec![0i32; c.n_codebooks as usize];
        let mut stop = Stop::Cap;
        for frame in 0..max_len {
            for i in suppress_start..c.codec_vocab_size {
                if i != c.codec_eos_id {
                    logits[i as usize] = f32::NEG_INFINITY;
                }
            }
            repetition_penalty(&mut logits, &history, sampling.repetition_penalty);
            let wants = on_step.as_ref().map_or(Wants::NONE, |(w, _)| *w);
            let p_stop = wants
                .p_stop
                .then(|| stop_probability(&logits, c.codec_eos_id, sampling));
            let seen = wants.logits.then(|| logits.clone());
            let next = if sampling.temperature <= 0.0 {
                argmax(&logits)
            } else {
                sample_top_k_p(&mut logits, sampling, &mut rng)
            };
            if let Some((_, f)) = on_step.as_mut() {
                f(frame, next, p_stop, seen.as_deref());
            }
            if next == c.codec_eos_id {
                stop = Stop::Eos;
                break;
            }
            let thinking = next >= c.codec_think_id && next <= c.codec_think_eos_id;
            frame_codes[0] = next;
            history.push(next);
            let rest = self.predict_codes(&hidden, next, sampling, &mut rng)?;
            frame_codes[1..].copy_from_slice(&rest);
            if !thinking {
                out.extend_from_slice(&frame_codes);
                if let Some(f) = on_frame.as_mut() {
                    f(&frame_codes);
                    if self.plant == Plant::FrameHookDraw {
                        rng.next_uniform();
                    }
                }
            }
            if frame + 1 >= max_len {
                break;
            }
            let row = if (frame as usize) < trailing_len {
                &trailing[frame as usize * h..(frame as usize + 1) * h]
            } else {
                &tts_pad[..]
            };
            let row = row.to_vec();
            let (l, hid) = self.talker_step(&frame_codes, &row, n_past)?;
            logits = l;
            hidden = hid;
            n_past += 1;
        }
        Ok(Generation { codes: out, stop })
    }

    fn new_cache(&self, ty: g::ggml_type, ne: [i64; 3], n_layers: usize, n_ctx: i64) -> Result<Cache, EngineError> {
        // SAFETY: a fresh context sized for the tensors made in it; the buffer is on the compute backend.
        unsafe {
            let ctx = g::ggml_init(g::ggml_init_params {
                mem_size: g::ggml_tensor_overhead() * (2 * n_layers + 2),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            });
            let mut k = Vec::new();
            let mut v = Vec::new();
            for _ in 0..n_layers {
                k.push(g::ggml_new_tensor_3d(ctx, ty, ne[0], ne[1], ne[2]));
                v.push(g::ggml_new_tensor_3d(ctx, ty, ne[0], ne[1], ne[2]));
            }
            let buffer = g::ggml_backend_alloc_ctx_tensors(ctx, self.cpu.backend);
            if buffer.is_null() {
                g::ggml_free(ctx);
                return Err(err("failed to allocate a KV cache"));
            }
            // Masked rows are never read, so this changes no result; it keeps the buffer's contents defined.
            g::ggml_backend_buffer_clear(buffer, 0);
            Ok(Cache {
                ctx,
                buffer,
                k,
                v,
                n_ctx,
            })
        }
    }

    /// Runs a graph built in `ctx` on the scheduler: allocate, `fill` the inputs, compute, `read` the outputs, reset.
    unsafe fn run<R>(
        &self,
        ctx: Ctx,
        gf: *mut g::ggml_cgraph,
        fill: impl FnOnce(),
        read: impl FnOnce() -> R,
    ) -> Result<R, EngineError> {
        let sched = self.cpu.sched;
        if !g::ggml_backend_sched_alloc_graph(sched, gf) {
            g::ggml_free(ctx);
            return Err(err("failed to allocate a graph"));
        }
        fill();
        if g::ggml_backend_sched_graph_compute(sched, gf) != g::GGML_STATUS_SUCCESS {
            g::ggml_backend_sched_reset(sched);
            g::ggml_free(ctx);
            return Err(err("graph compute failed"));
        }
        let r = read();
        g::ggml_backend_sched_reset(sched);
        g::ggml_free(ctx);
        Ok(r)
    }

    /// `lookup_embedding_rows` (`transformer_embeddings.cpp:11-97`): F16 and F32 tables read on the host when there
    /// are at most 32 rows; otherwise, and for quantized tables, `get_rows` and a cast to F32 in a graph.
    fn embedding_rows(&mut self, table: T, ids: &[i32]) -> Result<Vec<f32>, EngineError> {
        // SAFETY: table is a live weight in the CPU buffer; graph inputs are set after allocation.
        unsafe {
            let dim = (*table).ne[0] as usize;
            let vocab = (*table).ne[1];
            let ty = (*table).type_;
            if ids.iter().any(|&i| i < 0 || i as i64 >= vocab) {
                return Err(err("embedding token id out of range"));
            }
            if ids.len() <= 32 && (ty == g::GGML_TYPE_F16 || ty == g::GGML_TYPE_F32) {
                let mut out = Vec::with_capacity(dim * ids.len());
                for &id in ids {
                    let off = id as usize * (*table).nb[1];
                    if ty == g::GGML_TYPE_F32 {
                        let mut row = vec![0f32; dim];
                        g::ggml_backend_tensor_get(table, row.as_mut_ptr().cast(), off, dim * 4);
                        out.extend_from_slice(&row);
                    } else {
                        let mut row = vec![0u16; dim];
                        g::ggml_backend_tensor_get(table, row.as_mut_ptr().cast(), off, dim * 2);
                        out.extend(row.iter().map(|&x| g::ggml_fp16_to_fp32(x)));
                    }
                }
                return Ok(out);
            }
            let ctx = self.meta.context();
            let gf = g::ggml_new_graph_custom(ctx, MAX_NODES, false);
            let inp = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, ids.len() as i64);
            g::ggml_set_input(inp);
            let rows = g::ggml_cast(ctx, g::ggml_get_rows(ctx, table, inp), g::GGML_TYPE_F32);
            g::ggml_set_output(rows);
            g::ggml_build_forward_expand(gf, rows);
            self.run(
                ctx,
                gf,
                || g::ggml_backend_tensor_set(inp, ids.as_ptr().cast(), 0, ids.len() * 4),
                || {
                    let mut out = vec![0f32; dim * ids.len()];
                    g::ggml_backend_tensor_get(rows, out.as_mut_ptr().cast(), 0, out.len() * 4);
                    out
                },
            )
        }
    }

    /// `project_text_tokens` (`transformer_embeddings.cpp:147-217`): text embedding, fc1, SiLU, fc2.
    fn project_text(&mut self, ids: &[i32]) -> Result<Vec<f32>, EngineError> {
        let h = self.config.hidden_size as usize;
        // SAFETY: as in embedding_rows.
        unsafe {
            let ctx = self.meta.context();
            let gf = g::ggml_new_graph_custom(ctx, MAX_NODES, false);
            let inp = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, ids.len() as i64);
            g::ggml_set_input(inp);
            let mut cur = g::ggml_get_rows(ctx, self.text_embd, inp);
            cur = g::ggml_mul_mat(ctx, self.text_fc1, cur);
            cur = g::ggml_add(ctx, cur, self.text_fc1_b);
            cur = g::ggml_silu(ctx, cur);
            cur = g::ggml_mul_mat(ctx, self.text_fc2, cur);
            cur = g::ggml_add(ctx, cur, self.text_fc2_b);
            g::ggml_set_output(cur);
            g::ggml_build_forward_expand(gf, cur);
            self.run(
                ctx,
                gf,
                || g::ggml_backend_tensor_set(inp, ids.as_ptr().cast(), 0, ids.len() * 4),
                || {
                    let mut out = vec![0f32; h * ids.len()];
                    g::ggml_backend_tensor_get(cur, out.as_mut_ptr().cast(), 0, out.len() * 4);
                    out
                },
            )
        }
    }

    /// One talker layer stack over `cur` (`transformer_graph_talker.cpp:84-173` and `:271-389`), shared by prefill
    /// and step. Returns the final normed hidden states.
    #[allow(clippy::too_many_arguments)]
    unsafe fn talker_layers(
        &self,
        ctx: Ctx,
        gf: *mut g::ggml_cgraph,
        inp: T,
        n_tokens: i64,
        n_kv_pad: i64,
        pos: T,
        mrope_pos: T,
        mask: T,
        prefill: bool,
    ) -> T {
        let c = &self.config;
        let cache = self.cache.as_ref().unwrap();
        let (hd, nh, nkv) = (c.head_dim as i64, c.n_heads as i64, c.n_kv_heads as i64);
        let eps = c.rms_norm_eps;
        let mut sections = c.mrope_section.map(|s| [s[0], s[1], s[2], 0]).unwrap_or([0; 4]);
        let mut inp_l = inp;
        for (il, l) in self.layers.iter().enumerate() {
            let mut cur = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, inp_l, eps), l.attn_norm);
            let packed = !prefill && self.cpu.device.is_cuda();
            let (mut q, mut k, v) = qkv(ctx, l, cur, hd, nh, nkv, n_tokens, packed);
            if !l.attn_q_norm.is_null() {
                q = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, q, eps), l.attn_q_norm);
            }
            if !l.attn_k_norm.is_null() {
                k = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, k, eps), l.attn_k_norm);
            }
            if self.plant == Plant::RopeNormal {
                q = g::ggml_rope_ext(
                    ctx,
                    q,
                    pos,
                    ptr::null_mut(),
                    hd as i32,
                    0,
                    0,
                    c.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
                k = g::ggml_rope_ext(
                    ctx,
                    k,
                    pos,
                    ptr::null_mut(),
                    hd as i32,
                    0,
                    0,
                    c.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
            } else if c.mrope_section.is_some() {
                let mode = g::GGML_ROPE_TYPE_MROPE as i32;
                q = g::ggml_rope_multi(
                    ctx,
                    q,
                    mrope_pos,
                    ptr::null_mut(),
                    hd as i32,
                    sections.as_mut_ptr(),
                    mode,
                    0,
                    c.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
                k = g::ggml_rope_multi(
                    ctx,
                    k,
                    mrope_pos,
                    ptr::null_mut(),
                    hd as i32,
                    sections.as_mut_ptr(),
                    mode,
                    0,
                    c.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
            } else {
                let mode = g::GGML_ROPE_TYPE_NEOX as i32;
                q = g::ggml_rope_ext(
                    ctx,
                    q,
                    pos,
                    ptr::null_mut(),
                    hd as i32,
                    mode,
                    0,
                    c.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
                k = g::ggml_rope_ext(
                    ctx,
                    k,
                    pos,
                    ptr::null_mut(),
                    hd as i32,
                    mode,
                    0,
                    c.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
            }
            let kc = cache.k[il];
            let vc = cache.v[il];
            let kc2 = g::ggml_view_2d(ctx, kc, hd * nkv, cache.n_ctx, (*kc).nb[2], 0);
            let vc2 = g::ggml_view_2d(ctx, vc, hd * nkv, cache.n_ctx, (*vc).nb[2], 0);
            let k2 = g::ggml_view_2d(ctx, k, hd * nkv, n_tokens, (*k).nb[2], 0);
            let v2 = g::ggml_view_2d(ctx, v, hd * nkv, n_tokens, (*v).nb[2], 0);
            g::ggml_build_forward_expand(gf, g::ggml_set_rows(ctx, kc2, k2, pos));
            g::ggml_build_forward_expand(gf, g::ggml_set_rows(ctx, vc2, v2, pos));
            let kk = g::ggml_view_3d(ctx, kc, hd, nkv, n_kv_pad, (*kc).nb[1], (*kc).nb[2], 0);
            let vv = g::ggml_view_3d(ctx, vc, hd, nkv, n_kv_pad, (*vc).nb[1], (*vc).nb[2], 0);
            let qq = g::ggml_permute(ctx, q, 0, 2, 1, 3);
            let kk = g::ggml_permute(ctx, kk, 0, 2, 1, 3);
            let vv = g::ggml_permute(ctx, vv, 0, 2, 1, 3);
            let fa = g::ggml_flash_attn_ext(ctx, qq, kk, vv, mask, 1.0 / (hd as f32).sqrt(), 0.0, 0.0);
            if prefill {
                g::ggml_flash_attn_ext_set_prec(fa, g::GGML_PREC_F32);
            }
            cur = g::ggml_cont_2d(ctx, fa, nh * hd, n_tokens);
            cur = g::ggml_mul_mat(ctx, l.attn_output, cur);
            cur = g::ggml_add(ctx, cur, inp_l);
            let inp_ff = cur;
            cur = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, inp_ff, eps), l.ffn_norm);
            let gate = g::ggml_mul_mat(ctx, l.ffn_gate, cur);
            let up = g::ggml_mul_mat(ctx, l.ffn_up, cur);
            cur = g::ggml_swiglu_split(ctx, gate, up);
            cur = ffn_down(ctx, l.ffn_down, cur);
            inp_l = g::ggml_add(ctx, cur, inp_ff);
        }
        g::ggml_mul(ctx, g::ggml_rms_norm(ctx, inp_l, eps), self.output_norm)
    }

    /// Positions, M-RoPE positions and the F16 causal mask as `forward_prefill` and `forward_step_internal` fill
    /// them.
    unsafe fn set_positions(&self, pos: T, mrope_pos: T, mask: T, n_past: i64, n_tokens: i64, n_kv_pad: i64) {
        let p: Vec<i32> = (0..n_tokens).map(|i| (n_past + i) as i32).collect();
        g::ggml_backend_tensor_set(pos, p.as_ptr().cast(), 0, p.len() * 4);
        if !mrope_pos.is_null() {
            let mut m = vec![0i32; 4 * n_tokens as usize];
            for (i, &v) in p.iter().enumerate() {
                m[i] = v;
                m[i + n_tokens as usize] = v;
                m[i + 2 * n_tokens as usize] = v;
            }
            g::ggml_backend_tensor_set(mrope_pos, m.as_ptr().cast(), 0, m.len() * 4);
        }
        let zero = g::ggml_fp32_to_fp16(0.0);
        let neg = g::ggml_fp32_to_fp16(f32::NEG_INFINITY);
        let mut mk = vec![neg; (n_tokens * n_kv_pad) as usize];
        for q in 0..n_tokens {
            let last = (n_past + q).min(n_kv_pad - 1);
            for k in 0..=last {
                mk[(q * n_kv_pad + k) as usize] = zero;
            }
        }
        g::ggml_backend_tensor_set(mask, mk.as_ptr().cast(), 0, mk.len() * 2);
    }

    fn n_kv_pad(&self, n_past: i64, n_tokens: i64) -> i64 {
        let n_ctx = self.cache.as_ref().unwrap().n_ctx;
        n_ctx.min(ggtts_ggml_sys::pad_to(n_past + n_tokens, 256))
    }

    unsafe fn talker_inputs(&self, ctx: Ctx, n_tokens: i64, n_kv_pad: i64) -> (T, T, T) {
        let pos = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, n_tokens);
        g::ggml_set_input(pos);
        // An input the graph does not use must not exist: setting it would abort inside ggml.
        let mrope_pos = if self.config.mrope_section.is_some() && self.plant != Plant::RopeNormal {
            let t = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, 4 * n_tokens);
            g::ggml_set_input(t);
            t
        } else {
            ptr::null_mut()
        };
        let mask = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F16, n_kv_pad, n_tokens);
        g::ggml_set_input(mask);
        (pos, mrope_pos, mask)
    }

    /// `forward_prefill` from position 0: returns the last row's codec logits and hidden state.
    fn talker_prefill(&mut self, embd: &[f32], n_tokens: i64) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        let h = self.config.hidden_size as i64;
        let vocab = self.config.codec_vocab_size as usize;
        let n_kv_pad = self.n_kv_pad(0, n_tokens);
        // SAFETY: graph tensors are live in ctx until run() frees it; inputs are set after allocation.
        unsafe {
            let ctx = self.meta.context();
            let gf = g::ggml_new_graph_custom(ctx, MAX_NODES, false);
            let inp = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, h, n_tokens);
            g::ggml_set_input(inp);
            let (pos, mrope_pos, mask) = self.talker_inputs(ctx, n_tokens, n_kv_pad);
            let hidden = self.talker_layers(ctx, gf, inp, n_tokens, n_kv_pad, pos, mrope_pos, mask, true);
            g::ggml_set_output(hidden);
            let logits = g::ggml_mul_mat(ctx, self.codec_head, hidden);
            g::ggml_set_output(logits);
            g::ggml_build_forward_expand(gf, logits);
            self.run(
                ctx,
                gf,
                || {
                    g::ggml_backend_tensor_set(inp, embd.as_ptr().cast(), 0, embd.len() * 4);
                    self.set_positions(pos, mrope_pos, mask, 0, n_tokens, n_kv_pad);
                },
                || {
                    let mut l = vec![0f32; vocab];
                    g::ggml_backend_tensor_get(
                        logits,
                        l.as_mut_ptr().cast(),
                        (n_tokens as usize - 1) * vocab * 4,
                        vocab * 4,
                    );
                    let mut hid = vec![0f32; h as usize];
                    g::ggml_backend_tensor_get(
                        hidden,
                        hid.as_mut_ptr().cast(),
                        (n_tokens - 1) as usize * h as usize * 4,
                        h as usize * 4,
                    );
                    (l, hid)
                },
            )
        }
    }

    /// `forward_step_internal` with frame codes and an overlay row (`build_step_graph`, `use_frame_codes`).
    fn talker_step(
        &mut self,
        codes: &[i32],
        overlay: &[f32],
        n_past: i64,
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        let h = self.config.hidden_size as i64;
        let vocab = self.config.codec_vocab_size as usize;
        let ncb = self.config.n_codebooks as i64;
        let n_kv_pad = self.n_kv_pad(n_past, 1);
        if n_past + 1 > self.cache.as_ref().unwrap().n_ctx {
            return Err(err("context length exceeded"));
        }
        // SAFETY: as in talker_prefill.
        unsafe {
            let ctx = self.meta.context();
            let gf = g::ggml_new_graph_custom(ctx, MAX_NODES, false);
            let inp_codes = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, ncb);
            g::ggml_set_input(inp_codes);
            let inp_overlay = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, h, 1);
            g::ggml_set_input(inp_overlay);
            let idx = g::ggml_view_1d(ctx, inp_codes, 1, 0);
            let mut cur = g::ggml_reshape_2d(ctx, g::ggml_get_rows(ctx, self.codec_embd, idx), h, 1);
            for cb in 1..ncb as usize {
                let idx = g::ggml_view_1d(ctx, inp_codes, 1, cb * 4);
                let e = g::ggml_reshape_2d(ctx, g::ggml_get_rows(ctx, self.cp_embd[cb - 1], idx), h, 1);
                cur = g::ggml_add(ctx, cur, e);
            }
            cur = g::ggml_add(ctx, cur, inp_overlay);
            let (pos, mrope_pos, mask) = self.talker_inputs(ctx, 1, n_kv_pad);
            let hidden = self.talker_layers(ctx, gf, cur, 1, n_kv_pad, pos, mrope_pos, mask, false);
            g::ggml_set_output(hidden);
            let logits = g::ggml_mul_mat(ctx, self.codec_head, hidden);
            g::ggml_set_output(logits);
            g::ggml_build_forward_expand(gf, logits);
            self.run(
                ctx,
                gf,
                || {
                    g::ggml_backend_tensor_set(inp_codes, codes.as_ptr().cast(), 0, codes.len() * 4);
                    g::ggml_backend_tensor_set(inp_overlay, overlay.as_ptr().cast(), 0, overlay.len() * 4);
                    self.set_positions(pos, mrope_pos, mask, n_past, 1, n_kv_pad);
                },
                || {
                    let mut l = vec![0f32; vocab];
                    g::ggml_backend_tensor_get(logits, l.as_mut_ptr().cast(), 0, vocab * 4);
                    let mut hid = vec![0f32; h as usize];
                    g::ggml_backend_tensor_get(hidden, hid.as_mut_ptr().cast(), 0, h as usize * 4);
                    (l, hid)
                },
            )
        }
    }

    /// The code predictor's layer stack (`transformer_graph_code_pred.cpp:242-321`, `:439-549`).
    unsafe fn cp_layers(&self, ctx: Ctx, gf: *mut g::ggml_cgraph, inp: T, n_tokens: i64, pos: T, mask: T) -> T {
        let c = &self.config;
        let cache = self.cp_cache.as_ref().unwrap();
        let (hd, nh, nkv) = (c.cp_head_dim as i64, c.cp_n_heads as i64, c.cp_n_kv_heads as i64);
        let eps = c.cp_rms_norm_eps;
        let neox = g::GGML_ROPE_TYPE_NEOX as i32;
        let mut inp_l = inp;
        for (il, l) in self.cp_layers.iter().enumerate() {
            let mut cur = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, inp_l, eps), l.attn_norm);
            let packed = n_tokens == 1 && self.cpu.device.is_cuda();
            let (mut q, mut k, v) = qkv(ctx, l, cur, hd, nh, nkv, n_tokens, packed);
            if !l.attn_q_norm.is_null() {
                q = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, q, eps), l.attn_q_norm);
            }
            if !l.attn_k_norm.is_null() {
                k = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, k, eps), l.attn_k_norm);
            }
            q = g::ggml_rope_ext(
                ctx,
                q,
                pos,
                ptr::null_mut(),
                hd as i32,
                neox,
                0,
                c.cp_rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
            k = g::ggml_rope_ext(
                ctx,
                k,
                pos,
                ptr::null_mut(),
                hd as i32,
                neox,
                0,
                c.cp_rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
            let kc = cache.k[il];
            let vc = cache.v[il];
            let kcache = g::ggml_cont(ctx, g::ggml_permute(ctx, k, 0, 2, 1, 3));
            let vcache = g::ggml_cont(ctx, g::ggml_permute(ctx, v, 0, 2, 1, 3));
            g::ggml_build_forward_expand(gf, g::ggml_set_rows(ctx, kc, kcache, pos));
            g::ggml_build_forward_expand(gf, g::ggml_set_rows(ctx, vc, vcache, pos));
            let kfull = g::ggml_view_3d(ctx, kc, hd, cache.n_ctx, nkv, (*kc).nb[1], (*kc).nb[2], 0);
            let vfull = g::ggml_view_3d(ctx, vc, hd, cache.n_ctx, nkv, (*vc).nb[1], (*vc).nb[2], 0);
            let qq = g::ggml_permute(ctx, q, 0, 2, 1, 3);
            let fa = g::ggml_flash_attn_ext(ctx, qq, kfull, vfull, mask, 1.0 / (hd as f32).sqrt(), 0.0, 0.0);
            g::ggml_flash_attn_ext_set_prec(fa, g::GGML_PREC_F32);
            cur = g::ggml_reshape_2d(ctx, fa, nh * hd, n_tokens);
            cur = g::ggml_mul_mat(ctx, l.attn_output, cur);
            cur = g::ggml_add(ctx, cur, inp_l);
            let inp_ff = cur;
            cur = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, inp_ff, eps), l.ffn_norm);
            let gate = g::ggml_mul_mat(ctx, l.ffn_gate, cur);
            let up = g::ggml_mul_mat(ctx, l.ffn_up, cur);
            cur = g::ggml_swiglu_split(ctx, gate, up);
            cur = ffn_down(ctx, l.ffn_down, cur);
            inp_l = g::ggml_add(ctx, cur, inp_ff);
        }
        g::ggml_mul(ctx, g::ggml_rms_norm(ctx, inp_l, eps), self.cp_output_norm)
    }

    unsafe fn small_to_mtp(&self, ctx: Ctx, cur: T) -> T {
        if self.cp_small_to_mtp.is_null() {
            return cur;
        }
        let mut cur = g::ggml_mul_mat(ctx, self.cp_small_to_mtp, cur);
        if !self.cp_small_to_mtp_b.is_null() {
            let b = g::ggml_reshape_2d(ctx, self.cp_small_to_mtp_b, self.config.cp_hidden_size as i64, 1);
            cur = g::ggml_add(ctx, cur, g::ggml_repeat(ctx, b, cur));
        }
        cur
    }

    /// `predict_codes_autoregressive`: codebooks 1 to 15 for one frame, drawing from the shared stream.
    fn predict_codes(
        &mut self,
        hidden: &[f32],
        cb0: i32,
        sampling: Sampling,
        rng: &mut Philox,
    ) -> Result<Vec<i32>, EngineError> {
        let c = self.config.clone();
        let h = c.hidden_size as i64;
        let vocab = c.cp_vocab_size as usize;
        let n_ctx = CODE_PRED_CTX;
        let zero = unsafe { g::ggml_fp32_to_fp16(0.0) };
        let neg = unsafe { g::ggml_fp32_to_fp16(f32::NEG_INFINITY) };
        let mut out = vec![0i32; (c.n_codebooks - 1) as usize];

        // Prefill: [hidden; codec_embd[cb0]] at positions 0 and 1.
        // SAFETY: as in talker_prefill.
        let mut logits = unsafe {
            let ctx = self.cp_meta.context();
            let gf = g::ggml_new_graph_custom(ctx, CODE_PRED_MAX_NODES, false);
            let inp_hidden = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_F32, h);
            g::ggml_set_input(inp_hidden);
            let inp_cb0 = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, 1);
            g::ggml_set_input(inp_cb0);
            let pos = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, 2);
            g::ggml_set_input(pos);
            let mask = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F16, n_ctx, 2);
            g::ggml_set_input(mask);
            let hid2 = g::ggml_reshape_2d(ctx, inp_hidden, h, 1);
            let cb0_2d = g::ggml_reshape_2d(ctx, g::ggml_get_rows(ctx, self.codec_embd, inp_cb0), h, 1);
            let cur = g::ggml_concat(ctx, hid2, cb0_2d, 1);
            let cur = self.small_to_mtp(ctx, cur);
            let cur = self.cp_layers(ctx, gf, cur, 2, pos, mask);
            let hs = c.cp_hidden_size as i64;
            let last = g::ggml_view_2d(ctx, cur, hs, 1, (*cur).nb[1], hs as usize * 4);
            let logits = g::ggml_mul_mat(ctx, self.cp_head[0], last);
            g::ggml_set_output(logits);
            g::ggml_build_forward_expand(gf, logits);
            let mut m = vec![neg; 2 * n_ctx as usize];
            m[0] = zero;
            m[n_ctx as usize] = zero;
            m[n_ctx as usize + 1] = zero;
            self.run(
                ctx,
                gf,
                || {
                    g::ggml_backend_tensor_set(inp_hidden, hidden.as_ptr().cast(), 0, hidden.len() * 4);
                    g::ggml_backend_tensor_set(inp_cb0, (&cb0 as *const i32).cast(), 0, 4);
                    let p = [0i32, 1];
                    g::ggml_backend_tensor_set(pos, p.as_ptr().cast(), 0, 8);
                    g::ggml_backend_tensor_set(mask, m.as_ptr().cast(), 0, m.len() * 2);
                },
                || {
                    let mut l = vec![0f32; vocab];
                    g::ggml_backend_tensor_get(logits, l.as_mut_ptr().cast(), 0, vocab * 4);
                    l
                },
            )?
        };
        out[0] = sample_top_k_p(
            &mut logits,
            Sampling {
                repetition_penalty: 1.0,
                ..sampling
            },
            rng,
        );

        for step in 1..out.len() {
            let n_past = step as i32 + 1;
            // SAFETY: as in talker_prefill.
            let mut logits = unsafe {
                let ctx = self.cp_meta.context();
                let gf = g::ggml_new_graph_custom(ctx, CODE_PRED_MAX_NODES, false);
                let inp_code = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, 1);
                g::ggml_set_input(inp_code);
                let pos = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, 1);
                g::ggml_set_input(pos);
                let mask = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F16, n_ctx, 1);
                g::ggml_set_input(mask);
                let cur = g::ggml_reshape_2d(ctx, g::ggml_get_rows(ctx, self.cp_embd[step - 1], inp_code), h, 1);
                let cur = self.small_to_mtp(ctx, cur);
                let cur = self.cp_layers(ctx, gf, cur, 1, pos, mask);
                let logits = g::ggml_mul_mat(ctx, self.cp_head[step], cur);
                g::ggml_set_output(logits);
                g::ggml_build_forward_expand(gf, logits);
                let mut m = vec![neg; n_ctx as usize];
                let max_unmasked = (n_ctx as i32 - 1).min(step as i32 + 1);
                for x in m.iter_mut().take(max_unmasked as usize + 1) {
                    *x = zero;
                }
                let prev = out[step - 1];
                self.run(
                    ctx,
                    gf,
                    || {
                        g::ggml_backend_tensor_set(inp_code, (&prev as *const i32).cast(), 0, 4);
                        g::ggml_backend_tensor_set(pos, (&n_past as *const i32).cast(), 0, 4);
                        g::ggml_backend_tensor_set(mask, m.as_ptr().cast(), 0, m.len() * 2);
                    },
                    || {
                        let mut l = vec![0f32; vocab];
                        g::ggml_backend_tensor_get(logits, l.as_mut_ptr().cast(), 0, vocab * 4);
                        l
                    },
                )?
            };
            out[step] = sample_top_k_p(
                &mut logits,
                Sampling {
                    repetition_penalty: 1.0,
                    ..sampling
                },
                rng,
            );
        }
        Ok(out)
    }
}

/// Q, K and V for one layer: three projections, or on a CUDA step graph one packed projection viewed three ways
/// (`transformer_graph_talker.cpp:280-299`, `transformer_graph_code_pred.cpp:448-467`).
#[allow(clippy::too_many_arguments)]
unsafe fn qkv(ctx: Ctx, l: &Layer, cur: T, hd: i64, nh: i64, nkv: i64, n_tokens: i64, packed: bool) -> (T, T, T) {
    if packed && !l.attn_qkv.is_null() {
        let all = g::ggml_mul_mat(ctx, l.attn_qkv, cur);
        let nb0 = (*all).nb[0];
        let row = hd as usize * nb0;
        let q_bytes = hd as usize * nh as usize * nb0;
        let kv_bytes = hd as usize * nkv as usize * nb0;
        let q = g::ggml_view_3d(ctx, all, hd, nh, n_tokens, row, (*all).nb[1], 0);
        let k = g::ggml_view_3d(ctx, all, hd, nkv, n_tokens, row, (*all).nb[1], q_bytes);
        let v = g::ggml_view_3d(ctx, all, hd, nkv, n_tokens, row, (*all).nb[1], q_bytes + kv_bytes);
        return (q, k, v);
    }
    (
        g::ggml_reshape_3d(ctx, g::ggml_mul_mat(ctx, l.attn_q, cur), hd, nh, n_tokens),
        g::ggml_reshape_3d(ctx, g::ggml_mul_mat(ctx, l.attn_k, cur), hd, nkv, n_tokens),
        g::ggml_reshape_3d(ctx, g::ggml_mul_mat(ctx, l.attn_v, cur), hd, nkv, n_tokens),
    )
}

/// `ffn_down_matmul`: an F16 down projection is cast to F32 first.
unsafe fn ffn_down(ctx: Ctx, w: T, x: T) -> T {
    let w = if (*w).type_ == g::GGML_TYPE_F16 {
        g::ggml_cast(ctx, w, g::GGML_TYPE_F32)
    } else {
        w
    };
    g::ggml_mul_mat(ctx, w, x)
}

/// The sampler's random stream: Philox4x32-10 keyed by the seed, one counter per draw
/// (`transformer_sampling.h:18-77`).
#[derive(Debug, Clone, Copy)]
pub struct Philox {
    pub seed: i64,
    pub subseq: i64,
}

impl Philox {
    fn round(c: [u32; 4], k0: u32, k1: u32) -> [u32; 4] {
        let p0 = 0xD251_1F53u64 * c[0] as u64;
        let p1 = 0xCD9E_8D57u64 * c[2] as u64;
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
    }

    /// `transformer_philox_uniform(seed, subseq)`, then advances.
    pub fn next_uniform(&mut self) -> f32 {
        let s = self.subseq as u64;
        self.subseq += 1;
        let mut c = [0u32, 0, s as u32, (s >> 32) as u32];
        let (mut k0, mut k1) = (self.seed as u32, ((self.seed as u64) >> 32) as u32);
        for i in 0..10 {
            c = Self::round(c, k0, k1);
            if i != 9 {
                k0 = k0.wrapping_add(0x9E37_79B9);
                k1 = k1.wrapping_add(0xBB67_AE85);
            }
        }
        (c[0] as f32 + 0.5f32) * (1.0f32 / 4_294_967_296.0f32)
    }
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as i32
}

/// `transformer_apply_repetition_penalty`: each distinct earlier token once.
fn repetition_penalty(logits: &mut [f32], history: &[i32], penalty: f32) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &t in history {
        if t < 0 || t as usize >= logits.len() || seen[t as usize] {
            continue;
        }
        seen[t as usize] = true;
        let s = logits[t as usize];
        logits[t as usize] = if s < 0.0 { s * penalty } else { s / penalty };
    }
}

/// `transformer_sample_top_k_p` (`transformer_sampling.h:113-213`).
fn sample_top_k_p(logits: &mut [f32], s: Sampling, rng: &mut Philox) -> i32 {
    let n = logits.len();
    if s.temperature <= 0.0 {
        return argmax(logits);
    }
    let inv = 1.0f32 / s.temperature;
    for x in logits.iter_mut() {
        *x *= inv;
    }
    if s.top_k > 0 && (s.top_k as usize) < n {
        // The k-th largest value, as nth_element with std::greater finds it.
        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[s.top_k as usize - 1];
        for x in logits.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
        let mut sum_exp = 0f32;
        for &x in logits.iter() {
            sum_exp += (x - max).exp();
        }
        let inv_sum = 1.0f32 / sum_exp;
        let cutoff = max - 16.0;
        let mut probs: Vec<(usize, f32)> = Vec::new();
        for (i, x) in logits.iter_mut().enumerate() {
            if *x >= cutoff {
                probs.push((i, (*x - max).exp() * inv_sum));
            } else {
                *x = f32::NEG_INFINITY;
            }
        }
        // std::sort is not stable; equal probabilities keep index order here, which can differ.
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0f32;
        for (j, &(i, p)) in probs.iter().enumerate() {
            if j > 0 && cum >= s.top_p {
                logits[i] = f32::NEG_INFINITY;
            }
            cum += p;
        }
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
    let mut sum = 0f32;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let target = rng.next_uniform() * sum;
    let mut acc = 0f32;
    for (i, &x) in logits.iter().enumerate() {
        acc += x;
        if acc >= target {
            return i as i32;
        }
    }
    n as i32 - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_is_deterministic_and_in_range() {
        let mut a = Philox {
            seed: 20260905,
            subseq: 0,
        };
        let mut b = Philox {
            seed: 20260905,
            subseq: 0,
        };
        for _ in 0..100 {
            let x = a.next_uniform();
            assert_eq!(x.to_bits(), b.next_uniform().to_bits());
            assert!(x > 0.0 && x < 1.0);
        }
    }

    #[test]
    fn languages_match_the_cli_table() {
        assert_eq!(language_id("zh"), Some(2055));
        assert_eq!(language_id("auto"), Some(-1));
        assert_eq!(language_id("yue"), None);
    }
}
