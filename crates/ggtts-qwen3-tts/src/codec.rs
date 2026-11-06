//! The codec decoder: codec tokens to 24 kHz samples.
//!
//! A port of qwen3-tts.cpp at `16bb5af`, `src/decoder/`, the non-streaming path the CLI takes
//! (`pipeline_synthesize.cpp:1667`, `AudioTokenizerDecoder::decode`): the weights as `decoder_loader.cpp` loads and
//! prepares them, and the graph as `decoder_graph.cpp:82-283` and `decoder_layers.cpp` build it, with the summed
//! rest-codebook projection the reference picks on CPU. On the CPU backend the samples are bit-identical to the
//! reference's; `tests/decoder_identity.rs` checks that.

use std::collections::HashMap;
use std::path::Path;
use std::ptr;

use ggtts_core::engine::{EngineError, Pcm};
use ggtts_ggml_sys as g;

use crate::cpu::{Cpu, Device, Meta};
use crate::gguf::{set_name, Gguf, Weights};

type T = *mut g::ggml_tensor;
type Ctx = *mut g::ggml_context;

const MAX_NODES: usize = 32768;

/// `audio_decoder_config` (`src/audio_tokenizer_decoder.h:32-47`), with the four values the GGUF may override.
#[derive(Debug, Clone)]
pub struct Config {
    pub sample_rate: i32,
    pub n_codebooks: i32,
    pub codebook_size: i32,
    pub codebook_dim: i32,
    pub latent_dim: i32,
    pub hidden_dim: i32,
    pub n_pre_tfm_layers: i32,
    pub n_heads: i32,
    pub sliding_window: i32,
    pub decoder_dim: i32,
    pub upsample_rates: [i32; 4],
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

#[cfg(feature = "plant")]
pub use planted::Plant;
#[cfg(not(feature = "plant"))]
use planted::Plant;

mod planted {
    /// A deliberate defect, for showing that the identity check can fail. Public only with the `plant` feature,
    /// which exists for the identity tests: a build without it cannot plant one.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[cfg_attr(not(feature = "plant"), allow(dead_code))]
    pub enum Plant {
        #[default]
        None,
        /// The pre-transformer's rope in normal (adjacent-pair) mode instead of NEOX.
        RopeNormal,
        /// The snake activation's `1/(exp(beta) + 1e-9)` without the epsilon.
        SnakeNoEpsilon,
        /// The sliding window one frame narrower.
        WindowOffByOne,
    }

    #[cfg(feature = "plant")]
    impl Plant {
        /// `none`, `rope-normal`, `snake-no-epsilon` or `window-off-by-one`.
        pub fn parse(s: &str) -> Option<Self> {
            Some(match s {
                "" | "none" => Self::None,
                "rope-normal" => Self::RopeNormal,
                "snake-no-epsilon" => Self::SnakeNoEpsilon,
                "window-off-by-one" => Self::WindowOffByOne,
                _ => return None,
            })
        }
    }
}

#[derive(Default)]
struct Snake {
    alpha_exp: T,
    inv_beta_exp: T,
}

struct Residual {
    dilation: i32,
    act1: Snake,
    conv1_w: T,
    conv1_b: T,
    act2: Snake,
    conv2_w: T,
    conv2_b: T,
}

struct Block {
    snake: Snake,
    conv_t_w: T,
    conv_t_w_perm: T,
    conv_t_b: T,
    res: [Residual; 3],
}

struct Upsample {
    conv_w_perm: T,
    conv_b: T,
    dwconv_w: T,
    dwconv_b: T,
    norm_w: T,
    norm_b: T,
    pwconv1_w: T,
    pwconv1_b: T,
    pwconv2_w: T,
    pwconv2_b: T,
    gamma: T,
}

struct Layer {
    attn_norm_w: T,
    attn_q_w: T,
    attn_k_w: T,
    attn_v_w: T,
    attn_output_w: T,
    attn_scale: T,
    ffn_norm_w: T,
    ffn_gate_w: T,
    ffn_up_w: T,
    ffn_down_w: T,
    ffn_scale: T,
}

/// The loaded codec decoder.
pub struct Decoder {
    pub config: Config,
    _weights: Weights,
    vq_first_codebook: T,
    vq_first_output_proj: T,
    vq_rest_codebook: [T; 15],
    vq_rest_output_proj: T,
    pre_conv_w: T,
    pre_conv_b: T,
    pre_tfm_input_proj_w: T,
    pre_tfm_input_proj_b: T,
    layers: Vec<Layer>,
    pre_tfm_norm_w: T,
    pre_tfm_output_proj_w: T,
    pre_tfm_output_proj_b: T,
    upsample: [Upsample; 2],
    dec0_conv_w: T,
    dec0_conv_b: T,
    blocks: Vec<Block>,
    dec5: Snake,
    dec6_conv_w: T,
    dec6_conv_b: T,
    cpu: Cpu,
    meta: Meta,
    plant: Plant,
}

/// `is_runtime_f16_conv_weight` (`decoder_loader.cpp:32-55`): conv weights the reference holds as F16 whatever the
/// file's type.
fn is_runtime_f16_conv_weight(name: &str) -> bool {
    if name == "tok_dec.pre_conv.weight" || name == "tok_dec.dec.0.conv.weight" || name == "tok_dec.dec.6.conv.weight" {
        return true;
    }
    if name.starts_with("tok_dec.upsample.") && name.ends_with(".dwconv.weight") {
        return true;
    }
    name.starts_with("tok_dec.dec.")
        && name.contains(".res.")
        && (name.ends_with(".conv1.weight") || name.ends_with(".conv2.weight"))
}

fn err(msg: impl Into<String>) -> EngineError {
    EngineError(msg.into())
}

impl Decoder {
    /// Loads the codec GGUF (the `tok_dec.*` tensors) with `threads` CPU threads.
    pub fn load(path: &Path, threads: i32) -> Result<Self, EngineError> {
        Self::load_on(path, threads, Device::Cpu)
    }

    /// Loads onto `device`: the weights in its buffer, the graphs on its scheduler.
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
        let config = Config {
            sample_rate: gguf.u32_or(
                "qwen3-tts-tokenizer.sample_rate",
                gguf.u32_or(
                    "qwen3-tts-tokenizer.output_sample_rate",
                    gguf.u32_or("qwen3-tts.tokenizer.sample_rate", 24000),
                ),
            ),
            n_codebooks: gguf.u32_or(
                "qwen3-tts-tokenizer.num_codebooks",
                gguf.u32_or(
                    "qwen3-tts-tokenizer.decoder.num_quantizers",
                    gguf.u32_or("qwen3-tts.tokenizer.num_codebooks", 16),
                ),
            ),
            codebook_size: gguf.u32_or(
                "qwen3-tts-tokenizer.codebook_size",
                gguf.u32_or(
                    "qwen3-tts-tokenizer.decoder.codebook_size",
                    gguf.u32_or("qwen3-tts.tokenizer.codebook_size", 2048),
                ),
            ),
            codebook_dim: 256,
            latent_dim: 1024,
            hidden_dim: 512,
            n_pre_tfm_layers: 8,
            n_heads: 16,
            sliding_window: gguf.u32_or(
                "qwen3-tts-tokenizer.decoder.sliding_window",
                gguf.u32_or("qwen3-tts.tokenizer.decoder.sliding_window", 72),
            ),
            decoder_dim: 1536,
            upsample_rates: [8, 5, 4, 3],
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        };
        if config.n_codebooks != 16 {
            return Err(err(format!(
                "codec has {} codebooks; this decoder needs 16",
                config.n_codebooks
            )));
        }

        // The tensor context: every tok_dec.* tensor, plus room for the derived snake and permuted tensors.
        let names: Vec<String> = (0..gguf.n_tensors())
            .map(|i| gguf.tensor_name(i))
            .filter(|n| n.starts_with("tok_dec."))
            .collect();
        if names.is_empty() {
            return Err(err("no decoder tensors in the codec GGUF"));
        }
        // SAFETY: size query; the context is owned by Weights below.
        let ctx = unsafe {
            g::ggml_init(g::ggml_init_params {
                mem_size: g::ggml_tensor_overhead() * (names.len() + 72),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            return Err(err("failed to initialize the decoder weight context"));
        }
        let mut tensors = HashMap::new();
        for name in &names {
            let meta = gguf.meta_tensor(name);
            if meta.is_null() {
                continue;
            }
            // SAFETY: meta is a live tensor of the file's metadata context; ctx has room by construction.
            let t = unsafe {
                if is_runtime_f16_conv_weight(name) {
                    g::ggml_new_tensor(ctx, g::GGML_TYPE_F16, g::ggml_n_dims(meta), (*meta).ne.as_ptr())
                } else {
                    g::ggml_dup_tensor(ctx, meta)
                }
            };
            set_name(t, name);
            tensors.insert(name.clone(), t);
        }
        let get = |n: &str| tensors.get(n).copied().unwrap_or(ptr::null_mut());
        let need = |n: &str| -> Result<T, EngineError> {
            tensors
                .get(n)
                .copied()
                .ok_or_else(|| err(format!("codec GGUF lacks {n}")))
        };

        // Residual indices are 2..4 in the converted files, 0..2 in some (`decoder_loader.cpp:219-232`).
        let zero_based = names
            .iter()
            .any(|n| (1..=4).any(|b| (0..=1).any(|r| n.starts_with(&format!("tok_dec.dec.{b}.res.{r}.")))));
        let res_name = |r: usize| if zero_based { r } else { r + 2 };

        // `prepare_snake_tensors` (`decoder_loader.cpp:509-553`): F32 [1, channels] tensors for exp(alpha) and
        // 1/(exp(beta) + 1e-9), created in the weight context, filled after the load.
        let mut snake_src: Vec<(T, T, T, T)> = Vec::new();
        let mut snake = |prefix: &str| -> Snake {
            let alpha = get(&format!("{prefix}.alpha"));
            let beta = get(&format!("{prefix}.beta"));
            if alpha.is_null() || beta.is_null() {
                return Snake::default();
            }
            // SAFETY: alpha is a live tensor; ctx has room.
            let (ae, ib) = unsafe {
                let ch = (*alpha).ne[0];
                (
                    g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, 1, ch),
                    g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, 1, ch),
                )
            };
            set_name(ae, &format!("{prefix}.alpha_exp"));
            set_name(ib, &format!("{prefix}.inv_beta_exp"));
            snake_src.push((alpha, beta, ae, ib));
            Snake {
                alpha_exp: ae,
                inv_beta_exp: ib,
            }
        };

        let mut blocks = Vec::new();
        for i in 1..=4usize {
            let s = snake(&format!("tok_dec.dec.{i}.snake"));
            let mut res = Vec::new();
            for (r, dilation) in [(0usize, 1), (1, 3), (2, 9)] {
                let p = format!("tok_dec.dec.{i}.res.{}", res_name(r));
                let act1 = snake(&format!("{p}.act1"));
                let act2 = snake(&format!("{p}.act2"));
                res.push(Residual {
                    dilation,
                    act1,
                    conv1_w: need(&format!("{p}.conv1.weight"))?,
                    conv1_b: get(&format!("{p}.conv1.bias")),
                    act2,
                    conv2_w: need(&format!("{p}.conv2.weight"))?,
                    conv2_b: get(&format!("{p}.conv2.bias")),
                });
            }
            let res: [Residual; 3] = res.try_into().map_err(|_| err("residual blocks"))?;
            blocks.push(Block {
                snake: s,
                conv_t_w: need(&format!("tok_dec.dec.{i}.conv_t.weight"))?,
                conv_t_w_perm: ptr::null_mut(),
                conv_t_b: get(&format!("tok_dec.dec.{i}.conv_t.bias")),
                res,
            });
        }
        let dec5 = snake("tok_dec.dec.5.snake");

        // `prepare_transconv_tensors` (`decoder_loader.cpp:626-658`): F32 [in, kernel*out] permuted copies.
        let perm_of = |src: T, name: &str| -> T {
            // SAFETY: src is a live tensor; ctx has room.
            unsafe {
                let (k, oc, ic) = ((*src).ne[0], (*src).ne[1], (*src).ne[2]);
                let t = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, ic, k * oc);
                set_name(t, &format!("{name}.perm_f32"));
                t
            }
        };
        let mut upsample_conv_w = [ptr::null_mut(); 2];
        let mut upsample = Vec::new();
        for (i, slot) in upsample_conv_w.iter_mut().enumerate() {
            let p = format!("tok_dec.upsample.{i}");
            let conv_w = need(&format!("{p}.conv.weight"))?;
            *slot = conv_w;
            upsample.push(Upsample {
                conv_w_perm: perm_of(conv_w, &format!("{p}.conv.weight")),
                conv_b: get(&format!("{p}.conv.bias")),
                dwconv_w: get(&format!("{p}.dwconv.weight")),
                dwconv_b: get(&format!("{p}.dwconv.bias")),
                norm_w: get(&format!("{p}.norm.weight")),
                norm_b: get(&format!("{p}.norm.bias")),
                pwconv1_w: need(&format!("{p}.pwconv1.weight"))?,
                pwconv1_b: get(&format!("{p}.pwconv1.bias")),
                pwconv2_w: need(&format!("{p}.pwconv2.weight"))?,
                pwconv2_b: get(&format!("{p}.pwconv2.bias")),
                gamma: get(&format!("{p}.gamma")),
            });
        }
        for (i, b) in blocks.iter_mut().enumerate() {
            b.conv_t_w_perm = perm_of(b.conv_t_w, &format!("tok_dec.dec.{}.conv_t.weight", i + 1));
        }
        let upsample: [Upsample; 2] = upsample.try_into().map_err(|_| err("upsample blocks"))?;

        let mut layers = Vec::new();
        for i in 0..config.n_pre_tfm_layers {
            let p = format!("tok_dec.pre_tfm.blk.{i}");
            layers.push(Layer {
                attn_norm_w: need(&format!("{p}.attn_norm.weight"))?,
                attn_q_w: need(&format!("{p}.attn_q.weight"))?,
                attn_k_w: need(&format!("{p}.attn_k.weight"))?,
                attn_v_w: need(&format!("{p}.attn_v.weight"))?,
                attn_output_w: need(&format!("{p}.attn_output.weight"))?,
                attn_scale: get(&format!("{p}.attn_scale")),
                ffn_norm_w: need(&format!("{p}.ffn_norm.weight"))?,
                ffn_gate_w: need(&format!("{p}.ffn_gate.weight"))?,
                ffn_up_w: need(&format!("{p}.ffn_up.weight"))?,
                ffn_down_w: need(&format!("{p}.ffn_down.weight"))?,
                ffn_scale: get(&format!("{p}.ffn_scale")),
            });
        }

        let mut vq_rest_codebook = [ptr::null_mut(); 15];
        let mut vq_rest_usage = [ptr::null_mut(); 15];
        for i in 0..15 {
            vq_rest_codebook[i] = need(&format!("tok_dec.vq_rest.{i}.codebook"))?;
            vq_rest_usage[i] = get(&format!("tok_dec.vq_rest.{i}.usage"));
        }
        let vq_first_codebook = need("tok_dec.vq_first.0.codebook")?;
        let vq_first_usage = get("tok_dec.vq_first.0.usage");

        // The derived tensors are in ctx, so they share the allocation; only the file's tensors are read.
        let lookup = |n: &str| tensors.get(n).copied().unwrap_or(ptr::null_mut());
        let vq_first_output_proj = lookup("tok_dec.vq_first.output_proj.weight");
        let vq_rest_output_proj = lookup("tok_dec.vq_rest.output_proj.weight");
        let pre_conv_w = lookup("tok_dec.pre_conv.weight");
        let pre_conv_b = lookup("tok_dec.pre_conv.bias");
        let pre_tfm_input_proj_w = lookup("tok_dec.pre_tfm.input_proj.weight");
        let pre_tfm_input_proj_b = lookup("tok_dec.pre_tfm.input_proj.bias");
        let pre_tfm_norm_w = lookup("tok_dec.pre_tfm.norm.weight");
        let pre_tfm_output_proj_w = lookup("tok_dec.pre_tfm.output_proj.weight");
        let pre_tfm_output_proj_b = lookup("tok_dec.pre_tfm.output_proj.bias");
        let dec0_conv_w = lookup("tok_dec.dec.0.conv.weight");
        let dec0_conv_b = lookup("tok_dec.dec.0.conv.bias");
        let dec6_conv_w = lookup("tok_dec.dec.6.conv.weight");
        let dec6_conv_b = lookup("tok_dec.dec.6.conv.bias");
        for (name, t) in [
            ("vq_first.output_proj", vq_first_output_proj),
            ("vq_rest.output_proj", vq_rest_output_proj),
            ("pre_conv.weight", pre_conv_w),
            ("pre_tfm.input_proj.weight", pre_tfm_input_proj_w),
            ("pre_tfm.output_proj.weight", pre_tfm_output_proj_w),
            ("dec.0.conv.weight", dec0_conv_w),
            ("dec.6.conv.weight", dec6_conv_w),
        ] {
            if t.is_null() {
                return Err(err(format!("codec GGUF lacks tok_dec.{name}")));
            }
        }
        // SAFETY: ctx is the live no-alloc context every tensor above was made in.
        // SAFETY: ctx is the live no-alloc context every tensor above was made in.
        let buffer = unsafe { gguf.load_into(ctx, &tensors, device)? };
        let weights = Weights { ctx, buffer, tensors };

        // SAFETY: every tensor below lives in the buffer just filled; it is read and written through the backend
        // API, as the reference does, so this holds on CUDA as on CPU.
        unsafe {
            // `upload_snake_tensors` (`decoder_loader.cpp:555-624`).
            for &(alpha, beta, ae, ib) in &snake_src {
                if (*alpha).type_ != g::GGML_TYPE_F32 || (*beta).type_ != g::GGML_TYPE_F32 {
                    return Err(err("snake tensors must be F32"));
                }
                let ch = (*alpha).ne[0] as usize;
                let a = get_f32(alpha);
                let b = get_f32(beta);
                let eps = if plant == Plant::SnakeNoEpsilon {
                    0.0f32
                } else {
                    1e-9f32
                };
                let ae_host: Vec<f32> = a.iter().map(|&x| x.exp()).collect();
                let ib_host: Vec<f32> = b.iter().map(|&x| 1.0f32 / (x.exp() + eps)).collect();
                g::ggml_backend_tensor_set(ae, ae_host.as_ptr().cast(), 0, ch * 4);
                g::ggml_backend_tensor_set(ib, ib_host.as_ptr().cast(), 0, ch * 4);
            }
            // `upload_transconv_tensors` (`decoder_loader.cpp:660-729`).
            let perms = upsample_conv_w
                .iter()
                .zip(upsample.iter().map(|u| u.conv_w_perm))
                .chain(blocks.iter().map(|b| (&b.conv_t_w, b.conv_t_w_perm)));
            for (&src, dst) in perms {
                let (k, oc, ic) = ((*src).ne[0] as usize, (*src).ne[1] as usize, (*src).ne[2] as usize);
                let ty = (*src).type_;
                if ty != g::GGML_TYPE_F32 && ty != g::GGML_TYPE_F16 && ty != g::GGML_TYPE_BF16 {
                    return Err(err("a ConvTranspose weight cannot be pre-permuted"));
                }
                let bytes = get_bytes(src);
                let load = |idx: usize| -> f32 {
                    let d = bytes.as_ptr();
                    match ty {
                        g::GGML_TYPE_F32 => (d as *const f32).add(idx).read_unaligned(),
                        g::GGML_TYPE_F16 => g::ggml_fp16_to_fp32((d as *const u16).add(idx).read_unaligned()),
                        _ => g::ggml_bf16_to_fp32((d as *const g::ggml_bf16_t).add(idx).read_unaligned()),
                    }
                };
                let mut host = vec![0f32; k * oc * ic];
                for i in 0..ic {
                    for o in 0..oc {
                        for kk in 0..k {
                            host[(o * k + kk) * ic + i] = load(i * oc * k + o * k + kk);
                        }
                    }
                }
                g::ggml_backend_tensor_set(dst, host.as_ptr().cast(), 0, host.len() * 4);
            }
            // `normalize_codebooks` (`decoder_loader.cpp:120-157`): each F16 codebook row divided by its usage.
            // The reference writes through the tensor's data pointer, which is host memory on CPU. On a device
            // buffer this reads, normalizes and writes back the same values.
            let normalize = |cb: T, usage: T| {
                if cb.is_null() || usage.is_null() {
                    return;
                }
                let dim = (*cb).ne[0] as usize;
                let size = (*cb).ne[1] as usize;
                let mut data: Vec<u16> = get_bytes(cb)
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let u = get_f32(usage);
                for e in 0..size {
                    let mut uu = u[e];
                    if uu < 1e-5f32 {
                        uu = 1e-5f32;
                    }
                    let inv = 1.0f32 / uu;
                    for d in 0..dim {
                        let p = &mut data[d + e * dim];
                        *p = g::ggml_fp32_to_fp16(g::ggml_fp16_to_fp32(*p) * inv);
                    }
                }
                g::ggml_backend_tensor_set(cb, data.as_ptr().cast(), 0, data.len() * 2);
            };
            normalize(vq_first_codebook, vq_first_usage);
            for i in 0..15 {
                normalize(vq_rest_codebook[i], vq_rest_usage[i]);
            }
        }

        let cpu = Cpu::on(device, threads, MAX_NODES)?;
        Ok(Self {
            vq_first_codebook,
            vq_first_output_proj,
            vq_rest_codebook,
            vq_rest_output_proj,
            pre_conv_w,
            pre_conv_b,
            pre_tfm_input_proj_w,
            pre_tfm_input_proj_b,
            pre_tfm_norm_w,
            pre_tfm_output_proj_w,
            pre_tfm_output_proj_b,
            dec0_conv_w,
            dec0_conv_b,
            dec6_conv_w,
            dec6_conv_b,
            layers,
            upsample,
            blocks,
            dec5,
            config,
            _weights: weights,
            cpu,
            meta: Meta::new(MAX_NODES),
            plant,
        })
    }

    /// Decodes frame-major codes (`codes[f * 16 + c]`) to samples, `AudioTokenizerDecoder::decode`
    /// (`decoder_runtime.cpp:25-138`).
    pub fn decode(&mut self, codes: &[i32]) -> Result<Pcm, EngineError> {
        let nc = self.config.n_codebooks as usize;
        if codes.is_empty() || codes.len() % nc != 0 {
            return Err(err("codes must be a non-empty whole number of 16-code frames"));
        }
        let n = (codes.len() / nc) as i32;
        let limit = self.config.codebook_size;
        if let Some(bad) = codes.iter().find(|&&c| c < 0 || c >= limit) {
            return Err(err(format!("codec token {bad} is outside 0..{limit}")));
        }
        let ctx = self.meta.context();
        // SAFETY: ctx is a fresh no-alloc context over self.meta; every tensor used is live; inputs are set only
        // after the scheduler allocates the graph, and every input set is in the graph.
        unsafe {
            let (gf, inputs, positions, mask, audio) = self.build(ctx, n);
            let sched = self.cpu.sched;
            if !g::ggml_backend_sched_alloc_graph(sched, gf) {
                g::ggml_free(ctx);
                return Err(err("failed to allocate the decoder graph"));
            }
            for (cb, &t) in inputs.iter().enumerate() {
                let col: Vec<i32> = (0..n as usize).map(|f| codes[f * nc + cb]).collect();
                g::ggml_backend_tensor_set(t, col.as_ptr().cast(), 0, col.len() * 4);
            }
            let pos: Vec<i32> = (0..n).collect();
            g::ggml_backend_tensor_set(positions, pos.as_ptr().cast(), 0, pos.len() * 4);
            let mut window = if self.config.sliding_window > 0 {
                self.config.sliding_window
            } else {
                n
            };
            if self.plant == Plant::WindowOffByOne {
                window -= 1;
            }
            let nn = n as usize;
            let mut m = vec![f32::NEG_INFINITY; nn * nn];
            for q in 0..n {
                let k_min = (q - window + 1).max(0);
                for k in k_min..=q {
                    m[q as usize * nn + k as usize] = 0.0;
                }
            }
            g::ggml_backend_tensor_set(mask, m.as_ptr().cast(), 0, m.len() * 4);
            if g::ggml_backend_sched_graph_compute(sched, gf) != g::GGML_STATUS_SUCCESS {
                g::ggml_backend_sched_reset(sched);
                g::ggml_free(ctx);
                return Err(err("decoder graph compute failed"));
            }
            let ns = (*audio).ne[0] as usize;
            let mut samples = vec![0f32; ns];
            g::ggml_backend_tensor_get(audio, samples.as_mut_ptr().cast(), 0, ns * 4);
            g::ggml_backend_sched_reset(sched);
            g::ggml_free(ctx);
            Ok(Pcm {
                samples,
                sample_rate: self.config.sample_rate as u32,
            })
        }
    }

    /// `build_graph_impl` (`decoder_graph.cpp:82-283`).
    unsafe fn build(&self, ctx: Ctx, n: i32) -> (*mut g::ggml_cgraph, [T; 16], T, T, T) {
        let c = &self.config;
        let n64 = n as i64;
        let gf = g::ggml_new_graph_custom(ctx, MAX_NODES, false);
        let mut inputs = [ptr::null_mut(); 16];
        for (cb, slot) in inputs.iter_mut().enumerate() {
            let t = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, n64);
            set_name(t, &format!("codes_cb{cb}"));
            g::ggml_set_input(t);
            *slot = t;
        }
        let first_emb = g::ggml_get_rows(ctx, self.vq_first_codebook, inputs[0]);
        let first_emb_2d = g::ggml_reshape_2d(ctx, first_emb, c.codebook_dim as i64, n64);
        let first_w = g::ggml_reshape_2d(
            ctx,
            self.vq_first_output_proj,
            c.codebook_dim as i64,
            c.hidden_dim as i64,
        );
        let first_proj = g::ggml_mul_mat(ctx, first_w, first_emb_2d);
        let rest_w = g::ggml_reshape_2d(
            ctx,
            self.vq_rest_output_proj,
            c.codebook_dim as i64,
            c.hidden_dim as i64,
        );
        // The reference sums the 15 embeddings and projects once on CPU, and on CUDA below 64 frames; otherwise it
        // projects each and sums the projections (`decoder_graph.cpp:61-74,147-166`).
        let summed = !self.cpu.device.is_cuda() || n < 64;
        let mut rest_proj: T = ptr::null_mut();
        let mut sum: T = ptr::null_mut();
        for cb in 0..15 {
            let e = g::ggml_get_rows(ctx, self.vq_rest_codebook[cb], inputs[cb + 1]);
            let e2 = g::ggml_reshape_2d(ctx, e, c.codebook_dim as i64, n64);
            if summed {
                sum = if sum.is_null() { e2 } else { g::ggml_add(ctx, sum, e2) };
            } else {
                let p = g::ggml_mul_mat(ctx, rest_w, e2);
                rest_proj = if rest_proj.is_null() {
                    p
                } else {
                    g::ggml_add(ctx, rest_proj, p)
                };
            }
        }
        if summed {
            rest_proj = g::ggml_mul_mat(ctx, rest_w, sum);
        }
        let latent = g::ggml_add(ctx, first_proj, rest_proj);
        let latent = g::ggml_cont(ctx, g::ggml_transpose(ctx, latent));
        let latent = g::ggml_reshape_3d(ctx, latent, n64, c.hidden_dim as i64, 1);
        let latent = g::ggml_cont(ctx, latent);
        let padded = g::ggml_pad_ext(ctx, latent, 2, 0, 0, 0, 0, 0, 0, 0);
        let mut cur = g::ggml_conv_1d(ctx, f16w(ctx, self.pre_conv_w), padded, 1, 0, 1);
        if !self.pre_conv_b.is_null() {
            cur = g::ggml_add(
                ctx,
                cur,
                g::ggml_reshape_3d(ctx, self.pre_conv_b, 1, c.latent_dim as i64, 1),
            );
        }
        let cur2 = g::ggml_reshape_2d(ctx, cur, n64, c.latent_dim as i64);
        cur = g::ggml_cont(ctx, g::ggml_transpose(ctx, cur2));
        cur = g::ggml_mul_mat(ctx, self.pre_tfm_input_proj_w, cur);
        if !self.pre_tfm_input_proj_b.is_null() {
            cur = g::ggml_add(ctx, cur, self.pre_tfm_input_proj_b);
        }
        let positions = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, n64);
        set_name(positions, "positions");
        g::ggml_set_input(positions);
        let mask = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, n64, n64);
        set_name(mask, "mask");
        g::ggml_set_input(mask);
        for layer in &self.layers {
            cur = self.pre_tfm_layer(ctx, cur, layer, n64, positions, mask);
        }
        if !self.pre_tfm_norm_w.is_null() {
            cur = rms(ctx, cur, self.pre_tfm_norm_w, c.rms_norm_eps);
        }
        cur = g::ggml_mul_mat(ctx, self.pre_tfm_output_proj_w, cur);
        if !self.pre_tfm_output_proj_b.is_null() {
            cur = g::ggml_add(ctx, cur, self.pre_tfm_output_proj_b);
        }
        cur = g::ggml_permute(ctx, cur, 1, 0, 2, 3);
        cur = g::ggml_cont(ctx, cur);
        cur = g::ggml_reshape_3d(ctx, cur, n64, c.latent_dim as i64, 1);
        for u in &self.upsample {
            cur = upsample_block(ctx, cur, u);
        }
        cur = g::ggml_pad_ext(ctx, cur, 6, 0, 0, 0, 0, 0, 0, 0);
        cur = g::ggml_conv_1d(ctx, f16w(ctx, self.dec0_conv_w), cur, 1, 0, 1);
        if !self.dec0_conv_b.is_null() {
            cur = g::ggml_add(
                ctx,
                cur,
                g::ggml_reshape_3d(ctx, self.dec0_conv_b, 1, c.decoder_dim as i64, 1),
            );
        }
        for (i, b) in self.blocks.iter().enumerate() {
            cur = decoder_block(ctx, cur, b, c.upsample_rates[i]);
        }
        if !self.dec5.alpha_exp.is_null() {
            cur = snake(ctx, cur, &self.dec5);
        }
        cur = g::ggml_pad_ext(ctx, cur, 6, 0, 0, 0, 0, 0, 0, 0);
        cur = g::ggml_conv_1d(ctx, f16w(ctx, self.dec6_conv_w), cur, 1, 0, 1);
        if !self.dec6_conv_b.is_null() {
            cur = g::ggml_add(ctx, cur, g::ggml_reshape_3d(ctx, self.dec6_conv_b, 1, 1, 1));
        }
        cur = g::ggml_tanh(ctx, cur);
        cur = g::ggml_reshape_1d(ctx, cur, (*cur).ne[0]);
        set_name(cur, "audio");
        g::ggml_set_output(cur);
        g::ggml_build_forward_expand(gf, cur);
        (gf, inputs, positions, mask, cur)
    }

    /// `apply_pre_tfm_layer` (`decoder_layers.cpp:128-203`).
    unsafe fn pre_tfm_layer(&self, ctx: Ctx, x: T, l: &Layer, n: i64, positions: T, mask: T) -> T {
        let c = &self.config;
        let heads = c.n_heads as i64;
        let hd = (c.latent_dim / c.n_heads) as i64;
        let mode = if self.plant == Plant::RopeNormal {
            0
        } else {
            g::GGML_ROPE_TYPE_NEOX as i32
        };
        let residual = x;
        let normed = rms(ctx, x, l.attn_norm_w, c.rms_norm_eps);
        let q = g::ggml_reshape_3d(ctx, g::ggml_mul_mat(ctx, l.attn_q_w, normed), hd, heads, n);
        let k = g::ggml_reshape_3d(ctx, g::ggml_mul_mat(ctx, l.attn_k_w, normed), hd, heads, n);
        let v = g::ggml_reshape_3d(ctx, g::ggml_mul_mat(ctx, l.attn_v_w, normed), hd, heads, n);
        let q = g::ggml_rope_ext(
            ctx,
            q,
            positions,
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
        let k = g::ggml_rope_ext(
            ctx,
            k,
            positions,
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
        let q = g::ggml_permute(ctx, q, 0, 2, 1, 3);
        let k = g::ggml_permute(ctx, k, 0, 2, 1, 3);
        let v = g::ggml_permute(ctx, v, 0, 2, 1, 3);
        let kq = g::ggml_mul_mat(ctx, k, q);
        let kq = g::ggml_soft_max_ext(ctx, kq, mask, 1.0 / (hd as f32).sqrt(), 0.0);
        let v = g::ggml_cont(ctx, g::ggml_transpose(ctx, v));
        let kqv = g::ggml_mul_mat(ctx, v, kq);
        let kqv = g::ggml_permute(ctx, kqv, 0, 2, 1, 3);
        let mut attn = g::ggml_cont_2d(ctx, kqv, heads * hd, n);
        attn = g::ggml_mul_mat(ctx, l.attn_output_w, attn);
        if !l.attn_scale.is_null() {
            attn = g::ggml_mul(ctx, attn, l.attn_scale);
        }
        let x = g::ggml_add(ctx, residual, attn);
        let residual = x;
        let normed = rms(ctx, x, l.ffn_norm_w, c.rms_norm_eps);
        let gate = g::ggml_mul_mat(ctx, l.ffn_gate_w, normed);
        let up = g::ggml_mul_mat(ctx, l.ffn_up_w, normed);
        let gate = g::ggml_silu(ctx, gate);
        let mut ffn = g::ggml_mul(ctx, gate, up);
        ffn = g::ggml_mul_mat(ctx, l.ffn_down_w, ffn);
        if !l.ffn_scale.is_null() {
            ffn = g::ggml_mul(ctx, ffn, l.ffn_scale);
        }
        g::ggml_add(ctx, residual, ffn)
    }
}

/// A tensor's bytes, read through the backend.
unsafe fn get_bytes(t: T) -> Vec<u8> {
    let mut b = vec![0u8; g::ggml_nbytes(t)];
    g::ggml_backend_tensor_get(t, b.as_mut_ptr().cast(), 0, b.len());
    b
}

/// An F32 tensor's values, read through the backend.
unsafe fn get_f32(t: T) -> Vec<f32> {
    get_bytes(t)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `as_f16_conv_weight`.
unsafe fn f16w(ctx: Ctx, w: T) -> T {
    if w.is_null() || (*w).type_ == g::GGML_TYPE_F16 {
        w
    } else {
        g::ggml_cont(ctx, g::ggml_cast(ctx, w, g::GGML_TYPE_F16))
    }
}

unsafe fn rms(ctx: Ctx, x: T, w: T, eps: f32) -> T {
    g::ggml_mul(ctx, g::ggml_rms_norm(ctx, x, eps), w)
}

/// `apply_snake` with precomputed exponentials (`decoder_layers.cpp:82-94`).
unsafe fn snake(ctx: Ctx, x: T, s: &Snake) -> T {
    let ax = g::ggml_mul(ctx, x, s.alpha_exp);
    let sin_sq = g::ggml_sqr(ctx, g::ggml_sin(ctx, ax));
    g::ggml_add(ctx, x, g::ggml_mul(ctx, sin_sq, s.inv_beta_exp))
}

/// `qwen_causal_trans_conv1d` (`decoder_layers.cpp:17-40`).
unsafe fn trans_conv(ctx: Ctx, w_perm: T, b: T, x: T, stride: i32, kernel: i32, out_channels: i32) -> T {
    let trim = kernel - stride;
    let xt = g::ggml_cont(ctx, g::ggml_transpose(ctx, x));
    let col = g::ggml_mul_mat(ctx, w_perm, xt);
    let mut y = g::ggml_col2im_1d(ctx, col, stride, out_channels, 0);
    if trim > 0 {
        let keep = (*y).ne[0] - trim as i64;
        y = g::ggml_view_2d(ctx, y, keep, (*y).ne[1], (*y).nb[1], 0);
    }
    if !b.is_null() {
        y = g::ggml_add(ctx, y, g::ggml_reshape_2d(ctx, b, 1, out_channels as i64));
    }
    y
}

/// `apply_upsample_block` (`decoder_layers.cpp:291-355`), with the permuted transposed-conv weight.
unsafe fn upsample_block(ctx: Ctx, x: T, u: &Upsample) -> T {
    let seq = (*x).ne[0];
    let ch = (*x).ne[1];
    let x2 = g::ggml_reshape_2d(ctx, x, seq, ch);
    let x2 = trans_conv(ctx, u.conv_w_perm, u.conv_b, x2, 2, 2, ch as i32);
    let new_seq = (*x2).ne[0];
    let mut x = g::ggml_reshape_3d(ctx, x2, new_seq, ch, 1);
    let residual = x;
    if !u.dwconv_w.is_null() {
        x = g::ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
        x = g::ggml_conv_1d_dw(ctx, f16w(ctx, u.dwconv_w), x, 1, 0, 1);
        if !u.dwconv_b.is_null() {
            x = g::ggml_add(ctx, x, g::ggml_reshape_3d(ctx, u.dwconv_b, 1, ch, 1));
        }
    }
    x = g::ggml_cont(ctx, g::ggml_permute(ctx, x, 1, 0, 2, 3));
    if !u.norm_w.is_null() && !u.norm_b.is_null() {
        x = g::ggml_norm(ctx, x, 1e-6);
        x = g::ggml_mul(ctx, x, u.norm_w);
        x = g::ggml_add(ctx, x, u.norm_b);
    }
    x = g::ggml_mul_mat(ctx, u.pwconv1_w, x);
    if !u.pwconv1_b.is_null() {
        x = g::ggml_add(ctx, x, u.pwconv1_b);
    }
    x = g::ggml_gelu(ctx, x);
    x = g::ggml_mul_mat(ctx, u.pwconv2_w, x);
    if !u.pwconv2_b.is_null() {
        x = g::ggml_add(ctx, x, u.pwconv2_b);
    }
    x = g::ggml_cont(ctx, g::ggml_permute(ctx, x, 1, 0, 2, 3));
    if !u.gamma.is_null() {
        let gamma = g::ggml_reshape_3d(ctx, u.gamma, 1, ch, 1);
        let shape = g::ggml_new_tensor_3d(ctx, g::GGML_TYPE_F32, new_seq, ch, 1);
        x = g::ggml_mul(ctx, x, g::ggml_repeat(ctx, gamma, shape));
    }
    g::ggml_add(ctx, residual, x)
}

/// `apply_residual_block` (`decoder_layers.cpp:393-423`).
unsafe fn residual_block(ctx: Ctx, x: T, r: &Residual) -> T {
    let residual = x;
    let mut x = x;
    if !r.act1.alpha_exp.is_null() {
        x = snake(ctx, x, &r.act1);
    }
    let oc = (*r.conv1_w).ne[2];
    x = g::ggml_pad_ext(ctx, x, 6 * r.dilation, 0, 0, 0, 0, 0, 0, 0);
    x = g::ggml_conv_1d(ctx, f16w(ctx, r.conv1_w), x, 1, 0, r.dilation);
    if !r.conv1_b.is_null() {
        x = g::ggml_add(ctx, x, g::ggml_reshape_3d(ctx, r.conv1_b, 1, oc, 1));
    }
    if !r.act2.alpha_exp.is_null() {
        x = snake(ctx, x, &r.act2);
    }
    let oc = (*r.conv2_w).ne[2];
    x = g::ggml_conv_1d(ctx, f16w(ctx, r.conv2_w), x, 1, 0, 1);
    if !r.conv2_b.is_null() {
        x = g::ggml_add(ctx, x, g::ggml_reshape_3d(ctx, r.conv2_b, 1, oc, 1));
    }
    g::ggml_add(ctx, residual, x)
}

/// `apply_decoder_block` (`decoder_layers.cpp:425-474`), with the permuted transposed-conv weight.
unsafe fn decoder_block(ctx: Ctx, x: T, b: &Block, rate: i32) -> T {
    let mut x = x;
    if !b.snake.alpha_exp.is_null() {
        x = snake(ctx, x, &b.snake);
    }
    let seq = (*x).ne[0];
    let ic = (*x).ne[1];
    let oc = (*b.conv_t_w).ne[1];
    let kernel = (*b.conv_t_w).ne[0] as i32;
    let x2 = g::ggml_reshape_2d(ctx, x, seq, ic);
    let x2 = trans_conv(ctx, b.conv_t_w_perm, b.conv_t_b, x2, rate, kernel, oc as i32);
    let new_seq = (*x2).ne[0];
    x = g::ggml_reshape_3d(ctx, x2, new_seq, oc, 1);
    for r in &b.res {
        x = residual_block(ctx, x, r);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::is_runtime_f16_conv_weight as f16;

    #[test]
    fn runtime_f16_weights_match_the_reference_list() {
        assert!(f16("tok_dec.pre_conv.weight"));
        assert!(f16("tok_dec.dec.0.conv.weight"));
        assert!(f16("tok_dec.dec.6.conv.weight"));
        assert!(f16("tok_dec.upsample.1.dwconv.weight"));
        assert!(f16("tok_dec.dec.3.res.2.conv1.weight"));
        assert!(f16("tok_dec.dec.3.res.4.conv2.weight"));
        assert!(!f16("tok_dec.dec.3.conv_t.weight"));
        assert!(!f16("tok_dec.upsample.0.conv.weight"));
        assert!(!f16("tok_dec.pre_conv.bias"));
    }
}
