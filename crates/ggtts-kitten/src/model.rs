//! The Kitten-TTS 2 language model as a ggml graph: Qwen3 with a KV cache, on the CPU backend.
//!
//! Per layer (Hugging Face `Qwen3DecoderLayer`): RMSNorm, q/k/v projections, per-head RMSNorm on q and k, NeoX
//! RoPE, grouped-query attention (16 query heads over 8 KV heads, a query head `h` reading KV head `h / 2`, as
//! `repeat_kv` and ggml's broadcast both do), the output projection, a residual; RMSNorm, SwiGLU, a residual. The
//! head is the tied bf16 embedding.
//!
//! The ternary matrices are padded `TQ2_0` (see [`crate::ternary`]), so every activation that enters one is padded
//! the same way first: `[n, T]` viewed as `[128, n / 128, T]`, padded to `[256, n / 128, T]` with zeros. With
//! [`Weights::F32`] the matrices are dequantized to f32 at load instead, which isolates the graph from the
//! arithmetic for the parity check.
//!
//! The token embedding is looked up on the host (it is bf16, and widening is exact), and position 0 is overwritten
//! by the projected speaker embedding before the first forward pass, as `kittenml/speaker.py` does at prefill.

use std::ffi::CString;
use std::ptr;

use ggtts_ggml_sys as g;

use crate::convert::{ARCH, EMBED, LAYOUT};
use crate::gguf::{GgmlType, Gguf};
use crate::safetensors::bf16_to_f32;
use crate::ternary::{self, GROUP};

/// How the ternary matrices are held at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weights {
    /// Padded `TQ2_0`, ggml's ternary kernels with `Q8_K` activations; the head stays bf16. The runtime path.
    Ternary,
    /// Every matrix dequantized to f32, the head included. The graph check.
    F32,
    /// Padded `TQ2_0` body with an f32 head: a diagnostic arm that separates the body's error from the head's.
    TernaryF32Head,
}

#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub spk_dim: usize,
}

struct Layer {
    attn_norm: *mut g::ggml_tensor,
    q: *mut g::ggml_tensor,
    k: *mut g::ggml_tensor,
    v: *mut g::ggml_tensor,
    o: *mut g::ggml_tensor,
    q_norm: *mut g::ggml_tensor,
    k_norm: *mut g::ggml_tensor,
    ffn_norm: *mut g::ggml_tensor,
    gate: *mut g::ggml_tensor,
    up: *mut g::ggml_tensor,
    down: *mut g::ggml_tensor,
}

/// A loaded model. Owns its ggml context.
pub struct Model {
    pub hp: Hparams,
    pub weights: Weights,
    ctx: *mut g::ggml_context,
    layers: Vec<Layer>,
    norm: *mut g::ggml_tensor,
    head: *mut g::ggml_tensor,
    /// The bf16 embedding, for the host-side lookup.
    embed: Vec<u16>,
    spk_w: Vec<f32>,
    spk_b: Vec<f32>,
    spk_ln_w: Vec<f32>,
    spk_ln_b: Vec<f32>,
    /// The GGUF's metadata, for the token map and the sampler's defaults.
    pub meta: std::collections::BTreeMap<String, crate::gguf::Value>,
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { g::ggml_free(self.ctx) };
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Model {
    pub fn load(gguf: &Gguf, weights: Weights) -> Result<Model, String> {
        if gguf.str("general.architecture")? != ARCH || gguf.str("kitten.ternary.layout")? != LAYOUT {
            return Err("not a ggtts Kitten-TTS 2 GGUF with a known ternary layout".into());
        }
        let u = |k: &str| gguf.u32(&format!("kitten.{k}")).map(|v| v as usize);
        let hp = Hparams {
            n_layer: u("block_count")?,
            n_embd: u("embedding_length")?,
            n_ff: u("feed_forward_length")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_vocab: u("vocab_size")?,
            eps: gguf.f32("kitten.attention.layer_norm_rms_epsilon")?,
            rope_base: gguf.f32("kitten.rope.freq_base")?,
            spk_dim: u("speaker_dim")?,
        };
        // Size the context: every tensor, f32 for dequantized matrices.
        let mut mem = 0usize;
        for (name, t) in &gguf.tensors {
            let n: usize = t.dims.iter().map(|&d| d as usize).product();
            mem += match (t.ty, weights) {
                (GgmlType::TQ2_0, Weights::F32) => n / 2 * 4,
                (GgmlType::BF16, Weights::F32 | Weights::TernaryF32Head) => n * 4,
                _ => t.len,
            };
            mem += unsafe { g::ggml_tensor_overhead() } + 64;
            let _ = name;
        }
        let ctx = unsafe {
            g::ggml_init(g::ggml_init_params {
                mem_size: mem + (1 << 20),
                mem_buffer: ptr::null_mut(),
                no_alloc: false,
            })
        };
        if ctx.is_null() {
            return Err("ggml_init failed for the weights".into());
        }
        let mut model = Model {
            hp,
            weights,
            ctx,
            layers: Vec::new(),
            norm: ptr::null_mut(),
            head: ptr::null_mut(),
            embed: Vec::new(),
            spk_w: Vec::new(),
            spk_b: Vec::new(),
            spk_ln_w: Vec::new(),
            spk_ln_b: Vec::new(),
            meta: gguf.kv.clone(),
        };
        let load = |name: &str| -> Result<*mut g::ggml_tensor, String> {
            let (t, data) = gguf.tensor(name)?;
            let cname = CString::new(name).unwrap();
            unsafe {
                let out = match (t.ty, weights) {
                    (GgmlType::TQ2_0, Weights::Ternary | Weights::TernaryF32Head) => {
                        let x = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_TQ2_0, t.dims[0] as i64, t.dims[1] as i64);
                        ptr::copy_nonoverlapping(data.as_ptr(), (*x).data as *mut u8, data.len());
                        x
                    }
                    (GgmlType::TQ2_0, Weights::F32) => {
                        let (cols, rows) = (t.dims[0] as usize / 2, t.dims[1] as usize);
                        let x = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, cols as i64, rows as i64);
                        let dst = std::slice::from_raw_parts_mut((*x).data as *mut f32, cols * rows);
                        let rb = cols / GROUP * ternary::TQ2_0_BLOCK_BYTES;
                        for r in 0..rows {
                            let row = ternary::dequantize_row(&data[r * rb..(r + 1) * rb], cols)?;
                            dst[r * cols..(r + 1) * cols].copy_from_slice(&row);
                        }
                        x
                    }
                    (GgmlType::BF16, Weights::Ternary) => {
                        let x = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_BF16, t.dims[0] as i64, t.dims[1] as i64);
                        ptr::copy_nonoverlapping(data.as_ptr(), (*x).data as *mut u8, data.len());
                        x
                    }
                    (GgmlType::BF16, Weights::F32 | Weights::TernaryF32Head) => {
                        let x = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, t.dims[0] as i64, t.dims[1] as i64);
                        let n = data.len() / 2;
                        let dst = std::slice::from_raw_parts_mut((*x).data as *mut f32, n);
                        for (d, c) in dst.iter_mut().zip(data.chunks_exact(2)) {
                            *d = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                        }
                        x
                    }
                    (GgmlType::F32, _) => {
                        let x = match t.dims.len() {
                            1 => g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_F32, t.dims[0] as i64),
                            _ => g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, t.dims[0] as i64, t.dims[1] as i64),
                        };
                        ptr::copy_nonoverlapping(data.as_ptr(), (*x).data as *mut u8, data.len());
                        x
                    }
                    (ty, _) => return Err(format!("{name}: unexpected type {ty:?}")),
                };
                g::ggml_set_name(out, cname.as_ptr());
                Ok(out)
            }
        };
        for l in 0..model.hp.n_layer {
            let p = |s: &str| format!("model.layers.{l}.{s}.weight");
            model.layers.push(Layer {
                attn_norm: load(&p("input_layernorm"))?,
                q: load(&p("self_attn.q_proj"))?,
                k: load(&p("self_attn.k_proj"))?,
                v: load(&p("self_attn.v_proj"))?,
                o: load(&p("self_attn.o_proj"))?,
                q_norm: load(&p("self_attn.q_norm"))?,
                k_norm: load(&p("self_attn.k_norm"))?,
                ffn_norm: load(&p("post_attention_layernorm"))?,
                gate: load(&p("mlp.gate_proj"))?,
                up: load(&p("mlp.up_proj"))?,
                down: load(&p("mlp.down_proj"))?,
            });
        }
        model.norm = load("model.norm.weight")?;
        model.head = load(EMBED)?;
        let (_, emb) = gguf.tensor(EMBED)?;
        model.embed = emb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        model.spk_w = f32s(gguf.tensor("spk_proj.0.weight")?.1);
        model.spk_b = f32s(gguf.tensor("spk_proj.0.bias")?.1);
        model.spk_ln_w = f32s(gguf.tensor("spk_proj.1.weight")?.1);
        model.spk_ln_b = f32s(gguf.tensor("spk_proj.1.bias")?.1);
        if model.spk_w.len() != model.hp.spk_dim * model.hp.n_embd {
            return Err("spk_proj.0.weight has the wrong size".into());
        }
        Ok(model)
    }

    /// `spk_proj`: Linear(512 -> hidden) then LayerNorm (eps 1e-5, PyTorch's default), in f32.
    pub fn project_speaker(&self, emb: &[f32]) -> Vec<f32> {
        let (n, d) = (self.hp.n_embd, self.hp.spk_dim);
        assert_eq!(emb.len(), d);
        let mut y: Vec<f32> = (0..n)
            .map(|o| {
                self.spk_b[o]
                    + self.spk_w[o * d..(o + 1) * d]
                        .iter()
                        .zip(emb)
                        .map(|(w, x)| w * x)
                        .sum::<f32>()
            })
            .collect();
        let mean = y.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = y.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (i, v) in y.iter_mut().enumerate() {
            *v = ((f64::from(*v) - mean) * inv) as f32 * self.spk_ln_w[i] + self.spk_ln_b[i];
        }
        y
    }

    /// The input embeddings for `ids`, widened from bf16.
    pub fn embed_tokens(&self, ids: &[u32], out: &mut Vec<f32>) {
        let n = self.hp.n_embd;
        for &id in ids {
            let row = &self.embed[id as usize * n..(id as usize + 1) * n];
            out.extend(row.iter().map(|&b| bf16_to_f32(b)));
        }
    }
}

/// One generation's state: the KV cache, the backend and the graph allocator.
pub struct Session<'m> {
    model: &'m Model,
    kv_ctx: *mut g::ggml_context,
    k: Vec<*mut g::ggml_tensor>,
    v: Vec<*mut g::ggml_tensor>,
    pub n_ctx: usize,
    pub n_past: usize,
    backend: g::ggml_backend_t,
    galloc: g::ggml_gallocr_t,
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        unsafe {
            g::ggml_gallocr_free(self.galloc);
            g::ggml_backend_free(self.backend);
            g::ggml_free(self.kv_ctx);
        }
    }
}

const GRAPH_SIZE: usize = 8192;

/// What one forward pass returns.
pub struct Output {
    /// Logits of the last `n_logits` positions, `n_vocab` each.
    pub logits: Vec<f32>,
    /// With `hidden` requested: the residual after each layer (`n_layer` of `[T, n_embd]`), then the final norm's
    /// output for the logit positions.
    pub hidden: Vec<Vec<f32>>,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model, n_ctx: usize, threads: usize) -> Result<Self, String> {
        let hp = &model.hp;
        let per = hp.head_dim * hp.n_head_kv * n_ctx * 4;
        let mem = hp.n_layer * 2 * (per + unsafe { g::ggml_tensor_overhead() }) + (1 << 20);
        unsafe {
            let kv_ctx = g::ggml_init(g::ggml_init_params {
                mem_size: mem,
                mem_buffer: ptr::null_mut(),
                no_alloc: false,
            });
            if kv_ctx.is_null() {
                return Err("ggml_init failed for the KV cache".into());
            }
            let mut k = Vec::new();
            let mut v = Vec::new();
            for _ in 0..hp.n_layer {
                let shape = (hp.head_dim as i64, hp.n_head_kv as i64, n_ctx as i64);
                k.push(g::ggml_new_tensor_3d(
                    kv_ctx,
                    g::GGML_TYPE_F32,
                    shape.0,
                    shape.1,
                    shape.2,
                ));
                v.push(g::ggml_new_tensor_3d(
                    kv_ctx,
                    g::GGML_TYPE_F32,
                    shape.0,
                    shape.1,
                    shape.2,
                ));
            }
            let backend = g::ggml_backend_cpu_init();
            if backend.is_null() {
                g::ggml_free(kv_ctx);
                return Err("ggml_backend_cpu_init failed".into());
            }
            g::ggml_backend_cpu_set_n_threads(backend, threads as i32);
            let galloc = g::ggml_gallocr_new(g::ggml_backend_cpu_buffer_type());
            Ok(Session {
                model,
                kv_ctx,
                k,
                v,
                n_ctx,
                n_past: 0,
                backend,
                galloc,
            })
        }
    }

    /// Run `embd` (`T` rows of `n_embd`) at positions `n_past..n_past + T`, append to the cache, and return the
    /// logits of the last `n_logits` positions.
    pub fn forward(&mut self, embd: &[f32], n_logits: usize, hidden: bool) -> Result<Output, String> {
        let m = self.model;
        let hp = &m.hp;
        let t = embd.len() / hp.n_embd;
        if t == 0 || t * hp.n_embd != embd.len() || n_logits == 0 || n_logits > t {
            return Err("bad forward shape".into());
        }
        if self.n_past + t > self.n_ctx {
            return Err(format!("context full: {} + {t} > {}", self.n_past, self.n_ctx));
        }
        let pos0 = self.n_past;
        let n_kv = pos0 + t;
        let (hd, nh, nkv) = (hp.head_dim as i64, hp.n_head as i64, hp.n_head_kv as i64);
        let ti = t as i64;
        let padded = m.weights != Weights::F32;
        unsafe {
            let size = g::ggml_tensor_overhead() * GRAPH_SIZE + g::ggml_graph_overhead_custom(GRAPH_SIZE, false);
            let ctx = g::ggml_init(g::ggml_init_params {
                mem_size: size,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            });
            if ctx.is_null() {
                return Err("ggml_init failed for the graph".into());
            }
            let gf = g::ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

            let inp = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, hp.n_embd as i64, ti);
            g::ggml_set_input(inp);
            let pos = g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_I32, ti);
            g::ggml_set_input(pos);
            let mask = if t > 1 {
                let x = g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, n_kv as i64, ti);
                g::ggml_set_input(x);
                x
            } else {
                ptr::null_mut()
            };

            // A ternary matrix product: pad each 128-group of the input with 128 zeros first.
            let mm = |w: *mut g::ggml_tensor, x: *mut g::ggml_tensor, n_in: i64| -> *mut g::ggml_tensor {
                if padded {
                    let x3 = g::ggml_reshape_3d(ctx, x, GROUP as i64, n_in / GROUP as i64, ti);
                    let xp = g::ggml_pad(ctx, x3, GROUP as i32, 0, 0, 0);
                    let x2 = g::ggml_reshape_2d(ctx, xp, 2 * n_in, ti);
                    g::ggml_mul_mat(ctx, w, x2)
                } else {
                    g::ggml_mul_mat(ctx, w, x)
                }
            };

            let mut x = inp;
            let mut outs = Vec::new();
            let scale = 1.0 / (hp.head_dim as f32).sqrt();
            for (l, ly) in m.layers.iter().enumerate() {
                let h = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, x, hp.eps), ly.attn_norm);
                let ne = hp.n_embd as i64;
                let q = g::ggml_reshape_3d(ctx, mm(ly.q, h, ne), hd, nh, ti);
                let k = g::ggml_reshape_3d(ctx, mm(ly.k, h, ne), hd, nkv, ti);
                let v = g::ggml_reshape_3d(ctx, mm(ly.v, h, ne), hd, nkv, ti);
                let q = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, q, hp.eps), ly.q_norm);
                let k = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, k, hp.eps), ly.k_norm);
                let rope = |a: *mut g::ggml_tensor| {
                    g::ggml_rope_ext(
                        ctx,
                        a,
                        pos,
                        ptr::null_mut(),
                        hp.head_dim as i32,
                        ggtts_ggml_sys::rope_mode::NEOX,
                        0,
                        hp.rope_base,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    )
                };
                let q = rope(q);
                let k = rope(k);
                let ck = self.k[l];
                let cv = self.v[l];
                let (nb1, nb2) = ((*ck).nb[1], (*ck).nb[2]);
                let kdst = g::ggml_view_3d(ctx, ck, hd, nkv, ti, nb1, nb2, pos0 * nb2);
                let vdst = g::ggml_view_3d(ctx, cv, hd, nkv, ti, nb1, nb2, pos0 * nb2);
                g::ggml_build_forward_expand(gf, g::ggml_cpy(ctx, k, kdst));
                g::ggml_build_forward_expand(gf, g::ggml_cpy(ctx, v, vdst));
                let kall = g::ggml_view_3d(ctx, ck, hd, nkv, n_kv as i64, nb1, nb2, 0);
                let vall = g::ggml_view_3d(ctx, cv, hd, nkv, n_kv as i64, nb1, nb2, 0);
                let qp = g::ggml_permute(ctx, q, 0, 2, 1, 3); // [hd, T, nh]
                let kp = g::ggml_permute(ctx, kall, 0, 2, 1, 3); // [hd, n_kv, nkv]
                let kq = g::ggml_mul_mat(ctx, kp, qp); // [n_kv, T, nh]
                let kq = g::ggml_soft_max_ext(ctx, kq, mask, scale, 0.0);
                let vt = g::ggml_cont(ctx, g::ggml_permute(ctx, vall, 1, 2, 0, 3)); // [n_kv, hd, nkv]
                let kqv = g::ggml_mul_mat(ctx, vt, kq); // [hd, T, nh]
                let att = g::ggml_cont(ctx, g::ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [hd, nh, T]
                let att = g::ggml_reshape_2d(ctx, att, hd * nh, ti);
                x = g::ggml_add(ctx, x, mm(ly.o, att, hd * nh));
                let h = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, x, hp.eps), ly.ffn_norm);
                let gt = mm(ly.gate, h, ne);
                let up = mm(ly.up, h, ne);
                let a = g::ggml_mul(ctx, g::ggml_silu(ctx, gt), up);
                x = g::ggml_add(ctx, x, mm(ly.down, a, hp.n_ff as i64));
                if hidden {
                    g::ggml_set_output(x);
                    g::ggml_build_forward_expand(gf, x);
                    outs.push(x);
                }
            }
            let last = g::ggml_view_2d(
                ctx,
                x,
                hp.n_embd as i64,
                n_logits as i64,
                (*x).nb[1],
                (t - n_logits) * (*x).nb[1],
            );
            let last = g::ggml_cont(ctx, last);
            let normed = g::ggml_mul(ctx, g::ggml_rms_norm(ctx, last, hp.eps), m.norm);
            let logits = g::ggml_mul_mat(ctx, m.head, normed);
            g::ggml_set_output(logits);
            if hidden {
                g::ggml_set_output(normed);
                g::ggml_build_forward_expand(gf, normed);
                outs.push(normed);
            }
            g::ggml_build_forward_expand(gf, logits);

            if !g::ggml_gallocr_alloc_graph(self.galloc, gf) {
                g::ggml_free(ctx);
                return Err("ggml_gallocr_alloc_graph failed".into());
            }
            g::ggml_backend_tensor_set(inp, embd.as_ptr().cast(), 0, embd.len() * 4);
            let p: Vec<i32> = (pos0..pos0 + t).map(|p| p as i32).collect();
            g::ggml_backend_tensor_set(pos, p.as_ptr().cast(), 0, t * 4);
            if !mask.is_null() {
                let mut mk = vec![0f32; n_kv * t];
                for i in 0..t {
                    for j in 0..n_kv {
                        if j > pos0 + i {
                            mk[i * n_kv + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                g::ggml_backend_tensor_set(mask, mk.as_ptr().cast(), 0, mk.len() * 4);
            }
            let st = g::ggml_backend_graph_compute(self.backend, gf);
            if st != g::GGML_STATUS_SUCCESS {
                g::ggml_free(ctx);
                return Err(format!("graph compute failed: {st}"));
            }
            let read = |t: *mut g::ggml_tensor| -> Vec<f32> {
                let n = g::ggml_nbytes(t) / 4;
                let mut v = vec![0f32; n];
                g::ggml_backend_tensor_get(t, v.as_mut_ptr().cast(), 0, n * 4);
                v
            };
            let out = Output {
                logits: read(logits),
                hidden: outs.iter().map(|&t| read(t)).collect(),
            };
            g::ggml_free(ctx);
            self.n_past += t;
            Ok(out)
        }
    }
}
