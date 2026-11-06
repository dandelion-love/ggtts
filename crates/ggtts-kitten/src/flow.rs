//! S3Gen's flow, tokens to mel, as ggml graphs (`kittenml/s3gen.py`: `CausalMaskedDiffWithXvec.inference`,
//! `UpsampleConformerEncoder`, `RelPositionMultiHeadedAttention`, `CausalConditionalCFM.basic_euler`,
//! `ConditionalDecoder` with `BasicTransformerBlock`, `CausalResnetBlock1D`, `CausalBlock1D`).
//!
//! Batch 1 and no padding, so every mask in the PyTorch code is all ones and is left out. The meanflow decoder takes
//! two Euler steps, `t` = 0 then 0.5 to `r` = 0.5 then 1, without classifier-free guidance.
//!
//! Layouts: the residual streams are `[C, T]` (channels fastest, so a linear layer is one matrix product); a
//! convolution transposes to `[T, C]` and back (see [`crate::hift`]).

use ggtts_ggml_sys as g;

use crate::gguf::Gguf;
use crate::hift::{Graph, Ops, Weights};

type T = *mut g::ggml_tensor;

pub const MEL: usize = 80;
const D_ENC: i64 = 512;
const HEADS: i64 = 8;
const HEAD_DIM: i64 = 64;

pub struct Flow {
    w: Weights,
    pub threads: usize,
    /// `flow.input_embedding.weight`, `[6561, 512]` row-major, for the host-side lookup.
    tok_emb: Vec<f32>,
    /// `flow.spk_embed_affine_layer`: `[80, 192]` and `[80]`.
    spk_w: Vec<f32>,
    spk_b: Vec<f32>,
}

/// The conditioning one reference clip gives the flow (`S3Gen.embed_ref`).
pub struct FlowRef {
    pub prompt_token: Vec<u32>,
    /// `[frames, 80]`, frame-major, as PyTorch's `prompt_feat[0]`.
    pub prompt_feat: Vec<f32>,
    pub embedding: Vec<f32>,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `EspnetRelPositionalEncoding` for `t` frames: `2t - 1` rows of `d`, relative positions `t-1` down to `-(t-1)`;
/// even dims sin, odd dims cos. Column-major `[d, 2t-1]` for ggml.
pub fn rel_pos(t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; d * (2 * t - 1)];
    for m in 0..2 * t - 1 {
        let pos = (t as f32 - 1.0) - m as f32;
        for i in 0..d / 2 {
            let div = (-((2 * i) as f32) * (10000f32.ln() / d as f32)).exp();
            let a = pos * div;
            out[m * d + 2 * i] = a.sin();
            out[m * d + 2 * i + 1] = a.cos();
        }
    }
    out
}

/// `SinusoidalPosEmb(320)` of a time in [0, 1], scale 1000.
pub fn time_sin(t: f32) -> Vec<f32> {
    let half = 160usize;
    let e = 10000f32.ln() / (half as f32 - 1.0);
    let a: Vec<f32> = (0..half).map(|k| 1000.0 * t * (-(k as f32) * e).exp()).collect();
    a.iter().map(|x| x.sin()).chain(a.iter().map(|x| x.cos())).collect()
}

struct B<'a> {
    o: Ops<'a>,
}

impl B<'_> {
    fn c(&self) -> *mut g::ggml_context {
        self.o.ctx
    }
    fn w(&self, n: &str) -> Result<T, String> {
        self.o.w.t(n)
    }
    fn tr(&self, x: T) -> T {
        unsafe { g::ggml_cont(self.c(), g::ggml_transpose(self.c(), x)) }
    }
    /// A linear layer on `[in, T]`.
    fn lin(&self, x: T, name: &str, bias: bool) -> Result<T, String> {
        let y = unsafe { g::ggml_mul_mat(self.c(), self.w(&format!("{name}.weight"))?, x) };
        Ok(if bias {
            unsafe { g::ggml_add(self.c(), y, self.w(&format!("{name}.bias"))?) }
        } else {
            y
        })
    }
    fn ln(&self, x: T, name: &str, eps: f32) -> Result<T, String> {
        unsafe {
            let y = g::ggml_norm(self.c(), x, eps);
            let y = g::ggml_mul(self.c(), y, self.w(&format!("{name}.weight"))?);
            Ok(g::ggml_add(self.c(), y, self.w(&format!("{name}.bias"))?))
        }
    }
    fn mish(&self, x: T) -> T {
        unsafe { g::ggml_mul(self.c(), x, g::ggml_tanh(self.c(), g::ggml_softplus(self.c(), x))) }
    }
    /// A convolution on `[T, C]` padded by `pad` on both sides, keeping outputs `[from, from + len)`.
    fn conv_window(&self, x: T, name: &str, pad: i32, from: i64, len: i64) -> Result<T, String> {
        let y = self.o.conv_named(x, name, 1, pad, 1)?;
        unsafe {
            let v = g::ggml_view_2d(self.c(), y, len, (*y).ne[1], (*y).nb[1], from as usize * 4);
            Ok(g::ggml_cont(self.c(), v))
        }
    }
    /// `CausalConv1d` on a `[C, T]` stream: left padding `k - 1`.
    fn causal_conv(&self, x: T, name: &str, k: i32) -> Result<T, String> {
        let t = unsafe { (*x).ne[1] };
        let y = self.conv_window(self.tr(x), name, k - 1, 0, t)?;
        Ok(self.tr(y))
    }

    /// `RelPositionMultiHeadedAttention` plus the conformer layer around it (no macaron, no conv module).
    fn conformer(&self, x: T, pos: T, p: &str) -> Result<T, String> {
        let c = self.c();
        let t = unsafe { (*x).ne[1] };
        unsafe {
            let h = self.ln(x, &format!("{p}.norm_mha"), 1e-12)?;
            let q = g::ggml_reshape_3d(
                c,
                self.lin(h, &format!("{p}.self_attn.linear_q"), true)?,
                HEAD_DIM,
                HEADS,
                t,
            );
            let k = g::ggml_reshape_3d(
                c,
                self.lin(h, &format!("{p}.self_attn.linear_k"), true)?,
                HEAD_DIM,
                HEADS,
                t,
            );
            let v = g::ggml_reshape_3d(
                c,
                self.lin(h, &format!("{p}.self_attn.linear_v"), true)?,
                HEAD_DIM,
                HEADS,
                t,
            );
            let pe = g::ggml_mul_mat(c, self.w(&format!("{p}.self_attn.linear_pos.weight"))?, pos);
            let pe = g::ggml_reshape_3d(c, pe, HEAD_DIM, HEADS, 2 * t - 1);
            let bu = g::ggml_reshape_3d(c, self.w(&format!("{p}.self_attn.pos_bias_u"))?, HEAD_DIM, HEADS, 1);
            let bv = g::ggml_reshape_3d(c, self.w(&format!("{p}.self_attn.pos_bias_v"))?, HEAD_DIM, HEADS, 1);
            let qu = g::ggml_permute(c, g::ggml_add(c, q, bu), 0, 2, 1, 3); // [64, T, 8]
            let qv = g::ggml_permute(c, g::ggml_add(c, q, bv), 0, 2, 1, 3);
            let kp = g::ggml_permute(c, k, 0, 2, 1, 3);
            let pp = g::ggml_permute(c, pe, 0, 2, 1, 3); // [64, 2T-1, 8]
            let ac = g::ggml_mul_mat(c, kp, qu); // [T, T, 8]
            let bd = g::ggml_mul_mat(c, pp, qv); // [2T-1, T, 8]
                                                 // rel_shift: row i keeps columns T-1-i .. 2T-1-i.
            let bd = g::ggml_view_3d(
                c,
                bd,
                t,
                t,
                HEADS,
                (2 * t - 2) as usize * 4,
                (*bd).nb[2],
                (t - 1) as usize * 4,
            );
            let scores = g::ggml_add(c, ac, g::ggml_cont(c, bd));
            let attn = g::ggml_soft_max_ext(c, scores, std::ptr::null_mut(), 1.0 / (HEAD_DIM as f32).sqrt(), 0.0);
            let vt = g::ggml_cont(c, g::ggml_permute(c, v, 1, 2, 0, 3)); // [T, 64, 8]
            let o = g::ggml_mul_mat(c, vt, attn); // [64, T, 8]
            let o = g::ggml_reshape_2d(c, g::ggml_cont(c, g::ggml_permute(c, o, 0, 2, 1, 3)), D_ENC, t);
            let x = g::ggml_add(c, x, self.lin(o, &format!("{p}.self_attn.linear_out"), true)?);
            let h = self.ln(x, &format!("{p}.norm_ff"), 1e-12)?;
            let f = g::ggml_silu(c, self.lin(h, &format!("{p}.feed_forward.w_1"), true)?);
            Ok(g::ggml_add(c, x, self.lin(f, &format!("{p}.feed_forward.w_2"), true)?))
        }
    }

    /// `BasicTransformerBlock`: self-attention (8 heads of 64, no q/k/v bias) and a GELU feed-forward.
    fn transformer(&self, x: T, p: &str) -> Result<T, String> {
        let c = self.c();
        let t = unsafe { (*x).ne[1] };
        unsafe {
            let h = self.ln(x, &format!("{p}.norm1"), 1e-5)?;
            let q = g::ggml_reshape_3d(c, self.lin(h, &format!("{p}.attn1.to_q"), false)?, HEAD_DIM, HEADS, t);
            let k = g::ggml_reshape_3d(c, self.lin(h, &format!("{p}.attn1.to_k"), false)?, HEAD_DIM, HEADS, t);
            let v = g::ggml_reshape_3d(c, self.lin(h, &format!("{p}.attn1.to_v"), false)?, HEAD_DIM, HEADS, t);
            let kq = g::ggml_mul_mat(c, g::ggml_permute(c, k, 0, 2, 1, 3), g::ggml_permute(c, q, 0, 2, 1, 3));
            let attn = g::ggml_soft_max_ext(c, kq, std::ptr::null_mut(), 1.0 / (HEAD_DIM as f32).sqrt(), 0.0);
            let vt = g::ggml_cont(c, g::ggml_permute(c, v, 1, 2, 0, 3));
            let o = g::ggml_mul_mat(c, vt, attn);
            let o = g::ggml_reshape_2d(
                c,
                g::ggml_cont(c, g::ggml_permute(c, o, 0, 2, 1, 3)),
                HEAD_DIM * HEADS,
                t,
            );
            let x = g::ggml_add(c, x, self.lin(o, &format!("{p}.attn1.to_out.0"), true)?);
            let h = self.ln(x, &format!("{p}.norm3"), 1e-5)?;
            let f = g::ggml_gelu_erf(c, self.lin(h, &format!("{p}.ff.net.0.proj"), true)?);
            Ok(g::ggml_add(c, x, self.lin(f, &format!("{p}.ff.net.2"), true)?))
        }
    }

    /// `CausalBlock1D`: causal conv k3, LayerNorm over channels, Mish.
    fn causal_block(&self, x: T, p: &str) -> Result<T, String> {
        let y = self.causal_conv(x, &format!("{p}.block.0"), 3)?;
        let y = self.ln(y, &format!("{p}.block.2"), 1e-5)?;
        Ok(self.mish(y))
    }

    /// `CausalResnetBlock1D` with its time embedding `temb` (`[1024]`).
    fn resnet(&self, x: T, temb: T, p: &str) -> Result<T, String> {
        let h = self.causal_block(x, &format!("{p}.block1"))?;
        let m = self.lin(self.mish(temb), &format!("{p}.mlp.1"), true)?; // [256]
        let h = unsafe { g::ggml_add(self.c(), h, m) };
        let h = self.causal_block(h, &format!("{p}.block2"))?;
        let t = unsafe { (*x).ne[1] };
        let r = self.tr(self.conv_window(self.tr(x), &format!("{p}.res_conv"), 0, 0, t)?);
        Ok(unsafe { g::ggml_add(self.c(), h, r) })
    }
}

impl Flow {
    pub fn load(gguf: &Gguf, threads: usize) -> Result<Self, String> {
        let w = Weights::load(gguf, "flow.")?;
        Ok(Flow {
            w,
            threads,
            tok_emb: f32s(gguf.tensor("flow.input_embedding.weight")?.1),
            spk_w: f32s(gguf.tensor("flow.spk_embed_affine_layer.weight")?.1),
            spk_b: f32s(gguf.tensor("flow.spk_embed_affine_layer.bias")?.1),
        })
    }

    /// `F.normalize` then `spk_embed_affine_layer`: 192 to 80.
    pub fn spks(&self, emb: &[f32]) -> Vec<f32> {
        let n = emb
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        let e: Vec<f64> = emb.iter().map(|&v| f64::from(v) / n).collect();
        (0..MEL)
            .map(|o| {
                let row = &self.spk_w[o * e.len()..(o + 1) * e.len()];
                (f64::from(self.spk_b[o]) + row.iter().zip(&e).map(|(w, x)| f64::from(*w) * x).sum::<f64>()) as f32
            })
            .collect()
    }

    /// The encoder and `encoder_proj`: tokens (prompt and target) to `mu`, `[80, 2N]` channel-fastest, and the
    /// encoder output `[512, 2N]`.
    pub fn encode(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n = tokens.len() as i64;
        let mut emb = Vec::with_capacity(tokens.len() * 512);
        for &t in tokens {
            let t = t as usize;
            if t >= 6561 {
                return Err(format!("flow token {t} out of range"));
            }
            emb.extend_from_slice(&self.tok_emb[t * 512..(t + 1) * 512]);
        }
        let mut gr = Graph::new(self.threads)?;
        let b = B {
            o: Ops {
                ctx: gr.ctx,
                w: &self.w,
            },
        };
        let c = b.c();
        let x0 = gr.input(&[D_ENC, n]);
        let pos1 = gr.input(&[D_ENC, 2 * n - 1]);
        let pos2 = gr.input(&[D_ENC, 4 * n - 1]);
        let scale = (D_ENC as f32).sqrt();
        let (enc, mu) = unsafe {
            let x = b.lin(x0, "flow.encoder.embed.out.0", true)?;
            let x = g::ggml_scale(c, b.ln(x, "flow.encoder.embed.out.1", 1e-5)?, scale);
            // PreLookaheadLayer: right pad 3, conv k4, leaky ReLU; left pad 2, conv k3; residual.
            let xt = b.tr(x);
            let y = b.conv_window(xt, "flow.encoder.pre_lookahead_layer.conv1", 3, 3, n)?;
            let y = g::ggml_leaky_relu(c, y, 0.01, false);
            let y = b.conv_window(y, "flow.encoder.pre_lookahead_layer.conv2", 2, 0, n)?;
            let mut x = b.tr(g::ggml_add(c, xt, y));
            for i in 0..6 {
                x = b.conformer(x, pos1, &format!("flow.encoder.encoders.{i}"))?;
            }
            // Upsample1D: nearest x2, left pad 4, conv k5.
            let up = g::ggml_upscale_ext(
                c,
                b.tr(x),
                (2 * n) as i32,
                D_ENC as i32,
                1,
                1,
                g::GGML_SCALE_MODE_NEAREST,
            );
            let up = b.conv_window(up, "flow.encoder.up_layer.conv", 4, 0, 2 * n)?;
            let x = b.lin(b.tr(up), "flow.encoder.up_embed.out.0", true)?;
            let mut x = g::ggml_scale(c, b.ln(x, "flow.encoder.up_embed.out.1", 1e-5)?, scale);
            for i in 0..4 {
                x = b.conformer(x, pos2, &format!("flow.encoder.up_encoders.{i}"))?;
            }
            let enc = b.ln(x, "flow.encoder.after_norm", 1e-5)?;
            let mu = b.lin(enc, "flow.encoder_proj", true)?;
            (enc, mu)
        };
        let p1 = rel_pos(n as usize, 512);
        let p2 = rel_pos(2 * n as usize, 512);
        let mut out = gr.run(&[(x0, &emb), (pos1, &p1), (pos2, &p2)], &[mu, enc])?;
        let enc = out.pop().unwrap();
        Ok((out.pop().unwrap(), enc))
    }

    /// One call of the estimator (`ConditionalDecoder.forward`, meanflow): `inp` is `[T, 320]` channel-major
    /// (x, mu, spks, cond), returns `dx/dt` as `[T, 80]` channel-major.
    pub fn estimate(&self, inp: &[f32], t_len: usize, t: f32, r: f32) -> Result<Vec<f32>, String> {
        let p = "flow.decoder.estimator";
        let mut gr = Graph::new(self.threads)?;
        let b = B {
            o: Ops {
                ctx: gr.ctx,
                w: &self.w,
            },
        };
        let c = b.c();
        let x0 = gr.input(&[t_len as i64, 320]);
        let ts = gr.input(&[320]);
        let rs = gr.input(&[320]);
        let out = unsafe {
            let mlp = |s: T| -> Result<T, String> {
                let h = g::ggml_silu(c, b.lin(s, &format!("{p}.time_mlp.linear_1"), true)?);
                b.lin(h, &format!("{p}.time_mlp.linear_2"), true)
            };
            let te = g::ggml_concat(c, mlp(ts)?, mlp(rs)?, 0); // [2048]
            let temb = b.lin(te, &format!("{p}.time_embed_mixer"), false)?; // [1024]
            let mut x = b.tr(x0); // [320, T]
            x = b.resnet(x, temb, &format!("{p}.down_blocks.0.0"))?;
            for j in 0..4 {
                x = b.transformer(x, &format!("{p}.down_blocks.0.1.{j}"))?;
            }
            let skip = x;
            x = b.causal_conv(x, &format!("{p}.down_blocks.0.2"), 3)?;
            for i in 0..12 {
                x = b.resnet(x, temb, &format!("{p}.mid_blocks.{i}.0"))?;
                for j in 0..4 {
                    x = b.transformer(x, &format!("{p}.mid_blocks.{i}.1.{j}"))?;
                }
            }
            x = g::ggml_concat(c, x, skip, 0); // [512, T]
            x = b.resnet(x, temb, &format!("{p}.up_blocks.0.0"))?;
            for j in 0..4 {
                x = b.transformer(x, &format!("{p}.up_blocks.0.1.{j}"))?;
            }
            x = b.causal_conv(x, &format!("{p}.up_blocks.0.2"), 3)?;
            x = b.causal_block(x, &format!("{p}.final_block"))?;
            b.conv_window(b.tr(x), &format!("{p}.final_proj"), 0, 0, t_len as i64)?
            // [T, 80]
        };
        Ok(gr
            .run(&[(x0, inp), (ts, &time_sin(t)), (rs, &time_sin(r))], &[out])?
            .remove(0))
    }

    /// `CausalMaskedDiffWithXvec.inference` with `finalize=True`: target tokens and the reference to the target's
    /// mel, `[mel_frames, 80]`... returned channel-major (`80` rows of `frames`), with `z` the initial noise over
    /// the prompt and the target (`[2N, 80]` channel-major, as PyTorch's `[1, 80, 2N]`).
    pub fn mel(&self, tokens: &[u32], r: &FlowRef, z: &[f32]) -> Result<(Vec<f32>, usize), String> {
        let mut all = r.prompt_token.clone();
        all.extend_from_slice(tokens);
        let (mu, _) = self.encode(&all)?;
        let tn = 2 * all.len();
        let p_frames = r.prompt_feat.len() / MEL;
        if p_frames > tn || z.len() != MEL * tn {
            return Err("prompt mel longer than the flow's output, or noise of the wrong size".into());
        }
        let spks = self.spks(&r.embedding);
        // Channel-major [320][T]: x (filled per step), mu, spks, cond.
        let mut inp = vec![0f32; 320 * tn];
        for ch in 0..MEL {
            for f in 0..tn {
                inp[(MEL + ch) * tn + f] = mu[f * MEL + ch];
                inp[(2 * MEL + ch) * tn + f] = spks[ch];
                if f < p_frames {
                    inp[(3 * MEL + ch) * tn + f] = r.prompt_feat[f * MEL + ch];
                }
            }
        }
        let mut x = z.to_vec();
        for (t, rr) in [(0.0f32, 0.5f32), (0.5, 1.0)] {
            inp[..MEL * tn].copy_from_slice(&x);
            let d = self.estimate(&inp, tn, t, rr)?;
            for (xi, di) in x.iter_mut().zip(&d) {
                *xi += (rr - t) * di;
            }
        }
        let out_frames = tn - p_frames;
        let mut mel = Vec::with_capacity(MEL * out_frames);
        for ch in 0..MEL {
            mel.extend_from_slice(&x[ch * tn + p_frames..(ch + 1) * tn]);
        }
        Ok((mel, out_frames))
    }
}
