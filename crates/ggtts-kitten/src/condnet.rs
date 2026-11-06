//! Reference-clip preparation's three networks as ggml graphs:
//!
//! * the S3 tokenizer v2 (`s3tokenizer/model_v2.py`: `AudioEncoderV2` with rotary, FSMN-augmented attention, and
//!   `FSQCodebook.encode`), which gives the clip's S3 tokens for the LM prompt and for S3Gen;
//! * CAMPPlus (`kittenml/s3gen.py` `CAMPPlus`, `FCM`, `CAMDenseTDNNBlock`), S3Gen's 192-dim x-vector;
//! * Kitten's pyannote x-vector (`kittenml/speaker_embedding.py` `XVectorSincNet`), the 512-dim embedding
//!   `spk_proj` reads.
//!
//! The signal processing in front of them is [`crate::refdsp`]. Signals are `[T, C]` (time fastest) for the
//! convolutional parts and `[C, T]` for the transformer, as in [`crate::flow`].

use std::f64::consts::PI;

use ggtts_ggml_sys as g;

use crate::gguf::Gguf;
use crate::hift::{Graph, Ops, Weights};
use crate::refdsp;

type T = *mut g::ggml_tensor;

/// Planted faults, for showing that each parity check can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plant {
    None,
    /// The S3 tokenizer without its rotary embedding.
    NoRope,
    /// CAMPPlus without its BatchNorm folding (scale 1, shift 0).
    NoBatchNorm,
    /// pyannote without the absolute value after the sinc filters.
    NoAbs,
    /// Kaldi fbank without pre-emphasis.
    NoPreemphasis,
}

pub struct CondNets {
    tok: Weights,
    cam: Weights,
    py: Weights,
    mel_filters: Vec<f32>,
    pub threads: usize,
    pub plant: Plant,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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
    /// A folded BatchNorm on `[T, C]` (or `[W, H, C]` with `dims3`).
    fn bn(&self, x: T, name: &str, dims3: bool, off: bool) -> Result<T, String> {
        if off {
            return Ok(x);
        }
        let s = self.w(&format!("{name}.bn_scale"))?;
        let b = self.w(&format!("{name}.bn_shift"))?;
        unsafe {
            let n = (*s).ne[0];
            let (s, b) = if dims3 {
                (
                    g::ggml_reshape_3d(self.c(), s, 1, 1, n),
                    g::ggml_reshape_3d(self.c(), b, 1, 1, n),
                )
            } else {
                (
                    g::ggml_reshape_2d(self.c(), s, 1, n),
                    g::ggml_reshape_2d(self.c(), b, 1, n),
                )
            };
            Ok(g::ggml_add(self.c(), g::ggml_mul(self.c(), x, s), b))
        }
    }
    /// `nn.Conv2d` on `[W, H, C]` with stride `(sh, sw)` and padding `p` on both axes; kernel `[KW, KH, IC, OC]`.
    fn conv2d(&self, x: T, name: &str, sh: i32, sw: i32, p: i32) -> Result<T, String> {
        let w = self.w(&format!("{name}.weight"))?;
        unsafe {
            let col = g::ggml_im2col(self.c(), w, x, sw, sh, p, p, 1, 1, true, g::GGML_TYPE_F32);
            let (k, ow, oh) = ((*col).ne[0], (*col).ne[1], (*col).ne[2]);
            let col2 = g::ggml_reshape_2d(self.c(), col, k, ow * oh);
            let w2 = g::ggml_reshape_2d(self.c(), w, (*w).ne[0] * (*w).ne[1] * (*w).ne[2], (*w).ne[3]);
            let y = g::ggml_mul_mat(self.c(), col2, w2); // [OW*OH, OC]
            Ok(g::ggml_reshape_3d(self.c(), y, ow, oh, (*w).ne[3]))
        }
    }
    /// Depthwise `nn.Conv1d` (`groups = C`) on `[T, C]`, kernel `[K, 1, C]`, f32 throughout.
    fn conv_dw(&self, x: T, w: T, pad: i32) -> T {
        unsafe {
            let c = self.c();
            let xb = g::ggml_reshape_4d(c, x, (*x).ne[0], 1, (*x).ne[1], 1);
            let col = g::ggml_im2col(c, w, xb, 1, 0, pad, 0, 1, 0, false, g::GGML_TYPE_F32); // [K, OW, C]
            let y = g::ggml_mul_mat(c, col, w); // [OW, 1, C]
            g::ggml_reshape_2d(c, y, (*y).ne[0], (*y).ne[2])
        }
    }
    /// Mean and unbiased standard deviation over time of `[T, C]`, concatenated: `[2C]`.
    fn stats(&self, x: T) -> T {
        unsafe {
            let c = self.c();
            let t = (*x).ne[0] as f32;
            let mean = g::ggml_mean(c, x); // [1, C]
            let d = g::ggml_sub(c, x, mean);
            let var = g::ggml_scale(c, g::ggml_mean(c, g::ggml_sqr(c, d)), t / (t - 1.0));
            let std = g::ggml_sqrt(c, var);
            let mean = g::ggml_reshape_1d(c, mean, (*mean).ne[1]);
            let std = g::ggml_reshape_1d(c, std, (*std).ne[1]);
            g::ggml_concat(c, mean, std, 0)
        }
    }
}

impl CondNets {
    pub fn load(gguf: &Gguf, threads: usize) -> Result<Self, String> {
        Ok(CondNets {
            tok: Weights::load(gguf, "tokenizer.")?,
            cam: Weights::load(gguf, "speaker_encoder.")?,
            py: Weights::load(gguf, "pyannote.")?,
            mel_filters: f32s(gguf.tensor("tokenizer._mel_filters")?.1),
            threads,
            plant: Plant::None,
        })
    }

    /// The S3 tokenizer's log-mel of 16 kHz audio, `[128, frames]`.
    pub fn whisper_mel(&self, wav16: &[f32]) -> Vec<f32> {
        refdsp::whisper_mel(wav16, &self.mel_filters)
    }

    /// The S3 tokenizer on 16 kHz audio: one token per 40 ms.
    pub fn s3_tokens(&self, wav16: &[f32]) -> Result<Vec<u32>, String> {
        let mel = refdsp::whisper_mel(wav16, &self.mel_filters);
        let t = mel.len() / 128;
        let mut gr = Graph::new(self.threads)?;
        let b = B {
            o: Ops {
                ctx: gr.ctx,
                w: &self.tok,
            },
        };
        let c = b.c();
        let x0 = gr.input(&[t as i64, 128]);
        let t1 = (t + 2 - 3) / 2 + 1;
        let t2 = (t1 + 2 - 3) / 2 + 1;
        let pos = unsafe {
            let p = g::ggml_new_tensor_1d(c, g::GGML_TYPE_I32, t2 as i64);
            g::ggml_set_input(p);
            p
        };
        let (hd, nh, d) = (64i64, 20i64, 1280i64);
        let out = unsafe {
            let x = g::ggml_gelu_erf(c, b.o.conv_named(x0, "tokenizer.encoder.conv1", 2, 1, 1)?);
            let x = g::ggml_gelu_erf(c, b.o.conv_named(x, "tokenizer.encoder.conv2", 2, 1, 1)?);
            let mut x = b.tr(x); // [1280, T2]
            let tt = t2 as i64;
            for i in 0..6 {
                let p = format!("tokenizer.encoder.blocks.{i}");
                let h = b.ln(x, &format!("{p}.attn_ln"), 1e-5)?;
                let q = g::ggml_reshape_3d(c, b.lin(h, &format!("{p}.attn.query"), true)?, hd, nh, tt);
                let k = g::ggml_reshape_3d(c, b.lin(h, &format!("{p}.attn.key"), false)?, hd, nh, tt);
                let v = b.lin(h, &format!("{p}.attn.value"), true)?; // [1280, T]
                let rope = |a: T| {
                    if self.plant == Plant::NoRope {
                        return a;
                    }
                    g::ggml_rope_ext(
                        c,
                        a,
                        pos,
                        std::ptr::null_mut(),
                        64,
                        ggtts_ggml_sys::rope_mode::NEOX,
                        0,
                        10000.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    )
                };
                let (q, k) = (rope(q), rope(k));
                // FSMN memory: a depthwise conv (k 31, pad 15) over v, plus v.
                let vt = b.tr(v);
                let fsm = g::ggml_add(c, b.conv_dw(vt, b.w(&format!("{p}.attn.fsmn_block.weight"))?, 15), vt);
                let fsm = b.tr(fsm);
                let kq = g::ggml_mul_mat(c, g::ggml_permute(c, k, 0, 2, 1, 3), g::ggml_permute(c, q, 0, 2, 1, 3));
                let attn = g::ggml_soft_max_ext(c, kq, std::ptr::null_mut(), 1.0 / 8.0, 0.0);
                let v3 = g::ggml_reshape_3d(c, v, hd, nh, tt);
                let vt3 = g::ggml_cont(c, g::ggml_permute(c, v3, 1, 2, 0, 3));
                let o = g::ggml_mul_mat(c, vt3, attn);
                let o = g::ggml_reshape_2d(c, g::ggml_cont(c, g::ggml_permute(c, o, 0, 2, 1, 3)), d, tt);
                let o = g::ggml_add(c, b.lin(o, &format!("{p}.attn.out"), true)?, fsm);
                x = g::ggml_add(c, x, o);
                let h = b.ln(x, &format!("{p}.mlp_ln"), 1e-5)?;
                let f = g::ggml_gelu_erf(c, b.lin(h, &format!("{p}.mlp.0"), true)?);
                x = g::ggml_add(c, x, b.lin(f, &format!("{p}.mlp.2"), true)?);
            }
            b.lin(x, "tokenizer.quantizer._codebook.project_down", true)? // [8, T2]
        };
        let posv: Vec<i32> = (0..t2 as i32).collect();
        let h = gr.run_ints(&[(x0, &mel)], &[(pos, &posv)], &[out])?.remove(0);
        // FSQ: tanh, times 0.999, round half to even, plus 1, base 3.
        Ok(h.chunks_exact(8)
            .map(|v| {
                v.iter().enumerate().fold(0u32, |acc, (i, &x)| {
                    let q = (x.tanh() * 0.999).round_ties_even() + 1.0;
                    acc + q as u32 * 3u32.pow(i as u32)
                })
            })
            .collect())
    }

    /// CAMPPlus on Kaldi fbank features (`[frames, 80]`, mean-normalised): the 192-dim x-vector.
    pub fn xvector(&self, fbank: &[f32]) -> Result<Vec<f32>, String> {
        let t = fbank.len() / 80;
        // [W = T, H = 80] with time fastest, as PyTorch's (B, 1, F, T).
        let mut img = vec![0f32; fbank.len()];
        for f in 0..t {
            for m in 0..80 {
                img[m * t + f] = fbank[f * 80 + m];
            }
        }
        let off = self.plant == Plant::NoBatchNorm;
        let mut gr = Graph::new(self.threads)?;
        let b = B {
            o: Ops {
                ctx: gr.ctx,
                w: &self.cam,
            },
        };
        let c = b.c();
        let x0 = gr.input(&[t as i64, 80, 1]);
        let p = "speaker_encoder";
        // CAM's segment pooling: 1 / (segment length) per 100-frame segment of the stride-2 sequence.
        let t2 = (t + 4 - 5) / 2 + 1;
        let nseg = t2.div_ceil(100);
        let inv: Vec<f32> = (0..nseg).map(|s| 1.0 / (t2 - s * 100).min(100) as f32).collect();
        let invt = gr.input(&[1, nseg as i64, 1]);
        let out = unsafe {
            let relu = |x: T| g::ggml_relu(c, x);
            let block = |x: T, q: &str, stride: i32| -> Result<T, String> {
                let o = b.conv2d(x, &format!("{q}.conv1"), stride, 1, 1)?;
                let o = relu(b.bn(o, &format!("{q}.bn1"), true, off)?);
                let o = b.bn(
                    b.conv2d(o, &format!("{q}.conv2"), 1, 1, 1)?,
                    &format!("{q}.bn2"),
                    true,
                    off,
                )?;
                let sc = if stride != 1 {
                    let s = b.conv2d(x, &format!("{q}.shortcut.0"), stride, 1, 0)?;
                    b.bn(s, &format!("{q}.shortcut.1"), true, off)?
                } else {
                    x
                };
                Ok(relu(g::ggml_add(c, o, sc)))
            };
            let x = b.conv2d(x0, &format!("{p}.head.conv1"), 1, 1, 1)?;
            let mut x = relu(b.bn(x, &format!("{p}.head.bn1"), true, off)?);
            for l in ["layer1", "layer2"] {
                x = block(x, &format!("{p}.head.{l}.0"), 2)?;
                x = block(x, &format!("{p}.head.{l}.1"), 1)?;
            }
            let x = b.conv2d(x, &format!("{p}.head.conv2"), 2, 1, 1)?;
            let x = relu(b.bn(x, &format!("{p}.head.bn2"), true, off)?);
            let mut x = g::ggml_reshape_2d(c, x, (*x).ne[0], (*x).ne[1] * (*x).ne[2]); // [T, 320]
            let xv = format!("{p}.xvector");
            x = b.o.conv_named(x, &format!("{xv}.tdnn.linear"), 2, 2, 1)?;
            x = relu(b.bn(x, &format!("{xv}.tdnn.nonlinear.batchnorm"), false, off)?);
            for (bi, (layers, dil)) in [(12, 1), (24, 2), (16, 2)].into_iter().enumerate() {
                for j in 1..=layers {
                    let q = format!("{xv}.block{}.tdnnd{j}", bi + 1);
                    let h = relu(b.bn(x, &format!("{q}.nonlinear1.batchnorm"), false, off)?);
                    let h = b.o.conv_named(h, &format!("{q}.linear1"), 1, 0, 1)?;
                    let h = relu(b.bn(h, &format!("{q}.nonlinear2.batchnorm"), false, off)?);
                    let y = b.o.conv_named(h, &format!("{q}.cam_layer.linear_local"), 1, dil, dil)?;
                    // Context: the clip mean plus the mean of the 100-frame segment each frame sits in.
                    let tn = (*h).ne[0];
                    let ch = (*h).ne[1];
                    let tp = nseg as i64 * 100;
                    let hp = g::ggml_pad(c, h, (tp - tn) as i32, 0, 0, 0);
                    let r = g::ggml_reshape_3d(c, hp, 100, nseg as i64, ch);
                    let s = g::ggml_mul(c, g::ggml_sum_rows(c, r), invt);
                    let e = g::ggml_reshape_2d(c, g::ggml_repeat(c, s, r), tp, ch);
                    let seg = g::ggml_cont(c, g::ggml_view_2d(c, e, tn, ch, (*e).nb[1], 0));
                    let ctx = g::ggml_add(c, seg, g::ggml_mean(c, h));
                    let m = relu(b.o.conv_named(ctx, &format!("{q}.cam_layer.linear1"), 1, 0, 1)?);
                    let m = g::ggml_sigmoid(c, b.o.conv_named(m, &format!("{q}.cam_layer.linear2"), 1, 0, 1)?);
                    x = g::ggml_concat(c, x, g::ggml_mul(c, y, m), 1);
                }
                let q = format!("{xv}.transit{}", bi + 1);
                let h = relu(b.bn(x, &format!("{q}.nonlinear.batchnorm"), false, off)?);
                x = b.o.conv_named(h, &format!("{q}.linear"), 1, 0, 1)?;
            }
            x = relu(b.bn(x, &format!("{xv}.out_nonlinear.batchnorm"), false, off)?);
            let st = b.stats(x); // [1024]
            let st = g::ggml_reshape_2d(c, st, 1, (*st).ne[0]);
            let y = b.o.conv_named(st, &format!("{xv}.dense.linear"), 1, 0, 1)?; // [1, 192]
            b.bn(y, &format!("{xv}.dense.nonlinear.batchnorm"), false, false)?
        };
        Ok(gr.run(&[(x0, &img), (invt, &inv)], &[out])?.remove(0))
    }

    /// The SincNet filterbank's 80 filters of 251 taps (`ParamSincFilterbank.filters`), `[251, 1, 80]`.
    fn sinc_filters(&self) -> Result<Vec<f32>, String> {
        let lowp = self.py.host("pyannote.sincnet.conv1d.0.filterbank.low_hz_")?;
        let band = self.py.host("pyannote.sincnet.conv1d.0.filterbank.band_hz_")?;
        let (k, half, sr) = (251usize, 125usize, 16000.0f64);
        // np.hamming(251)[:125], and 2 pi arange(-125, 0) / sr.
        let win: Vec<f64> = (0..half)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (k - 1) as f64).cos())
            .collect();
        let n: Vec<f64> = (0..half).map(|i| 2.0 * PI * (i as f64 - half as f64) / sr).collect();
        let mut out = vec![0f32; 80 * k];
        for (kind, base) in [(0usize, 0usize), (1, 40)] {
            for f in 0..40 {
                let low = 50.0 + f64::from(lowp[f].abs());
                let high = (low + 50.0 + f64::from(band[f].abs())).clamp(50.0, sr / 2.0);
                let bw = high - low;
                let row = &mut out[(base + f) * k..(base + f + 1) * k];
                for i in 0..half {
                    let v = if kind == 0 {
                        (high * n[i]).sin() - (low * n[i]).sin()
                    } else {
                        (low * n[i]).cos() - (high * n[i]).cos()
                    };
                    let left = v / (n[i] / 2.0) * win[i];
                    row[i] = (left / (2.0 * bw)) as f32;
                    let right = if kind == 0 { left } else { -left };
                    row[k - 1 - i] = (right / (2.0 * bw)) as f32;
                }
                row[half] = if kind == 0 { 1.0 } else { 0.0 };
            }
        }
        Ok(out)
    }

    /// Kitten's pyannote x-vector on 16 kHz audio, before the unit normalisation `SpeakerEncoder` applies.
    pub fn pyannote(&self, wav16: &[f32]) -> Result<Vec<f32>, String> {
        let n = wav16.len() as f64;
        let mean = wav16.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = wav16.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n;
        let w0 = f64::from(self.py.host("pyannote.sincnet.wav_norm1d.weight")?[0]);
        let b0 = f64::from(self.py.host("pyannote.sincnet.wav_norm1d.bias")?[0]);
        let x: Vec<f32> = wav16
            .iter()
            .map(|&v| ((f64::from(v) - mean) / (var + 1e-5).sqrt() * w0 + b0) as f32)
            .collect();
        let filters = self.sinc_filters()?;
        let mut gr = Graph::new(self.threads)?;
        let b = B {
            o: Ops {
                ctx: gr.ctx,
                w: &self.py,
            },
        };
        let c = b.c();
        let x0 = gr.input(&[x.len() as i64, 1]);
        let fw = gr.input(&[251, 1, 80]);
        let out = unsafe {
            let inorm = |x: T, i: usize| -> Result<T, String> {
                let y = g::ggml_norm(c, x, 1e-5);
                let ch = (*y).ne[1];
                let w = g::ggml_reshape_2d(c, b.w(&format!("pyannote.sincnet.norm1d.{i}.weight"))?, 1, ch);
                let bb = g::ggml_reshape_2d(c, b.w(&format!("pyannote.sincnet.norm1d.{i}.bias"))?, 1, ch);
                Ok(g::ggml_add(c, g::ggml_mul(c, y, w), bb))
            };
            let pool = |x: T| g::ggml_pool_1d(c, x, g::GGML_OP_POOL_MAX, 3, 3, 0);
            let leaky = |x: T| g::ggml_leaky_relu(c, x, 0.01, false);
            let mut y = b.o.conv(x0, fw, None, 10, 0, 1);
            if self.plant != Plant::NoAbs {
                y = g::ggml_abs(c, y);
            }
            y = leaky(inorm(pool(y), 0)?);
            for i in 1..3 {
                y = b.o.conv_named(y, &format!("pyannote.sincnet.conv1d.{i}"), 1, 0, 1)?;
                y = leaky(inorm(pool(y), i)?);
            }
            for (ci, bi, dil) in [(0, 2, 1), (3, 5, 2), (6, 8, 3), (9, 11, 1), (12, 14, 1)] {
                y = leaky(b.o.conv_named(y, &format!("pyannote.tdnns.{ci}"), 1, 0, dil)?);
                y = b.bn(y, &format!("pyannote.tdnns.{bi}"), false, false)?;
            }
            let st = b.stats(y); // [3000]
            b.lin(st, "pyannote.embedding", true)?
        };
        Ok(gr.run(&[(x0, &x), (fw, &filters)], &[out])?.remove(0))
    }

    /// The whole preparation of one clip (`KittenTTS2._clone_reference` with `S3Codec.reference_conditioning`):
    /// the clip at 24 kHz (first 10 s, tiled to at least 6 s) for S3Gen's mel, and at 16 kHz for its tokens and
    /// x-vector; the whole clip at 16 kHz for the LM prompt's tokens and the pyannote embedding (unit norm).
    /// Resampling is torchaudio's sinc throughout, where Python uses soxr (through librosa) for the 24 kHz load and
    /// the LM prompt's tokens; on a 24 kHz clip only the LM prompt's path differs.
    pub fn prepare(&self, wav: &[f32], sr: u32) -> Result<Prepared, String> {
        let mut w24 = refdsp::resample(wav, sr, 24000);
        if w24.is_empty() {
            return Err("empty reference clip".into());
        }
        let need = 6 * 24000;
        if w24.len() < need {
            let reps = need.div_ceil(w24.len());
            w24 = w24.repeat(reps);
        }
        w24.truncate(10 * 24000);
        let prompt_feat = refdsp::mel24(&w24);
        let w16 = refdsp::resample(&w24, 24000, 16000);
        let mut prompt_token = self.s3_tokens(&w16)?;
        prompt_token.truncate(prompt_feat.len() / 80 / 2);
        let fbank = refdsp::kaldi_fbank(&w16, self.plant == Plant::NoPreemphasis);
        let xvector = self.xvector(&fbank)?;
        let full16 = refdsp::resample(wav, sr, 16000);
        let reference_tokens = self.s3_tokens(&full16)?;
        let e = self.pyannote(&full16)?;
        let norm = e.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
        let embedding = e.iter().map(|&v| (f64::from(v) / norm) as f32).collect();
        Ok(Prepared {
            reference_tokens,
            embedding,
            prompt_token,
            prompt_feat,
            xvector,
        })
    }
}

/// The conditioning `tools/kitten_ref.py reference` writes, computed without Python.
pub struct Prepared {
    pub reference_tokens: Vec<u32>,
    /// pyannote, unit norm (for `spk_proj`).
    pub embedding: Vec<f32>,
    /// S3Gen's prompt tokens, its prompt mel (`[frames, 80]`) and its CAMPPlus x-vector.
    pub prompt_token: Vec<u32>,
    pub prompt_feat: Vec<f32>,
    pub xvector: Vec<f32>,
}
