//! S3Gen's HiFT vocoder (`kittenml/s3gen.py` `HiFTGenerator`, `ConvRNNF0Predictor`, `ResBlock`, `Snake`) as ggml
//! graphs: mel to f0, then mel and the source's STFT to the inverse-STFT's magnitude and phase. The source, the
//! STFT and the inverse STFT are host-side ([`crate::dsp`]).
//!
//! Layout: a signal is `[T, C]` in ggml (time fastest), which is PyTorch's `[1, C, T]` memory exactly, and a
//! convolution kernel is ggml `[K, Cin, Cout]`, PyTorch's `[Cout, Cin, K]`. Convolutions run as an f32 `im2col`
//! and a matrix product (`ggml_conv_1d` would round the patches to f16).

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::rc::Rc;

use ggtts_ggml_sys as g;

use crate::backend::{Backend, GRAPH_SIZE};
use crate::dsp;
use crate::gguf::{GgmlType, Gguf};

type T = *mut g::ggml_tensor;

/// Weights of one GGUF in a device buffer, by name, with host copies of the small ones.
pub struct Weights {
    ctx: *mut g::ggml_context,
    buf: g::ggml_backend_buffer_t,
    map: HashMap<String, T>,
    host: HashMap<String, Vec<f32>>,
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe {
            if !self.buf.is_null() {
                g::ggml_backend_buffer_free(self.buf);
            }
            g::ggml_free(self.ctx);
        }
    }
}

impl Weights {
    pub fn load(gguf: &Gguf, prefix: &str, be: &Backend) -> Result<Self, String> {
        let mut items = Vec::new();
        for n in gguf.order.iter().filter(|n| n.starts_with(prefix)) {
            let (t, data) = gguf.tensor(n)?;
            if t.ty != GgmlType::F32 {
                return Err(format!("{n}: expected f32"));
            }
            let d: Vec<i64> = t.dims.iter().map(|&x| x as i64).collect();
            items.push((n.clone(), d, data));
        }
        Self::from_bytes(be, &items)
    }

    /// Small f32 tensors computed on the host (for example the inverse of every Snake alpha), one dimension each.
    pub fn from_host(be: &Backend, items: &[(String, Vec<f32>)]) -> Result<Self, String> {
        let bytes: Vec<(String, Vec<i64>, Vec<u8>)> = items
            .iter()
            .map(|(n, v)| {
                (
                    n.clone(),
                    vec![v.len() as i64],
                    v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                )
            })
            .collect();
        let refs: Vec<(String, Vec<i64>, &[u8])> = bytes
            .iter()
            .map(|(n, d, b)| (n.clone(), d.clone(), b.as_slice()))
            .collect();
        Self::from_bytes(be, &refs)
    }

    fn from_bytes(be: &Backend, items: &[(String, Vec<i64>, &[u8])]) -> Result<Self, String> {
        let mem = (items.len() + 1) * unsafe { g::ggml_tensor_overhead() };
        let ctx = unsafe {
            g::ggml_init(g::ggml_init_params {
                mem_size: mem,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            })
        };
        if ctx.is_null() {
            return Err("ggml_init failed for the decoder weights".into());
        }
        let mut map = HashMap::new();
        let mut host = HashMap::new();
        for (n, d, data) in items {
            let x = unsafe {
                match d.len() {
                    1 => g::ggml_new_tensor_1d(ctx, g::GGML_TYPE_F32, d[0]),
                    2 => g::ggml_new_tensor_2d(ctx, g::GGML_TYPE_F32, d[0], d[1]),
                    3 => g::ggml_new_tensor_3d(ctx, g::GGML_TYPE_F32, d[0], d[1], d[2]),
                    _ => g::ggml_new_tensor_4d(ctx, g::GGML_TYPE_F32, d[0], d[1], d[2], d[3]),
                }
            };
            unsafe {
                let c = CString::new(n.as_str()).unwrap();
                g::ggml_set_name(x, c.as_ptr());
            }
            if data.len() <= 4 * 4096 {
                host.insert(
                    n.clone(),
                    data.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            }
            map.insert(n.clone(), x);
        }
        let buf = if items.is_empty() {
            ptr::null_mut()
        } else {
            be.alloc(ctx)?
        };
        for (n, _, data) in items {
            unsafe { g::ggml_backend_tensor_set(map[n], data.as_ptr().cast(), 0, data.len()) };
        }
        Ok(Weights { ctx, buf, map, host })
    }

    pub fn t(&self, name: &str) -> Result<T, String> {
        self.map
            .get(name)
            .copied()
            .ok_or_else(|| format!("decoder tensor {name} missing"))
    }

    pub fn host(&self, name: &str) -> Result<&[f32], String> {
        self.host
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("decoder tensor {name} missing on the host"))
    }
}

/// A graph context, built once and run through the backend's scheduler.
pub struct Graph<'a> {
    pub ctx: *mut g::ggml_context,
    pub gf: *mut g::ggml_cgraph,
    be: &'a Backend,
}

impl<'a> Graph<'a> {
    pub fn new(be: &'a Backend) -> Result<Self, String> {
        unsafe {
            let size = g::ggml_tensor_overhead() * GRAPH_SIZE + g::ggml_graph_overhead_custom(GRAPH_SIZE, false);
            let ctx = g::ggml_init(g::ggml_init_params {
                mem_size: size,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            });
            if ctx.is_null() {
                return Err("ggml_init failed for a decoder graph".into());
            }
            let gf = g::ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
            Ok(Graph { ctx, gf, be })
        }
    }

    pub fn input(&self, ne: &[i64]) -> T {
        unsafe {
            let x = match ne.len() {
                1 => g::ggml_new_tensor_1d(self.ctx, g::GGML_TYPE_F32, ne[0]),
                2 => g::ggml_new_tensor_2d(self.ctx, g::GGML_TYPE_F32, ne[0], ne[1]),
                _ => g::ggml_new_tensor_3d(self.ctx, g::GGML_TYPE_F32, ne[0], ne[1], ne[2]),
            };
            g::ggml_set_input(x);
            x
        }
    }

    /// Allocate, set the inputs, compute, and read the outputs back.
    pub fn run(&mut self, inputs: &[(T, &[f32])], outputs: &[T]) -> Result<Vec<Vec<f32>>, String> {
        self.run_ints(inputs, &[], outputs)
    }

    /// [`Graph::run`] with `i32` inputs too (positions).
    pub fn run_ints(
        &mut self,
        inputs: &[(T, &[f32])],
        ints: &[(T, &[i32])],
        outputs: &[T],
    ) -> Result<Vec<Vec<f32>>, String> {
        unsafe {
            for &o in outputs {
                g::ggml_set_output(o);
                g::ggml_build_forward_expand(self.gf, o);
            }
            let sched = self.be.sched;
            g::ggml_backend_sched_reset(sched);
            if !g::ggml_backend_sched_alloc_graph(sched, self.gf) {
                return Err("ggml_backend_sched_alloc_graph failed".into());
            }
            // An input the graph never reads is never allocated, and setting it would abort.
            for (t, d) in inputs {
                if (**t).buffer.is_null() {
                    continue;
                }
                if g::ggml_nbytes(*t) != d.len() * 4 {
                    return Err("decoder input size mismatch".into());
                }
                g::ggml_backend_tensor_set(*t, d.as_ptr().cast(), 0, d.len() * 4);
            }
            for (t, d) in ints {
                if (**t).buffer.is_null() {
                    continue;
                }
                if g::ggml_nbytes(*t) != d.len() * 4 {
                    return Err("decoder input size mismatch".into());
                }
                g::ggml_backend_tensor_set(*t, d.as_ptr().cast(), 0, d.len() * 4);
            }
            if g::ggml_backend_sched_graph_compute(sched, self.gf) != g::GGML_STATUS_SUCCESS {
                return Err("decoder graph compute failed".into());
            }
            Ok(outputs
                .iter()
                .map(|&o| {
                    let n = g::ggml_nbytes(o) / 4;
                    let mut v = vec![0f32; n];
                    g::ggml_backend_tensor_get(o, v.as_mut_ptr().cast(), 0, n * 4);
                    v
                })
                .collect())
        }
    }
}

impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe {
            g::ggml_backend_sched_reset(self.be.sched);
            g::ggml_free(self.ctx);
        }
    }
}

/// Graph-building helpers over `[T, C]` signals.
pub(crate) struct Ops<'a> {
    pub(crate) ctx: *mut g::ggml_context,
    pub(crate) w: &'a Weights,
}

impl Ops<'_> {
    /// `nn.Conv1d`: kernel `[K, Cin, Cout]`, optional bias `[Cout]`.
    pub(crate) fn conv(&self, x: T, w: T, b: Option<T>, stride: i32, pad: i32, dil: i32) -> T {
        unsafe {
            let col = g::ggml_im2col(self.ctx, w, x, stride, 0, pad, 0, dil, 0, false, g::GGML_TYPE_F32);
            let col = g::ggml_reshape_2d(self.ctx, col, (*col).ne[0], (*col).ne[1] * (*col).ne[2]);
            let k = g::ggml_reshape_2d(self.ctx, w, (*w).ne[0] * (*w).ne[1], (*w).ne[2]);
            let y = g::ggml_mul_mat(self.ctx, col, k); // [T_out, Cout]
            match b {
                Some(b) => g::ggml_add(self.ctx, y, g::ggml_reshape_2d(self.ctx, b, 1, (*b).ne[0])),
                None => y,
            }
        }
    }

    pub(crate) fn conv_named(&self, x: T, name: &str, stride: i32, pad: i32, dil: i32) -> Result<T, String> {
        let b = self.w.t(&format!("{name}.bias")).ok();
        Ok(self.conv(x, self.w.t(&format!("{name}.weight"))?, b, stride, pad, dil))
    }

    /// `nn.ConvTranspose1d` with `padding = p`: ggml's has none, so the full output is cropped by `p` each side.
    pub(crate) fn conv_transpose(&self, x: T, name: &str, stride: i32, pad: i64) -> Result<T, String> {
        let w = self.w.t(&format!("{name}.weight"))?;
        let b = self.w.t(&format!("{name}.bias"))?;
        unsafe {
            let y = g::ggml_conv_transpose_1d(self.ctx, w, x, stride, 0, 1);
            let len = (*y).ne[0] - 2 * pad;
            let y = g::ggml_view_2d(self.ctx, y, len, (*y).ne[1], (*y).nb[1], pad as usize * 4);
            let y = g::ggml_cont(self.ctx, y);
            Ok(g::ggml_add(self.ctx, y, g::ggml_reshape_2d(self.ctx, b, 1, (*b).ne[0])))
        }
    }

    /// `Snake`: `x + 1 / (alpha + 1e-9) * sin(alpha * x)^2`, alpha per channel.
    pub(crate) fn snake(&self, x: T, alpha: T, inv_alpha: T) -> T {
        unsafe {
            let c = (*alpha).ne[0];
            let a = g::ggml_reshape_2d(self.ctx, alpha, 1, c);
            let ia = g::ggml_reshape_2d(self.ctx, inv_alpha, 1, c);
            let s = g::ggml_sqr(self.ctx, g::ggml_sin(self.ctx, g::ggml_mul(self.ctx, x, a)));
            g::ggml_add(self.ctx, x, g::ggml_mul(self.ctx, s, ia))
        }
    }
}

/// The vocoder: weights, and the `1 / (alpha + 1e-9)` of every Snake, precomputed.
pub struct Hift {
    w: Weights,
    inv: Weights,
    pub be: Rc<Backend>,
}

const UP_RATES: [i32; 3] = [8, 5, 3];
const UP_KERNELS: [i64; 3] = [16, 11, 7];
const RES_KERNELS: [i32; 3] = [3, 7, 11];
const SRC_KERNELS: [i32; 3] = [7, 7, 11];
const DILATIONS: [i32; 3] = [1, 3, 5];
const SRC_DOWN: [(i32, i32); 3] = [(15, 7), (3, 1), (1, 0)]; // (stride, padding); kernel 2*stride, or 1
/// Samples per mel frame: the product of the upsampling rates times the STFT hop.
pub const UPSAMPLE: usize = 480;

impl Hift {
    pub fn load(gguf: &Gguf, be: Rc<Backend>) -> Result<Self, String> {
        let w = Weights::load(gguf, "mel2wav.", &be)?;
        // The inverse alphas, named after the alpha they invert.
        let mut inv = Vec::new();
        for a in w.map.keys().filter(|n| n.ends_with(".alpha")) {
            inv.push((a.clone(), w.host(a)?.iter().map(|&x| 1.0 / (x + 1e-9)).collect()));
        }
        let inv = Weights::from_host(&be, &inv)?;
        Ok(Hift { w, inv, be })
    }

    fn resblock(&self, o: &Ops, mut x: T, name: &str, k: i32) -> Result<T, String> {
        for (j, &d) in DILATIONS.iter().enumerate() {
            let a1 = format!("{name}.activations1.{j}.alpha");
            let a2 = format!("{name}.activations2.{j}.alpha");
            let xt = o.snake(x, self.w.t(&a1)?, self.inv.t(&a1)?);
            let xt = o.conv_named(xt, &format!("{name}.convs1.{j}"), 1, (k * d - d) / 2, d)?;
            let xt = o.snake(xt, self.w.t(&a2)?, self.inv.t(&a2)?);
            let xt = o.conv_named(xt, &format!("{name}.convs2.{j}"), 1, (k - 1) / 2, 1)?;
            x = unsafe { g::ggml_add(o.ctx, x, xt) };
        }
        Ok(x)
    }

    /// `ConvRNNF0Predictor`: mel (`[T, 80]`, channel-major) to f0 (`T`).
    pub fn f0(&self, mel: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let mut gr = Graph::new(&self.be)?;
        let o = Ops {
            ctx: gr.ctx,
            w: &self.w,
        };
        let x0 = gr.input(&[t as i64, 80]);
        let mut x = x0;
        for i in [0, 2, 4, 6, 8] {
            x = o.conv_named(x, &format!("mel2wav.f0_predictor.condnet.{i}"), 1, 1, 1)?;
            x = unsafe { g::ggml_elu(o.ctx, x) };
        }
        let y = unsafe {
            let xt = g::ggml_cont(o.ctx, g::ggml_transpose(o.ctx, x)); // [512, T]
            let y = g::ggml_mul_mat(o.ctx, self.w.t("mel2wav.f0_predictor.classifier.weight")?, xt); // [1, T]
            let y = g::ggml_add(o.ctx, y, self.w.t("mel2wav.f0_predictor.classifier.bias")?);
            g::ggml_abs(o.ctx, y)
        };
        Ok(gr.run(&[(x0, mel)], &[y])?.remove(0))
    }

    /// `HiFTGenerator.decode` up to the inverse STFT: mel and the source's STFT (18 channels of `t * 120 + 1`) to
    /// the magnitude and phase (9 channels each, channel-major), then the inverse STFT and the clamp.
    pub fn decode(&self, mel: &[f32], t: usize, s_stft: &[f32]) -> Result<Vec<f32>, String> {
        let frames = t * 120 + 1;
        let mut gr = Graph::new(&self.be)?;
        let o = Ops {
            ctx: gr.ctx,
            w: &self.w,
        };
        let xm = gr.input(&[t as i64, 80]);
        let xs = gr.input(&[frames as i64, 18]);
        let mut x = o.conv_named(xm, "mel2wav.conv_pre", 1, 3, 1)?;
        for i in 0..3 {
            x = unsafe { g::ggml_leaky_relu(o.ctx, x, 0.1, false) };
            let p = (UP_KERNELS[i] - i64::from(UP_RATES[i])) / 2;
            x = o.conv_transpose(x, &format!("mel2wav.ups.{i}"), UP_RATES[i], p)?;
            if i == 2 {
                x = unsafe { g::ggml_pad_reflect_1d(o.ctx, x, 1, 0) };
            }
            let (stride, pad) = SRC_DOWN[i];
            let si = o.conv_named(xs, &format!("mel2wav.source_downs.{i}"), stride, pad, 1)?;
            let si = self.resblock(&o, si, &format!("mel2wav.source_resblocks.{i}"), SRC_KERNELS[i])?;
            x = unsafe { g::ggml_add(o.ctx, x, si) };
            let mut xs_sum: Option<T> = None;
            for (j, &k) in RES_KERNELS.iter().enumerate() {
                let r = self.resblock(&o, x, &format!("mel2wav.resblocks.{}", i * 3 + j), k)?;
                xs_sum = Some(match xs_sum {
                    None => r,
                    Some(s) => unsafe { g::ggml_add(o.ctx, s, r) },
                });
            }
            x = unsafe { g::ggml_scale(o.ctx, xs_sum.unwrap(), 1.0 / 3.0) };
        }
        let y = unsafe {
            let x = g::ggml_leaky_relu(o.ctx, x, 0.01, false);
            o.conv_named(x, "mel2wav.conv_post", 1, 3, 1)?
        };
        let out = gr.run(&[(xm, mel), (xs, s_stft)], &[y])?.remove(0);
        let n = dsp::N_BINS * frames;
        let mag: Vec<f32> = out[..n].iter().map(|v| v.exp()).collect();
        let phase: Vec<f32> = out[n..2 * n].iter().map(|v| v.sin()).collect();
        Ok(dsp::istft(&mag, &phase, frames)
            .into_iter()
            .map(|v| v.clamp(-0.99, 0.99))
            .collect())
    }

    /// The source, from f0: [`dsp::source`] with this vocoder's `l_linear`.
    pub fn source(&self, f0: &[f32], noise: &mut dsp::Noise) -> Result<Vec<f32>, String> {
        let lw = self.w.host("mel2wav.m_source.l_linear.weight")?;
        let lb = self.w.host("mel2wav.m_source.l_linear.bias")?[0];
        Ok(dsp::source(f0, UPSAMPLE, 24000.0, lw, lb, noise))
    }
}

/// S3Gen's fade-in over the first 40 ms (`trim_fade`): 20 ms of silence, then a raised-cosine ramp.
pub fn trim_fade(wav: &mut [f32]) {
    let n = 24000 / 50;
    for (i, v) in wav.iter_mut().enumerate().take(2 * n) {
        let f = if i < n {
            0.0
        } else {
            // torch.linspace(pi, 0, n)[i - n]
            let x = std::f64::consts::PI * (1.0 - (i - n) as f64 / (n - 1) as f64);
            (x.cos() + 1.0) / 2.0
        };
        *v = (f64::from(*v) * f) as f32;
    }
}
