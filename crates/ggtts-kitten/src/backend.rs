//! Where the graphs run: the CPU, or the first CUDA device with the CPU as the scheduler's fallback for any op
//! the CUDA backend lacks (the arrangement ggtts's Qwen3-TTS engine uses). Weights and the KV cache live in the
//! device's own buffers; every graph runs through one scheduler.

use std::ptr;
use std::rc::Rc;

use ggtts_ggml_sys as g;

/// Where the graphs run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// The first CUDA device. Needs a ggml built with CUDA (`ggtts-ggml-sys/cuda`).
    Cuda,
}

impl Device {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(Device::Cpu),
            "cuda" => Some(Device::Cuda),
            _ => None,
        }
    }
}

/// The environment variable NVIDIA's libraries read to override TF32 tensor-core math; `0` disables it in cuBLAS.
pub const TF32_OVERRIDE: &str = "NVIDIA_TF32_OVERRIDE";

/// Ask cuBLAS for full f32 math, before this process's CUDA backend exists, unless the caller set the override.
///
/// ggml's CUDA backend puts every cuBLAS handle in TF32 tensor-op mode (`ggml/src/ggml-cuda/common.cuh:1478`,
/// `cublasSetMathMode(..., CUBLAS_TF32_TENSOR_OP_MATH)`), so an f32 matrix product of more than 16 columns, which
/// ggml hands to cuBLAS, rounds both operands to a 10-bit mantissa. Every convolution here is an f32 `im2col` and
/// such a product, and so is attention over a prompt. ggml has no option for it at the pin; NVIDIA's override does.
/// It is set before the backend is created, and ggml creates its cuBLAS handles lazily, at the first product.
/// Whether cuBLAS honours a value set inside the process is measured, not assumed: `scripts/cuda_probe.sh` runs the
/// decoder check once more with the override exported from the shell. A value the caller set is respected, so
/// `NVIDIA_TF32_OVERRIDE=1` reproduces ggml's TF32 behaviour for comparison.
fn disable_tf32() {
    if std::env::var_os(TF32_OVERRIDE).is_none() {
        // Before any CUDA or cuBLAS state exists, and before this crate starts a thread.
        std::env::set_var(TF32_OVERRIDE, "0");
    }
}

/// One line for a log: what this process asks of cuBLAS's TF32 math.
pub fn tf32_note() -> String {
    match std::env::var(TF32_OVERRIDE).as_deref() {
        Ok("0") => format!("{TF32_OVERRIDE}=0: cuBLAS asked for full f32 products"),
        Ok(v) => format!("{TF32_OVERRIDE}={v}: cuBLAS may use TF32"),
        Err(_) => format!("{TF32_OVERRIDE} unset: cuBLAS uses TF32 (ggml's default)"),
    }
}

/// The largest graph any Kitten graph builds, in nodes.
pub const GRAPH_SIZE: usize = 16384;

pub struct Backend {
    pub device: Device,
    pub threads: usize,
    pub backend: g::ggml_backend_t,
    fallback: g::ggml_backend_t,
    pub sched: g::ggml_backend_sched_t,
}

impl Backend {
    pub fn new(device: Device, threads: usize) -> Result<Rc<Self>, String> {
        // SAFETY: constructors; Self owns every handle and frees them in Drop.
        unsafe {
            let cpu = || {
                let b = g::ggml_backend_init_by_type(g::GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
                if !b.is_null() {
                    g::ggml_backend_cpu_set_n_threads(b, threads as i32);
                }
                b
            };
            let backend = match device {
                Device::Cpu => cpu(),
                Device::Cuda => {
                    disable_tf32();
                    g::ggml_backend_init_by_type(g::GGML_BACKEND_DEVICE_TYPE_GPU, ptr::null())
                }
            };
            if backend.is_null() {
                return Err(format!(
                    "failed to initialise the {device:?} backend (is ggml built with it?)"
                ));
            }
            let fallback = if device == Device::Cuda { cpu() } else { ptr::null_mut() };
            let mut list = vec![backend];
            if !fallback.is_null() {
                list.push(fallback);
            }
            let sched = g::ggml_backend_sched_new(
                list.as_mut_ptr(),
                ptr::null_mut(),
                list.len() as i32,
                GRAPH_SIZE,
                false,
                true,
            );
            if sched.is_null() {
                g::ggml_backend_free(backend);
                if !fallback.is_null() {
                    g::ggml_backend_free(fallback);
                }
                return Err("failed to create the backend scheduler".into());
            }
            Ok(Rc::new(Backend {
                device,
                threads,
                backend,
                fallback,
                sched,
            }))
        }
    }

    /// Allocate every tensor of a no-alloc context in this device's buffer.
    pub(crate) fn alloc(&self, ctx: *mut g::ggml_context) -> Result<g::ggml_backend_buffer_t, String> {
        // SAFETY: ctx is a live no-alloc context; the caller owns the returned buffer.
        let buf = unsafe { g::ggml_backend_alloc_ctx_tensors(ctx, self.backend) };
        if buf.is_null() {
            return Err(format!("failed to allocate a {:?} buffer", self.device));
        }
        Ok(buf)
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: owned; the scheduler first.
        unsafe {
            g::ggml_backend_sched_free(self.sched);
            g::ggml_backend_free(self.backend);
            if !self.fallback.is_null() {
                g::ggml_backend_free(self.fallback);
            }
        }
    }
}
