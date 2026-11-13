//! The compute backend and its scheduler, as qwen3-tts.cpp sets them up (`src/gguf_loader.cpp:29-55`,
//! `src/decoder/decoder_loader.cpp:453-484`, `src/transformer/transformer_loader.cpp:252-266`): on CPU, one CPU
//! backend; on CUDA, the CUDA backend first and a CPU backend as the scheduler's fallback.

use std::ptr;

use ggtts_core::engine::EngineError;
use ggtts_ggml_sys as g;

/// Where the graphs run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// The first CUDA device. Needs the `cuda` feature.
    Cuda,
}

impl Device {
    pub fn is_cuda(self) -> bool {
        self == Device::Cuda
    }

    /// `cpu` or `cuda`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(Device::Cpu),
            "cuda" => Some(Device::Cuda),
            _ => None,
        }
    }

    /// A fresh backend of this device's type; the CPU one gets `threads` threads. Null on failure.
    pub(crate) fn init_backend(self, threads: i32) -> g::ggml_backend_t {
        // SAFETY: plain constructor calls; the caller owns the handle.
        unsafe {
            match self {
                Device::Cpu => {
                    let b = g::ggml_backend_init_by_type(g::GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
                    if !b.is_null() {
                        g::ggml_backend_cpu_set_n_threads(b, threads);
                    }
                    b
                }
                Device::Cuda => g::ggml_backend_init_by_type(g::GGML_BACKEND_DEVICE_TYPE_GPU, ptr::null()),
            }
        }
    }
}

/// The reference's default thread count: half the hardware threads, at least one
/// (`default_cpu_thread_count`, `src/gguf_loader.cpp`).
pub fn default_threads() -> i32 {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) / 2;
    n.max(1) as i32
}

/// The compute backend (CPU, or CUDA with a CPU fallback) and a scheduler over them.
pub struct Cpu {
    pub device: Device,
    pub backend: g::ggml_backend_t,
    fallback: g::ggml_backend_t,
    pub sched: g::ggml_backend_sched_t,
    /// The CPU threads asked for; each compute runs with no more than the process may use then
    /// ([`g::cap_sched_threads`]).
    pub threads: i32,
}

impl Cpu {
    /// Before a compute: the CPU backends get the configured threads, capped at the CPUs this process may use now.
    ///
    /// # Safety
    /// `self.sched` is live for as long as `self` is.
    pub(crate) unsafe fn cap_threads(&self) {
        g::cap_sched_threads(self.sched, self.threads);
    }

    pub fn new(threads: i32, graph_size: usize) -> Result<Self, EngineError> {
        Self::on(Device::Cpu, threads, graph_size)
    }

    pub fn on(device: Device, threads: i32, graph_size: usize) -> Result<Self, EngineError> {
        let backend = device.init_backend(threads);
        if backend.is_null() {
            return Err(EngineError(format!("failed to initialize the {device:?} backend")));
        }
        let fallback = if device.is_cuda() {
            Device::Cpu.init_backend(threads)
        } else {
            ptr::null_mut()
        };
        // SAFETY: both handles are live; Self owns them and the scheduler.
        unsafe {
            let mut backends = vec![backend];
            if !fallback.is_null() {
                backends.push(fallback);
            }
            let sched = g::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                backends.len() as i32,
                graph_size,
                false,
                true,
            );
            if sched.is_null() {
                g::ggml_backend_free(backend);
                if !fallback.is_null() {
                    g::ggml_backend_free(fallback);
                }
                return Err(EngineError("failed to create the backend scheduler".into()));
            }
            Ok(Self {
                device,
                backend,
                fallback,
                sched,
                threads,
            })
        }
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        // SAFETY: owned; freed once, scheduler first.
        unsafe {
            g::ggml_backend_sched_free(self.sched);
            g::ggml_backend_free(self.backend);
            if !self.fallback.is_null() {
                g::ggml_backend_free(self.fallback);
            }
        }
    }
}

/// A reusable buffer for graph metadata, and a no-alloc context over it for one graph build.
pub struct Meta {
    buf: Vec<u8>,
}

impl Meta {
    pub fn new(n_nodes: usize) -> Self {
        // SAFETY: size queries only.
        let size = unsafe { g::ggml_tensor_overhead() * n_nodes + g::ggml_graph_overhead_custom(n_nodes, false) };
        Self { buf: vec![0; size] }
    }

    /// A fresh no-alloc context over the buffer. The caller frees it with `ggml_free` before the next call.
    pub fn context(&mut self) -> *mut g::ggml_context {
        let params = g::ggml_init_params {
            mem_size: self.buf.len(),
            mem_buffer: self.buf.as_mut_ptr().cast(),
            no_alloc: true,
        };
        // SAFETY: the buffer outlives the context by the contract above.
        unsafe { g::ggml_init(params) }
    }
}
