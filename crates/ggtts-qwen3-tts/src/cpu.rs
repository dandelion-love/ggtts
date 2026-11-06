//! The CPU backend and a one-backend scheduler, as qwen3-tts.cpp sets them up (`src/gguf_loader.cpp:29-55`,
//! `src/decoder/decoder_loader.cpp:473-484`).

use std::ptr;

use ggtts_core::engine::EngineError;
use ggtts_ggml_sys as g;

/// The reference's default thread count: half the hardware threads, at least one
/// (`default_cpu_thread_count`, `src/gguf_loader.cpp`).
pub fn default_threads() -> i32 {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) / 2;
    n.max(1) as i32
}

/// A CPU backend with a fixed thread count, and a scheduler over it alone.
pub struct Cpu {
    pub backend: g::ggml_backend_t,
    pub sched: g::ggml_backend_sched_t,
}

impl Cpu {
    pub fn new(threads: i32, graph_size: usize) -> Result<Self, EngineError> {
        // SAFETY: plain constructor calls; both handles are owned by Self and freed in Drop.
        unsafe {
            let backend = g::ggml_backend_init_by_type(g::GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
            if backend.is_null() {
                return Err(EngineError("failed to initialize the CPU backend".into()));
            }
            g::ggml_backend_cpu_set_n_threads(backend, threads);
            let mut backends = [backend];
            let sched = g::ggml_backend_sched_new(backends.as_mut_ptr(), ptr::null_mut(), 1, graph_size, false, true);
            if sched.is_null() {
                g::ggml_backend_free(backend);
                return Err(EngineError("failed to create the backend scheduler".into()));
            }
            Ok(Self { backend, sched })
        }
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        // SAFETY: owned; freed once, scheduler first.
        unsafe {
            g::ggml_backend_sched_free(self.sched);
            g::ggml_backend_free(self.backend);
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
