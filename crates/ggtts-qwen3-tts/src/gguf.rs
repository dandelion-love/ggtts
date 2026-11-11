//! Reading a GGUF file and loading its weights into a CPU buffer, as qwen3-tts.cpp's `GGUFLoader` and
//! `load_tensor_data_from_file` do (`src/gguf_loader.cpp:254-433` at `16bb5af`): metadata through ggml's gguf API,
//! tensor data read at its file offset, and a type conversion only from F32 or BF16 to F16.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::ptr;

use ggtts_core::engine::EngineError;
use ggtts_ggml_sys as g;

fn err(msg: impl Into<String>) -> EngineError {
    EngineError(msg.into())
}

/// An open GGUF file: its key-value metadata and tensor directory. Tensor data is not read here.
pub struct Gguf {
    ctx: *mut g::gguf_context,
    meta: *mut g::ggml_context,
    path: PathBuf,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, EngineError> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| err("path has a NUL byte"))?;
        let mut meta: *mut g::ggml_context = ptr::null_mut();
        let params = g::gguf_init_params {
            no_alloc: true,
            ctx: &mut meta,
        };
        // SAFETY: c is a valid C string; meta receives a context owned by Self.
        let ctx = unsafe { g::gguf_init_from_file(c.as_ptr(), params) };
        if ctx.is_null() {
            return Err(err(format!("failed to open GGUF file: {}", path.display())));
        }
        Ok(Self {
            ctx,
            meta,
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn key(&self, key: &str) -> Option<i64> {
        let c = CString::new(key).ok()?;
        // SAFETY: ctx is live for self's lifetime.
        let id = unsafe { g::gguf_find_key(self.ctx, c.as_ptr()) };
        (id >= 0).then_some(id)
    }

    /// `GGUFLoader::get_u32`: the value read as u32 and cast to i32, or the default when the key is absent.
    pub fn u32_or(&self, key: &str, default: i32) -> i32 {
        match self.key(key) {
            // SAFETY: id came from gguf_find_key on this context.
            Some(id) => unsafe { g::gguf_get_val_u32(self.ctx, id) as i32 },
            None => default,
        }
    }

    /// `GGUFLoader::get_f32`.
    pub fn f32_or(&self, key: &str, default: f32) -> f32 {
        match self.key(key) {
            // SAFETY: as above.
            Some(id) => unsafe { g::gguf_get_val_f32(self.ctx, id) },
            None => default,
        }
    }

    /// A flag as `get_bool_any` reads one (`transformer_loader_metadata.cpp:45-87`): a bool, or any integer type
    /// read as not-zero. `None` when the key is absent or holds anything else, which the reference also treats as
    /// absent.
    pub fn flag(&self, key: &str) -> Option<bool> {
        let id = self.key(key)?;
        // SAFETY: as above; each read matches the type checked first.
        unsafe {
            Some(match g::gguf_get_kv_type(self.ctx, id) {
                g::GGUF_TYPE_BOOL => g::gguf_get_val_bool(self.ctx, id),
                g::GGUF_TYPE_UINT8 => g::gguf_get_val_u8(self.ctx, id) != 0,
                g::GGUF_TYPE_INT8 => g::gguf_get_val_i8(self.ctx, id) != 0,
                g::GGUF_TYPE_UINT16 => g::gguf_get_val_u16(self.ctx, id) != 0,
                g::GGUF_TYPE_INT16 => g::gguf_get_val_i16(self.ctx, id) != 0,
                g::GGUF_TYPE_UINT32 => g::gguf_get_val_u32(self.ctx, id) != 0,
                g::GGUF_TYPE_INT32 => g::gguf_get_val_i32(self.ctx, id) != 0,
                g::GGUF_TYPE_UINT64 => g::gguf_get_val_u64(self.ctx, id) != 0,
                g::GGUF_TYPE_INT64 => g::gguf_get_val_i64(self.ctx, id) != 0,
                _ => return None,
            })
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.key(key).is_some()
    }

    pub fn kv_type(&self, key: &str) -> Option<g::gguf_type> {
        // SAFETY: as above.
        self.key(key).map(|id| unsafe { g::gguf_get_kv_type(self.ctx, id) })
    }

    pub fn string(&self, key: &str) -> Option<String> {
        let id = self.key(key)?;
        // SAFETY: as above; the type is checked before the read.
        unsafe {
            if g::gguf_get_kv_type(self.ctx, id) != g::GGUF_TYPE_STRING {
                return None;
            }
            Some(
                CStr::from_ptr(g::gguf_get_val_str(self.ctx, id))
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    /// A string array, or None if the key is absent or not an array of strings.
    pub fn strings(&self, key: &str) -> Option<Vec<String>> {
        let id = self.key(key)?;
        // SAFETY: as above; types are checked before the reads.
        unsafe {
            if g::gguf_get_kv_type(self.ctx, id) != g::GGUF_TYPE_ARRAY
                || g::gguf_get_arr_type(self.ctx, id) != g::GGUF_TYPE_STRING
            {
                return None;
            }
            let n = g::gguf_get_arr_n(self.ctx, id);
            Some(
                (0..n)
                    .map(|i| {
                        CStr::from_ptr(g::gguf_get_arr_str(self.ctx, id, i))
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect(),
            )
        }
    }

    /// An integer array of any 32-bit integer type, as i32.
    pub fn ints(&self, key: &str) -> Option<Vec<i32>> {
        let id = self.key(key)?;
        // SAFETY: as above; the element type is checked and the data has n elements of it.
        unsafe {
            if g::gguf_get_kv_type(self.ctx, id) != g::GGUF_TYPE_ARRAY {
                return None;
            }
            let t = g::gguf_get_arr_type(self.ctx, id);
            if t != g::GGUF_TYPE_INT32 && t != g::GGUF_TYPE_UINT32 {
                return None;
            }
            let n = g::gguf_get_arr_n(self.ctx, id);
            let data = g::gguf_get_arr_data(self.ctx, id) as *const i32;
            Some(std::slice::from_raw_parts(data, n).to_vec())
        }
    }

    pub fn n_tensors(&self) -> i64 {
        // SAFETY: ctx is live.
        unsafe { g::gguf_get_n_tensors(self.ctx) }
    }

    pub fn tensor_name(&self, i: i64) -> String {
        // SAFETY: i < n_tensors by the callers' loops.
        unsafe {
            CStr::from_ptr(g::gguf_get_tensor_name(self.ctx, i))
                .to_string_lossy()
                .into_owned()
        }
    }

    /// The tensor's metadata (type, shape) from the file, without data.
    pub fn meta_tensor(&self, name: &str) -> *mut g::ggml_tensor {
        let c = CString::new(name).unwrap();
        // SAFETY: meta is the no-alloc context gguf_init_from_file filled.
        unsafe { g::ggml_get_tensor(self.meta, c.as_ptr()) }
    }

    /// `load_tensor_data_from_file` (`src/gguf_loader.cpp:303-433`): allocates every tensor of `ctx` in one buffer of
    /// `device` and fills the ones named in `tensors` from the file, in file order. Same type: bytes copied. F32 or
    /// BF16 in the file to an F16 tensor: converted with ggml's own routines. Anything else is refused.
    ///
    /// # Safety
    /// `ctx` must be a live no-alloc context, and every tensor in `tensors` must belong to it.
    pub unsafe fn load_into(
        &self,
        ctx: *mut g::ggml_context,
        tensors: &HashMap<String, *mut g::ggml_tensor>,
        device: crate::cpu::Device,
    ) -> Result<g::ggml_backend_buffer_t, EngineError> {
        // SAFETY: a fresh CPU backend, used only to allocate and freed before returning, as the reference does.
        unsafe {
            let backend = device.init_backend(1);
            if backend.is_null() {
                return Err(err("failed to initialize a backend for the tensor loader"));
            }
            let buffer = g::ggml_backend_alloc_ctx_tensors(ctx, backend);
            g::ggml_backend_free(backend);
            if buffer.is_null() {
                return Err(err("failed to allocate the tensor buffer"));
            }
            g::ggml_backend_buffer_set_usage(buffer, g::GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            let mut f = File::open(&self.path).map_err(|e| err(format!("{}: {e}", self.path.display())))?;
            let data_offset = g::gguf_get_data_offset(self.ctx) as u64;
            let mut buf = Vec::new();
            for i in 0..self.n_tensors() {
                let name = self.tensor_name(i);
                let Some(&tensor) = tensors.get(&name) else { continue };
                let offset = g::gguf_get_tensor_offset(self.ctx, i) as u64;
                let source_type = g::gguf_get_tensor_type(self.ctx, i);
                let source_nbytes = g::gguf_get_tensor_size(self.ctx, i);
                let target_nbytes = g::ggml_nbytes(tensor);
                buf.resize(source_nbytes, 0u8);
                f.seek(SeekFrom::Start(data_offset + offset))
                    .and_then(|_| f.read_exact(&mut buf))
                    .map_err(|e| err(format!("reading tensor {name}: {e}")))?;
                let ttype = (*tensor).type_;
                if source_type == ttype {
                    if source_nbytes != target_nbytes {
                        return Err(err(format!("tensor byte size mismatch: {name}")));
                    }
                    g::ggml_backend_tensor_set(tensor, buf.as_ptr().cast(), 0, target_nbytes);
                } else if ttype == g::GGML_TYPE_F16
                    && (source_type == g::GGML_TYPE_F32 || source_type == g::GGML_TYPE_BF16)
                {
                    let n = g::ggml_nelements(tensor);
                    let mut converted = vec![0u16; n as usize];
                    if source_type == g::GGML_TYPE_F32 {
                        g::ggml_fp32_to_fp16_row(buf.as_ptr().cast(), converted.as_mut_ptr(), n);
                    } else {
                        let src = buf.as_ptr() as *const u16;
                        for (e, out) in converted.iter_mut().enumerate() {
                            let b = g::ggml_bf16_t {
                                bits: src.add(e).read_unaligned(),
                            };
                            *out = g::ggml_fp32_to_fp16(g::ggml_bf16_to_fp32(b));
                        }
                    }
                    if converted.len() * 2 != target_nbytes {
                        return Err(err(format!("converted tensor byte size mismatch: {name}")));
                    }
                    g::ggml_backend_tensor_set(tensor, converted.as_ptr().cast(), 0, target_nbytes);
                } else {
                    return Err(err(format!(
                        "unsupported tensor conversion for {name}: type {source_type} -> {ttype}"
                    )));
                }
            }
            Ok(buffer)
        }
    }
}

impl Drop for Gguf {
    fn drop(&mut self) {
        // SAFETY: both were created in open() and are freed once.
        unsafe {
            g::gguf_free(self.ctx);
            if !self.meta.is_null() {
                g::ggml_free(self.meta);
            }
        }
    }
}

/// A set of weights: a no-alloc context holding the tensors, their CPU buffer, and a name index.
pub struct Weights {
    pub ctx: *mut g::ggml_context,
    pub buffer: g::ggml_backend_buffer_t,
    pub tensors: HashMap<String, *mut g::ggml_tensor>,
}

impl Weights {
    pub fn get(&self, name: &str) -> Option<*mut g::ggml_tensor> {
        self.tensors.get(name).copied()
    }
}

impl Drop for Weights {
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

/// Sets a tensor's name. Callers pass a live tensor; it is a raw pointer, as every ggml handle here is.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub(crate) fn set_name(t: *mut g::ggml_tensor, name: &str) {
    let c = CString::new(name).unwrap();
    // SAFETY: t is a live tensor; ggml copies the name into the tensor.
    unsafe {
        g::ggml_set_name(t, c.as_ptr());
    }
}
