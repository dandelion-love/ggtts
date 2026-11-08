//! ggtts behind a C-ABI, for the Android binding (and any other host that speaks C). The surface is
//! `docs/android/binding-api.md`; the header is `include/ggtts.h`.
//!
//! * **Handles are ids, not pointers.** Engines live in a registry behind a `u64` id that is never reused, so a call
//!   after [`ggtts_free`], or with id `0`, is [`GGTTS_ERR_HANDLE`], never a use of freed memory.
//! * **The registry is locked only to look an engine up, never while it renders.** Each engine has its own lock, taken
//!   with `try_lock`: a second call on an engine that is rendering (from its own stream callback, or another thread)
//!   is [`GGTTS_ERR_BUSY`], never a deadlock. Different engines render at once, and [`ggtts_free`] never waits for a
//!   render: it drops the registry's reference, and the engine goes when the render holding it returns.
//! * **No panic unwinds across the boundary.** The entry points that reach an engine or the registry
//!   ([`ggtts_load`], [`ggtts_render`], [`ggtts_render_stream`], [`ggtts_free`]) run inside `catch_unwind`, and a
//!   panic comes back as [`GGTTS_ERR_PANIC`] with its message; the engine that panicked is dropped from the
//!   registry. [`ggtts_pcm_free`] and [`ggtts_last_error`] are not wrapped: one drops a box, the other reads a
//!   thread-local. `catch_unwind` catches unwinding panics only: an abort, a stack overflow or a crash in C code
//!   still ends the process.
//! * **Errors are a code plus a message:** [`ggtts_last_error`] holds the calling thread's last message.
//!
//! With the `jni` feature, `jni.rs` adds the `Java_ggtts_Native_*` entry points over the same functions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use ggtts_core::engine::Engine;
use ggtts_core::pipeline::{plan_story, render_story, render_story_streamed, RenderError, RenderOptions};
use ggtts_core::trace::NullEvents;

#[cfg(feature = "jni")]
mod jni;

pub const GGTTS_OK: i32 = 0;
/// A NULL, non-UTF-8 or out-of-range argument.
pub const GGTTS_ERR_ARG: i32 = -1;
/// No engine with this id: never loaded, already freed, or `0`.
pub const GGTTS_ERR_HANDLE: i32 = -2;
/// The engine refused: a missing file, not a GGUF, an unknown engine, a failed render.
pub const GGTTS_ERR_ENGINE: i32 = -3;
/// A panic inside ggtts, caught at the boundary.
pub const GGTTS_ERR_PANIC: i32 = -4;
/// The stream callback asked to stop.
pub const GGTTS_ERR_STOPPED: i32 = -5;
/// The engine is rendering: a call on it from its own stream callback, or from another thread meanwhile.
pub const GGTTS_ERR_BUSY: i32 = -6;

/// The story id every binding render uses: the seed of each chunk is derived from it, the text and the seed given.
const STORY: &str = "ggtts-binding";

type Fail = (i32, String);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Run `f`, turning an error or a panic into a status, and record the message.
fn guard(f: impl FnOnce() -> Result<(), Fail>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            set_error("");
            GGTTS_OK
        }
        Ok(Err((code, msg))) => {
            set_error(&msg);
            code
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".into());
            set_error(&format!("panic inside ggtts: {msg}"));
            GGTTS_ERR_PANIC
        }
    }
}

/// An engine in the registry. `Engine` has no `Send` bound, and the real engines are not `Send` to the compiler (raw
/// ggml pointers; Kitten-TTS 2 shares its backend through `Rc`), so a `Held` is built in one of two ways only:
/// [`Held::checked`], where the compiler proves the engine `Send`, or [`Held::assume_send`], whose every caller argues
/// it for that one engine.
struct Held(Box<dyn Engine>);

impl Held {
    fn checked(e: Box<dyn Engine + Send>) -> Self {
        Held(e)
    }

    /// # Safety
    /// The engine may be moved to another thread whole and used there by one thread at a time: nothing in it is tied
    /// to the thread that made it, and nothing it shares (an `Rc`, a pointer) is reachable from outside it.
    #[cfg_attr(not(feature = "ggml"), allow(dead_code))]
    unsafe fn assume_send(e: Box<dyn Engine>) -> Self {
        Held(e)
    }
}

// SAFETY: every `Held` is `Send` by the compiler's proof or by its constructor's per-engine argument (above), and its
// engine is used under its own mutex, one call at a time.
unsafe impl Send for Held {}

type Slot = Arc<Mutex<Held>>;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static ENGINES: Mutex<Option<HashMap<u64, Slot>>> = Mutex::new(None);

/// The registry, locked for a lookup, an insert or a remove and never longer. None of those panics, so a poisoned lock
/// still holds a whole map.
fn registry() -> MutexGuard<'static, Option<HashMap<u64, Slot>>> {
    ENGINES.lock().unwrap_or_else(|p| p.into_inner())
}

fn no_engine(handle: u64) -> Fail {
    (
        GGTTS_ERR_HANDLE,
        format!("no engine with handle {handle} (never loaded, or freed)"),
    )
}

/// Run `f` on the engine. The registry lock is released before `f` runs, so `f` (and a stream callback inside it) may
/// call any entry point; the engine's own lock is taken with `try_lock`, so a second call on a rendering engine is
/// [`GGTTS_ERR_BUSY`]. If `f` panics, the engine is dropped from the registry before the panic goes on to [`guard`].
fn with_engine<T>(handle: u64, f: impl FnOnce(&mut dyn Engine) -> Result<T, Fail>) -> Result<T, Fail> {
    let slot = registry()
        .get_or_insert_with(HashMap::new)
        .get(&handle)
        .cloned()
        .ok_or_else(|| no_engine(handle))?;
    let mut held = match slot.try_lock() {
        Ok(h) => h,
        Err(TryLockError::WouldBlock) => {
            return Err((
                GGTTS_ERR_BUSY,
                format!(
                    "the engine with handle {handle} is rendering (a call from its own stream callback, or from \
                     another thread)"
                ),
            ))
        }
        // A panic inside a render evicts the engine and poisons its lock (below), so a call that looked it up just
        // before the eviction never uses it either.
        Err(TryLockError::Poisoned(_)) => return Err(no_engine(handle)),
    };
    match catch_unwind(AssertUnwindSafe(|| f(held.0.as_mut()))) {
        Ok(r) => r,
        Err(p) => {
            // The registry is never locked while an engine's lock is waited on, so taking it here cannot deadlock.
            registry().get_or_insert_with(HashMap::new).remove(&handle);
            // `held` drops while this panic unwinds, which poisons the engine's lock on purpose.
            std::panic::resume_unwind(p)
        }
    }
}

/// A required C string argument.
///
/// # Safety
/// `p` is NULL or points to a NUL-terminated string.
unsafe fn arg<'a>(p: *const c_char, name: &str) -> Result<&'a str, Fail> {
    if p.is_null() {
        return Err((GGTTS_ERR_ARG, format!("{name} is NULL")));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| (GGTTS_ERR_ARG, format!("{name} is not UTF-8")))
}

/// An optional C string argument: NULL is `None`.
///
/// # Safety
/// As [`arg`].
unsafe fn opt_arg<'a>(p: *const c_char, name: &str) -> Result<Option<&'a str>, Fail> {
    if p.is_null() {
        Ok(None)
    } else {
        unsafe { arg(p, name) }.map(Some)
    }
}

/// A model file must exist and be a GGUF before any engine is asked to load it.
fn check_gguf(what: &str, path: &str) -> Result<(), Fail> {
    let mut magic = [0u8; 4];
    let read = std::fs::File::open(path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic));
    match read {
        Err(e) => Err((GGTTS_ERR_ENGINE, format!("{what} {path}: {e}"))),
        Ok(()) if &magic != b"GGUF" => Err((GGTTS_ERR_ENGINE, format!("{what} {path}: not a GGUF file"))),
        Ok(()) => Ok(()),
    }
}

/// The engine for a name. The fake engine needs no files.
fn open_engine(
    engine: &str,
    model: Option<&str>,
    codec: Option<&str>,
    voice: Option<&str>,
    threads: i32,
) -> Result<Held, Fail> {
    let threads = if threads <= 0 { 4 } else { threads };
    if engine == "fake" {
        return Ok(Held::checked(Box::new(ggtts_fake::FakeEngine::default())));
    }
    #[cfg(feature = "plant")]
    if engine == "panic" {
        return Ok(Held::checked(Box::new(planted::Panics(
            ggtts_fake::FakeEngine::default(),
        ))));
    }
    if !matches!(engine, "qwen3-tts" | "kitten-tts-2") {
        return Err((GGTTS_ERR_ENGINE, format!("no engine named {engine:?}")));
    }
    let model = model.ok_or((GGTTS_ERR_ARG, format!("the {engine} engine needs a model")))?;
    check_gguf("model", model)?;
    let codec = codec.ok_or((GGTTS_ERR_ARG, format!("the {engine} engine needs a codec")))?;
    check_gguf("codec", codec)?;
    load_real(engine, Path::new(model), Path::new(codec), voice, threads)
}

#[cfg(feature = "ggml")]
fn load_real(engine: &str, model: &Path, codec: &Path, voice: Option<&str>, threads: i32) -> Result<Held, Fail> {
    let refused = |e: String| (GGTTS_ERR_ENGINE, e);
    match engine {
        "qwen3-tts" => ggtts_qwen3_tts::engine::Qwen3Tts::load(model, codec, threads, Default::default())
            // SAFETY: Qwen3-TTS is not `Send` to the compiler only for its raw ggml pointers (contexts, tensors,
            // graphs, backend buffers), every one allocated by this engine and reachable only through it. It holds no
            // `Rc` and no thread-local, and ggml's CPU backend keeps no per-thread state between graph runs, so the
            // engine is sound to move whole and use from one thread at a time.
            .map(|e| unsafe { Held::assume_send(Box::new(e)) })
            .map_err(|e| refused(e.to_string())),
        _ => {
            let voice = voice.ok_or((GGTTS_ERR_ARG, "the kitten-tts-2 engine needs a voice".to_string()))?;
            ggtts_kitten::engine::KittenEngine::open(
                model,
                codec,
                Path::new(voice),
                threads as usize,
                ggtts_kitten::backend::Device::Cpu,
            )
            // SAFETY: Kitten-TTS 2 is not `Send` to the compiler for its raw ggml pointers and for the `Rc<Backend>`
            // its parts share. Every clone of that `Rc` lives inside this engine, and the registry hands out no
            // reference to its parts, so the clones always move together and their count is only ever touched by the
            // one thread using the engine. It runs on the CPU backend here, which keeps no per-thread state.
            .map(|e| unsafe { Held::assume_send(Box::new(e)) })
            .map_err(refused)
        }
    }
}

#[cfg(not(feature = "ggml"))]
fn load_real(engine: &str, _: &Path, _: &Path, _: Option<&str>, _: i32) -> Result<Held, Fail> {
    Err((
        GGTTS_ERR_ENGINE,
        format!("this build has no {engine} engine (built without the ggml feature)"),
    ))
}

/// Text to the story path's chunks: one paragraph per non-empty line.
fn plan(engine: &dyn Engine, text: &str) -> Result<Vec<ggtts_core::text::PlannedChunk>, Fail> {
    let paragraphs: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    if paragraphs.is_empty() {
        return Err((GGTTS_ERR_ARG, "text is empty".into()));
    }
    plan_story(engine.profile(), &paragraphs).map_err(|e| (GGTTS_ERR_ARG, e.to_string()))
}

fn options(engine: &dyn Engine, seed: u64) -> RenderOptions {
    RenderOptions {
        seed_salt: seed.to_string(),
        ..RenderOptions::for_profile(engine.profile())
    }
}

fn render_error(e: RenderError) -> Fail {
    match e {
        RenderError::Sink(m) if m == STOPPED => (GGTTS_ERR_STOPPED, "the stream callback asked to stop".into()),
        other => (GGTTS_ERR_ENGINE, other.to_string()),
    }
}

const STOPPED: &str = "stopped by the callback";

/// Load an engine and write its handle.
///
/// # Safety
/// The strings are NULL or NUL-terminated; `handle` is NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn ggtts_load(
    engine: *const c_char,
    model: *const c_char,
    codec: *const c_char,
    voice: *const c_char,
    threads: i32,
    handle: *mut u64,
) -> i32 {
    guard(|| {
        if handle.is_null() {
            return Err((GGTTS_ERR_ARG, "handle is NULL".into()));
        }
        let e = open_engine(
            unsafe { arg(engine, "engine") }?,
            unsafe { opt_arg(model, "model") }?,
            unsafe { opt_arg(codec, "codec") }?,
            unsafe { opt_arg(voice, "voice") }?,
            threads,
        )?;
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        registry()
            .get_or_insert_with(HashMap::new)
            .insert(id, Arc::new(Mutex::new(e)));
        unsafe { *handle = id };
        Ok(())
    })
}

/// Render `text` to mono f32 PCM. Free the buffer with [`ggtts_pcm_free`].
///
/// # Safety
/// `text` is NUL-terminated; `pcm`, `len` and `rate` are writable.
#[no_mangle]
pub unsafe extern "C" fn ggtts_render(
    handle: u64,
    text: *const c_char,
    seed: u64,
    pcm: *mut *mut f32,
    len: *mut usize,
    rate: *mut u32,
) -> i32 {
    guard(|| {
        if pcm.is_null() || len.is_null() || rate.is_null() {
            return Err((GGTTS_ERR_ARG, "pcm, len and rate must not be NULL".into()));
        }
        let text = unsafe { arg(text, "text") }?;
        let read = with_engine(handle, |e| {
            let chunks = plan(e, text)?;
            let opts = options(e, seed);
            render_story(e, STORY, &chunks, &opts).map_err(render_error)
        })?;
        let boxed = read.samples.into_boxed_slice();
        let n = boxed.len();
        unsafe {
            *pcm = Box::into_raw(boxed) as *mut f32;
            *len = n;
            *rate = read.sample_rate;
        }
        Ok(())
    })
}

/// The stream callback: `user`, a piece's samples and its rate. Return 0 to go on, anything else to stop.
pub type PieceFn = extern "C" fn(user: *mut c_void, samples: *const f32, len: usize, rate: u32) -> i32;

/// Render `text`, handing each kept chunk's piece to `on_piece` before the level step.
///
/// # Safety
/// `text` is NUL-terminated; `on_piece` is a valid function; `user` is passed through untouched.
#[no_mangle]
pub unsafe extern "C" fn ggtts_render_stream(
    handle: u64,
    text: *const c_char,
    seed: u64,
    on_piece: Option<PieceFn>,
    user: *mut c_void,
) -> i32 {
    guard(|| {
        let on_piece = on_piece.ok_or((GGTTS_ERR_ARG, "on_piece is NULL".to_string()))?;
        let text = unsafe { arg(text, "text") }?;
        with_engine(handle, |e| {
            let chunks = plan(e, text)?;
            let opts = options(e, seed);
            let mut sink = |p: ggtts_core::pipeline::StreamPiece<'_>| {
                if on_piece(user, p.samples.as_ptr(), p.samples.len(), p.sample_rate) == 0 {
                    Ok(())
                } else {
                    Err(STOPPED.to_string())
                }
            };
            render_story_streamed(e, STORY, &chunks, &opts, &mut NullEvents, &mut sink).map_err(render_error)
        })?;
        Ok(())
    })
}

/// Free a buffer [`ggtts_render`] returned.
///
/// # Safety
/// `pcm` and `len` are exactly what one `ggtts_render` call wrote, freed once.
#[no_mangle]
pub unsafe extern "C" fn ggtts_pcm_free(pcm: *mut f32, len: usize) {
    if !pcm.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pcm, len)) });
    }
}

/// Release an engine. Its id is never reused, so a later call with it is [`GGTTS_ERR_HANDLE`]. It never waits: if the
/// engine is rendering (on another thread, or this call comes from its own stream callback), that render finishes and
/// the engine is dropped when it returns.
#[no_mangle]
pub extern "C" fn ggtts_free(handle: u64) -> i32 {
    guard(|| {
        // The registry guard is a temporary, so it is released before `slot`, and the engine with it, drops.
        let slot = registry().get_or_insert_with(HashMap::new).remove(&handle);
        slot.map(drop).ok_or_else(|| no_engine(handle))
    })
}

/// The calling thread's last error message, empty after a success. Valid until the thread's next call.
#[no_mangle]
pub extern "C" fn ggtts_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// The last error as a Rust string, for the JNI layer and the tests.
pub fn last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().to_string_lossy().into_owned())
}

#[cfg(feature = "plant")]
mod planted {
    use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
    use ggtts_core::profile::Profile;

    /// The fake engine, except that generating panics: the planted defect for the boundary test.
    pub struct Panics(pub ggtts_fake::FakeEngine);

    impl Engine for Panics {
        fn info(&self) -> &EngineInfo {
            self.0.info()
        }
        fn profile(&self) -> &Profile {
            self.0.profile()
        }
        fn generate(&mut self, _req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
            panic!("planted panic in generate")
        }
        fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
            self.0.decode(codes)
        }
    }
}

#[cfg(test)]
mod chunk_input_tests {
    use super::*;

    #[test]
    fn punctuation_only_is_an_argument_error_before_render() {
        let engine = ggtts_fake::FakeEngine::default();
        let (code, message) = plan(&engine, "？！").unwrap_err();
        assert_eq!(code, GGTTS_ERR_ARG);
        assert!(message.contains("punctuation-only"));
    }
    extern "C" fn count(user: *mut std::ffi::c_void, _: *const f32, _: usize, _: u32) -> i32 {
        unsafe {
            *(user as *mut usize) += 1;
        }
        0
    }

    struct CountingEngine {
        inner: ggtts_fake::FakeEngine,
        calls: Arc<AtomicU64>,
    }

    impl Engine for CountingEngine {
        fn info(&self) -> &ggtts_core::engine::EngineInfo {
            self.inner.info()
        }
        fn profile(&self) -> &ggtts_core::profile::Profile {
            self.inner.profile()
        }
        fn generate(
            &mut self,
            req: &ggtts_core::engine::ChunkRequest<'_>,
        ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.generate(req)
        }
        fn decode(
            &mut self,
            codes: &ggtts_core::engine::Codes,
        ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
            self.inner.decode(codes)
        }
    }

    #[test]
    fn exported_render_and_stream_reject_before_any_output() {
        use std::ffi::CString;
        let text = CString::new("好。\n？！").unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let engine = CountingEngine {
            inner: ggtts_fake::FakeEngine::default(),
            calls: calls.clone(),
        };
        let handle = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        registry()
            .get_or_insert_with(HashMap::new)
            .insert(handle, Arc::new(Mutex::new(Held::checked(Box::new(engine)))));
        let (mut pcm, mut len, mut rate) = (std::ptr::null_mut(), 0, 0);
        assert_eq!(
            unsafe { ggtts_render(handle, text.as_ptr(), 1, &mut pcm, &mut len, &mut rate) },
            GGTTS_ERR_ARG
        );
        assert!(pcm.is_null());
        let mut callbacks = 0usize;
        assert_eq!(
            unsafe {
                ggtts_render_stream(
                    handle,
                    text.as_ptr(),
                    1,
                    Some(count),
                    &mut callbacks as *mut usize as *mut std::ffi::c_void,
                )
            },
            GGTTS_ERR_ARG
        );
        assert_eq!(callbacks, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "invalid input invoked generate");
        assert_eq!(ggtts_free(handle), GGTTS_OK);
    }
}
