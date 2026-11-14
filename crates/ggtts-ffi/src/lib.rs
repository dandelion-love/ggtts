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
//!   ([`ggtts_load`], [`ggtts_load_with`], [`ggtts_render`], [`ggtts_render_stream`], [`ggtts_free`]) run inside
//!   `catch_unwind`, and a panic comes back as [`GGTTS_ERR_PANIC`] with its message; the engine that panicked is
//!   dropped from the registry. [`ggtts_pcm_free`], [`ggtts_last_error`] and [`ggtts_abi_version`] are not wrapped:
//!   one drops a box, one reads a thread-local, one returns a constant. `catch_unwind` catches unwinding panics only: an abort, a stack overflow or a crash in C code
//!   still ends the process.
//! * **Errors are a code plus a message:** [`ggtts_last_error`] holds the calling thread's last message.
//! * **The surface is versioned and only grows.** [`GGTTS_ABI_VERSION`] (and [`ggtts_abi_version`]) goes up by one
//!   whenever a function, a struct or a code is added; an exported function never changes its signature or its
//!   behaviour. Version 2 added [`ggtts_load_with`] and [`GgttsSettings`] (a Qwen3-TTS speaker and style
//!   instruction). Version 3 added three trailing fields to [`GgttsSettings`]: the device, and the default model
//!   files' cache directory and download permission. A version-2 struct (its `size` is
//!   [`GGTTS_SETTINGS_V2_SIZE`]) still loads exactly as it did.
//!
//! With the `jni` feature, `jni.rs` adds the `Java_ggtts_Native_*` entry points over the same functions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use ggtts_load::binding::{self, Held, Slot};

#[cfg(feature = "jni")]
mod jni;

pub const GGTTS_OK: i32 = binding::OK;
/// A NULL, non-UTF-8 or out-of-range argument.
pub const GGTTS_ERR_ARG: i32 = binding::ERR_ARG;
/// No engine with this id: never loaded, already freed, or `0`.
pub const GGTTS_ERR_HANDLE: i32 = binding::ERR_HANDLE;
/// The engine refused: a missing file, not a GGUF, an unknown engine, a failed render.
pub const GGTTS_ERR_ENGINE: i32 = binding::ERR_ENGINE;
/// A panic inside ggtts, caught at the boundary.
pub const GGTTS_ERR_PANIC: i32 = binding::ERR_PANIC;
/// The stream callback asked to stop.
pub const GGTTS_ERR_STOPPED: i32 = binding::ERR_STOPPED;
/// The engine is rendering: a call on it from its own stream callback, or from another thread meanwhile.
pub const GGTTS_ERR_BUSY: i32 = binding::ERR_BUSY;

/// The C-ABI's version: 1 was the surface before [`ggtts_load_with`]; 2 added it and [`GgttsSettings`]; 3 added the
/// struct's `device`, `cache_dir` and `allow_download` fields.
pub const GGTTS_ABI_VERSION: u32 = 3;

/// The settings [`ggtts_load_with`] takes, by pointer. `size` is `sizeof(ggtts_settings)` as the caller's header
/// declares it, so a later version can add fields and still tell an older caller's struct apart. A NULL or empty
/// string is no setting: the result is what [`ggtts_load`] gives.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GgttsSettings {
    /// `sizeof(ggtts_settings)`.
    pub size: usize,
    /// A Qwen3-TTS speaker name (case is ignored); NULL or empty for the engine's default speaker.
    pub speaker: *const c_char,
    /// A Qwen3-TTS style instruction ("Speak slowly and softly."); NULL or empty for none.
    pub instruction: *const c_char,
    /// Version 3: `cpu` or `cuda`; NULL or empty for `cpu`.
    pub device: *const c_char,
    /// Version 3: the root of the default model files' cache; NULL or empty for ggtts-render's (`GGTTS_CACHE_DIR`,
    /// else the platform's per-user cache directory).
    pub cache_dir: *const c_char,
    /// Version 3: 1 downloads a default model file missing from the cache; 0 (the default) never downloads.
    pub allow_download: i32,
}

/// `sizeof(ggtts_settings)` in a version-2 header: `size`, `speaker` and `instruction`. A struct of this size loads
/// as it did in version 2: on the CPU, from the paths given only.
pub const GGTTS_SETTINGS_V2_SIZE: usize = std::mem::offset_of!(GgttsSettings, device);

/// The C-ABI's version, [`GGTTS_ABI_VERSION`] of the library actually loaded: a caller built against a newer header
/// checks it before calling a function the library may not have.
#[no_mangle]
pub extern "C" fn ggtts_abi_version() -> u32 {
    GGTTS_ABI_VERSION
}

type Fail = binding::Fail;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Run `f`, turning an error or a panic into a status ([`binding::guard`]), and record the message.
fn guard(f: impl FnOnce() -> Result<(), Fail>) -> i32 {
    match binding::guard(f) {
        Ok(()) => {
            set_error("");
            GGTTS_OK
        }
        Err((code, msg)) => {
            set_error(&msg);
            code
        }
    }
}

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

/// Run `f` on the engine ([`binding::with_engine`]). The registry lock is released before `f` runs, so `f` (and a
/// stream callback inside it) may call any entry point; the engine's own lock is taken with `try_lock`, so a second
/// call on a rendering engine is [`GGTTS_ERR_BUSY`]. If `f` panics, the engine is dropped from the registry and the
/// call is [`GGTTS_ERR_PANIC`]. `f` gets the engine and the instruction it was loaded with.
fn with_engine<T>(
    handle: u64,
    f: impl FnOnce(&mut dyn ggtts_core::engine::Engine, Option<&str>) -> Result<T, Fail>,
) -> Result<T, Fail> {
    let slot = registry()
        .get_or_insert_with(HashMap::new)
        .get(&handle)
        .cloned()
        .ok_or_else(|| no_engine(handle))?;
    binding::with_engine(
        &slot,
        |why| match why {
            binding::Unusable::Busy => (
                GGTTS_ERR_BUSY,
                format!(
                    "the engine with handle {handle} is rendering (a call from its own stream callback, or from \
                     another thread)"
                ),
            ),
            // A panic inside a render evicts the engine (below), so a call that looked it up just before the
            // eviction never uses it either.
            binding::Unusable::Poisoned | binding::Unusable::Panicked => no_engine(handle),
        },
        // The registry is never locked while an engine's lock is waited on, so taking it here cannot deadlock.
        || {
            registry().get_or_insert_with(HashMap::new).remove(&handle);
        },
        f,
    )
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

/// The device setting: `cpu` or `cuda`; `None` (no setting) is `cpu`.
fn parse_device(s: Option<&str>) -> Result<binding::Device, Fail> {
    match s {
        None => Ok(binding::Device::Cpu),
        Some(name) => binding::Device::from_name(name).ok_or_else(|| {
            (
                GGTTS_ERR_ARG,
                format!("settings.device must be cpu or cuda, not {name:?}"),
            )
        }),
    }
}

/// How a version-3 load finds the default model files: the cache root (`None`: ggtts-render's) and whether a missing
/// file may be downloaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Models<'a> {
    cache_dir: Option<&'a str>,
    allow_download: bool,
}

/// The settings of one load: a Qwen3-TTS speaker and style instruction, the device, and (version 3 only) how the
/// default model files are found, `None` for each one not given. An empty string is no setting. `models` is `None`
/// for [`ggtts_load`], NULL settings and a version-2 struct, which load only the paths given, as they always have.
#[derive(Debug, Clone, Copy, Default)]
struct Settings<'a> {
    speaker: Option<&'a str>,
    instruction: Option<&'a str>,
    device: Option<&'a str>,
    models: Option<Models<'a>>,
}

impl<'a> Settings<'a> {
    fn new(speaker: Option<&'a str>, instruction: Option<&'a str>) -> Self {
        Settings {
            speaker: speaker.filter(|s| !s.is_empty()),
            instruction: instruction.filter(|s| !s.is_empty()),
            ..Settings::default()
        }
    }

    /// The settings struct a C caller passed: NULL is no settings. A version-2 struct (`size` is
    /// [`GGTTS_SETTINGS_V2_SIZE`]) is read up to `instruction` and nothing after it.
    ///
    /// # Safety
    /// `p` is NULL or points to a readable [`GgttsSettings`] of at least `size` bytes whose strings are NULL or
    /// NUL-terminated.
    unsafe fn from_c(p: *const GgttsSettings) -> Result<Self, Fail> {
        if p.is_null() {
            return Ok(Settings::default());
        }
        let size = unsafe { std::ptr::addr_of!((*p).size).read_unaligned() };
        let known = std::mem::size_of::<GgttsSettings>();
        if size != known && size != GGTTS_SETTINGS_V2_SIZE {
            return Err((
                GGTTS_ERR_ARG,
                format!(
                    "settings.size is {size}; this library (C-ABI version {GGTTS_ABI_VERSION}) reads a ggtts_settings \
                     of {known} bytes, or {GGTTS_SETTINGS_V2_SIZE} from a version-2 header: set size to \
                     sizeof(ggtts_settings)"
                ),
            ));
        }
        // Only the fields the caller's struct has are read: a version-2 caller's memory ends at `instruction`.
        let speaker = unsafe { std::ptr::addr_of!((*p).speaker).read_unaligned() };
        let instruction = unsafe { std::ptr::addr_of!((*p).instruction).read_unaligned() };
        let mut settings = Settings::new(unsafe { opt_arg(speaker, "settings.speaker") }?, unsafe {
            opt_arg(instruction, "settings.instruction")
        }?);
        if size == known {
            let s = unsafe { &*p };
            let empty_is_none = |o: Option<&'a str>| o.filter(|s| !s.is_empty());
            settings.device = empty_is_none(unsafe { opt_arg(s.device, "settings.device") }?);
            let allow_download = match s.allow_download {
                0 => false,
                1 => true,
                other => {
                    return Err((
                        GGTTS_ERR_ARG,
                        format!("settings.allow_download must be 0 or 1, not {other}"),
                    ))
                }
            };
            settings.models = Some(Models {
                cache_dir: empty_is_none(unsafe { opt_arg(s.cache_dir, "settings.cache_dir") }?),
                allow_download,
            });
        }
        Ok(settings)
    }
}

/// How the C-ABI words the refusals that name its own settings ([`binding::Wording`]).
const WORDING: binding::Wording = binding::Wording {
    choose_speaker: ". Choose one with the speaker setting",
    device_setting: "settings.device",
    cpu_only,
    no_engines_for_defaults: "",
    offline_hint: " (downloading is off: allow it with the allow_download setting)",
    file_checks: binding::FileChecks::InTurn,
};

fn cpu_only(engine: &str, _: binding::Device) -> String {
    format!("the {engine} engine runs on the CPU only")
}

/// The cache a version-3 load resolves the default model files in: `cache_dir`, else ggtts-render's.
fn cache_for(cache_dir: Option<&str>) -> Result<ggtts_models::Cache, String> {
    match cache_dir {
        Some(dir) => Ok(ggtts_models::Cache::new(PathBuf::from(dir))),
        None => ggtts_models::Cache::default_root()
            .map(ggtts_models::Cache::new)
            .map_err(|e| format!("{e}, or the cache_dir setting")),
    }
}

/// The engine for a name ([`binding::open`]). The fake engine needs no files. With version-3 settings and engine
/// `qwen3-tts`, a model or codec not given is the pinned default `ggtts-render` uses, from the same cache and verified
/// against the same sha256, downloaded only when `allow_download` is set.
fn open_engine(
    engine: &str,
    model: Option<&str>,
    codec: Option<&str>,
    voice: Option<&str>,
    threads: i32,
    settings: Settings<'_>,
) -> Result<Held, Fail> {
    // The C-ABI refuses a setting the engine does not take before it reads the device; `binding::open` checks again.
    if engine != "qwen3-tts" {
        binding::refuse_settings(engine, settings.speaker, settings.instruction)?;
    }
    let device = parse_device(settings.device)?;
    let cache_dir = settings.models.and_then(|m| m.cache_dir);
    let cache = || cache_for(cache_dir);
    let http = ggtts_models::Http::default();
    let models = settings.models.map(|m| binding::Models {
        cache: &cache,
        offline: !m.allow_download,
        fetch: &http,
    });
    let opened = binding::open(
        binding::Request {
            engine,
            model: model.map(PathBuf::from),
            codec: codec.map(PathBuf::from),
            voice: voice.map(PathBuf::from),
            speaker: settings.speaker,
            instruction: settings.instruction,
            device,
            threads,
            models,
        },
        &WORDING,
        &mut |_| {},
    )?;
    Ok(opened.held)
}

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
    guard(|| unsafe { load(engine, model, codec, voice, threads, Settings::default(), handle) })
}

/// [`ggtts_load`] with voice settings (C-ABI version 2). `settings` may be NULL, and a NULL or empty string in it is
/// no setting: with neither set, the engine and its output are what `ggtts_load` gives. A setting is refused, never
/// ignored: an unknown speaker is [`GGTTS_ERR_ENGINE`] with the model's speakers in the message; an instruction on a
/// checkpoint that takes none (a Base checkpoint, or a `custom_voice` one below 1.7B) is `GGTTS_ERR_ENGINE` with the
/// library's message; either setting on an engine other than `qwen3-tts` is [`GGTTS_ERR_ARG`]. Each refusal comes before
/// any text is rendered, and no handle is written.
///
/// Version 3 (a struct of the full size): `device` chooses `cpu` (the default) or `cuda`, and a build without the CUDA
/// backend refuses `cuda` with [`GGTTS_ERR_ENGINE`] before reading any file. With engine `qwen3-tts` and the model or
/// codec NULL, the missing file is the pinned default `ggtts-render` uses, from the same cache and verified against the
/// same sha256, downloaded only when `allow_download` is 1; and with no model and no speaker, the speaker is the
/// default model's default. A version-2 struct, or NULL settings, loads only the paths given, on the CPU, as before.
///
/// # Safety
/// As [`ggtts_load`]; `settings` is NULL or points to a readable [`GgttsSettings`] of at least `size` bytes, whose
/// `size` is set.
#[no_mangle]
pub unsafe extern "C" fn ggtts_load_with(
    engine: *const c_char,
    model: *const c_char,
    codec: *const c_char,
    voice: *const c_char,
    threads: i32,
    settings: *const GgttsSettings,
    handle: *mut u64,
) -> i32 {
    guard(|| unsafe {
        let settings = Settings::from_c(settings)?;
        load(engine, model, codec, voice, threads, settings, handle)
    })
}

/// The body of both loads.
///
/// # Safety
/// As [`ggtts_load`].
unsafe fn load(
    engine: *const c_char,
    model: *const c_char,
    codec: *const c_char,
    voice: *const c_char,
    threads: i32,
    settings: Settings<'_>,
    handle: *mut u64,
) -> Result<(), Fail> {
    if handle.is_null() {
        return Err((GGTTS_ERR_ARG, "handle is NULL".into()));
    }
    let e = open_engine(
        unsafe { arg(engine, "engine") }?,
        unsafe { opt_arg(model, "model") }?,
        unsafe { opt_arg(codec, "codec") }?,
        unsafe { opt_arg(voice, "voice") }?,
        threads,
        settings,
    )?;
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    registry().get_or_insert_with(HashMap::new).insert(id, binding::slot(e));
    unsafe { *handle = id };
    Ok(())
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
        let (samples, sample_rate) = with_engine(handle, |e, instruction| binding::render(e, text, seed, instruction))?;
        let boxed = samples.into_boxed_slice();
        let n = boxed.len();
        unsafe {
            *pcm = Box::into_raw(boxed) as *mut f32;
            *len = n;
            *rate = sample_rate;
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
        with_engine(handle, |e, instruction| {
            binding::render_stream(e, text, seed, instruction, &mut |samples, rate| {
                on_piece(user, samples.as_ptr(), samples.len(), rate) == 0
            })
        })
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

/// The backend a loaded engine runs on, as its engine reports it (`ggml-cpu`, `ggml-cuda`, or the fake engine's), for
/// a Rust host and the tests. Not part of the C-ABI.
pub fn backend_of(handle: u64) -> Result<String, String> {
    with_engine(handle, |e, _| Ok(e.info().backend.clone())).map_err(|(_, m)| m)
}

/// The last error as a Rust string, for the JNI layer and the tests.
pub fn last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().to_string_lossy().into_owned())
}

#[cfg(test)]
mod chunk_input_tests {
    use super::*;
    use ggtts_core::engine::Engine;
    use std::sync::Arc;

    #[test]
    fn punctuation_only_is_an_argument_error_before_render() {
        let engine = ggtts_fake::FakeEngine::default();
        let (code, message) = binding::plan(&engine, "？！").unwrap_err();
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
            .insert(handle, binding::slot(Held::checked(Box::new(engine), None)));
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

    /// The fake engine, recording each request's instruction and rendering it as none.
    struct Recording {
        inner: ggtts_fake::FakeEngine,
        seen: Arc<Mutex<Vec<Option<String>>>>,
    }

    impl Engine for Recording {
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
            self.seen.lock().unwrap().push(req.instruction.map(String::from));
            self.inner.generate(&ggtts_core::engine::ChunkRequest {
                instruction: None,
                ..*req
            })
        }
        fn decode(
            &mut self,
            codes: &ggtts_core::engine::Codes,
        ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
            self.inner.decode(codes)
        }
    }

    /// What every request of a render and a stream carries, on an engine held with `instruction`.
    fn seen_with(instruction: Option<&str>) -> Vec<Option<String>> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let engine = Recording {
            inner: ggtts_fake::FakeEngine::default(),
            seen: seen.clone(),
        };
        let handle = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        registry()
            .get_or_insert_with(HashMap::new)
            .insert(handle, binding::slot(Held::checked(Box::new(engine), instruction)));
        let text = std::ffi::CString::new("今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。").unwrap();
        let (mut pcm, mut len, mut rate) = (std::ptr::null_mut(), 0, 0);
        assert_eq!(
            unsafe { ggtts_render(handle, text.as_ptr(), 1, &mut pcm, &mut len, &mut rate) },
            GGTTS_OK,
            "{}",
            last_error()
        );
        unsafe { ggtts_pcm_free(pcm, len) };
        let mut pieces = 0usize;
        assert_eq!(
            unsafe {
                ggtts_render_stream(
                    handle,
                    text.as_ptr(),
                    1,
                    Some(count),
                    &mut pieces as *mut usize as *mut std::ffi::c_void,
                )
            },
            GGTTS_OK,
            "{}",
            last_error()
        );
        assert_eq!(ggtts_free(handle), GGTTS_OK);
        let seen = seen.lock().unwrap().clone();
        assert!(seen.len() >= 4, "two chunks on each path: {seen:?}");
        seen
    }

    /// The instruction an engine is loaded with reaches every request of its renders and streams; without one, none.
    #[test]
    fn the_loaded_instruction_reaches_every_request() {
        let seen = seen_with(Some("Speak slowly."));
        assert!(seen.iter().all(|i| i.as_deref() == Some("Speak slowly.")), "{seen:?}");
        let seen = seen_with(None);
        assert!(seen.iter().all(Option::is_none), "{seen:?}");
    }

    /// An empty string in the settings is no setting, so a caller passing `""` gets what `ggtts_load` gives.
    #[test]
    fn empty_settings_are_no_settings() {
        let s = Settings::new(Some(""), Some(""));
        assert_eq!((s.speaker, s.instruction), (None, None));
        let s = Settings::new(Some("hk_voice_777"), Some("Speak slowly."));
        assert_eq!(
            (s.speaker, s.instruction),
            (Some("hk_voice_777"), Some("Speak slowly."))
        );
    }
}

#[cfg(test)]
mod default_model_tests {
    use super::*;
    use std::cell::RefCell;

    /// A fetcher that records each URL it is asked for and fails, so a test sees whether a download was tried.
    #[derive(Default)]
    struct Recording(RefCell<Vec<String>>);

    impl ggtts_models::Fetch for Recording {
        fn open(&self, url: &str) -> Result<Box<dyn std::io::Read>, String> {
            self.0.borrow_mut().push(url.to_string());
            Err("no network in this test".into())
        }
    }

    fn empty_cache(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("ggtts-ffi-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    /// A cache already holding the default files, as ggtts-render's tests make one: each a sparse file of the pinned
    /// size with the record of an earlier check, so nothing is downloaded or hashed.
    fn prepared_cache(name: &str) -> PathBuf {
        let root = empty_cache(name);
        let cache = ggtts_models::Cache::new(root.clone());
        for p in ggtts_models::QWEN3_TTS.files {
            let path = cache.path_for(p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(p.size).unwrap();
            let mtime = std::fs::metadata(&path)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let record = serde_json::json!({"sha256": p.sha256, "size": p.size, "mtime_ns": mtime.to_string()});
            std::fs::write(
                path.with_file_name(format!("{}.verified", p.file_name)),
                record.to_string(),
            )
            .unwrap();
        }
        root
    }

    /// The default files a version-3 load with these settings resolves, as [`open_engine`] resolves them: the C-ABI's
    /// cache and wording.
    fn default_files(
        model: Option<&str>,
        codec: Option<&str>,
        models: Models<'_>,
        engines_built: bool,
        fetch: &Recording,
    ) -> Result<binding::DefaultFiles, Fail> {
        let cache = || cache_for(models.cache_dir);
        let models = binding::Models {
            cache: &cache,
            offline: !models.allow_download,
            fetch,
        };
        binding::resolve_defaults(
            &ggtts_models::QWEN3_TTS,
            model.map(PathBuf::from),
            codec.map(PathBuf::from),
            &models,
            engines_built,
            &WORDING,
            &mut |_| {},
        )
    }

    fn resolve(
        root: &std::path::Path,
        model: Option<&str>,
        codec: Option<&str>,
        allow_download: bool,
        fetch: &Recording,
    ) -> Result<binding::DefaultFiles, Fail> {
        let models = Models {
            cache_dir: Some(root.to_str().unwrap()),
            allow_download,
        };
        default_files(model, codec, models, true, fetch)
    }

    /// With no paths, the pinned default files from the cache and the default model's default speaker; a given path
    /// is used as it is, and a given model keeps its own speaker. Nothing is fetched.
    #[test]
    fn the_default_files_and_speaker_come_from_the_cache() {
        let root = prepared_cache("defaults");
        let cache = ggtts_models::Cache::new(root.clone());
        let [talker, codec] = ggtts_models::QWEN3_TTS.files else {
            panic!("two default files")
        };
        let fetch = Recording::default();
        let files = resolve(&root, None, None, false, &fetch).unwrap();
        assert_eq!(
            (files.model, files.codec, files.speaker),
            (cache.path_for(talker), cache.path_for(codec), Some("hk_voice_343"))
        );
        let files = resolve(&root, Some("my-talker.gguf"), None, false, &fetch).unwrap();
        assert_eq!(files.model, PathBuf::from("my-talker.gguf"));
        assert_eq!(files.codec, cache.path_for(codec));
        assert_eq!(files.speaker, None, "a given model keeps its own default speaker");
        let files = resolve(&root, None, Some("my-codec.gguf"), false, &fetch).unwrap();
        assert_eq!(files.model, cache.path_for(talker));
        assert_eq!(files.codec, PathBuf::from("my-codec.gguf"));
        assert_eq!(files.speaker, Some("hk_voice_343"));
        assert!(fetch.0.borrow().is_empty(), "fetched {:?}", fetch.0.borrow());
        let _ = std::fs::remove_dir_all(root);
    }

    /// Downloading off (the default): a file missing from the cache is an error naming the file and its URL, and
    /// nothing is fetched. Downloading on: the pinned URL is fetched, and its failure is the error.
    #[test]
    fn a_missing_file_is_downloaded_only_when_allowed() {
        let root = empty_cache("offline");
        let fetch = Recording::default();
        let (code, msg) = resolve(&root, None, None, false, &fetch).unwrap_err();
        assert_eq!(code, GGTTS_ERR_ENGINE);
        for p in ggtts_models::QWEN3_TTS.files {
            assert!(msg.contains(p.file_name) && msg.contains(p.url), "{msg}");
        }
        assert!(msg.contains("allow_download"), "{msg}");
        assert!(fetch.0.borrow().is_empty(), "fetched {:?}", fetch.0.borrow());

        let (code, msg) = resolve(&root, None, None, true, &fetch).unwrap_err();
        assert_eq!(code, GGTTS_ERR_ENGINE);
        let talker = &ggtts_models::QWEN3_TTS.files[0];
        assert_eq!(*fetch.0.borrow(), [talker.url.to_string()]);
        assert!(
            msg.contains("no network in this test") && msg.contains(talker.url),
            "{msg}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A build without engines looks nothing up and fetches nothing, as ggtts-render.
    #[test]
    fn a_build_without_engines_fetches_nothing() {
        let root = empty_cache("no-engines");
        let fetch = Recording::default();
        let models = Models {
            cache_dir: Some(root.to_str().unwrap()),
            allow_download: true,
        };
        let (code, msg) = default_files(None, None, models, false, &fetch).unwrap_err();
        assert_eq!(code, GGTTS_ERR_ENGINE);
        assert_eq!(
            msg,
            "this build has no qwen3-tts engine (built without the ggml feature)"
        );
        assert!(fetch.0.borrow().is_empty());
        assert!(!root.exists(), "the cache was touched");
    }

    /// A version-2 struct is read up to `instruction` only: whatever follows it in the caller's memory is not a
    /// setting, so it loads only the paths given, on the CPU. A version-3 struct reads every field.
    #[test]
    fn a_version_2_struct_reads_no_version_3_field() {
        #[repr(C)]
        struct V2ThenJunk {
            size: usize,
            speaker: *const c_char,
            instruction: *const c_char,
            junk: [u8; 64],
        }
        let v2 = V2ThenJunk {
            size: GGTTS_SETTINGS_V2_SIZE,
            speaker: c"hk_voice_777".as_ptr(),
            instruction: std::ptr::null(),
            junk: [0xff; 64],
        };
        assert_eq!(GGTTS_SETTINGS_V2_SIZE, 3 * std::mem::size_of::<usize>());
        let s = unsafe { Settings::from_c(&v2 as *const V2ThenJunk as *const GgttsSettings) }.unwrap();
        assert_eq!(s.speaker, Some("hk_voice_777"));
        assert_eq!((s.device, s.models), (None, None));

        let v3 = GgttsSettings {
            size: std::mem::size_of::<GgttsSettings>(),
            speaker: std::ptr::null(),
            instruction: std::ptr::null(),
            device: c"cuda".as_ptr(),
            cache_dir: c"".as_ptr(),
            allow_download: 1,
        };
        let s = unsafe { Settings::from_c(&v3) }.unwrap();
        assert_eq!(s.device, Some("cuda"));
        assert_eq!(
            s.models,
            Some(Models {
                cache_dir: None,
                allow_download: true
            })
        );
        let bad = GgttsSettings {
            allow_download: 2,
            ..v3
        };
        let (code, msg) = unsafe { Settings::from_c(&bad) }.unwrap_err();
        assert_eq!(code, GGTTS_ERR_ARG);
        assert_eq!(msg, "settings.allow_download must be 0 or 1, not 2");
    }
}
