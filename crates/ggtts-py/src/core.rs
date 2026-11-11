//! The binding's engine layer, in plain Rust: loading with the settings of the C-ABI's `ggtts_load_with`, rendering,
//! streaming with a stop, and the guard that turns an error or a panic into a code and a message. `lib.rs` puts the
//! Python module over it; nothing here needs Python, so `cargo test` runs it without an interpreter.
//!
//! It mirrors `crates/ggtts-ffi/src/lib.rs` on the engine API directly (no C strings, no handle registry): the same
//! codes, the same refusals before any file is read or any text rendered, the same story path, and a panic caught at
//! the boundary, with the engine that panicked dropped.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, TryLockError};

use ggtts_core::engine::{refuse_instruction, ChunkRequest, Engine};
use ggtts_core::pipeline::{plan_story, render_story, render_story_streamed, RenderError, RenderOptions};
use ggtts_core::trace::NullEvents;

/// The codes are the C-ABI's (`ggtts.h`): a test holds them equal to `ggtts-ffi`'s.
pub const OK: i32 = 0;
/// An invalid argument: an unknown device, a setting the engine does not take, empty or punctuation-only text.
pub const ERR_ARG: i32 = -1;
/// The engine was closed, or dropped after a panic.
pub const ERR_HANDLE: i32 = -2;
/// The engine refused: a missing file, not a GGUF, an unknown engine or speaker, a failed download, a failed render.
pub const ERR_ENGINE: i32 = -3;
/// A panic inside ggtts, caught at the boundary.
pub const ERR_PANIC: i32 = -4;
/// The stream was stopped. The Python stream returns instead of raising it; it is here for parity.
pub const ERR_STOPPED: i32 = -5;
/// The engine is rendering: a call on it from its own stream callback, or from another thread meanwhile.
pub const ERR_BUSY: i32 = -6;

/// The seed a render uses when given none, as `ggtts-render`.
pub const DEFAULT_SEED: u64 = 20260905;

/// The story id every binding render uses, as the C-ABI: the seed of each chunk is derived from it, the text and the
/// seed given, so a render here is the C-ABI's render at the same seed.
const STORY: &str = "ggtts-binding";
const STOPPED: &str = "stopped by the callback";

/// A code and a message.
pub type Fail = (i32, String);

fn arg_err(msg: impl Into<String>) -> Fail {
    (ERR_ARG, msg.into())
}

fn engine_err(msg: impl Into<String>) -> Fail {
    (ERR_ENGINE, msg.into())
}

/// The message of a caught panic.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    let msg = p
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".into());
    format!("panic inside ggtts: {msg}")
}

/// Run `f`, turning a panic into [`ERR_PANIC`] with its message.
pub fn guard<T>(f: impl FnOnce() -> Result<T, Fail>) -> Result<T, Fail> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| Err((ERR_PANIC, panic_message(p.as_ref()))))
}

/// Where the graphs run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// The first CUDA device. Needs a build with the `cuda` feature; another build refuses it when loading.
    Cuda,
}

impl Device {
    pub fn parse(s: &str) -> Result<Self, Fail> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda),
            other => Err(arg_err(format!("device must be \"cpu\" or \"cuda\", not {other:?}"))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        }
    }
}

/// What to load. A path not given is the engine's pinned default where it has one (Qwen3-TTS), from the per-user
/// cache, downloaded first unless `offline`. An empty speaker or instruction is no setting, as in the C-ABI.
#[derive(Debug, Clone, Default)]
pub struct LoadRequest {
    pub engine: String,
    pub model: Option<PathBuf>,
    pub codec: Option<PathBuf>,
    pub voice: Option<PathBuf>,
    pub speaker: Option<String>,
    pub instruction: Option<String>,
    pub device: Device,
    /// 0 or less is the C-ABI's default, 4.
    pub threads: i32,
    pub offline: bool,
}

/// One model file a loaded engine uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    pub role: String,
    pub path: PathBuf,
    /// `path` (given) or `cache` (the pinned default, verified).
    pub source: String,
}

/// What a load gives besides the engine: the settings in effect and where the files came from.
#[derive(Debug, Clone, Default)]
pub struct Loaded {
    pub engine: String,
    pub family: String,
    pub backend: String,
    pub parity: String,
    /// The speaker in effect: the one given, or the default model's default speaker; `None` for the engine's own.
    pub speaker: Option<String>,
    pub instruction: Option<String>,
    pub device: Device,
    pub files: Vec<ModelFile>,
}

/// An engine and the instruction its renders carry. `Engine` has no `Send` bound and the real engines are not `Send` to
/// the compiler (raw ggml pointers; Kitten-TTS 2 shares its backend through `Rc`), so a `Held` is built in one of two
/// ways only: [`Held::checked`], where the compiler proves the engine `Send`, or [`Held::assume_send`], whose every
/// caller argues it for that one engine, as in the C-ABI.
struct Held {
    engine: Box<dyn Engine>,
    instruction: Option<String>,
    /// The engine panicked: it is never used again.
    dead: bool,
}

impl Held {
    fn checked(e: Box<dyn Engine + Send>) -> Self {
        Held {
            engine: e,
            instruction: None,
            dead: false,
        }
    }

    /// # Safety
    /// The engine may be moved to another thread whole and used there by one thread at a time: nothing in it is tied
    /// to the thread that made it, and nothing it shares (an `Rc`, a pointer) is reachable from outside it.
    #[cfg_attr(not(feature = "ggml"), allow(dead_code))]
    unsafe fn assume_send(e: Box<dyn Engine>) -> Self {
        Held {
            engine: e,
            instruction: None,
            dead: false,
        }
    }
}

// SAFETY: every `Held` is `Send` by the compiler's proof or by its constructor's per-engine argument (above), and its
// engine is used under its own mutex, one call at a time.
unsafe impl Send for Held {}

/// A loaded engine. Its lock is taken with `try_lock`, so a second call while it renders is [`ERR_BUSY`], never a
/// deadlock; [`Handle::close`] never waits for a render, which drops the engine when it returns. A panic inside a
/// render drops the engine, and later calls are [`ERR_HANDLE`].
pub struct Handle {
    slot: Mutex<Option<Arc<Mutex<Held>>>>,
}

impl Handle {
    fn new(held: Held) -> Self {
        Handle {
            slot: Mutex::new(Some(Arc::new(Mutex::new(held)))),
        }
    }

    /// A handle over any engine, for tests of the boundary.
    #[cfg(test)]
    fn from_engine(e: Box<dyn Engine + Send>, instruction: Option<&str>) -> Self {
        let mut held = Held::checked(e);
        held.instruction = instruction.map(String::from);
        Handle::new(held)
    }

    fn slot(&self) -> std::sync::MutexGuard<'_, Option<Arc<Mutex<Held>>>> {
        // Nothing panics while this lock is held, so a poisoned lock still holds a whole value.
        self.slot.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Drop the engine: a later call is [`ERR_HANDLE`]. Returns whether it was open.
    pub fn close(&self) -> bool {
        // The guard is a temporary, released before the engine (if no render holds it) drops.
        let taken = self.slot().take();
        taken.is_some()
    }

    pub fn is_closed(&self) -> bool {
        self.slot().is_none()
    }

    /// Run `f` on the engine, catching a panic: the engine is dropped and the panic becomes [`ERR_PANIC`].
    fn with_engine<T>(&self, f: impl FnOnce(&mut dyn Engine, Option<&str>) -> Result<T, Fail>) -> Result<T, Fail> {
        let slot = self
            .slot()
            .clone()
            .ok_or_else(|| (ERR_HANDLE, "the engine is closed".to_string()))?;
        let mut held = match slot.try_lock() {
            Ok(h) => h,
            Err(TryLockError::WouldBlock) => {
                return Err((
                    ERR_BUSY,
                    "the engine is rendering (a call from its own stream callback, or from another thread)".into(),
                ))
            }
            // Nothing panics while the lock is held (a panic is caught inside it), but a poisoned lock is closed too.
            Err(TryLockError::Poisoned(_)) => return Err((ERR_HANDLE, "the engine is closed".to_string())),
        };
        if held.dead {
            return Err((ERR_HANDLE, "the engine is closed (it panicked)".to_string()));
        }
        let h = &mut *held;
        match catch_unwind(AssertUnwindSafe(|| f(h.engine.as_mut(), h.instruction.as_deref()))) {
            Ok(r) => r,
            Err(p) => {
                let msg = panic_message(p.as_ref());
                // Only this slot is dropped, and only if it is still the one that panicked.
                let mut current = self.slot();
                if current.as_ref().is_some_and(|s| Arc::ptr_eq(s, &slot)) {
                    *current = None;
                }
                drop(current);
                // A call that took a clone of the slot before it was dropped sees the engine closed, never uses it.
                held.dead = true;
                Err((ERR_PANIC, msg))
            }
        }
    }

    /// Render `text` (one paragraph per non-empty line) to mono f32 samples and their rate.
    pub fn render(&self, text: &str, seed: u64) -> Result<(Vec<f32>, u32), Fail> {
        self.with_engine(|e, instruction| {
            let chunks = plan(e, text)?;
            let opts = options(e, seed, instruction);
            let read = render_story(e, STORY, &chunks, &opts).map_err(render_error)?;
            Ok((read.samples, read.sample_rate))
        })
    }

    /// Render `text`, handing each kept chunk's piece (samples and rate) to `on_piece` before the next chunk is
    /// generated. `on_piece` returns false to stop: the render then returns `Ok(false)`. Text that cannot be rendered
    /// is refused before the first piece.
    pub fn render_stream(
        &self,
        text: &str,
        seed: u64,
        on_piece: &mut dyn FnMut(&[f32], u32) -> bool,
    ) -> Result<bool, Fail> {
        self.with_engine(|e, instruction| {
            let chunks = plan(e, text)?;
            let opts = options(e, seed, instruction);
            let mut sink = |p: ggtts_core::pipeline::StreamPiece<'_>| {
                if on_piece(p.samples, p.sample_rate) {
                    Ok(())
                } else {
                    Err(STOPPED.to_string())
                }
            };
            match render_story_streamed(e, STORY, &chunks, &opts, &mut NullEvents, &mut sink) {
                Ok(_) => Ok(true),
                Err(err) => match render_error(err) {
                    (ERR_STOPPED, _) => Ok(false),
                    other => Err(other),
                },
            }
        })
    }
}

/// Text to the story path's chunks: one paragraph per non-empty line. Empty and punctuation-only text is refused here,
/// before the engine generates anything.
fn plan(engine: &dyn Engine, text: &str) -> Result<Vec<ggtts_core::text::PlannedChunk>, Fail> {
    let paragraphs: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    if paragraphs.is_empty() {
        return Err(arg_err("text is empty"));
    }
    plan_story(engine.profile(), &paragraphs).map_err(|e| arg_err(e.to_string()))
}

fn options(engine: &dyn Engine, seed: u64, instruction: Option<&str>) -> RenderOptions {
    RenderOptions {
        seed_salt: seed.to_string(),
        instruction: instruction.map(String::from),
        ..RenderOptions::for_profile(engine.profile())
    }
}

fn render_error(e: RenderError) -> Fail {
    match e {
        RenderError::Sink(m) if m == STOPPED => (ERR_STOPPED, "the stream callback asked to stop".into()),
        other => engine_err(other.to_string()),
    }
}

/// A model file must exist and be a GGUF before any engine is asked to load it.
fn check_gguf(what: &str, path: &Path) -> Result<(), Fail> {
    let mut magic = [0u8; 4];
    let read = std::fs::File::open(path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic));
    match read {
        Err(e) => Err(engine_err(format!("{what} {}: {e}", path.display()))),
        Ok(()) if &magic != b"GGUF" => Err(engine_err(format!("{what} {}: not a GGUF file", path.display()))),
        Ok(()) => Ok(()),
    }
}

fn non_empty(s: &Option<String>) -> Option<&str> {
    s.as_deref().filter(|s| !s.is_empty())
}

/// The engines other than Qwen3-TTS take neither a speaker nor an instruction, and refuse one before any file is read,
/// rather than ignore it: [`ERR_ARG`], as the C-ABI.
fn refuse_settings_for(engine: &str, speaker: Option<&str>, instruction: Option<&str>) -> Result<(), Fail> {
    if speaker.is_some() {
        return Err(arg_err(format!(
            "a speaker is only supported by the qwen3-tts engine, not {engine}"
        )));
    }
    let req = ChunkRequest {
        text: "",
        seed: 0,
        chunk: 0,
        draw: 0,
        instruction,
    };
    refuse_instruction(&req, engine).map_err(|e| arg_err(e.0))
}

/// The engines a build can load, by name, besides the test engines.
pub const ENGINES: &[&str] = &["qwen3-tts", "kitten-tts-2"];

/// Load an engine. `notice` receives the download notice and progress of a default model, which is never silent.
pub fn load(req: &LoadRequest, notice: &mut dyn FnMut(&str)) -> Result<(Handle, Loaded), Fail> {
    guard(|| {
        load_inner(
            req,
            notice,
            &|| ggtts_models::Cache::default_root().map(ggtts_models::Cache::new),
            &ggtts_models::Http::default(),
        )
    })
}

/// [`load`] with the cache and the fetcher passed in, for tests.
fn load_inner(
    req: &LoadRequest,
    notice: &mut dyn FnMut(&str),
    cache: &dyn Fn() -> Result<ggtts_models::Cache, String>,
    fetch: &dyn ggtts_models::Fetch,
) -> Result<(Handle, Loaded), Fail> {
    let name = req.engine.as_str();
    let mut speaker = non_empty(&req.speaker).map(String::from);
    let instruction = non_empty(&req.instruction).map(String::from);
    let threads = if req.threads <= 0 { 4 } else { req.threads };
    if name != "qwen3-tts" {
        refuse_settings_for(name, speaker.as_deref(), instruction.as_deref())?;
    }
    let mut loaded = Loaded {
        engine: name.to_string(),
        instruction: instruction.clone(),
        device: req.device,
        ..Default::default()
    };
    let test_engine = match name {
        "fake" => Some(Held::checked(Box::new(ggtts_fake::FakeEngine::default()))),
        #[cfg(feature = "plant")]
        "panic" => Some(Held::checked(Box::new(planted::Panics(
            ggtts_fake::FakeEngine::default(),
        )))),
        #[cfg(feature = "plant")]
        "slow" => Some(Held::checked(Box::new(
            planted::Slow(ggtts_fake::FakeEngine::default()),
        ))),
        #[cfg(feature = "plant")]
        "panic-load" => panic!("planted panic in load"),
        _ => None,
    };
    if let Some(held) = test_engine {
        if req.device != Device::Cpu {
            return Err(arg_err(format!(
                "the {name} engine runs on the CPU only, not {}",
                req.device.as_str()
            )));
        }
        return Ok(finish(held, loaded));
    }
    if !ENGINES.contains(&name) {
        return Err(engine_err(format!("no engine named {name:?}")));
    }
    // As the C-ABI: refused before any file is read or downloaded, with the same code and message.
    if req.device == Device::Cuda && !cfg!(feature = "cuda") {
        return Err(engine_err(
            "this build has no CUDA backend (built without the cuda feature): device cuda is not available; use cpu",
        ));
    }
    // The model files: the given paths, or the engine's pinned defaults from the cache, as ggtts-render resolves them.
    let given = vec![req.model.clone(), req.codec.clone()];
    let given_file = |role: &str, path: &PathBuf| ModelFile {
        role: role.to_string(),
        path: path.clone(),
        source: "path".into(),
    };
    let mut files: Vec<ModelFile> = match ggtts_models::defaults_for(name) {
        // A file not given is the pinned default (the talker and the codec), from the cache.
        Some(d) if given.len() == d.files.len() && given.iter().any(Option::is_none) => {
            if !cfg!(feature = "ggml") {
                return Err(engine_err(format!(
                    "this build has no {name} engine (built without the ggml feature), so no default model is fetched"
                )));
            }
            let cache = cache().map_err(engine_err)?;
            let resolved = ggtts_models::resolve(d, &given, &cache, req.offline, fetch, notice).map_err(engine_err)?;
            if req.model.is_none() && speaker.is_none() {
                speaker = d.default_speaker.map(str::to_string);
            }
            resolved
                .into_iter()
                .map(|f| ModelFile {
                    role: f.role,
                    path: f.path,
                    source: f.source.as_str().to_string(),
                })
                .collect()
        }
        _ => {
            let model = req
                .model
                .as_ref()
                .ok_or_else(|| arg_err(format!("the {name} engine needs a model")))?;
            let codec = req
                .codec
                .as_ref()
                .ok_or_else(|| arg_err(format!("the {name} engine needs a codec")))?;
            vec![given_file("model", model), given_file("codec", codec)]
        }
    };
    let (model, codec) = (files[0].path.clone(), files[1].path.clone());
    check_gguf("model", &model)?;
    check_gguf("codec", &codec)?;
    if let Some(v) = &req.voice {
        files.push(given_file("voice", v));
    }
    loaded.speaker = speaker.clone();
    let held = load_real(
        name,
        &model,
        &codec,
        req.voice.as_deref(),
        threads,
        req.device,
        speaker.as_deref(),
        instruction.as_deref(),
    )?;
    loaded.files = files;
    Ok(finish(held, loaded))
}

fn finish(mut held: Held, mut loaded: Loaded) -> (Handle, Loaded) {
    let info = held.engine.info();
    loaded.family = info.family.clone();
    loaded.backend = info.backend.clone();
    loaded.parity = info.parity.clone();
    held.instruction = loaded.instruction.clone();
    (Handle::new(held), loaded)
}

#[cfg(feature = "ggml")]
#[allow(clippy::too_many_arguments)]
fn load_real(
    engine: &str,
    model: &Path,
    codec: &Path,
    voice: Option<&Path>,
    threads: i32,
    device: Device,
    speaker: Option<&str>,
    instruction: Option<&str>,
) -> Result<Held, Fail> {
    match engine {
        "qwen3-tts" => {
            let mut v = ggtts_qwen3_tts::engine::Voice {
                device: match device {
                    Device::Cpu => ggtts_qwen3_tts::cpu::Device::Cpu,
                    Device::Cuda => ggtts_qwen3_tts::cpu::Device::Cuda,
                },
                ..Default::default()
            };
            if let Some(s) = speaker {
                v.speaker = s.to_string();
            }
            // The library checks both settings while loading, before any text is rendered: an instruction the
            // checkpoint takes none of, then a speaker the model does not have (with the model's speakers).
            ggtts_qwen3_tts::engine::Qwen3Tts::load_for(model, codec, threads, v, instruction)
                // SAFETY: as in the C-ABI. Qwen3-TTS is not `Send` to the compiler only for its raw ggml pointers,
                // every one allocated by this engine and reachable only through it. It holds no `Rc` and no
                // thread-local, and ggml's backends keep no per-thread state between graph runs, so the engine is
                // sound to move whole and use from one thread at a time.
                .map(|e| unsafe { Held::assume_send(Box::new(e)) })
                .map_err(|e| {
                    // The library's unknown-speaker message names the CLI flag, which a Python caller does not have.
                    engine_err(e.0.replace(". Choose one with --speaker", ". Choose one with speaker="))
                })
        }
        _ => {
            let voice = voice.ok_or_else(|| arg_err("the kitten-tts-2 engine needs a voice"))?;
            let device = match device {
                Device::Cpu => ggtts_kitten::backend::Device::Cpu,
                Device::Cuda => ggtts_kitten::backend::Device::Cuda,
            };
            ggtts_kitten::engine::KittenEngine::open(model, codec, voice, threads as usize, device)
                // SAFETY: as in the C-ABI. Kitten-TTS 2 is not `Send` to the compiler for its raw ggml pointers and
                // the `Rc<Backend>` its parts share. Every clone of that `Rc` lives inside this engine and nothing
                // hands out a reference to its parts, so the clones move together and their count is only touched
                // by the one thread using the engine.
                .map(|e| unsafe { Held::assume_send(Box::new(e)) })
                .map_err(engine_err)
        }
    }
}

#[cfg(not(feature = "ggml"))]
#[allow(clippy::too_many_arguments)]
fn load_real(
    engine: &str,
    _: &Path,
    _: &Path,
    _: Option<&Path>,
    _: i32,
    _: Device,
    _: Option<&str>,
    _: Option<&str>,
) -> Result<Held, Fail> {
    Err(engine_err(format!(
        "this build has no {engine} engine (built without the ggml feature)"
    )))
}

/// Write mono f32 `samples` as 16-bit FLAC at `sample_rate`, with the library's writer (the suffix is forced to
/// `.flac`). Returns the path written.
pub fn write_flac(path: &Path, samples: &[f32], sample_rate: u32) -> Result<PathBuf, Fail> {
    if sample_rate == 0 {
        return Err(arg_err("sample_rate must be above 0"));
    }
    ggtts_core::flac::write_flac(path, samples, sample_rate).map_err(|e| engine_err(format!("{}: {e}", path.display())))
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

    /// The fake engine, taking a quarter second per chunk: long enough for a test to see another thread run.
    pub struct Slow(pub ggtts_fake::FakeEngine);

    impl Engine for Slow {
        fn info(&self) -> &EngineInfo {
            self.0.info()
        }
        fn profile(&self) -> &Profile {
            self.0.profile()
        }
        fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
            std::thread::sleep(std::time::Duration::from_millis(250));
            self.0.generate(req)
        }
        fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
            self.0.decode(codes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    const TEXT: &str = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。";

    fn fake() -> (Handle, Loaded) {
        load(
            &LoadRequest {
                engine: "fake".into(),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap()
    }

    fn req(engine: &str) -> LoadRequest {
        LoadRequest {
            engine: engine.into(),
            ..Default::default()
        }
    }

    fn load_err(r: &LoadRequest) -> Fail {
        match load(r, &mut |_| {}) {
            Ok(_) => panic!("{r:?} loaded"),
            Err(e) => e,
        }
    }

    #[test]
    fn the_codes_are_the_c_abi_codes() {
        assert_eq!(OK, ggtts_ffi::GGTTS_OK);
        assert_eq!(ERR_ARG, ggtts_ffi::GGTTS_ERR_ARG);
        assert_eq!(ERR_HANDLE, ggtts_ffi::GGTTS_ERR_HANDLE);
        assert_eq!(ERR_ENGINE, ggtts_ffi::GGTTS_ERR_ENGINE);
        assert_eq!(ERR_PANIC, ggtts_ffi::GGTTS_ERR_PANIC);
        assert_eq!(ERR_STOPPED, ggtts_ffi::GGTTS_ERR_STOPPED);
        assert_eq!(ERR_BUSY, ggtts_ffi::GGTTS_ERR_BUSY);
    }

    /// The same text and seed give the C-ABI's samples: the same story id, plan and options.
    #[test]
    fn a_fake_render_is_the_c_abi_render() {
        let (h, loaded) = fake();
        assert_eq!((loaded.family.as_str(), loaded.device), ("fake", Device::Cpu));
        let (ours, rate) = h.render(TEXT, 7).unwrap();
        assert!(!ours.is_empty() && rate > 0);
        let (e, t) = (
            std::ffi::CString::new("fake").unwrap(),
            std::ffi::CString::new(TEXT).unwrap(),
        );
        let mut handle = 0u64;
        let code = unsafe {
            ggtts_ffi::ggtts_load(
                e.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut handle,
            )
        };
        assert_eq!(code, 0);
        let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
        assert_eq!(
            unsafe { ggtts_ffi::ggtts_render(handle, t.as_ptr(), 7, &mut p, &mut n, &mut r) },
            0
        );
        let theirs = unsafe { std::slice::from_raw_parts(p, n) }.to_vec();
        unsafe { ggtts_ffi::ggtts_pcm_free(p, n) };
        assert_eq!(ggtts_ffi::ggtts_free(handle), 0);
        assert_eq!((ours, rate), (theirs, r));
    }

    /// The pieces of a stream are the render, in order; a stop ends it with `Ok(false)` and the engine still works.
    #[test]
    fn a_stream_is_the_render_in_pieces_and_stops() {
        let (h, _) = fake();
        let (whole, _) = h.render(TEXT, 3).unwrap();
        let mut pieces: Vec<f32> = Vec::new();
        let mut count = 0;
        assert!(h
            .render_stream(TEXT, 3, &mut |s, _| {
                pieces.extend_from_slice(s);
                count += 1;
                true
            })
            .unwrap());
        assert!(count >= 2, "{count}");
        assert_eq!(pieces.len(), whole.len(), "the level step scales but keeps the length");
        let mut seen = 0;
        assert!(!h
            .render_stream(TEXT, 3, &mut |_, _| {
                seen += 1;
                false
            })
            .unwrap());
        assert_eq!(seen, 1);
        assert!(h.render(TEXT, 3).is_ok());
    }

    struct Counting {
        inner: ggtts_fake::FakeEngine,
        calls: Arc<AtomicU64>,
    }

    impl Engine for Counting {
        fn info(&self) -> &ggtts_core::engine::EngineInfo {
            self.inner.info()
        }
        fn profile(&self) -> &ggtts_core::profile::Profile {
            self.inner.profile()
        }
        fn generate(
            &mut self,
            req: &ChunkRequest<'_>,
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
    fn punctuation_only_and_empty_text_are_refused_before_any_generate() {
        let calls = Arc::new(AtomicU64::new(0));
        let h = Handle::from_engine(
            Box::new(Counting {
                inner: ggtts_fake::FakeEngine::default(),
                calls: calls.clone(),
            }),
            None,
        );
        for text in ["好。\n？！", "？！", "  \n ", ""] {
            assert_eq!(h.render(text, 1).unwrap_err().0, ERR_ARG, "{text:?}");
            let mut pieces = 0;
            let e = h
                .render_stream(text, 1, &mut |_, _| {
                    pieces += 1;
                    true
                })
                .unwrap_err();
            assert_eq!((e.0, pieces), (ERR_ARG, 0), "{text:?}");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0, "invalid input invoked generate");
    }

    /// A setting the engine does not take, or a device that is not one, is refused before anything loads.
    #[test]
    fn settings_another_engine_does_not_take_are_refused() {
        let speaker = LoadRequest {
            speaker: Some("hk_voice_343".into()),
            ..req("fake")
        };
        let (code, msg) = load_err(&speaker);
        assert_eq!(code, ERR_ARG);
        assert!(msg.contains("only supported by the qwen3-tts engine"), "{msg}");
        let instruction = LoadRequest {
            instruction: Some("Speak slowly.".into()),
            ..req("kitten-tts-2")
        };
        let (code, msg) = load_err(&instruction);
        assert_eq!(code, ERR_ARG);
        assert!(msg.contains("takes no style instruction"), "{msg}");
        assert_eq!(
            load_err(&LoadRequest {
                device: Device::Cuda,
                ..req("fake")
            })
            .0,
            ERR_ARG
        );
        assert_eq!(Device::parse("tpu").unwrap_err().0, ERR_ARG);
        // An empty setting is no setting.
        let empty = LoadRequest {
            speaker: Some(String::new()),
            instruction: Some(String::new()),
            ..req("fake")
        };
        assert!(load(&empty, &mut |_| {}).is_ok());
        assert_eq!(load_err(&req("nonesuch")).0, ERR_ENGINE);
    }

    #[test]
    fn a_missing_or_wrong_model_file_is_an_engine_refusal() {
        let dir = std::env::temp_dir().join(format!("ggtts-py-core-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let not_gguf = dir.join("not.gguf");
        std::fs::write(&not_gguf, b"not a model").unwrap();
        let r = LoadRequest {
            model: Some(not_gguf.clone()),
            codec: Some(not_gguf.clone()),
            ..req("qwen3-tts")
        };
        let (code, msg) = load_err(&r);
        assert_eq!(code, ERR_ENGINE);
        assert!(msg.contains("not a GGUF file"), "{msg}");
        let r = LoadRequest {
            model: Some(dir.join("absent.gguf")),
            codec: Some(not_gguf.clone()),
            ..req("kitten-tts-2")
        };
        assert_eq!(load_err(&r).0, ERR_ENGINE);
        // Kitten-TTS 2 has no default model: a model is an argument it needs.
        assert_eq!(load_err(&req("kitten-tts-2")).0, ERR_ARG);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A fetcher that fails the test if it is used.
    struct NoNetwork;
    impl ggtts_models::Fetch for NoNetwork {
        fn open(&self, url: &str) -> Result<Box<dyn std::io::Read>, String> {
            panic!("offline mode fetched {url}")
        }
    }

    /// No model given: the default model from the cache. Offline with an empty cache is a refusal naming the files,
    /// and nothing is fetched; a build without engines fetches nothing either.
    #[test]
    fn the_default_model_offline_with_an_empty_cache_is_refused_without_fetching() {
        let root = std::env::temp_dir().join(format!("ggtts-py-cache-{}", std::process::id()));
        let r = LoadRequest {
            offline: true,
            ..req("qwen3-tts")
        };
        let mut said = Vec::new();
        let got = guard(|| {
            load_inner(
                &r,
                &mut |m| said.push(m.to_string()),
                &|| Ok(ggtts_models::Cache::new(root.clone())),
                &NoNetwork,
            )
        });
        let (code, msg) = match got {
            Ok(_) => panic!("loaded from an empty cache"),
            Err(e) => e,
        };
        assert_eq!(code, ERR_ENGINE, "{msg}");
        if cfg!(feature = "ggml") {
            assert!(msg.contains("offline mode does not download"), "{msg}");
            assert!(msg.contains(ggtts_models::QWEN3_TTS.files[0].file_name), "{msg}");
        } else {
            assert!(msg.contains("built without the ggml feature"), "{msg}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `cuda` on a build without the CUDA backend is refused as the C-ABI refuses it (`ERR_ENGINE`, before any file
    /// is looked up or fetched), for the default model as for given paths.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_the_backend_is_refused_before_any_file() {
        let root = std::env::temp_dir().join(format!("ggtts-py-cuda-{}", std::process::id()));
        for model in [None, Some(PathBuf::from("/no/such/model.gguf"))] {
            let r = LoadRequest {
                device: Device::Cuda,
                model: model.clone(),
                ..req("qwen3-tts")
            };
            let got = guard(|| {
                load_inner(
                    &r,
                    &mut |_| {},
                    &|| Ok(ggtts_models::Cache::new(root.clone())),
                    &NoNetwork,
                )
            });
            let (code, msg) = match got {
                Ok(_) => panic!("loaded on cuda without the backend"),
                Err(e) => e,
            };
            assert_eq!(code, ERR_ENGINE, "{msg}");
            assert!(
                msg.starts_with("this build has no CUDA backend (built without the cuda feature)"),
                "{model:?}: {msg}"
            );
        }
        assert!(!root.exists(), "the cache was touched");
    }

    struct Panics(ggtts_fake::FakeEngine);
    impl Engine for Panics {
        fn info(&self) -> &ggtts_core::engine::EngineInfo {
            self.0.info()
        }
        fn profile(&self) -> &ggtts_core::profile::Profile {
            self.0.profile()
        }
        fn generate(
            &mut self,
            _: &ChunkRequest<'_>,
        ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
            panic!("planted panic in generate")
        }
        fn decode(
            &mut self,
            codes: &ggtts_core::engine::Codes,
        ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
            self.0.decode(codes)
        }
    }

    /// A panic is a code and a message, never an unwind; the engine that panicked is dropped.
    #[test]
    fn a_panic_is_err_panic_and_drops_the_engine() {
        let h = Handle::from_engine(Box::new(Panics(ggtts_fake::FakeEngine::default())), None);
        let (code, msg) = h.render(TEXT, 1).unwrap_err();
        assert_eq!(code, ERR_PANIC);
        assert!(msg.contains("planted panic in generate"), "{msg}");
        assert!(h.is_closed());
        assert_eq!(h.render(TEXT, 1).unwrap_err().0, ERR_HANDLE);
        let (code, msg) = guard::<()>(|| panic!("boom")).unwrap_err();
        assert_eq!((code, msg.as_str()), (ERR_PANIC, "panic inside ggtts: boom"));
    }

    /// A call on an engine from its own stream callback is busy; a closed engine is a handle error.
    #[test]
    fn busy_and_closed() {
        let (h, _) = fake();
        let mut inner = None;
        h.render_stream(TEXT, 1, &mut |_, _| {
            inner = Some(h.render(TEXT, 1).map(|_| ()));
            false
        })
        .unwrap();
        assert_eq!(inner.unwrap().unwrap_err().0, ERR_BUSY);
        assert!(h.close());
        assert!(!h.close());
        assert_eq!(h.render(TEXT, 1).unwrap_err().0, ERR_HANDLE);
    }

    /// The loaded instruction reaches every request of a render and a stream; without one, none.
    #[test]
    fn the_instruction_reaches_every_request() {
        struct Recording(ggtts_fake::FakeEngine, Arc<Mutex<Vec<Option<String>>>>);
        impl Engine for Recording {
            fn info(&self) -> &ggtts_core::engine::EngineInfo {
                self.0.info()
            }
            fn profile(&self) -> &ggtts_core::profile::Profile {
                self.0.profile()
            }
            fn generate(
                &mut self,
                req: &ChunkRequest<'_>,
            ) -> Result<ggtts_core::engine::Generated, ggtts_core::engine::EngineError> {
                self.1.lock().unwrap().push(req.instruction.map(String::from));
                self.0.generate(&ChunkRequest {
                    instruction: None,
                    ..*req
                })
            }
            fn decode(
                &mut self,
                codes: &ggtts_core::engine::Codes,
            ) -> Result<ggtts_core::engine::Pcm, ggtts_core::engine::EngineError> {
                self.0.decode(codes)
            }
        }
        for instruction in [Some("Speak slowly."), None] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let h = Handle::from_engine(
                Box::new(Recording(ggtts_fake::FakeEngine::default(), seen.clone())),
                instruction,
            );
            h.render(TEXT, 1).unwrap();
            h.render_stream(TEXT, 1, &mut |_, _| true).unwrap();
            let seen = seen.lock().unwrap().clone();
            assert!(seen.len() >= 4, "{seen:?}");
            assert!(seen.iter().all(|i| i.as_deref() == instruction), "{seen:?}");
        }
    }

    #[test]
    fn flac_is_written_by_the_library_writer() {
        let path = std::env::temp_dir().join(format!("ggtts-py-{}.wav", std::process::id()));
        let written = write_flac(&path, &[0.0, 0.5, -0.5], 24000).unwrap();
        assert_eq!(written.extension().unwrap(), "flac");
        let bytes = std::fs::read(&written).unwrap();
        assert_eq!(&bytes[..4], b"fLaC");
        std::fs::remove_file(written).unwrap();
        assert_eq!(write_flac(&path, &[0.0], 0).unwrap_err().0, ERR_ARG);
    }
}
