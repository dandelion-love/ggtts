//! The engine glue every binding shares: the C-ABI (`ggtts-ffi`, and Kotlin over it) and Python (`ggtts-py`) both
//! load, refuse, render and guard through this module, so the two cannot drift.
//!
//! * **Loading** ([`open`]): an engine by name, with a Qwen3-TTS speaker and style instruction, a device, and the
//!   pinned default model files resolved from the cache as `ggtts-render` resolves them. Every refusal comes before any
//!   text is rendered, and a setting an engine does not take is refused before any file is read.
//! * **Rendering** ([`render`], [`render_stream`]): one paragraph per non-empty line, through the story path, with the
//!   options built from a seed and the loaded instruction ([`options`]).
//! * **The guard** ([`guard`], [`with_engine`]): a panic is a code and a message, never an unwind; an engine that is
//!   rendering is [`ERR_BUSY`] to a second call, never a deadlock; an engine that panicked is never used again.
//!
//! What stays in each binding is what only it has: the C-ABI's strings, struct versions and handle registry; Python's
//! handle object, its arguments and its module. Where the two bindings word a refusal differently (each names its own
//! argument), the difference is a field of [`Wording`], so it is stated once rather than found by comparing copies.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, TryLockError};

use ggtts_core::capabilities::Capabilities;
use ggtts_core::engine::Engine;
use ggtts_core::hooks::Hooks;
use ggtts_core::options::{self, Options};
use ggtts_core::pipeline::{
    plan_story, render_plan, render_story, render_story_streamed, DurationJudge, Observers, Output, RenderError,
    RenderOptions, StorySeeds, StreamMode, StreamPiece,
};
use ggtts_core::trace::{emit_header, Events, Header, JsonlEvents, NullEvents};
use ggtts_models::{Cache, Fetch, ModelFile};

/// The codes, the C-ABI's (`ggtts.h`): success.
pub const OK: i32 = 0;
/// A NULL, non-UTF-8 or out-of-range argument; a setting the engine does not take; empty or punctuation-only text.
pub const ERR_ARG: i32 = -1;
/// No such engine: never loaded, closed, or dropped after a panic.
pub const ERR_HANDLE: i32 = -2;
/// The engine refused: a missing file, not a GGUF, an unknown engine or speaker, a failed download, a failed render.
pub const ERR_ENGINE: i32 = -3;
/// A panic inside ggtts, caught at the boundary.
pub const ERR_PANIC: i32 = -4;
/// The stream callback asked to stop.
pub const ERR_STOPPED: i32 = -5;
/// The engine is rendering: a call on it from its own stream callback, or from another thread meanwhile.
pub const ERR_BUSY: i32 = -6;

/// A code and a message.
pub type Fail = (i32, String);

/// The story id every binding render uses: the seed of each chunk is derived from it, the text and the seed given, so
/// a render is the same in every binding at the same seed.
pub const STORY: &str = "ggtts-binding";

/// The CPU threads a load uses when given 0 or less.
pub const DEFAULT_THREADS: i32 = 4;

const STOPPED: &str = "stopped by the callback";

fn arg_err(msg: impl Into<String>) -> Fail {
    (ERR_ARG, msg.into())
}

fn engine_err(msg: impl Into<String>) -> Fail {
    (ERR_ENGINE, msg.into())
}

/// The message of a caught panic.
pub fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
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
    /// `cpu` or `cuda`; anything else is `None`, which each binding refuses in its own words.
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(Device::Cpu),
            "cuda" => Some(Device::Cuda),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        }
    }
}

/// An engine and the style instruction its renders carry. `Engine` has no `Send` bound and the real engines are not
/// `Send` to the compiler (raw ggml pointers; Kitten-TTS 2 shares its backend through `Rc`), so a `Held` is built in
/// one of two ways only: [`Held::checked`], where the compiler proves the engine `Send`, or `Held::assume_send`, whose
/// every caller (in [`open`]) argues it for that one engine.
pub struct Held {
    engine: Box<dyn Engine>,
    instruction: Option<String>,
    /// The engine panicked: it is never used again.
    dead: bool,
}

impl Held {
    /// Any engine the compiler proves `Send`, with the instruction its renders carry (`None` for none).
    pub fn checked(e: Box<dyn Engine + Send>, instruction: Option<&str>) -> Self {
        Held {
            engine: e,
            instruction: instruction.map(String::from),
            dead: false,
        }
    }

    /// # Safety
    /// The engine may be moved to another thread whole and used there by one thread at a time: nothing in it is tied
    /// to the thread that made it, and nothing it shares (an `Rc`, a pointer) is reachable from outside it.
    #[cfg_attr(not(feature = "ggml"), allow(dead_code))]
    unsafe fn assume_send(e: Box<dyn Engine>, instruction: Option<&str>) -> Self {
        Held {
            engine: e,
            instruction: instruction.map(String::from),
            dead: false,
        }
    }

    /// The engine, to read its info.
    pub fn engine(&self) -> &dyn Engine {
        self.engine.as_ref()
    }
}

// SAFETY: every `Held` is `Send` by the compiler's proof or by its constructor's per-engine argument (above), and its
// engine is used under its own mutex, one call at a time.
unsafe impl Send for Held {}

/// An engine behind its own lock, as a binding keeps it.
pub type Slot = Arc<Mutex<Held>>;

/// The slot for a loaded engine.
pub fn slot(held: Held) -> Slot {
    Arc::new(Mutex::new(held))
}

/// Why [`with_engine`] did not run: each binding words these with its own handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unusable {
    /// Another call holds the engine: it is rendering.
    Busy,
    /// The engine's lock is poisoned.
    Poisoned,
    /// The engine panicked in an earlier call.
    Panicked,
}

/// Run `f` on the engine in `slot`, with the instruction it was loaded with. The engine's lock is taken with
/// `try_lock`, so a second call while it renders is `unusable(Busy)`, never a deadlock. A panic inside `f` is caught:
/// the engine is marked dead (every later call is `unusable(Panicked)`), `on_panic` runs so the binding can drop its
/// reference, and the result is [`ERR_PANIC`] with the panic's message. `on_panic` runs while the engine's lock is
/// held, so it must not wait on that lock.
pub fn with_engine<T>(
    slot: &Slot,
    unusable: impl FnOnce(Unusable) -> Fail,
    on_panic: impl FnOnce(),
    f: impl FnOnce(&mut dyn Engine, Option<&str>) -> Result<T, Fail>,
) -> Result<T, Fail> {
    let mut held = match slot.try_lock() {
        Ok(h) => h,
        Err(TryLockError::WouldBlock) => return Err(unusable(Unusable::Busy)),
        // Nothing panics while the lock is held (a panic is caught inside it), but a poisoned lock is never used.
        Err(TryLockError::Poisoned(_)) => return Err(unusable(Unusable::Poisoned)),
    };
    if held.dead {
        return Err(unusable(Unusable::Panicked));
    }
    let h = &mut *held;
    match catch_unwind(AssertUnwindSafe(|| f(h.engine.as_mut(), h.instruction.as_deref()))) {
        Ok(r) => r,
        Err(p) => {
            let msg = panic_message(p.as_ref());
            // A call that took the slot before the binding dropped it sees the engine dead, and never uses it.
            held.dead = true;
            on_panic();
            Err((ERR_PANIC, msg))
        }
    }
}

/// Text to the story path's chunks: one paragraph per non-empty line. Empty and punctuation-only text is refused here,
/// before the engine generates anything.
pub fn plan(engine: &dyn Engine, text: &str) -> Result<Vec<ggtts_core::text::PlannedChunk>, Fail> {
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

/// The story options: the seed as the salt, and the instruction the engine was loaded with (`None` renders exactly as
/// before instructions existed).
pub fn options(engine: &dyn Engine, seed: u64, instruction: Option<&str>) -> RenderOptions {
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

/// Render `text` to mono f32 samples and their rate.
pub fn render(
    engine: &mut dyn Engine,
    text: &str,
    seed: u64,
    instruction: Option<&str>,
) -> Result<(Vec<f32>, u32), Fail> {
    let chunks = plan(engine, text)?;
    let opts = options(engine, seed, instruction);
    let read = render_story(engine, STORY, &chunks, &opts).map_err(render_error)?;
    Ok((read.samples, read.sample_rate))
}

/// Render `text`, handing each kept chunk's piece (samples and rate) to `on_piece` before the level step and before
/// the next chunk is generated. `on_piece` returns false to stop: the render then ends with [`ERR_STOPPED`]. Text that
/// cannot be rendered is refused before the first piece.
pub fn render_stream(
    engine: &mut dyn Engine,
    text: &str,
    seed: u64,
    instruction: Option<&str>,
    on_piece: &mut dyn FnMut(&[f32], u32) -> bool,
) -> Result<(), Fail> {
    let chunks = plan(engine, text)?;
    let opts = options(engine, seed, instruction);
    let mut sink = |p: ggtts_core::pipeline::StreamPiece<'_>| {
        if on_piece(p.samples, p.sample_rate) {
            Ok(())
        } else {
            Err(STOPPED.to_string())
        }
    };
    render_story_streamed(engine, STORY, &chunks, &opts, &mut NullEvents, &mut sink)
        .map(|_| ())
        .map_err(render_error)
}

// ── options documents ──

/// Why a binding refuses the option `key` in a document, or `None` when it takes it: in a load document (`at_load`)
/// the load section and the render defaults; in a render document the render, the trace and the stream.
pub fn document_refusal(key: &str, at_load: bool) -> Option<&'static str> {
    let (section, name) = key.split_once('.').unwrap_or((key, ""));
    let why = match (section, name) {
        ("load", _) if at_load => return None,
        ("load", _) => "a load option: it is fixed when the engine loads, so give it in the load document",
        ("render", "text" | "story") => "the text is the render's own argument",
        ("render", "seed") => "a binding's seed is the story seed's salt: give render.seed_salt",
        ("render", "measure_timings") | ("observe", "timing_report" | "duration_report") => {
            "a binding returns the samples, not timings or reports"
        }
        ("observe", "log_level" | "log_format") => "a binding writes no log: its errors are its codes and messages",
        ("observe", "trace" | "trace_level") if at_load => "a trace is per render: give it in the render document",
        ("observe", "trace" | "trace_level") => return None,
        ("observe", _) => "instruments are out-of-process programs the CLI runs on a finished file",
        ("output", "out" | "stream_format") => "a binding hands the PCM to its caller and writes no file",
        ("output", "stream_raw" | "stream_level_seconds") => {
            "a levelled stream is not taken yet: output.stream \"raw\" streams the pieces before the level step"
        }
        ("output", "stream") if at_load => "a stream is per render: give it in the render document",
        ("output", "stream") => return None,
        ("book", _) => "ggtts-book's own options",
        ("render", _) => return None,
        _ => "not an option of a binding",
    };
    Some(why)
}

/// Refuses an option a binding does not take in this document ([`document_refusal`]), and a stream other than
/// `"raw"`, as [`ERR_ARG`] naming the option.
pub fn check_document(o: &Options, at_load: bool) -> Result<(), Fail> {
    for key in o.marks.sources.keys() {
        if let Some(why) = document_refusal(key, at_load) {
            return Err(arg_err(format!("{key}: {why}")));
        }
    }
    match o.output.stream.as_deref() {
        None | Some("raw") => Ok(()),
        Some(other) => Err(arg_err(format!(
            "output.stream: a binding streams to its callback: \"raw\" hands it the pieces before the level step \
             (a levelled stream is not taken yet), not {other:?}"
        ))),
    }
}

/// Render `text` with an options value: the whole read, or, with `on_piece`, each kept chunk handed to it before the
/// level step (returning false stops the render with [`ERR_STOPPED`]); the whole read's samples and rate either way.
/// An option the engine does not support is refused ([`options::check`]) as [`ERR_ARG`] before anything is
/// rendered, and so is steering it cannot read. `observe.trace` writes the event stream, as `ggtts-render` does.
/// With only `render.seed_salt` and `render.instruction` set, this is [`render`] (and with `on_piece`,
/// [`render_stream`]) sample for sample.
pub fn render_document(
    engine: &mut dyn Engine,
    text: &str,
    options: &Options,
    on_piece: Option<&mut dyn FnMut(&[f32], u32) -> bool>,
) -> Result<(Vec<f32>, u32), Fail> {
    let chunks = plan(engine, text)?;
    let mut o = options::check(options, &engine.capabilities())
        .map_err(|r| arg_err(r.to_string()))?
        .options;
    o.read_steering().map_err(|r| arg_err(r.to_string()))?;
    let opts = o.render_options(engine.profile()).map_err(|r| arg_err(r.to_string()))?;
    let level = o.observe.trace_level.unwrap_or(1);
    let mut trace = match &o.observe.trace {
        Some(path) => Some(JsonlEvents::new(
            std::fs::File::create(path).map_err(|e| arg_err(format!("observe.trace {}: {e}", path.display())))?,
            level,
        )),
        None => None,
    };
    let mut null = NullEvents;
    let events: &mut dyn Events = match trace.as_mut() {
        Some(w) => w,
        None => &mut null,
    };
    if events.level() > 0 {
        let mut header = Header::for_engine(engine, level);
        header.set_reported(&options::record(&o, engine.info()));
        emit_header(events, &header);
    }
    let mut hooks = Hooks::new();
    let mut obs = Observers::new(events, &mut hooks);
    let seeds = StorySeeds {
        story_id: STORY,
        salt: &opts.seed_salt,
    };
    let read = match on_piece {
        None => render_plan(
            engine,
            &chunks,
            &seeds,
            &mut DurationJudge,
            &opts,
            &mut obs,
            Output::Whole,
        ),
        Some(f) => {
            let mut sink = |p: StreamPiece<'_>| {
                if f(p.samples, p.sample_rate) {
                    Ok(())
                } else {
                    Err(STOPPED.to_string())
                }
            };
            let output = Output::Stream {
                sink: &mut sink,
                level: StreamMode::Raw,
            };
            render_plan(engine, &chunks, &seeds, &mut DurationJudge, &opts, &mut obs, output)
        }
    }
    .map_err(render_error)?;
    if let Some(w) = trace {
        w.finish_with_stats()
            .map_err(|e| engine_err(format!("observe.trace: {e}")))?;
    }
    Ok((read.samples, read.sample_rate))
}

/// A speaker or an instruction the engine does not take is refused before any file is read, rather than ignored, as
/// [`ERR_ARG`], with [`options::check`]'s refusal: the words `ggtts-render` shows for the same option and engine. An
/// engine name the library does not know is left to the load, which names it.
pub fn refuse_settings(engine: &str, speaker: Option<&str>, instruction: Option<&str>) -> Result<(), Fail> {
    let Some(caps) = Capabilities::of(engine) else {
        return Ok(());
    };
    let mut o = Options::default();
    o.load.speaker = speaker.map(str::to_string);
    o.render.instruction = instruction.map(str::to_string);
    options::check(&o, &caps)
        .map(|_| ())
        .map_err(|r| arg_err(r.to_string()))
}

/// The order a load checks the model files given as paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChecks {
    /// The model is needed, then read; then the codec is needed, then read (the C-ABI).
    InTurn,
    /// Both are needed, then both are read (Python).
    NeededFirst,
}

/// The words a binding uses where it names its own arguments in a refusal. Every other message is the same in every
/// binding. A difference here is a difference between the bindings' messages today, kept byte for byte.
#[derive(Debug, Clone, Copy)]
pub struct Wording {
    /// What replaces the library's ". Choose one with --speaker" in an unknown-speaker refusal, which names a CLI flag
    /// a binding caller does not have.
    pub choose_speaker: &'static str,
    /// The binding's name for the device setting, in the refusal of `cuda` by a build without the CUDA backend.
    pub device_setting: &'static str,
    /// The refusal of a device other than the CPU by a test engine (`fake`, and the planted ones).
    pub cpu_only: fn(&str, Device) -> String,
    /// Added to the refusal of a default model by a build without engines.
    pub no_engines_for_defaults: &'static str,
    /// Added to the offline refusal of a default model file missing from the cache.
    pub offline_hint: &'static str,
    /// The order the model files given as paths are checked.
    pub file_checks: FileChecks,
}

/// How a load finds the default model files.
pub struct Models<'a> {
    /// The cache to use (a binding's setting, or the per-user default); its error is the refusal's message.
    pub cache: &'a dyn Fn() -> Result<Cache, String>,
    /// Never download: a default file missing from the cache is refused, naming it.
    pub offline: bool,
    /// How a missing file is downloaded.
    pub fetch: &'a dyn Fetch,
}

/// What to load. A speaker or instruction that is empty is no setting.
pub struct Request<'a> {
    pub engine: &'a str,
    pub model: Option<PathBuf>,
    pub codec: Option<PathBuf>,
    /// Kitten-TTS 2's reference voice.
    pub voice: Option<PathBuf>,
    pub speaker: Option<&'a str>,
    pub instruction: Option<&'a str>,
    pub device: Device,
    /// 0 or less is [`DEFAULT_THREADS`].
    pub threads: i32,
    /// `None`: only the paths given are loaded (the C-ABI's version-2 struct and NULL settings). `Some`: a model or
    /// codec not given is the engine's pinned default, from the cache, as `ggtts-render` resolves it.
    pub models: Option<Models<'a>>,
}

/// A loaded engine, the speaker in effect (the one given, or the default model's default speaker; `None` for the
/// engine's own) and the files it was loaded from.
pub struct Opened {
    pub held: Held,
    pub speaker: Option<String>,
    pub files: Vec<ModelFile>,
}

/// The engines a build can load by name, besides the test engines.
pub use crate::ENGINES;

/// Load an engine. `notice` receives the download notice and progress of a default model file. Every refusal comes
/// before any text is rendered; a setting the engine does not take, and `cuda` on a build without it, before any file
/// is read or downloaded.
pub fn open(req: Request<'_>, wording: &Wording, notice: &mut dyn FnMut(&str)) -> Result<Opened, Fail> {
    let name = req.engine;
    let speaker = req.speaker.filter(|s| !s.is_empty());
    let instruction = req.instruction.filter(|s| !s.is_empty());
    let threads = if req.threads <= 0 { DEFAULT_THREADS } else { req.threads };
    if name != "qwen3-tts" {
        refuse_settings(name, speaker, instruction)?;
    }
    if let Some(held) = test_engine(name) {
        if req.device != Device::Cpu {
            return Err(arg_err((wording.cpu_only)(name, req.device)));
        }
        return Ok(Opened {
            held,
            speaker: None,
            files: Vec::new(),
        });
    }
    if !ENGINES.contains(&name) {
        return Err(engine_err(format!("no engine named {name:?}")));
    }
    // Refused before any file is read or downloaded.
    if req.device == Device::Cuda && !cfg!(feature = "cuda") {
        return Err(engine_err(format!(
            "this build has no CUDA backend (built without the cuda feature): {} cuda is not available; use cpu",
            wording.device_setting
        )));
    }
    let mut speaker = speaker.map(String::from);
    let need = |p: &Option<PathBuf>, what: &str| {
        p.clone()
            .ok_or_else(|| arg_err(format!("the {name} engine needs a {what}")))
    };
    let (model, codec, mut files) = match (&req.models, ggtts_models::defaults_for(name)) {
        (Some(models), Some(defaults)) if req.model.is_none() || req.codec.is_none() => {
            let files = resolve_defaults(
                defaults,
                req.model.clone(),
                req.codec.clone(),
                models,
                cfg!(feature = "ggml"),
                wording,
                notice,
            )?;
            if speaker.is_none() {
                speaker = files.speaker.map(String::from);
            }
            check_gguf("model", &files.model)?;
            check_gguf("codec", &files.codec)?;
            (files.model, files.codec, files.files)
        }
        _ => {
            let (model, codec) = match wording.file_checks {
                FileChecks::InTurn => {
                    let model = need(&req.model, "model")?;
                    check_gguf("model", &model)?;
                    let codec = need(&req.codec, "codec")?;
                    check_gguf("codec", &codec)?;
                    (model, codec)
                }
                FileChecks::NeededFirst => {
                    let (model, codec) = (need(&req.model, "model")?, need(&req.codec, "codec")?);
                    check_gguf("model", &model)?;
                    check_gguf("codec", &codec)?;
                    (model, codec)
                }
            };
            let files = vec![
                ModelFile::given("model", model.clone()),
                ModelFile::given("codec", codec.clone()),
            ];
            (model, codec, files)
        }
    };
    if let Some(v) = &req.voice {
        files.push(ModelFile::given("voice", v.clone()));
    }
    let held = load_real(
        name,
        &model,
        &codec,
        req.voice.as_deref(),
        threads,
        req.device,
        speaker.as_deref(),
        instruction,
        wording,
    )?;
    Ok(Opened { held, speaker, files })
}

/// The test engines, which need no files: `fake`, and with the `plant` feature `panic` (generating panics), `slow`
/// (each chunk takes a quarter second) and `panic-load` (loading panics).
fn test_engine(name: &str) -> Option<Held> {
    match name {
        "fake" => Some(Held::checked(Box::new(ggtts_fake::FakeEngine::default()), None)),
        #[cfg(feature = "plant")]
        "panic" => Some(Held::checked(
            Box::new(planted::Panics(ggtts_fake::FakeEngine::default())),
            None,
        )),
        #[cfg(feature = "plant")]
        "slow" => Some(Held::checked(
            Box::new(planted::Slow(ggtts_fake::FakeEngine::default())),
            None,
        )),
        #[cfg(feature = "plant")]
        "panic-load" => panic!("planted panic in load"),
        _ => None,
    }
}

/// What [`resolve_defaults`] found: the model and codec paths, the speaker to use when none is given (the default
/// model's default speaker; a given model keeps its own), and every file with where it came from.
#[derive(Debug)]
pub struct DefaultFiles {
    pub model: PathBuf,
    pub codec: PathBuf,
    pub speaker: Option<&'static str>,
    pub files: Vec<ModelFile>,
}

/// The files a load uses when the model or the codec is not given, resolved as `ggtts-render` resolves them with no
/// `--model` or `--codec` (`ggtts_models`): a path given is used as it is; a path not given is the engine's pinned file
/// in the cache, verified against its sha256, and downloaded first unless `models.offline`. Nothing is looked up for a
/// build that cannot render it (`engines_built` false).
pub fn resolve_defaults(
    defaults: &ggtts_models::Defaults,
    model: Option<PathBuf>,
    codec: Option<PathBuf>,
    models: &Models<'_>,
    engines_built: bool,
    wording: &Wording,
    notice: &mut dyn FnMut(&str),
) -> Result<DefaultFiles, Fail> {
    if !engines_built {
        return Err(engine_err(format!(
            "this build has no {} engine (built without the ggml feature){}",
            defaults.engine, wording.no_engines_for_defaults
        )));
    }
    let cache = (models.cache)().map_err(engine_err)?;
    let model_given = model.is_some();
    let mut given = vec![model, codec];
    given.resize(defaults.files.len(), None);
    let files = ggtts_models::resolve(defaults, &given, &cache, models.offline, models.fetch, notice).map_err(|e| {
        if e.contains("offline mode does not download") {
            engine_err(format!("{e}{}", wording.offline_hint))
        } else {
            engine_err(e)
        }
    })?;
    let mut paths = files.iter().map(|f| f.path.clone());
    match (paths.next(), paths.next()) {
        (Some(model), Some(codec)) => Ok(DefaultFiles {
            model,
            codec,
            speaker: if model_given { None } else { defaults.default_speaker },
            files,
        }),
        _ => Err(engine_err(format!(
            "{}: the default files are not a model and a codec",
            defaults.engine
        ))),
    }
}

/// A model file must exist and be a GGUF before any engine is asked to load it.
fn check_gguf(what: &str, path: &Path) -> Result<(), Fail> {
    let mut magic = [0u8; 4];
    let read = std::fs::File::open(path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic));
    let path = path.display();
    match read {
        Err(e) => Err(engine_err(format!("{what} {path}: {e}"))),
        Ok(()) if &magic != b"GGUF" => Err(engine_err(format!("{what} {path}: not a GGUF file"))),
        Ok(()) => Ok(()),
    }
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
    wording: &Wording,
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
                // SAFETY: Qwen3-TTS is not `Send` to the compiler only for its raw ggml pointers (contexts, tensors,
                // graphs, backend buffers), every one allocated by this engine and reachable only through it. It holds
                // no `Rc` and no thread-local, and neither of ggml's backends it may run on keeps per-thread state
                // between graph runs: the CPU backend keeps none, and the CUDA backend has no `thread_local` and
                // selects its device on every call (`ggml_cuda_set_device`), so its streams and cuBLAS handles, used by
                // one thread at a time, may be used from any thread. The engine is sound to move whole and use from
                // one thread at a time.
                .map(|e| unsafe { Held::assume_send(Box::new(e), instruction) })
                // The library's unknown-speaker message names the CLI flag, which a binding caller does not have.
                .map_err(|e| engine_err(e.0.replace(". Choose one with --speaker", wording.choose_speaker)))
        }
        _ => {
            let voice = voice.ok_or_else(|| arg_err("the kitten-tts-2 engine needs a voice"))?;
            let device = match device {
                Device::Cpu => ggtts_kitten::backend::Device::Cpu,
                Device::Cuda => ggtts_kitten::backend::Device::Cuda,
            };
            ggtts_kitten::engine::KittenEngine::open(model, codec, voice, threads as usize, device)
                // SAFETY: Kitten-TTS 2 is not `Send` to the compiler for its raw ggml pointers and for the
                // `Rc<Backend>` its parts share. Every clone of that `Rc` lives inside this engine, and nothing hands
                // out a reference to its parts, so the clones always move together and their count is only ever
                // touched by the one thread using the engine. Neither backend keeps per-thread state (as for
                // Qwen3-TTS above). It takes no instruction (refused before loading).
                .map(|e| unsafe { Held::assume_send(Box::new(e), None) })
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
    _: &Wording,
) -> Result<Held, Fail> {
    Err(engine_err(format!(
        "this build has no {engine} engine (built without the ggml feature)"
    )))
}

#[cfg(feature = "plant")]
mod planted {
    use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
    use ggtts_core::profile::Profile;

    /// The fake engine, except that generating panics: the planted defect for the boundary tests.
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

    const TEXT: &str = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。";

    fn wording() -> Wording {
        Wording {
            choose_speaker: ". Choose one with the test's setting",
            device_setting: "test.device",
            cpu_only: |name, _| format!("the {name} engine runs on the CPU only"),
            no_engines_for_defaults: "",
            offline_hint: "",
            file_checks: FileChecks::InTurn,
        }
    }

    fn fake_slot() -> Slot {
        let opened = open(
            Request {
                engine: "fake",
                model: None,
                codec: None,
                voice: None,
                speaker: None,
                instruction: None,
                device: Device::Cpu,
                threads: 0,
                models: None,
            },
            &wording(),
            &mut |_| {},
        )
        .unwrap_or_else(|e| panic!("{e:?}"));
        slot(opened.held)
    }

    /// A second call while the engine renders is busy; a panic marks it dead and runs `on_panic` once.
    #[test]
    fn busy_then_panicked() {
        let s = fake_slot();
        let inner = with_engine(
            &s,
            |_| (ERR_HANDLE, "unusable".into()),
            || {},
            |_, _| Ok(with_engine(&s, |u| (ERR_BUSY, format!("{u:?}")), || {}, |_, _| Ok(()))),
        )
        .unwrap();
        assert_eq!(inner.unwrap_err(), (ERR_BUSY, "Busy".to_string()));
        let mut evicted = 0;
        let (code, msg) = with_engine::<()>(
            &s,
            |_| (ERR_HANDLE, "unusable".into()),
            || evicted += 1,
            |_, _| panic!("boom"),
        )
        .unwrap_err();
        assert_eq!(
            (code, msg.as_str(), evicted),
            (ERR_PANIC, "panic inside ggtts: boom", 1)
        );
        let after = with_engine(&s, |u| (ERR_HANDLE, format!("{u:?}")), || {}, |_, _| Ok(()));
        assert_eq!(after.unwrap_err(), (ERR_HANDLE, "Panicked".to_string()));
    }

    /// A render and a stream of the same text and seed give the same samples, in pieces; a stop is ERR_STOPPED.
    #[test]
    fn a_stream_is_the_render_in_pieces() {
        let mut e = ggtts_fake::FakeEngine::default();
        let (whole, _) = render(&mut e, TEXT, 3, None).unwrap();
        let mut len = 0;
        render_stream(&mut e, TEXT, 3, None, &mut |s, _| {
            len += s.len();
            true
        })
        .unwrap();
        assert_eq!(len, whole.len());
        let stopped = render_stream(&mut e, TEXT, 3, None, &mut |_, _| false).unwrap_err();
        assert_eq!(stopped, (ERR_STOPPED, "the stream callback asked to stop".to_string()));
    }
}
