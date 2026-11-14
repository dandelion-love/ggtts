//! Load a ggtts engine by name, once, for a tool that renders many requests with it.
//!
//! [`resolve`] finds the model files: a path given is used as it is; a path not given is the engine's pinned default
//! (`ggtts_models`), taken from the per-user cache and downloaded first when missing (never when `offline`). With the
//! default talker and no speaker named, the default speaker is used. [`load`] then opens the engine from those files.
//! [`load_default`] does both with the real cache and HTTPS.
//!
//! The rules are `ggtts-render`'s (`resolve_models_with` and `load_engine` in `crates/ggtts-cli/src/bin/ggtts-render.rs`),
//! kept the same so a tool built on this crate loads exactly what `ggtts-render` would for the same flags.
//! `ggtts-render` keeps its own copy for now; moving it onto this crate is a separate change.
//!
//! Without the `ggml` feature only the fake engine loads; `qwen3-tts` and `kitten-tts-2` are refused with a named
//! reason, and nothing is downloaded for a build that cannot render it.
//!
//! [`binding`] is the engine glue the bindings share: the C-ABI (`ggtts-ffi`, and Kotlin over it) and Python
//! (`ggtts-py`) load, refuse, render and guard against panics through it, with the codes and messages they give.

pub mod binding;

use std::path::PathBuf;

use ggtts_core::engine::Engine;
use ggtts_models::{Cache, Fetch, ModelFile};

/// The model engines this crate can load, by name. `fake` loads too, and needs no files.
pub const ENGINES: &[&str] = &["qwen3-tts", "kitten-tts-2"];

/// The refusal for a model engine in a build without engines.
pub const NO_ENGINES: &str = "this build has no engines: rebuild with --features ggml";

fn unknown(engine: &str) -> String {
    format!("unknown engine {engine:?}: expected qwen3-tts, kitten-tts-2 or fake")
}

/// Whether this build carries the model engines.
pub fn engines_built() -> bool {
    cfg!(feature = "ggml")
}

/// Whether this build carries the CUDA backend.
pub fn cuda_built() -> bool {
    cfg!(feature = "cuda")
}

/// What to load. The paths are in the order an engine takes them: model (the talker or LM), codec, voice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Request {
    pub engine: String,
    pub model: Option<PathBuf>,
    pub codec: Option<PathBuf>,
    /// A reference voice, for an engine that clones one (kitten-tts-2).
    pub voice: Option<PathBuf>,
    /// A named speaker in a Qwen3-TTS talker.
    pub speaker: Option<String>,
    /// `cpu` (the default) or `cuda`.
    pub device: Option<String>,
    /// CPU threads; 4 when not given.
    pub threads: Option<i32>,
    /// Never download a default file.
    pub offline: bool,
}

/// A loaded engine and the files it was loaded from, for provenance.
pub struct Loaded {
    pub engine: Box<dyn Engine>,
    pub files: Vec<ModelFile>,
    /// The speaker it renders with, when the engine takes one.
    pub speaker: Option<String>,
}

/// The files `req` renders with. Fills `req`'s paths, and its speaker when the talker is the default one and none
/// was named. `cache` is called only when a default file is needed; `say` receives the download notices.
pub fn resolve(
    req: &mut Request,
    cache: &dyn Fn() -> Result<Cache, String>,
    fetch: &dyn Fetch,
    say: &mut dyn FnMut(&str),
) -> Result<Vec<ModelFile>, String> {
    if req.engine != "fake" && !ENGINES.contains(&req.engine.as_str()) {
        return Err(unknown(&req.engine));
    }
    let given = [req.model.clone(), req.codec.clone(), req.voice.clone()];
    let Some(defaults) = ggtts_models::defaults_for(&req.engine) else {
        if req.model.is_none() && req.engine != "fake" {
            return Err(format!(
                "--model is required: the {} engine has no default model files",
                req.engine
            ));
        }
        let roles = ["model", "codec", "voice"];
        return Ok(roles
            .iter()
            .zip(given)
            .filter_map(|(role, path)| path.map(|p| ModelFile::given(role, p)))
            .collect());
    };
    let n = defaults.files.len();
    if given[..n].iter().all(Option::is_some) {
        return Ok(defaults
            .files
            .iter()
            .zip(given)
            .filter_map(|(p, path)| path.map(|path| ModelFile::given(p.role, path)))
            .collect());
    }
    // Nothing is fetched for a build that cannot render it.
    if !engines_built() {
        return Err(NO_ENGINES.into());
    }
    let files = ggtts_models::resolve(defaults, &given[..n], &cache()?, req.offline, fetch, say)?;
    if req.model.is_none() && req.speaker.is_none() {
        req.speaker = defaults.default_speaker.map(str::to_string);
    }
    let mut paths = files.iter().map(|f| Some(f.path.clone()));
    req.model = paths.next().flatten();
    req.codec = paths.next().flatten().or(req.codec.take());
    req.voice = paths.next().flatten().or(req.voice.take());
    Ok(files)
}

/// Open the engine `req` names from its (resolved) paths.
pub fn load(req: &Request) -> Result<Box<dyn Engine>, String> {
    if req.engine == "fake" {
        return Ok(Box::new(ggtts_fake::FakeEngine::default()));
    }
    if !ENGINES.contains(&req.engine.as_str()) {
        return Err(unknown(&req.engine));
    }
    load_model_engine(req)
}

#[cfg(feature = "ggml")]
fn load_model_engine(req: &Request) -> Result<Box<dyn Engine>, String> {
    let threads = req.threads.unwrap_or(4);
    let model = req.model.as_ref().ok_or("no model file: resolve the request first")?;
    match req.engine.as_str() {
        "qwen3-tts" => {
            let codec = req.codec.as_ref().ok_or("the qwen3-tts engine needs --codec")?;
            let device = req.device.as_deref().unwrap_or("cpu");
            let mut voice = ggtts_qwen3_tts::engine::Voice {
                device: ggtts_qwen3_tts::cpu::Device::parse(device)
                    .ok_or(format!("--device must be cpu or cuda, not {device}"))?,
                ..Default::default()
            };
            if let Some(speaker) = &req.speaker {
                voice.speaker.clone_from(speaker);
            }
            ggtts_qwen3_tts::engine::Qwen3Tts::load(model, codec, threads, voice)
                .map(|e| Box::new(e) as Box<dyn Engine>)
                .map_err(|e| e.to_string())
        }
        "kitten-tts-2" => {
            let codec = req.codec.as_ref().ok_or(
                "the kitten-tts-2 engine needs --codec, the S3Gen GGUF that ggtts-kitten-convert --s3gen writes",
            )?;
            let voice = req.voice.as_ref().ok_or(
                "the kitten-tts-2 engine needs --voice, a reference voice JSON that ggtts-kitten prepare writes",
            )?;
            let device = match req.device.as_deref().unwrap_or("cpu") {
                "cuda" => ggtts_kitten::backend::Device::Cuda,
                "cpu" => ggtts_kitten::backend::Device::Cpu,
                other => return Err(format!("--device must be cpu or cuda, not {other}")),
            };
            ggtts_kitten::engine::KittenEngine::open(model, codec, voice, threads as usize, device)
                .map(|e| Box::new(e) as Box<dyn Engine>)
        }
        other => Err(format!("unknown engine {other:?}")),
    }
}

#[cfg(not(feature = "ggml"))]
fn load_model_engine(_req: &Request) -> Result<Box<dyn Engine>, String> {
    Err(NO_ENGINES.into())
}

/// [`resolve`] with the per-user cache and HTTPS (notices to stderr), then [`load`].
pub fn load_default(mut req: Request) -> Result<Loaded, String> {
    let files = resolve(
        &mut req,
        &|| Cache::default_root().map(Cache::new),
        &ggtts_models::Http::default(),
        &mut |m| eprintln!("ggtts: {m}"),
    )?;
    let engine = load(&req)?;
    Ok(Loaded {
        engine,
        files,
        speaker: req.speaker,
    })
}
