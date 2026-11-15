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

/// What to load: the options value's load section ([`ggtts_core::options::Load`]), one type for every front end. The
/// paths are in the order an engine takes them: model (the talker or LM), codec, voice.
pub use ggtts_core::options::Load as Request;

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
            Tuning::of(req).apply(&mut voice);
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
            ggtts_kitten::engine::KittenEngine::open(model, codec, voice, threads as usize, device).map(|mut e| {
                e.full_head = req.kitten_full_head;
                Box::new(e) as Box<dyn Engine>
            })
        }
        other => Err(format!("unknown engine {other:?}")),
    }
}

#[cfg(not(feature = "ggml"))]
fn load_model_engine(_req: &Request) -> Result<Box<dyn Engine>, String> {
    Err(NO_ENGINES.into())
}

/// The cache root: `cache_dir` when given, else the per-user cache (`GGTTS_CACHE_DIR`, else the platform's).
pub fn cache_root(cache_dir: Option<&std::path::Path>) -> Result<PathBuf, String> {
    match cache_dir {
        Some(dir) => Ok(dir.to_path_buf()),
        None => Cache::default_root(),
    }
}

/// A steering file named by URL (`render.steer` `https://...`, optionally `#sha256=<hex>`), swapped for its copy in
/// the cache: downloaded once into the options' cache (`load.cache_dir`, else the per-user cache), never when
/// `load.offline`, and checked against the pin when there is one ([`ggtts_models::Cache::fetch_url`]). A path is left
/// as it is. Every front end calls this before it reads the steering file, so the URL form is one option everywhere.
/// `say` gets the download and the file's sha256.
pub fn resolve_steer(
    o: &mut ggtts_core::options::Options,
    say: &mut dyn FnMut(&str),
) -> Result<(), ggtts_core::options::Refusal> {
    let cache_dir = o.load.cache_dir.clone();
    resolve_steer_with(
        o,
        &|| cache_root(cache_dir.as_deref()).map(Cache::new),
        &ggtts_models::Http::default(),
        say,
    )
}

/// [`resolve_steer`] with the cache and the fetcher passed in, for tests.
pub fn resolve_steer_with(
    o: &mut ggtts_core::options::Options,
    cache: &dyn Fn() -> Result<Cache, String>,
    fetch: &dyn Fetch,
    say: &mut dyn FnMut(&str),
) -> Result<(), ggtts_core::options::Refusal> {
    let refusal = |message: String| ggtts_core::options::Refusal {
        key: "render.steer".into(),
        message,
    };
    let Some(text) = o.render.steer.as_ref().map(|p| p.to_string_lossy().into_owned()) else {
        return Ok(());
    };
    let spec = match ggtts_models::UrlSpec::parse(&text) {
        None => return Ok(()),
        Some(parsed) => parsed.map_err(refusal)?,
    };
    let cache = cache().map_err(refusal)?;
    let file = cache.fetch_url(&spec, o.load.offline, fetch, say).map_err(refusal)?;
    o.render.steer = Some(file.path);
    Ok(())
}

/// [`resolve`] with the request's cache (`cache_dir`, else the per-user cache) and HTTPS (notices to stderr), then [`load`].
pub fn load_default(mut req: Request) -> Result<Loaded, String> {
    let cache_dir = req.cache_dir.clone();
    let files = resolve(
        &mut req,
        &|| cache_root(cache_dir.as_deref()).map(Cache::new),
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

/// The Qwen3-TTS sampler settings, frame cap and language of a load ([`ggtts_core::options::SAMPLER`]), each unset
/// when not given. Every front end hands the load section's values here, and [`Tuning::apply`] is the one place they
/// reach the engine's [`Voice`](ggtts_qwen3_tts::engine::Voice).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Tuning {
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub subtalker_temperature: Option<f64>,
    pub subtalker_top_k: Option<i32>,
    pub subtalker_top_p: Option<f64>,
    pub max_frames: Option<i32>,
    /// One of [`ggtts_core::options::LANGUAGES`].
    pub language: Option<&'static str>,
}

impl Tuning {
    /// The settings `load` gives. A language the table does not list is left unset here: the table refuses it when
    /// the options are read.
    pub fn of(load: &ggtts_core::options::Load) -> Self {
        Tuning {
            temperature: load.temperature,
            top_k: load.top_k,
            top_p: load.top_p,
            repetition_penalty: load.repetition_penalty,
            subtalker_temperature: load.subtalker_temperature,
            subtalker_top_k: load.subtalker_top_k,
            subtalker_top_p: load.subtalker_top_p,
            max_frames: load.max_frames,
            language: load
                .language
                .as_deref()
                .and_then(|l| ggtts_core::options::LANGUAGES.iter().copied().find(|k| *k == l)),
        }
    }

    /// Writes the settings given into `voice`; the rest keep its defaults (the reference CLI's, 1200 frames, `zh`).
    #[cfg(feature = "ggml")]
    pub fn apply(&self, voice: &mut ggtts_qwen3_tts::engine::Voice) {
        let s = &mut voice.sampling;
        if let Some(t) = self.temperature {
            s.temperature = t as f32;
        }
        if let Some(k) = self.top_k {
            s.top_k = k;
        }
        if let Some(p) = self.top_p {
            s.top_p = p as f32;
        }
        if let Some(p) = self.repetition_penalty {
            s.repetition_penalty = p as f32;
        }
        s.subtalker_temperature = self.subtalker_temperature.map(|t| t as f32);
        s.subtalker_top_k = self.subtalker_top_k;
        s.subtalker_top_p = self.subtalker_top_p.map(|p| p as f32);
        if let Some(n) = self.max_frames {
            voice.max_frames = n;
        }
        if let Some(l) = self.language {
            voice.language = l.to_string();
        }
    }
}

#[cfg(all(test, feature = "ggml"))]
mod tuning_tests {
    use super::Tuning;
    use ggtts_core::options::Options;
    use ggtts_qwen3_tts::engine::Voice;

    fn voice(args: &[&str]) -> Voice {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let o = Options::from_args(&args).unwrap();
        let mut v = Voice::default();
        Tuning::of(&o.load).apply(&mut v);
        v
    }

    /// Review of #351: the settings never touch the speaker (resolved before, from the load section and the default
    /// model, #332), the device or the trace.
    #[test]
    fn the_settings_leave_the_speaker_device_and_trace_alone() {
        let args: Vec<String> = ["--temperature", "0.7", "--language", "en", "--max-frames", "600"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let o = Options::from_args(&args).unwrap();
        let mut v = Voice {
            speaker: "a_speaker".into(),
            ..Voice::default()
        };
        let before = (v.speaker.clone(), format!("{:?}", v.device), v.trace.clone());
        Tuning::of(&o.load).apply(&mut v);
        assert_eq!((v.speaker.clone(), format!("{:?}", v.device), v.trace.clone()), before);
    }

    /// #351 AC1: with none given, the voice is the engine's default, the one every render used before.
    #[test]
    fn no_setting_leaves_the_default_voice() {
        let (a, b) = (voice(&[]), Voice::default());
        assert_eq!(
            (a.sampling, a.max_frames, a.language),
            (b.sampling, b.max_frames, b.language)
        );
    }

    #[test]
    fn each_setting_reaches_the_voice() {
        let v = voice(&[
            "--temperature",
            "0.7",
            "--top-k",
            "30",
            "--top-p",
            "0.95",
            "--repetition-penalty",
            "1.1",
            "--subtalker-temperature",
            "0.8",
            "--subtalker-top-k",
            "40",
            "--subtalker-top-p",
            "0.9",
            "--max-frames",
            "600",
            "--language",
            "en",
        ]);
        let s = v.sampling;
        assert_eq!(
            (s.temperature, s.top_k, s.top_p, s.repetition_penalty),
            (0.7, 30, 0.95, 1.1)
        );
        assert_eq!(
            (s.subtalker_temperature, s.subtalker_top_k, s.subtalker_top_p),
            (Some(0.8), Some(40), Some(0.9))
        );
        assert_eq!((v.max_frames, v.language.as_str()), (600, "en"));
        // The sub-talker samples with its own settings, and with the talker's where none is given.
        let sub = s.subtalker();
        assert_eq!(
            (sub.temperature, sub.top_k, sub.top_p, sub.repetition_penalty),
            (0.8, 40, 0.9, 1.0)
        );
        let talker_only = voice(&["--temperature", "0.7"]).sampling.subtalker();
        assert_eq!(
            (talker_only.temperature, talker_only.top_k, talker_only.top_p),
            (0.7, 50, 1.0)
        );
    }
}
