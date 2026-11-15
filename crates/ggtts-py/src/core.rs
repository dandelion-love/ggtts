//! The binding's engine layer, in plain Rust: Python's handle and load request over the engine glue it shares with the
//! C-ABI (`ggtts_load::binding`): loading with the settings of `ggtts_load_with`, rendering, streaming with a stop, and
//! the guard that turns an error or a panic into a code and a message. `lib.rs` puts the Python module over it; nothing
//! here needs Python, so `cargo test` runs it without an interpreter.
//!
//! What is here is only what Python has and the C-ABI does not: its handle object (no registry, no ids), its load
//! request and the report of what was loaded, its device argument, and its wording of the refusals that name its own
//! arguments ([`WORDING`]). The codes, the refusals, the story path and the panic guard are the shared module's.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ggtts_core::engine::Engine;
use ggtts_load::binding::{self, Held, Slot};

pub use ggtts_load::binding::{
    guard, Device, Fail, ERR_ARG, ERR_BUSY, ERR_ENGINE, ERR_HANDLE, ERR_PANIC, ERR_STOPPED, OK,
};
/// The engines a build can load, by name, besides the test engines.
pub use ggtts_load::ENGINES;

/// The seed a render uses when given none, as `ggtts-render`.
pub const DEFAULT_SEED: u64 = 20260905;

fn arg_err(msg: impl Into<String>) -> Fail {
    (ERR_ARG, msg.into())
}

fn engine_err(msg: impl Into<String>) -> Fail {
    (ERR_ENGINE, msg.into())
}

/// Python's `device` argument: `cpu` or `cuda`.
pub fn parse_device(s: &str) -> Result<Device, Fail> {
    Device::from_name(s).ok_or_else(|| arg_err(format!("device must be \"cpu\" or \"cuda\", not {s:?}")))
}

/// How Python words the refusals that name its own arguments ([`binding::Wording`]).
const WORDING: binding::Wording = binding::Wording {
    choose_speaker: ". Choose one with speaker=",
    device_setting: "device",
    cpu_only,
    no_engines_for_defaults: ", so no default model is fetched",
    offline_hint: "",
    file_checks: binding::FileChecks::NeededFirst,
};

fn cpu_only(engine: &str, device: Device) -> String {
    format!("the {engine} engine runs on the CPU only, not {}", device.as_str())
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

/// A loaded engine. Its lock is taken with `try_lock`, so a second call while it renders is [`ERR_BUSY`], never a
/// deadlock; [`Handle::close`] never waits for a render, which drops the engine when it returns. A panic inside a
/// render drops the engine, and later calls are [`ERR_HANDLE`].
pub struct Handle {
    slot: Mutex<Option<Slot>>,
}

impl Handle {
    fn new(held: Held) -> Self {
        Handle {
            slot: Mutex::new(Some(binding::slot(held))),
        }
    }

    /// A handle over any engine, for tests of the boundary.
    #[cfg(test)]
    fn from_engine(e: Box<dyn Engine + Send>, instruction: Option<&str>) -> Self {
        Handle::new(Held::checked(e, instruction))
    }

    fn slot(&self) -> std::sync::MutexGuard<'_, Option<Slot>> {
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

    /// Run `f` on the engine ([`binding::with_engine`]): a panic drops the engine and becomes [`ERR_PANIC`].
    fn with_engine<T>(&self, f: impl FnOnce(&mut dyn Engine, Option<&str>) -> Result<T, Fail>) -> Result<T, Fail> {
        let slot = self
            .slot()
            .clone()
            .ok_or_else(|| (ERR_HANDLE, "the engine is closed".to_string()))?;
        binding::with_engine(
            &slot,
            |why| match why {
                binding::Unusable::Busy => (
                    ERR_BUSY,
                    "the engine is rendering (a call from its own stream callback, or from another thread)".into(),
                ),
                binding::Unusable::Poisoned => (ERR_HANDLE, "the engine is closed".to_string()),
                binding::Unusable::Panicked => (ERR_HANDLE, "the engine is closed (it panicked)".to_string()),
            },
            || {
                // Only this slot is dropped, and only if it is still the one that panicked.
                let mut current = self.slot();
                if current.as_ref().is_some_and(|s| std::sync::Arc::ptr_eq(s, &slot)) {
                    *current = None;
                }
            },
            f,
        )
    }

    /// Render `text` (one paragraph per non-empty line) to mono f32 samples and their rate.
    pub fn render(&self, text: &str, seed: u64) -> Result<(Vec<f32>, u32), Fail> {
        self.with_engine(|e, instruction| binding::render(e, text, seed, instruction))
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
        self.with_engine(
            |e, instruction| match binding::render_stream(e, text, seed, instruction, on_piece) {
                Ok(()) => Ok(true),
                Err((ERR_STOPPED, _)) => Ok(false),
                Err(other) => Err(other),
            },
        )
    }
}

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
    let opened = binding::open(
        binding::Request {
            engine: &req.engine,
            model: req.model.clone(),
            codec: req.codec.clone(),
            voice: req.voice.clone(),
            speaker: req.speaker.as_deref(),
            instruction: req.instruction.as_deref(),
            device: req.device,
            threads: req.threads,
            models: Some(binding::Models {
                cache,
                offline: req.offline,
                fetch,
            }),
        },
        &WORDING,
        notice,
    )?;
    let info = opened.held.engine().info();
    let loaded = Loaded {
        engine: req.engine.clone(),
        family: info.family.clone(),
        backend: info.backend.clone(),
        parity: info.parity.clone(),
        speaker: opened.speaker,
        instruction: req.instruction.clone().filter(|s| !s.is_empty()),
        device: req.device,
        files: opened
            .files
            .into_iter()
            .map(|f| ModelFile {
                role: f.role,
                path: f.path,
                source: f.source.as_str().to_string(),
            })
            .collect(),
    };
    Ok((Handle::new(opened.held), loaded))
}

/// Write mono f32 `samples` as 16-bit FLAC at `sample_rate`, with the library's writer (the suffix is forced to
/// `.flac`). Returns the path written.
pub fn write_flac(path: &Path, samples: &[f32], sample_rate: u32) -> Result<PathBuf, Fail> {
    if sample_rate == 0 {
        return Err(arg_err("sample_rate must be above 0"));
    }
    ggtts_core::flac::write_flac(path, samples, sample_rate).map_err(|e| engine_err(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ggtts_core::engine::ChunkRequest;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

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
        assert_eq!(parse_device("tpu").unwrap_err().0, ERR_ARG);
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
        /// A stand-in for an engine that takes an instruction, so it says so.
        struct Recording(ggtts_fake::FakeEngine, Arc<Mutex<Vec<Option<String>>>>);
        impl Engine for Recording {
            fn capabilities(&self) -> ggtts_core::capabilities::Capabilities {
                ggtts_core::capabilities::Capabilities::of("qwen3-tts").unwrap()
            }
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
