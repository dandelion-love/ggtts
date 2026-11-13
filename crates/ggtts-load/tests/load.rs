//! What `ggtts-load` loads, and what it refuses, without any model file. The model engines' own loading is covered
//! by their crates and by `ggtts-render`'s tests.

use std::io::Read;
use std::path::PathBuf;

#[cfg(not(feature = "ggml"))]
use ggtts_load::NO_ENGINES;
use ggtts_load::{engines_built, load, resolve, Request};
#[cfg(feature = "ggml")]
use ggtts_models::Cache;
use ggtts_models::Fetch;

/// A fetcher that must never be called.
struct NoNetwork;

impl Fetch for NoNetwork {
    fn open(&self, url: &str) -> Result<Box<dyn Read>, String> {
        panic!("tried to download {url}")
    }
}

#[cfg(feature = "ggml")]
fn empty_cache() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-load-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_fake_engine_loads_with_no_files() {
    let mut req = Request {
        engine: "fake".into(),
        ..Default::default()
    };
    let files = resolve(&mut req, &|| panic!("no cache is needed"), &NoNetwork, &mut |_| {}).unwrap();
    assert!(files.is_empty());
    let engine = load(&req).unwrap();
    assert_eq!(engine.info().family, "fake");
}

#[test]
fn an_unknown_engine_is_refused_by_name() {
    let req = Request {
        engine: "parrot".into(),
        ..Default::default()
    };
    let err = load(&req).err().unwrap();
    assert!(err.contains("unknown engine \"parrot\""), "{err}");
}

#[test]
fn an_engine_with_no_defaults_needs_a_model() {
    let mut req = Request {
        engine: "kitten-tts-2".into(),
        ..Default::default()
    };
    let err = resolve(&mut req, &|| panic!("no cache is needed"), &NoNetwork, &mut |_| {}).unwrap_err();
    assert!(err.contains("--model is required"), "{err}");
}

/// Given paths are used as they are: nothing is fetched and no cache is opened.
#[test]
fn given_paths_are_used_as_given() {
    let mut req = Request {
        engine: "qwen3-tts".into(),
        model: Some("talker.gguf".into()),
        codec: Some("codec.gguf".into()),
        ..Default::default()
    };
    let files = resolve(&mut req, &|| panic!("no cache is needed"), &NoNetwork, &mut |_| {}).unwrap();
    let roles: Vec<&str> = files.iter().map(|f| f.role.as_str()).collect();
    assert_eq!(roles, ["talker", "codec"]);
    assert_eq!(files[0].path, PathBuf::from("talker.gguf"));
    assert!(req.speaker.is_none(), "a given talker keeps its own default speaker");
}

/// A build without engines downloads nothing and refuses a model engine with the named reason.
#[test]
#[cfg(not(feature = "ggml"))]
fn a_build_without_engines_refuses_and_downloads_nothing() {
    assert!(!engines_built());
    let mut req = Request {
        engine: "qwen3-tts".into(),
        ..Default::default()
    };
    let err = resolve(&mut req, &|| panic!("no cache is opened"), &NoNetwork, &mut |_| {}).unwrap_err();
    assert_eq!(err, NO_ENGINES);
    let req = Request {
        engine: "qwen3-tts".into(),
        model: Some("m".into()),
        codec: Some("c".into()),
        ..Default::default()
    };
    assert_eq!(load(&req).err().unwrap(), NO_ENGINES);
}

/// Offline with an empty cache: the default files are refused by name and URL, and nothing is fetched.
#[test]
#[cfg(feature = "ggml")]
fn offline_names_the_missing_default_files() {
    assert!(engines_built());
    let dir = empty_cache();
    let mut req = Request {
        engine: "qwen3-tts".into(),
        offline: true,
        ..Default::default()
    };
    let err = resolve(&mut req, &|| Ok(Cache::new(dir.clone())), &NoNetwork, &mut |_| {}).unwrap_err();
    assert!(err.contains("offline mode does not download"), "{err}");
    assert!(err.contains("huggingface.co"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
