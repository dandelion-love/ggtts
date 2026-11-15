//! A steering URL through the C ABI resolves with the settings the engine was loaded with: an engine loaded offline
//! with a cache directory refuses an uncached URL, naming it and that cache. The load here is the version-3 struct
//! load, the Kotlin binding's `loadWithModels` (the Android app's `allowDownload = false`).
//!
//! The planted `steerable` engine is the fake engine stating Qwen3-TTS's capabilities, so steering passes the
//! capability check and reaches the resolution. The URL's host is `.invalid`, so a resolution that ignored the
//! load's offline setting fails on the network instead, and the check below sees it.
#![cfg(feature = "plant")]

use std::ffi::{CStr, CString};
use std::ptr;

use ggtts_ffi::*;

const URL: &str = "https://example.invalid/presets/angry.safetensors";

fn last_error() -> String {
    unsafe { CStr::from_ptr(ggtts_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn last_key() -> Option<String> {
    let k = ggtts_last_error_key();
    (!k.is_null()).then(|| unsafe { CStr::from_ptr(k) }.to_string_lossy().into_owned())
}

/// A whole read with `render.steer` the uncached URL: the code, the message and the refused key.
fn render_steered(h: u64) -> (i32, String, Option<String>) {
    let text = CString::new("你好。").unwrap();
    let doc = CString::new(format!(r#"{{"render": {{"steer": "{URL}"}}}}"#)).unwrap();
    let (mut p, mut n, mut r) = (ptr::null_mut(), 0usize, 0u32);
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            doc.as_ptr(),
            None,
            ptr::null_mut(),
            &mut p,
            &mut n,
            &mut r,
        )
    };
    if !p.is_null() {
        unsafe { ggtts_pcm_free(p, n) };
    }
    (code, last_error(), last_key())
}

/// The offline refusal of the URL, naming the load's cache, or what was seen instead.
fn refused_offline_in(seen: (i32, String, Option<String>), cache: &std::path::Path) -> Result<(), String> {
    let (code, msg, key) = seen;
    let want = msg.contains(URL)
        && msg.contains("offline mode does not download it")
        && msg.contains(&cache.display().to_string());
    if code == GGTTS_ERR_ARG && key.as_deref() == Some("render.steer") && want {
        Ok(())
    } else {
        Err(format!("code {code}, key {key:?}: {msg}"))
    }
}

#[test]
fn an_engine_loaded_offline_resolves_a_steering_url_offline_in_its_own_cache() {
    let dir = std::env::temp_dir().join(format!("ggtts-ffi-steer-url-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // A load document cannot name a test engine (its load.engine is a choice of real ones), and both loads record
    // the settings in the same place (load_options), so the struct load stands for both here; Python covers a
    // document beside a test engine.
    // The version-3 struct load with downloading off: Kotlin's loadWithModels, and Android's allowDownload = false.
    let cache = dir.join("struct-cache");
    let (engine, cache_c) = (
        CString::new("steerable").unwrap(),
        CString::new(cache.to_str().unwrap()).unwrap(),
    );
    let settings = GgttsSettings {
        size: std::mem::size_of::<GgttsSettings>(),
        speaker: ptr::null(),
        instruction: ptr::null(),
        device: ptr::null(),
        cache_dir: cache_c.as_ptr(),
        allow_download: 0,
    };
    let mut h = 0u64;
    let null = ptr::null();
    let code = unsafe { ggtts_load_with(engine.as_ptr(), null, null, null, 0, &settings, &mut h) };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    refused_offline_in(render_steered(h), &cache).unwrap();
    ggtts_free(h);
    assert!(!cache.join("urls").exists(), "an offline engine wrote into its cache");
    let _ = std::fs::remove_dir_all(dir);
}
