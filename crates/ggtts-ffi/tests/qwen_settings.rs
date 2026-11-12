//! The voice settings of `ggtts_load_with` on real Qwen3-TTS checkpoints: the speaker and the instruction reach the
//! engine, neither set gives `ggtts_load`'s output, and an unknown speaker or an instruction on a Base checkpoint is a
//! code and a message. Opt-in, as the other model tests:
//!
//! ```text
//! GGTTS_TALKER=<custom_voice 1.7B talker with two speakers> GGTTS_CODEC=<codec> \
//!     cargo test -p ggtts-ffi --features ggml --test qwen_settings -- --ignored --nocapture
//! ```
//!
//! The published v0.0.5 GGUF (`jmtl/hk-cantonese-cc0-qwen3-tts-GGUF`, speakers `hk_voice_343` and `hk_voice_777`)
//! is such a talker. `GGTTS_BASE_TALKER`, a Qwen3-TTS Base talker GGUF, is needed by the Base test only. The
//! default-model test needs `GGTTS_TALKER` and `GGTTS_CODEC` to be the pinned v0.0.5 files themselves (it puts them
//! in a cache of its own, where the library verifies their sha256). The CUDA test needs a build with the `cuda`
//! feature and a GPU:
//!
//! ```text
//! cargo test -p ggtts-ffi --features cuda --test qwen_settings -- --ignored --nocapture cuda
//! ```
#![cfg(feature = "ggml")]

use std::ffi::{c_char, CString};
use std::ptr;
use std::time::Instant;

use ggtts_ffi::*;

const LINE: &str = "今日天氣好好，我哋去海邊行下。";
const INSTRUCTION: &str = "Speak slowly and softly.";

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

/// Load Qwen3-TTS from `talker` and `GGTTS_CODEC`, through `ggtts_load` (`settings` `None`) or `ggtts_load_with`.
fn load(talker: &str, settings: Option<(Option<&str>, Option<&str>)>) -> (i32, u64) {
    let (e, m, k) = (c("qwen3-tts"), c(talker), c(&env("GGTTS_CODEC")));
    let strings = settings.map(|(s, i)| (s.map(c), i.map(c)));
    let p = |s: &Option<CString>| s.as_ref().map_or(ptr::null::<c_char>(), |s| s.as_ptr());
    let mut h = 0u64;
    let code = match &strings {
        None => unsafe { ggtts_load(e.as_ptr(), m.as_ptr(), k.as_ptr(), ptr::null(), 4, &mut h) },
        Some((s, i)) => {
            let st = GgttsSettings {
                speaker: p(s),
                instruction: p(i),
                ..unset()
            };
            unsafe { ggtts_load_with(e.as_ptr(), m.as_ptr(), k.as_ptr(), ptr::null(), 4, &st, &mut h) }
        }
    };
    (code, h)
}

/// A version-3 struct with no setting: `GGTTS_SETTINGS_INIT`.
fn unset() -> GgttsSettings {
    GgttsSettings {
        size: std::mem::size_of::<GgttsSettings>(),
        speaker: ptr::null(),
        instruction: ptr::null(),
        device: ptr::null(),
        cache_dir: ptr::null(),
        allow_download: 0,
    }
}

/// `ggtts_load_with` of Qwen3-TTS with a version-3 struct: `model` and `codec` NULL for `None`.
fn load_v3(model: Option<&str>, codec: Option<&str>, st: GgttsSettings) -> (i32, u64) {
    let (e, m, k) = (c("qwen3-tts"), model.map(c), codec.map(c));
    let p = |s: &Option<CString>| s.as_ref().map_or(ptr::null::<c_char>(), |s| s.as_ptr());
    let mut h = 0u64;
    let code = unsafe { ggtts_load_with(e.as_ptr(), p(&m), p(&k), ptr::null(), 4, &st, &mut h) };
    (code, h)
}

/// Render `LINE` at the default seed and free the engine.
fn render_once(h: u64, what: &str) -> Vec<f32> {
    let t = c(LINE);
    let (mut p, mut n, mut r) = (ptr::null_mut(), 0usize, 0u32);
    let start = Instant::now();
    let code = unsafe { ggtts_render(h, t.as_ptr(), 20260905, &mut p, &mut n, &mut r) };
    assert_eq!(code, GGTTS_OK, "{what}: {}", last_error());
    let v = unsafe { std::slice::from_raw_parts(p, n) }.to_vec();
    unsafe { ggtts_pcm_free(p, n) };
    assert_eq!(ggtts_free(h), GGTTS_OK);
    println!(
        "{what}: {:.2} s of audio at {r} Hz in {:.1} s",
        n as f64 / r as f64,
        start.elapsed().as_secs_f64()
    );
    v
}

fn spoken(talker: &str, settings: Option<(Option<&str>, Option<&str>)>, what: &str) -> Vec<f32> {
    let (code, h) = load(talker, settings);
    assert_eq!(code, GGTTS_OK, "{what}: {}", last_error());
    render_once(h, what)
}

/// The model's speakers, read from the unknown-speaker refusal, which lists them.
fn speakers(talker: &str) -> Vec<String> {
    let (code, h) = load(talker, Some((Some("no_such_speaker"), None)));
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    let msg = last_error();
    println!("unknown speaker: {msg}");
    assert!(msg.contains("no speaker no_such_speaker"), "{msg}");
    assert!(!msg.contains("--speaker"), "a binding caller has no flag: {msg}");
    let list = msg
        .split("speakers are: ")
        .nth(1)
        .and_then(|s| s.split(". ").next())
        .unwrap_or_else(|| panic!("no speaker list: {msg}"));
    list.split(", ").map(String::from).collect()
}

/// AC4 (speaker) and AC1: an unknown speaker is refused with the model's speakers; each listed speaker loads and
/// renders, and two speakers render differently, so the speaker reaches the engine.
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC, a custom_voice talker with two speakers and its codec"]
fn the_speaker_reaches_the_engine_and_an_unknown_one_is_refused() {
    let talker = env("GGTTS_TALKER");
    let names = speakers(&talker);
    assert!(names.len() >= 2, "{names:?}");
    let a = spoken(&talker, Some((Some(&names[0]), None)), &names[0]);
    let b = spoken(&talker, Some((Some(&names[1]), None)), &names[1]);
    assert!(!a.is_empty() && !b.is_empty());
    assert_ne!(a, b, "{} and {} rendered the same samples", names[0], names[1]);
    // Case is ignored, as in the library.
    let upper = names[0].to_uppercase();
    assert_eq!(spoken(&talker, Some((Some(&upper), None)), &upper), a);
}

/// AC1 and AC3 for the instruction: it changes the render; an empty one is none.
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC, a custom_voice 1.7B talker and its codec"]
fn the_instruction_reaches_the_engine_and_an_empty_one_is_none() {
    let talker = env("GGTTS_TALKER");
    let speaker = speakers(&talker).remove(0);
    let plain = spoken(&talker, Some((Some(&speaker), None)), "no instruction");
    let styled = spoken(
        &talker,
        Some((Some(&speaker), Some(INSTRUCTION))),
        "with the instruction",
    );
    assert_ne!(plain, styled, "the instruction left the samples unchanged");
    assert_eq!(
        spoken(&talker, Some((Some(&speaker), Some(""))), "an empty instruction"),
        plain
    );
}

/// AC3 on a real checkpoint: `ggtts_load_with` with NULL settings, NULL strings or empty strings gives what
/// `ggtts_load` gives: the same samples, or (a talker without the default speaker) the same refusal.
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC"]
fn neither_setting_is_ggtts_load_on_qwen() {
    let talker = env("GGTTS_TALKER");
    let (code, h) = load(&talker, None);
    let today = if code == GGTTS_OK {
        Ok(render_once(h, "ggtts_load"))
    } else {
        Err((code, last_error()))
    };
    for settings in [Some((None, None)), Some((Some(""), Some("")))] {
        let (code, h) = load(&talker, settings);
        let now = if code == GGTTS_OK {
            Ok(render_once(h, &format!("ggtts_load_with {settings:?}")))
        } else {
            Err((code, last_error()))
        };
        assert!(
            now == today,
            "{settings:?}: differs from ggtts_load ({:?})",
            now.as_ref().err()
        );
    }
}

/// AC4 (instruction): a Base checkpoint refuses an instruction while loading, with the library's message, and writes
/// no handle.
#[test]
#[ignore = "needs GGTTS_BASE_TALKER, a Qwen3-TTS Base talker GGUF, and GGTTS_CODEC"]
fn an_instruction_on_a_base_checkpoint_is_refused() {
    let base = env("GGTTS_BASE_TALKER");
    let (code, h) = load(&base, Some((None, Some(INSTRUCTION))));
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    let msg = last_error();
    println!("base + instruction: {msg}");
    assert!(
        msg.contains("base model") && msg.contains("takes no style instruction"),
        "{msg}"
    );
}

/// #164: with no paths and no speaker, a version-3 load finds the pinned default files in the cache, verifies them,
/// and renders with the default model's default speaker: the same samples as the same files given as paths with
/// `hk_voice_343`. A given model keeps `ggtts_load`'s default speaker, and a given codec is used as it is. Nothing is
/// downloaded (downloading stays off).
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC to be the pinned v0.0.5 talker and codec"]
fn the_default_model_and_its_default_speaker_come_from_the_cache() {
    let (talker, codec) = (env("GGTTS_TALKER"), env("GGTTS_CODEC"));
    let root = std::env::temp_dir().join(format!("ggtts-ffi-default-model-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let cache = ggtts_models::Cache::new(root.clone());
    for (pinned, file) in ggtts_models::QWEN3_TTS.files.iter().zip([&talker, &codec]) {
        let path = cache.path_for(pinned);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(std::fs::canonicalize(file).unwrap(), &path).unwrap();
    }
    let dir = c(root.to_str().unwrap());
    let st = GgttsSettings {
        cache_dir: dir.as_ptr(),
        ..unset()
    };

    let (code, h) = load_v3(None, None, st);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let by_default = render_once(h, "the default model and speaker");
    let explicit = spoken(
        &talker,
        Some((Some("hk_voice_343"), None)),
        "the same files with hk_voice_343",
    );
    assert!(!by_default.is_empty());
    assert_eq!(by_default, explicit, "the default speaker is not hk_voice_343");

    // A given codec is used as it is, with the default talker and speaker.
    let (code, h) = load_v3(None, Some(&codec), st);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(render_once(h, "a given codec"), explicit);

    // A given talker keeps ggtts_load's default speaker, which the v0.0.5 talker does not have.
    let (code, h) = load_v3(Some(&talker), None, st);
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0));
    assert!(
        last_error().contains("no speaker hk_cantonese_speaker"),
        "{}",
        last_error()
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The memory the GPU has in use, in MiB, from `nvidia-smi`; `None` without it.
#[cfg(feature = "cuda")]
fn gpu_memory_used() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()
}

/// #167: a CUDA build renders Qwen3-TTS on the GPU through `ggtts_load_with` with `device` `cuda`. The engine reports
/// the ggml CUDA backend (which loads only when ggml initialises a GPU device), the GPU's memory grows by the model
/// while the engine is held, and the samples differ from the CPU's at the same seed (the CUDA path packs Q, K and V,
/// so it is a different computation). `cpu`, and no device, render on the CPU.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA build (--features cuda), a GPU, GGTTS_TALKER and GGTTS_CODEC"]
fn cuda_renders_qwen_on_the_gpu() {
    let (talker, codec) = (env("GGTTS_TALKER"), env("GGTTS_CODEC"));
    let speaker = speakers(&talker).remove(0);
    let s = c(&speaker);
    let on = |device: Option<&str>| {
        let d = device.map(c);
        let st = GgttsSettings {
            speaker: s.as_ptr(),
            device: d.as_ref().map_or(ptr::null(), |d| d.as_ptr()),
            ..unset()
        };
        load_v3(Some(&talker), Some(&codec), st)
    };

    let before = gpu_memory_used();
    let (code, h) = on(Some("cuda"));
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(backend_of(h).as_deref(), Ok("ggml-cuda"));
    let held = gpu_memory_used();
    println!("GPU memory in use: {before:?} MiB before the load, {held:?} MiB with the engine held");
    if let (Some(before), Some(held)) = (before, held) {
        assert!(
            held >= before + 1024,
            "the GPU's memory grew by {} MiB",
            held as i64 - before as i64
        );
    }
    let gpu = render_once(h, "cuda");

    let (code, h) = on(Some("cpu"));
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(backend_of(h).as_deref(), Ok("ggml-cpu"));
    let cpu = render_once(h, "cpu");
    let (code, h) = on(None);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(backend_of(h).as_deref(), Ok("ggml-cpu"), "no device is the CPU");
    assert_eq!(render_once(h, "no device"), cpu);

    assert!(!gpu.is_empty() && !cpu.is_empty());
    assert_ne!(gpu, cpu, "the GPU rendered the CPU's samples");
}
