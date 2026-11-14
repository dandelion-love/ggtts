//! The C ABI's options documents (version 4): `ggtts_load_opts` and `ggtts_render_opts` on the fake engine. A load
//! document gives what the equivalent version-3 struct gives, field by field; a render document gives what the struct
//! functions give; every malformed or refused document is `GGTTS_ERR_ARG` before anything renders; and the struct
//! callers of versions 1, 2 and 3 still pass. The real-model cases (steering on the tiny Qwen3-TTS model, a speaker and
//! an instruction) are opt-in:
//!
//! ```text
//! GGTTS_TALKER=<talker> GGTTS_CODEC=<codec> cargo test -p ggtts-ffi --features ggml --test options_documents -- --ignored
//! ```

use std::ffi::{c_void, CString};
use std::ptr;

use ggtts_ffi::*;

const TEXT: &str = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。";

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// `ggtts_load_opts(json)`: its code and handle.
fn load_doc(json: &str) -> (i32, u64) {
    let mut h = 0u64;
    let code = unsafe { ggtts_load_opts(c(json).as_ptr(), &mut h) };
    (code, h)
}

/// `ggtts_load_with` with a full version-3 struct, its strings given or NULL.
#[allow(clippy::too_many_arguments)]
fn load_struct(
    engine: &str,
    model: Option<&str>,
    codec: Option<&str>,
    speaker: Option<&str>,
    instruction: Option<&str>,
    device: Option<&str>,
    cache_dir: Option<&str>,
    allow_download: i32,
) -> (i32, u64) {
    let (e, m, k) = (c(engine), model.map(c), codec.map(c));
    let (s, i, d, cd) = (speaker.map(c), instruction.map(c), device.map(c), cache_dir.map(c));
    let p = |x: &Option<CString>| x.as_ref().map_or(ptr::null(), |x| x.as_ptr());
    let settings = GgttsSettings {
        size: std::mem::size_of::<GgttsSettings>(),
        speaker: p(&s),
        instruction: p(&i),
        device: p(&d),
        cache_dir: p(&cd),
        allow_download,
    };
    let mut h = 0u64;
    let code = unsafe { ggtts_load_with(e.as_ptr(), p(&m), p(&k), ptr::null(), 0, &settings, &mut h) };
    (code, h)
}

/// `ggtts_render(handle, TEXT, seed)`: its samples' bits.
fn render_struct(h: u64, seed: u64) -> Vec<u32> {
    let text = c(TEXT);
    let (mut p, mut n, mut r) = (ptr::null_mut(), 0usize, 0u32);
    let code = unsafe { ggtts_render(h, text.as_ptr(), seed, &mut p, &mut n, &mut r) };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let out = unsafe { std::slice::from_raw_parts(p, n) }
        .iter()
        .map(|x| x.to_bits())
        .collect();
    unsafe { ggtts_pcm_free(p, n) };
    out
}

/// `ggtts_render_opts(handle, TEXT, json)` as a whole read: its code and samples' bits.
fn render_doc(h: u64, json: Option<&str>) -> (i32, Vec<u32>) {
    let text = c(TEXT);
    let doc = json.map(c);
    let (mut p, mut n, mut r) = (ptr::null_mut(), 0usize, 0u32);
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            doc.as_ref().map_or(ptr::null(), |d| d.as_ptr()),
            None,
            ptr::null_mut(),
            &mut p,
            &mut n,
            &mut r,
        )
    };
    let out = if code == GGTTS_OK {
        let v = unsafe { std::slice::from_raw_parts(p, n) }
            .iter()
            .map(|x| x.to_bits())
            .collect();
        unsafe { ggtts_pcm_free(p, n) };
        v
    } else {
        Vec::new()
    };
    (code, out)
}

extern "C" fn collect(user: *mut c_void, samples: *const f32, len: usize, _: u32) -> i32 {
    let out = unsafe { &mut *(user as *mut Vec<u32>) };
    out.extend(
        unsafe { std::slice::from_raw_parts(samples, len) }
            .iter()
            .map(|x| x.to_bits()),
    );
    0
}

/// A fresh fake engine, loaded by document. The fake engine remembers what it rendered (redraws of a text, and the
/// texts it has seen), so every comparison starts from a fresh one.
fn fresh() -> u64 {
    let (code, h) = load_doc(r#"{"load": {"engine": "fake"}}"#);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    h
}

/// A render document is what the struct functions give for the same seed: the whole read, and the stream.
#[test]
fn a_render_document_gives_what_the_struct_functions_give() {
    let h = fresh();
    let want = render_struct(h, 7);
    ggtts_free(h);
    let h = fresh();
    assert_eq!(
        render_doc(h, Some(r#"{"render": {"seed_salt": "7"}}"#)),
        (GGTTS_OK, want.clone())
    );
    ggtts_free(h);
    assert!(!want.is_empty());
    // The stream: the pieces of ggtts_render_stream, through the document.
    let stream = |doc: Option<&str>| {
        let h = fresh();
        let text = c(TEXT);
        let mut got: Vec<u32> = Vec::new();
        let user = &mut got as *mut Vec<u32> as *mut c_void;
        let code = match doc {
            None => unsafe { ggtts_render_stream(h, text.as_ptr(), 7, Some(collect), user) },
            Some(d) => {
                let d = c(d);
                let null = ptr::null_mut();
                unsafe {
                    ggtts_render_opts(
                        h,
                        text.as_ptr(),
                        d.as_ptr(),
                        Some(collect),
                        user,
                        null,
                        null as _,
                        null as _,
                    )
                }
            }
        };
        assert_eq!(code, GGTTS_OK, "{}", last_error());
        assert_eq!(ggtts_free(h), GGTTS_OK);
        got
    };
    let by_struct = stream(None);
    assert!(!by_struct.is_empty());
    assert_eq!(
        stream(Some(r#"{"render": {"seed_salt": "7"}, "output": {"stream": "raw"}}"#)),
        by_struct
    );
}

/// The struct functions render as the shared glue always has: the document path changed no sample.
#[test]
fn the_struct_render_is_the_glues_render_sample_for_sample() {
    let (code, h) = load_struct("fake", None, None, None, None, None, None, 0);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let mut engine = ggtts_fake::FakeEngine::default();
    let (want, _) = ggtts_load::binding::render(&mut engine, TEXT, 20260905, None).unwrap();
    let want: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
    assert_eq!(render_struct(h, 20260905), want);
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

/// AC3 of #280, on the fake engine: each field of a version-3 struct and its document key give the same: the same
/// samples where the engine takes it, the same refusal where it does not.
#[test]
fn a_load_document_gives_what_its_struct_gives_field_by_field() {
    let render_of = |(code, h): (i32, u64)| {
        assert_eq!(code, GGTTS_OK, "{}", last_error());
        let pcm = render_struct(h, 7);
        ggtts_free(h);
        pcm
    };
    let base = render_of(load_struct("fake", None, None, None, None, None, None, 0));
    // engine: samples.
    let by_doc = render_of(load_doc(r#"{"load": {"engine": "fake"}}"#));
    assert_eq!(by_doc, base, "engine");
    // device: the fake engine runs on the CPU only, so a document's "cuda" reaches it and is refused, as the struct's is.
    let (code, h) = load_struct("fake", None, None, None, None, Some("cuda"), None, 0);
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0), "struct device: {}", last_error());
    let (code, h) = load_doc(r#"{"load": {"engine": "fake", "device": "cuda"}}"#);
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0), "document device: {}", last_error());
    assert!(last_error().contains("cuda"), "{}", last_error());
    // speaker and instruction: the fake engine refuses both, in the same words either way.
    for (field, doc, st) in [
        (
            "speaker",
            r#"{"load": {"engine": "fake", "speaker": "hk_voice_777"}}"#,
            ("hk_voice_777", None),
        ),
        (
            "instruction",
            r#"{"load": {"engine": "fake"}, "render": {"instruction": "Speak slowly."}}"#,
            ("", Some("Speak slowly.")),
        ),
    ] {
        let speaker = Some(st.0).filter(|s| !s.is_empty());
        let (code, _) = load_struct("fake", None, None, speaker, st.1, None, None, 0);
        let by_struct = (code, last_error());
        let (code, _) = load_doc(doc);
        assert_eq!((code, last_error()), by_struct, "{field}");
        assert_eq!(code, GGTTS_ERR_ARG, "{field}");
    }
}

/// AC4 of #280: each is GGTTS_ERR_ARG before anything renders, naming what is wrong.
#[test]
fn a_bad_document_is_an_argument_error_before_anything_renders() {
    let (code, h) = load_doc(r#"{"load": {"engine": "fake"}}"#);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    for (doc, names) in [
        (r#"{"render": {"no_such_option": 1}}"#, "render.no_such_option"),
        (r#"{"render": "#, "not JSON"),
        (r#"{"load": {"threads": 2}}"#, "load.threads"),
        (r#"{"render": {"redraw": "two"}}"#, "render.redraw"),
        (r#"{"output": {"stream": "levelled"}}"#, "output.stream"),
        (r#"{"observe": {"timing_report": "t.json"}}"#, "observe.timing_report"),
        (r#"{"render": {"steer": "s.safetensors"}}"#, "render.steer"),
    ] {
        let (code, _) = render_doc(h, Some(doc));
        assert_eq!(code, GGTTS_ERR_ARG, "{doc}: {}", last_error());
        assert!(last_error().contains(names), "{doc}: {}", last_error());
    }
    // A stream with no callback, and a whole read with one.
    let text = c(TEXT);
    let stream = c(r#"{"output": {"stream": "raw"}}"#);
    let null = ptr::null_mut();
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            stream.as_ptr(),
            None,
            null,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, GGTTS_ERR_ARG, "{}", last_error());
    assert!(last_error().contains("on_piece"), "{}", last_error());
    let whole = c("{}");
    let (mut p, mut n, mut r) = (ptr::null_mut(), 0usize, 0u32);
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            whole.as_ptr(),
            Some(collect),
            null,
            &mut p,
            &mut n,
            &mut r,
        )
    };
    assert_eq!(code, GGTTS_ERR_ARG, "{}", last_error());
    assert!(last_error().contains("on_piece"), "{}", last_error());
    assert_eq!(ggtts_free(h), GGTTS_OK);
    // A bad load document writes no handle.
    for doc in [
        r#"{"load": {"engine": "fake", "threads": 0}}"#,
        r#"{"observe": {"trace": "t.jsonl"}}"#,
        r#"not json"#,
    ] {
        let (code, h) = load_doc(doc);
        assert_eq!((code, h), (GGTTS_ERR_ARG, 0), "{doc}: {}", last_error());
    }
}

/// AC3 of #280, the default model files: a document's `cache_dir` and `offline` reach the resolution as the struct's
/// `cache_dir` and `allow_download` do. An empty cache with downloading off is refused naming a file under that cache,
/// in each caller's own words.
#[cfg(feature = "ggml")]
#[test]
fn a_documents_cache_dir_and_offline_reach_the_default_files() {
    let dir = std::env::temp_dir().join(format!("ggtts-doc-empty-cache-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cache = dir.to_string_lossy().into_owned();
    let (code, h) = load_struct("qwen3-tts", None, None, None, None, None, Some(&cache), 0);
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    let by_struct = last_error();
    assert!(
        by_struct.contains(&cache) && by_struct.contains("allow_download"),
        "{by_struct}"
    );
    let doc = format!(
        r#"{{"load": {{"engine": "qwen3-tts", "cache_dir": {}, "offline": true}}}}"#,
        serde_json::to_string(&cache).unwrap()
    );
    let (code, h) = load_doc(&doc);
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    let by_doc = last_error();
    assert!(by_doc.contains(&cache), "the document's cache_dir: {by_doc}");
    assert!(
        by_doc.contains("load.offline") && !by_doc.contains("allow_download"),
        "{by_doc}"
    );
    assert_eq!(
        by_doc.replace(
            "set load.offline to false to allow it",
            "allow it with the allow_download setting"
        ),
        by_struct,
        "the same refusal, in the document's words"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// AC3 of #280, the device: a document's `load.device` reaches the load as the struct's `device` does. A build without
/// the CUDA backend refuses "cuda" before reading any file, naming the setting in each caller's words.
#[cfg(all(feature = "ggml", not(feature = "cuda")))]
#[test]
fn a_documents_device_reaches_the_load() {
    let dir = std::env::temp_dir().join(format!("ggtts-doc-device-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cache = dir.to_string_lossy().into_owned();
    let (code, h) = load_struct("qwen3-tts", None, None, None, None, Some("cuda"), Some(&cache), 0);
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    let by_struct = last_error();
    assert!(
        by_struct.contains("settings.device cuda is not available"),
        "{by_struct}"
    );
    let doc = format!(
        r#"{{"load": {{"engine": "qwen3-tts", "device": "cuda", "cache_dir": {}, "offline": true}}}}"#,
        serde_json::to_string(&cache).unwrap()
    );
    let (code, h) = load_doc(&doc);
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    assert_eq!(last_error(), by_struct.replace("settings.device", "load.device"));
    let _ = std::fs::remove_dir_all(dir);
}

/// A struct load names its engine: "auto" is a document's only, and the struct loads refuse it as they always have.
#[test]
fn auto_is_a_documents_engine_and_the_struct_loads_still_refuse_it() {
    let mut h = 0u64;
    let auto = c("auto");
    let null = ptr::null();
    let code = unsafe { ggtts_load(auto.as_ptr(), null, null, null, 0, &mut h) };
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    assert!(last_error().contains(r#"no engine named "auto""#), "{}", last_error());
    let (code, h) = load_struct("auto", None, None, None, None, None, None, 0);
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{}", last_error());
    assert!(last_error().contains(r#"no engine named "auto""#), "{}", last_error());
}

/// A stream's outputs are all three or none: one alone has nowhere to write, and is refused.
#[test]
fn a_streams_outputs_are_all_set_or_all_null() {
    let h = fresh();
    let text = c(TEXT);
    let doc = c(r#"{"output": {"stream": "raw"}}"#);
    let mut p = ptr::null_mut();
    let mut got: Vec<u32> = Vec::new();
    let user = &mut got as *mut Vec<u32> as *mut c_void;
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            doc.as_ptr(),
            Some(collect),
            user,
            &mut p,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, GGTTS_ERR_ARG, "{}", last_error());
    assert!(got.is_empty(), "refused before anything rendered");
    // A whole read with only some of them set is told they must not be NULL.
    let whole = c("{}");
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            whole.as_ptr(),
            None,
            ptr::null_mut(),
            &mut p,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, GGTTS_ERR_ARG, "{}", last_error());
    assert_eq!(last_error(), "pcm, len and rate must not be NULL");
    // All three set: the pieces and the whole read.
    let (mut n, mut r) = (0usize, 0u32);
    let code = unsafe {
        ggtts_render_opts(
            h,
            text.as_ptr(),
            doc.as_ptr(),
            Some(collect),
            user,
            &mut p,
            &mut n,
            &mut r,
        )
    };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert!(!got.is_empty() && n > 0);
    unsafe { ggtts_pcm_free(p, n) };
    ggtts_free(h);
}

/// AC2 of #280, the refusal half: steering for kitten-tts-2 is refused before any file is read.
#[test]
fn steering_on_kitten_is_refused_by_name_before_anything_is_read() {
    let (code, h) = load_doc(r#"{"load": {"engine": "kitten-tts-2"}, "render": {"steer": "s.safetensors"}}"#);
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0), "{}", last_error());
    assert!(
        last_error().starts_with("render.steer: the kitten-tts-2 engine cannot apply steering"),
        "{}",
        last_error()
    );
}

/// The struct callers of versions 1, 2 and 3 still load and render as they did.
#[test]
fn struct_callers_of_every_version_still_pass() {
    let mut h = 0u64;
    let e = c("fake");
    let null = ptr::null();
    assert_eq!(unsafe { ggtts_load(e.as_ptr(), null, null, null, 0, &mut h) }, GGTTS_OK);
    let v1 = render_struct(h, 7);
    ggtts_free(h);
    let v2 = GgttsSettings {
        size: GGTTS_SETTINGS_V2_SIZE,
        speaker: null,
        instruction: null,
        device: null,
        cache_dir: null,
        allow_download: 0,
    };
    assert_eq!(
        unsafe { ggtts_load_with(e.as_ptr(), null, null, null, 0, &v2, &mut h) },
        GGTTS_OK
    );
    assert_eq!(render_struct(h, 7), v1, "version 2");
    ggtts_free(h);
    let (code, h) = load_struct("fake", None, None, None, None, None, None, 0);
    assert_eq!(code, GGTTS_OK);
    assert_eq!(render_struct(h, 7), v1, "version 3");
    ggtts_free(h);
}

/// AC2 of #280 on the tiny (or any) real Qwen3-TTS model: steering through a render document renders steered audio,
/// different from the same render without it, and repeatably.
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC (a real Qwen3-TTS talker and codec) and a ggml build"]
fn a_steered_render_document_renders_steered_audio_on_a_real_model() {
    let env = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("set {n}"));
    let talker = env("GGTTS_TALKER");
    let doc = format!(
        r#"{{"load": {{"engine": "qwen3-tts", "model": {}, "codec": {}, "offline": true}}}}"#,
        serde_json::to_string(&talker).unwrap(),
        serde_json::to_string(&env("GGTTS_CODEC")).unwrap()
    );
    let (code, h) = load_doc(&doc);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    // One vector per talker layer 0, of the talker's hidden size, read from the steering refusal for a wrong one.
    let dir = std::env::temp_dir().join(format!("ggtts-doc-steer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("steer.safetensors");
    let hidden = std::env::var("GGTTS_HIDDEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048usize);
    let v = vec![0.01f32; hidden];
    std::fs::write(&path, ggtts_core::steer::write_safetensors(&[("talker.layer.0", &v)])).unwrap();
    let steer = format!(
        r#"{{"render": {{"seed_salt": "7", "steer": {}, "steer_scale": 0.5}}}}"#,
        serde_json::to_string(&path.to_string_lossy()).unwrap()
    );
    let plain = render_doc(h, Some(r#"{"render": {"seed_salt": "7"}}"#));
    assert_eq!(plain.0, GGTTS_OK, "{}", last_error());
    let steered = render_doc(h, Some(&steer));
    assert_eq!(steered.0, GGTTS_OK, "{}", last_error());
    assert_ne!(steered.1, plain.1, "steering changed nothing");
    assert_eq!(render_doc(h, Some(&steer)), steered, "a steered render repeats");
    ggtts_free(h);
    let _ = std::fs::remove_dir_all(dir);
}

/// AC3 of #280 on a real Qwen3-TTS model: a speaker, an empty instruction and the device, by document and by struct,
/// give the same samples.
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC (a real Qwen3-TTS talker and codec) and a ggml build"]
fn a_load_document_gives_what_its_struct_gives_on_a_real_model() {
    let env = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("set {n}"));
    let (talker, codec) = (env("GGTTS_TALKER"), env("GGTTS_CODEC"));
    let speaker = std::env::var("GGTTS_SPEAKER").ok();
    let j = |s: &str| serde_json::to_string(s).unwrap();
    let mut load = format!(
        r#""engine": "qwen3-tts", "model": {}, "codec": {}, "device": "cpu""#,
        j(&talker),
        j(&codec)
    );
    if let Some(s) = &speaker {
        load.push_str(&format!(r#", "speaker": {}"#, j(s)));
    }
    let doc = format!(r#"{{"load": {{{load}}}, "render": {{"instruction": ""}}}}"#);
    let (code, hd) = load_doc(&doc);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let (code, hs) = load_struct(
        "qwen3-tts",
        Some(&talker),
        Some(&codec),
        speaker.as_deref(),
        Some(""),
        Some("cpu"),
        None,
        0,
    );
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(render_struct(hd, 7), render_struct(hs, 7));
    ggtts_free(hd);
    ggtts_free(hs);
}
