//! The C ABI's introspection (version 5): `ggtts_options_schema` is the committed schema byte for byte,
//! `ggtts_capabilities` states what a loaded engine supports and its speakers, and `ggtts_last_error_key` names the
//! option a `GGTTS_ERR_ARG` refused, and nothing otherwise. The fake engine runs everywhere; the real models are
//! opt-in, one model loaded at a time:
//!
//! ```text
//! GGTTS_TALKER=<talker> GGTTS_CODEC=<codec> cargo test -p ggtts-ffi --features ggml --test introspection -- --ignored
//! GGTTS_KITTEN_LM=<lm> GGTTS_KITTEN_S3GEN=<s3gen> GGTTS_KITTEN_VOICE=<voice.json> cargo test -p ggtts-ffi \
//!     --features ggml --test introspection -- --ignored
//! ```

use std::ffi::{c_char, CStr, CString};
use std::ptr;

use ggtts_core::capabilities::Capabilities;
use ggtts_ffi::*;
use serde_json::{json, Value};

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// `ggtts_load_opts(json)`: its code and handle.
fn load_doc(json: &str) -> (i32, u64) {
    let mut h = 0u64;
    let code = unsafe { ggtts_load_opts(c(json).as_ptr(), &mut h) };
    (code, h)
}

/// `ggtts_capabilities(handle)`: its code and the statement, parsed (`None` when `*json` stayed NULL).
fn capabilities(handle: u64) -> (i32, Option<Value>) {
    let mut out: *mut c_char = ptr::null_mut();
    let code = unsafe { ggtts_capabilities(handle, &mut out) };
    if out.is_null() {
        return (code, None);
    }
    let text = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_string();
    unsafe { ggtts_string_free(out) };
    (code, Some(serde_json::from_str(&text).unwrap()))
}

/// `ggtts_last_error_key()` as a Rust value: `None` for NULL.
fn key() -> Option<String> {
    let p = ggtts_last_error_key();
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string())
}

/// The statement an engine with `caps` and `speakers` must give, spelled out here rather than by the library's own
/// function, so a field the library drops, renames or reshapes is red.
fn expected(caps: &Capabilities, speakers: &[String]) -> Value {
    json!({
        "format": "ggtts-capabilities/1",
        "family": caps.family,
        "instruction": caps.instruction,
        "no_instruction_why": caps.no_instruction_why,
        "steering": caps.steering,
        "speakers": caps.speakers,
        "speaker_names": speakers,
        "devices": caps.devices,
        "hooks": {"p_stop": caps.hooks.p_stop, "logits": caps.hooks.logits},
        "step_events": caps.step_events,
    })
}

/// AC1: the schema is the committed file, byte for byte, and the same static string on every call.
#[test]
fn the_options_schema_is_the_committed_file() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // A Windows checkout may hold the file with CRLF line ends (core.autocrlf): the bytes git stores are LF.
    let committed = std::fs::read_to_string(root.join("schemas/ggtts-options-1.schema.json"))
        .unwrap()
        .replace("\r\n", "\n");
    let p = ggtts_options_schema();
    assert!(!p.is_null());
    let got = unsafe { CStr::from_ptr(p) }.to_bytes();
    assert!(
        got == committed.as_bytes(),
        "ggtts_options_schema() is not schemas/ggtts-options-1.schema.json"
    );
    assert_eq!(ggtts_options_schema(), p, "one static string");
    // It is the schema itself, not a copy of some file: it names its own id.
    let v: Value = serde_json::from_slice(got).unwrap();
    assert_eq!(v["$id"], ggtts_core::options::SCHEMA_ID);
    assert_eq!(ggtts_abi_version(), 5);
}

/// AC2 on the fake engine: what `Engine::capabilities` reports, with no speakers.
#[test]
fn the_fake_engines_capabilities() {
    let (code, h) = load_doc(r#"{"load": {"engine": "fake"}}"#);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let (code, statement) = capabilities(h);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let caps = ggtts_core::engine::Engine::capabilities(&ggtts_fake::FakeEngine::default());
    assert_eq!(statement.unwrap(), expected(&caps, &[]));
    assert_eq!(last_error(), "");
    assert_eq!(key(), None);
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

/// AC4 (rejection): a freed, never-loaded or zero handle is GGTTS_ERR_HANDLE, and `*json` is NULL; a NULL `json` is
/// GGTTS_ERR_ARG with no key.
#[test]
fn capabilities_of_no_engine_is_a_handle_error() {
    let (code, h) = load_doc(r#"{"load": {"engine": "fake"}}"#);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(ggtts_free(h), GGTTS_OK);
    for handle in [h, 0, u64::MAX] {
        let mut out: *mut c_char = ptr::dangling_mut(); // not NULL, so the call must clear it
        let code = unsafe { ggtts_capabilities(handle, &mut out) };
        assert_eq!(code, GGTTS_ERR_HANDLE, "handle {handle}: {}", last_error());
        assert!(out.is_null(), "handle {handle}: *json was left set");
        assert!(last_error().contains("no engine with handle"), "{}", last_error());
        assert_eq!(key(), None);
    }
    let (code, h) = load_doc(r#"{"load": {"engine": "fake"}}"#);
    assert_eq!(code, GGTTS_OK);
    assert_eq!(unsafe { ggtts_capabilities(h, ptr::null_mut()) }, GGTTS_ERR_ARG);
    assert_eq!(last_error(), "json is NULL");
    assert_eq!(key(), None);
    ggtts_free(h);
    unsafe { ggtts_string_free(ptr::null_mut()) };
}

/// AC3: a refused option sets its schema key; a refusal that is not about one option, and every success, leave it NULL.
#[test]
fn the_refused_key_names_the_option_and_only_that() {
    // Steering an engine that cannot steer is refused by name, before any file is read.
    let (code, h) = load_doc(r#"{"load": {"engine": "kitten-tts-2"}, "render": {"steer": "s.safetensors"}}"#);
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
    assert_eq!(key().as_deref(), Some("render.steer"), "{}", last_error());
    assert!(last_error().starts_with("render.steer: "), "{}", last_error());

    // A success clears it.
    let (code, h) = load_doc(r#"{"load": {"engine": "fake"}}"#);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(key(), None);

    // An unknown key, in a render document and in a load document: that key.
    let text = c("今日天氣好好。");
    let (mut pcm, mut len, mut rate) = (ptr::null_mut(), 0usize, 0u32);
    let render = |doc: &str, pcm: &mut *mut f32, len: &mut usize, rate: &mut u32| unsafe {
        ggtts_render_opts(h, text.as_ptr(), c(doc).as_ptr(), None, ptr::null_mut(), pcm, len, rate)
    };
    assert_eq!(
        render(r#"{"render": {"no_such_option": 1}}"#, &mut pcm, &mut len, &mut rate),
        GGTTS_ERR_ARG
    );
    assert_eq!(key().as_deref(), Some("render.no_such_option"), "{}", last_error());
    let (code, _) = load_doc(r#"{"load": {"engine": "fake", "no_such_option": 1}}"#);
    assert_eq!(code, GGTTS_ERR_ARG);
    assert_eq!(key().as_deref(), Some("load.no_such_option"), "{}", last_error());

    // An option a binding does not take, a load option in a render document, and steering the fake engine.
    for (doc, want) in [
        (r#"{"observe": {"log_level": "info"}}"#, "observe.log_level"),
        (r#"{"load": {"threads": 2}}"#, "load.threads"),
        (r#"{"render": {"steer": "s.safetensors"}}"#, "render.steer"),
        (r#"{"output": {"stream": "levelled"}}"#, "output.stream"),
    ] {
        assert_eq!(render(doc, &mut pcm, &mut len, &mut rate), GGTTS_ERR_ARG, "{doc}");
        assert_eq!(key().as_deref(), Some(want), "{doc}: {}", last_error());
    }

    // A render that succeeds after a refusal clears it.
    assert_eq!(
        render(r#"{"render": {"seed_salt": "7"}}"#, &mut pcm, &mut len, &mut rate),
        GGTTS_OK,
        "{}",
        last_error()
    );
    assert_eq!(key(), None);
    unsafe { ggtts_pcm_free(pcm, len) };

    // Not about one option: a document that is not JSON (the whole document), a NULL text, and a missing model file.
    assert_eq!(render(r#"{"render": "#, &mut pcm, &mut len, &mut rate), GGTTS_ERR_ARG);
    assert_eq!(key(), None, "{}", last_error());
    let code = unsafe {
        ggtts_render_opts(
            h,
            ptr::null(),
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut pcm,
            &mut len,
            &mut rate,
        )
    };
    assert_eq!(code, GGTTS_ERR_ARG);
    assert_eq!(key(), None, "{}", last_error());
    let missing = std::env::temp_dir().join(format!("ggtts-ffi-no-such-{}.gguf", std::process::id()));
    let doc = json!({"load": {"engine": "qwen3-tts", "model": missing, "codec": missing, "offline": true}});
    let (code, _) = load_doc(&doc.to_string());
    assert_ne!(code, GGTTS_OK);
    assert_eq!(key(), None, "{code}: {}", last_error());

    // A key is per thread: another thread's refusal does not reach this one.
    let (code, _) = load_doc(r#"{"load": {"engine": "fake", "no_such_option": 1}}"#);
    assert_eq!(code, GGTTS_ERR_ARG);
    std::thread::spawn(|| {
        assert_eq!(key(), None, "a fresh thread has no key");
    })
    .join()
    .unwrap();
    assert_eq!(key().as_deref(), Some("load.no_such_option"));
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

/// The speakers a Qwen3-TTS talker has, as its unknown-speaker refusal lists them: a path that is not the
/// capability statement's, so the two are checked against each other.
fn speakers_from_the_refusal(load: &str) -> Vec<String> {
    let (code, _) = load_doc(&format!(r#"{{"load": {{{load}, "speaker": "no-such-speaker"}}}}"#));
    assert_eq!(code, GGTTS_ERR_ENGINE, "{}", last_error());
    let msg = last_error();
    let list = msg
        .split_once("speakers are: ")
        .unwrap_or_else(|| panic!("no speaker list in {msg:?}"))
        .1;
    let list = list.split(". Choose").next().unwrap();
    list.split(", ").map(str::to_string).collect()
}

/// AC2 on a real Qwen3-TTS talker (the tiny model in CI's fixture, or any): its capabilities as the engine reports
/// them, and its speakers, the same list its unknown-speaker refusal names.
#[test]
#[ignore = "needs GGTTS_TALKER and GGTTS_CODEC (a real Qwen3-TTS talker and codec) and a ggml build"]
fn a_qwen3_tts_models_capabilities_and_speakers() {
    let env = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("set {n}"));
    let load = format!(
        r#""engine": "qwen3-tts", "model": {}, "codec": {}, "device": "cpu", "offline": true"#,
        serde_json::to_string(&env("GGTTS_TALKER")).unwrap(),
        serde_json::to_string(&env("GGTTS_CODEC")).unwrap()
    );
    let speakers = speakers_from_the_refusal(&load);
    assert!(!speakers.is_empty());
    let (code, h) = load_doc(&format!(r#"{{"load": {{{load}}}}}"#));
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let (code, statement) = capabilities(h);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let statement = statement.unwrap();
    // Everything but the instruction is the engine's; whether this checkpoint takes an instruction is its own.
    let mut caps = Capabilities::of("qwen3-tts").unwrap();
    caps.instruction = statement["instruction"].as_bool().unwrap();
    caps.no_instruction_why = statement["no_instruction_why"].as_str().map(str::to_string);
    assert_eq!(caps.instruction, caps.no_instruction_why.is_none(), "{statement}");
    assert_eq!(statement, expected(&caps, &speakers));
    println!("qwen3-tts statement: {statement}");
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

/// AC2 on kitten-tts-2: no instruction, steering or speakers, CPU and CUDA, nothing to hooks.
#[test]
#[ignore = "needs GGTTS_KITTEN_LM, GGTTS_KITTEN_S3GEN and GGTTS_KITTEN_VOICE (a kitten-tts-2 model) and a ggml build"]
fn a_kitten_tts_2_models_capabilities() {
    let env = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("set {n}"));
    let doc = json!({"load": {
        "engine": "kitten-tts-2",
        "model": env("GGTTS_KITTEN_LM"),
        "codec": env("GGTTS_KITTEN_S3GEN"),
        "voice": env("GGTTS_KITTEN_VOICE"),
        "device": "cpu",
        "offline": true,
    }});
    let (code, h) = load_doc(&doc.to_string());
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let (code, statement) = capabilities(h);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let statement = statement.unwrap();
    assert_eq!(statement, expected(&Capabilities::of("kitten-tts-2").unwrap(), &[]));
    println!("kitten-tts-2 statement: {statement}");
    assert_eq!(ggtts_free(h), GGTTS_OK);
}
