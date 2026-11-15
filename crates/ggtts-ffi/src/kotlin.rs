//! The Kotlin binding's options-document calls, in plain Rust: what `Native.loadOpts`, `Native.renderOpts` and
//! `Native.renderStreamOpts` do once JNI has turned their arguments into Rust values (`jni.rs` is only that turning,
//! and the throwing). Nothing here needs a JVM, so `cargo test` runs it.
//!
//! A Kotlin document is the C ABI's ([`crate::ggtts_load_opts`], [`crate::ggtts_render_opts`]), read and checked by the
//! same code, with what only Kotlin has on top:
//!
//! * **Its own refusals** ([`refusal`]): a device (the Android build is CPU only) and the trace (no file sink a phone
//!   user can read).
//! * **Its arguments beside the document**: `load`'s parameters, and a render's `seed`, laid into the document; one
//!   given both ways is refused, naming both ([`binding::refuse_given_twice`]).
//! * **Its seed**: a render with no seed and no `render.seed_salt` in the document or the load's defaults renders at
//!   [`binding::DEFAULT_SEED`], as `Ggtts.render` always has.

use std::collections::HashMap;
use std::path::PathBuf;

use ggtts_core::options::{Options, Refusal, Source};
use ggtts_load::binding;

use crate::{defaults, load_document, refused, render};

/// A code and a message.
pub use ggtts_load::binding::Fail;

/// Why the Kotlin binding refuses the option `key` beyond what every binding refuses ([`binding::check_document`]),
/// or `None`. The words are the conformance table's for the Kotlin front end.
pub fn refusal(key: &str) -> Option<&'static str> {
    match key {
        "load.device" => Some("the Android build is CPU only"),
        "observe.trace" | "observe.trace_level" => {
            Some("the trace is a desktop observer: refused on Android, where no file a phone user can read is written")
        }
        _ => None,
    }
}

/// Refuses an option the Kotlin binding does not take ([`refusal`]), naming it.
fn check(o: &Options) -> Result<(), Fail> {
    for key in o.marks.sources.keys() {
        if let Some(why) = refusal(key) {
            return Err(binding::refusal_err(Refusal {
                key: key.clone(),
                message: why.to_string(),
            }));
        }
    }
    Ok(())
}

/// `Ggtts.load`'s parameters beside a load document, each `None` when not given.
#[derive(Debug, Default, Clone)]
pub struct LoadArgs {
    pub engine: Option<String>,
    pub model: Option<String>,
    pub codec: Option<String>,
    pub voice: Option<String>,
    pub speaker: Option<String>,
    pub instruction: Option<String>,
    pub cache_dir: Option<String>,
    pub threads: Option<i32>,
    pub allow_download: Option<bool>,
}

/// Load the engine a load document names, with `Ggtts.load`'s parameters laid in: the handle, its render defaults
/// kept as [`crate::ggtts_load_opts`] keeps them.
pub fn load(json: &str, a: &LoadArgs) -> Result<u64, Fail> {
    let mut o = Options::from_json(json).map_err(refused)?;
    binding::check_document(&o, true)?;
    check(&o)?;
    let set = |s: &Option<String>| s.as_deref().is_some_and(|s| !s.is_empty());
    let given: Vec<(&str, &str)> = [
        ("engine", "load.engine", a.engine.is_some()),
        ("model", "load.model", a.model.is_some()),
        ("codec", "load.codec", a.codec.is_some()),
        ("voice", "load.voice", a.voice.is_some()),
        ("speaker", "load.speaker", set(&a.speaker)),
        ("instruction", "render.instruction", set(&a.instruction)),
        ("cacheDir", "load.cache_dir", a.cache_dir.is_some()),
        ("threads", "load.threads", a.threads.is_some()),
        ("allowDownload", "load.offline", a.allow_download.is_some()),
    ]
    .into_iter()
    .filter(|(_, _, given)| *given)
    .map(|(argument, key, _)| (argument, key))
    .collect();
    binding::refuse_given_twice(&o, &given)?;
    let mark = |o: &mut Options, key: &str| {
        o.marks.sources.insert(key.into(), Source::Document);
    };
    if let Some(e) = &a.engine {
        o.load.engine = e.clone();
        mark(&mut o, "load.engine");
    }
    for (given, key) in [
        (&a.model, "load.model"),
        (&a.codec, "load.codec"),
        (&a.voice, "load.voice"),
    ] {
        if let Some(p) = given {
            let path = Some(PathBuf::from(p));
            match key {
                "load.model" => o.load.model = path,
                "load.codec" => o.load.codec = path,
                _ => o.load.voice = path,
            }
            mark(&mut o, key);
        }
    }
    if set(&a.speaker) {
        o.load.speaker.clone_from(&a.speaker);
        mark(&mut o, "load.speaker");
    }
    if set(&a.instruction) {
        o.render.instruction.clone_from(&a.instruction);
        mark(&mut o, "render.instruction");
    }
    if let Some(dir) = &a.cache_dir {
        o.load.cache_dir = Some(PathBuf::from(dir));
        mark(&mut o, "load.cache_dir");
    }
    if let Some(t) = a.threads {
        o.load.threads = Some(t);
        mark(&mut o, "load.threads");
    }
    if let Some(allow) = a.allow_download {
        o.load.offline = !allow;
        mark(&mut o, "load.offline");
    }
    load_document(&o)
}

/// A render call's document ([`binding::render_request`]) with Kotlin's refusals and its default seed: the salt is
/// [`binding::DEFAULT_SEED`] when neither the call, the document nor the load's defaults give one.
fn document(handle: u64, json: Option<&str>, seed: Option<u64>, stream: bool) -> Result<Options, Fail> {
    let mut doc = binding::render_request(json, seed, stream)?;
    check(&doc)?;
    let salted = |o: &Options| o.marks.sources.contains_key("render.seed_salt");
    let defaults_salted = defaults()
        .get_or_insert_with(HashMap::new)
        .get(&handle)
        .is_some_and(salted);
    if !salted(&doc) && !defaults_salted {
        doc.render.seed_salt = Some(binding::DEFAULT_SEED.to_string());
        doc.marks.sources.insert("render.seed_salt".into(), Source::Document);
    }
    Ok(doc)
}

/// `Native.renderOpts`: the whole read of `text` with a render document, laid over the load's render defaults.
pub fn render_opts(handle: u64, text: &str, json: Option<&str>, seed: Option<u64>) -> Result<(Vec<f32>, u32), Fail> {
    let doc = document(handle, json, seed, false)?;
    render(handle, text, &doc, None)
}

/// `Native.renderStreamOpts`: each kept chunk's piece to `on_piece` before the level step, with a render document.
/// `Ok(false)` when `on_piece` stopped it.
pub fn render_stream_opts(
    handle: u64,
    text: &str,
    json: Option<&str>,
    seed: Option<u64>,
    on_piece: binding::PieceSink<'_>,
) -> Result<bool, Fail> {
    let doc = document(handle, json, seed, true)?;
    match render(handle, text, &doc, Some(on_piece)) {
        Ok(_) => Ok(true),
        Err((crate::GGTTS_ERR_STOPPED, _)) => Ok(false),
        Err(other) => Err(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ggtts_free, ggtts_load, ggtts_pcm_free, ggtts_render, GGTTS_ERR_ARG};
    use std::ffi::CString;

    const TEXT: &str = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。";

    fn keyed<T>(f: impl FnOnce() -> Result<T, Fail>) -> Result<T, (Fail, Option<String>)> {
        binding::guard_keyed(f)
    }

    fn fake_doc(json: &str) -> u64 {
        load(json, &LoadArgs::default()).unwrap()
    }

    /// `ggtts_render` on a fresh `ggtts_load("fake")`: the samples the old Kotlin `render` gives.
    fn struct_render(seed: u64) -> (Vec<f32>, u32) {
        let (e, t) = (CString::new("fake").unwrap(), CString::new(TEXT).unwrap());
        let mut h = 0u64;
        let null = std::ptr::null();
        assert_eq!(unsafe { ggtts_load(e.as_ptr(), null, null, null, 0, &mut h) }, 0);
        let (mut p, mut n, mut r) = (std::ptr::null_mut(), 0usize, 0u32);
        assert_eq!(unsafe { ggtts_render(h, t.as_ptr(), seed, &mut p, &mut n, &mut r) }, 0);
        let s = unsafe { std::slice::from_raw_parts(p, n) }.to_vec();
        unsafe { ggtts_pcm_free(p, n) };
        ggtts_free(h);
        (s, r)
    }

    /// An empty document renders as the struct render at the default seed, and the seed argument is the salt.
    #[test]
    fn an_empty_document_is_the_struct_render() {
        let h = fake_doc(r#"{"load": {"engine": "fake"}}"#);
        assert_eq!(
            render_opts(h, TEXT, None, None).unwrap(),
            struct_render(binding::DEFAULT_SEED)
        );
        ggtts_free(h);
        let h = fake_doc(r#"{"load": {"engine": "fake"}}"#);
        assert_eq!(render_opts(h, TEXT, Some("{}"), Some(7)).unwrap(), struct_render(7));
        ggtts_free(h);
        let h = fake_doc(r#"{"load": {"engine": "fake"}}"#);
        let (mut a, mut b) = (Vec::new(), 0usize);
        assert!(render_stream_opts(h, TEXT, None, Some(7), &mut |s, _| {
            a.extend_from_slice(s);
            b += 1;
            true
        })
        .unwrap());
        assert!(b >= 2 && a.len() == struct_render(7).0.len());
        ggtts_free(h);
    }

    /// A document's instruction reaches the engine: the fake engine's capabilities refuse it in their own words, keyed.
    #[test]
    fn an_instruction_in_a_document_reaches_the_engine() {
        let h = fake_doc(r#"{"load": {"engine": "fake"}}"#);
        let caps = ggtts_core::capabilities::Capabilities::of("fake").unwrap();
        let ((code, msg), key) =
            keyed(|| render_opts(h, TEXT, Some(r#"{"render": {"instruction": "Speak slowly."}}"#), None)).unwrap_err();
        assert_eq!(code, GGTTS_ERR_ARG);
        assert_eq!(msg, caps.refusal("render.instruction", None).to_string());
        assert_eq!(key.as_deref(), Some("render.instruction"));
        ggtts_free(h);
    }

    /// Steering on Kitten-TTS 2 is refused before any file is read, with the key `render.steer`.
    #[test]
    fn steering_on_kitten_is_refused_by_its_key() {
        let doc =
            r#"{"load": {"engine": "kitten-tts-2", "model": "/no/such.gguf"}, "render": {"steer": "s.safetensors"}}"#;
        let ((code, msg), key) = keyed(|| load(doc, &LoadArgs::default())).unwrap_err();
        assert_eq!((code, key.as_deref()), (GGTTS_ERR_ARG, Some("render.steer")), "{msg}");
    }

    /// An unknown key, malformed JSON, a load key in a render document, a device, the trace, and a value given twice
    /// are each refused, keyed where there is one option to name.
    #[test]
    fn refusals_name_their_key() {
        let h = fake_doc(r#"{"load": {"engine": "fake"}}"#);
        for (doc, want) in [
            (r#"{"render": {"nonesuch": 1}}"#, Some("render.nonesuch")),
            ("{", None),
            (r#"{"load": {"threads": 2}}"#, Some("load.threads")),
            (r#"{"observe": {"trace": "t.jsonl"}}"#, Some("observe.trace")),
            (r#"{"output": {"stream": "raw"}}"#, Some("output.stream")),
        ] {
            let ((code, msg), key) = keyed(|| render_opts(h, TEXT, Some(doc), None)).unwrap_err();
            assert_eq!((code, key.as_deref()), (GGTTS_ERR_ARG, want), "{doc}: {msg}");
        }
        let ((_, msg), key) =
            keyed(|| render_opts(h, TEXT, Some(r#"{"render": {"seed_salt": "3"}}"#), Some(3))).unwrap_err();
        assert_eq!(key.as_deref(), Some("render.seed_salt"));
        assert!(msg.contains("the argument seed"), "{msg}");
        ggtts_free(h);
        for (doc, args, want) in [
            (
                r#"{"load": {"engine": "fake", "device": "cpu"}}"#,
                LoadArgs::default(),
                "load.device",
            ),
            (
                r#"{"load": {"engine": "fake", "bogus": 1}}"#,
                LoadArgs::default(),
                "load.bogus",
            ),
            (
                r#"{"load": {"engine": "fake", "threads": 2}}"#,
                LoadArgs {
                    threads: Some(2),
                    ..Default::default()
                },
                "load.threads",
            ),
            (
                r#"{"load": {"engine": "fake"}}"#,
                LoadArgs {
                    engine: Some("fake".into()),
                    ..Default::default()
                },
                "load.engine",
            ),
        ] {
            let ((code, msg), key) = keyed(|| load(doc, &args)).unwrap_err();
            assert_eq!((code, key.as_deref()), (GGTTS_ERR_ARG, Some(want)), "{doc}: {msg}");
        }
    }

    /// The load's render defaults reach every render, and the arguments laid into the load document load the engine.
    #[test]
    fn load_arguments_and_render_defaults_are_carried() {
        let h = load(
            r#"{"render": {"edge_s": 1.5}}"#,
            &LoadArgs {
                engine: Some("fake".into()),
                threads: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let edged = render_opts(h, TEXT, None, Some(7)).unwrap();
        ggtts_free(h);
        assert_ne!(edged.0.len(), struct_render(7).0.len());
    }
}
