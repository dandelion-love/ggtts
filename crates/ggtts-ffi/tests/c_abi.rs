//! The C-ABI, driven through its `extern "C"` functions on the host with the fake engine: load, render, the stream's
//! order and its stop, every error a code and a message, a planted panic caught at the boundary, no use of a freed or
//! null handle, and calls back in from a stream callback or another thread that neither deadlock nor wait.

use std::ffi::{c_void, CString};
use std::ptr;

use ggtts_ffi::*;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn load(engine: &str, model: Option<&str>, codec: Option<&str>) -> (i32, u64) {
    let (e, m, k) = (c(engine), model.map(c), codec.map(c));
    let mut h = 0u64;
    let code = unsafe {
        ggtts_load(
            e.as_ptr(),
            m.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            k.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            ptr::null(),
            0,
            &mut h,
        )
    };
    (code, h)
}

fn render(h: u64, text: &str) -> (i32, Vec<f32>, u32) {
    let t = c(text);
    let (mut p, mut n, mut r) = (ptr::null_mut(), 0usize, 0u32);
    let code = unsafe { ggtts_render(h, t.as_ptr(), 7, &mut p, &mut n, &mut r) };
    let v = if p.is_null() {
        Vec::new()
    } else {
        let v = unsafe { std::slice::from_raw_parts(p, n) }.to_vec();
        unsafe { ggtts_pcm_free(p, n) };
        v
    };
    (code, v, r)
}

const TEXT: &str = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。";

#[test]
fn load_render_and_free_on_the_fake_engine() {
    let (code, h) = load("fake", None, None);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_ne!(h, 0);
    let (code, pcm, rate) = render(h, TEXT);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert!(
        pcm.len() > rate as usize,
        "at least a second of audio: {} at {rate}",
        pcm.len()
    );
    assert_eq!(last_error(), "", "a success clears the message");
    // A fresh engine, the same text and seed: the same samples. (One fake engine keeps per-text draw counts, so a
    // second render on it is a redraw, not a repeat.)
    let (_, fresh) = load("fake", None, None);
    assert_eq!(render(fresh, TEXT).1, pcm);
    assert_eq!(ggtts_free(fresh), GGTTS_OK);
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

struct Pieces {
    lens: Vec<usize>,
    total: usize,
    stop_after: Option<usize>,
}

extern "C" fn collect(user: *mut c_void, samples: *const f32, len: usize, rate: u32) -> i32 {
    let p = unsafe { &mut *(user as *mut Pieces) };
    assert!(!samples.is_null() && rate > 0);
    p.lens.push(len);
    p.total += len;
    match p.stop_after {
        Some(n) if p.lens.len() >= n => 1,
        _ => 0,
    }
}

#[test]
fn the_stream_hands_over_every_piece_in_order_and_can_be_stopped() {
    let (_, h) = load("fake", None, None);
    let t = c(TEXT);
    let mut p = Pieces {
        lens: Vec::new(),
        total: 0,
        stop_after: None,
    };
    let code = unsafe { ggtts_render_stream(h, t.as_ptr(), 7, Some(collect), &mut p as *mut Pieces as *mut c_void) };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert!(p.lens.len() >= 2, "one piece per chunk: {:?}", p.lens);
    // The pieces together are the read (before the level step, so the same length as the levelled render).
    let (_, pcm, _) = render(h, TEXT);
    assert_eq!(p.total, pcm.len());

    // A callback that asks to stop after the first piece stops the render there.
    let mut first = Pieces {
        lens: Vec::new(),
        total: 0,
        stop_after: Some(1),
    };
    let code = unsafe {
        ggtts_render_stream(
            h,
            t.as_ptr(),
            7,
            Some(collect),
            &mut first as *mut Pieces as *mut c_void,
        )
    };
    assert_eq!(code, GGTTS_ERR_STOPPED, "{}", last_error());
    assert_eq!(first.lens.len(), 1);
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

#[test]
fn every_failure_is_a_code_and_a_message() {
    // A missing model file.
    let (code, h) = load("qwen3-tts", Some("/no/such/model.gguf"), Some("/no/such/codec.gguf"));
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0));
    assert!(
        last_error().starts_with("model /no/such/model.gguf:"),
        "{}",
        last_error()
    );

    // A file that is not a GGUF.
    let bad = std::env::temp_dir().join(format!("ggtts-ffi-bad-{}.gguf", std::process::id()));
    std::fs::write(&bad, b"not a model").unwrap();
    let (code, _) = load("qwen3-tts", Some(bad.to_str().unwrap()), Some(bad.to_str().unwrap()));
    assert_eq!(code, GGTTS_ERR_ENGINE);
    assert!(last_error().ends_with("not a GGUF file"), "{}", last_error());
    let _ = std::fs::remove_file(bad);

    // An engine that does not exist, and a NULL argument.
    assert_eq!(load("nonesuch", None, None).0, GGTTS_ERR_ENGINE);
    assert!(last_error().contains("no engine named"), "{}", last_error());
    let mut h = 0;
    let code = unsafe { ggtts_load(ptr::null(), ptr::null(), ptr::null(), ptr::null(), 0, &mut h) };
    assert_eq!(code, GGTTS_ERR_ARG);
    assert_eq!(last_error(), "engine is NULL");

    // Empty text.
    let (_, h) = load("fake", None, None);
    assert_eq!(render(h, "  \n ").0, GGTTS_ERR_ARG);
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

#[test]
fn a_freed_or_null_handle_is_an_error_never_a_crash() {
    let (_, h) = load("fake", None, None);
    assert_eq!(ggtts_free(h), GGTTS_OK);
    // After free, with 0, and freed twice: each an error, and the process goes on.
    assert_eq!(render(h, TEXT).0, GGTTS_ERR_HANDLE);
    assert!(last_error().contains(&format!("handle {h}")), "{}", last_error());
    assert_eq!(render(0, TEXT).0, GGTTS_ERR_HANDLE);
    assert_eq!(ggtts_free(h), GGTTS_ERR_HANDLE);
    // Ids are not reused: a new engine never answers to the freed id.
    let (_, h2) = load("fake", None, None);
    assert_ne!(h2, h);
    assert_eq!(render(h, TEXT).0, GGTTS_ERR_HANDLE);
    assert_eq!(ggtts_free(h2), GGTTS_OK);
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

/// `ggtts_load_with` with `settings` NULL (`None`) or a struct of the given speaker and instruction.
fn load_with(engine: &str, settings: Option<(Option<&str>, Option<&str>)>) -> (i32, u64) {
    let e = c(engine);
    let strings = settings.map(|(s, i)| (s.map(c), i.map(c)));
    let ptr_of = |s: &Option<CString>| s.as_ref().map_or(ptr::null(), |s| s.as_ptr());
    let st = strings.as_ref().map(|(s, i)| GgttsSettings {
        speaker: ptr_of(s),
        instruction: ptr_of(i),
        ..unset()
    });
    let mut h = 0u64;
    let code = unsafe {
        ggtts_load_with(
            e.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            st.as_ref().map_or(ptr::null(), |s| s as *const GgttsSettings),
            &mut h,
        )
    };
    (code, h)
}

/// The header's lines with their endings and trailing space removed, so a checkout with CRLF endings (Windows) reads
/// as one with LF.
fn header_lines(header: &str) -> Vec<&str> {
    header.lines().map(str::trim_end).collect()
}

/// The value of `#define GGTTS_ABI_VERSION`, whatever the line endings.
fn header_version(header: &str) -> Option<u32> {
    header_lines(header).iter().find_map(|l| {
        let mut words = l.split_whitespace();
        match (words.next(), words.next(), words.next()) {
            (Some("#define"), Some("GGTTS_ABI_VERSION"), Some(v)) => v.parse().ok(),
            _ => None,
        }
    })
}

#[test]
fn the_header_reads_the_same_with_lf_and_crlf_endings() {
    let lf = "/* x */\n#define GGTTS_ABI_VERSION 2\ntypedef struct a {\n    size_t size;\n} a;\n";
    let crlf = lf.replace('\n', "\r\n");
    assert_eq!(header_version(lf), Some(2));
    assert_eq!(header_version(&crlf), Some(2), "a CRLF header");
    assert_eq!(header_lines(&crlf), header_lines(lf));
    assert_eq!(header_version("#define GGTTS_ABI_VERSIONS 2\n"), None);
}

#[test]
fn the_abi_version_is_3_in_the_library_and_the_header() {
    assert_eq!(ggtts_abi_version(), 3);
    assert_eq!(GGTTS_ABI_VERSION, 3);
    let header = include_str!("../include/ggtts.h");
    assert_eq!(header_version(header), Some(3), "the header's version");
    // The header's struct is the library's: version 2's three fields, size first, then version 3's.
    let joined = header_lines(header).join("\n");
    assert!(
        joined.contains(
            "typedef struct ggtts_settings {\n    size_t size;\n    const char *speaker;\n    const char *instruction;\n    \
             const char *device;\n    const char *cache_dir;\n    int32_t allow_download;\n} ggtts_settings;"
        ),
        "the header's ggtts_settings"
    );
    assert!(
        joined.contains("#define GGTTS_SETTINGS_INIT {sizeof(ggtts_settings), NULL, NULL, NULL, NULL, 0}"),
        "the header's GGTTS_SETTINGS_INIT"
    );
    let word = std::mem::size_of::<usize>();
    assert_eq!(GGTTS_SETTINGS_V2_SIZE, 3 * word, "a version-2 struct");
    assert_eq!(
        std::mem::size_of::<GgttsSettings>(),
        (5 * word + 4).next_multiple_of(word)
    );
}

/// A caller built against the version-2 header passes a struct of three fields. It loads as it did, and nothing past
/// its end is read (the struct here is exactly that long).
#[test]
fn a_version_2_struct_still_loads() {
    #[repr(C)]
    struct V2 {
        size: usize,
        speaker: *const std::ffi::c_char,
        instruction: *const std::ffi::c_char,
    }
    let e = c("fake");
    let (code, h) = load("fake", None, None);
    assert_eq!(code, GGTTS_OK);
    let today = render(h, TEXT).1;
    assert_eq!(ggtts_free(h), GGTTS_OK);
    let v2 = Box::new(V2 {
        size: std::mem::size_of::<V2>(),
        speaker: ptr::null(),
        instruction: ptr::null(),
    });
    assert_eq!(v2.size, GGTTS_SETTINGS_V2_SIZE);
    let mut h = 0u64;
    let code = unsafe {
        ggtts_load_with(
            e.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            &*v2 as *const V2 as *const GgttsSettings,
            &mut h,
        )
    };
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(render(h, TEXT).1, today);
    assert_eq!(ggtts_free(h), GGTTS_OK);
    // A speaker on another engine is still refused through a version-2 struct.
    let st = V2 {
        size: GGTTS_SETTINGS_V2_SIZE,
        speaker: c"hk_voice_777".as_ptr(),
        instruction: ptr::null(),
    };
    let code = unsafe {
        ggtts_load_with(
            e.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            &st as *const V2 as *const GgttsSettings,
            &mut h,
        )
    };
    assert_eq!(code, GGTTS_ERR_ARG);
}

/// `ggtts_load_with` with a version-3 struct of `engine` and `device`, and no paths.
fn load_on(engine: &str, device: &str) -> (i32, u64) {
    let (e, d) = (c(engine), c(device));
    let st = GgttsSettings {
        device: d.as_ptr(),
        ..unset()
    };
    let mut h = 0u64;
    let code = unsafe { ggtts_load_with(e.as_ptr(), ptr::null(), ptr::null(), ptr::null(), 0, &st, &mut h) };
    (code, h)
}

/// The device setting: `cpu` (or none) is what `ggtts_load` gives; an unknown device is an argument error; the fake
/// engine takes the CPU only; and a build without the CUDA backend refuses `cuda` before reading any file. Each
/// refusal writes no handle.
#[test]
fn the_device_is_cpu_unless_asked_and_a_missing_backend_is_refused() {
    let (code, h) = load_on("fake", "cpu");
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(
        backend_of(h),
        Ok("none".to_string()),
        "the fake engine has no compute backend"
    );
    assert_eq!(ggtts_free(h), GGTTS_OK);

    let (code, h) = load_on("fake", "gpu");
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
    assert_eq!(last_error(), "settings.device must be cpu or cuda, not \"gpu\"");

    let (code, h) = load_on("fake", "cuda");
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
    assert_eq!(last_error(), "the fake engine runs on the CPU only");

    // The library's engines and backends are ggtts-load's, whose features cargo may unify on from another crate in
    // the same build (`--workspace --features ggtts-cli/ggml`): ask the library, not this crate's own features.
    if !ggtts_load::cuda_built() {
        for engine in ["qwen3-tts", "kitten-tts-2"] {
            let (code, h) = load_on(engine, "cuda");
            assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0), "{engine}: {}", last_error());
            assert!(
                last_error().starts_with("this build has no CUDA backend (built without the cuda feature)"),
                "{}",
                last_error()
            );
        }
    }
}

/// With no model, a version-3 load of `qwen3-tts` looks for the default files, and with downloading off a missing one
/// is an error naming the file and its URL (a build with engines); a build without engines refuses before looking.
#[test]
fn no_model_resolves_the_default_files_and_never_downloads_unless_allowed() {
    let root = std::env::temp_dir().join(format!("ggtts-ffi-c-abi-offline-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (e, dir) = (c("qwen3-tts"), c(root.to_str().unwrap()));
    let st = GgttsSettings {
        cache_dir: dir.as_ptr(),
        ..unset()
    };
    let mut h = 0u64;
    let code = unsafe { ggtts_load_with(e.as_ptr(), ptr::null(), ptr::null(), ptr::null(), 0, &st, &mut h) };
    assert_eq!((code, h), (GGTTS_ERR_ENGINE, 0));
    let msg = last_error();
    // Whether the library has its engines is ggtts-load's feature, not this crate's (as for CUDA above).
    if ggtts_load::engines_built() {
        assert!(
            msg.contains("qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf")
                && msg.contains("https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF/resolve/")
                && msg.contains("allow_download"),
            "{msg}"
        );
    } else {
        assert_eq!(
            msg,
            "this build has no qwen3-tts engine (built without the ggml feature)"
        );
    }
    assert!(!root.join("models").exists(), "nothing was downloaded");
    let _ = std::fs::remove_dir_all(&root);

    // ggtts_load and NULL settings keep version 1's answer for no model: an argument error, nothing looked up.
    let (code, h) = load("qwen3-tts", None, None);
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
    assert_eq!(last_error(), "the qwen3-tts engine needs a model");
}

/// AC3: a caller that sets neither speaker nor instruction gets today's output, sample for sample, whether it calls
/// `ggtts_load`, or `ggtts_load_with` with NULL settings, NULL strings or empty strings (fresh engines, same seed).
#[test]
fn neither_setting_renders_exactly_as_ggtts_load() {
    let (code, h) = load("fake", None, None);
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    let today = render(h, TEXT).1;
    assert_eq!(ggtts_free(h), GGTTS_OK);
    assert!(!today.is_empty());
    for settings in [None, Some((None, None)), Some((Some(""), Some("")))] {
        let (code, h) = load_with("fake", settings);
        assert_eq!(code, GGTTS_OK, "{settings:?}: {}", last_error());
        let (code, pcm, _) = render(h, TEXT);
        assert_eq!(code, GGTTS_OK, "{settings:?}: {}", last_error());
        assert_eq!(pcm, today, "{settings:?}: the samples differ from ggtts_load's");
        assert_eq!(ggtts_free(h), GGTTS_OK);
    }
}

/// AC4 and the struct's own refusals: each a code and a message, no handle written, and the process goes on. A
/// setting on an engine that does not take it is refused, never ignored (the instruction with the library's message);
/// the Qwen3-TTS refusals of an unknown speaker and of an instruction on a Base checkpoint are in `qwen_settings.rs`.
#[test]
fn a_refused_setting_is_a_code_and_a_message_and_no_handle() {
    let (code, h) = load_with("fake", Some((Some("hk_voice_777"), None)));
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
    assert_eq!(
        last_error(),
        "a speaker is only supported by the qwen3-tts engine, not fake"
    );

    // Both settings on another engine are the same code: an argument that engine does not take.
    let (code, h) = load_with("fake", Some((None, Some("Speak slowly and softly."))));
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
    assert!(
        last_error().contains("the fake engine takes no style instruction"),
        "{}",
        last_error()
    );

    // A struct of another size (a caller built against another header) is refused, not misread.
    let e = c("fake");
    let mut h = 0u64;
    for size in [8, GGTTS_SETTINGS_V2_SIZE + 8, std::mem::size_of::<GgttsSettings>() + 8] {
        let wrong = GgttsSettings { size, ..unset() };
        let code = unsafe { ggtts_load_with(e.as_ptr(), ptr::null(), ptr::null(), ptr::null(), 0, &wrong, &mut h) };
        assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
        assert!(
            last_error().starts_with(&format!("settings.size is {size};")),
            "{}",
            last_error()
        );
    }

    // A speaker, a device or a cache directory that is not UTF-8.
    let bad = [0xffu8, 0xfe, 0];
    let bad = bad.as_ptr() as *const std::ffi::c_char;
    for (st, name) in [
        (
            GgttsSettings {
                speaker: bad,
                ..unset()
            },
            "speaker",
        ),
        (GgttsSettings { device: bad, ..unset() }, "device"),
        (
            GgttsSettings {
                cache_dir: bad,
                ..unset()
            },
            "cache_dir",
        ),
    ] {
        let code = unsafe { ggtts_load_with(e.as_ptr(), ptr::null(), ptr::null(), ptr::null(), 0, &st, &mut h) };
        assert_eq!((code, h), (GGTTS_ERR_ARG, 0));
        assert_eq!(last_error(), format!("settings.{name} is not UTF-8"));
    }

    // The engine still loads and renders after every refusal.
    let (code, h) = load_with("fake", None);
    assert_eq!(code, GGTTS_OK);
    assert_eq!(render(h, TEXT).0, GGTTS_OK);
    assert_eq!(ggtts_free(h), GGTTS_OK);
}

/// The new entry point keeps the boundary: a panic while loading, and one while rendering an engine it loaded, each
/// come back as `GGTTS_ERR_PANIC` with the message, and the process goes on.
#[cfg(feature = "plant")]
#[test]
fn a_panic_on_the_new_path_is_caught_at_the_boundary() {
    let (code, h) = load_with("panic-load", Some((None, None)));
    assert_eq!((code, h), (GGTTS_ERR_PANIC, 0));
    assert!(last_error().contains("planted panic in load"), "{}", last_error());

    let (code, h) = load_with("panic", Some((Some(""), Some(""))));
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(render(h, TEXT).0, GGTTS_ERR_PANIC);
    assert!(last_error().contains("planted panic in generate"), "{}", last_error());
    assert_eq!(render(h, TEXT).0, GGTTS_ERR_HANDLE);

    let (code, ok) = load_with("fake", None);
    assert_eq!(code, GGTTS_OK);
    assert_eq!(render(ok, TEXT).0, GGTTS_OK);
    assert_eq!(ggtts_free(ok), GGTTS_OK);
}

#[cfg(feature = "plant")]
#[test]
fn a_panic_inside_is_caught_at_the_boundary_and_the_process_goes_on() {
    let (code, h) = load("panic", None, None);
    assert_eq!(code, GGTTS_OK);
    let (code, pcm, _) = render(h, TEXT);
    assert_eq!(code, GGTTS_ERR_PANIC);
    assert!(pcm.is_empty());
    assert!(last_error().contains("planted panic in generate"), "{}", last_error());
    // The engine that panicked is gone: its id is unknown from then on.
    assert_eq!(render(h, TEXT).0, GGTTS_ERR_HANDLE);
    assert_eq!(ggtts_free(h), GGTTS_ERR_HANDLE);
    // The registry survived the panic: another engine still loads and renders.
    let (_, ok) = load("fake", None, None);
    assert_eq!(render(ok, TEXT).0, GGTTS_OK);
    assert_eq!(ggtts_free(ok), GGTTS_OK);
}

/// What a re-entering callback does on its first piece, and what it saw.
struct Reenter {
    own: u64,
    other: u64,
    action: &'static str,
    codes: Vec<i32>,
}

extern "C" fn reenter(user: *mut c_void, _samples: *const f32, _len: usize, _rate: u32) -> i32 {
    let r = unsafe { &mut *(user as *mut Reenter) };
    if r.codes.is_empty() {
        r.codes.push(match r.action {
            "render-own" => render(r.own, TEXT).0,
            "render-other" => render(r.other, TEXT).0,
            "free-own" => ggtts_free(r.own),
            _ => unreachable!(),
        });
    }
    0
}

fn stream_with(own: u64, other: u64, action: &'static str) -> (i32, Vec<i32>) {
    let t = c(TEXT);
    let mut r = Reenter {
        own,
        other,
        action,
        codes: Vec::new(),
    };
    let code = unsafe { ggtts_render_stream(own, t.as_ptr(), 7, Some(reenter), &mut r as *mut Reenter as *mut c_void) };
    (code, r.codes)
}

/// A stream callback may call back in: on its own engine it is told the engine is busy, never deadlocked; another
/// engine renders; and freeing its own engine lets the render finish, after which the id is gone.
#[test]
fn a_stream_callback_may_call_back_in_without_a_deadlock() {
    let (_, own) = load("fake", None, None);
    let (_, other) = load("fake", None, None);

    let (code, inner) = stream_with(own, other, "render-own");
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(inner, vec![GGTTS_ERR_BUSY]);

    let (code, inner) = stream_with(own, other, "render-other");
    assert_eq!(code, GGTTS_OK, "{}", last_error());
    assert_eq!(inner, vec![GGTTS_OK], "two engines render at once");

    let (code, inner) = stream_with(own, other, "free-own");
    assert_eq!(
        (code, inner),
        (GGTTS_OK, vec![GGTTS_OK]),
        "the render outlives the free"
    );
    assert_eq!(render(own, TEXT).0, GGTTS_ERR_HANDLE);
    assert_eq!(ggtts_free(other), GGTTS_OK);
}

/// Two channels the stream callback uses to hold a render open while another thread acts.
struct Hold {
    started: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
    pieces: usize,
}

extern "C" fn hold(user: *mut c_void, _samples: *const f32, _len: usize, _rate: u32) -> i32 {
    let h = unsafe { &mut *(user as *mut Hold) };
    if h.pieces == 0 {
        h.started.send(()).unwrap();
        h.release.recv().unwrap();
    }
    h.pieces += 1;
    0
}

/// Freeing an engine from another thread while it renders returns at once (an app's close() on its UI thread must not
/// wait out a background render), and the render it interrupted runs to the end.
#[test]
fn free_from_another_thread_never_waits_for_a_render() {
    let (_, h) = load("fake", None, None);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let renderer = std::thread::spawn(move || {
        let t = c(TEXT);
        let mut state = Hold {
            started: started_tx,
            release: release_rx,
            pieces: 0,
        };
        let code = unsafe { ggtts_render_stream(h, t.as_ptr(), 7, Some(hold), &mut state as *mut Hold as *mut c_void) };
        (code, state.pieces)
    });
    started_rx.recv().unwrap();
    // The render is inside its callback now. With the registry locked across a render, this call would block forever.
    assert_eq!(ggtts_free(h), GGTTS_OK);
    assert_eq!(render(h, TEXT).0, GGTTS_ERR_HANDLE);
    release_tx.send(()).unwrap();
    let (code, pieces) = renderer.join().unwrap();
    assert_eq!(code, GGTTS_OK);
    assert!(pieces >= 2, "the render ran to the end: {pieces} pieces");
}
