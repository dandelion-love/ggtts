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
