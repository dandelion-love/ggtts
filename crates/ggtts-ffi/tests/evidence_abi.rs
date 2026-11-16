use ggtts_ffi::*;
use std::{
    ffi::{c_void, CString},
    ptr,
};
#[derive(Default)]
struct Capture {
    data: Vec<f32>,
    chunks: usize,
    stop: bool,
}
extern "C" fn evidence(user: *mut c_void, ptr: *const StreamEvidence) -> i32 {
    let c = unsafe { &mut *(user as *mut Capture) };
    let p = unsafe { &*ptr };
    assert_eq!(p.chunk, c.chunks);
    assert_eq!(p.start, c.data.len());
    assert!(p.start <= p.speech_end && p.speech_end <= p.start + p.len && p.rate > 0 && p.stop <= 1);
    assert!(
        !std::str::from_utf8(unsafe { std::slice::from_raw_parts(p.text, p.text_len) })
            .unwrap()
            .is_empty()
    );
    c.chunks += 1;
    c.data
        .extend_from_slice(unsafe { std::slice::from_raw_parts(p.samples, p.len) });
    i32::from(c.stop)
}
extern "C" fn plain(user: *mut c_void, ptr: *const f32, len: usize, _: u32) -> i32 {
    unsafe { &mut *(user as *mut Capture) }
        .data
        .extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
    0
}
fn load() -> u64 {
    let mut h = 0;
    let name = CString::new("fake").unwrap();
    assert_eq!(
        unsafe { ggtts_load(name.as_ptr(), ptr::null(), ptr::null(), ptr::null(), 0, &mut h) },
        GGTTS_OK
    );
    h
}
#[test]
fn evidence_audio_matches_legacy_and_failure_is_visible() {
    let text = CString::new("The first sentence is here. The second sentence follows after it.").unwrap();
    let h = load();
    let mut c = Capture::default();
    assert_eq!(
        unsafe { ggtts_render_stream_evidence(h, text.as_ptr(), 7, Some(evidence), &mut c as *mut _ as *mut c_void) },
        GGTTS_OK
    );
    assert!(c.chunks > 0);
    let h2 = load();
    let mut old = Capture::default();
    assert_eq!(
        unsafe { ggtts_render_stream(h2, text.as_ptr(), 7, Some(plain), &mut old as *mut _ as *mut c_void) },
        GGTTS_OK
    );
    assert_eq!(c.data, old.data);
    let mut stopped = Capture {
        stop: true,
        ..Capture::default()
    };
    assert_eq!(
        unsafe {
            ggtts_render_document_evidence(
                h,
                text.as_ptr(),
                ptr::null(),
                Some(evidence),
                &mut stopped as *mut _ as *mut c_void,
            )
        },
        GGTTS_ERR_STOPPED
    );
    assert_eq!(
        unsafe { ggtts_render_stream_evidence(h, text.as_ptr(), 7, None, ptr::null_mut()) },
        GGTTS_ERR_ARG
    );
    assert_eq!(ggtts_free(h), GGTTS_OK);
    assert_eq!(ggtts_free(h2), GGTTS_OK);
}
