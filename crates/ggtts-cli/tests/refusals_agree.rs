//! AC3 of #277: for the same option and engine, a `ggtts-render` user and a C ABI caller are refused with the same
//! text, which carries the option's schema key.

use std::ffi::CString;
use std::process::Command;
use std::ptr;

use ggtts_ffi::{ggtts_load_with, last_error, GgttsSettings, GGTTS_ERR_ARG};

/// The message `ggtts-render` prints for a refused command line: its `error:` line.
fn render_refusal(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_ggtts-render"))
        .args(args)
        .args(["--text", "今日天氣好好。", "--out", "unused.flac"])
        .output()
        .unwrap();
    assert!(!out.status.success(), "{args:?} rendered");
    let stderr = String::from_utf8(out.stderr).unwrap();
    stderr
        .lines()
        .find_map(|l| l.strip_prefix("error: "))
        .unwrap_or_else(|| panic!("{args:?}: no error line in {stderr:?}"))
        .to_string()
}

/// The message a C ABI caller reads after `ggtts_load_with` refuses `engine` with a speaker or an instruction.
fn c_abi_refusal(engine: &str, speaker: Option<&str>, instruction: Option<&str>) -> String {
    let c = |s: &str| CString::new(s).unwrap();
    let (e, s, i) = (c(engine), speaker.map(c), instruction.map(c));
    let settings = GgttsSettings {
        size: std::mem::size_of::<GgttsSettings>(),
        speaker: s.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        instruction: i.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        device: ptr::null(),
        cache_dir: ptr::null(),
        allow_download: 0,
    };
    let mut h = 0u64;
    let code = unsafe { ggtts_load_with(e.as_ptr(), ptr::null(), ptr::null(), ptr::null(), 0, &settings, &mut h) };
    assert_eq!((code, h), (GGTTS_ERR_ARG, 0), "{engine}: {}", last_error());
    last_error()
}

#[test]
fn ggtts_render_and_the_c_abi_refuse_an_option_in_the_same_words() {
    for engine in ["fake", "kitten-tts-2"] {
        let speaker = render_refusal(&["--engine", engine, "--speaker", "hk_voice_777"]);
        assert_eq!(
            speaker,
            c_abi_refusal(engine, Some("hk_voice_777"), None),
            "{engine}: speaker"
        );
        assert!(speaker.starts_with("load.speaker: "), "{speaker}");
        let instruction = render_refusal(&["--engine", engine, "--instruction", "Speak slowly."]);
        assert_eq!(
            instruction,
            c_abi_refusal(engine, None, Some("Speak slowly.")),
            "{engine}: instruction"
        );
        assert!(instruction.starts_with("render.instruction: "), "{instruction}");
    }
}
