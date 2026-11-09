//! Bridge ggml's process-global log callback to the CLI's structured log sink.
//!
//! Install only in a standalone CLI process. ggml's callback is global and replacing it in an embedding
//! application would override that application's own handler. Continuation fragments inherit their thread's last
//! known level; an unclassified continuation is conservatively treated as an error.
use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr};
use std::sync::OnceLock;

use ggtts_core::logging::{Level, Settings};
use ggtts_ggml_sys as ggml;

static SETTINGS: OnceLock<Settings> = OnceLock::new();
thread_local! { static LAST_LEVEL: Cell<Level> = const { Cell::new(Level::Error) }; }

fn map_level(level: ggml::ggml_log_level) -> Level {
    let mapped = match level {
        ggml::GGML_LOG_LEVEL_DEBUG => Level::Debug,
        ggml::GGML_LOG_LEVEL_INFO => Level::Info,
        ggml::GGML_LOG_LEVEL_WARN => Level::Warn,
        ggml::GGML_LOG_LEVEL_ERROR => Level::Error,
        ggml::GGML_LOG_LEVEL_CONT => LAST_LEVEL.with(Cell::get),
        _ => Level::Warn,
    };
    LAST_LEVEL.with(|last| last.set(mapped));
    mapped
}

unsafe extern "C" fn callback(level: ggml::ggml_log_level, text: *const c_char, _data: *mut c_void) {
    // Never unwind across the C boundary. ggml owns the NUL-terminated text for the duration of this callback.
    let _ = std::panic::catch_unwind(|| {
        if text.is_null() {
            return;
        }
        let Some(settings) = SETTINGS.get().copied() else {
            return;
        };
        let level = map_level(level);
        if settings.enabled(level) {
            let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
            settings.stderr(
                level,
                "ggml",
                "native",
                format_args!("{}", text.trim_end_matches(['\r', '\n'])),
            );
        }
    });
}

/// Set the process-wide handler once, before initializing any backend.
pub fn install(settings: Settings) -> Result<(), &'static str> {
    SETTINGS
        .set(settings)
        .map_err(|_| "native log handler already initialized")?;
    unsafe { ggml::ggml_log_set(Some(callback), std::ptr::null_mut()) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_preserves_the_previous_native_severity() {
        assert_eq!(map_level(ggml::GGML_LOG_LEVEL_INFO), Level::Info);
        assert_eq!(map_level(ggml::GGML_LOG_LEVEL_CONT), Level::Info);
        assert_eq!(map_level(ggml::GGML_LOG_LEVEL_ERROR), Level::Error);
        assert_eq!(map_level(ggml::GGML_LOG_LEVEL_CONT), Level::Error);
    }
}
