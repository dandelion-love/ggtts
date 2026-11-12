//! The JNI entry points behind `ggtts.Native` (the Kotlin class `Ggtts` calls them). Each goes through the C-ABI
//! functions, so there is one implementation; a non-zero status becomes a thrown `ggtts.GgttsException(code,
//! message)`, never a crash.
//!
//! Two rules hold throughout. **No JNI call is made with an exception pending** beyond the ones JNI allows then
//! (`exception_check`, `delete_local_ref`): after a failed JNI call the function returns at once and lets the pending
//! exception reach Kotlin, and [`throw`] never throws over one. **No panic leaves a `Java_*` function or the stream
//! callback:** each body runs inside `catch_unwind`, and a panic becomes a `GgttsException` with `GGTTS_ERR_PANIC`.

use std::ffi::{c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use jni::objects::{JClass, JFloatArray, JIntArray, JObject, JString, JThrowable, JValue};
use jni::sys::{jboolean, jint, jlong};
use jni::JNIEnv;

use crate::{
    ggtts_free, ggtts_load, ggtts_load_with, ggtts_pcm_free, ggtts_render, ggtts_render_stream, last_error,
    GgttsSettings, GGTTS_ERR_ARG, GGTTS_ERR_PANIC,
};

/// Whether an exception is pending. A failed check is read as pending, so nothing is thrown over an unknown state.
fn pending(env: &mut JNIEnv) -> bool {
    env.exception_check().unwrap_or(true)
}

/// Throw `ggtts.GgttsException(code, message)`, unless an exception is already pending: that one goes to Kotlin as it
/// is. Building the exception can itself fail (out of memory, a missing class), and then whatever that left pending is
/// thrown instead.
fn throw(env: &mut JNIEnv, code: i32, msg: &str) {
    if pending(env) {
        return;
    }
    let built = env.new_string(msg).and_then(|m| {
        env.new_object(
            "ggtts/GgttsException",
            "(ILjava/lang/String;)V",
            &[JValue::Int(code), JValue::Object(&m)],
        )
    });
    match built {
        Ok(obj) => {
            let _ = env.throw(JThrowable::from(obj));
        }
        Err(_) if !pending(env) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("ggtts error {code}: {msg}"));
        }
        Err(_) => {}
    }
}

/// Run a `Java_*` body, turning a panic into a thrown `GgttsException` and `fallback`.
fn jni_guard<'l, T>(env: &mut JNIEnv<'l>, fallback: impl FnOnce() -> T, f: impl FnOnce(&mut JNIEnv<'l>) -> T) -> T {
    match catch_unwind(AssertUnwindSafe(|| f(&mut *env))) {
        Ok(v) => v,
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".into());
            throw(env, GGTTS_ERR_PANIC, &format!("panic inside ggtts: {msg}"));
            fallback()
        }
    }
}

/// A Java string as a C string; `None` for a Java null. An unusable string throws and returns `Err`.
fn cstr(env: &mut JNIEnv, s: &JString, name: &str) -> Result<Option<CString>, ()> {
    if s.is_null() {
        return Ok(None);
    }
    let text: String = match env.get_string(s) {
        Ok(t) => t.into(),
        Err(_) => {
            throw(env, GGTTS_ERR_ARG, &format!("{name} is not a readable string"));
            return Err(());
        }
    };
    match CString::new(text) {
        Ok(c) => Ok(Some(c)),
        Err(_) => {
            throw(env, GGTTS_ERR_ARG, &format!("{name} contains a NUL character"));
            Err(())
        }
    }
}

fn or_null(c: &Option<CString>) -> *const std::ffi::c_char {
    c.as_ref().map_or(ptr::null(), |c| c.as_ptr())
}

#[no_mangle]
pub extern "system" fn Java_ggtts_Native_load<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    engine: JString<'l>,
    model: JString<'l>,
    codec: JString<'l>,
    voice: JString<'l>,
    threads: jint,
) -> jlong {
    jni_guard(
        &mut env,
        || 0,
        |env| {
            let Ok(engine) = cstr(env, &engine, "engine") else {
                return 0;
            };
            let Ok(model) = cstr(env, &model, "model") else {
                return 0;
            };
            let Ok(codec) = cstr(env, &codec, "codec") else {
                return 0;
            };
            let Ok(voice) = cstr(env, &voice, "voice") else {
                return 0;
            };
            let mut handle = 0u64;
            let code = unsafe {
                ggtts_load(
                    or_null(&engine),
                    or_null(&model),
                    or_null(&codec),
                    or_null(&voice),
                    threads,
                    &mut handle,
                )
            };
            if code != 0 {
                throw(env, code, &last_error());
                return 0;
            }
            handle as jlong
        },
    )
}

/// `Native.loadWith`: `load` with a Qwen3-TTS speaker and style instruction, each nullable, through
/// [`ggtts_load_with`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_ggtts_Native_loadWith<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    engine: JString<'l>,
    model: JString<'l>,
    codec: JString<'l>,
    voice: JString<'l>,
    threads: jint,
    speaker: JString<'l>,
    instruction: JString<'l>,
) -> jlong {
    jni_guard(
        &mut env,
        || 0,
        |env| {
            let Ok(engine) = cstr(env, &engine, "engine") else {
                return 0;
            };
            let Ok(model) = cstr(env, &model, "model") else {
                return 0;
            };
            let Ok(codec) = cstr(env, &codec, "codec") else {
                return 0;
            };
            let Ok(voice) = cstr(env, &voice, "voice") else {
                return 0;
            };
            let Ok(speaker) = cstr(env, &speaker, "speaker") else {
                return 0;
            };
            let Ok(instruction) = cstr(env, &instruction, "instruction") else {
                return 0;
            };
            let settings = GgttsSettings {
                size: std::mem::size_of::<GgttsSettings>(),
                speaker: or_null(&speaker),
                instruction: or_null(&instruction),
            };
            let mut handle = 0u64;
            let code = unsafe {
                ggtts_load_with(
                    or_null(&engine),
                    or_null(&model),
                    or_null(&codec),
                    or_null(&voice),
                    threads,
                    &settings,
                    &mut handle,
                )
            };
            if code != 0 {
                throw(env, code, &last_error());
                return 0;
            }
            handle as jlong
        },
    )
}

#[no_mangle]
pub extern "system" fn Java_ggtts_Native_render<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    text: JString<'l>,
    seed: jlong,
    rate_out: JIntArray<'l>,
) -> JFloatArray<'l> {
    let null = || JFloatArray::from(JObject::null());
    jni_guard(&mut env, null, |env| {
        let Ok(text) = cstr(env, &text, "text") else {
            return null();
        };
        let (mut pcm, mut len, mut rate) = (ptr::null_mut(), 0usize, 0u32);
        let code = unsafe {
            ggtts_render(
                handle as u64,
                or_null(&text),
                seed as u64,
                &mut pcm,
                &mut len,
                &mut rate,
            )
        };
        if code != 0 {
            throw(env, code, &last_error());
            return null();
        }
        let samples = unsafe { std::slice::from_raw_parts(pcm, len) };
        let out = env
            .new_float_array(len as jint)
            .and_then(|a| env.set_float_array_region(&a, 0, samples).map(|_| a));
        unsafe { ggtts_pcm_free(pcm, len) };
        // A failed JNI call leaves its exception pending (out of memory, or a rate array too short): it goes to
        // Kotlin as it is, and no further JNI call is made.
        let Ok(out) = out else {
            throw(env, GGTTS_ERR_ARG, "could not allocate the sample array");
            return null();
        };
        if !rate_out.is_null() && env.set_int_array_region(&rate_out, 0, &[rate as jint]).is_err() {
            throw(
                env,
                GGTTS_ERR_ARG,
                "could not write the rate (rateOut must have room for one Int)",
            );
            return null();
        }
        out
    })
}

/// What the stream callback needs: the JNI environment and the listener.
struct Stream<'a, 'l> {
    env: &'a mut JNIEnv<'l>,
    listener: &'a JObject<'l>,
}

/// Hand one piece to the listener. Returns non-zero to stop: the listener returned false or threw, a JNI call failed
/// (its exception stays pending for `renderStream` to let through), or this panicked.
extern "C" fn on_piece(user: *mut c_void, samples: *const f32, len: usize, rate: u32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *(user as *mut Stream) };
        let piece = unsafe { std::slice::from_raw_parts(samples, len) };
        let arr = match s
            .env
            .new_float_array(len as jint)
            .and_then(|a| s.env.set_float_array_region(&a, 0, piece).map(|_| a))
        {
            Ok(a) => a,
            Err(_) => return 1,
        };
        let go_on = s
            .env
            .call_method(
                s.listener,
                "onPiece",
                "([FI)Z",
                &[JValue::Object(&arr), JValue::Int(rate as jint)],
            )
            .and_then(|v| v.z());
        // Deleting a local reference is allowed with an exception pending.
        let _ = s.env.delete_local_ref(arr);
        match go_on {
            Ok(true) if !pending(s.env) => 0,
            _ => 1,
        }
    }))
    .unwrap_or(1)
}

#[no_mangle]
pub extern "system" fn Java_ggtts_Native_renderStream<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    text: JString<'l>,
    seed: jlong,
    listener: JObject<'l>,
) -> jboolean {
    jni_guard(
        &mut env,
        || 0,
        |env| {
            let Ok(text) = cstr(env, &text, "text") else {
                return 0;
            };
            if listener.is_null() {
                throw(env, GGTTS_ERR_ARG, "listener is null");
                return 0;
            }
            let code = {
                let mut stream = Stream {
                    env: &mut *env,
                    listener: &listener,
                };
                unsafe {
                    ggtts_render_stream(
                        handle as u64,
                        or_null(&text),
                        seed as u64,
                        Some(on_piece),
                        &mut stream as *mut Stream as *mut c_void,
                    )
                }
            };
            match code {
                0 => 1,
                // Stopped by the listener: returning false, or throwing, whose exception is still pending and goes to
                // Kotlin as it is.
                crate::GGTTS_ERR_STOPPED => 0,
                // `throw` lets an exception a failed JNI call left pending through instead.
                _ => {
                    throw(env, code, &last_error());
                    0
                }
            }
        },
    )
}

#[no_mangle]
pub extern "system" fn Java_ggtts_Native_free<'l>(mut env: JNIEnv<'l>, _class: JClass<'l>, handle: jlong) {
    jni_guard(
        &mut env,
        || (),
        |env| {
            let code = ggtts_free(handle as u64);
            if code != 0 {
                throw(env, code, &last_error());
            }
        },
    )
}
