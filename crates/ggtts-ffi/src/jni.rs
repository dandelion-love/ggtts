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
    ggtts_free, ggtts_load, ggtts_load_with, ggtts_pcm_free, ggtts_render, ggtts_render_stream, kotlin, last_error,
    last_error_key, GgttsSettings, GGTTS_ERR_ARG, GGTTS_ERR_PANIC, GGTTS_SETTINGS_V2_SIZE,
};

/// Whether an exception is pending. A failed check is read as pending, so nothing is thrown over an unknown state.
fn pending(env: &mut JNIEnv) -> bool {
    env.exception_check().unwrap_or(true)
}

/// Throw `ggtts.GgttsException(code, message)`, with no refused key.
fn throw(env: &mut JNIEnv, code: i32, msg: &str) {
    throw_keyed(env, code, msg, None);
}

/// Throw the error the last C-ABI call on this thread recorded: its message and the key of the option it refused.
fn throw_last(env: &mut JNIEnv, code: i32) {
    throw_keyed(env, code, &last_error(), last_error_key().as_deref());
}

/// Throw `ggtts.GgttsException(code, message, key)`, `key` the schema key of the option refused or null, unless an
/// exception is already pending: that one goes to Kotlin as it is. Building the exception can itself fail (out of
/// memory, a missing class), and then whatever that left pending is thrown instead.
fn throw_keyed(env: &mut JNIEnv, code: i32, msg: &str, key: Option<&str>) {
    if pending(env) {
        return;
    }
    let built = env.new_string(msg).and_then(|m| {
        let k = match key {
            Some(k) => env.new_string(k)?,
            None => JString::from(JObject::null()),
        };
        env.new_object(
            "ggtts/GgttsException",
            "(ILjava/lang/String;Ljava/lang/String;)V",
            &[JValue::Int(code), JValue::Object(&m), JValue::Object(&k)],
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
                throw_last(env, code);
                return 0;
            }
            handle as jlong
        },
    )
}

/// The Java strings of one `ggtts_load_with` call; `cache_dir` and `allow_download` are `None` for a version-2 call.
struct LoadArgs<'a, 'l> {
    engine: &'a JString<'l>,
    model: &'a JString<'l>,
    codec: &'a JString<'l>,
    voice: &'a JString<'l>,
    threads: jint,
    speaker: &'a JString<'l>,
    instruction: &'a JString<'l>,
    models: Option<(&'a JString<'l>, bool)>,
}

/// Convert `a` and call [`ggtts_load_with`]: with a version-2 struct when `a.models` is `None`, so that call loads
/// exactly as it did in version 2, else with the whole struct (the device stays unset: the CPU). Returns the handle,
/// or 0 with an exception thrown.
fn load_with(env: &mut JNIEnv, a: LoadArgs) -> jlong {
    let strings = [
        (a.engine, "engine"),
        (a.model, "model"),
        (a.codec, "codec"),
        (a.voice, "voice"),
        (a.speaker, "speaker"),
        (a.instruction, "instruction"),
    ];
    let mut c = Vec::with_capacity(strings.len() + 1);
    for (s, name) in strings {
        let Ok(s) = cstr(env, s, name) else {
            return 0;
        };
        c.push(s);
    }
    let Ok(cache_dir) = a.models.map_or(Ok(None), |(dir, _)| cstr(env, dir, "cacheDir")) else {
        return 0;
    };
    let settings = GgttsSettings {
        size: if a.models.is_some() {
            std::mem::size_of::<GgttsSettings>()
        } else {
            GGTTS_SETTINGS_V2_SIZE
        },
        speaker: or_null(&c[4]),
        instruction: or_null(&c[5]),
        device: ptr::null(),
        cache_dir: or_null(&cache_dir),
        allow_download: a.models.is_some_and(|(_, allow)| allow) as i32,
    };
    let mut handle = 0u64;
    let code = unsafe {
        ggtts_load_with(
            or_null(&c[0]),
            or_null(&c[1]),
            or_null(&c[2]),
            or_null(&c[3]),
            a.threads,
            &settings,
            &mut handle,
        )
    };
    if code != 0 {
        throw_last(env, code);
        return 0;
    }
    handle as jlong
}

/// `Native.loadWith`: `load` with a Qwen3-TTS speaker and style instruction, each nullable, through
/// [`ggtts_load_with`] with a version-2 struct, so it loads as it did when it was added: the paths given only.
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
            load_with(
                env,
                LoadArgs {
                    engine: &engine,
                    model: &model,
                    codec: &codec,
                    voice: &voice,
                    threads,
                    speaker: &speaker,
                    instruction: &instruction,
                    models: None,
                },
            )
        },
    )
}

/// `Native.loadWithModels`: `loadWith` that also resolves the default model files (C-ABI version 3): a null model or
/// codec of `qwen3-tts` is the pinned default from the cache at `cacheDir` (null: ggtts-render's), downloaded only
/// when `allowDownload` is true; with no model and no speaker, the default model's default speaker.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_ggtts_Native_loadWithModels<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    engine: JString<'l>,
    model: JString<'l>,
    codec: JString<'l>,
    voice: JString<'l>,
    threads: jint,
    speaker: JString<'l>,
    instruction: JString<'l>,
    cache_dir: JString<'l>,
    allow_download: jboolean,
) -> jlong {
    jni_guard(
        &mut env,
        || 0,
        |env| {
            load_with(
                env,
                LoadArgs {
                    engine: &engine,
                    model: &model,
                    codec: &codec,
                    voice: &voice,
                    threads,
                    speaker: &speaker,
                    instruction: &instruction,
                    models: Some((&cache_dir, allow_download != 0)),
                },
            )
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
            throw_last(env, code);
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
                    throw_last(env, code);
                    0
                }
            }
        },
    )
}

/// A Java string as a Rust string; `None` for a Java null. An unusable string throws and returns `Err`.
fn string(env: &mut JNIEnv, s: &JString, name: &str) -> Result<Option<String>, ()> {
    if s.is_null() {
        return Ok(None);
    }
    match env.get_string(s) {
        Ok(t) => Ok(Some(t.into())),
        Err(_) => {
            throw(env, GGTTS_ERR_ARG, &format!("{name} is not a readable string"));
            Err(())
        }
    }
}

/// `Native.loadOpts`: `Ggtts.load` with an options document ([`kotlin::load`]). `threads` 0 or less and
/// `allowDownload` -1 are not given; a null string is not given.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_ggtts_Native_loadOpts<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    options_json: JString<'l>,
    engine: JString<'l>,
    model: JString<'l>,
    codec: JString<'l>,
    voice: JString<'l>,
    threads: jint,
    speaker: JString<'l>,
    instruction: JString<'l>,
    cache_dir: JString<'l>,
    allow_download: jint,
) -> jlong {
    jni_guard(
        &mut env,
        || 0,
        |env| {
            let mut s = Vec::new();
            for (j, name) in [
                (&options_json, "optionsJson"),
                (&engine, "engine"),
                (&model, "model"),
                (&codec, "codec"),
                (&voice, "voice"),
                (&speaker, "speaker"),
                (&instruction, "instruction"),
                (&cache_dir, "cacheDir"),
            ] {
                let Ok(v) = string(env, j, name) else {
                    return 0;
                };
                s.push(v);
            }
            let mut s = s.into_iter();
            let Some(json) = s.next().flatten() else {
                throw(env, GGTTS_ERR_ARG, "optionsJson is null");
                return 0;
            };
            let mut next = || s.next().flatten();
            let args = kotlin::LoadArgs {
                engine: next(),
                model: next(),
                codec: next(),
                voice: next(),
                speaker: next(),
                instruction: next(),
                cache_dir: next(),
                threads: (threads > 0).then_some(threads),
                allow_download: (allow_download >= 0).then_some(allow_download != 0),
            };
            let mut handle = 0u64;
            let code = crate::guard(|| {
                handle = kotlin::load(&json, &args)?;
                Ok(())
            });
            if code != 0 {
                throw_last(env, code);
                return 0;
            }
            handle as jlong
        },
    )
}

/// The seed a Kotlin document render passes: `None` when `seedGiven` is false.
fn given_seed(seed: jlong, seed_given: jboolean) -> Option<u64> {
    (seed_given != 0).then_some(seed as u64)
}

/// `Native.renderOpts`: the whole read of `text` with a render document ([`kotlin::render_opts`]); a null document is
/// the empty one.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_ggtts_Native_renderOpts<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    text: JString<'l>,
    options_json: JString<'l>,
    seed: jlong,
    seed_given: jboolean,
    rate_out: JIntArray<'l>,
) -> JFloatArray<'l> {
    let null = || JFloatArray::from(JObject::null());
    jni_guard(&mut env, null, |env| {
        let Ok(Some(text)) = string(env, &text, "text") else {
            throw(env, GGTTS_ERR_ARG, "text is null");
            return null();
        };
        let Ok(json) = string(env, &options_json, "optionsJson") else {
            return null();
        };
        let mut read = None;
        let code = crate::guard(|| {
            read = Some(kotlin::render_opts(
                handle as u64,
                &text,
                json.as_deref(),
                given_seed(seed, seed_given),
            )?);
            Ok(())
        });
        let Some((samples, rate)) = read.filter(|_| code == 0) else {
            throw_last(env, code);
            return null();
        };
        let out = env
            .new_float_array(samples.len() as jint)
            .and_then(|a| env.set_float_array_region(&a, 0, &samples).map(|_| a));
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

/// `Native.renderStreamOpts`: each kept chunk's piece to `listener` with a render document, levelled unless the
/// document asks for the pieces before the level step ([`kotlin::render_stream_opts`]). Returns false when the listener stopped it (returning false, or throwing, whose
/// exception goes to Kotlin as it is).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_ggtts_Native_renderStreamOpts<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
    text: JString<'l>,
    options_json: JString<'l>,
    seed: jlong,
    seed_given: jboolean,
    listener: JObject<'l>,
) -> jboolean {
    jni_guard(
        &mut env,
        || 0,
        |env| {
            let Ok(Some(text)) = string(env, &text, "text") else {
                throw(env, GGTTS_ERR_ARG, "text is null");
                return 0;
            };
            let Ok(json) = string(env, &options_json, "optionsJson") else {
                return 0;
            };
            if listener.is_null() {
                throw(env, GGTTS_ERR_ARG, "listener is null");
                return 0;
            }
            let mut finished = false;
            let code = {
                let mut stream = Stream {
                    env: &mut *env,
                    listener: &listener,
                };
                let user = &mut stream as *mut Stream as *mut c_void;
                let mut piece = |samples: &[f32], rate: u32| on_piece(user, samples.as_ptr(), samples.len(), rate) == 0;
                crate::guard(|| {
                    finished = kotlin::render_stream_opts(
                        handle as u64,
                        &text,
                        json.as_deref(),
                        given_seed(seed, seed_given),
                        &mut piece,
                    )?;
                    Ok(())
                })
            };
            if code != 0 {
                throw_last(env, code);
                return 0;
            }
            // Stopped by the listener: returning false, or throwing, whose exception is still pending and goes to
            // Kotlin as it is.
            finished as jboolean
        },
    )
}

/// `Native.optionsSchema`: the options schema as JSON ([`kotlin::options_schema`]).
#[no_mangle]
pub extern "system" fn Java_ggtts_Native_optionsSchema<'l>(mut env: JNIEnv<'l>, _class: JClass<'l>) -> JString<'l> {
    let null = || JString::from(JObject::null());
    jni_guard(&mut env, null, |env| match env.new_string(kotlin::options_schema()) {
        Ok(s) => s,
        Err(_) => {
            throw(env, GGTTS_ERR_ARG, "could not allocate the schema string");
            null()
        }
    })
}

/// `Native.capabilities`: the engine's capability statement as JSON ([`kotlin::capabilities`]).
#[no_mangle]
pub extern "system" fn Java_ggtts_Native_capabilities<'l>(
    mut env: JNIEnv<'l>,
    _class: JClass<'l>,
    handle: jlong,
) -> JString<'l> {
    let null = || JString::from(JObject::null());
    jni_guard(&mut env, null, |env| {
        let mut text = None;
        let code = crate::guard(|| {
            text = Some(kotlin::capabilities(handle as u64)?);
            Ok(())
        });
        let Some(text) = text.filter(|_| code == 0) else {
            throw_last(env, code);
            return null();
        };
        match env.new_string(text) {
            Ok(s) => s,
            Err(_) => {
                throw(env, GGTTS_ERR_ARG, "could not allocate the statement string");
                null()
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_ggtts_Native_free<'l>(mut env: JNIEnv<'l>, _class: JClass<'l>, handle: jlong) {
    jni_guard(
        &mut env,
        || (),
        |env| {
            let code = ggtts_free(handle as u64);
            if code != 0 {
                throw_last(env, code);
            }
        },
    )
}
