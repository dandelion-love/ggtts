//! The JNI entry points behind `ggtts.Native` (the Kotlin class `Ggtts` calls them). Each goes through the C-ABI
//! functions, so there is one implementation; a non-zero status becomes a thrown `ggtts.GgttsException(code,
//! message)`, never a crash.

use std::ffi::{c_void, CString};
use std::ptr;

use jni::objects::{JClass, JFloatArray, JIntArray, JObject, JString, JThrowable, JValue};
use jni::sys::{jboolean, jint, jlong};
use jni::JNIEnv;

use crate::{ggtts_free, ggtts_load, ggtts_pcm_free, ggtts_render, ggtts_render_stream, last_error, GGTTS_ERR_ARG};

/// Throw `ggtts.GgttsException(code, message)`.
fn throw(env: &mut JNIEnv, code: i32, msg: &str) {
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
        // The exception class is missing: still throw, never crash.
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("ggtts error {code}: {msg}"));
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
    let Ok(engine) = cstr(&mut env, &engine, "engine") else {
        return 0;
    };
    let Ok(model) = cstr(&mut env, &model, "model") else {
        return 0;
    };
    let Ok(codec) = cstr(&mut env, &codec, "codec") else {
        return 0;
    };
    let Ok(voice) = cstr(&mut env, &voice, "voice") else {
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
        throw(&mut env, code, &last_error());
        return 0;
    }
    handle as jlong
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
    let null = JFloatArray::from(JObject::null());
    let Ok(text) = cstr(&mut env, &text, "text") else {
        return null;
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
        throw(&mut env, code, &last_error());
        return null;
    }
    let samples = unsafe { std::slice::from_raw_parts(pcm, len) };
    let out = env
        .new_float_array(len as jint)
        .and_then(|a| env.set_float_array_region(&a, 0, samples).map(|_| a));
    unsafe { ggtts_pcm_free(pcm, len) };
    if !rate_out.is_null() {
        let _ = env.set_int_array_region(&rate_out, 0, &[rate as jint]);
    }
    match out {
        Ok(a) => a,
        Err(_) => {
            throw(&mut env, GGTTS_ERR_ARG, "could not allocate the sample array");
            null
        }
    }
}

/// What the stream callback needs: the JNI environment and the listener.
struct Stream<'a, 'l> {
    env: &'a mut JNIEnv<'l>,
    listener: &'a JObject<'l>,
}

extern "C" fn on_piece(user: *mut c_void, samples: *const f32, len: usize, rate: u32) -> i32 {
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
    let _ = s.env.delete_local_ref(arr);
    // A listener that threw, or returned false, stops the render.
    match go_on {
        Ok(true) if !s.env.exception_check().unwrap_or(true) => 0,
        _ => 1,
    }
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
    let Ok(text) = cstr(&mut env, &text, "text") else {
        return 0;
    };
    if listener.is_null() {
        throw(&mut env, GGTTS_ERR_ARG, "listener is null");
        return 0;
    }
    let code = {
        let mut stream = Stream {
            env: &mut env,
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
    // A listener's own exception is already pending: let it propagate as is.
    if env.exception_check().unwrap_or(false) {
        return 0;
    }
    match code {
        0 => 1,
        crate::GGTTS_ERR_STOPPED => 0,
        _ => {
            throw(&mut env, code, &last_error());
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ggtts_Native_free<'l>(mut env: JNIEnv<'l>, _class: JClass<'l>, handle: jlong) {
    let code = ggtts_free(handle as u64);
    if code != 0 {
        throw(&mut env, code, &last_error());
    }
}
