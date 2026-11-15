# ggtts from Kotlin and Android

The Kotlin binding is one class, `ggtts.Ggtts`, in
[`android/ggtts/src/main/kotlin/ggtts/Ggtts.kt`](../../android/ggtts/src/main/kotlin/ggtts/Ggtts.kt). It calls the C
library over JNI, so it needs `libggtts_ffi` built with the JNI entry points. It runs in an Android app and on a
desktop JVM. [All bindings](README.md) compares it with C and Python.

## Get it

**No Android package is published**: there is no Maven artifact, and the C library in the release archives does not
export the JNI entry points. There are two ways to get it.

**The AAR.** [`android/`](../../android/README.md) is a Gradle project. Its `ggtts` module packs `Ggtts.kt` and the
arm64-v8a library into an AAR, and its `demo` module is a one-screen app that uses it.
`scripts/android_build.sh app OUT_DIR` builds both, and CI's `android-arm64` job uploads them as its artifact. Add the
AAR to your app as a file dependency. It carries its licence notices and build record under `META-INF/ggtts/`
([what the AAR carries](../../android/README.md#what-the-aar-carries)).

**By hand.** Build two pieces from source and add them to your app:

1. **The Kotlin class.** Copy `Ggtts.kt` into your app as it is. Keep its package, `ggtts`: the JNI function names
   (`Java_ggtts_Native_*`) are tied to the package and class names.
2. **The native library**, `libggtts_ffi.so`, built for Android with the `jni` and `ggml` features, in your app's
   `src/main/jniLibs/arm64-v8a/`. `Ggtts` loads it with `System.loadLibrary("ggtts_ffi")`.

To build the library you need the Rust toolchain that `rust-toolchain.toml` names, the Android NDK, cargo-ndk, CMake
and libclang (the Android build generates its ggml bindings). This repository's CI builds `ggtts-render` for Android
this way, with NDK 27.2.12479018, cargo-ndk 3.5.4 and API level 24
([`scripts/android_build.sh`](../../scripts/android_build.sh)). The library is the same build with `ggtts-ffi` in
place of the command-line tool:

```sh
git clone https://github.com/dandelion-love/ggtts && cd ggtts
git submodule update --init crates/ggtts-ggml-sys/ggml
rustup target add aarch64-linux-android
export ANDROID_NDK_HOME=/path/to/ndk/27.2.12479018
export BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android="--sysroot=$(ls -d "$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*/sysroot) --target=aarch64-linux-android24"
cargo ndk -t arm64-v8a --platform 24 build --release --locked -p ggtts-ffi -p ggtts-ggml-sys \
    --features ggtts-ffi/ggml,ggtts-ffi/jni,ggtts-ggml-sys/bindgen
# the library: target/aarch64-linux-android/release/libggtts_ffi.so
```

CI builds this library for Android inside the AAR (above), and also runs the binding against a host build (below). Check what the library needs with the NDK's `llvm-readelf -d libggtts_ffi.so`, and ship any `NEEDED`
library that is not an Android system library beside it. For `ggtts-render` that is the NDK's `libc++_shared.so`,
which `scripts/android_build.sh` copies.

**On a desktop JVM** (Linux; how CI tests the binding), build the library for the host and name it with the system
property `ggtts.lib`:

```sh
cargo build --release --locked -p ggtts-ffi --features jni,ggml     # without ggml: only the fake engine
kotlinc Ggtts.kt FirstRender.kt -d classes
java -Dggtts.lib="$PWD/target/release/libggtts_ffi.so" \
     -cp "classes:/path/to/kotlin-stdlib.jar" firstrender.FirstRenderKt
```

[`scripts/kotlin_binding_test.sh`](../../scripts/kotlin_binding_test.sh) does exactly this for the tests and for the
examples on this page.

## Requirements

* **Android:** arm64-v8a only. That is the one ABI the project builds and checks (ggml at its pin, NEON in the
  kernels); 32-bit ARM and x86 are not built. API level 24 is what the build targets.
* **The CPU only.** ggtts has no GPU backend on Android, so the Kotlin binding has no device setting.
* **Storage and network**, for Qwen3-TTS's default model: 2.33 GB, downloaded once, only if you allow it (below).
* **Desktop JVM:** a JDK, `kotlinc`, and the library built on that machine as above. CI runs it on Linux x86-64.

## A first render

This program loads Qwen3-TTS, renders one line and writes `first_render.wav`. With no arguments, the first run
downloads the default model (2.33 GB) into its cache directory and checks its sha256; later runs download nothing.

<!-- example: examples/kotlin/FirstRender.kt -->
```kotlin
// FirstRender.kt: load an engine, render one line, write it to a WAV file. It runs on a desktop JVM as it is; in an
// Android app, the same calls run off the UI thread, with cacheDir under context.filesDir.
//
//   main(arrayOf())                                        Qwen3-TTS with its default model (downloaded once, 2.33 GB)
//   main(arrayOf("qwen3-tts", "talker.gguf", "codec.gguf"))  model files you already have
//   main(arrayOf("kitten-tts-2", "lm.gguf", "s3gen.gguf", "voice.json"))
//   main(arrayOf("fake"))                                  the test engine: no files, a tone instead of speech
package firstrender

import ggtts.Ggtts
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun main(args: Array<String>) {
    val engine = args.getOrElse(0) { "qwen3-tts" }
    val cacheDir = File(System.getProperty("java.io.tmpdir"), "ggtts-cache").path  // Android: File(context.filesDir, "ggtts").path

    Ggtts.load(
        engine,
        model = args.getOrNull(1),          // null: the default model (qwen3-tts only)
        codec = args.getOrNull(2),
        voice = args.getOrNull(3),          // kitten-tts-2's prepared reference voice
        cacheDir = cacheDir,
        allowDownload = true,               // ask the user before this on a phone: it is 2.33 GB
    ).use { tts ->
        val pcm = tts.render("今日天氣好好。")    // seed = Ggtts.DEFAULT_SEED unless you pass one
        writeWav(File("first_render.wav"), pcm.samples, pcm.sampleRate)
        println("${pcm.samples.size} samples at ${pcm.sampleRate} Hz, written to first_render.wav")
    }
}

/** Mono float samples as a 16-bit PCM WAV file. */
fun writeWav(file: File, samples: FloatArray, rate: Int) {
    val data = samples.size * 2
    val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
    b.put("RIFF".toByteArray()).putInt(36 + data).put("WAVEfmt ".toByteArray())
    b.putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
    b.put("data".toByteArray()).putInt(data)
    for (x in samples) b.putShort((x.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
    file.writeBytes(b.array())
}
```

In an app, `load` and `render` block: call them off the main thread. To play a `Pcm` rather than save it, Android's
`AudioTrack` takes float samples with `AudioFormat.ENCODING_PCM_FLOAT`, mono, at `pcm.sampleRate`.

## The class

| Member | What it does |
|---|---|
| `Ggtts.load(engine, model, codec, voice, threads, speaker, instruction, cacheDir, allowDownload, options)` | Load an engine. Every argument has a default; name the ones you set. `options` is a load document ([Options](#options)). |
| `render(text, seed = null, options = null): Pcm` | Render the whole text. `Pcm` has `samples: FloatArray` (mono) and `sampleRate: Int`. |
| `renderStream(text, seed = null, options = null, listener): Boolean` | Render, handing each piece to `listener` as soon as it is ready. False if the listener stopped it. |
| `capabilities(): String` | What the engine supports, as JSON. `Ggtts.optionsSchema()` is the options schema. See [Options](#options). |
| `close()` | Release the engine. `Ggtts` is `AutoCloseable`, so `use { }` works. |
| `Ggtts.ERR_ARG` ... `Ggtts.ERR_BUSY`, `Ggtts.DEFAULT_SEED` | The error codes, and the default seed, 20260905. |

## Loading

| Argument | Default | Meaning |
|---|---|---|
| `engine` | `"qwen3-tts"` | `"qwen3-tts"`, `"kitten-tts-2"`, or `"fake"` (a test engine that needs no files). With `options`, null is the document's `load.engine`, whose default is `"auto"`: the model file's GGUF architecture, or `"qwen3-tts"` with no model. |
| `model` | null | The model GGUF path: Qwen3-TTS's talker, or Kitten-TTS 2's LM. Null for Qwen3-TTS means the default model (below). |
| `codec` | null | The audio decoder GGUF: Qwen3-TTS's 12 Hz codec, or Kitten-TTS 2's S3Gen GGUF. Null for Qwen3-TTS means the default codec. |
| `voice` | null | Kitten-TTS 2's prepared reference voice (a JSON file). |
| `threads` | null: 4 | CPU threads. 0 or less means 4. Before each graph the count is capped at the CPUs the app may use, so an app moved to the little cores when it leaves the screen does not stall; the first cap is said once on stderr, which an Android app does not show in logcat. |
| `speaker` | null | A Qwen3-TTS speaker the model has, such as `"hk_voice_777"`. Case is ignored. |
| `instruction` | null | A Qwen3-TTS style instruction for every chunk, such as `"Speak slowly and softly."` |
| `cacheDir` | null | Where the default model files are cached. An app passes its own (below). |
| `allowDownload` | null: false | True: download a default model file the cache lacks. False: never download. |
| `options` | null | A load document, JSON: every option a load takes, and render defaults. See [Options](#options). |

Paths are file paths on the device. For `speaker`, `instruction` and `cacheDir`, an empty string is the same as null:
no setting. That is not true of `model`, `codec` and `voice`: an empty one is taken as a path and fails with
`ERR_ENGINE`, so pass null, not `""`, to get the default model. A setting the engine does not take is refused, never
ignored. A file that is missing or is not a GGUF is refused before any engine loads. The settings hold
for the engine's life.

`GGTTS_KITTEN_FULL_HEAD=1` (exactly `1`) makes Kitten-TTS 2 compute its whole output head instead of the rows a step can
keep (see the crate README). It is a diagnostic switch read from the process environment only; no binding setting
carries it.

### The default model (Qwen3-TTS)

With engine `"qwen3-tts"` and a null `model` or `codec`, the binding uses the same default files as `ggtts-render`:
the v0.0.5 talker and codec of
[jmtl/hk-cantonese-cc0-qwen3-tts-GGUF](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF), 2.33 GB in all,
each checked against a pinned sha256. A path you give is used as it is, and you may give one and not the other.
Read the model's card before you use its voice; its licence is the model's, not this repository's.

* **Downloading is off unless you turn it on.** With `allowDownload = false`, a default file missing from the cache
  throws `ERR_ENGINE`, and the message names the file and its URL. With `true`, the missing files are downloaded over
  HTTPS inside `load`, which blocks until they are verified and reports no progress. On a phone, ask the user first,
  on your own screen, and call `load` off the main thread. Or put the files in the cache yourself and never allow a
  download.
* **The cache.** Pass `cacheDir`, for example `File(context.filesDir, "ggtts").path`. Null means the desktop
  default (`GGTTS_CACHE_DIR`, else the per-user cache directory), which an Android process does not have. Each file is
  at `models/<sha256>/<file name>` under it. The README's
  [default model files](../../README.md#default-model-files-what-ggtts-downloads-and-when) section lists the files and
  their hashes.
* **The speaker.** The default model's speakers are `hk_voice_343` (used when you give none) and `hk_voice_777`. A
  model you give by path uses `hk_cantonese_speaker` when you give none, so a talker without that speaker needs
  `speaker`. An unknown speaker throws `ERR_ENGINE`, and the message lists the model's speakers.

### Kitten-TTS 2

Kitten-TTS 2 needs `model` (the LM GGUF), `codec` (the S3Gen GGUF) and `voice` (a reference voice made by
`ggtts-kitten prepare`). It has no default model, and no ggtts-ready Kitten files are published: make them once on a
computer with `ggtts-kitten-convert` and `ggtts-kitten` from a release archive, as
[the Kitten README](../../crates/ggtts-kitten/README.md) shows, then copy them to the device. It takes no speaker and
no instruction (`ERR_ARG`). Its licence is the model's: audio rendered with it must not be used to train other models.

## Rendering

* **Text** is one paragraph per line. Empty lines are skipped. Text with nothing to say (empty, or only punctuation)
  throws `ERR_ARG`, before any rendering.
* **The render** splits the text into chunks, renders each, checks it, trims it, joins them and levels the whole read,
  as `ggtts-render --story` does.
* **The seed** fixes the random draws. It is the **seed salt**: each chunk's sampler seed is derived from the text, a
  fixed story id and this salt. In an options document its key is `render.seed_salt` (a string). With no `seed` and
  no `options`, the salt is `Ggtts.DEFAULT_SEED`, 20260905; with `options` and no salt given anywhere, it is empty,
  as the C library's `ggtts_render_opts` leaves it, so a document renders the same samples in every binding. Pass
  `seed` to keep a render the same with and without a document. The same engine, model, text and seed give the same samples on the same
  platform.
* **`options`** is a render document, laid over the load's render defaults. See [Options](#options).
* **The result** is a `Pcm`: mono float samples in a `FloatArray` the JVM owns, at `sampleRate` Hz (24000 for the
  models above).

## Streaming and stopping

`renderStream(text, seed, listener)` calls `listener.onPiece(samples, sampleRate)` once for each kept chunk, in order,
so playback can start before the render ends.

* The listener runs on the thread that called `renderStream`.
* Each piece is a new `FloatArray`; you may keep it.
* Return true to go on, false to stop. A stopped render returns false; it does not throw.
* An exception the listener throws stops the render and is thrown from `renderStream` as it is.
* The pieces are the render's samples *before* its final level step. Joined, they have the render's length, but not
  its exact values.

This program streams, stops after two pieces, tries a render from inside the listener, and shows three errors:

<!-- example: examples/kotlin/Stream.kt -->
```kotlin
// Stream.kt: render piece by piece, stop early, and see the threading rules and error codes at work.
//
//   main(arrayOf())                                        Qwen3-TTS with its default model (it must be in the cache)
//   main(arrayOf("fake"))                                  the test engine
//   main(arrayOf("qwen3-tts", "talker.gguf", "codec.gguf"))
package streaming

import ggtts.Ggtts
import ggtts.GgttsException

const val TEXT = "今日天氣好好。\n我哋去海邊行下。\n行完之後，去食飯。"

fun main(args: Array<String>) {
    val tts = try {
        // No allowDownload: a default file missing from the cache throws ERR_ENGINE naming the file and its URL.
        Ggtts.load(args.getOrElse(0) { "qwen3-tts" }, model = args.getOrNull(1), codec = args.getOrNull(2))
    } catch (e: GgttsException) {
        System.err.println("load failed (${e.code}): ${e.message}")
        return
    }

    tts.use {
        // The listener runs on the thread that called renderStream, once per kept chunk, in order.
        // Return true to go on, false to stop. Here it also tries to render from inside the listener.
        var pieces = 0
        var busy = 0
        val finished = tts.renderStream(TEXT) { samples, sampleRate ->
            pieces++
            println("piece $pieces: ${samples.size} samples at $sampleRate Hz")
            try {
                tts.render("x")                    // one render at a time per engine: this throws ERR_BUSY
            } catch (e: GgttsException) {
                busy = e.code
            }
            pieces < 2                             // stop after two pieces
        }
        println("renderStream returned $finished after $pieces pieces; a render from the listener threw $busy")
        check(!finished && pieces == 2 && busy == Ggtts.ERR_BUSY)

        // Text with nothing to say is refused before rendering.
        val refused = runCatching { tts.render("？！") }.exceptionOrNull() as GgttsException
        println("punctuation only: ${refused.code} (ERR_ARG): ${refused.message}")
        check(refused.code == Ggtts.ERR_ARG)
    }

    // After close (here, the end of use), every call throws ERR_HANDLE.
    val closed = runCatching { tts.render(TEXT) }.exceptionOrNull() as GgttsException
    println("after close: ${closed.code} (ERR_HANDLE)")
    check(closed.code == Ggtts.ERR_HANDLE)
}
```

## Options

Every option of ggtts's options table reaches Kotlin through one parameter, `options`, so an option added to the table
needs no new parameter. A document is a JSON string, the C library's options document
([C: options documents](c.md#options-documents)), read and checked over JNI by the same code. The schema is
`schemas/ggtts-options-1.schema.json`, and [docs/options.md](../options.md) describes the sections.

* **`Ggtts.load(options = ...)`** takes the `load` section (each `load` parameter has its key there) and `render`
  defaults that every render on the engine starts from. A parameter left null is not given; one given both as a
  parameter and in the document throws `ERR_ARG` naming both. The whole document is checked against the engine before
  any file is read.
* **`render(text, options = ...)`** and **`renderStream(text, options = ...) { ... }`** take a render document, laid
  over the load's defaults: `render` `seed_salt`, `instruction`, `steer` with `steer_scale` and `steer_layers`,
  `redraw` and `edge_s`. A stream's pieces come before the level step; `{"output": {"stream": "raw"}}` says so and
  may be left out. `render` refuses `output.stream`: stream with `renderStream`.
* **Refused:** `load.device` (the binding is CPU only), the trace (`observe.trace`: a desktop observer, with no file a
  phone user can read), instruments, a `load` key in a render document, `render.seed` (use `seed_salt`), the reports
  and logging, `render.measure_timings`, `output.out` and the book options. A levelled stream is not taken yet.

**What the library states.** `Ggtts.optionsSchema()` is the options schema as JSON: every option a document may
give, with its type, default and phase, from the library (the C library's `ggtts_options_schema`), so no schema file
ships with the binding. `capabilities()` on an instance is what its engine supports, as JSON (`ggtts-capabilities/1`,
the C library's `ggtts_capabilities`): `family`, `instruction` and `no_instruction_why`, `steering`, `speakers` and
`speaker_names` (the model's speakers), `devices`, `hooks` and `step_events`. Read them to offer only what the engine
takes, for example to show steering only when `steering` is true.

A refused option throws `GgttsException` with `ERR_ARG`, and its `key` is the option's schema key, read from the
library (`ggtts_last_error_key`) and never parsed from the message: `render.steer` for steering on an engine that has
none, `render.nonesuch` for an unknown key. A document that is not JSON is refused whole, with `key` null.

This program loads with a document, renders and streams with render documents, and catches a refusal:

<!-- example: examples/kotlin/Options.kt -->
```kotlin
// Options.kt: every option through one argument, `options`, at load and per render; a refused option names its key.
//
//   main(arrayOf())                                          Qwen3-TTS with its default model (from the cache)
//   main(arrayOf("qwen3-tts", "talker.gguf", "codec.gguf"))  model files you already have
//   main(arrayOf("fake"))                                    the test engine: no files, a tone instead of speech
package options

import ggtts.Ggtts
import ggtts.GgttsException

fun main(args: Array<String>) {
    val engine = args.getOrElse(0) { "qwen3-tts" }
    // The load document: the load section, and render defaults every render on this engine starts from. The
    // parameters of load still work beside it; one given both ways is refused.
    Ggtts.load(
        engine,
        model = args.getOrNull(1),
        codec = args.getOrNull(2),
        options = """{"render": {"edge_s": 0.25}}""",
    ).use { tts ->
        // A render document, laid over those defaults. seed = 7 is the same value as "render": {"seed_salt": "7"}.
        val pcm = tts.render("今日天氣好好。", seed = 7, options = """{"render": {"redraw": 1}}""")
        println("${pcm.samples.size} samples at ${pcm.sampleRate} Hz")

        // A stream through a document: the pieces before the level step, as renderStream always hands them.
        var pieces = 0
        tts.renderStream("今日天氣好好。\n我哋去海邊行下。", options = """{"output": {"stream": "raw"}}""") { _, _ ->
            pieces++
            true
        }
        println("$pieces pieces")

        // A refused option throws GGTTS_ERR_ARG, and the exception's key is the option's schema key.
        try {
            tts.render("今日天氣好好。", options = """{"render": {"nonesuch": 1}}""")
        } catch (e: GgttsException) {
            println("refused ${e.key}: ${e.message}")
            check(e.code == Ggtts.ERR_ARG && e.key == "render.nonesuch")
        }
    }
}
```

## Errors

Every failure is a `ggtts.GgttsException` (a `RuntimeException`) with `code`, one of the constants below (the C
library's codes), the library's `message`, and `key`: the schema key of the one option refused, or null. A bad argument, a missing file or a closed instance never crashes the
process.

| Code | Value | What happened | What to do |
|---|---|---|---|
| `Ggtts.ERR_ARG` | -1 | An argument is wrong: empty or punctuation-only text, a string with a NUL character, a missing `model`, `codec` or `voice` the engine needs, or a speaker or instruction for an engine that takes none. | Fix the argument the message names. |
| `Ggtts.ERR_HANDLE` | -2 | The instance is closed, or its engine was dropped after a panic. | Load again. Do not use an instance after `close()`. |
| `Ggtts.ERR_ENGINE` | -3 | The engine refused: an unknown engine name, a missing file or one that is not a GGUF, an unknown speaker, an instruction the checkpoint takes none of, a default model file missing with `allowDownload = false`, a failed or mismatched download, an engine this library was built without, or a failed render. | Read the message. For a missing default file, allow the download, put the file in the cache, or pass paths. For an unknown speaker, pick one the message lists. |
| `Ggtts.ERR_PANIC` | -4 | A bug inside ggtts, caught at the boundary. The engine is dropped. | Load again. Please report it as an issue, with the message. |
| `Ggtts.ERR_STOPPED` | -5 | Never thrown in Kotlin: a stopped `renderStream` returns false. It exists so the codes match C. | Nothing. |
| `Ggtts.ERR_BUSY` | -6 | The engine is already rendering, on another thread or in its own listener. | Wait for that render, or load a second engine. |

## Threads

* **`load` and `render` block.** In an app, call them from a background thread or a coroutine dispatcher for blocking
  work, never the main thread.
* **One render at a time per instance.** A second call while one runs throws `ERR_BUSY` at once. It never waits and
  never deadlocks, including a call from the instance's own listener.
* **Different instances render at the same time**, on different threads. A second instance is a second loaded model,
  with its own memory, which matters on a phone.
* **`close()` never waits.** Called while a render runs (on another thread, or from its own listener), it returns at
  once; the render finishes, and the model is released when it returns.

## Lifetime

* Close an instance when you are done: `close()`, or `use { }`. After that every call throws `ERR_HANDLE`. Closing
  twice is harmless.
* A `Pcm` and each streamed `FloatArray` belong to the JVM. They stay valid after `close()`.

## Compatibility

The binding grew `speaker` and `instruction` (C ABI version 2), then `cacheDir` and `allowDownload` (version 3), then
`options` (version 4's documents). The older `load`, `render` and `renderStream` signatures stay in the compiled
class, hidden from source, so code built against them still runs and renders exactly as it did. New code calls the
ones above. A new option arrives through `options`, with no new parameter.
