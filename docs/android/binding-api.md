# The ggtts binding: C-ABI and Kotlin

Written before the code (#54), and kept to what a first app needs. Each item names why it is here; an option that has
no "why" is left out.

## The C-ABI (`crates/ggtts-ffi`, `crates/ggtts-ffi/include/ggtts.h`)

Every function that returns a status returns `0` for ok and a negative code otherwise. A panic inside `ggtts_load`,
`ggtts_render`, `ggtts_render_stream` or `ggtts_free` is caught and comes back as `GGTTS_ERR_PANIC`, and the engine
that panicked is dropped. `ggtts_pcm_free` and `ggtts_last_error` do nothing that can panic. Only unwinding panics are
caught: an abort or a crash in C code still ends the process. The message for the last error on the calling thread is
available from `ggtts_last_error()`.

| Function | Why |
|---|---|
| `int ggtts_load(const char *engine, const char *model, const char *codec, const char *voice, int threads, uint64_t *handle)` | Load an engine by name (`qwen3-tts`, `kitten-tts-2`, or `fake` for tests) from GGUF paths. `voice` is Kitten-TTS 2's reference voice (a `ggtts-kitten prepare` JSON); NULL for the others. A missing file or a file that is not a GGUF is refused before any engine loads. `threads <= 0` means 4. |
| `int ggtts_load_with(const char *engine, const char *model, const char *codec, const char *voice, int threads, const ggtts_settings *settings, uint64_t *handle)` | `ggtts_load` with a Qwen3-TTS speaker and style instruction (C-ABI version 2), so an app can pick a voice the model has and a speaking style; and (version 3) the device, so a CUDA build can render on the GPU, and the default model files, so an app gets the model `ggtts-render` uses without shipping paths. `settings` may be NULL; see "Settings" below. |
| `uint32_t ggtts_abi_version(void)` | The loaded library's C-ABI version, so a caller built against a newer header can check before calling a newer function. |
| `int ggtts_render(uint64_t handle, const char *text, uint64_t seed, float **pcm, size_t *len, uint32_t *rate)` | Render text on the story path (chunked, judged, levelled, as `ggtts-render --story` does) to mono f32 PCM. The seed is the story's salt. |
| `int ggtts_render_stream(uint64_t handle, const char *text, uint64_t seed, ggtts_piece_fn on_piece, void *user)` | The same render, delivered piece by piece as each chunk is kept (before the level step, as `--stream-raw`), so playback can start early. `on_piece(user, const float *samples, size_t len, uint32_t rate)` returns `0` to go on, non-zero to stop the render (`GGTTS_ERR_STOPPED`). |
| `void ggtts_pcm_free(float *pcm, size_t len)` | Free a buffer `ggtts_render` returned. |
| `int ggtts_free(uint64_t handle)` | Release the engine. |
| `const char *ggtts_last_error(void)` | The message for this thread's last error; valid until the next call on the thread. |

**Handles are ids, not pointers.** An engine lives in a registry behind its id. So `ggtts_render` after `ggtts_free`,
or with handle `0`, returns `GGTTS_ERR_HANDLE` rather than touching freed memory. An id is never reused.

**No call waits on another.** The registry is locked only to look an id up, never while an engine renders, and each
engine has its own lock, which is tried rather than waited on. So:

* A second call on an engine that is rendering returns `GGTTS_ERR_BUSY`. That covers a stream callback calling back
  into its own engine, which would otherwise deadlock, and another thread using the same engine meanwhile. One render
  at a time per engine; different engines render at once.
* `ggtts_free` returns at once, even mid-render, including from the render's own callback. The render runs to the end
  and the engine is dropped when it returns. An app's `close()` on its UI thread never waits out a background render.

Codes: `GGTTS_OK 0`, `GGTTS_ERR_ARG -1` (a NULL or non-UTF-8 argument), `GGTTS_ERR_HANDLE -2`, `GGTTS_ERR_ENGINE -3`
(the engine refused: a missing file, not a GGUF, unknown engine), `GGTTS_ERR_PANIC -4`, `GGTTS_ERR_STOPPED -5`,
`GGTTS_ERR_BUSY -6`.

**Settings (version 2, extended in version 3).** `ggtts_settings` is passed by pointer and starts with its own size,
so a later version can add a field and still tell an older caller's struct apart: later versions accept any earlier
`sizeof(ggtts_settings)` and treat the fields it lacks as unset. A size this library does not know is
`GGTTS_ERR_ARG`. A caller built against the version-2 header passes the three-field struct, and loads exactly as it
did in version 2: on the CPU, from the paths it gives.

```c
typedef struct ggtts_settings {
    size_t size;              /* sizeof(ggtts_settings); GGTTS_SETTINGS_INIT sets it */
    const char *speaker;      /* NULL or "": the model's default speaker */
    const char *instruction;  /* NULL or "": no instruction */
    const char *device;       /* version 3. NULL or "": "cpu"; or "cuda" */
    const char *cache_dir;    /* version 3. NULL or "": ggtts-render's cache of the default model files */
    int32_t allow_download;   /* version 3. 0 (the default): never download; 1: download a missing default file */
} ggtts_settings;
```

Strings are UTF-8 and NUL-terminated, owned by the caller and read only during the call. A NULL or empty string is
no setting, and with no setting the engine and its output are `ggtts_load`'s, sample for sample. The settings are
fixed for the engine's life: every render on the handle uses them. All are checked while loading, before any text is
rendered, and a refused one writes no handle:

* **Speaker.** A speaker the model has, case ignored. An unknown one is `GGTTS_ERR_ENGINE`, and the message lists the
  model's speakers. Without one, a model given by path uses `hk_cantonese_speaker` as `ggtts_load` always has, so a
  talker without that speaker (the v0.0.5 GGUF has `hk_voice_343` and `hk_voice_777`) needs one; the default model
  uses its own default speaker (below).
* **Instruction.** A natural-language style instruction given to every chunk, as `ggtts-render --instruction`. A
  checkpoint that takes none (Base, or `custom_voice` below 1.7B) refuses it with `GGTTS_ERR_ENGINE` and the
  library's message.
* **Another engine** refuses either one with `GGTTS_ERR_ARG`: the refusal is about the engine, not a checkpoint.
* **Device (version 3).** `cpu` (the default, and what every earlier caller gets) or `cuda`, as `ggtts-render
  --device`, for Qwen3-TTS and Kitten-TTS 2. A library built without the CUDA backend (the `cuda` feature of
  `ggtts-ffi`) refuses `cuda` with `GGTTS_ERR_ENGINE` before reading any file; a CUDA build on a machine without a GPU
  refuses it with `GGTTS_ERR_ENGINE` while loading. Another value is `GGTTS_ERR_ARG`, and the fake engine takes `cpu`
  only. Android has no GPU backend in ggtts, so an Android library is CPU only.
* **Default model files (version 3).** With engine `qwen3-tts` and the model or the codec NULL, the missing file is
  the pinned default that `ggtts-render` uses with no `--model` or `--codec` (the v0.0.5 talker and codec of
  `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF`, 2.33 GB), resolved by the same code (`ggtts-models`): the same per-user
  cache, the same file names, the same sha256 check before use. A path given is used as it is, and the two may be
  mixed. With no model and no speaker the speaker is the default model's, `hk_voice_343`, as in `ggtts-render`.
  `cache_dir` moves the cache; NULL is `ggtts-render`'s (`GGTTS_CACHE_DIR`, else the platform's per-user cache
  directory). An Android process has no such directory, so an app passes its own, for example under
  `Context.getFilesDir()`.
* **Downloading is opt-in, everywhere (version 3).** The binding never downloads unless `allow_download` is 1. With 0
  (the default, and `GGTTS_SETTINGS_INIT`'s), a default file missing from the cache is `GGTTS_ERR_ENGINE`, and the
  message names the file, where it would be, and its URL, as `ggtts-render --offline` does. With 1, the missing files
  are downloaded over HTTPS inside the call, which blocks until they are verified and reports no progress. The
  default is off because a load call is not a place to spend 2.33 GB of a phone's data or storage without the user
  agreeing: an app asks first, on its own screen, then loads with `allow_download = 1` off the UI thread (or puts the
  files in the cache itself and never allows a download).

```c
ggtts_settings s = GGTTS_SETTINGS_INIT;
s.speaker = "hk_voice_777";                   /* a speaker */
s.instruction = "Speak slowly and softly.";   /* a style instruction */
s.device = "cuda";                            /* the GPU, in a CUDA build */
uint64_t h;
if (ggtts_load_with("qwen3-tts", talker, codec, NULL, 4, &s, &h) != GGTTS_OK)
    fprintf(stderr, "%s\n", ggtts_last_error());

ggtts_settings d = GGTTS_SETTINGS_INIT;       /* the default model and its default speaker */
d.allow_download = 1;                         /* downloaded once, if the cache does not have it */
ggtts_load_with("qwen3-tts", NULL, NULL, NULL, 4, &d, &h);
```

`ggtts_load`, NULL settings and a version-2 struct never resolve default files: with no model they are the argument
error they always were.

**Versioning.** `GGTTS_ABI_VERSION` in the header, and `ggtts_abi_version()` in the library, go up by one whenever a
function, a struct, a struct field or a code is added. An exported function never changes its signature or its
behaviour. Version 1 is the six functions above `ggtts_load_with`; version 2 added `ggtts_load_with`,
`ggtts_settings` and `ggtts_abi_version`; version 3 added `device`, `cache_dir` and `allow_download` to
`ggtts_settings`. A caller built against the version-3 header checks `ggtts_abi_version() >= 3` before passing the
longer struct: a version-2 library refuses it with `GGTTS_ERR_ARG` (its size is unknown there), so the check is for a
clearer message, not for safety.

Left out, for now: per-render speakers and instructions (the settings are per engine), Kitten-TTS 2 voices beyond the
reference voice file, per-chunk seeds, tracing and instruments over the binding (#37 AC6 is a later story), and FLAC
output (the app has the PCM, and Android plays PCM).

## Kotlin (`android/ggtts/src/main/kotlin/ggtts/Ggtts.kt`)

```kotlin
class Ggtts private constructor(private var handle: Long) : AutoCloseable {
    companion object {
        // voice is Kitten-TTS 2's reference voice; speaker and instruction are Qwen3-TTS voice settings, null for
        // none; for "qwen3-tts", a null model or codec is the default file from the cache at cacheDir, downloaded
        // only when allowDownload is true (ggtts_load_with, version 3)
        fun load(engine: String, model: String? = null, codec: String? = null, voice: String? = null, threads: Int = 4,
                 speaker: String? = null, instruction: String? = null,
                 cacheDir: String? = null, allowDownload: Boolean = false): Ggtts
    }
    fun render(text: String, seed: Long = 20260905): Pcm             // Pcm(samples: FloatArray, sampleRate: Int)
    fun renderStream(text: String, seed: Long = 20260905, listener: PieceListener): Boolean  // false: the listener stopped it
    override fun close()                                             // never waits for a running render
}
fun interface PieceListener { fun onPiece(samples: FloatArray, sampleRate: Int): Boolean }  // false: stop
class GgttsException(val code: Int, message: String) : RuntimeException(message)
```

```kotlin
val g = Ggtts.load("qwen3-tts", talker, codec, speaker = "hk_voice_777")                     // a speaker
val s = Ggtts.load("qwen3-tts", talker, codec, speaker = "hk_voice_777",
                   instruction = "Speak slowly and softly.")                                  // and a style
```

A call that names neither gets what it got before. A refused setting throws as any other error: an unknown speaker
or an instruction on a checkpoint that takes none is a `GgttsException` with `ERR_ENGINE` and the library's message.

```kotlin
// The default model and its default speaker (hk_voice_343), from the app's own cache directory. Without
// allowDownload a missing file throws ERR_ENGINE naming the file and its URL; ask the user before allowing the
// 2.33 GB download, and call this off the UI thread: it blocks until the files are verified.
val dir = File(context.filesDir, "ggtts").path
val d = Ggtts.load("qwen3-tts", cacheDir = dir, allowDownload = userAgreed)
```

Kotlin has no device setting: Android has no GPU backend in ggtts, so the binding always renders on the CPU. The
seven-parameter and five-parameter `load` from earlier versions stay in the compiled class, hidden from source, so
code built against them runs without recompiling and loads exactly as it did (the paths given only).

Every non-zero C status becomes a `GgttsException` with the code and `ggtts_last_error()`'s message. Nothing on the
Kotlin side can crash the process on a bad argument, a missing file or a closed instance: `render` after `close`
throws with `GGTTS_ERR_HANDLE`.

The JNI glue is in the same crate behind the `jni` feature (`Java_ggtts_Native_*`), so the C-ABI and the JNI layer
share one implementation. It makes no JNI call with an exception pending: a listener's exception, or one a failed JNI
call left, goes to Kotlin as it is, and nothing is thrown over it. A panic in a `Java_*` body or the stream callback
becomes a `GgttsException` with `GGTTS_ERR_PANIC` rather than an abort.

**Revised after review** (the first version held the registry lock for a whole render, so a listener calling back in
deadlocked and `close()` waited out a render): the `GGTTS_ERR_BUSY` code, the no-wait rules above, the narrower
`catch_unwind` claim, and the Kotlin signatures as built (`voice`, a nullable `model`, `renderStream` returning
`Boolean`).
