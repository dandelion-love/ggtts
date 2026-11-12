# Using ggtts from your own code

ggtts turns text into speech. Besides the `ggtts-render` command, you can call it from your own program through one of
three bindings. All three run the same engine code, take the same settings and return the same error codes.

| Binding | Use it from | Page |
|---|---|---|
| **C** | C, C++, or any language that calls a C library (Rust, Go, C#, Swift, ...) | [ggtts from C](c.md) |
| **Kotlin** | An Android app, or a desktop JVM | [ggtts from Kotlin and Android](kotlin-android.md) |
| **Python** | Python 3.9 or later | [ggtts from Python](python.md) |

Each page shows how to get the binding, then a first render that works, then covers loading, streaming, errors,
threads and freeing. Every C, Kotlin and Python example on these pages is a real file in [`examples/`](examples/), and CI compiles and runs each
one on every change that could break it, so the examples match the code.

## The ideas they share

* **An engine** is one model architecture: `qwen3-tts` (Qwen3-TTS), `kitten-tts-2` (Kitten-TTS 2), or `fake`, a test
  engine that needs no files and makes a tone instead of speech. You load an engine once and render with it many
  times.
* **Model files** are GGUF files: a model (Qwen3-TTS's talker, Kitten-TTS 2's LM) and an audio decoder, called the
  codec. Kitten-TTS 2 also needs a prepared reference voice. Releases do not include model files.
* **The default model.** Qwen3-TTS has one: given no model, every binding uses the same files `ggtts-render` uses (a
  Hong Kong Cantonese model, 2.33 GB), from the same per-user cache, checked against the same sha256. Kitten-TTS 2
  has none: you pass its files.
* **Text** is UTF-8, one paragraph per line. A render splits it into chunks, renders, checks and trims each, joins
  them and levels the whole read, as `ggtts-render --story` does.
* **The seed** fixes the random draws: the same engine, model, text and seed give the same samples on the same
  platform.
* **The result** is mono 32-bit float samples and a sample rate (24000 Hz for the models here).
* **Streaming** hands you each chunk as soon as it is ready, so playback can start early. You can stop it at any
  chunk.
* **One render at a time per engine.** A second call on a busy engine fails at once with `ERR_BUSY`; it never waits
  or deadlocks. Different engines render at the same time.

## The bindings compared

| | C | Kotlin / Android | Python |
|---|---|---|---|
| How you get it | Release archives (after v0.1.0-rc.10): `lib/` and `include/ggtts.h`. Or build from source. | From source only: copy `Ggtts.kt`, build `libggtts_ffi.so` with `jni`. No AAR is published. | Release wheels (after v0.1.0-rc.10). Or build with maturin. |
| Platforms of the released files | Linux x86-64 (glibc 2.35+), Windows x86-64 | None released. Android arm64-v8a from source; a desktop JVM for tests | Linux x86-64 (`manylinux_2_35`), Windows x86-64 (`win_amd64`) |
| Load | `ggtts_load_with(...)` | `Ggtts.load(...)` | `ggtts.load(...)` |
| Engines | `qwen3-tts`, `kitten-tts-2`, `fake` | the same | the same |
| No model given (Qwen3-TTS) | The default model | The default model | The default model |
| Downloads a missing default file | **Only if asked**: `allow_download = 1` | **Only if asked**: `allowDownload = true` | **By default**; `offline=True` turns it off |
| Cache location | `cache_dir` setting, else `GGTTS_CACHE_DIR`, else the per-user cache | `cacheDir` (an app passes its own) | `GGTTS_CACHE_DIR`, else the per-user cache |
| Speaker, style instruction (Qwen3-TTS) | `settings.speaker`, `settings.instruction` | `speaker =`, `instruction =` | `speaker=`, `instruction=` |
| Device | `settings.device`: `"cpu"` (default) or `"cuda"`, in a library built with CUDA. The archives' library is CPU only. | None: CPU only (no GPU backend on Android) | `device=`: `"cpu"` (default) or `"cuda"`, in a build with CUDA. The wheels are CPU only. |
| Threads | `threads` argument; 0 or less is 4 | `threads = 4` | `threads=None` is 4 |
| Seed | You pass one | Default 20260905 | Default 20260905 |
| Result | `float *` and its length and rate; free with `ggtts_pcm_free` | `Pcm(samples: FloatArray, sampleRate)` | `Audio`: numpy or `array` samples, `sample_rate`, `write_flac()` |
| Streaming | `ggtts_render_stream` with a callback | `renderStream` with a listener | `tts.stream()` generator, or `render_stream` with a callback |
| Stopping a stream | Callback returns non-zero; the call returns `GGTTS_ERR_STOPPED` | Listener returns false; the call returns false | `break`, or callback returns `False`; the call returns `False` |
| Errors | A negative code, and `ggtts_last_error()` on the same thread | `GgttsException(code, message)` | `ggtts.GgttsError(code, message)` |
| Releasing | `ggtts_free(handle)` | `close()` or `use { }` | `close()` or `with` |
| Concurrency | One render per handle; `GGTTS_ERR_BUSY` | One render per instance; `ERR_BUSY` | One render per engine; `ERR_BUSY`; the GIL is released while loading and rendering |
| Writes audio files | No (you get PCM) | No (you get PCM) | FLAC |

## The error codes

All three bindings use these codes. Each page says what causes each one in that binding and what to do.

| Code | Value | Meaning |
|---|---|---|
| `OK` | 0 | Success (C only; the others just return). |
| `ERR_ARG` | -1 | An argument is wrong: bad text, a missing file argument, a setting the engine does not take. |
| `ERR_HANDLE` | -2 | The engine is closed, freed, or was never loaded. |
| `ERR_ENGINE` | -3 | The engine refused: a missing or bad file, an unknown speaker, a default file not downloaded, no such engine or device in this build, a failed render. |
| `ERR_PANIC` | -4 | A bug inside ggtts, caught at the boundary; the engine is dropped. |
| `ERR_STOPPED` | -5 | The stream was stopped. Only C returns it. |
| `ERR_BUSY` | -6 | The engine is already rendering. |

## Where else to look

* [The README](../../README.md): the command-line tools, the release archives, and the default model files.
* [`crates/ggtts-ffi/include/ggtts.h`](../../crates/ggtts-ffi/include/ggtts.h): the C header, with comments.
* [The Kitten README](../../crates/ggtts-kitten/README.md): how to make Kitten-TTS 2's files.
* [`CHANGELOG.md`](../../CHANGELOG.md): what changed in each release.
