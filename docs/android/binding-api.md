# The ggtts binding: C-ABI and Kotlin

Written before the code (#54), and kept to what a first app needs. Each item names why it is here; an option that has
no "why" is left out.

## The C-ABI (`crates/ggtts-ffi`, `include/ggtts.h`)

Every function returns a status (`0` ok, a negative code otherwise) and never unwinds across the boundary: a panic
inside is caught, and it comes back as `GGTTS_ERR_PANIC`. The message for the last error on the calling thread is
available from `ggtts_last_error()`.

| Function | Why |
|---|---|
| `int ggtts_load(const char *engine, const char *model, const char *codec, int threads, uint64_t *handle)` | Load an engine by name (`qwen3-tts`, `kitten-tts-2`, or `fake` for tests) from GGUF paths. `codec` may be NULL for an engine that needs none. `threads <= 0` means 4. |
| `int ggtts_render(uint64_t handle, const char *text, uint64_t seed, float **pcm, size_t *len, uint32_t *rate)` | Render text on the story path (chunked, judged, levelled, as `ggtts-render --story` does) to mono f32 PCM. The seed is the story's salt. |
| `int ggtts_render_stream(uint64_t handle, const char *text, uint64_t seed, ggtts_piece_fn on_piece, void *user)` | The same render, delivered piece by piece as each chunk is kept (before the level step, as `--stream-raw`), so playback can start early. `on_piece(user, const float *samples, size_t len, uint32_t rate)` returns `0` to go on, non-zero to stop the render (`GGTTS_ERR_STOPPED`). |
| `void ggtts_pcm_free(float *pcm, size_t len)` | Free a buffer `ggtts_render` returned. |
| `int ggtts_free(uint64_t handle)` | Release the engine. |
| `const char *ggtts_last_error(void)` | The message for this thread's last error; valid until the next call on the thread. |

**Handles are ids, not pointers.** An engine lives in a registry behind its id. So `ggtts_render` after `ggtts_free`,
or with handle `0`, returns `GGTTS_ERR_HANDLE` rather than touching freed memory. An id is never reused.

Codes: `GGTTS_OK 0`, `GGTTS_ERR_ARG -1` (a NULL or non-UTF-8 argument), `GGTTS_ERR_HANDLE -2`, `GGTTS_ERR_ENGINE -3`
(the engine refused: a missing file, not a GGUF, unknown engine), `GGTTS_ERR_PANIC -4`, `GGTTS_ERR_STOPPED -5`.

Left out, for now: voices and speakers beyond each engine's default, per-chunk seeds, tracing and instruments over the
binding (#37 AC6 is a later story), and FLAC output (the app has the PCM, and Android plays PCM).

## Kotlin (`android/ggtts/src/main/kotlin/ggtts/Ggtts.kt`)

```kotlin
class Ggtts private constructor(private var handle: Long) : AutoCloseable {
    companion object {
        fun load(engine: String, model: String, codec: String? = null, threads: Int = 4): Ggtts
    }
    fun render(text: String, seed: Long = 20260905): Pcm             // Pcm(samples: FloatArray, sampleRate: Int)
    fun renderStream(text: String, seed: Long = 20260905, onPiece: PieceListener)
    override fun close()
}
fun interface PieceListener { fun onPiece(samples: FloatArray, sampleRate: Int): Boolean }  // false: stop
class GgttsException(val code: Int, message: String) : RuntimeException(message)
```

Every non-zero C status becomes a `GgttsException` with the code and `ggtts_last_error()`'s message. Nothing on the
Kotlin side can crash the process on a bad argument, a missing file or a closed instance: `render` after `close`
throws with `GGTTS_ERR_HANDLE`.

The JNI glue is in the same crate behind the `jni` feature (`Java_ggtts_Native_*`), so the C-ABI and the JNI layer
share one implementation.
