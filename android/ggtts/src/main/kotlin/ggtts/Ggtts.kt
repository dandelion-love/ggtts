// The Kotlin binding over the ggtts C-ABI (docs/android/binding-api.md). Every failure is a GgttsException; nothing
// here can crash the process on a bad argument, a missing file or a closed instance.
package ggtts

/**
 * A ggtts error: [code] is the C-ABI's status (GGTTS_ERR_*), [message] its explanation, and [key] the schema key of
 * the one option refused (for example "render.steer"), read from the library (`ggtts_last_error_key`), or null.
 */
class GgttsException(val code: Int, message: String, val key: String?) : RuntimeException(message) {
    constructor(code: Int, message: String) : this(code, message, null)
}

/** Mono f32 samples at [sampleRate]. */
class Pcm(val samples: FloatArray, val sampleRate: Int)

/** Receives each kept chunk's piece; return false to stop the render. */
fun interface PieceListener {
    fun onPiece(samples: FloatArray, sampleRate: Int): Boolean
}

/** The JNI entry points (crates/ggtts-ffi, feature `jni`). */
internal object Native {
    init {
        // A host test names the built library; an app loads libggtts_ffi.so from its jniLibs.
        val path = System.getProperty("ggtts.lib")
        if (path != null) System.load(path) else System.loadLibrary("ggtts_ffi")
    }

    @JvmStatic external fun load(engine: String, model: String?, codec: String?, voice: String?, threads: Int): Long
    @JvmStatic external fun loadWith(
        engine: String,
        model: String?,
        codec: String?,
        voice: String?,
        threads: Int,
        speaker: String?,
        instruction: String?,
    ): Long
    @JvmStatic external fun loadWithModels(
        engine: String,
        model: String?,
        codec: String?,
        voice: String?,
        threads: Int,
        speaker: String?,
        instruction: String?,
        cacheDir: String?,
        allowDownload: Boolean,
    ): Long
    @JvmStatic external fun loadOpts(
        optionsJson: String,
        engine: String?,
        model: String?,
        codec: String?,
        voice: String?,
        threads: Int,
        speaker: String?,
        instruction: String?,
        cacheDir: String?,
        allowDownload: Int,
    ): Long
    @JvmStatic external fun render(handle: Long, text: String, seed: Long, rateOut: IntArray): FloatArray
    @JvmStatic external fun renderOpts(
        handle: Long,
        text: String,
        optionsJson: String?,
        seed: Long,
        seedGiven: Boolean,
        rateOut: IntArray,
    ): FloatArray
    @JvmStatic external fun renderStream(handle: Long, text: String, seed: Long, listener: PieceListener): Boolean
    @JvmStatic external fun renderStreamOpts(
        handle: Long,
        text: String,
        optionsJson: String?,
        seed: Long,
        seedGiven: Boolean,
        listener: PieceListener,
    ): Boolean
    @JvmStatic external fun optionsSchema(): String
    @JvmStatic external fun capabilities(handle: Long): String
    @JvmStatic external fun free(handle: Long)
}

/**
 * One loaded engine. Close it to release the model; any call after that throws GGTTS_ERR_HANDLE. Close never waits: a
 * render already running (on another thread, or the one whose listener closes it) finishes, and the model goes then.
 * One render at a time per engine: a second call while one runs throws GGTTS_ERR_BUSY.
 */
class Ggtts private constructor(private var handle: Long) : AutoCloseable {
    companion object {
        const val ERR_ARG = -1
        const val ERR_HANDLE = -2
        const val ERR_ENGINE = -3
        const val ERR_PANIC = -4
        const val ERR_STOPPED = -5
        /** The engine is rendering: a call on it from its own listener, or from another thread meanwhile. */
        const val ERR_BUSY = -6
        const val DEFAULT_SEED = 20260905L

        /**
         * The JSON Schema of an options document, as the library states it (the C-ABI's `ggtts_options_schema`):
         * every option `options` may give, with its type, default and phase. No schema file ships with the binding.
         */
        fun optionsSchema(): String = Native.optionsSchema()

        /**
         * [engine]: "qwen3-tts", "kitten-tts-2" or "fake". [voice] is Kitten-TTS 2's reference voice.
         *
         * [speaker] and [instruction] are Qwen3-TTS settings; null or empty is no setting, and the result is the same
         * as without them. [speaker] is a speaker the model has (for example "hk_voice_777"); an unknown one throws
         * GGTTS_ERR_ENGINE with the model's speakers in the message. [instruction] is a style instruction for every
         * render (for example "Speak slowly and softly."); a checkpoint that takes none (Base, or custom_voice below
         * 1.7B) throws GGTTS_ERR_ENGINE. Another engine refuses either one with GGTTS_ERR_ARG.
         *
         * The default model: for "qwen3-tts", a null [model] or [codec] is the pinned default file ggtts-render uses
         * with no --model (the v0.0.5 talker and codec, 2.33 GB), from the cache under [cacheDir] and verified against
         * its sha256; a path given is used as it is. With a null [model] and [speaker], the speaker is the default
         * model's ("hk_voice_343"). [cacheDir] null is ggtts-render's cache (GGTTS_CACHE_DIR, else the per-user cache
         * directory), which an Android process does not have: an app passes its own, for example under
         * `context.filesDir`. A default file missing from the cache throws GGTTS_ERR_ENGINE naming the file and its
         * URL, unless [allowDownload] is true: then it is downloaded over HTTPS inside this call, which blocks until
         * the file is verified, so call it off the UI thread and only after the user has agreed to the download.
         *
         * The engine runs on the CPU: Android has no GPU backend in ggtts, so the binding has no device setting.
         *
         * [options] is a load document, JSON, as the C-ABI's `ggtts_load_opts` takes it: the `load` section (every
         * parameter above has its key there) and `render` defaults every render on the engine starts from. With it,
         * a parameter left null is not given, and one given both here and in the document throws GGTTS_ERR_ARG naming
         * both. A refused option throws GGTTS_ERR_ARG with its [GgttsException.key]; so do `load.device` (CPU only)
         * and the trace (no file sink a phone user can read). Without it, null [engine] is "qwen3-tts", [threads] 4
         * and [allowDownload] false, and the load is the one this function always made.
         */
        fun load(
            engine: String? = null,
            model: String? = null,
            codec: String? = null,
            voice: String? = null,
            threads: Int? = null,
            speaker: String? = null,
            instruction: String? = null,
            cacheDir: String? = null,
            allowDownload: Boolean? = null,
            options: String? = null,
        ): Ggtts = Ggtts(
            if (options == null) {
                Native.loadWithModels(
                    engine ?: "qwen3-tts",
                    model,
                    codec,
                    voice,
                    threads ?: 4,
                    speaker,
                    instruction,
                    cacheDir,
                    allowDownload ?: false,
                )
            } else {
                Native.loadOpts(
                    options,
                    engine,
                    model,
                    codec,
                    voice,
                    threads ?: 0,
                    speaker,
                    instruction,
                    cacheDir,
                    when (allowDownload) {
                        null -> -1
                        true -> 1
                        false -> 0
                    },
                )
            },
        )

        /**
         * The nine-parameter `load` from before the options document, kept so code compiled against it (including its
         * default-argument form) still links and loads exactly as it did. Hidden from source.
         */
        @Deprecated("Kept for binary compatibility; call load with options", level = DeprecationLevel.HIDDEN)
        fun load(
            engine: String,
            model: String? = null,
            codec: String? = null,
            voice: String? = null,
            threads: Int = 4,
            speaker: String? = null,
            instruction: String? = null,
            cacheDir: String? = null,
            allowDownload: Boolean = false,
        ): Ggtts = Ggtts(
            Native.loadWithModels(engine, model, codec, voice, threads, speaker, instruction, cacheDir, allowDownload),
        )

        /**
         * The seven-parameter `load` from before the default model, kept so code compiled against it (including its
         * default-argument form) still links and loads exactly as it did: the paths given only. Hidden from source,
         * where the `load` above covers every call.
         */
        @Deprecated("Kept for binary compatibility; call load with cacheDir and allowDownload", level = DeprecationLevel.HIDDEN)
        fun load(
            engine: String,
            model: String? = null,
            codec: String? = null,
            voice: String? = null,
            threads: Int = 4,
            speaker: String? = null,
            instruction: String? = null,
        ): Ggtts = Ggtts(Native.loadWith(engine, model, codec, voice, threads, speaker, instruction))

        /**
         * The five-parameter `load` from before the voice settings, kept so code compiled against it (including its
         * default-argument form) still links and loads exactly as it did. Hidden from source.
         */
        @Deprecated("Kept for binary compatibility; call load with speaker and instruction", level = DeprecationLevel.HIDDEN)
        fun load(engine: String, model: String? = null, codec: String? = null, voice: String? = null, threads: Int = 4): Ggtts =
            Ggtts(Native.loadWith(engine, model, codec, voice, threads, null, null))
    }

    /**
     * Render [text] (one paragraph per line) to levelled PCM. [seed] is the salt of the story seed every chunk's sampler
     * seed is derived from (`render.seed_salt`). Not given, it is [DEFAULT_SEED] without [options], and with them the
     * document's salt, empty when nothing gives one, as the C-ABI's `ggtts_render_opts` renders it. [options] is
     * a render document, JSON, as the C-ABI's `ggtts_render_opts` takes it, laid over the load's `render` defaults:
     * steering, an edge, redraws, an instruction. A load option in it, `output.stream` (stream with [renderStream]),
     * the trace and any option the engine or this binding does not take throw GGTTS_ERR_ARG with the option's
     * [GgttsException.key]; so does a [seed] beside `render.seed_salt`.
     */
    fun render(text: String, seed: Long? = null, options: String? = null): Pcm {
        val rate = IntArray(1)
        val samples = if (options == null) {
            Native.render(handle, text, seed ?: DEFAULT_SEED, rate)
        } else {
            Native.renderOpts(handle, text, options, seed ?: 0L, seed != null, rate)
        }
        return Pcm(samples, rate[0])
    }

    /** The `render` from before the options document, kept so compiled code still links. Hidden from source. */
    @Deprecated("Kept for binary compatibility; call render with options", level = DeprecationLevel.HIDDEN)
    fun render(text: String, seed: Long = DEFAULT_SEED): Pcm = render(text, seed, null)

    /**
     * Render [text], handing each kept chunk's piece to [listener] as it is produced (before the level step).
     * Returns false if the listener stopped it. [seed] and [options] are as [render]'s; a stream's document may say
     * `"output": {"stream": "raw"}`, which is what it streams anyway.
     */
    fun renderStream(text: String, seed: Long? = null, options: String? = null, listener: PieceListener): Boolean =
        if (options == null) {
            Native.renderStream(handle, text, seed ?: DEFAULT_SEED, listener)
        } else {
            Native.renderStreamOpts(handle, text, options, seed ?: 0L, seed != null, listener)
        }

    /** The `renderStream` from before the options document, kept so compiled code still links. Hidden from source. */
    @Deprecated("Kept for binary compatibility; call renderStream with options", level = DeprecationLevel.HIDDEN)
    fun renderStream(text: String, seed: Long = DEFAULT_SEED, listener: PieceListener): Boolean =
        renderStream(text, seed, null, listener)

    /**
     * What this engine supports, as JSON, as the library states it (the C-ABI's `ggtts_capabilities`, format
     * `ggtts-capabilities/1`): `family`, `instruction` (and `no_instruction_why`), `steering`, `speakers` and
     * `speaker_names` (the model's speakers), `devices`, `hooks` and `step_events`.
     */
    fun capabilities(): String = Native.capabilities(handle)

    override fun close() {
        if (handle != 0L) {
            val h = handle
            handle = 0L
            Native.free(h)
        }
    }
}
