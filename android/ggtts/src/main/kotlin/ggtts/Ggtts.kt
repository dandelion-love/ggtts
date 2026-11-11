// The Kotlin binding over the ggtts C-ABI (docs/android/binding-api.md). Every failure is a GgttsException; nothing
// here can crash the process on a bad argument, a missing file or a closed instance.
package ggtts

/** A ggtts error: [code] is the C-ABI's status (GGTTS_ERR_*), [message] its explanation. */
class GgttsException(val code: Int, message: String) : RuntimeException(message)

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
    @JvmStatic external fun render(handle: Long, text: String, seed: Long, rateOut: IntArray): FloatArray
    @JvmStatic external fun renderStream(handle: Long, text: String, seed: Long, listener: PieceListener): Boolean
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
         * [engine]: "qwen3-tts", "kitten-tts-2" or "fake". [voice] is Kitten-TTS 2's reference voice.
         *
         * [speaker] and [instruction] are Qwen3-TTS settings; null or empty is no setting, and the result is the same
         * as without them. [speaker] is a speaker the model has (for example "hk_voice_777"); an unknown one throws
         * GGTTS_ERR_ENGINE with the model's speakers in the message. [instruction] is a style instruction for every
         * render (for example "Speak slowly and softly."); a checkpoint that takes none (Base, or custom_voice below
         * 1.7B) throws GGTTS_ERR_ENGINE. Another engine refuses either one with GGTTS_ERR_ARG.
         */
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
         * default-argument form) still links. Hidden from source, where the `load` above covers every call.
         */
        @Deprecated("Kept for binary compatibility; call load with speaker and instruction", level = DeprecationLevel.HIDDEN)
        fun load(engine: String, model: String? = null, codec: String? = null, voice: String? = null, threads: Int = 4): Ggtts =
            load(engine, model, codec, voice, threads, null, null)
    }

    /** Render [text] (one paragraph per line) to levelled PCM. */
    fun render(text: String, seed: Long = DEFAULT_SEED): Pcm {
        val rate = IntArray(1)
        val samples = Native.render(handle, text, seed, rate)
        return Pcm(samples, rate[0])
    }

    /**
     * Render [text], handing each kept chunk's piece to [listener] as it is produced (before the level step).
     * Returns false if the listener stopped it.
     */
    fun renderStream(text: String, seed: Long = DEFAULT_SEED, listener: PieceListener): Boolean =
        Native.renderStream(handle, text, seed, listener)

    override fun close() {
        if (handle != 0L) {
            val h = handle
            handle = 0L
            Native.free(h)
        }
    }
}
