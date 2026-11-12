// The small interface the demo codes against, so a test can swap the native binding for a fake one.
package ggtts.demo

import ggtts.Ggtts
import ggtts.Pcm

/** Which engine to load, from which files. Paths are files on the device; null is "none picked". */
data class LoadRequest(
    val engine: String,
    val model: String?,
    val codec: String?,
    val voice: String?,
    /** A Qwen3-TTS speaker the talker has, such as hk_voice_343; null for the model's default. */
    val speaker: String? = null,
    val threads: Int = 4,
)

/** Loads an engine. A failure is a ggtts.GgttsException carrying the binding's code and message. */
fun interface TtsBinding {
    fun load(request: LoadRequest): TtsEngine
}

/** A loaded engine. */
interface TtsEngine : AutoCloseable {
    fun render(text: String, seed: Long): Pcm

    /**
     * Render [text], handing each chunk's samples to [onPiece] as soon as it is ready (before the level step).
     * Return false from [onPiece] to stop. Returns false if it was stopped.
     */
    fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean
}

/**
 * The real binding: ggtts.Ggtts over JNI. [cacheDir] is where Qwen3-TTS looks for its default model files when no
 * model is picked; the demo never downloads them (allowDownload = false), so a missing one is the binding's error.
 */
class GgttsBinding(private val cacheDir: String) : TtsBinding {
    override fun load(request: LoadRequest): TtsEngine {
        val tts = Ggtts.load(
            request.engine,
            model = request.model,
            codec = request.codec,
            voice = request.voice,
            threads = request.threads,
            speaker = request.speaker,
            cacheDir = cacheDir,
            allowDownload = false,
        )
        return object : TtsEngine {
            override fun render(text: String, seed: Long): Pcm = tts.render(text, seed)
            override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
                tts.renderStream(text, seed) { samples, rate -> onPiece(samples, rate) }
            override fun close() = tts.close()
        }
    }
}
