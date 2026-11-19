// The small interface the demo codes against, so a test can swap the native binding for a fake one. Plain Kotlin over
// the binding (no Android), so scripts/kotlin_binding_test.sh runs the real GgttsBinding on the host too.
package ggtts.demo

import ggtts.Ggtts
import ggtts.Pcm

/**
 * Which engine to load, from which files, and with what: one options document ([Options]), as the library's schema
 * names its keys. The screen's choices (the files, the speaker, the threads) are values in it; anything else the
 * binding takes (a style instruction, steering, an edge, redraws, the stream's level step) is a key in the same
 * document, carried to the binding unread. A file is a path on the device, or a content URI in the picked models
 * folder (Storage.kt); a key not given is "none picked".
 */
data class LoadRequest(val options: Options) {
    /** The screen's request: [engine] and the files picked; null is "none picked". */
    constructor(
        engine: String,
        model: String?,
        codec: String?,
        voice: String?,
        /** A Qwen3-TTS speaker the talker has, such as hk_voice_343; null for the model's default. */
        speaker: String? = null,
        threads: Int = DEFAULT_THREADS,
    ) : this(
        Options.EMPTY
            .with(LOAD, "engine", engine)
            .with(LOAD, "model", model)
            .with(LOAD, "codec", codec)
            .with(LOAD, "voice", voice)
            .with(LOAD, "speaker", speaker)
            .with(LOAD, "threads", threads),
    )

    val engine: String get() = options[LOAD, "engine"] as? String ?: "qwen3-tts"
    val model: String? get() = options[LOAD, "model"] as? String
    val codec: String? get() = options[LOAD, "codec"] as? String
    val voice: String? get() = options[LOAD, "voice"] as? String
    val speaker: String? get() = options[LOAD, "speaker"] as? String
    val threads: Int get() = (options[LOAD, "threads"] as? Long)?.toInt() ?: DEFAULT_THREADS

    /** This request with [section].[key] set to [value] (null removes it). */
    fun with(section: String, key: String, value: Any?) = LoadRequest(options.with(section, key, value))

    /** This request on [threads] CPU threads: the screen's thread choice is a value in the document. */
    fun withThreads(threads: Int) = with(LOAD, "threads", threads)

    /**
     * The document `Ggtts.load` takes: the `load` section, with [cacheDir] as `load.cache_dir` unless the document
     * gives one, and the `render` section as the defaults every render starts from.
     */
    fun loadDocument(cacheDir: String?): String {
        var d = options.only(LOAD, RENDER)
        if (cacheDir != null && d[LOAD, "cache_dir"] == null) d = d.with(LOAD, "cache_dir", cacheDir)
        return d.toJson()
    }

    /**
     * The document a whole render takes: every section but `load`, `render` (given at load) and `output` (a whole
     * render writes no stream). What the binding does not take, such as the trace or a log level, it refuses by key.
     */
    fun renderDocument(): String = options.without(LOAD, RENDER, OUTPUT).toJson()

    /**
     * The document a stream takes: as [renderDocument], with the `output` section. With no `output` section the app
     * streams the pieces before the level step (`"stream": "raw"`), as it did before the document, so the first audio
     * plays as soon as the first chunk is ready. A document with an `output` section streams as that section says,
     * for example levelled (`"stream": "levelled"`).
     */
    fun streamDocument(): String {
        val d = options.without(LOAD, RENDER)
        return (if (d.has(OUTPUT)) d else d.with(OUTPUT, "stream", "raw")).toJson()
    }

    override fun toString() = options.toJson()

    companion object {
        const val LOAD = "load"
        const val RENDER = "render"
        const val OUTPUT = "output"

        /** The binding's default thread count. See android/README.md for what more threads buy on a phone. */
        const val DEFAULT_THREADS = 4
    }
}

/** A path the engine can open (Storage.kt's ContentPaths makes one for a picked file), and what keeps it open (closed when the engine is). */
class Opened(val path: String, private val handle: AutoCloseable? = null) : AutoCloseable {
    override fun close() {
        handle?.close()
    }
}

/** Turns a model location into a path the engine opens. */
fun interface PathResolver {
    fun open(location: String): Opened

    companion object {
        /** Paths only: what the JVM tests and a binding with no picked folder use. */
        val PLAIN = PathResolver { Opened(it) }
    }
}

/** Loads an engine. A failure is a ggtts.GgttsException carrying the binding's code, message and key. */
fun interface TtsBinding {
    fun load(request: LoadRequest): TtsEngine
}

/** A loaded engine. */
interface TtsEngine : AutoCloseable {
    fun render(text: String, seed: Long): Pcm

    /**
     * Render [text], handing each chunk's samples to [onPiece] as soon as it is ready (before the level step, unless
     * the request's document asks for a levelled stream). Return false from [onPiece] to stop. Returns false if it
     * was stopped.
     */
    fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean
    val supportsEvidence: Boolean get() = false
    fun renderEvidence(text: String, seed: Long, onPiece: (Int, String, Long, Int, Int, Long, Long, FloatArray, Int) -> Boolean): Boolean = error("evidence unsupported")
}

/**
 * The real binding: ggtts.Ggtts over JNI, given the request's documents. [cacheDir] is where Qwen3-TTS looks for its
 * default model files when no model is picked. The demo never downloads them: `allowDownload = false` is given beside
 * the document, so a document that names `load.offline` is refused by the binding, and a missing default file is the
 * binding's error. [paths] turns each file's location into a path the engine opens: a file in the picked models
 * folder is a content URI, opened as `/proc/self/fd/<n>` and held open until the engine is closed (Storage.kt).
 */
class GgttsBinding(private val cacheDir: String?, private val paths: PathResolver = PathResolver.PLAIN) : TtsBinding {
    override fun load(request: LoadRequest): TtsEngine {
        val held = mutableListOf<Opened>()
        val tts = try {
            var resolved = request
            for (key in listOf("model", "codec", "voice")) {
                val location = request.options[LoadRequest.LOAD, key] as? String ?: continue
                val opened = paths.open(location).also { held += it }
                resolved = resolved.with(LoadRequest.LOAD, key, opened.path)
            }
            Ggtts.load(allowDownload = false, options = resolved.loadDocument(cacheDir))
        } catch (t: Throwable) {
            held.forEach { runCatching { it.close() } }
            throw t
        }
        val whole = request.renderDocument()
        val stream = request.streamDocument()
        return object : TtsEngine {
            override fun render(text: String, seed: Long): Pcm = tts.render(text, seed, options = whole)
            override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
                tts.renderStream(text, seed, options = stream) { samples, rate -> onPiece(samples, rate) }
            override val supportsEvidence: Boolean get() = true
            override fun renderEvidence(text: String, seed: Long, onPiece: (Int, String, Long, Int, Int, Long, Long, FloatArray, Int) -> Boolean): Boolean =
                tts.renderStreamEvidence(text, seed, options = stream) { i, t, s, d, stop, start, speechEnd, samples, rate ->
                    onPiece(i, t, s, d, stop, start, speechEnd, samples, rate)
                }
            override fun close() {
                try { tts.close() } finally { held.forEach { runCatching { it.close() } } }
            }
        }
    }
}
