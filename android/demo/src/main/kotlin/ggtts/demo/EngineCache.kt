// One loaded engine for the whole app: the demo's screen and the system TTS engine render through the same one, so
// a model is resident once, not once per user. Loading another engine closes the one held (the binding's close never
// waits: a render in progress finishes, then its model is released).
package ggtts.demo

class EngineCache(private val binding: TtsBinding, private val log: (String) -> Unit = {}) {
    private var loaded: LoadRequest? = null
    private var engine: TtsEngine? = null

    /** The engine for [request]: the one held if it was loaded with the same request, else a fresh load. */
    @Synchronized
    fun acquire(request: LoadRequest): TtsEngine {
        engine?.let { if (loaded == request) return it }
        release()
        val t0 = System.nanoTime()
        val e = binding.load(request)
        log("loaded ${request.engine} in ${(System.nanoTime() - t0) / 1_000_000} ms: model=${request.model} codec=${request.codec} voice=${request.voice} threads=${request.threads}")
        engine = e
        loaded = request
        return e
    }

    /** Whether [request]'s engine is the one held. */
    @Synchronized
    fun holds(request: LoadRequest): Boolean = engine != null && loaded == request

    /** Close the engine held, if any. */
    @Synchronized
    fun release() {
        engine?.let {
            log("released ${loaded?.engine}")
            it.close()
        }
        engine = null
        loaded = null
    }
}

/** The app's one cache, shared by the screen and the system engine (created on first use). */
object SharedEngines {
    @Volatile private var cache: EngineCache? = null

    fun get(create: () -> EngineCache): EngineCache =
        cache ?: synchronized(this) { cache ?: create().also { cache = it } }
}
