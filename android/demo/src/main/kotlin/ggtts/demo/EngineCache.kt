// One loaded engine for the whole app: the demo's screen and the system TTS engine render through the same one, so
// a model is resident once, not once per user. One caller reads with it at a time. While one does, another caller's
// acquire is refused (EngineInUse) instead of loading a second model beside it, evicting it mid-read, or rendering on
// the same handle at once; and memory trims and close leave it alone.
package ggtts.demo

/** [EngineCache.acquire] was refused: another caller (the screen or the system engine) is reading with the engine. */
class EngineInUse(engine: String) :
    RuntimeException("the $engine engine is busy with another read (this screen or Android's TTS); try again when it ends")

class EngineCache(private val binding: TtsBinding, private val log: (String) -> Unit = {}) {
    private var loaded: LoadRequest? = null
    private var engine: TtsEngine? = null
    private var inUse = false

    /**
     * The engine for [request], the caller's until it calls [done]: the one held if it was loaded with the same
     * request, else a fresh load (closing the one held). Throws [EngineInUse] while another caller has it. The load
     * runs outside the lock, so a trim on the main thread never waits for it.
     */
    fun acquire(request: LoadRequest): TtsEngine {
        synchronized(this) {
            if (inUse) throw EngineInUse(loaded?.engine ?: request.engine)
            inUse = true
            engine?.let { if (loaded == request) return it }
            closeHeld()
        }
        try {
            val t0 = System.nanoTime()
            val e = binding.load(request)
            log("loaded ${request.engine} in ${(System.nanoTime() - t0) / 1_000_000} ms: model=${request.model} codec=${request.codec} voice=${request.voice} threads=${request.threads}")
            synchronized(this) {
                engine = e
                loaded = request
            }
            return e
        } catch (t: Throwable) {
            synchronized(this) { inUse = false }
            throw t
        }
    }

    /** The caller is finished with the engine from [acquire]. [failed]: close it too, so the next read loads afresh. */
    @Synchronized
    fun done(failed: Boolean = false) {
        inUse = false
        if (failed) closeHeld()
    }

    /** Whether a caller is reading with the engine now. */
    val busy: Boolean
        @Synchronized get() = inUse

    /** Whether [request]'s engine is the one held. */
    @Synchronized
    fun holds(request: LoadRequest): Boolean = engine != null && loaded == request

    /** Close the engine held, unless a caller is reading with it. Returns whether it was closed (or none was held). */
    @Synchronized
    fun release(): Boolean {
        if (inUse) {
            log("release refused: ${loaded?.engine} is reading")
            return false
        }
        closeHeld()
        return true
    }

    private fun closeHeld() {
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
