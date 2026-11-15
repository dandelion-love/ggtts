// ggtts as Android's system TTS engine, apart from Android: which engine speaks which language, which model files
// each uses (the ones its last successful Speak in the demo used), and one synthesis streamed to the platform's
// callback chunk by chunk. GgttsTtsService adapts this to TextToSpeechService; the tests drive it directly.
package ggtts.demo

import ggtts.Ggtts
import ggtts.GgttsException
import java.util.Locale

/** A small string store (SharedPreferences on a device, a map in tests). */
interface KeyValue {
    fun get(key: String): String?
    fun put(key: String, value: String)
}

/**
 * The request each engine last spoke with in the demo; the system engine uses the same. Stored as its options
 * document, so whatever the request carries (an instruction, steering, a stream mode) is saved with it. A voice saved
 * before the document, as the old positional list (model, codec, voice, speaker, threads on five lines), still loads as
 * the same request.
 *
 * A saved voice the binding refuses (an unknown key, say) is marked unusable with the binding's message, not dropped:
 * [load] then gives null, [unusable] says why, and the next [save] for that engine clears it.
 */
class VoiceStore(private val kv: KeyValue) {
    fun save(request: LoadRequest) {
        kv.put("voice." + request.engine, request.options.toJson())
        kv.put("voice.${request.engine}.unusable", "")
    }

    fun load(engine: String): LoadRequest? {
        if (unusable(engine) != null) return null
        val stored = kv.get("voice.$engine") ?: return null
        return try {
            read(engine, stored)
        } catch (e: IllegalArgumentException) {
            refuse(engine, "the saved voice is not an options document: ${e.message}")
            null
        }
    }

    /** Why [engine]'s saved voice is unusable, or null when it is not marked. */
    fun unusable(engine: String): String? = kv.get("voice.$engine.unusable")?.ifEmpty { null }

    /** Mark [engine]'s saved voice unusable, for [why] (the binding's message). */
    fun refuse(engine: String, why: String) = kv.put("voice.$engine.unusable", why)

    companion object {
        /** A stored voice of [engine]: an options document, or the positional list from before it. */
        fun read(engine: String, stored: String): LoadRequest? {
            if (stored.trimStart().startsWith("{")) {
                val r = LoadRequest(Options.parse(stored))
                return when (r.options[LoadRequest.LOAD, "engine"]) {
                    null -> r.with(LoadRequest.LOAD, "engine", engine)
                    engine -> r
                    else -> throw IllegalArgumentException("it is saved for $engine and names ${r.engine}")
                }
            }
            val parts = stored.split("\n")
            if (parts.size != 5) return null
            fun s(i: Int) = parts[i].ifEmpty { null }
            return LoadRequest(engine, s(0), s(1), s(2), s(3), parts[4].toIntOrNull() ?: LoadRequest.DEFAULT_THREADS)
        }
    }
}

/** Android's TextToSpeech codes the system engine reports (the platform's values). */
object TtsCodes {
    const val ERROR_SYNTHESIS = -3
    const val ERROR_SERVICE = -4
    const val ERROR_INVALID_REQUEST = -8
    const val ERROR_NOT_INSTALLED_YET = -9
    const val LANG_COUNTRY_AVAILABLE = 1
    const val LANG_AVAILABLE = 0
    const val LANG_MISSING_DATA = -1
    const val LANG_NOT_SUPPORTED = -2
}

/** Which engine speaks a language: the Hong Kong Cantonese voice for Cantonese, Kitten-TTS 2 for English. */
object Languages {
    /** [iso3] is a three-letter language code as Android passes it (yue, zho, eng); [country] may be empty. */
    fun engineFor(iso3: String, country: String): String? = when (iso3.lowercase()) {
        "yue" -> "qwen3-tts"
        "zho", "chi" -> if (country.uppercase() in setOf("HK", "HKG")) "qwen3-tts" else null
        "eng" -> "kitten-tts-2"
        else -> null
    }

    /** The country each engine's voice speaks for, in the two- and three-letter forms Android may pass. */
    fun countriesOf(engine: String): Set<String> = when (engine) {
        "qwen3-tts" -> setOf("HK", "HKG")
        "kitten-tts-2" -> setOf("US", "USA")
        else -> emptySet()
    }
}

/** The system engine's voices: name, locale, and the engine that speaks it. */
object SystemVoices {
    val ALL = listOf(
        Triple("yue-hk-ggtts-qwen3", Locale("yue", "HK"), "qwen3-tts"),
        Triple("en-us-ggtts-kitten", Locale.US, "kitten-tts-2"),
    )

    /** A voice is usable when its engine's model files are known: the rule onGetVoices lists voices by. */
    fun usable(name: String?, voices: VoiceStore): Boolean {
        val v = ALL.firstOrNull { it.first == name } ?: return false
        return voices.load(v.third) != null
    }
}

/** Where a synthesis goes: the platform's SynthesisCallback on a device. */
interface SynthesisOut {
    /** Most bytes one [audio] call may take. */
    val maxBytes: Int
    fun start(sampleRate: Int)
    fun audio(pcm16: ByteArray, offset: Int, length: Int)
    fun done()
    fun error(code: Int)
}

class SynthesisSession(
    private val binding: TtsBinding,
    private val voices: VoiceStore,
    private val log: (String) -> Unit = {},
    /** Where engines are loaded and held; the app passes the one it shares with the demo's screen. */
    private val cache: EngineCache = EngineCache(binding, log),
) {
    @Volatile
    private var stopped = false

    /**
     * Whether [iso3]/[country] is spoken, and whether its model files are known. The country counts only when it is
     * the voice's own: English asked for Britain is English, available, not British English.
     */
    fun availability(iso3: String, country: String): Int {
        val e = Languages.engineFor(iso3, country) ?: return TtsCodes.LANG_NOT_SUPPORTED
        if (voices.load(e) == null) return TtsCodes.LANG_MISSING_DATA
        return if (country.uppercase() in Languages.countriesOf(e)) TtsCodes.LANG_COUNTRY_AVAILABLE
        else TtsCodes.LANG_AVAILABLE
    }

    /** Called from another thread by the platform: end the synthesis at the next chunk. */
    fun stop() {
        stopped = true
    }

    /**
     * Synthesize [text] in [iso3]/[country] into [out], streaming each chunk as it is ready. Blocks.
     *
     * [rate] and [pitch] are the platform's speech rate and pitch, 100 being normal. ggtts has no rate or pitch
     * option, so both are refused: a read asked at another value is rendered as at 100, and says so in one log line.
     */
    fun synthesize(
        iso3: String,
        country: String,
        text: String,
        seed: Long,
        out: SynthesisOut,
        rate: Int = NORMAL,
        pitch: Int = NORMAL,
    ) {
        stopped = false
        ignored(rate, pitch)?.let(log)
        val name = Languages.engineFor(iso3, country) ?: return out.error(TtsCodes.ERROR_INVALID_REQUEST)
        val request = voices.load(name) ?: run {
            val why = voices.unusable(name)
            if (why != null) {
                log("system engine: the saved $name voice is unusable: $why; speak once with it in the demo")
                return out.error(TtsCodes.ERROR_INVALID_REQUEST)
            }
            log("system engine: no model files known for $name; speak once with it in the demo")
            return out.error(TtsCodes.ERROR_NOT_INSTALLED_YET)
        }
        val tts = try {
            cache.acquire(request)
        } catch (e: EngineInUse) {
            log("system engine: ${e.message}")
            return out.error(TtsCodes.ERROR_SERVICE)
        } catch (e: GgttsException) {
            if (refused(name, e)) return out.error(TtsCodes.ERROR_INVALID_REQUEST)
            log("system engine: load failed ${e.code}: ${e.message}")
            return out.error(TtsCodes.ERROR_NOT_INSTALLED_YET)
        } catch (e: RuntimeException) {
            log("system engine: load failed: $e")
            return out.error(TtsCodes.ERROR_NOT_INSTALLED_YET)
        }
        var ok = false
        try {
            ok = stream(name, tts, text, seed, out)
        } finally {
            cache.done(failed = !ok)
        }
    }

    /**
     * A refusal of the saved voice's document: the binding's argument error naming a key. The voice is marked
     * unusable with the binding's message and the read reports it, rather than the key being dropped.
     */
    private fun refused(engine: String, e: GgttsException): Boolean {
        val key = e.key
        if (e.code != Ggtts.ERR_ARG || key == null) return false
        voices.refuse(engine, e.message ?: key)
        log("system engine: the saved $engine voice is unusable: the binding refused $key: ${e.message}")
        return true
    }

    /** Stream one render into [out]. Returns false when the render failed (the engine is then closed). */
    private fun stream(engine: String, tts: TtsEngine, text: String, seed: Long, out: SynthesisOut): Boolean {
        var started = false
        val finished = try {
            tts.renderStream(text, seed) { samples, rate ->
                if (stopped) return@renderStream false
                if (!started) {
                    out.start(rate)
                    started = true
                }
                val bytes = pcm16(samples)
                var at = 0
                while (at < bytes.size && !stopped) {
                    val n = minOf(out.maxBytes, bytes.size - at)
                    out.audio(bytes, at, n)
                    at += n
                }
                true
            }
        } catch (e: GgttsException) {
            if (refused(engine, e)) {
                out.error(TtsCodes.ERROR_INVALID_REQUEST)
                return false
            }
            log("system engine: render failed ${e.code}: ${e.message}")
            out.error(TtsCodes.ERROR_SYNTHESIS)
            return false
        } catch (e: RuntimeException) {
            log("system engine: render failed: $e")
            out.error(TtsCodes.ERROR_SYNTHESIS)
            return false
        }
        if (!finished || stopped) {
            log("system engine: stopped")
            return true
        }
        out.done()
        return true
    }

    companion object {
        /** The platform's normal speech rate and pitch (SynthesisRequest.getSpeechRate and getPitch). */
        const val NORMAL = 100

        /** The log line for a read asked at a rate or pitch ggtts cannot give, or null when both are normal. */
        fun ignored(rate: Int, pitch: Int): String? {
            val asked = listOfNotNull(
                if (rate != NORMAL) "speech rate $rate" else null,
                if (pitch != NORMAL) "pitch $pitch" else null,
            )
            if (asked.isEmpty()) return null
            return "system engine: ignored ${asked.joinToString(" and ")} ($NORMAL is normal): " +
                "ggtts has no rate or pitch option, so this read is rendered at $NORMAL"
        }

        /** Float samples as 16-bit little-endian PCM, the format the session starts its stream in. */
        fun pcm16(samples: FloatArray): ByteArray {
            val b = ByteArray(samples.size * 2)
            for (i in samples.indices) {
                val v = (samples[i].coerceIn(-1f, 1f) * 32767f).toInt()
                b[2 * i] = (v and 0xff).toByte()
                b[2 * i + 1] = (v shr 8 and 0xff).toByte()
            }
            return b
        }
    }
}
