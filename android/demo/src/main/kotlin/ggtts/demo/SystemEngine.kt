// ggtts as Android's system TTS engine, apart from Android: which engine speaks which language, which model files
// each uses (the ones its last successful Speak in the demo used), and one synthesis streamed to the platform's
// callback chunk by chunk. GgttsTtsService adapts this to TextToSpeechService; the tests drive it directly.
package ggtts.demo

import ggtts.GgttsException

/** A small string store (SharedPreferences on a device, a map in tests). */
interface KeyValue {
    fun get(key: String): String?
    fun put(key: String, value: String)
}

/** The model files each engine last spoke with in the demo; the system engine uses the same. */
class VoiceStore(private val kv: KeyValue) {
    fun save(request: LoadRequest) {
        kv.put(
            "voice." + request.engine,
            listOf(request.model, request.codec, request.voice, request.speaker, request.threads.toString())
                .joinToString("\n") { it ?: "" },
        )
    }

    fun load(engine: String): LoadRequest? {
        val parts = kv.get("voice.$engine")?.split("\n") ?: return null
        if (parts.size != 5) return null
        fun s(i: Int) = parts[i].ifEmpty { null }
        return LoadRequest(engine, s(0), s(1), s(2), s(3), parts[4].toIntOrNull() ?: MainActivity.DEFAULT_THREADS)
    }
}

/** Android's TextToSpeech codes the system engine reports (the platform's values). */
object TtsCodes {
    const val ERROR_SYNTHESIS = -3
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
) {
    private var loaded: LoadRequest? = null
    private var engine: TtsEngine? = null

    @Volatile
    private var stopped = false

    /** Whether [iso3]/[country] is spoken, and whether its model files are known. */
    fun availability(iso3: String, country: String): Int {
        val e = Languages.engineFor(iso3, country) ?: return TtsCodes.LANG_NOT_SUPPORTED
        if (voices.load(e) == null) return TtsCodes.LANG_MISSING_DATA
        return if (country.isEmpty()) TtsCodes.LANG_AVAILABLE else TtsCodes.LANG_COUNTRY_AVAILABLE
    }

    /** Called from another thread by the platform: end the synthesis at the next chunk. */
    fun stop() {
        stopped = true
    }

    /** Synthesize [text] in [iso3]/[country] into [out], streaming each chunk as it is ready. Blocks. */
    fun synthesize(iso3: String, country: String, text: String, seed: Long, out: SynthesisOut) {
        stopped = false
        val name = Languages.engineFor(iso3, country) ?: return out.error(TtsCodes.ERROR_INVALID_REQUEST)
        val request = voices.load(name) ?: run {
            log("system engine: no model files known for $name; speak once with it in the demo")
            return out.error(TtsCodes.ERROR_NOT_INSTALLED_YET)
        }
        val tts = try {
            engineFor(request)
        } catch (e: GgttsException) {
            log("system engine: load failed ${e.code}: ${e.message}")
            return out.error(TtsCodes.ERROR_NOT_INSTALLED_YET)
        } catch (e: RuntimeException) {
            log("system engine: load failed: $e")
            return out.error(TtsCodes.ERROR_NOT_INSTALLED_YET)
        }
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
            log("system engine: render failed ${e.code}: ${e.message}")
            drop()
            return out.error(TtsCodes.ERROR_SYNTHESIS)
        } catch (e: RuntimeException) {
            log("system engine: render failed: $e")
            drop()
            return out.error(TtsCodes.ERROR_SYNTHESIS)
        }
        if (!finished || stopped) {
            log("system engine: stopped")
            return
        }
        out.done()
    }

    fun close() = drop()

    private fun engineFor(request: LoadRequest): TtsEngine {
        engine?.let { if (loaded == request) return it }
        drop()
        val e = binding.load(request)
        engine = e
        loaded = request
        return e
    }

    private fun drop() {
        engine?.close()
        engine = null
        loaded = null
    }

    companion object {
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
