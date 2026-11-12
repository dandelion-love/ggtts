// The system engine's logic on the host JVM, with a fake binding: which engine speaks a language, the model files it
// remembers, the stream into the platform callback (start, then audio in pieces no larger than the callback takes,
// then done), Stop, and the codes Android gets for each failure.
package ggtts.demo

import ggtts.Ggtts
import ggtts.GgttsException
import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SystemEngineTest {
    private class MapKv : KeyValue {
        val m = mutableMapOf<String, String>()
        override fun get(key: String) = m[key]
        override fun put(key: String, value: String) { m[key] = value }
    }

    private class Out(override val maxBytes: Int = 1000) : SynthesisOut {
        val events = mutableListOf<String>()
        var bytes = 0
        override fun start(sampleRate: Int) { events += "start $sampleRate" }
        override fun audio(pcm16: ByteArray, offset: Int, length: Int) {
            assertTrue("a piece fits the callback's buffer", length <= maxBytes)
            bytes += length
            if (events.lastOrNull()?.startsWith("audio") != true) events += "audio"
        }
        override fun done() { events += "done" }
        override fun error(code: Int) { events += "error $code" }
    }

    private class Engine(val chunks: Int = 2, val during: (Int) -> Unit = {}) : TtsEngine {
        override fun render(text: String, seed: Long): Pcm = error("streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean {
            for (i in 0 until chunks) {
                during(i)
                if (!onPiece(FloatArray(2400) { 0.5f }, 24000)) return false
            }
            return true
        }
        override fun close() {}
    }

    private val kv = MapKv()
    private val voices = VoiceStore(kv)
    private val qwen = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343", 4)

    @Test
    fun languagesMapToEngines() {
        assertEquals("qwen3-tts", Languages.engineFor("yue", "HKG"))
        assertEquals("qwen3-tts", Languages.engineFor("yue", ""))
        assertEquals("qwen3-tts", Languages.engineFor("zho", "HKG"))
        assertNull("Mandarin is not offered", Languages.engineFor("zho", "CHN"))
        assertEquals("kitten-tts-2", Languages.engineFor("eng", "USA"))
        assertNull(Languages.engineFor("fra", "FRA"))
    }

    @Test
    fun theVoiceStoreRoundTrips() {
        voices.save(qwen)
        assertEquals(qwen, voices.load("qwen3-tts"))
        assertNull(voices.load("kitten-tts-2"))
    }

    @Test
    fun availabilityNeedsTheModelFiles() {
        val s = SynthesisSession({ Engine() }, voices)
        assertEquals(TtsCodes.LANG_MISSING_DATA, s.availability("yue", "HKG"))
        voices.save(qwen)
        assertEquals(TtsCodes.LANG_COUNTRY_AVAILABLE, s.availability("yue", "HKG"))
        assertEquals(TtsCodes.LANG_NOT_SUPPORTED, s.availability("fra", ""))
    }

    @Test
    fun aSynthesisStreamsStartAudioDone() {
        voices.save(qwen)
        val out = Out(maxBytes = 1000)
        SynthesisSession({ Engine(chunks = 2) }, voices).synthesize("yue", "HKG", "今日天氣好好。", Ggtts.DEFAULT_SEED, out)
        assertEquals(listOf("start 24000", "audio", "done"), out.events)
        assertEquals("two chunks of 2400 samples, 16-bit", 2 * 2400 * 2, out.bytes)
    }

    @Test
    fun stopEndsTheSynthesisWithoutDone() {
        voices.save(qwen)
        val out = Out()
        lateinit var s: SynthesisSession
        s = SynthesisSession({ Engine(chunks = 5) { i -> if (i == 1) s.stop() } }, voices)
        s.synthesize("yue", "HKG", "今日天氣好好。", Ggtts.DEFAULT_SEED, out)
        assertEquals("only the first chunk went out, and no done", listOf("start 24000", "audio"), out.events)
        assertEquals(2400 * 2, out.bytes)
    }

    @Test
    fun noModelFilesIsNotInstalledYet() {
        val out = Out()
        SynthesisSession({ Engine() }, voices).synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, out)
        assertEquals(listOf("error ${TtsCodes.ERROR_NOT_INSTALLED_YET}"), out.events)
    }

    @Test
    fun aFailedLoadIsNotInstalledYetAndAFailedRenderIsASynthesisError() {
        voices.save(qwen)
        val loadFails = Out()
        SynthesisSession({ throw GgttsException(Ggtts.ERR_ENGINE, "model missing") }, voices)
            .synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, loadFails)
        assertEquals(listOf("error ${TtsCodes.ERROR_NOT_INSTALLED_YET}"), loadFails.events)

        val renderFails = Out()
        SynthesisSession({ Engine { throw GgttsException(Ggtts.ERR_ARG, "nothing to say") } }, voices)
            .synthesize("yue", "HKG", "？", Ggtts.DEFAULT_SEED, renderFails)
        assertEquals(listOf("error ${TtsCodes.ERROR_SYNTHESIS}"), renderFails.events)
    }

    @Test
    fun anUnsupportedLanguageIsAnInvalidRequest() {
        val out = Out()
        SynthesisSession({ Engine() }, voices).synthesize("fra", "FRA", "Bonjour", Ggtts.DEFAULT_SEED, out)
        assertEquals(listOf("error ${TtsCodes.ERROR_INVALID_REQUEST}"), out.events)
    }

    @Test
    fun theDemoRemembersTheFilesOfAFinishedRead() {
        val saved = mutableListOf<LoadRequest>()
        val view = object : DemoView {
            override fun setBusy(busy: Boolean) {}
            override fun showStatus(text: String) {}
            override fun showError(text: String) {}
        }
        val audio = object : AudioOut {
            override fun begin(rate: Int) {}
            override fun play(samples: FloatArray) {}
            override fun finish() {}
            override fun stop() {}
            override fun save(samples: FloatArray, rate: Int, engine: String) = "/x.wav"
        }
        val c = DemoController({ Engine() }, view, audio, { it.run() }, { it.run() }, onSpoke = { saved += it })
        c.speak(qwen, "今日天氣好好。", Ggtts.DEFAULT_SEED)
        assertEquals(listOf(qwen), saved)
        val failing = DemoController({ throw GgttsException(Ggtts.ERR_ENGINE, "no") }, view, audio, { it.run() },
            { it.run() }, onSpoke = { saved += it })
        failing.speak(qwen, "x", Ggtts.DEFAULT_SEED)
        assertFalse("a failed read is not remembered", saved.size > 1)
    }

    @Test
    fun pcm16IsLittleEndianAndClamped() {
        val b = SynthesisSession.pcm16(floatArrayOf(1f, -1f, 2f, 0f))
        assertEquals(listOf(0xff, 0x7f, 0x01, 0x80, 0xff, 0x7f, 0x00, 0x00), b.map { it.toInt() and 0xff })
    }
}
