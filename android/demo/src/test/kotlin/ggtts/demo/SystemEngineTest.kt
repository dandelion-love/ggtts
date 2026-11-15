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
        /** Chunks rendered: Stop must end rendering, not only the audio. */
        var rendered = 0
        override fun render(text: String, seed: Long): Pcm = error("streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean {
            for (i in 0 until chunks) {
                during(i)
                rendered++
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
    fun theCountryCountsOnlyWhenItIsTheVoicesOwn() {
        val s = SynthesisSession({ Engine() }, voices)
        voices.save(qwen)
        voices.save(LoadRequest("kitten-tts-2", "/m/lm.gguf", "/m/s3gen.gguf", "/m/voice.json", null, 4))
        assertEquals(TtsCodes.LANG_COUNTRY_AVAILABLE, s.availability("yue", "HK"))
        assertEquals("Cantonese for mainland China is Cantonese, not the HK voice", TtsCodes.LANG_AVAILABLE, s.availability("yue", "CHN"))
        assertEquals(TtsCodes.LANG_AVAILABLE, s.availability("yue", ""))
        assertEquals(TtsCodes.LANG_COUNTRY_AVAILABLE, s.availability("eng", "USA"))
        assertEquals("British English is English, not the US voice", TtsCodes.LANG_AVAILABLE, s.availability("eng", "GBR"))
        assertEquals(TtsCodes.LANG_NOT_SUPPORTED, s.availability("zho", "CHN"))
    }

    @Test
    fun aVoiceIsValidOnlyWhenItsFilesAreKnown() {
        val hk = "yue-hk-ggtts-qwen3"
        assertFalse("listed nowhere yet, so not loadable", SystemVoices.usable(hk, voices))
        voices.save(qwen)
        assertTrue(SystemVoices.usable(hk, voices))
        assertFalse("the English voice's files are still unknown", SystemVoices.usable("en-us-ggtts-kitten", voices))
        assertFalse(SystemVoices.usable("no-such-voice", voices))
        assertFalse(SystemVoices.usable(null, voices))
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
        val engine = Engine(chunks = 5) { i -> if (i == 1) s.stop() }
        s = SynthesisSession({ engine }, voices)
        s.synthesize("yue", "HKG", "今日天氣好好。", Ggtts.DEFAULT_SEED, out)
        assertEquals("only the first chunk went out, and no done", listOf("start 24000", "audio"), out.events)
        assertEquals(2400 * 2, out.bytes)
        assertEquals("rendering ended at the chunk in progress", 2, engine.rendered)
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

    /** #282 AC2: a voice saved before the document, as the positional list, loads as the same request. */
    @Test
    fun aVoiceSavedBeforeTheDocumentLoadsAsTheSameRequest() {
        kv.m["voice.qwen3-tts"] = "/m/talker.gguf\n/m/codec.gguf\n\nhk_voice_343\n4"
        assertEquals(qwen, voices.load("qwen3-tts"))
        kv.m["voice.kitten-tts-2"] = "/m/lm.gguf\n/m/s3gen.gguf\n/m/voice.json\n\n8"
        assertEquals(
            LoadRequest("kitten-tts-2", "/m/lm.gguf", "/m/s3gen.gguf", "/m/voice.json", null, 8),
            voices.load("kitten-tts-2"),
        )
        // Saved again, it is stored as its document and is still the same request.
        voices.save(qwen)
        assertTrue(kv.m["voice.qwen3-tts"]!!.startsWith("{"))
        assertEquals(qwen, voices.load("qwen3-tts"))
    }

    /** The document carries what the screen never set: an instruction saved with a voice is loaded with it. */
    @Test
    fun aSavedVoiceKeepsEveryKeyOfItsDocument() {
        val styled = qwen.with(LoadRequest.RENDER, "instruction", "Speak slowly.")
            .with(LoadRequest.OUTPUT, "stream", "levelled")
        voices.save(styled)
        val loaded = voices.load("qwen3-tts")!!
        assertEquals(styled, loaded)
        assertEquals("Speak slowly.", loaded.options[LoadRequest.RENDER, "instruction"])
        val loads = mutableListOf<LoadRequest>()
        SynthesisSession({ loads += it; Engine() }, voices).synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, Out())
        assertEquals("the system engine loads the saved document as it is", listOf(styled), loads)
    }

    /**
     * #282 AC4: a saved document with a key the binding does not take is refused with the binding's error, and the
     * voice is reported unusable (not listed, not available, and the read says why), not silently dropped.
     */
    @Test
    fun aSavedDocumentTheBindingRefusesMakesTheVoiceUnusable() {
        kv.m["voice.qwen3-tts"] = """{"load": {"engine": "qwen3-tts", "model": "/m/talker.gguf", "nonesuch": 1}}"""
        val refusal = GgttsException(Ggtts.ERR_ARG, "load.nonesuch: unknown option", "load.nonesuch")
        val logs = mutableListOf<String>()
        val s = SynthesisSession({ throw refusal }, voices, log = { logs += it })
        assertEquals("usable until the binding has seen it", TtsCodes.LANG_COUNTRY_AVAILABLE, s.availability("yue", "HKG"))
        val out = Out()
        s.synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, out)
        assertEquals(listOf("error ${TtsCodes.ERROR_INVALID_REQUEST}"), out.events)
        assertTrue(logs.toString(), logs.any { "unusable" in it && "load.nonesuch" in it })
        assertEquals("load.nonesuch: unknown option", voices.unusable("qwen3-tts"))
        assertFalse(SystemVoices.usable("yue-hk-ggtts-qwen3", voices))
        assertEquals(TtsCodes.LANG_MISSING_DATA, s.availability("yue", "HKG"))
        val again = Out()
        s.synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, again)
        assertEquals(listOf("error ${TtsCodes.ERROR_INVALID_REQUEST}"), again.events)
        assertTrue(logs.last(), "unusable" in logs.last() && "load.nonesuch" in logs.last())
        // A new Speak in the demo saves the voice again and clears the mark.
        voices.save(qwen)
        assertNull(voices.unusable("qwen3-tts"))
        assertTrue(SystemVoices.usable("yue-hk-ggtts-qwen3", voices))
    }

    /** A refusal at render (a render-phase key) marks the voice unusable the same way. */
    @Test
    fun aRenderRefusalByKeyMakesTheVoiceUnusable() {
        voices.save(qwen.with("observe", "log_level", "info"))
        val out = Out()
        val refusal = GgttsException(Ggtts.ERR_ARG, "observe.log_level: refused", "observe.log_level")
        SynthesisSession({ Engine { throw refusal } }, voices).synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, out)
        assertEquals(listOf("error ${TtsCodes.ERROR_INVALID_REQUEST}"), out.events)
        assertEquals("observe.log_level: refused", voices.unusable("qwen3-tts"))
    }

    /** A stored voice that is not a document at all is unusable with the reason, not a crash. */
    @Test
    fun aMalformedSavedDocumentIsUnusable() {
        kv.m["voice.qwen3-tts"] = """{"load": {"engine": """
        assertNull(voices.load("qwen3-tts"))
        assertTrue(voices.unusable("qwen3-tts")!!.startsWith("the saved voice is not an options document"))
        kv.m["voice.kitten-tts-2"] = """{"load": {"engine": "qwen3-tts"}}"""
        assertNull("a document for another engine", voices.load("kitten-tts-2"))
        assertTrue(voices.unusable("kitten-tts-2")!!.contains("names qwen3-tts"))
    }

    /** #282 AC5: a non-normal speech rate or pitch logs one line per read naming it, and the audio is the same. */
    @Test
    fun aSpeechRateOrPitchIsIgnoredOutLoudWithTheSameAudio() {
        voices.save(qwen)
        class Keep : TtsEngine {
            val pcm = mutableListOf<Float>()
            override fun render(text: String, seed: Long): Pcm = error("streams")
            override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean {
                val s = FloatArray(240) { ((it * 7 + seed) % 100).toFloat() / 100f }
                pcm += s.toList()
                return onPiece(s, 24000)
            }
            override fun close() {}
        }
        fun read(rate: Int, pitch: Int): Pair<List<Float>, List<String>> {
            val logs = mutableListOf<String>()
            val e = Keep()
            SynthesisSession({ e }, voices, log = { logs += it })
                .synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, Out(), rate = rate, pitch = pitch)
            return e.pcm to logs.filter { "ignored" in it }
        }
        val (normal, normalLogs) = read(100, 100)
        assertEquals("normal rate and pitch log nothing", emptyList<String>(), normalLogs)
        val (fast, fastLogs) = read(150, 100)
        assertEquals("the same audio", normal, fast)
        assertEquals(1, fastLogs.size)
        assertTrue(fastLogs[0], "ignored speech rate 150 (" in fastLogs[0])
        val (low, bothLogs) = read(80, 120)
        assertEquals("the same audio", normal, low)
        assertEquals(1, bothLogs.size)
        assertTrue(bothLogs[0], "ignored speech rate 80 and pitch 120" in bothLogs[0])
    }

    @Test
    fun pcm16IsLittleEndianAndClamped() {
        val b = SynthesisSession.pcm16(floatArrayOf(1f, -1f, 2f, 0f))
        assertEquals(listOf(0xff, 0x7f, 0x01, 0x80, 0xff, 0x7f, 0x00, 0x00), b.map { it.toInt() and 0xff })
    }
}
