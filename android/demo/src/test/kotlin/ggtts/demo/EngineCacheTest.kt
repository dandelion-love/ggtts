// One loaded engine for the screen and the system engine, kept between reads, and given back only when Android asks
// for memory while nothing is being read.
package ggtts.demo

import ggtts.Ggtts
import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EngineCacheTest {
    private class Engine : TtsEngine {
        var closed = false
        override fun render(text: String, seed: Long): Pcm = error("streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
            onPiece(FloatArray(100), 24000)
        override fun close() { closed = true }
    }

    private val engines = mutableListOf<Engine>()
    private val loads = mutableListOf<LoadRequest>()
    private val binding = TtsBinding { r -> loads += r; Engine().also { engines += it } }
    private val qwen = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343", 4)

    private val view = object : DemoView {
        override fun setBusy(busy: Boolean) {}
        override fun showStatus(text: String) {}
        override fun showError(text: String) {}
    }
    private val audio = object : AudioOut {
        override fun begin(rate: Int) {}
        override fun play(samples: FloatArray) {}
        override fun finish() {}
        override fun stop() {}
        override fun save(samples: FloatArray, rate: Int, engine: String) = "/x.wav"
    }
    private val out = object : SynthesisOut {
        override val maxBytes = 4096
        val events = mutableListOf<String>()
        override fun start(sampleRate: Int) { events += "start" }
        override fun audio(pcm16: ByteArray, offset: Int, length: Int) {}
        override fun done() { events += "done" }
        override fun error(code: Int) { events += "error $code" }
    }

    @Test
    fun theScreenAndTheSystemEngineShareOneLoad() {
        val cache = EngineCache(binding)
        val voices = VoiceStore(object : KeyValue {
            val m = mutableMapOf<String, String>()
            override fun get(key: String) = m[key]
            override fun put(key: String, value: String) { m[key] = value }
        })
        val screen = DemoController(binding, view, audio, { it.run() }, { it.run() },
            onSpoke = { voices.save(it) }, cache = cache)
        val system = SynthesisSession(binding, voices, cache = cache)

        screen.speak(qwen, "今日天氣好好。", Ggtts.DEFAULT_SEED)
        system.synthesize("yue", "HKG", "今日天氣好好。", Ggtts.DEFAULT_SEED, out)
        assertEquals("loaded once for both", 1, loads.size)
        assertEquals(listOf("start", "done"), out.events)

        // Another engine replaces it, and the first is closed.
        screen.speak(qwen.copy(threads = 8), "x", Ggtts.DEFAULT_SEED)
        assertEquals(2, loads.size)
        assertTrue(engines[0].closed)
        assertFalse(engines[1].closed)
    }

    @Test
    fun theEngineIsKeptBetweenReads() {
        val cache = EngineCache(binding)
        val c = DemoController(binding, view, audio, { it.run() }, { it.run() }, cache = cache)
        c.speak(qwen, "One.", Ggtts.DEFAULT_SEED)
        c.speak(qwen, "Two.", Ggtts.DEFAULT_SEED)
        assertEquals(1, loads.size)
        assertTrue(cache.holds(qwen))
    }

    @Test
    fun closeWhileReadingKeepsTheEngine() {
        val cache = EngineCache(binding)
        val queued = mutableListOf<Runnable>()
        val c = DemoController(binding, view, audio, { queued += it }, { it.run() }, cache = cache)
        cache.acquire(qwen)
        c.speak(qwen, "Hello.", Ggtts.DEFAULT_SEED)
        c.close()
        assertTrue("a read is running: the engine stays", cache.holds(qwen))
        queued.single().run()
        c.close()
        assertFalse("idle: close releases it", cache.holds(qwen))
    }

    @Test
    fun memoryIsGivenBackOnlyWhenIdleAndAskedHardEnough() {
        assertFalse("never during a read", TrimPolicy.shouldRelease(TrimPolicy.RUNNING_CRITICAL, busy = true))
        assertFalse(TrimPolicy.shouldRelease(80, busy = true))
        assertTrue(TrimPolicy.shouldRelease(TrimPolicy.RUNNING_CRITICAL, busy = false))
        assertTrue("the app went to the background", TrimPolicy.shouldRelease(TrimPolicy.BACKGROUND, busy = false))
        assertFalse("moderate pressure while in use keeps it", TrimPolicy.shouldRelease(5, busy = false))
        assertFalse("the screen was hidden: a read may follow", TrimPolicy.shouldRelease(20, busy = false))
    }
}
