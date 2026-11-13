// What the screen says an engine costs, and the free-memory check before a load: a warning first, the load on a
// second press, and no warning for an engine already loaded.
package ggtts.demo

import ggtts.Ggtts
import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class EngineCostsTest {
    @Test
    fun eachEngineSaysWhatItNeedsAndWhatItIsNotFor() {
        for (engine in listOf("qwen3-tts", "kitten-tts-2")) {
            val d = EngineCosts.describe(engine)
            assertTrue(d, d.contains("GB of model files") && d.contains("GB of memory"))
            assertTrue(d, d.contains(EngineCosts.MEASURED_ON))
            assertTrue(d, d.contains("Not suitable for screen readers"))
        }
        assertEquals("", EngineCosts.describe("fake"))
    }

    @Test
    fun theWarningComesBelowTheMeasuredPeakOrWhenTheSystemSaysLow() {
        assertNull(EngineCosts.memoryWarning("kitten-tts-2", 4_000_000_000, lowMemory = false))
        val w = EngineCosts.memoryWarning("kitten-tts-2", 2_000_000_000, lowMemory = false)!!
        assertTrue(w, w.startsWith("Low memory: kitten-tts-2 used up to 3.8 GB") && w.contains("2.0 GB is available"))
        // The peak is VmHWM's 3,693,672 KiB, 3.78 GB of 10^9 bytes, compared with available bytes: 3.75 GB is short.
        assertTrue("3.75 GB is below the 3.78 GB peak", EngineCosts.memoryWarning("kitten-tts-2", 3_750_000_000, lowMemory = false) != null)
        val low = EngineCosts.memoryWarning("qwen3-tts", 8_000_000_000, lowMemory = true)!!
        assertTrue(low, low.contains("the system reports memory is low"))
        assertNull("an engine with no figures is not checked", EngineCosts.memoryWarning("fake", 0, lowMemory = true))
    }

    private class View : DemoView {
        val errors = mutableListOf<String>()
        override fun setBusy(busy: Boolean) {}
        override fun showStatus(text: String) {}
        override fun showError(text: String) { errors += text }
    }

    private class Audio : AudioOut {
        override fun begin(rate: Int) {}
        override fun play(samples: FloatArray) {}
        override fun finish() {}
        override fun stop() {}
        override fun save(samples: FloatArray, rate: Int, engine: String) = "/fake.wav"
    }

    private class Engine : TtsEngine {
        override fun render(text: String, seed: Long): Pcm = error("streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
            onPiece(FloatArray(10), 24000)
        override fun close() {}
    }

    @Test
    fun shortMemoryWarnsFirstAndTheSamePressAgainLoads() {
        val loads = mutableListOf<LoadRequest>()
        val view = View()
        var available = 1_000_000_000L
        val c = DemoController({ loads += it; Engine() }, view, Audio(), { it.run() }, { it.run() },
            memory = { available to false })
        val kitten = LoadRequest("kitten-tts-2", "/m/lm.gguf", "/m/s3gen.gguf", "/m/voice.json")

        c.speak(kitten, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals("warned, not loaded", 0, loads.size)
        assertTrue(view.errors.single().startsWith("Low memory"))

        c.speak(kitten, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals("the second press loads", 1, loads.size)

        // Loaded now: speaking again does not warn, however short memory is.
        c.speak(kitten, "Again.", Ggtts.DEFAULT_SEED)
        assertEquals(1, view.errors.size)

        // A different request warns again before its own load.
        c.speak(kitten.copy(threads = 8), "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(2, view.errors.size)
        assertEquals(1, loads.size)

        // With room, no warning at all.
        available = 8_000_000_000L
        c.speak(kitten.copy(threads = 6), "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(2, view.errors.size)
        assertEquals(2, loads.size)
    }
}
