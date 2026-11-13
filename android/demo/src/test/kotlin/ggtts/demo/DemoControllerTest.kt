// The demo's controller on the host JVM, with a fake binding in place of the native library: streaming playback (the
// first chunk plays before the read is rendered), Stop, and the error path (a missing or wrong model file shows the
// binding's message and the screen keeps working). This tests the app, not the native library; the real library's
// errors are checked on a device (android/README.md).
package ggtts.demo

import ggtts.Ggtts
import ggtts.GgttsException
import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoControllerTest {
    /** Everything that happened, in order, across the fakes. */
    private val events = mutableListOf<String>()

    private inner class FakeView : DemoView {
        val statuses = mutableListOf<String>()
        val errors = mutableListOf<String>()
        var shownBusy = false
        override fun setBusy(busy: Boolean) { shownBusy = busy }
        override fun showStatus(text: String) { statuses += text }
        override fun showError(text: String) { errors += text }
    }

    private inner class FakeAudio : AudioOut {
        val played = mutableListOf<FloatArray>()
        var saved: FloatArray? = null
        var stops = 0
        /** Runs inside [finish], while the last chunks play out. */
        var duringFinish: () -> Unit = {}
        override fun begin(rate: Int) { events += "begin $rate" }
        override fun play(samples: FloatArray) { played += samples; events += "play ${samples.size}" }
        override fun finish() { events += "finish"; duringFinish() }
        override fun stop() { stops++; events += "stop" }
        override fun save(samples: FloatArray, rate: Int, engine: String): String {
            saved = samples
            return "/fake/$engine.wav"
        }
    }

    /** Renders [chunks] pieces of 24000 samples (1 s each), calling [during] before each piece. */
    private inner class FakeEngine(val chunks: Int = 3, val during: (Int) -> Unit = {}) : TtsEngine {
        override fun render(text: String, seed: Long): Pcm = error("the controller streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean {
            if (text.isBlank()) throw GgttsException(Ggtts.ERR_ARG, "the text has nothing to say")
            for (i in 0 until chunks) {
                during(i)
                events += "render chunk $i"
                if (!onPiece(FloatArray(24000) { i.toFloat() }, 24000)) return false
            }
            events += "render end"
            return true
        }
        override fun close() {}
    }

    /** Throws [failures] in order, one per load, then loads [engine]. */
    private inner class FakeBinding(vararg failures: GgttsException, val engine: () -> TtsEngine = { FakeEngine() }) :
        TtsBinding {
        val pending = ArrayDeque(failures.toList())
        val loads = mutableListOf<LoadRequest>()
        override fun load(request: LoadRequest): TtsEngine {
            loads += request
            pending.removeFirstOrNull()?.let { throw it }
            return engine()
        }
    }

    private val view = FakeView()
    private val audio = FakeAudio()
    private val request = LoadRequest("kitten-tts-2", "/m/lm.gguf", "/m/s3gen.gguf", "/m/voice.json")

    /** Runs background and main work inline, so each speak has finished when it returns. */
    private fun controller(binding: TtsBinding) =
        DemoController(binding, view, audio, background = { it.run() }, main = { it.run() })

    @Test
    fun theFirstChunkPlaysBeforeTheReadIsRendered() {
        controller(FakeBinding()).speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(
            listOf("render chunk 0", "begin 24000", "play 24000", "render chunk 1", "play 24000", "render chunk 2",
                "play 24000", "render end", "finish"),
            events,
        )
        assertEquals("the whole read is saved", 72000, audio.saved!!.size)
        assertTrue(view.statuses.last(), view.statuses.last().startsWith("kitten-tts-2: 3.00 s of audio"))
        assertTrue(view.statuses.last(), view.statuses.last().contains("first audio after"))
        assertTrue("progress was shown", view.statuses.any { it.startsWith("Speaking: chunk 2 ready") })
    }

    @Test
    fun stopEndsPlaybackAtOnceAndRenderingAtTheNextChunk() {
        lateinit var c: DemoController
        // Stop is pressed while chunk 1 renders.
        c = controller(FakeBinding(engine = { FakeEngine(chunks = 5) { i -> if (i == 1) c.stop() } }))
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertTrue("playback stopped", audio.stops >= 1)
        assertEquals("only chunk 0 was played", 1, audio.played.size)
        assertFalse("rendering did not go on", events.contains("render chunk 2"))
        assertFalse("nothing is saved for a stopped read", audio.saved != null)
        assertEquals("Stopped after 1 chunk(s).", view.statuses.last())
        assertFalse(c.busy)
        assertTrue(view.errors.isEmpty())
    }

    @Test
    fun aStopWhileTheLastChunksPlayReportsStoppedNotFinished() {
        val c = controller(FakeBinding())
        audio.duringFinish = { c.stop() }
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertTrue("playback stopped", audio.stops >= 1)
        assertTrue(view.statuses.last(), view.statuses.last().startsWith("Stopped during playback"))
        assertEquals("the whole read was rendered, so it is saved", 72000, audio.saved!!.size)
        assertFalse(c.busy)
    }

    @Test
    fun stopWhenIdleDoesNothing() {
        val c = controller(FakeBinding())
        c.stop()
        assertEquals(0, audio.stops)
        assertTrue(view.statuses.isEmpty())
    }

    @Test
    fun aMissingModelShowsTheBindingsMessageAndTheScreenKeepsWorking() {
        val missing = GgttsException(Ggtts.ERR_ENGINE, "cannot open /m/lm.gguf: No such file or directory")
        val binding = FakeBinding(missing)
        val c = controller(binding)

        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(listOf("ggtts error -3: cannot open /m/lm.gguf: No such file or directory"), view.errors)
        assertFalse("the screen is usable again", view.shownBusy)
        assertFalse(c.busy)
        assertTrue("nothing played", audio.played.isEmpty())

        // The next try, with the file in place, renders and plays.
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(1, view.errors.size)
        assertEquals(3, audio.played.size)
        assertEquals(2, binding.loads.size)
    }

    @Test
    fun aFileThatIsNotAGgufShowsTheBindingsMessage() {
        val c = controller(FakeBinding(GgttsException(Ggtts.ERR_ENGINE, "/m/voice.json is not a GGUF file")))
        c.speak(request.copy(model = "/m/voice.json"), "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(listOf("ggtts error -3: /m/voice.json is not a GGUF file"), view.errors)
        assertFalse(view.shownBusy)
    }

    @Test
    fun aRenderErrorStopsPlaybackDropsTheEngineAndTheNextLoadIsFresh() {
        val binding = FakeBinding()
        val c = controller(binding)
        c.speak(request, "   ", Ggtts.DEFAULT_SEED)
        assertEquals(listOf("ggtts error -1: the text has nothing to say"), view.errors)
        assertEquals(1, audio.stops)
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals("a failed render drops the engine, so it is loaded again", 2, binding.loads.size)
        assertEquals(3, audio.played.size)
    }

    @Test
    fun anErrorMidReadStopsWhatIsPlaying() {
        val c = controller(FakeBinding(engine = {
            FakeEngine(chunks = 3) { i -> if (i == 2) throw GgttsException(Ggtts.ERR_ENGINE, "decode failed") }
        }))
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(listOf("ggtts error -3: decode failed"), view.errors)
        assertEquals("the two chunks played are cut off", "stop", events.last())
        assertFalse(c.busy)
    }

    @Test
    fun anUnexpectedExceptionIsShownNotThrown() {
        val c = controller { throw IllegalStateException("out of memory mapping the model") }
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals(listOf("IllegalStateException: out of memory mapping the model"), view.errors)
        assertFalse(c.busy)
    }

    @Test
    fun aSecondSpeakWhileOneRunsIsRefusedNotQueued() {
        val queued = mutableListOf<Runnable>()
        val binding = FakeBinding()
        val c = DemoController(binding, view, audio, background = { queued += it }, main = { it.run() })
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertTrue(c.busy)
        c.speak(request, "Hello again.", Ggtts.DEFAULT_SEED)
        assertEquals(1, queued.size)
        assertEquals("Still speaking; press Stop first.", view.statuses.last())
        queued.single().run()
        assertFalse(c.busy)
        assertEquals(3, audio.played.size)
    }

    @Test
    fun theSameFilesReuseTheLoadedEngineAndOtherThreadsReload() {
        val binding = FakeBinding()
        val c = controller(binding)
        c.speak(request, "One.", Ggtts.DEFAULT_SEED)
        c.speak(request, "Two.", Ggtts.DEFAULT_SEED)
        assertEquals(1, binding.loads.size)
        c.speak(request.copy(threads = 8), "Three.", Ggtts.DEFAULT_SEED)
        assertEquals(2, binding.loads.size)
        c.speak(request.copy(engine = "qwen3-tts", voice = null), "Four.", Ggtts.DEFAULT_SEED)
        assertEquals(3, binding.loads.size)
    }
}
