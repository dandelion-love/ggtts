// The demo's error path on the host JVM, with a fake binding in place of the native library: a missing or wrong model
// file shows the binding's message and the screen keeps working. This tests the app, not the native library; the
// real library's errors are checked on a device (android/README.md).
package ggtts.demo

import ggtts.Ggtts
import ggtts.GgttsException
import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DemoControllerTest {
    private class FakeView : DemoView {
        val statuses = mutableListOf<String>()
        val errors = mutableListOf<String>()
        var shownBusy = false
        override fun setBusy(busy: Boolean) { shownBusy = busy }
        override fun showStatus(text: String) { statuses += text }
        override fun showError(text: String) { errors += text }
    }

    private class FakeAudio : AudioOut {
        val played = mutableListOf<Pcm>()
        override fun save(pcm: Pcm, engine: String) = "/fake/$engine.wav"
        override fun play(pcm: Pcm) { played += pcm }
    }

    private class FakeEngine : TtsEngine {
        var closed = false
        override fun render(text: String, seed: Long): Pcm {
            if (text.isBlank()) throw GgttsException(Ggtts.ERR_ARG, "the text has nothing to say")
            return Pcm(FloatArray(24000), 24000)
        }
        override fun close() { closed = true }
    }

    /** Throws [failures] in order, one per load, then loads a working engine. */
    private class FakeBinding(vararg failures: GgttsException) : TtsBinding {
        val pending = ArrayDeque(failures.toList())
        val loads = mutableListOf<LoadRequest>()
        override fun load(request: LoadRequest): TtsEngine {
            loads += request
            pending.removeFirstOrNull()?.let { throw it }
            return FakeEngine()
        }
    }

    private val view = FakeView()
    private val audio = FakeAudio()
    private val request = LoadRequest("kitten-tts-2", "/m/lm.gguf", "/m/s3gen.gguf", "/m/voice.json")

    /** Runs background and main work inline, so each speak has finished when it returns. */
    private fun controller(binding: TtsBinding) =
        DemoController(binding, view, audio, background = { it.run() }, main = { it.run() })

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
        assertEquals(1, audio.played.size)
        assertTrue(view.statuses.last(), view.statuses.last().startsWith("kitten-tts-2: 1.00 s of audio"))
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
    fun aRenderErrorDropsTheEngineAndTheNextLoadIsFresh() {
        val binding = FakeBinding()
        val c = controller(binding)
        c.speak(request, "   ", Ggtts.DEFAULT_SEED)
        assertEquals(listOf("ggtts error -1: the text has nothing to say"), view.errors)
        c.speak(request, "Hello.", Ggtts.DEFAULT_SEED)
        assertEquals("a failed render drops the engine, so it is loaded again", 2, binding.loads.size)
        assertEquals(1, audio.played.size)
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
        assertEquals("Still speaking; wait for it to finish.", view.statuses.last())
        queued.single().run()
        assertFalse(c.busy)
        assertEquals(1, audio.played.size)
    }

    @Test
    fun theSameFilesReuseTheLoadedEngine() {
        val binding = FakeBinding()
        val c = controller(binding)
        c.speak(request, "One.", Ggtts.DEFAULT_SEED)
        c.speak(request, "Two.", Ggtts.DEFAULT_SEED)
        assertEquals(1, binding.loads.size)
        c.speak(request.copy(engine = "qwen3-tts", voice = null), "Three.", Ggtts.DEFAULT_SEED)
        assertEquals(2, binding.loads.size)
    }
}
