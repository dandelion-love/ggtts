// The media controls apart from Android: the state a media session publishes, the notification's toggle button, and
// that Pause, Play and Stop reach the read. Each test names the planted fault it catches.
package ggtts.demo

import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MediaControlsTest {
    /** Plant: a state that reads "playing" while paused, which shows the wrong lock-screen button. */
    @Test
    fun theSessionStateFollowsTheRead() {
        assertEquals(MediaControls.State.NONE, MediaControls.state(busy = false, paused = false))
        assertEquals(MediaControls.State.NONE, MediaControls.state(busy = false, paused = true))
        assertEquals(MediaControls.State.PLAYING, MediaControls.state(busy = true, paused = false))
        assertEquals(MediaControls.State.PAUSED, MediaControls.state(busy = true, paused = true))
    }

    /** Plant: a toggle that sends Pause while paused. The button must always offer the other state. */
    @Test
    fun theNotificationsButtonOffersTheOtherState() {
        assertEquals(ForegroundGate.ACTION_PAUSE, MediaControls.toggleAction(MediaControls.State.PLAYING))
        assertEquals("Pause", MediaControls.toggleLabel(MediaControls.State.PLAYING))
        assertEquals(ForegroundGate.ACTION_PLAY, MediaControls.toggleAction(MediaControls.State.PAUSED))
        assertEquals("Play", MediaControls.toggleLabel(MediaControls.State.PAUSED))
        assertNull(MediaControls.toggleAction(MediaControls.State.NONE))
    }

    private class Audio : AudioOut {
        override var paused = false
        val events = mutableListOf<String>()
        override fun begin(rate: Int) {}
        override fun play(samples: FloatArray) {}
        override fun finish() {}
        override fun stop() { events += "stop" }
        override fun save(samples: FloatArray, rate: Int, engine: String) = "/x.wav"
        override fun pause() { paused = true; events += "pause" }
        override fun resume() { paused = false; events += "resume" }
    }

    private fun controller(audio: Audio, statuses: MutableList<String>): DemoController {
        val engine = object : TtsEngine {
            override fun render(text: String, seed: Long): Pcm = error("streams")
            override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean) = true
            override fun close() {}
        }
        val view = object : DemoView {
            override fun setBusy(busy: Boolean) {}
            override fun showStatus(text: String) { statuses += text }
            override fun showError(text: String) { statuses += "error: $text" }
        }
        // The read's background step is held back, so the controller stays busy while the controls are pressed.
        return DemoController(TtsBinding { engine }, view, audio, background = {}, main = { it.run() })
    }

    /** Plant: Pause and Play swapped, or either ignored. Each reaches the player once, and only during a read. */
    @Test
    fun pausePlayAndStopReachTheRead() {
        val audio = Audio()
        val statuses = mutableListOf<String>()
        val c = controller(audio, statuses)
        assertEquals("pause", MediaControls.apply(ForegroundGate.ACTION_PAUSE, c))
        assertTrue("Pause with no read touched the player", audio.events.isEmpty())
        c.speak(LoadRequest("qwen3-tts", "/m.gguf", "/c.gguf", null), "line", 1)
        MediaControls.apply(ForegroundGate.ACTION_PAUSE, c)
        assertTrue(c.paused)
        MediaControls.apply(ForegroundGate.ACTION_PAUSE, c) // a second Pause does nothing more
        MediaControls.apply(ForegroundGate.ACTION_PLAY, c)
        assertFalse(c.paused)
        MediaControls.apply(ForegroundGate.ACTION_STOP, c)
        assertEquals(listOf("pause", "resume", "stop"), audio.events)
        assertTrue(statuses.any { it.startsWith("Paused") })
        assertNull(MediaControls.apply("something else", c))
    }
}
