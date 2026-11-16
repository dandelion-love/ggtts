package ggtts.demo

import ggtts.Pcm
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class EvidenceControllerTest {
    private class View : DemoView {
        val errors = mutableListOf<String>()
        override fun setBusy(busy: Boolean) {}
        override fun showStatus(text: String) {}
        override fun showError(text: String) { errors += text }
    }
    private class Audio(private val path: String) : AudioOut {
        val played = mutableListOf<FloatArray>()
        override fun begin(rate: Int) {}
        override fun play(samples: FloatArray) { played += samples }
        override fun finish() {}
        override fun stop() {}
        override fun save(samples: FloatArray, rate: Int, engine: String): String {
            java.io.File(path).writeBytes(Wav.encode(samples, rate))
            return path
        }
    }
    private class Engine(private val success: Boolean) : TtsEngine {
        override val supportsEvidence = true
        override fun render(text: String, seed: Long): Pcm = error("not used")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
            error("legacy stream must not be called")
        override fun renderEvidence(text: String, seed: Long,
            onPiece: (Int, String, Long, Int, Int, Long, Long, FloatArray, Int) -> Boolean): Boolean {
            if (!onPiece(0, "first", -1L, 2, 0, 0, 2, floatArrayOf(0.5f, -0.5f, 0f), 24000)) return false
            if (!onPiece(1, "second", 7L, 0, 1, 3, 4, floatArrayOf(1f), 24000)) return false
            return success
        }
        override fun close() {}
    }

    @Test fun successfulStreamSavesAndReopensEvidence() {
        val dir = Files.createTempDirectory("controller-evidence").toFile()
        try {
            val file = dir.resolve("read.wav"); val audio = Audio(file.path); val view = View()
            val controller = DemoController(TtsBinding { Engine(true) }, view, audio,
                background = { it.run() }, main = { it.run() }, evidenceStore = FileSavedEvidenceStore())
            controller.speak(LoadRequest("fake", null, null, null), "first second", 7)
            assertTrue(view.errors.toString(), view.errors.isEmpty())
            assertEquals(2, audio.played.size)
            val manifest = FileSavedEvidenceStore().load(file.path)!!
            assertEquals("first", manifest.chunks[0].text)
            assertEquals(2, manifest.chunks[0].draw)
            assertEquals(-1L, manifest.chunks[0].seedBits)
            assertEquals(SampleSpan(0, 3), manifest.chunks[0].delivered)
        } finally { dir.deleteRecursively() }
    }

    @Test fun abortedStreamDoesNotSaveAudioOrEvidence() {
        val dir = Files.createTempDirectory("controller-evidence").toFile()
        try {
            val file = dir.resolve("read.wav"); val audio = Audio(file.path)
            val controller = DemoController(TtsBinding { Engine(false) }, View(), audio,
                background = { it.run() }, main = { it.run() }, evidenceStore = FileSavedEvidenceStore())
            controller.speak(LoadRequest("fake", null, null, null), "x", 7)
            assertFalse(file.exists())
            assertNull(FileSavedEvidenceStore().load(file.path))
        } finally { dir.deleteRecursively() }
    }
}
