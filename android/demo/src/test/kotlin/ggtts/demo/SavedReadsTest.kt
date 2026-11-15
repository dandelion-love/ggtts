// The saved-reads list apart from Android: which files are reads, their title and length from the file itself (with
// MediaStore's only as a fallback), merging this install's with the picked folder's, and each row's text. Each test
// names the planted fault it catches.
package ggtts.demo

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.TimeZone
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SavedReadsTest {
    private val flac = Flac.encode(FloatArray(24000 * 3 + 1200) { 0.1f }, 24000, "今日天氣好好")

    /** A WAV as DeviceAudio writes it: 16-bit mono at [rate], [seconds] long. */
    private fun wav(rate: Int, seconds: Double): ByteArray {
        val data = (rate * seconds).toInt() * 2
        val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + data).put("WAVEfmt ".toByteArray())
        b.putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
        b.put("data".toByteArray()).putInt(data)
        return b.array()
    }

    /**
     * Plant: the title or length taken from MediaStore (which a reinstall forgets) over the file's own, or a FLAC
     * length read from the wrong bits. The file's TITLE and STREAMINFO win.
     */
    @Test
    fun theTitleAndLengthComeFromTheFileItself() {
        val (title, seconds) = SavedReads.fromHead(flac.copyOf(SavedReads.HEAD.coerceAtMost(flac.size)))
        assertEquals("今日天氣好好", title)
        assertEquals(3.05, seconds!!, 1e-9)
        // A rate whose low 4 bits are not zero (22050 = 0x5622): 24000's are, which hides a misread of those bits.
        assertEquals(1.0, Flac.seconds(Flac.encode(FloatArray(22050), 22050))!!, 1e-9)
        val r =SavedReads.read("content://x/1", "ggtts-qwen3-tts-20261115-055044.flac", flac, "stale title", 999_000, 1L, owned = true)
        assertEquals("今日天氣好好", r.title)
        assertEquals(3.05, r.seconds!!, 1e-9)
        assertEquals(Flac.MIME, r.mime)
    }

    @Test
    fun aWavGivesItsLengthAndMediaStoreFillsWhatAFileDoesNotSay() {
        assertEquals(2.5, SavedReads.fromHead(wav(24000, 2.5)).second!!, 1e-9)
        val w = SavedReads.read("content://x/2", "ggtts-qwen3-tts-20261114-214645.wav", wav(24000, 1.83), "A title", null, 2L, owned = true)
        assertEquals("A title", w.title)
        assertEquals(1.83, w.seconds!!, 1e-3)
        // No head at all (unreadable): MediaStore's title and duration; and a title that is only the file name says so.
        val bare = SavedReads.read("content://x/3", "ggtts-kitten-tts-2-20261115-010000.flac", null, "ggtts-kitten-tts-2-20261115-010000", 4200, 3L, owned = false)
        assertEquals("ggtts-kitten-tts-2-20261115-010000", bare.title)
        assertEquals(4.2, bare.seconds!!, 1e-9)
        assertNull(SavedReads.fromHead("not audio".toByteArray()).second)
        assertNull("a cut-short FLAC has no title rather than an error", Flac.title(flac.copyOf(60)))
    }

    /** Plant: a read in both places listed twice, or a voice sample listed as a read. */
    @Test
    fun aReadInBothPlacesIsListedOnceAsThisInstallsAndNewestFirst() {
        fun r(name: String, date: Long, owned: Boolean) = SavedRead("u/$name/$owned", name, name, 1.0, date, owned)
        val owned = listOf(r("ggtts-a.flac", 2, true))
        val picked = listOf(r("ggtts-a.flac", 2, false), r("ggtts-b.wav", 3, false), r("hk_voice_343.sample.wav", 9, false), r("ggtts-x.sample.wav", 9, false))
        val list = SavedReads.merge(owned, picked)
        assertEquals(listOf("ggtts-b.wav", "ggtts-a.flac"), list.map { it.name })
        assertTrue(list.single { it.name == "ggtts-a.flac" }.owned)
        assertFalse(SavedReads.isRead("notes.txt"))
    }

    @Test
    fun aRowSaysItsLengthAndWhenItWasSaved() {
        val r = SavedRead("u", "ggtts-a.flac", "t", 9.56, 1_763_200_000_000L, true)
        assertEquals("9.6 s · 15 Nov 2025 09:46", SavedReads.describe(r, TimeZone.getTimeZone("UTC")))
        assertEquals("2 min 05 s · 15 Nov 2025 09:46", SavedReads.describe(r.copy(seconds = 125.0), TimeZone.getTimeZone("UTC")))
        assertTrue(SavedReads.describe(r.copy(seconds = null), TimeZone.getTimeZone("UTC")).startsWith("length unknown"))
    }
}
