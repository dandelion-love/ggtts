package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream

class VerifiedWavEvidenceTest {
    private val first = floatArrayOf(-1f, -0.5f, 0.5f, 0f)
    private val second = floatArrayOf(1f, 0.1f)
    private fun wav(rate: Int = 24000): ByteArray {
        return Wav.encode(first + second, rate)
    }
    // Android test compilation omits java.desktop; resolve the host JVM decoder at runtime.
    private fun decodeWithJavaSound(bytes: ByteArray): ByteArray {
        val system = Class.forName("javax.sound.sampled.AudioSystem")
        val method = system.getMethod("getAudioInputStream", java.io.InputStream::class.java)
        return (method.invoke(null, ByteArrayInputStream(bytes)) as java.io.InputStream).use { it.readBytes() }
    }
    private fun evidence(): VerifiedWavEvidence = VerifiedWavEvidence().apply {
        add(0, "first", -1L, 2, 0, 0, 3, first, 24000)
        add(1, "second", 7L, 0, 1, 4, 6, second, 24000)
    }
    private fun rejected(block: () -> Unit) {
        try { block(); fail("expected refusal") } catch (_: IllegalArgumentException) {} catch (_: IllegalStateException) {}
    }

    @Test fun roundTripAndIndependentDecoder() {
        val bytes = wav()
        val pcm = decodeWithJavaSound(bytes)
        assertArrayEquals(bytes.copyOfRange(44, bytes.size), pcm)
        val e = evidence()
        e.complete(true)
        val m = e.verify("read-1", bytes)
        assertEquals(2, m.chunks[0].draw)
        assertEquals(-1L, m.chunks[0].seedBits)
        assertEquals(SampleSpan(0, 3), m.chunks[0].speech)
        assertEquals(SampleSpan(0, 4), m.chunks[0].delivered)
        assertEquals(m, AudioBoundReadManifest.decodeJson(m.encodeJson()))
    }

    @Test fun sharedWriterClipsAndTruncates() {
        val samples = floatArrayOf(-2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f)
        val bytes = Wav.encode(samples, 24000)
        val pcm = decodeWithJavaSound(bytes)
        val shorts = java.nio.ByteBuffer.wrap(pcm).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val actual = ShortArray(shorts.remaining()); shorts.get(actual)
        assertArrayEquals(shortArrayOf(-32767, -32767, -16383, 0, 16383, 32767, 32767), actual)
        val e = VerifiedWavEvidence()
        e.add(0, "boundary", 1, 0, 0, 0, samples.size.toLong(), samples, 24000)
        e.complete(true)
        e.verify("boundary", bytes)
    }

    @Test fun refusesCorruptionAndIncompleteStreams() {
        val e = evidence()
        rejected { e.verify("r", wav()) }
        e.complete(true)
        val changed = wav().also { it[44] = (it[44].toInt() xor 1).toByte() }
        rejected { e.verify("r", changed) }
        val gap = wav().also { it[50] = 1 }
        rejected { e.verify("r", gap) }
        rejected { e.verify("r", wav(22050)) }
        rejected { e.verify("r", wav().copyOf(wav().size - 1)) }
        rejected { e.verify("r", wav().also { it[22] = 2 }) }
        val original = wav()
        val extended = original.copyOfRange(0, 36) + byteArrayOf(0, 0) + original.copyOfRange(36, original.size)
        extended[16] = 17
        extended[4] = (extended.size - 8).toByte()
        rejected { e.verify("r", extended) }
        val stopped = evidence().apply { complete(false) }
        rejected { stopped.verify("r", wav()) }
        rejected { VerifiedWavEvidence().apply { add(0, "x", 0, 0, 0, 0, 1, floatArrayOf(1f, 1f), 24000) } }
    }
}
