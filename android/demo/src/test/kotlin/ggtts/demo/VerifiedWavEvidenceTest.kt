package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.sound.sampled.AudioSystem
import java.io.ByteArrayInputStream

class VerifiedWavEvidenceTest {
    private val first = floatArrayOf(-1f, -0.5f, 0.5f, 0f)
    private val second = floatArrayOf(1f, 0.1f)
    private fun wav(rate: Int = 24000): ByteArray {
        val samples = first + second
        val b = ByteBuffer.allocate(44 + samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + samples.size * 2).put("WAVEfmt ".toByteArray())
        b.putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16)
        b.put("data".toByteArray()).putInt(samples.size * 2)
        for (x in samples) b.putShort((x.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        return b.array()
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
        val audio = AudioSystem.getAudioInputStream(ByteArrayInputStream(bytes))
        assertEquals(24000f, audio.format.sampleRate)
        assertEquals(6L, audio.frameLength)
        val pcm = audio.readBytes()
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
        rejected { e.verify("r", wav().also { it[16] = 17 }) }
        val stopped = evidence().apply { complete(false) }
        rejected { stopped.verify("r", wav()) }
        rejected { VerifiedWavEvidence().apply { add(0, "x", 0, 0, 0, 0, 1, floatArrayOf(1f, 1f), 24000) } }
    }
}
