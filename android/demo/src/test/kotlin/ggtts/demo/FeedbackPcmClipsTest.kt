package ggtts.demo

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class FeedbackPcmClipsTest {
    private fun fixture(n: Int): Pair<ByteArray, AudioBoundReadManifest> {
        val samples = FloatArray(n) { i ->
            when (i % 7) { 0 -> -1f; 1 -> -0.5f; 2 -> -0.0001f; 3 -> 0f; 4 -> 0.0001f; 5 -> 0.5f; else -> 1f }
        }
        val wav = Wav.encode(samples, 24000)
        val evidence = VerifiedWavEvidence()
        evidence.add(0, "first", 1, 0, 0, 0, n.toLong(), samples, 24000)
        evidence.complete(true)
        return wav to evidence.verify("test-read", wav)
    }

    @Test fun refusesChangedAndMalformedWav() {
        val (wav, manifest) = fixture(16)
        val changed = wav.clone().also { it[44] = (it[44].toInt() xor 1).toByte() }
        assertTrue(runCatching { FeedbackPcmClips.fromWav(changed, manifest) }.isFailure)
        val malformed = wav.clone().also { it[22] = 2 }
        assertTrue(runCatching { FeedbackPcmClips.fromWav(malformed, manifest) }.isFailure)
    }

    private fun decodeFlac(bytes: ByteArray): ByteArray {
        val file = File.createTempFile("ggtts-pcm-", ".flac")
        val raw = File.createTempFile("ggtts-pcm-", ".raw")
        try {
            file.writeBytes(bytes)
            val process = ProcessBuilder("ffmpeg", "-v", "error", "-y", "-i", file.path,
                "-f", "s16le", "-acodec", "pcm_s16le", raw.path).redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            assertEquals("ffmpeg failed: $output", 0, process.waitFor())
            return raw.readBytes()
        } finally { file.delete(); raw.delete() }
    }

    @Test fun eachDeliveredSpanIncludesItsOwnGapAndMetadata() {
        val a = floatArrayOf(-1f, 0.5f, 0f)
        val b = floatArrayOf(1f, -0.5f)
        val wav = Wav.encode(a + b, 24000)
        val evidence = VerifiedWavEvidence()
        evidence.add(0, "first", 2, 0, 0, 0, 2, a, 24000)
        evidence.add(1, "second", 3, 1, 1, 3, 5, b, 24000)
        evidence.complete(true)
        val manifest = evidence.verify("two-chunks", wav)
        val clips = FeedbackPcmClips.fromWav(wav, manifest)
        assertEquals(manifest.chunks, clips.map { it.chunk })
        assertEquals(3.0 / 24000, Flac.seconds(clips[0].flac)!!, 0.000000001)
        assertEquals(2.0 / 24000, Flac.seconds(clips[1].flac)!!, 0.000000001)
        val ffmpeg = runCatching { ProcessBuilder("ffmpeg", "-version").start().waitFor() == 0 }.getOrDefault(false)
        assumeTrue("FFmpeg required for independent decoder test", ffmpeg)
        assertArrayEquals(wav.copyOfRange(44, 50), decodeFlac(clips[0].flac))
        assertArrayEquals(wav.copyOfRange(50, 54), decodeFlac(clips[1].flac))
    }

    @Test fun flacDecodesToExactSavedPcmAcrossFrameBoundaries() {
        val available = runCatching { ProcessBuilder("ffmpeg", "-version").start().waitFor() == 0 }.getOrDefault(false)
        assumeTrue("FFmpeg required for independent decoder test", available)
        for (n in listOf(1, 15, 4095, 4096, 4097, 8193)) {
            val (wav, manifest) = fixture(n)
            val clip = FeedbackPcmClips.fromWav(wav, manifest).single()
            val file = File.createTempFile("ggtts-pcm-", ".flac")
            val raw = File.createTempFile("ggtts-pcm-", ".raw")
            try {
                file.writeBytes(clip.flac)
                val process = ProcessBuilder("ffmpeg", "-v", "error", "-y", "-i", file.path,
                    "-f", "s16le", "-acodec", "pcm_s16le", raw.path).redirectErrorStream(true).start()
                val output = process.inputStream.bufferedReader().readText()
                assertEquals("ffmpeg failed for $n: $output", 0, process.waitFor())
                assertArrayEquals("PCM mismatch for $n", wav.copyOfRange(44, wav.size), raw.readBytes())
                assertEquals(manifest.chunks.single(), clip.chunk)
            } finally { file.delete(); raw.delete() }
        }
    }
}
