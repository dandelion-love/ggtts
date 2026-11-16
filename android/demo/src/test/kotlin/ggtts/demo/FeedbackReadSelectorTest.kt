package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class FeedbackReadSelectorTest {
    @Test fun onlyVerifiedCompletedReadOpensAndIdentityIsStable() {
        val dir = Files.createTempDirectory("feedback-read").toFile()
        try {
            val audio = dir.resolve("read.wav")
            val pcm = floatArrayOf(0.5f, -0.5f, 0f)
            audio.writeBytes(Wav.encode(pcm, 24000))
            val e = VerifiedWavEvidence()
            e.add(0, "\u4e2d", -1L, 2, 0, 0, 2, pcm, 24000)
            e.complete(true)
            val store = FileSavedEvidenceStore()
            val selector = FeedbackReadSelector(store)
            assertFalse(selector.eligible(audio))
            assertNull(selector.prepare(audio))
            store.save(audio.path, e)
            assertNull(selector.prepare(audio))
            store.markPlaybackComplete(audio.path)
            assertTrue(selector.eligible(audio))
            val first = selector.prepare(audio)!!
            assertEquals(first.digest, FeedbackReadSelector().prepare(audio)!!.digest)
            assertTrue(first.canonicalJson.contains("chunk-0"))
            audio.writeBytes(audio.readBytes().also { it[44] = (it[44].toInt() xor 1).toByte() })
            assertFalse(selector.eligible(audio))
            assertNull(selector.prepare(audio))
        } finally { dir.deleteRecursively() }
    }

    @Test fun missingAndCorruptRecordsAreUnavailable() {
        val dir = Files.createTempDirectory("feedback-read").toFile()
        try {
            val audio = dir.resolve("old.wav")
            audio.writeBytes(Wav.encode(floatArrayOf(1f), 24000))
            assertNull(FeedbackReadSelector().prepare(audio))
            dir.resolve("old.wav.manifest.json").writeText("{")
            assertNull(FeedbackReadSelector().prepare(audio))
        } finally { dir.deleteRecursively() }
    }
}
