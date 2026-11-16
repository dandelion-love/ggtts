package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class SavedEvidenceStoreTest {
    private fun evidence(): VerifiedWavEvidence = VerifiedWavEvidence().apply {
        add(0, "first", -1L, 2, 0, 0, 2, floatArrayOf(0.5f, -0.5f, 0f), 24000)
        add(1, "second", 7L, 0, 1, 3, 4, floatArrayOf(1f), 24000)
        complete(true)
    }
    @Test fun reopenAndCorruptionFailClosed() {
        val dir = Files.createTempDirectory("read-evidence").toFile()
        try {
            val audio = dir.resolve("read.wav")
            audio.writeBytes(Wav.encode(floatArrayOf(0.5f, -0.5f, 0f, 1f), 24000))
            val store = FileSavedEvidenceStore()
            assertNull(store.load(audio.path))
            assertTrue(store.list(dir).isEmpty())
            store.save(audio.path, evidence())
            // A fresh store instance models process restart; only valid adjacent sidecars are listed.
            val reopenedStore = FileSavedEvidenceStore()
            assertEquals(listOf(audio), reopenedStore.list(dir).map { it.audioFile })
            val reopened = reopenedStore.load(audio.path)!!
            assertEquals(2, reopened.chunks[0].draw)
            assertEquals(-1L, reopened.chunks[0].seedBits)
            audio.writeBytes(audio.readBytes().also { it[44] = (it[44].toInt() xor 1).toByte() })
            assertNull(store.load(audio.path))
            assertTrue(reopenedStore.list(dir).isEmpty())
            dir.resolve("read.wav.manifest.json").writeText("{")
            assertNull(store.load(audio.path))
            assertTrue(reopenedStore.list(dir).isEmpty())
        } finally { dir.deleteRecursively() }
    }

    @Test fun failedOrIncompleteEvidenceDoesNotPublish() {
        val dir = Files.createTempDirectory("read-evidence").toFile()
        try {
            val audio = dir.resolve("read.wav")
            audio.writeBytes(Wav.encode(floatArrayOf(1f), 24000))
            val incomplete = VerifiedWavEvidence()
            incomplete.add(0, "x", 1, 0, 0, 0, 1, floatArrayOf(1f), 24000)
            incomplete.complete(false)
            try { FileSavedEvidenceStore().save(audio.path, incomplete); fail("expected refusal") }
            catch (_: IllegalStateException) {}
            assertNull(FileSavedEvidenceStore().load(audio.path))
            assertFalse(dir.resolve("read.wav.manifest.json").exists())
        } finally { dir.deleteRecursively() }
    }
    @Test fun interruptedWriteAndFailedRenameNeverPublish() {
        val dir = Files.createTempDirectory("read-evidence").toFile()
        try {
            val audio = dir.resolve("read.wav")
            audio.writeBytes(Wav.encode(floatArrayOf(0.5f, -0.5f, 0f, 1f), 24000))
            val failedWrite = FileSavedEvidenceStore(write = { file, bytes ->
                file.writeBytes(bytes.copyOfRange(0, 8))
                throw java.io.IOException("planted interrupted write")
            })
            try { failedWrite.save(audio.path, evidence()); fail("expected write failure") }
            catch (_: java.io.IOException) {}
            assertNull(FileSavedEvidenceStore().load(audio.path))
            assertEquals(listOf("read.wav"), dir.list()!!.sorted())
            val failedPublish = FileSavedEvidenceStore(publish = { _, _ -> false })
            try { failedPublish.save(audio.path, evidence()); fail("expected publish failure") }
            catch (_: IllegalStateException) {}
            assertNull(FileSavedEvidenceStore().load(audio.path))
            assertEquals(listOf("read.wav"), dir.list()!!.sorted())
        } finally { dir.deleteRecursively() }
    }
}
