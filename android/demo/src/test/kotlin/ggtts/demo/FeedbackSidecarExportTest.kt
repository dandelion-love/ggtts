package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class FeedbackSidecarExportTest {
    private class Provider : FeedbackDocumentProvider {
        data class Entry(var name: String, var bytes: ByteArray = byteArrayOf())
        val entries = linkedMapOf<String, Entry>()
        var seq = 0
        var fail: String? = null
        var writes = 0
        var publishes = 0
        var renameIds = false
        override fun find(name: String): String? {
            val found = entries.filterValues { it.name == name }.keys.toList()
            check(found.size <= 1)
            return found.singleOrNull()
        }
        override fun createStaging(name: String, mime: String): String =
            (++seq).toString().also { entries[it] = Entry(name) }
        override fun write(id: String, bytes: ByteArray) {
            writes++
            if (fail == "write$writes") {
                entries.getValue(id).bytes = bytes.copyOf(bytes.size / 2)
                error("planted write")
            }
            entries.getValue(id).bytes = bytes.copyOf()
        }
        override fun read(id: String) = entries.getValue(id).bytes.copyOf()
        override fun publish(id: String, name: String): String {
            publishes++
            if (fail == "publish$publishes") error("planted publish")
            check(find(name) == null)
            val entry = entries.getValue(id)
            entry.name = name
            if (!renameIds) return id
            entries.remove(id)
            return (++seq).toString().also { entries[it] = entry }
        }
        override fun delete(id: String) { entries.remove(id) }
    }

    private fun fixture(block: (java.io.File, FileSavedEvidenceStore, FeedbackResultStore) -> Unit) {
        val dir = Files.createTempDirectory("feedback-export").toFile()
        try {
            val audio = dir.resolve("read.wav")
            val pcm = floatArrayOf(0.5f, -0.5f, 0f)
            audio.writeBytes(Wav.encode(pcm, 24000))
            val e = VerifiedWavEvidence()
            e.add(0, "\u4e2d", -1, 2, 0, 0, 2, pcm, 24000); e.complete(true)
            val store = FileSavedEvidenceStore()
            store.save(audio.path, e); store.markPlaybackComplete(audio.path)
            val input = FeedbackReadSelector(store).prepare(audio)!!
            val result = feedbackCanonical(mapOf(
                "format" to "ggtts-feedback-result/1", "id" to input.id, "input_sha256" to input.digest,
                "note" to "", "items" to listOf(mapOf("id" to "chunk-0", "marks" to listOf(0L),
                    "rating" to "accept", "note" to ""))))
            val results = FeedbackResultStore(dir.resolve("results"))
            results.save(input, result)
            block(audio, store, results)
        } finally { dir.deleteRecursively() }
    }

    @Test fun roundTripAndTamper() = fixture { audio, store, results ->
        val provider = Provider()
        val export = FeedbackSidecarExport(store, results)
        val pair = export.export(provider, audio)
        val restored = export.reopen(provider, pair.audioName)
        assertNotNull(restored)
        val id = provider.find(pair.audioName)!!
        provider.entries.getValue(id).bytes[44] = 0
        assertNull(export.reopen(provider, pair.audioName))
        provider.delete(provider.find(pair.sidecarName)!!)
        assertNull(export.reopen(provider, pair.audioName))
    }

    @Test fun failureAndRestartNeverAcceptIncompletePair() = fixture { audio, store, results ->
        for (fault in listOf("write1", "write2", "publish1", "publish2")) {
            val provider = Provider().apply { fail = fault }
            val export = FeedbackSidecarExport(store, results)
            assertTrue(runCatching { export.export(provider, audio) }.isFailure)
            assertTrue("cleanup $fault", provider.entries.isEmpty())
            assertTrue(audio.exists())
            provider.fail = null
            val retry = export.export(provider, audio)
            assertNotNull(FeedbackSidecarExport(store, results).reopen(provider, retry.audioName))
            provider.delete(provider.find(retry.sidecarName)!!)
            assertNull(FeedbackSidecarExport(store, results).reopen(provider, retry.audioName))
        }
    }
    @Test fun swappedSidecarAndInterruptedArtifactsNeverRestore() = fixture { audio, store, results ->
        val provider = Provider()
        val export = FeedbackSidecarExport(store, results)
        val pair = export.export(provider, audio)
        val side = provider.find(pair.sidecarName)!!
        val original = provider.entries.getValue(side).bytes
        provider.entries.getValue(side).bytes = original.copyOf(original.size / 2)
        assertNull(export.reopen(provider, pair.audioName))
        provider.entries.getValue(side).bytes = original
        assertNotNull(export.reopen(provider, pair.audioName))
        provider.entries.getValue(side).bytes = String(original, Charsets.UTF_8)
            .replace("ggtts-feedback-sidecar/1", "ggtts-feedback-sidecar/2").toByteArray()
        assertNull(export.reopen(provider, pair.audioName))
        provider.entries.getValue(side).bytes = original
        // Simulate death after audio publication, with only a temporary sidecar.
        provider.entries.getValue(side).name = ".abandoned.feedback.tmp"
        assertNull(FeedbackSidecarExport(store, results).reopen(provider, pair.audioName))
    }

    @Test fun missingCompletionOrFeedbackRefusesExport() = fixture { audio, store, results ->
        val provider = Provider()
        val completion = java.io.File(audio.path + ".playback-complete")
        assertTrue(completion.delete())
        assertTrue(runCatching { FeedbackSidecarExport(store, results).export(provider, audio) }.isFailure)
        assertTrue(provider.entries.isEmpty())
    }
    @Test fun changedPublishIdsAreTrackedAndPreexistingDocumentsRemain() = fixture { audio, store, results ->
        val provider = Provider().apply { renameIds = true }
        provider.entries["prior"] = Provider.Entry("existing.wav", byteArrayOf(1, 2, 3))
        val export = FeedbackSidecarExport(store, results)
        provider.fail = "publish2"
        assertTrue(runCatching { export.export(provider, audio) }.isFailure)
        assertEquals(setOf("prior"), provider.entries.keys)
        assertArrayEquals(byteArrayOf(1, 2, 3), provider.entries.getValue("prior").bytes)
        provider.fail = null
        val pair = export.export(provider, audio)
        assertNotNull(export.reopen(provider, pair.audioName))
        assertArrayEquals(byteArrayOf(1, 2, 3), provider.entries.getValue("prior").bytes)
    }
    @Test fun providerFailuresHaveFixedStageWithoutExceptionText() = fixture { audio, store, results ->
        val logs = mutableListOf<String>()
        val provider = Provider().apply { fail = "publish2" }
        val export = FeedbackSidecarExport(store, results, DiagnosticLog(sink = { logs += it }))
        assertTrue(runCatching { export.export(provider, audio) }.isFailure)
        assertTrue(logs.toString(), logs.any { "stage=export_publish outcome=provider_failure" in it })
        assertTrue(logs.toString(), logs.none { "planted publish" in it || audio.path in it })
        assertTrue(logs.toString(), logs.all { "elapsed_ms=" in it })
    }
    @Test fun throwingDiagnosticSinkCannotChangeExportOrMaskProviderFailure() = fixture { audio, store, results ->
        val provider = Provider()
        val diagnostics = DiagnosticLog(sink = { throw IllegalStateException("diagnostic failure") })
        val export = FeedbackSidecarExport(store, results, diagnostics)
        val pair = export.export(provider, audio)
        assertNotNull(export.reopen(provider, pair.audioName))
        val failing = Provider().apply { fail = "publish2" }
        val error = runCatching { export.export(failing, audio) }.exceptionOrNull()
        assertEquals("planted publish", error?.message)
        assertTrue(failing.entries.isEmpty())
    }
}
