package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files

class DiagnosticLogTest {
    @Test fun fixedVocabularyAndMonotonicDurationExcludeSecrets() {
        val records = mutableListOf<String>()
        var now = 2_000_000L
        val d = DiagnosticLog(sink = { records += it }, clock = { now })
        val event = d.start(DiagnosticStage.EVIDENCE)
        now = 9_000_000L
        event.finish(DiagnosticOutcome.INVALID)
        assertEquals(1, records.size)
        assertTrue(records.single(), records.single().contains("stage=evidence outcome=invalid"))
        assertTrue(records.single(), records.single().endsWith("elapsed_ms=7"))
    }

    @Test fun preparationReportsUnavailableAndIncompleteWithoutSecretPathOrText() {
        val dir = Files.createTempDirectory("PRIVATE_PATH_781").toFile()
        try {
            val audio = dir.resolve("PRIVATE_FILENAME_781.wav")
            val logs = mutableListOf<String>()
            val diagnostics = DiagnosticLog(sink = { logs += it })
            val selector = FeedbackReadSelector(diagnostics = diagnostics)
            assertNull(selector.prepare(audio))
            assertTrue(logs.last(), "outcome=unavailable" in logs.last())
            val pcm = floatArrayOf(1f)
            audio.writeBytes(Wav.encode(pcm, 24000))
            val evidence = VerifiedWavEvidence()
            evidence.add(0, "PRIVATE_TEXT_781", 1, 0, 0, 0, 1, pcm, 24000)
            evidence.complete(true)
            FileSavedEvidenceStore().save(audio.path, evidence)
            assertNull(selector.prepare(audio))
            assertTrue(logs.last(), "outcome=incomplete" in logs.last())
            FileSavedEvidenceStore().markPlaybackComplete(audio.path)
            assertNotNull(selector.prepare(audio))
            assertTrue(logs.last(), "outcome=ok" in logs.last())
            assertTrue(logs.toString(), logs.none { "PRIVATE_" in it || audio.path in it })
            assertTrue(logs.toString(), logs.all { "elapsed_ms=" in it })
        } finally { dir.deleteRecursively() }
    }
    @Test fun bridgeLogsAtMostOneSuccessAndOneFailure() {
        val records = mutableListOf<String>()
        val log = DiagnosticLog(sink = { records += it })
        val limiter = BridgeDiagnosticLimiter()
        repeat(20) { limiter.report(log.start(DiagnosticStage.BRIDGE_SAVE), true) }
        repeat(20) { limiter.report(log.start(DiagnosticStage.BRIDGE_SAVE), false) }
        assertEquals(2, records.size)
        assertTrue(records[0], "outcome=ok" in records[0])
        assertTrue(records[1], "outcome=failure" in records[1])
    }
}
