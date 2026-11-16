package ggtts.demo

import java.io.File
import java.security.MessageDigest

/** A page may open only for a reopened, audio-bound read with confirmed playback-head consumption. */
class FeedbackReadSelector(
    private val store: FileSavedEvidenceStore = FileSavedEvidenceStore(),
    private val diagnostics: DiagnosticLog = DiagnosticLog(),
) {
    /** Checks saved identity and completion only; does not encode clips or build a page. */
    fun eligible(audio: File): Boolean = store.playbackComplete(audio.path)

    fun prepare(audio: File): PreparedFeedbackInput? {
        val event = diagnostics.start(DiagnosticStage.FEEDBACK_PREPARE)
        var outcome = DiagnosticOutcome.INVALID
        return try {
            val manifest = store.load(audio.path) ?: run {
                outcome = DiagnosticOutcome.UNAVAILABLE
                return null
            }
            if (!store.playbackComplete(audio.path)) {
                outcome = DiagnosticOutcome.INCOMPLETE
                return null
            }
            val readHash = MessageDigest.getInstance("SHA-256").digest(manifest.readId.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
            val page = FeedbackInputAdapter.fromWav(audio.readBytes(), manifest, "read-" + readHash)
            if (store.load(audio.path) != manifest || !store.playbackComplete(audio.path)) return null
            val input = PreparedFeedbackInput(page.canonicalJson)
            check(input.digest == page.sha256)
            outcome = DiagnosticOutcome.OK
            input
        } catch (_: Exception) { null }
        finally { event.finish(outcome) }
    }
}
