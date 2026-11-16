package ggtts.demo

import java.io.File
import java.security.MessageDigest
import java.util.UUID

/**
 * A provider creates staging objects, checks for existing names and verifies published names.
 * Publication is per document: provider behavior and concurrent writers can leave orphans;
 * validation on reopen, not the presence of a name, decides whether a pair is complete.
 */
interface FeedbackDocumentProvider {
    fun find(name: String): String?
    fun createStaging(name: String, mime: String): String
    fun write(id: String, bytes: ByteArray)
    fun read(id: String): ByteArray
    fun publish(id: String, name: String): String
    fun delete(id: String)
}

data class ExportedFeedbackPair(val audioName: String, val sidecarName: String)
data class ReopenedFeedback(val input: PreparedFeedbackInput, val result: String)

/** Export only verified completed private reads and validated feedback; no two-document atomicity is assumed. */
class FeedbackSidecarExport(
    private val evidence: FileSavedEvidenceStore,
    private val results: FeedbackResultStore,
) {
    fun export(provider: FeedbackDocumentProvider, audio: File): ExportedFeedbackPair {
        val manifest = evidence.load(audio.path) ?: error("missing audio evidence")
        check(evidence.playbackComplete(audio.path)) { "playback is not confirmed" }
        val bytes = audio.readBytes()
        val input = inputFor(bytes, manifest)
        val result = results.load(input) ?: error("missing or invalid feedback")
        input.validateResult(result)
        check(evidence.load(audio.path) == manifest && evidence.playbackComplete(audio.path)) { "read changed" }

        val id = UUID.randomUUID().toString()
        val audioName = "ggtts-" + id + ".wav"
        val metadataName = audioName + ".feedback.json"
        check(provider.find(audioName) == null && provider.find(metadataName) == null)
        val envelope = feedbackCanonical(mapOf(
            "format" to "ggtts-feedback-sidecar/1",
            "audio_sha256" to manifest.audio.sha256,
            "input_sha256" to input.digest,
            "manifest" to manifest.encodeJson(),
            "result" to result,
        )).toByteArray(Charsets.UTF_8)
        var audioStage: String? = null
        var metadataStage: String? = null
        var audioPublished: String? = null
        var metadataPublished: String? = null
        try {
            audioStage = provider.createStaging("." + id + ".audio.tmp", "audio/wav")
            provider.write(audioStage, bytes)
            check(provider.read(audioStage).contentEquals(bytes)) { "staged audio differs" }
            metadataStage = provider.createStaging("." + id + ".feedback.tmp", "application/json")
            provider.write(metadataStage, envelope)
            check(provider.read(metadataStage).contentEquals(envelope)) { "staged sidecar differs" }
            audioPublished = provider.publish(audioStage, audioName)
            audioStage = null
            metadataPublished = provider.publish(metadataStage, metadataName)
            metadataStage = null
            check(reopen(provider, audioName) != null) { "published pair does not validate" }
            return ExportedFeedbackPair(audioName, metadataName)
        } catch (t: Throwable) {
            listOfNotNull(metadataStage, audioStage, metadataPublished, audioPublished).forEach {
                runCatching { provider.delete(it) }
            }
            throw t
        }
    }

    /** Missing, interrupted, malformed, or mismatched sidecars return null; audio remains playable. */
    fun reopen(provider: FeedbackDocumentProvider, audioName: String): String? = reopenPage(provider, audioName)?.result

    fun reopenPage(provider: FeedbackDocumentProvider, audioName: String): ReopenedFeedback? {
        return try {
        require(audioName.matches(Regex("^ggtts-[A-Za-z0-9-]+\\.wav$")))
        val audioId = provider.find(audioName) ?: return null
        val sideId = provider.find(audioName + ".feedback.json") ?: return null
        val raw = provider.read(sideId)
        require(raw.size <= 2_000_000)
        val json = raw.toString(Charsets.UTF_8)
        require(json.toByteArray(Charsets.UTF_8).contentEquals(raw))
        @Suppress("UNCHECKED_CAST")
        val root = JsonReader(json).readDocument() as? Map<String, Any?> ?: error("invalid sidecar")
        require(root.keys == setOf("format", "audio_sha256", "input_sha256", "manifest", "result"))
        require(root["format"] == "ggtts-feedback-sidecar/1")
        val manifest = AudioBoundReadManifest.decodeJson(root["manifest"] as? String ?: error("manifest missing"))
        val bytes = provider.read(audioId)
        require(root["audio_sha256"] == manifest.audio.sha256)
        val input = inputFor(bytes, manifest)
        require(root["input_sha256"] == input.digest)
        val result = root["result"] as? String ?: error("result missing")
        input.validateResult(result)
        ReopenedFeedback(input, result)
        } catch (_: Exception) { null }
    }

    private fun inputFor(bytes: ByteArray, manifest: AudioBoundReadManifest): PreparedFeedbackInput {
        val readHash = MessageDigest.getInstance("SHA-256").digest(manifest.readId.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        val page = FeedbackInputAdapter.fromWav(bytes, manifest, "read-" + readHash)
        return PreparedFeedbackInput(page.canonicalJson).also { check(it.digest == page.sha256) }
    }
}
