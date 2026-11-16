package ggtts.demo

import java.io.File
import java.io.FileOutputStream
import java.util.UUID

/** Saved audio identity and chunk evidence. This does not claim playback completion or feedback readiness. */
data class SavedAudioEvidence(val audioFile: File, val manifest: AudioBoundReadManifest)

/** Audio identity only: this record establishes neither playback completion nor feedback readiness. */
interface SavedEvidenceStore {
    fun save(audioPath: String, evidence: VerifiedWavEvidence)
    fun markPlaybackComplete(audioPath: String)
}

/** Sidecar bound to a unique app-owned WAV. Missing, corrupt or mismatched records are unavailable. */
class FileSavedEvidenceStore(
    private val write: (File, ByteArray) -> Unit = { file, bytes ->
        FileOutputStream(file).use { it.write(bytes); it.fd.sync() }
    },
    private val publish: (File, File) -> Boolean = { from, to -> from.renameTo(to) },
) : SavedEvidenceStore {
    private fun sidecar(audio: File) = File(audio.path + ".manifest.json")

    override fun save(audioPath: String, evidence: VerifiedWavEvidence) {
        val audio = File(audioPath)
        val dest = sidecar(audio)
        require(!dest.exists()) { "evidence already exists" }
        val bytes = audio.readBytes()
        val manifest = evidence.verify(UUID.randomUUID().toString(), bytes)
        val temp = File(dest.path + "." + UUID.randomUUID() + ".tmp")
        try {
            write(temp, manifest.encodeJson().toByteArray(Charsets.UTF_8))
            check(!dest.exists() && publish(temp, dest)) { "cannot publish evidence" }
        } finally {
            temp.delete()
        }
    }

    fun load(audioPath: String): AudioBoundReadManifest? = try {
        val audio = File(audioPath)
        val manifest = AudioBoundReadManifest.decodeJson(sidecar(audio).readText(Charsets.UTF_8))
        val bytes = audio.readBytes()
        manifest.validateAudio(bytes, manifest.audio.sampleRate, manifest.audio.sampleCount)
        manifest
    } catch (_: Exception) { null }

    private fun completion(audio: File) = File(audio.path + ".playback-complete")

    /** Persist only a confirmed playback head, bound to this verified manifest and audio hash. */
    override fun markPlaybackComplete(audioPath: String) {
        val audio = File(audioPath)
        val manifest = load(audioPath) ?: error("saved evidence unavailable")
        val dest = completion(audio)
        check(!dest.exists())
        val content = completionContent(audio, manifest)
        val temp = File(dest.path + "." + UUID.randomUUID() + ".tmp")
        try {
            write(temp, content.toByteArray(Charsets.UTF_8))
            check(load(audioPath) == manifest && !dest.exists() && publish(temp, dest))
        } finally { temp.delete() }
    }

    /** False for missing, interrupted, mismatched or corrupt evidence; not proof a person heard it. */
    fun playbackComplete(audioPath: String): Boolean {
        return try {
            val audio = File(audioPath)
            val manifest = load(audioPath) ?: return false
            completion(audio).readText(Charsets.UTF_8) == completionContent(audio, manifest)
        } catch (_: Exception) { false }
    }

    private fun completionContent(audio: File, manifest: AudioBoundReadManifest): String =
        listOf("ggtts-playback-complete/1", manifest.readId, manifest.audio.sha256, manifestHash(audio), "").joinToString(10.toChar().toString())

    private fun manifestHash(audio: File): String = java.security.MessageDigest.getInstance("SHA-256")
        .digest(sidecar(audio).readBytes()).joinToString("") { "%02x".format(it) }

    /** Reopen valid evidence in an app-private audio directory after process restart. */
    fun list(directory: File): List<SavedAudioEvidence> {
        val files = directory.listFiles() ?: return emptyList()
        return files.asSequence()
            .filter { it.isFile && it.extension.equals("wav", ignoreCase = true) }
            .mapNotNull { audio -> load(audio.path)?.let { SavedAudioEvidence(audio, it) } }
            .sortedBy { it.audioFile.name }
            .toList()
    }
}
