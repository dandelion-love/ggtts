package ggtts.demo

/** A lossless replay clip of a verified saved read, including its delivered trailing silence. */
data class FeedbackPcmClip(val chunk: DeliveredChunk, val flac: ByteArray)

object FeedbackPcmClips {
    /**
     * Recheck exact WAV identity and structural facts before extracting each delivered span.
     * The caller must have obtained the manifest from trusted stream evidence. This does not
     * generate a feedback page, validate a result, or prove playback.
     */
    fun fromWav(wav: ByteArray, manifest: AudioBoundReadManifest): List<FeedbackPcmClip> {
        val snapshot = wav.copyOf()
        val info = MonoPcmWav.parse(snapshot)
        manifest.validateAudio(snapshot, info.rate, info.count)
        return manifest.chunks.map { chunk ->
            val span = chunk.delivered
            val pcm = ShortArray((span.end - span.start).toInt()) { i ->
                val at = info.data + ((span.start + i) * 2).toInt()
                ((snapshot[at].toInt() and 255) or (snapshot[at + 1].toInt() shl 8)).toShort()
            }
            FeedbackPcmClip(chunk, Flac.encodePcm16(pcm, info.rate))
        }
    }
}
