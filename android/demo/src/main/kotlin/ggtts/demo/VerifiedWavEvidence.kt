package ggtts.demo

import java.security.MessageDigest

/** Provisional callback evidence. This verifies saved PCM, not playback or publication. */
class VerifiedWavEvidence {
    private data class Piece(val chunk: DeliveredChunk, val digest: ByteArray)
    private val pieces = mutableListOf<Piece>()
    private var rate = 0
    private var end = 0L
    private var completed: Boolean? = null
    private var invalid = false

    fun add(index: Int, text: String, seed: Long, draw: Int, stop: Int,
            start: Long, speechEnd: Long, samples: FloatArray, sampleRate: Int) {
        check(completed == null && !invalid)
        try {
            require(index == pieces.size && start == end && samples.isNotEmpty())
            require(sampleRate > 0 && (rate == 0 || rate == sampleRate))
            val next = Math.addExact(start, samples.size.toLong())
            require(speechEnd > start && speechEnd <= next)
            val stopValue = when (stop) { 0 -> ReadStop.EOS; 1 -> ReadStop.CAP; else -> throw IllegalArgumentException("invalid stop") }
            val hash = MessageDigest.getInstance("SHA-256")
            samples.forEachIndexed { i, x ->
                require(x.isFinite())
                if (start + i >= speechEnd) require(x == 0f)
                val value = (x.coerceIn(-1f, 1f) * 32767f).toInt().toShort().toInt()
                hash.update(value.toByte()); hash.update((value shr 8).toByte())
            }
            pieces.add(Piece(DeliveredChunk(index, text, draw, seed, stopValue,
                SampleSpan(start, next), SampleSpan(start, speechEnd)), hash.digest()))
            end = next; rate = sampleRate
        } catch (e: Exception) { invalid = true; throw e }
    }

    /** Call once with the stream result; exceptions must be reported as false. */
    fun complete(success: Boolean) {
        check(completed == null)
        completed = success && !invalid
    }

    /** Bind reopened bytes after validating PCM. This does not establish playback completion. */
    fun verify(readId: String, wav: ByteArray): AudioBoundReadManifest {
        check(completed == true && pieces.isNotEmpty()) { "stream is incomplete" }
        val decoded = MonoPcmWav.parse(wav)
        require(decoded.rate == rate && decoded.count == end) { "WAV rate or count differs" }
        var offset = decoded.data
        for (piece in pieces) {
            val hash = MessageDigest.getInstance("SHA-256")
            val count = piece.chunk.delivered.end - piece.chunk.delivered.start
            repeat(count.toInt()) {
                val lo = wav[offset++]; val hi = wav[offset++]
                hash.update(lo); hash.update(hi)
                if (piece.chunk.delivered.start + it >= piece.chunk.speech.end)
                    require(lo == 0.toByte() && hi == 0.toByte()) { "nonzero saved gap" }
            }
            require(MessageDigest.isEqual(hash.digest(), piece.digest)) { "WAV PCM differs from evidence" }
        }
        return AudioBoundReadManifest.create(readId, wav, decoded.rate, decoded.count, pieces.map { it.chunk })
    }

}
