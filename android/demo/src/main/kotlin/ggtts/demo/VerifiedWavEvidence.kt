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
        val decoded = parseWav(wav)
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

    private data class Wav(val rate: Int, val count: Long, val data: Int)
    private fun parseWav(b: ByteArray): Wav {
        fun u16(p: Int): Int = (b[p].toInt() and 255) or ((b[p + 1].toInt() and 255) shl 8)
        fun u32(p: Int): Long = u16(p).toLong() or (u16(p + 2).toLong() shl 16)
        fun tag(p: Int, s: String) = s.indices.all { b[p + it].toInt() == s[it].code }
        require(b.size >= 12 && tag(0, "RIFF") && tag(8, "WAVE"))
        require(u32(4) == b.size.toLong() - 8)
        var pos = 12; var sampleRate = 0; var data = -1; var size = -1L; var formatSeen = false
        while (pos < b.size) {
            require(b.size - pos >= 8)
            val length = u32(pos + 4); val body = pos.toLong() + 8
            val next = body + length + (length and 1)
            require(next <= b.size.toLong())
            if (tag(pos, "fmt ")) {
                require(!formatSeen && length == 16L) { "unsupported PCM fmt chunk" }
                formatSeen = true
                val p = body.toInt(); val r = u32(p + 4)
                require(u16(p) == 1 && u16(p + 2) == 1 && r in 1..Int.MAX_VALUE.toLong() &&
                    u32(p + 8) == r * 2 && u16(p + 12) == 2 && u16(p + 14) == 16)
                sampleRate = r.toInt()
            } else if (tag(pos, "data")) {
                require(data < 0 && length % 2 == 0L)
                data = body.toInt(); size = length
            }
            pos = next.toInt()
        }
        require(formatSeen && data >= 0 && size > 0)
        return Wav(sampleRate, size / 2, data)
    }
}
