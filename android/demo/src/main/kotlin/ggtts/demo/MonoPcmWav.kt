package ggtts.demo

/** Strict mono PCM16 WAV parser shared by verification and extraction. */
internal object MonoPcmWav {
    data class Wav(val rate: Int, val count: Long, val data: Int)
    fun parse(b: ByteArray): Wav {
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
