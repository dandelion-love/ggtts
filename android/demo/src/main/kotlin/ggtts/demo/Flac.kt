// FLAC for a saved read: mono 16-bit PCM in VERBATIM subframes (lossless, valid for any decoder, not compressed), the
// same encoding as the engine's own (crates/ggtts-core/src/flac.rs), with a VORBIS_COMMENT block carrying the read's
// TITLE, so a list of saved reads can show it after a restart from the file alone.
package ggtts.demo

import java.io.ByteArrayOutputStream

object Flac {
    /** Samples per FLAC block, as the engine writes them. */
    const val BLOCK = 4096
    const val MIME = "audio/flac"

    /** One float sample to 16-bit PCM: scaled by 32767, rounded half to even, saturated (as the engine does). */
    fun toI16(x: Float): Short =
        Math.rint(x.toDouble() * 32767.0).coerceIn(Short.MIN_VALUE.toDouble(), Short.MAX_VALUE.toDouble()).toInt().toShort()

    /** Encode mono [samples] at [rate] Hz as a FLAC stream, with [title] as its TITLE tag when given. */
    fun encode(samples: FloatArray, rate: Int, title: String? = null): ByteArray {
        require(rate in 1 until (1 shl 20)) { "sample rate $rate is out of FLAC's range" }
        val out = ByteArrayOutputStream()
        out.write("fLaC".toByteArray())

        // STREAMINFO, then VORBIS_COMMENT (the last metadata block).
        val si = Bits()
        si.put(0, 1); si.put(0, 7); si.put(34, 24)
        si.put(BLOCK.toLong(), 16); si.put(BLOCK.toLong(), 16)
        si.put(0, 24); si.put(0, 24)
        si.put(rate.toLong(), 20)
        si.put(0, 3) // channels - 1
        si.put(15, 5) // bits per sample - 1
        si.put(samples.size.toLong(), 36)
        si.put(0, 64); si.put(0, 64) // MD5 unknown
        out.write(si.bytes())

        val comment = vorbisComment(title)
        val vh = Bits()
        vh.put(1, 1); vh.put(4, 7); vh.put(comment.size.toLong(), 24)
        out.write(vh.bytes())
        out.write(comment)

        var index = 0L
        var at = 0
        while (at < samples.size) {
            val n = minOf(BLOCK, samples.size - at)
            val h = Bits()
            h.put(0x3ffe, 14); h.put(0, 1); h.put(0, 1)
            val sizeCode = if (n == BLOCK) 0b1100 else 0b0111
            h.put(sizeCode.toLong(), 4)
            h.put(0, 4) // sample rate from STREAMINFO
            h.put(0, 4) // mono
            h.put(0b100, 3) // 16 bits per sample
            h.put(0, 1)
            val frame = ByteArrayOutputStream()
            frame.write(h.bytes())
            frame.write(utf8Number(index))
            if (sizeCode == 0b0111) { frame.write((n - 1) shr 8); frame.write((n - 1) and 0xff) }
            frame.write(crc8(frame.toByteArray()).toInt())
            frame.write(0b0000_0010) // pad bit, VERBATIM subframe, no wasted bits
            for (i in at until at + n) {
                val s = toI16(samples[i]).toInt()
                frame.write((s shr 8) and 0xff); frame.write(s and 0xff)
            }
            val crc = crc16(frame.toByteArray())
            frame.write(crc shr 8); frame.write(crc and 0xff)
            out.write(frame.toByteArray())
            at += n
            index++
        }
        return out.toByteArray()
    }

    /** The TITLE tag of a FLAC stream written by [encode], or null. */
    fun title(flac: ByteArray): String? {
        var p = 4
        while (p + 4 <= flac.size) {
            val header = flac[p].toInt() and 0xff
            val len = ((flac[p + 1].toInt() and 0xff) shl 16) or ((flac[p + 2].toInt() and 0xff) shl 8) or (flac[p + 3].toInt() and 0xff)
            val type = header and 0x7f
            if (type == 4) {
                var q = p + 4
                fun le32(): Int { val v = (flac[q].toInt() and 0xff) or ((flac[q + 1].toInt() and 0xff) shl 8) or ((flac[q + 2].toInt() and 0xff) shl 16) or ((flac[q + 3].toInt() and 0xff) shl 24); q += 4; return v }
                val vendor = le32() // read first: le32 moves q past the length
                q += vendor
                repeat(le32()) {
                    val n = le32()
                    val c = String(flac, q, n, Charsets.UTF_8); q += n
                    if (c.startsWith("TITLE=", ignoreCase = true)) return c.substring(6)
                }
            }
            if (header and 0x80 != 0) return null
            p += 4 + len
        }
        return null
    }

    private fun vorbisComment(title: String?): ByteArray {
        val out = ByteArrayOutputStream()
        fun le32(v: Int) { out.write(v and 0xff); out.write((v shr 8) and 0xff); out.write((v shr 16) and 0xff); out.write((v shr 24) and 0xff) }
        val vendor = "ggtts".toByteArray()
        le32(vendor.size); out.write(vendor)
        val comments = listOfNotNull(title?.let { "TITLE=$it".toByteArray(Charsets.UTF_8) })
        le32(comments.size)
        for (c in comments) { le32(c.size); out.write(c) }
        return out.toByteArray()
    }

    private class Bits {
        private val out = ByteArrayOutputStream()
        private var acc = 0
        private var n = 0
        fun put(value: Long, bits: Int) {
            for (i in bits - 1 downTo 0) {
                acc = (acc shl 1) or ((value ushr i) and 1L).toInt()
                if (++n == 8) { out.write(acc); acc = 0; n = 0 }
            }
        }
        fun bytes(): ByteArray { check(n == 0) { "FLAC structures end on a byte boundary" }; return out.toByteArray() }
    }

    fun crc8(data: ByteArray): Byte {
        var crc = 0
        for (b in data) {
            crc = crc xor (b.toInt() and 0xff)
            repeat(8) { crc = if (crc and 0x80 != 0) ((crc shl 1) xor 0x07) and 0xff else (crc shl 1) and 0xff }
        }
        return crc.toByte()
    }

    fun crc16(data: ByteArray): Int {
        var crc = 0
        for (b in data) {
            crc = crc xor ((b.toInt() and 0xff) shl 8)
            repeat(8) { crc = if (crc and 0x8000 != 0) ((crc shl 1) xor 0x8005) and 0xffff else (crc shl 1) and 0xffff }
        }
        return crc
    }

    fun utf8Number(v: Long): ByteArray {
        if (v < 0x80) return byteArrayOf(v.toByte())
        var n = 2
        while (n < 7 && v >= (1L shl (5 * n + 1))) n++
        val out = ByteArray(n)
        var rest = v
        for (i in n - 1 downTo 1) { out[i] = (0x80 or (rest and 0x3f).toInt()).toByte(); rest = rest ushr 6 }
        out[0] = ((0xff00 shr n) or rest.toInt()).toByte()
        return out
    }
}
