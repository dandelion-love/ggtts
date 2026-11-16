package ggtts.demo

import java.security.MessageDigest

/** Canonical shared-page input; this alone does not establish playback or persist feedback. */
data class FeedbackPageInput(val canonicalJson: String, val sha256: String)

object FeedbackInputAdapter {
    private val idPattern = Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    private const val alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

    /** Build shared ggtts-feedback-input/1 from validated saved PCM and a page-safe stable ID. */
    fun fromWav(wav: ByteArray, manifest: AudioBoundReadManifest, pageId: String): FeedbackPageInput {
        require(idPattern.matches(pageId)) { "invalid feedback page ID" }
        val clips = FeedbackPcmClips.fromWav(wav, manifest)
        require(clips.isNotEmpty())
        val json = buildString {
            // Keys are in Python sort_keys order, with no insignificant whitespace.
            append("{\"format\":\"ggtts-feedback-input/1\",\"id\":")
            appendJson(pageId)
            append(",\"items\":[")
            clips.forEachIndexed { i, clip ->
                if (i != 0) append(',')
                append("{\"audio\":")
                appendJson("data:audio/flac;base64," + base64(clip.flac))
                append(",\"id\":")
                appendJson("chunk-" + i)
                append(",\"text\":")
                appendJson(clip.chunk.text)
                append('}')
            }
            append("]}")
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(json.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        return FeedbackPageInput(json, digest)
    }

    /** Python ensure_ascii with HTML-sensitive characters escaped after JSON encoding. */
    private fun StringBuilder.appendJson(value: String) {
        append('"')
        var i = 0
        while (i < value.length) {
            val c = value[i]
            if (Character.isHighSurrogate(c)) {
                require(i + 1 < value.length && Character.isLowSurrogate(value[i + 1])) { "invalid Unicode" }
                appendUnicode(c)
                appendUnicode(value[++i])
            } else {
                require(!Character.isLowSurrogate(c)) { "invalid Unicode" }
                when (c) {
                    '"' -> append("\\\"")
                    '\\' -> append("\\\\")
                    '\b' -> append("\\b")
                    '\u000c' -> append("\\f")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> if (c.code < 32 || c.code >= 127 || c == '<' || c == '>' || c == '&') {
                        appendUnicode(c)
                    } else append(c)
                }
            }
            i++
        }
        append('"')
    }

    private fun StringBuilder.appendUnicode(c: Char) {
        val hex = "0123456789abcdef"
        append("\\u")
        for (shift in intArrayOf(12, 8, 4, 0)) append(hex[(c.code shr shift) and 15])
    }

    private fun base64(bytes: ByteArray): String = buildString {
        var i = 0
        while (i < bytes.size) {
            val a = bytes[i++].toInt() and 255
            val b = if (i < bytes.size) bytes[i++].toInt() and 255 else -1
            val c = if (i < bytes.size) bytes[i++].toInt() and 255 else -1
            append(alphabet[a shr 2])
            append(alphabet[((a and 3) shl 4) or (if (b >= 0) b shr 4 else 0)])
            append(if (b >= 0) alphabet[((b and 15) shl 2) or (if (c >= 0) c shr 6 else 0)] else '=')
            append(if (c >= 0) alphabet[c and 63] else '=')
        }
    }
}
