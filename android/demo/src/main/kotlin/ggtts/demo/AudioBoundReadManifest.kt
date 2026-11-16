package ggtts.demo

import java.math.BigInteger
import java.security.MessageDigest
import java.util.Collections

/** An absolute half-open range of samples in the final saved mono audio. */
data class SampleSpan(val start: Long, val end: Long)

enum class ReadStop(val wireValue: String) {
    EOS("eos"), CAP("cap");

    companion object {
        fun fromWire(value: String): ReadStop =
            values().firstOrNull { it.wireValue == value }
                ?: throw IllegalArgumentException("unsupported read stop: " + value)
    }
}

/** Seed bits are held in a signed JVM Long and encoded as unsigned decimal. */
data class DeliveredChunk(
    val index: Int,
    val text: String,
    val draw: Int,
    val seedBits: Long,
    val stop: ReadStop,
    val delivered: SampleSpan,
    val speech: SampleSpan,
)

/** Exact byte identity and decoded mono sample facts for the saved audio. */
data class ReadAudioIdentity(
    val sha256: String,
    val byteLength: Long,
    val sampleRate: Int,
    val sampleCount: Long,
)

/**
 * Versioned evidence binding delivered chunks to one saved audio file.
 * Legacy audio without this record remains playable but cannot be represented as feedback-ready by inventing evidence.
 */
class AudioBoundReadManifest private constructor(
    val readId: String,
    val audio: ReadAudioIdentity,
    chunks: List<DeliveredChunk>,
) {
    val chunks: List<DeliveredChunk> = Collections.unmodifiableList(chunks.toList())

    override fun equals(other: Any?): Boolean =
        other is AudioBoundReadManifest &&
            readId == other.readId &&
            audio == other.audio &&
            chunks == other.chunks

    override fun hashCode(): Int = 31 * (31 * readId.hashCode() + audio.hashCode()) + chunks.hashCode()

    /**
     * Compare exact stored bytes and independently decoded sample facts to this record.
     * This does not decode audio, inspect PCM, verify silence gaps, or prove playback.
     */
    fun validateAudio(audioBytes: ByteArray, sampleRate: Int, sampleCount: Long) {
        require(audioBytes.isNotEmpty()) { "audio is empty" }
        require(audioBytes.size.toLong() == audio.byteLength) { "audio byte length does not match manifest" }
        require(sha256(audioBytes) == audio.sha256) { "audio SHA-256 does not match manifest" }
        require(sampleRate == audio.sampleRate) { "audio sample rate does not match manifest" }
        require(sampleCount == audio.sampleCount) { "audio sample count does not match manifest" }
    }

    /** Stable JSON; text is kept exactly and is never trimmed or reconstructed. */
    fun encodeJson(): String = buildString {
        append("{\"format\":")
        append(jsonQuote(FORMAT))
        append(",\"version\":1,\"read_id\":")
        append(jsonQuote(readId))
        append(",\"audio\":{\"sha256\":")
        append(jsonQuote(audio.sha256))
        append(",\"byte_length\":")
        append(audio.byteLength)
        append(",\"sample_rate\":")
        append(audio.sampleRate)
        append(",\"sample_count\":")
        append(audio.sampleCount)
        append("},\"chunks\":[")
        chunks.forEachIndexed { i, chunk ->
            if (i != 0) append(',')
            append("{\"index\":")
            append(chunk.index)
            append(",\"text\":")
            append(jsonQuote(chunk.text))
            append(",\"draw\":")
            append(chunk.draw)
            append(",\"seed_u64\":")
            append(jsonQuote(unsignedDecimal(chunk.seedBits)))
            append(",\"stop\":")
            append(jsonQuote(chunk.stop.wireValue))
            append(",\"delivered\":")
            append(spanJson(chunk.delivered))
            append(",\"speech\":")
            append(spanJson(chunk.speech))
            append('}')
        }
        append("]}")
    }

    companion object {
        const val FORMAT = "ggtts-audio-read-manifest"
        const val VERSION = 1
        private val ID = Regex("^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
        private val SHA256 = Regex("^[0-9a-f]{64}$")
        private val TWO_63 = BigInteger.ONE.shiftLeft(63)
        private val TWO_64 = BigInteger.ONE.shiftLeft(64)
        private val MAX_U64 = TWO_64.subtract(BigInteger.ONE)

        /** Call only after final audio is written successfully. audioBytes are the exact bytes stored. */
        fun create(
            readId: String,
            audioBytes: ByteArray,
            sampleRate: Int,
            sampleCount: Long,
            chunks: List<DeliveredChunk>,
        ): AudioBoundReadManifest {
            require(audioBytes.isNotEmpty()) { "audio is empty" }
            require(sampleRate > 0) { "sample rate must be positive" }
            require(sampleCount > 0) { "sample count must be positive" }
            return AudioBoundReadManifest(
                readId,
                ReadAudioIdentity(sha256(audioBytes), audioBytes.size.toLong(), sampleRate, sampleCount),
                chunks,
            ).also { it.validate() }
        }

        /** Strict schema v1 decoder. Unknown, duplicate, missing, or mistyped fields are refused. */
        fun decodeJson(json: String): AudioBoundReadManifest {
            try {
                val root = JsonReader(json).readDocument().asObject("manifest")
                root.requireKeys(setOf("format", "version", "read_id", "audio", "chunks"), "manifest")
                require(root.string("format") == FORMAT) { "unsupported manifest format" }
                require(root.long("version") == VERSION.toLong()) { "unsupported manifest version" }

                val audioObject = root.objectValue("audio")
                audioObject.requireKeys(setOf("sha256", "byte_length", "sample_rate", "sample_count"), "audio")
                val audio = ReadAudioIdentity(
                    audioObject.string("sha256"),
                    audioObject.long("byte_length"),
                    audioObject.int("sample_rate"),
                    audioObject.long("sample_count"),
                )
                val chunks = root.arrayValue("chunks").mapIndexed { index, raw ->
                    val item = raw.asObject("chunks[" + index + "]")
                    item.requireKeys(
                        setOf("index", "text", "draw", "seed_u64", "stop", "delivered", "speech"),
                        "chunks[" + index + "]",
                    )
                    DeliveredChunk(
                        index = item.int("index"),
                        text = item.string("text"),
                        draw = item.int("draw"),
                        seedBits = parseUnsignedDecimal(item.string("seed_u64")),
                        stop = ReadStop.fromWire(item.string("stop")),
                        delivered = decodeSpan(item.objectValue("delivered"), "delivered"),
                        speech = decodeSpan(item.objectValue("speech"), "speech"),
                    )
                }
                return AudioBoundReadManifest(root.string("read_id"), audio, chunks).also { it.validate() }
            } catch (e: IllegalArgumentException) {
                throw e
            } catch (e: Exception) {
                throw IllegalArgumentException("invalid audio-bound read manifest", e)
            }
        }

        private fun decodeSpan(value: Map<String, Any?>, path: String): SampleSpan {
            value.requireKeys(setOf("start", "end"), path)
            return SampleSpan(value.long("start"), value.long("end"))
        }

        private fun AudioBoundReadManifest.validate() {
            require(ID.matches(readId)) { "read ID must be a stable opaque token (1 to 128 safe ASCII characters)" }
            require(SHA256.matches(audio.sha256)) { "audio SHA-256 must be 64 lowercase hexadecimal characters" }
            require(audio.byteLength > 0) { "audio byte length must be positive" }
            require(audio.sampleRate > 0) { "audio sample rate must be positive" }
            require(audio.sampleCount > 0) { "audio sample count must be positive" }
            require(chunks.isNotEmpty()) { "manifest has no delivered chunk evidence" }

            var expectedStart = 0L
            for ((position, chunk) in chunks.withIndex()) {
                require(chunk.index == position) { "chunk indices must be contiguous and ordered from zero" }
                require(chunk.text.isNotEmpty()) { "chunk text must not be empty" }
                requireWellFormedUnicode(chunk.text)
                require(chunk.draw >= 0) { "kept draw must be nonnegative" }

                val delivered = chunk.delivered
                val speech = chunk.speech
                require(delivered.start == expectedStart) { "delivered chunks must be contiguous without overlap or gaps" }
                require(delivered.start >= 0 && delivered.end > delivered.start) { "invalid delivered span at chunk " + position }
                require(delivered.end <= audio.sampleCount) { "delivered span exceeds audio sample count" }
                require(speech.start == delivered.start) { "speech span must begin at delivered chunk start" }
                require(speech.end > speech.start && speech.end <= delivered.end) {
                    "speech span is outside delivered span at chunk " + position
                }
                expectedStart = delivered.end
            }
            require(expectedStart == audio.sampleCount) { "delivered spans do not cover the complete saved audio" }
        }

        private fun parseUnsignedDecimal(value: String): Long {
            require(Regex("^(0|[1-9][0-9]{0,19})$").matches(value)) {
                "seed_u64 must be canonical unsigned decimal"
            }
            val unsigned = BigInteger(value)
            require(unsigned <= MAX_U64) { "seed_u64 exceeds unsigned 64-bit range" }
            val signed = if (unsigned >= TWO_63) unsigned.subtract(TWO_64) else unsigned
            return signed.toLong()
        }

        private fun unsignedDecimal(bits: Long): String {
            val signed = BigInteger.valueOf(bits)
            return if (bits >= 0) signed.toString() else signed.add(TWO_64).toString()
        }

        private fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

        private fun spanJson(span: SampleSpan): String =
            "{\"start\":" + span.start + ",\"end\":" + span.end + "}"

        private fun jsonQuote(value: String): String {
            requireWellFormedUnicode(value)
            val out = StringBuilder(value.length + 2).append('"')
            for (c in value) {
                when (c) {
                    '"' -> out.append("\\\"")
                    '\\' -> out.append("\\\\")
                    '\b' -> out.append("\\b")
                    '\u000c' -> out.append("\\f")
                    '\n' -> out.append("\\n")
                    '\r' -> out.append("\\r")
                    '\t' -> out.append("\\t")
                    else -> if (c.code < 0x20) out.append("\\u%04x".format(c.code)) else out.append(c)
                }
            }
            return out.append('"').toString()
        }
    }
}

private fun requireWellFormedUnicode(text: String) {
    var i = 0
    while (i < text.length) {
        val c = text[i]
        if (c.isHighSurrogate()) {
            require(i + 1 < text.length && text[i + 1].isLowSurrogate()) {
                "text contains an unpaired high surrogate"
            }
            i += 2
        } else {
            require(!c.isLowSurrogate()) { "text contains an unpaired low surrogate" }
            i++
        }
    }
}

/** Minimal strict JSON reader for this small schema; avoids Android-only JSON runtime behavior in JVM tests. */
internal class JsonReader(private val source: String) {
    private var at = 0

    fun readDocument(): Any? {
        skipWhitespace()
        val value = readValue(0)
        skipWhitespace()
        require(at == source.length) { "trailing data after JSON document" }
        return value
    }

    private fun readValue(depth: Int): Any? {
        require(depth <= 64) { "JSON nesting is too deep" }
        skipWhitespace()
        require(at < source.length) { "unexpected end of JSON" }
        return when (source[at]) {
            '{' -> readObject(depth + 1)
            '[' -> readArray(depth + 1)
            '"' -> readString()
            't' -> { literal("true"); true }
            'f' -> { literal("false"); false }
            'n' -> { literal("null"); null }
            '-', in '0'..'9' -> readInteger()
            else -> throw IllegalArgumentException("invalid JSON value at offset " + at)
        }
    }

    private fun readObject(depth: Int): Map<String, Any?> {
        expect('{')
        skipWhitespace()
        val result = linkedMapOf<String, Any?>()
        if (take('}')) return result
        while (true) {
            skipWhitespace()
            require(at < source.length && source[at] == '"') { "object key must be a string" }
            val key = readString()
            skipWhitespace()
            expect(':')
            val value = readValue(depth)
            require(!result.containsKey(key)) { "duplicate JSON field " + key }
            result[key] = value
            skipWhitespace()
            if (take('}')) return result
            expect(',')
        }
    }

    private fun readArray(depth: Int): List<Any?> {
        expect('[')
        skipWhitespace()
        val result = mutableListOf<Any?>()
        if (take(']')) return result
        while (true) {
            result += readValue(depth)
            skipWhitespace()
            if (take(']')) return result
            expect(',')
        }
    }

    private fun readString(): String {
        expect('"')
        val out = StringBuilder()
        while (at < source.length) {
            val c = source[at++]
            when {
                c == '"' -> {
                    val result = out.toString()
                    requireWellFormedUnicode(result)
                    return result
                }
                c == '\\' -> {
                    require(at < source.length) { "unfinished JSON escape" }
                    when (val escaped = source[at++]) {
                        '"', '\\', '/' -> out.append(escaped)
                        'b' -> out.append('\b')
                        'f' -> out.append('\u000c')
                        'n' -> out.append('\n')
                        'r' -> out.append('\r')
                        't' -> out.append('\t')
                        'u' -> {
                            require(at + 4 <= source.length) { "short unicode escape" }
                            val digits = source.substring(at, at + 4)
                            require(digits.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) {
                                "invalid unicode escape"
                            }
                            out.append(digits.toInt(16).toChar())
                            at += 4
                        }
                        else -> throw IllegalArgumentException("invalid JSON escape")
                    }
                }
                c.code < 0x20 -> throw IllegalArgumentException("unescaped control character in JSON string")
                else -> out.append(c)
            }
        }
        throw IllegalArgumentException("unterminated JSON string")
    }

    private fun readInteger(): Long {
        val start = at
        if (take('-')) require(at < source.length) { "unfinished JSON number" }
        if (take('0')) {
            require(at == source.length || source[at] !in '0'..'9') { "leading zero in JSON number" }
        } else {
            require(at < source.length && source[at] in '1'..'9') { "invalid JSON number" }
            while (at < source.length && source[at] in '0'..'9') at++
        }
        require(at == source.length || source[at] !in ".eE") { "manifest numeric fields must be integers" }
        return source.substring(start, at).toLongOrNull()
            ?: throw IllegalArgumentException("JSON integer is outside signed 64-bit range")
    }

    private fun literal(value: String) {
        require(source.startsWith(value, at)) { "invalid JSON literal" }
        at += value.length
    }

    private fun skipWhitespace() {
        while (at < source.length && (source[at] == ' ' || source[at] == '\t' ||
                source[at] == '\r' || source[at] == '\n')) at++
    }

    private fun expect(c: Char) {
        require(take(c)) { "expected " + c + " at offset " + at }
    }

    private fun take(c: Char): Boolean {
        if (at >= source.length || source[at] != c) return false
        at++
        return true
    }
}

private fun Any?.asObject(path: String): Map<String, Any?> =
    this as? Map<String, Any?> ?: throw IllegalArgumentException(path + " must be an object")

private fun Map<String, Any?>.requireKeys(expected: Set<String>, path: String) {
    require(keys == expected) { path + " fields must be exactly " + expected.sorted() }
}

private fun Map<String, Any?>.string(key: String): String =
    this[key] as? String ?: throw IllegalArgumentException(key + " must be a string")

private fun Map<String, Any?>.long(key: String): Long =
    this[key] as? Long ?: throw IllegalArgumentException(key + " must be an integer")

private fun Map<String, Any?>.int(key: String): Int {
    val value = long(key)
    require(value in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) { key + " is outside the integer range" }
    return value.toInt()
}

private fun Map<String, Any?>.objectValue(key: String): Map<String, Any?> = this[key].asObject(key)

private fun Map<String, Any?>.arrayValue(key: String): List<Any?> =
    this[key] as? List<Any?> ?: throw IllegalArgumentException(key + " must be an array")
