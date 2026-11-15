// What the screen offers first: voices, not an engine and its files. A voice is a bundle of model files that speak
// together, found by reading what each file in the models directory says it is (its GGUF general.architecture, a
// talker's speaker names, a voice JSON's fields), never by its file name.
package ggtts.demo

import java.io.BufferedInputStream
import java.io.EOFException
import java.io.File
import java.io.InputStream

/** What one file in the models directory is. */
enum class Kind { QWEN_TALKER, QWEN_CODEC, KITTEN_LM, KITTEN_DECODER, KITTEN_VOICE, OTHER }

/** One model file as [Voices.group] needs it: its name, what it is and, for a Qwen3-TTS talker, its speakers. */
data class ModelFile(val name: String, val kind: Kind, val speakers: List<String> = emptyList())

/**
 * A voice: the files that speak together, a name a person can choose by, the language its engine reads (the one
 * Android is told, [SystemVoices]), and a sample to play when one sits beside it in the models directory.
 */
data class Voice(
    val name: String,
    val engine: String,
    val language: String,
    val model: String,
    val codec: String,
    val voiceFile: String? = null,
    val speaker: String? = null,
    val sample: String? = null,
) {
    /** The one line a list shows. */
    val label: String get() = "$name · $language · ${Voices.ENGINE_NAMES[engine] ?: engine}"
}

/** A line to try a voice with, by the language it reads; put in the text box until the user types their own. */
val EXAMPLES = mapOf(
    "Cantonese" to "你好！今日天氣幾好，不如一齊出去行吓街，食個下午茶。",
    "English" to "Hello! This voice runs entirely on your phone, with no internet connection.",
)

/** Files that cannot speak yet, and what they lack. */
data class Incomplete(val name: String, val missing: String)

data class Found(val voices: List<Voice>, val incomplete: List<Incomplete>) {
    /** The voice selected when the screen opens: the first complete one, if any. */
    val first: Voice? get() = voices.firstOrNull()
}

object Voices {
    val ENGINE_NAMES = mapOf("qwen3-tts" to "Qwen3-TTS", "kitten-tts-2" to "Kitten-TTS 2")
    /** The language each engine reads, as its system voice declares it. Model files do not say. */
    val LANGUAGES = mapOf("qwen3-tts" to "Cantonese", "kitten-tts-2" to "English")
    /** A sample for a voice is `<voice name>.sample.wav` in the models directory. */
    const val SAMPLE_SUFFIX = ".sample.wav"

    /**
     * The voices the files make, in a stable order, and the files that make none. A Qwen3-TTS talker gives one voice
     * per speaker it names, with the first codec; a Kitten-TTS 2 language model gives one voice per voice JSON, with
     * the first decoder. A talker without a codec, a talker with no speakers, a language model without a decoder or a
     * voice, and a voice JSON without a language model are incomplete.
     */
    fun group(files: List<ModelFile>, samples: Set<String> = emptySet()): Found {
        val sorted = files.sortedBy { it.name }
        fun of(k: Kind) = sorted.filter { it.kind == k }
        fun sample(name: String) = (name + SAMPLE_SUFFIX).takeIf { it in samples }
        val voices = mutableListOf<Voice>()
        val incomplete = mutableListOf<Incomplete>()

        val codec = of(Kind.QWEN_CODEC).firstOrNull()
        for (t in of(Kind.QWEN_TALKER)) {
            when {
                codec == null -> incomplete += Incomplete(t.name, "a Qwen3-TTS 12 Hz codec")
                t.speakers.isEmpty() -> incomplete += Incomplete(t.name, "a named speaker (it has none)")
                else -> t.speakers.sorted().forEach { s ->
                    voices += Voice(s, "qwen3-tts", LANGUAGES.getValue("qwen3-tts"), t.name, codec.name,
                        speaker = s, sample = sample(s))
                }
            }
        }

        val decoder = of(Kind.KITTEN_DECODER).firstOrNull()
        val lms = of(Kind.KITTEN_LM)
        val jsons = of(Kind.KITTEN_VOICE)
        for (lm in lms) {
            when {
                decoder == null -> incomplete += Incomplete(lm.name, "a Kitten-TTS 2 S3Gen decoder")
                jsons.isEmpty() -> incomplete += Incomplete(lm.name, "a prepared voice JSON")
                else -> jsons.forEach { j ->
                    val stem = j.name.removeSuffix(".json")
                    // With one language model the voice's name is enough; with several, say which model it uses.
                    val name = if (lms.size == 1) stem else "$stem (${lm.name.removeSuffix(".gguf")})"
                    voices += Voice(name, "kitten-tts-2", LANGUAGES.getValue("kitten-tts-2"), lm.name, decoder.name,
                        voiceFile = j.name, sample = sample(stem))
                }
            }
        }
        if (lms.isEmpty()) jsons.forEach { incomplete += Incomplete(it.name, "a Kitten-TTS 2 language model") }
        return Found(voices, incomplete)
    }

    /** Reads what each listed file is. A file that cannot be read is [Kind.OTHER]. */
    fun scan(dir: File, names: List<String>): List<ModelFile> = names.map { name ->
        val f = File(dir, name)
        runCatching {
            when {
                name.endsWith(".gguf", true) -> f.inputStream().use { GgufInfo.read(it) }.toModelFile(name)
                name.endsWith(".json", true) -> ModelFile(name, if (isKittenVoice(f.readText())) Kind.KITTEN_VOICE else Kind.OTHER)
                else -> ModelFile(name, Kind.OTHER)
            }
        }.getOrElse { ModelFile(name, Kind.OTHER) }
    }

    /** A prepared Kitten-TTS 2 voice carries both its speaker embedding and its decoder's. */
    fun isKittenVoice(json: String): Boolean = "\"embedding\"" in json && "\"s3gen\"" in json
}

/** The GGUF header fields [Voices] needs. */
data class GgufInfo(val architecture: String?, val speakers: List<String>) {
    fun toModelFile(name: String) = ModelFile(
        name,
        when (architecture) {
            "qwen3-tts" -> Kind.QWEN_TALKER
            "qwen3-tts-tokenizer" -> Kind.QWEN_CODEC
            "kitten-tts-2" -> Kind.KITTEN_LM
            "s3gen-meanflow" -> Kind.KITTEN_DECODER
            else -> Kind.OTHER
        },
        speakers,
    )

    companion object {
        const val SPEAKER_NAMES = "qwen3-tts.codec.speaker_names"
        private const val STRING = 8
        private const val ARRAY = 9
        /** Byte sizes of the fixed-size GGUF value types, by type id. */
        private val FIXED = mapOf(0 to 1, 1 to 1, 2 to 2, 3 to 2, 4 to 4, 5 to 4, 6 to 4, 7 to 1, 10 to 8, 11 to 8, 12 to 8)

        /** Reads the key-value section only (never the tensors), so a 2 GB model costs its header. */
        fun read(input: InputStream): GgufInfo {
            val r = Le(BufferedInputStream(input, 1 shl 16))
            require(r.bytes(4).contentEquals("GGUF".toByteArray())) { "not a GGUF file" }
            val version = r.u32()
            require(version >= 2) { "GGUF version $version is too old" }
            r.u64() // tensor count
            val kvs = r.u64()
            var arch: String? = null
            var speakers = emptyList<String>()
            for (i in 0 until kvs) {
                val key = r.string()
                val type = r.u32().toInt()
                when {
                    key == "general.architecture" && type == STRING -> arch = r.string()
                    key == SPEAKER_NAMES && type == ARRAY -> {
                        val et = r.u32().toInt()
                        val n = r.u64()
                        speakers = if (et == STRING) (0 until n).map { r.string() } else { skipArray(r, et, n); emptyList() }
                    }
                    else -> skip(r, type)
                }
                if (arch != null && (arch != "qwen3-tts" || speakers.isNotEmpty())) break
            }
            return GgufInfo(arch, speakers)
        }

        private fun skip(r: Le, type: Int) {
            when (type) {
                STRING -> r.skip(r.u64())
                ARRAY -> { val et = r.u32().toInt(); skipArray(r, et, r.u64()) }
                else -> r.skip((FIXED[type] ?: error("unknown GGUF type $type")).toLong())
            }
        }

        private fun skipArray(r: Le, et: Int, n: Long) {
            val size = FIXED[et]
            if (size != null) r.skip(size * n) else for (i in 0 until n) skip(r, et)
        }
    }
}

/** Little-endian reads from a stream. */
private class Le(private val s: InputStream) {
    fun bytes(n: Int): ByteArray {
        val b = ByteArray(n)
        var off = 0
        while (off < n) {
            val k = s.read(b, off, n - off)
            if (k < 0) throw EOFException()
            off += k
        }
        return b
    }
    fun u32(): Long = bytes(4).foldIndexed(0L) { i, acc, b -> acc or ((b.toLong() and 0xff) shl (8 * i)) }
    fun u64(): Long = bytes(8).foldIndexed(0L) { i, acc, b -> acc or ((b.toLong() and 0xff) shl (8 * i)) }
    fun string(): String {
        val n = u64()
        require(n in 0..(1 shl 20)) { "GGUF string of $n bytes" }
        return String(bytes(n.toInt()), Charsets.UTF_8)
    }
    fun skip(n: Long) {
        var left = n
        while (left > 0) {
            val k = s.skip(left)
            if (k <= 0) { if (s.read() < 0) throw EOFException(); left-- } else left -= k
        }
    }
}
