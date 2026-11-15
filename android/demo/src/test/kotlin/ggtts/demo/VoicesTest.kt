package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VoicesTest {
    private val talker = ModelFile("talker.gguf", Kind.QWEN_TALKER, listOf("hk_voice_777", "hk_voice_343"))
    private val codec = ModelFile("codec.gguf", Kind.QWEN_CODEC)
    private val lm = ModelFile("lm.gguf", Kind.KITTEN_LM)
    private val decoder = ModelFile("s3gen.gguf", Kind.KITTEN_DECODER)
    private val voiceJson = ModelFile("voice.json", Kind.KITTEN_VOICE)

    @Test
    fun a_talker_and_its_codec_give_one_voice_per_speaker_in_order() {
        val found = Voices.group(listOf(talker, codec))
        assertEquals(listOf("hk_voice_343", "hk_voice_777"), found.voices.map { it.name })
        val v = found.first!!
        assertEquals("qwen3-tts", v.engine)
        assertEquals("talker.gguf", v.model)
        assertEquals("codec.gguf", v.codec)
        assertEquals("hk_voice_343", v.speaker)
        assertNull(v.voiceFile)
        assertEquals("hk_voice_343 · Cantonese · Qwen3-TTS", v.label)
        assertTrue(found.incomplete.isEmpty())
    }

    @Test
    fun a_kitten_model_gives_one_voice_per_voice_json_and_names_the_model_only_when_there_are_several() {
        val one = Voices.group(listOf(lm, decoder, voiceJson, ModelFile("yue.json", Kind.KITTEN_VOICE)))
        assertEquals(listOf("voice", "yue"), one.voices.map { it.name })
        assertEquals("s3gen.gguf", one.first!!.codec)
        assertEquals("voice.json", one.first!!.voiceFile)
        val two = Voices.group(listOf(lm, ModelFile("lm2.gguf", Kind.KITTEN_LM), decoder, voiceJson))
        assertEquals(listOf("voice (lm)", "voice (lm2)"), two.voices.map { it.name })
    }

    @Test
    fun incomplete_bundles_are_listed_with_what_they_lack_and_give_no_voice() {
        val found = Voices.group(listOf(talker, lm, voiceJson, ModelFile("notes.json", Kind.OTHER)))
        assertTrue(found.voices.isEmpty())
        assertNull(found.first)
        assertEquals(
            listOf(Incomplete("talker.gguf", "a Qwen3-TTS 12 Hz codec"), Incomplete("lm.gguf", "a Kitten-TTS 2 S3Gen decoder")),
            found.incomplete,
        )
        assertEquals(listOf(Incomplete("voice.json", "a Kitten-TTS 2 language model")),
            Voices.group(listOf(voiceJson, decoder)).incomplete)
        assertEquals(listOf(Incomplete("lm.gguf", "a prepared voice JSON")), Voices.group(listOf(lm, decoder)).incomplete)
        assertEquals(listOf(Incomplete("t.gguf", "a named speaker (it has none)")),
            Voices.group(listOf(ModelFile("t.gguf", Kind.QWEN_TALKER), codec)).incomplete)
    }

    @Test
    fun the_first_complete_voice_is_chosen_even_after_an_incomplete_one() {
        val found = Voices.group(listOf(talker, lm, decoder, voiceJson))
        assertEquals(listOf(Incomplete("talker.gguf", "a Qwen3-TTS 12 Hz codec")), found.incomplete)
        assertEquals("voice", found.first!!.name)
    }

    @Test
    fun a_sample_beside_a_voice_is_attached_to_it() {
        val found = Voices.group(listOf(talker, codec), samples = setOf("hk_voice_777.sample.wav"))
        assertEquals(listOf(null, "hk_voice_777.sample.wav"), found.voices.map { it.sample })
    }

    @Test
    fun the_header_reader_finds_the_architecture_and_speakers_past_other_values() {
        val bytes = gguf(
            "general.architecture" to "qwen3-tts",
            "general.name" to "x",
            "qwen3-tts.codec.language_ids" to intArrayOf(1, 2, 3),
            "qwen3-tts.codec.language_names" to listOf("chinese", "english"),
            "qwen3-tts.block_count" to 28,
            GgufInfo.SPEAKER_NAMES to listOf("hk_voice_343", "hk_voice_777"),
        )
        val info = GgufInfo.read(ByteArrayInputStream(bytes))
        assertEquals(GgufInfo("qwen3-tts", listOf("hk_voice_343", "hk_voice_777")), info)
        assertEquals(Kind.QWEN_TALKER, info.toModelFile("t.gguf").kind)
        assertEquals(Kind.KITTEN_DECODER, GgufInfo.read(ByteArrayInputStream(gguf("general.architecture" to "s3gen-meanflow")))
            .toModelFile("d.gguf").kind)
        assertEquals(Kind.OTHER, GgufInfo("llama", emptyList()).toModelFile("l.gguf").kind)
    }

    @Test
    fun a_voice_json_is_recognised_by_its_fields() {
        assertTrue(Voices.isKittenVoice("""{"embedding":[0.1],"prepared_by":"x","s3gen":{"embedding":[1]}}"""))
        assertFalse(Voices.isKittenVoice("""{"name":"not a voice"}"""))
    }

    /** A GGUF v3 header with the given key-values and no tensors. */
    private fun gguf(vararg kv: Pair<String, Any>): ByteArray {
        val out = ByteArrayOutputStream()
        fun le(n: Int, v: Long) = out.write(ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN).apply {
            if (n == 4) putInt(v.toInt()) else putLong(v)
        }.array())
        fun str(s: String) { val b = s.toByteArray(); le(8, b.size.toLong()); out.write(b) }
        out.write("GGUF".toByteArray()); le(4, 3); le(8, 0); le(8, kv.size.toLong())
        for ((k, v) in kv) {
            str(k)
            when (v) {
                is String -> { le(4, 8); str(v) }
                is Int -> { le(4, 4); le(4, v.toLong()) }
                is IntArray -> { le(4, 9); le(4, 5); le(8, v.size.toLong()); v.forEach { le(4, it.toLong()) } }
                is List<*> -> { le(4, 9); le(4, 8); le(8, v.size.toLong()); v.forEach { str(it as String) } }
            }
        }
        return out.toByteArray()
    }
}
