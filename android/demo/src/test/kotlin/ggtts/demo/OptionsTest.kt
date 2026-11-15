// The app's options document and the three documents a request hands the binding (load, a whole render, a stream).
package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class OptionsTest {
    private val qwen = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343", 4)

    @Test
    fun aDocumentRoundTripsThroughItsJson() {
        val o = Options.EMPTY
            .with("load", "engine", "qwen3-tts")
            .with("load", "threads", 6)
            .with("render", "instruction", "Say \"hi\"\n\\ slowly.")
            .with("render", "steer_scale", 0.5)
            .with("render", "steer_layers", listOf(10, 11))
            .with("output", "stream_raw", true)
        assertEquals(o, Options.parse(o.toJson()))
        assertEquals(6L, Options.parse(o.toJson())["load", "threads"])
        assertEquals("keys are sorted, so equal documents write the same JSON",
            """{"load": {"engine": "qwen3-tts", "threads": 6}, "output": {"stream_raw": true}, """ +
                """"render": {"instruction": "Say \"hi\"\n\\ slowly.", "steer_layers": [10, 11], "steer_scale": 0.5}}""",
            o.toJson())
    }

    @Test
    fun anUnknownKeyIsKeptForTheBindingToRefuse() {
        val o = Options.parse("""{"load": {"nonesuch": 1}, "elsewhere": {"x": "y"}}""")
        assertEquals(1L, o["load", "nonesuch"])
        assertEquals("y", o["elsewhere", "x"])
    }

    @Test
    fun whatIsNotADocumentIsRefused() {
        for (bad in listOf("", "[]", "{\"load\": 1}", "{\"load\": {\"a\": null}}", "{\"load\": {}} x", "{\"load\": {\"a\": 1, \"a\": 2}}")) {
            assertThrows(bad, IllegalArgumentException::class.java) { Options.parse(bad) }
        }
    }

    @Test
    fun nullRemovesAKeyAndAnEmptiedSection() {
        val o = Options.EMPTY.with("render", "edge_s", 0.25).with("render", "edge_s", null)
        assertFalse(o.has("render"))
        assertEquals(Options.EMPTY, o)
    }

    @Test
    fun theScreensRequestIsADocument() {
        assertEquals(
            """{"load": {"codec": "/m/codec.gguf", "engine": "qwen3-tts", "model": "/m/talker.gguf", "speaker": "hk_voice_343", "threads": 4}}""",
            qwen.options.toJson(),
        )
        assertEquals("qwen3-tts", qwen.engine)
        assertNull(qwen.voice)
        assertEquals(8, qwen.withThreads(8).threads)
        assertEquals("the thread choice is a value in the document", 8L, qwen.withThreads(8).options["load", "threads"])
        assertTrue("equal by content, so the cache keeps an engine for the same request", qwen == LoadRequest(Options.parse(qwen.options.toJson())))
    }

    @Test
    fun theLoadDocumentCarriesTheCacheAndTheRenderDefaults() {
        val r = qwen.with("render", "instruction", "Speak slowly.").with("observe", "log_level", "info")
        assertEquals(
            """{"load": {"cache_dir": "/c", "codec": "/m/codec.gguf", "engine": "qwen3-tts", "model": "/m/talker.gguf", "speaker": "hk_voice_343", "threads": 4}, "render": {"instruction": "Speak slowly."}}""",
            r.loadDocument("/c"),
        )
        assertTrue("a cache the document names wins", r.with("load", "cache_dir", "/mine").loadDocument("/c").contains("\"cache_dir\": \"/mine\""))
        assertEquals("the rest goes to each render, for the binding to take or refuse", """{"observe": {"log_level": "info"}}""", r.renderDocument())
    }

    @Test
    fun theStreamIsRawUnlessTheDocumentSaysOtherwise() {
        assertEquals("""{"output": {"stream": "raw"}}""", qwen.streamDocument())
        assertEquals("{}", qwen.renderDocument())
        val levelled = qwen.with("output", "stream", "levelled").with("output", "stream_level_seconds", 0)
        assertEquals("""{"output": {"stream": "levelled", "stream_level_seconds": 0}}""", levelled.streamDocument())
        assertEquals("a whole render carries no stream section", "{}", levelled.renderDocument())
    }
}
