// Options.kt: every option through one argument, `options`, at load and per render; a refused option names its key.
//
//   main(arrayOf())                                          Qwen3-TTS with its default model (from the cache)
//   main(arrayOf("qwen3-tts", "talker.gguf", "codec.gguf"))  model files you already have
//   main(arrayOf("fake"))                                    the test engine: no files, a tone instead of speech
package options

import ggtts.Ggtts
import ggtts.GgttsException

fun main(args: Array<String>) {
    val engine = args.getOrElse(0) { "qwen3-tts" }
    // The load document: the load section, and render defaults every render on this engine starts from. The
    // parameters of load still work beside it; one given both ways is refused.
    Ggtts.load(
        engine,
        model = args.getOrNull(1),
        codec = args.getOrNull(2),
        options = """{"render": {"edge_s": 0.25}}""",
    ).use { tts ->
        // A render document, laid over those defaults. seed = 7 is the same value as "render": {"seed_salt": "7"}.
        val pcm = tts.render("今日天氣好好。", seed = 7, options = """{"render": {"redraw": 1}}""")
        println("${pcm.samples.size} samples at ${pcm.sampleRate} Hz")

        // A stream through a document is levelled as ggtts-render --stream levels it: one gain from the first 5 s.
        // {"output": {"stream": "raw"}} asks for the pieces before the level step, as renderStream with no options.
        var pieces = 0
        tts.renderStream("今日天氣好好。\n我哋去海邊行下。", options = "{}") { _, _ ->
            pieces++
            true
        }
        println("$pieces levelled pieces")

        // A refused option throws GGTTS_ERR_ARG, and the exception's key is the option's schema key.
        try {
            tts.render("今日天氣好好。", options = """{"render": {"nonesuch": 1}}""")
        } catch (e: GgttsException) {
            println("refused ${e.key}: ${e.message}")
            check(e.code == Ggtts.ERR_ARG && e.key == "render.nonesuch")
        }
    }
}
