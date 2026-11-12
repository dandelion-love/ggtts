// Stream.kt: render piece by piece, stop early, and see the threading rules and error codes at work.
//
//   main(arrayOf())                                        Qwen3-TTS with its default model (it must be in the cache)
//   main(arrayOf("fake"))                                  the test engine
//   main(arrayOf("qwen3-tts", "talker.gguf", "codec.gguf"))
package streaming

import ggtts.Ggtts
import ggtts.GgttsException

const val TEXT = "今日天氣好好。\n我哋去海邊行下。\n行完之後，去食飯。"

fun main(args: Array<String>) {
    val tts = try {
        // No allowDownload: a default file missing from the cache throws ERR_ENGINE naming the file and its URL.
        Ggtts.load(args.getOrElse(0) { "qwen3-tts" }, model = args.getOrNull(1), codec = args.getOrNull(2))
    } catch (e: GgttsException) {
        System.err.println("load failed (${e.code}): ${e.message}")
        return
    }

    tts.use {
        // The listener runs on the thread that called renderStream, once per kept chunk, in order.
        // Return true to go on, false to stop. Here it also tries to render from inside the listener.
        var pieces = 0
        var busy = 0
        val finished = tts.renderStream(TEXT) { samples, sampleRate ->
            pieces++
            println("piece $pieces: ${samples.size} samples at $sampleRate Hz")
            try {
                tts.render("x")                    // one render at a time per engine: this throws ERR_BUSY
            } catch (e: GgttsException) {
                busy = e.code
            }
            pieces < 2                             // stop after two pieces
        }
        println("renderStream returned $finished after $pieces pieces; a render from the listener threw $busy")
        check(!finished && pieces == 2 && busy == Ggtts.ERR_BUSY)

        // Text with nothing to say is refused before rendering.
        val refused = runCatching { tts.render("？！") }.exceptionOrNull() as GgttsException
        println("punctuation only: ${refused.code} (ERR_ARG): ${refused.message}")
        check(refused.code == Ggtts.ERR_ARG)
    }

    // After close (here, the end of use), every call throws ERR_HANDLE.
    val closed = runCatching { tts.render(TEXT) }.exceptionOrNull() as GgttsException
    println("after close: ${closed.code} (ERR_HANDLE)")
    check(closed.code == Ggtts.ERR_HANDLE)
}
