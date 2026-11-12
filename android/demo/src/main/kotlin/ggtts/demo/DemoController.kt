// What the Speak button does, apart from Android: load (or reuse) the engine, render one line, save and play it, and
// show the result. Every failure is shown as a message, and the screen stays usable for the next try.
package ggtts.demo

import ggtts.GgttsException
import ggtts.Pcm

/** The screen, as the controller sees it. Called on the main thread only. */
interface DemoView {
    fun setBusy(busy: Boolean)
    fun showStatus(text: String)
    fun showError(text: String)
}

/** Where the rendered line goes. Called on the background thread. */
interface AudioOut {
    /** Save [pcm] for [engine] and return where it went. */
    fun save(pcm: Pcm, engine: String): String

    /** Play [pcm] to the end. */
    fun play(pcm: Pcm)
}

class DemoController(
    private val binding: TtsBinding,
    private val view: DemoView,
    private val audio: AudioOut,
    /** Runs blocking work off the main thread. */
    private val background: (Runnable) -> Unit,
    /** Posts to the main thread. */
    private val main: (Runnable) -> Unit,
    private val log: (String) -> Unit = {},
) {
    private var loaded: LoadRequest? = null
    private var engine: TtsEngine? = null

    /** True from a [speak] until its result is shown. Read and written on the main thread. */
    var busy = false
        private set

    fun speak(request: LoadRequest, text: String, seed: Long) {
        if (busy) {
            view.showStatus("Still speaking; wait for it to finish.")
            return
        }
        busy = true
        view.setBusy(true)
        view.showStatus("Loading ${request.engine}...")
        background(Runnable {
            val outcome: Result<String> = try {
                Result.success(run(request, text, seed))
            } catch (e: GgttsException) {
                drop()
                log("error ${e.code}: ${e.message}")
                Result.failure(IllegalStateException("ggtts error ${e.code}: ${e.message}"))
            } catch (e: RuntimeException) {
                drop()
                log("error: $e")
                Result.failure(IllegalStateException("${e.javaClass.simpleName}: ${e.message}"))
            }
            main(Runnable {
                busy = false
                view.setBusy(false)
                outcome.fold({ view.showStatus(it) }, { view.showError(it.message ?: "error") })
            })
        })
    }

    /** Release the loaded engine, if any. */
    fun close() = drop()

    private fun run(request: LoadRequest, text: String, seed: Long): String {
        val tts = engineFor(request)
        val t0 = System.nanoTime()
        val pcm = tts.render(text, seed)
        val renderMs = (System.nanoTime() - t0) / 1_000_000
        val seconds = pcm.samples.size.toDouble() / pcm.sampleRate
        val rtf = if (seconds > 0) renderMs / 1000.0 / seconds else Double.NaN
        val path = audio.save(pcm, request.engine)
        log(
            "rendered ${request.engine}: ${pcm.samples.size} samples at ${pcm.sampleRate} Hz " +
                "(%.2f s) in $renderMs ms, RTF %.2f, seed $seed, wrote $path".format(seconds, rtf),
        )
        audio.play(pcm)
        return "${request.engine}: %.2f s of audio in %.1f s (RTF %.2f). Saved to $path".format(
            seconds, renderMs / 1000.0, rtf,
        )
    }

    private fun engineFor(request: LoadRequest): TtsEngine {
        engine?.let { if (loaded == request) return it }
        drop()
        val t0 = System.nanoTime()
        val e = binding.load(request)
        log("loaded ${request.engine} in ${(System.nanoTime() - t0) / 1_000_000} ms: model=${request.model} codec=${request.codec} voice=${request.voice}")
        engine = e
        loaded = request
        return e
    }

    private fun drop() {
        engine?.close()
        engine = null
        loaded = null
    }
}
