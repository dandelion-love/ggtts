// What the Speak and Stop buttons do, apart from Android: load (or reuse) the engine, stream the line chunk by chunk
// into the player as each chunk is ready, save the whole read, and show progress and the result. Every failure is
// shown as a message, and the screen stays usable for the next try.
//
// Streaming is the point on a phone: a chunk can take tens of seconds to render, and waiting for the whole read before
// playing anything multiplies that by the number of chunks.
package ggtts.demo

import ggtts.GgttsException

/** The screen, as the controller sees it. Called on the main thread only. */
interface DemoView {
    fun setBusy(busy: Boolean)
    fun showStatus(text: String)
    fun showError(text: String)
}

/** Where the audio goes. Called on the background thread, except [stop]. */
interface AudioOut {
    /** A new stream at [rate] Hz begins; [play] follows with its pieces in order. */
    fun begin(rate: Int)

    /** Queue [samples] for playback and return at once; playback runs on its own. */
    fun play(samples: FloatArray)

    /** Wait until everything queued has played. Throws if playback failed before the end (the read is not finished). */
    fun finish()

    /** True only when the playback device reports all frames consumed; this does not prove a person heard them. */
    fun finishConfirmed(): Boolean { finish(); return false }

    /** Drop what is queued and stop playing now. Safe from any thread, and when nothing is playing. */
    fun stop()

    /** Save the whole read for [engine] and return where it went. */
    fun save(samples: FloatArray, rate: Int, engine: String): String

    /** Hold playback where it is; what arrives meanwhile waits. Safe from any thread. */
    fun pause() {}

    /** Play on from where [pause] held it. Safe from any thread. */
    fun resume() {}

    /** Whether playback is held. */
    val paused: Boolean get() = false
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
    /** The clock, in nanoseconds; tests pass their own. */
    private val clock: () -> Long = System::nanoTime,
    /** Available memory in bytes, and whether the system reports memory is low; tests pass their own. */
    private val memory: () -> Pair<Long, Boolean> = { Long.MAX_VALUE to false },
    /** Told the request of every read that finished, so the system engine can speak with the same files. */
    private val onSpoke: (LoadRequest) -> Unit = {},
    /** Where engines are loaded and held; the app passes the one it shares with the system engine. */
    private val cache: EngineCache = EngineCache(binding, log),
    /** What it reports, in the phone's language on a phone ([ResourceWords]); English by default. */
    private val words: Words = EnglishWords,
    private val evidenceStore: SavedEvidenceStore? = null,
) {
    /** A load the user was warned about; the same request again goes ahead. Main thread only. */
    private var warnedFor: LoadRequest? = null

    @Volatile
    private var stopRequested = false

    /** True from a [speak] until its result is shown. Read and written on the main thread. */
    var busy = false
        private set

    fun speak(request: LoadRequest, text: String, seed: Long) {
        if (busy) {
            view.showStatus(words.get("still_speaking"))
            return
        }
        // Before a load (not when this engine is already loaded), warn once if memory is short; the same request
        // again goes ahead.
        if (!cache.holds(request) && warnedFor != request) {
            val (available, low) = memory()
            EngineCosts.memoryWarning(request.engine, available, low, words)?.let {
                warnedFor = request
                log("memory warning: $it")
                view.showError(it)
                return
            }
        }
        warnedFor = null
        busy = true
        stopRequested = false
        view.setBusy(true)
        view.showStatus(words.get("preparing_voice"))
        background(Runnable {
            val outcome: Result<String> = try {
                Result.success(run(request, text, seed))
            } catch (e: GgttsException) {
                audio.stop()
                log("error ${e.code}: ${e.message}")
                Result.failure(IllegalStateException("ggtts error ${e.code}: ${e.message}"))
            } catch (e: EngineInUse) {
                log("refused: ${e.message}")
                Result.failure(IllegalStateException(words.get("not_started", e.message ?: "")))
            } catch (e: RuntimeException) {
                audio.stop()
                log("error: $e")
                Result.failure(IllegalStateException("${e.javaClass.simpleName}: ${e.message}"))
            } catch (e: Throwable) {
                // An Error (out of memory, say) too: shown, and the screen is left usable rather than busy for good.
                runCatching { audio.stop() }
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

    /** Stop the read: playback stops now, and rendering stops when the chunk in progress is done. */
    fun stop() {
        if (!busy) return
        stopRequested = true
        audio.stop()
        view.showStatus(words.get("stopping"))
    }

    /** Whether the read's playback is held ([pause]). */
    val paused: Boolean get() = busy && audio.paused

    /**
     * Hold the read's playback (a media session's Pause, a headset button). Rendering goes on, and what it makes
     * waits; [resume] plays on from there. Nothing happens when no read runs. Main thread.
     */
    fun pause() {
        if (!busy || audio.paused) return
        audio.pause()
        view.showStatus(words.get("paused_status"))
    }

    /** Play on from where [pause] held it. Main thread. */
    fun resume() {
        if (!busy || !audio.paused) return
        audio.resume()
        view.showStatus(words.get("playing_on"))
    }

    /** Release the loaded engine, if any, unless a read (this screen's or the system engine's) is using it. */
    fun close() {
        if (!busy) cache.release()
    }

    /** One read with the shared engine: it is this read's until the read ends, and a failed read closes it. */
    private fun run(request: LoadRequest, text: String, seed: Long): String {
        val tts = cache.acquire(request)
        var ok = false
        try {
            return read(tts, request, text, seed).also { ok = true }
        } finally {
            cache.done(failed = !ok)
        }
    }

    private fun read(tts: TtsEngine, request: LoadRequest, text: String, seed: Long): String {
        main(Runnable { view.showStatus(words.get("writing_first")) })
        val t0 = clock()
        var firstMs = -1L
        var rate = 0
        val pieces = mutableListOf<FloatArray>()
        val evidence = if (tts.supportsEvidence && evidenceStore != null) VerifiedWavEvidence() else null
        var evidenceValid = true
        val onPiece: (FloatArray, Int) -> Boolean = piece@ { samples, sampleRate ->
            if (stopRequested) return@piece false
            if (pieces.isEmpty()) {
                firstMs = (clock() - t0) / 1_000_000
                rate = sampleRate
                audio.begin(sampleRate)
                log("first audio after $firstMs ms")
            }
            pieces += samples
            audio.play(samples)
            val n = pieces.size
            main(Runnable {
                view.showStatus(words.get(if (audio.paused) "paused_chunk" else "speaking_chunk", n, firstMs / 1000.0))
            })
            true
        }
        val finished = if (evidence == null) tts.renderStream(text, seed, onPiece) else {
            try {
                val success = tts.renderEvidence(text, seed) { i, t, s, d, stop, start, speechEnd, samples, sampleRate ->
                    if (stopRequested) false else {
                        if (evidenceValid) {
                            runCatching { evidence.add(i, t, s, d, stop, start, speechEnd, samples, sampleRate) }
                                .onFailure { evidenceValid = false; log("stream evidence unavailable: " + it.message) }
                        }
                        onPiece(samples, sampleRate)
                    }
                }
                evidence.complete(success && evidenceValid)
                success
            } catch (t: Throwable) {
                evidence.complete(false)
                throw t
            }
        }
        val renderMs = (clock() - t0) / 1_000_000
        if (!finished) {
            // A Stop can land after the worker read stopRequested and before the first chunk began its stream; that
            // stream plays on unless it is stopped here (stop is idempotent).
            audio.stop()
            log("stopped after ${pieces.size} chunks, $renderMs ms")
            return words.get("stopped_after", pieces.size)
        }
        val all = FloatArray(pieces.sumOf { it.size })
        var at = 0
        for (p in pieces) {
            p.copyInto(all, at)
            at += p.size
        }
        val seconds = if (rate > 0) all.size.toDouble() / rate else 0.0
        val rtf = if (seconds > 0) renderMs / 1000.0 / seconds else Double.NaN
        val path = if (all.isEmpty()) null else audio.save(all, rate, request.engine)
        if (path != null && evidence != null && evidenceValid) {
            runCatching { evidenceStore!!.save(path, evidence) }
                .onFailure { log("saved audio evidence unavailable: " + it.message) }
        }
        log(
            "rendered ${request.engine}: ${all.size} samples at $rate Hz (%.2f s) in $renderMs ms, RTF %.2f, ".format(seconds, rtf) +
                "first audio after $firstMs ms, ${pieces.size} chunks, seed $seed, wrote $path",
        )
        // The files rendered: remember them now (the system engine can use them while this read still plays).
        onSpoke(request)
        try {
            audio.finish()
        } catch (e: PlaybackUnconfirmed) {
            // A deliberate Stop is reported below; an unexplained failure is never read_done.
            if (!stopRequested) throw e
        }
        // A Stop while the last chunks play out cuts the playback short: say so, not that the read finished.
        if (stopRequested) {
            log("stopped during playback, after rendering ${pieces.size} chunks")
            return words.get("stopped_during_playback", path)
        }
        return words.get("read_done", request.engine, seconds, renderMs / 1000.0, rtf, firstMs / 1000.0, path)
    }
}
