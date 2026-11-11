// The Kotlin binding on the host JVM, over the C-ABI built for the host (the fake engine): load, render, stream and
// stop, the voice settings (speaker, instruction), and every failure a GgttsException rather than a crash. With
// -Dggtts.talker and -Dggtts.codec (and a library built with the ggml feature) it also renders Qwen3-TTS with a
// speaker and an instruction. Run by scripts/kotlin_binding_test.sh:
//   java -Dggtts.lib=<libggtts_ffi.so> -cp <classes> ggtts.BindingTestKt
// Exits 0 when every check passes, 1 with the failures listed.
package ggtts

private val failures = mutableListOf<String>()

private fun check(ok: Boolean, what: String) {
    println((if (ok) "ok   " else "FAIL ") + what)
    if (!ok) failures.add(what)
}

private inline fun throwsCode(code: Int, what: String, saying: String = "", block: () -> Unit) {
    try {
        block()
        check(false, "$what: no exception")
    } catch (e: GgttsException) {
        val said = e.message ?: ""
        check(e.code == code && said.contains(saying), "$what: GgttsException code ${e.code} (want $code): $said")
    }
}

private fun samplesOf(g: Ggtts, text: String): FloatArray =
    try {
        g.render(text).samples
    } finally {
        g.close()
    }

/**
 * Speaker and instruction on Qwen3-TTS, when -Dggtts.talker and -Dggtts.codec name a custom_voice 1.7B talker with
 * more than one speaker (for example the v0.0.5 GGUF: hk_voice_343, hk_voice_777) and its codec. Skipped otherwise.
 */
private fun qwenSettings() {
    val talker = System.getProperty("ggtts.talker")
    val codec = System.getProperty("ggtts.codec")
    if (talker == null || codec == null) {
        println("skip qwen3-tts speaker and instruction (set -Dggtts.talker and -Dggtts.codec)")
        return
    }
    val line = "今日天氣好好。"
    val spoken = { speaker: String?, instruction: String? ->
        samplesOf(Ggtts.load("qwen3-tts", talker, codec, speaker = speaker, instruction = instruction), line)
    }
    // An unknown speaker names the model's speakers; use the first two.
    var speakers = listOf<String>()
    throwsCode(Ggtts.ERR_ENGINE, "qwen3-tts: an unknown speaker", "speakers are") {
        try {
            Ggtts.load("qwen3-tts", talker, codec, speaker = "no_such_speaker").close()
        } catch (e: GgttsException) {
            speakers = (e.message ?: "").substringAfter("speakers are: ").substringBefore(".").split(", ")
            throw e
        }
    }
    check(speakers.size >= 2, "qwen3-tts: the model has two speakers to choose from ($speakers)")
    if (speakers.size < 2) return
    val first = spoken(speakers[0], null)
    check(first.isNotEmpty() && !first.contentEquals(spoken(speakers[1], null)), "qwen3-tts: speaker ${speakers[1]} differs from ${speakers[0]}")
    check(!first.contentEquals(spoken(speakers[0], "Speak slowly and softly.")), "qwen3-tts: an instruction changes the render")
    check(first.contentEquals(spoken(speakers[0], "")), "qwen3-tts: an empty instruction is no instruction")
}

private const val TEXT = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。"

fun main() {
    // Load and render.
    val g = Ggtts.load("fake")
    val pcm = g.render(TEXT)
    check(pcm.samples.size > pcm.sampleRate && pcm.sampleRate > 0, "render: ${pcm.samples.size} samples at ${pcm.sampleRate} Hz")

    // Stream: every piece, in order, adding up to the read; then a listener that stops after one.
    val pieces = mutableListOf<Int>()
    val done = g.renderStream(TEXT) { s, rate -> pieces.add(s.size); rate > 0 }
    check(done && pieces.size >= 2, "stream: completed with ${pieces.size} pieces")
    val fresh = Ggtts.load("fake")
    check(pieces.sum() == fresh.render(TEXT).samples.size, "stream: the pieces add up to the read (${pieces.sum()})")
    fresh.close()
    var seen = 0
    val stopped = g.renderStream(TEXT) { _, _ -> seen++; false }
    check(!stopped && seen == 1, "stream: a listener returning false stops after one piece (saw $seen)")

    // A listener may call back in: rendering on its own engine is busy, never a deadlock; closing it lets the stream
    // finish, and the engine is gone after.
    var inner: Int? = null
    val reentered = g.renderStream(TEXT) { _, _ ->
        if (inner == null) {
            inner = try {
                g.render(TEXT)
                0
            } catch (e: GgttsException) {
                e.code
            }
        }
        true
    }
    check(reentered && inner == Ggtts.ERR_BUSY, "a listener rendering on its own engine: busy (got $inner), the stream finishes")
    val closing = Ggtts.load("fake")
    var closes = 0
    val closed = closing.renderStream(TEXT) { _, _ -> if (closes++ == 0) closing.close(); true }
    check(closed && closes >= 2, "a listener closing its own engine: the stream finishes ($closes pieces)")
    throwsCode(Ggtts.ERR_HANDLE, "render after a listener closed it") { closing.render(TEXT) }

    // A freed id, not just the zeroed handle a closed Ggtts holds.
    val freed = Native.load("fake", null, null, null, 0)
    Native.free(freed)
    throwsCode(Ggtts.ERR_HANDLE, "render with a freed id") { Native.render(freed, TEXT, 1L, IntArray(1)) }
    throwsCode(Ggtts.ERR_HANDLE, "free a freed id") { Native.free(freed) }

    // Errors are exceptions.
    throwsCode(Ggtts.ERR_ENGINE, "a missing model file") { Ggtts.load("qwen3-tts", "/no/such/model.gguf", "/no/such/codec.gguf") }
    throwsCode(Ggtts.ERR_ENGINE, "an unknown engine") { Ggtts.load("nonesuch") }
    throwsCode(Ggtts.ERR_ARG, "empty text") { g.render(" ") }

    // Voice settings. Neither set gives today's output, through the old entry point and the new one, with nulls and
    // with empty strings (fresh engines: one fake engine counts its draws per text).
    val old = Native.load("fake", null, null, null, 4)
    val oldSamples = Native.render(old, TEXT, Ggtts.DEFAULT_SEED, IntArray(1))
    Native.free(old)
    check(oldSamples.contentEquals(samplesOf(Ggtts.load("fake"), TEXT)), "no settings: the same samples as the old load")
    check(
        oldSamples.contentEquals(samplesOf(Ggtts.load("fake", speaker = "", instruction = ""), TEXT)),
        "empty settings: the same samples as the old load",
    )
    // Each setting reaches the library, which refuses it on an engine that does not take it.
    throwsCode(Ggtts.ERR_ARG, "a speaker on the fake engine", "speaker is only supported by the qwen3-tts engine") {
        Ggtts.load("fake", speaker = "hk_voice_777")
    }
    throwsCode(Ggtts.ERR_ENGINE, "an instruction on the fake engine", "takes no style instruction") {
        Ggtts.load("fake", instruction = "Speak slowly and softly.")
    }
    qwenSettings()

    // Closed: every call throws, none crashes; closing twice is harmless.
    g.close()
    throwsCode(Ggtts.ERR_HANDLE, "render after close") { g.render(TEXT) }
    throwsCode(Ggtts.ERR_HANDLE, "stream after close") { g.renderStream(TEXT) { _, _ -> true } }
    g.close()

    if (failures.isEmpty()) {
        println("all checks passed")
    } else {
        println("${failures.size} failed")
        kotlin.system.exitProcess(1)
    }
}
