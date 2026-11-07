// The Kotlin binding on the host JVM, over the C-ABI built for the host (the fake engine): load, render, stream and
// stop, and every failure a GgttsException rather than a crash. Run by scripts/kotlin_binding_test.sh:
//   java -Dggtts.lib=<libggtts_ffi.so> -cp <classes> ggtts.BindingTestKt
// Exits 0 when every check passes, 1 with the failures listed.
package ggtts

private val failures = mutableListOf<String>()

private fun check(ok: Boolean, what: String) {
    println((if (ok) "ok   " else "FAIL ") + what)
    if (!ok) failures.add(what)
}

private inline fun throwsCode(code: Int, what: String, block: () -> Unit) {
    try {
        block()
        check(false, "$what: no exception")
    } catch (e: GgttsException) {
        check(e.code == code, "$what: GgttsException code ${e.code} (want $code): ${e.message}")
    }
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
