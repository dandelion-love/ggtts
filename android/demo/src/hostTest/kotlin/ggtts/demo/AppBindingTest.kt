// The demo app's real binding seam (GgttsBinding, LoadRequest, Options) on the host JVM, over the C-ABI built for the
// host (the fake engine): the app's request reaches the library as its options document. Plain Kotlin, no Android, so
// scripts/kotlin_binding_test.sh compiles these three files beside the binding and runs this:
//   java -Dggtts.lib=<libggtts_ffi.so> -cp <classes> ggtts.demo.AppBindingTestKt
// Exits 0 when every check passes, 1 with the failures listed. The app's own JVM tests (android/demo/src/test) cover
// the rest with a fake binding; this is the one place the app's documents meet the library.
package ggtts.demo

import ggtts.Ggtts
import ggtts.GgttsException
import java.security.MessageDigest

private val failures = mutableListOf<String>()

private fun check(ok: Boolean, what: String) {
    println((if (ok) "ok   " else "FAIL ") + what)
    if (!ok) failures.add(what)
}

private inline fun refusesKey(key: String, what: String, block: () -> Unit) {
    try {
        block()
        check(false, "$what: no exception")
    } catch (e: GgttsException) {
        check(e.code == Ggtts.ERR_ARG && e.key == key, "$what: code ${e.code} key ${e.key} (want ${Ggtts.ERR_ARG} $key): ${e.message}")
    }
}

private const val TEXT = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。"

private fun hash(samples: FloatArray): String {
    val md = MessageDigest.getInstance("SHA-256")
    for (s in samples) {
        val b = java.lang.Float.floatToIntBits(s)
        md.update(byteArrayOf(b.toByte(), (b shr 8).toByte(), (b shr 16).toByte(), (b shr 24).toByte()))
    }
    return md.digest().joinToString("") { "%02x".format(it) }.take(16)
}

private fun streamed(engine: TtsEngine, seed: Long): FloatArray {
    val out = mutableListOf<Float>()
    engine.renderStream(TEXT, seed) { s, _ ->
        out += s.toList()
        true
    }
    return out.toFloatArray()
}

fun main() {
    val app = GgttsBinding(cacheDir = null)
    val fake = LoadRequest("fake", null, null, null)

    // AC1: the app's request, as a document, renders what the binding's parameter call rendered before it, whole and
    // streamed (the stream raw, as the app streamed before the document).
    val before = Ggtts.load("fake", threads = 4)
    val wholeBefore = before.render(TEXT, Ggtts.DEFAULT_SEED).samples
    val pieces = mutableListOf<Float>()
    before.renderStream(TEXT, Ggtts.DEFAULT_SEED) { s, _ ->
        pieces += s.toList()
        true
    }
    before.close()
    val streamBefore = pieces.toFloatArray()
    app.load(fake).use { e ->
        val whole = e.render(TEXT, Ggtts.DEFAULT_SEED).samples
        check(whole.contentEquals(wholeBefore), "AC1 whole read: document ${hash(whole)} = parameters ${hash(wholeBefore)}")
        val stream = streamed(e, Ggtts.DEFAULT_SEED)
        check(stream.contentEquals(streamBefore), "AC1 stream: document ${hash(stream)} = parameters ${hash(streamBefore)}")
    }

    // The stream's level step is a key of the same document: levelled, the pieces differ from the raw ones.
    app.load(fake.with("output", "stream", "levelled")).use { e ->
        val levelled = streamed(e, Ggtts.DEFAULT_SEED)
        check(levelled.size == streamBefore.size && !levelled.contentEquals(streamBefore),
            "output.stream levelled reaches the binding: ${hash(levelled)} != raw ${hash(streamBefore)}")
    }

    // AC3: an instruction in the request's document reaches the engine's chunk request; the fake engine takes none,
    // so its capability refusal, by the key, is what arriving looks like.
    refusesKey("render.instruction", "AC3 an instruction in the app's document reaches the fake engine") {
        app.load(fake.with("render", "instruction", "Speak slowly.")).use { it.render(TEXT, Ggtts.DEFAULT_SEED) }
    }
    refusesKey("render.steer", "render.steer in the app's document reaches the fake engine") {
        app.load(fake.with("render", "steer", "s.safetensors")).use { it.render(TEXT, Ggtts.DEFAULT_SEED) }
    }
    // An edge is a render default in the load document, and changes the read.
    app.load(fake.with("render", "edge_s", 1.5)).use { e ->
        val edged = e.render(TEXT, Ggtts.DEFAULT_SEED).samples
        check(edged.size != wholeBefore.size, "render.edge_s reaches the render: ${edged.size} samples, not ${wholeBefore.size}")
    }

    // AC4: a key the library does not have is refused with the binding's error and its key, not dropped.
    refusesKey("load.nonesuch", "AC4 an unknown load key is refused by key") { app.load(fake.with("load", "nonesuch", 1)).close() }
    refusesKey("render.nonesuch", "AC4 an unknown render key is refused by key") { app.load(fake.with("render", "nonesuch", 1)).close() }
    // What the binding refuses on Android, from the app's document.
    refusesKey("observe.log_level", "a log level is refused by key") {
        app.load(fake.with("observe", "log_level", "info")).use { it.render(TEXT, Ggtts.DEFAULT_SEED) }
    }
    refusesKey("observe.timing_report", "a timing report is refused by key") {
        app.load(fake.with("observe", "timing_report", "t.json")).use { it.render(TEXT, Ggtts.DEFAULT_SEED) }
    }
    refusesKey("render.measure_timings", "measured timings are refused by key") {
        app.load(fake.with("render", "measure_timings", true)).use { it.render(TEXT, Ggtts.DEFAULT_SEED) }
    }
    // The app never downloads: it gives allowDownload = false beside the document, so a document asking for a
    // download is refused rather than obeyed.
    try {
        app.load(fake.with("load", "offline", false)).close()
        check(false, "load.offline in the app's document: no exception")
    } catch (e: GgttsException) {
        check(e.code == Ggtts.ERR_ARG, "load.offline in the app's document is refused: ${e.message}")
    }

    if (failures.isEmpty()) {
        println("app binding: all checks passed")
    } else {
        println("app binding: ${failures.size} failed:")
        failures.forEach { println("  $it") }
        kotlin.system.exitProcess(1)
    }
}
