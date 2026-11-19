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

    // The default model, when -Dggtts.cachedir names a cache already holding the pinned v0.0.5 files: no paths and no
    // speaker render as those files with hk_voice_343, and nothing is downloaded.
    val cacheDir = System.getProperty("ggtts.cachedir")
    if (cacheDir == null) {
        println("skip qwen3-tts default model (set -Dggtts.cachedir to a cache holding the default files)")
        return
    }
    val byDefault = samplesOf(Ggtts.load("qwen3-tts", cacheDir = cacheDir), line)
    check(
        byDefault.isNotEmpty() && byDefault.contentEquals(spoken("hk_voice_343", null)),
        "qwen3-tts: the default model renders with its default speaker hk_voice_343",
    )
}

/** A refusal of one option: GGTTS_ERR_ARG with [key] as the exception's key (null: the whole document). */
private inline fun refusesKey(key: String?, what: String, block: () -> Unit) {
    try {
        block()
        check(false, "$what: no exception")
    } catch (e: GgttsException) {
        check(e.code == Ggtts.ERR_ARG && e.key == key, "$what: code ${e.code} key ${e.key} (want ${Ggtts.ERR_ARG} $key): ${e.message}")
    }
}

/** The options documents: `load(options = ...)`, `render(text, options = ...)` and `renderStream(..., options = ...)`. */
private fun optionsDocuments() {
    val fake = """{"load": {"engine": "fake"}}"""
    // Nothing given but the engine: the old render, at the default seed and at a seed.
    check(
        samplesOf(Ggtts.load(options = fake), TEXT).contentEquals(samplesOf(Ggtts.load("fake"), TEXT)),
        "a load document: the same samples as load(\"fake\")",
    )
    Ggtts.load(options = fake).use { a ->
        Ggtts.load("fake").use { b ->
            check(
                a.render(TEXT, seed = 7, options = "{}").samples.contentEquals(b.render(TEXT, seed = 7).samples),
                "an empty render document at seed 7: the same samples as render(seed = 7)",
            )
        }
    }
    // The load's render defaults reach every render: the edge kept at each chunk changes the read's length.
    val edged = Ggtts.load(options = """{"load": {"engine": "fake"}, "render": {"edge_s": 1.5}}""").use { it.render(TEXT, seed = 7).samples }
    check(edged.size != samplesOf(Ggtts.load("fake"), TEXT).size, "a load document's render.edge_s reaches the render")
    // A document's instruction reaches the engine: the fake engine's capabilities refuse it, by its key.
    Ggtts.load(options = fake).use { g ->
        refusesKey("render.instruction", "an instruction in a render document reaches the fake engine") {
            g.render(TEXT, options = """{"render": {"instruction": "Speak slowly."}}""")
        }
        refusesKey("render.steer", "steering in a render document reaches the fake engine") {
            g.render(TEXT, options = """{"render": {"steer": "s.safetensors"}}""")
        }
        refusesKey("render.nonesuch", "an unknown key") { g.render(TEXT, options = """{"render": {"nonesuch": 1}}""") }
        refusesKey(null, "malformed JSON") { g.render(TEXT, options = "{") }
        refusesKey("load.threads", "a load key in a render document") { g.render(TEXT, options = """{"load": {"threads": 2}}""") }
        refusesKey("observe.trace", "the trace (a desktop observer)") {
            g.render(TEXT, options = """{"observe": {"trace": "t.jsonl"}}""")
        }
        refusesKey("observe.instruments", "instruments") {
            g.render(TEXT, options = """{"observe": {"instruments": "instruments"}}""")
        }
        refusesKey("output.stream", "a stream on a whole read") { g.render(TEXT, options = """{"output": {"stream": "raw"}}""") }
        refusesKey("render.seed_salt", "a seed beside render.seed_salt") {
            g.render(TEXT, seed = 3, options = """{"render": {"seed_salt": "3"}}""")
        }
        // A stream through a document: the pieces before the level step, as renderStream hands them.
        var pieces = 0
        check(g.renderStream(TEXT, options = """{"output": {"stream": "raw"}}""") { _, _ -> pieces++; true } && pieces >= 2, "a stream document streams ($pieces pieces)")
    }
    // Steering on Kitten-TTS 2 is refused before any file is read.
    refusesKey("render.steer", "steering on kitten-tts-2") {
        Ggtts.load(options = """{"load": {"engine": "kitten-tts-2", "model": "/no/such.gguf"}, "render": {"steer": "s.safetensors"}}""")
    }
    refusesKey("load.device", "a device (CPU only)") { Ggtts.load(options = """{"load": {"engine": "fake", "device": "cpu"}}""") }
    refusesKey("load.engine", "an engine given twice") { Ggtts.load("fake", options = fake) }
    refusesKey("load.bogus", "an unknown load key") { Ggtts.load(options = """{"load": {"engine": "fake", "bogus": 1}}""") }
    // A refusal through the parameters carries its key too.
    refusesKey("load.speaker", "a refused speaker parameter") { Ggtts.load("fake", speaker = "hk_voice_777") }
}

/** A document stream is levelled unless it asks for the pieces before the level step; no document is the old call. */
private fun streamModes() {
    fun joined(options: String?): FloatArray {
        val out = mutableListOf<Float>()
        Ggtts.load("fake").use { it.renderStream(TEXT, seed = 7, options = options) { s, _ -> out.addAll(s.toList()); true } }
        return out.toFloatArray()
    }
    val old = joined(null)
    val levelled = joined("{}")
    check(joined("""{"output": {"stream": "raw"}}""").contentEquals(old), "a raw document stream is the old stream")
    check(!levelled.contentEquals(old) && levelled.size == old.size, "a document stream is levelled by default")
    check(
        !joined("""{"output": {"stream": "levelled", "stream_level_seconds": 0}}""").contentEquals(levelled),
        "the hold reaches the level step",
    )
}

/** The options schema and an engine's capability statement, from the library. */
private fun introspection() {
    val schema = Ggtts.optionsSchema()
    check(schema.contains("\"https://json-schema.org/") && schema.contains("\"steer\""), "the options schema is the library's")
    val g = Ggtts.load("fake")
    val caps = g.capabilities()
    check(
        caps.contains("\"format\":\"ggtts-capabilities/1\"") && caps.contains("\"family\":\"fake\"") &&
            caps.contains("\"speaker_names\":[]"),
        "the fake engine states its capabilities: $caps",
    )
    g.close()
    throwsCode(Ggtts.ERR_HANDLE, "capabilities after close") { g.capabilities() }
}

/** The default model files without a network: whatever the build, nothing is downloaded unless asked. */
private fun defaultModel() {
    val empty = java.nio.file.Files.createTempDirectory("ggtts-binding-cache").toFile()
    try {
        throwsCode(Ggtts.ERR_ENGINE, "qwen3-tts with no model and an empty cache, downloading off") {
            try {
                Ggtts.load("qwen3-tts", cacheDir = empty.path).close()
            } catch (e: GgttsException) {
                val said = e.message ?: ""
                // A build with engines names the missing file and its URL; one without says it has no engine.
                check(
                    said.contains("no qwen3-tts engine") ||
                        (said.contains("qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf") && said.contains("https://huggingface.co/")),
                    "the refusal names the missing file and its URL, or the missing engine: $said",
                )
                throw e
            }
        }
        check(empty.listFiles().isNullOrEmpty(), "nothing was written to the cache")
        // The fake engine takes the new parameters and renders as before.
        val plain = samplesOf(Ggtts.load("fake"), TEXT)
        check(
            plain.contentEquals(samplesOf(Ggtts.load("fake", cacheDir = empty.path, allowDownload = false), TEXT)),
            "the default-model parameters leave another engine's samples unchanged",
        )
    } finally {
        empty.deleteRecursively()
    }
    // The entry point the seven-parameter load used still loads only the paths given.
    throwsCode(Ggtts.ERR_ARG, "the version-2 entry point with no model", "needs a model") {
        Native.loadWith("qwen3-tts", null, null, null, 4, null, null)
    }
}

private const val TEXT = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。"

private fun evidenceStream() {
    Ggtts.load("fake").use { g ->
        for (options in listOf<String?>(null, "{}",
                """{"output":{"stream":"raw"}}""")) {
            val legacy = mutableListOf<Float>()
            val evidence = mutableListOf<Float>()
            var next = 0L
            var seen = 0
            val seenText = StringBuilder()
            val text = "One supplementary \uD840\uDC00 character.\nAnother line."
            // Warm the fake engine: its first distinct text has a different amplitude.
            g.renderStream(text, seed = 7, options = options) { _, _ -> true }
            check(g.renderStream(text, seed = 7, options = options) { samples, _ ->
                legacy.addAll(samples.toList()); true
            }, "legacy stream completed")
            check(g.renderStreamEvidence(text, seed = 7, options = options) { chunk, pieceText, _, draw, stop, start, end, samples, rate ->
                check(chunk == seen && start == next && end in start..(start + samples.size) &&
                    draw >= 0 && stop in 0..1 && rate > 0 && pieceText.isNotEmpty(),
                    "evidence identity and boundaries for chunk $chunk")
                evidence.addAll(samples.toList())
                seenText.append(pieceText)
                next += samples.size
                seen++
                true
            }, "evidence stream completed")
            check(seen >= 2 && legacy == evidence, "evidence PCM matches legacy for $options")
            check(seenText.contains("\uD840\uDC00"), "supplementary text survives callback")
        }
        var seen = 0
        check(!g.renderStreamEvidence(TEXT) { _, _, _, _, _, _, _, _, _ -> seen++; false } && seen == 1,
            "evidence listener can stop")
        val marker = IllegalStateException("listener marker")
        try {
            g.renderStreamEvidence(TEXT) { _, _, _, _, _, _, _, _, _ -> throw marker }
            check(false, "evidence listener exception propagated")
        } catch (e: IllegalStateException) {
            check(e === marker, "evidence listener exception propagated unchanged")
        }
        refusesKey("render.seed_salt", "evidence seed conflict") {
            g.renderStreamEvidence(TEXT, seed = 7, options = """{"render":{"seed_salt":"8"}}""") { _, _, _, _, _, _, _, _, _ -> true }
        }
    }
}

fun main() {
    evidenceStream()
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
    throwsCode(Ggtts.ERR_ARG, "an instruction on the fake engine", "takes no style instruction") {
        Ggtts.load("fake", instruction = "Speak slowly and softly.")
    }
    optionsDocuments()
    introspection()
    streamModes()
    qwenSettings()
    defaultModel()

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
