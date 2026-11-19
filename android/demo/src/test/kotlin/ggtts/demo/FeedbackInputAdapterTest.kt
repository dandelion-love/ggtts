package ggtts.demo

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class FeedbackInputAdapterTest {
    private fun fixture(): Pair<ByteArray, AudioBoundReadManifest> {
        val a = floatArrayOf(-1f, 0.5f, 0f)
        val b = floatArrayOf(1f, -0.5f)
        val wav = Wav.encode(a + b, 24000)
        val e = VerifiedWavEvidence()
        e.add(0, "A中😀𠀀，文𰀀</script>&\"\\\n\t\b\u000c\r\u0001\u007f", -1, 3, 0, 0, 2, a, 24000)
        e.add(1, "second / line", 2, 0, 1, 3, 5, b, 24000)
        e.complete(true)
        return wav to e.verify("read:manifest", wav)
    }

    @Test fun canonicalBytesAndDigestMatchPythonSharedGenerator() {
        val (wav, manifest) = fixture()
        val actual = FeedbackInputAdapter.fromWav(wav, manifest, "read-abc")
        var root = File(System.getProperty("user.dir") ?: error("missing user.dir")).absoluteFile
        while (!File(root, "scripts/feedback_page.py").isFile) root = root.parentFile ?: break
        assumeTrue("shared Python generator and python3 required", File(root, "scripts/feedback_page.py").isFile)
        val command = "import json,sys; from scripts.feedback_page import canonical_input,input_digest,generate; d=json.load(sys.stdin); print(canonical_input(d)); print(input_digest(d)); print(len(generate(d)))"
        val process = try { ProcessBuilder("python3", "-c", command).directory(root).redirectErrorStream(true).start() }
                      catch (_: java.io.IOException) { assumeTrue("python3 required", false); return }
        process.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(actual.canonicalJson) }
        val output = process.inputStream.bufferedReader(Charsets.UTF_8).readLines()
        assertEquals(output.joinToString("\n"), 0, process.waitFor())
        assertEquals(3, output.size)
        assertEquals(output[0], actual.canonicalJson)
        assertEquals(output[1], actual.sha256)
        assertTrue(output[2].toInt() > actual.canonicalJson.length)
        assertFalse(actual.canonicalJson.contains("</script>"))
        assertTrue(actual.canonicalJson.contains("\\ud83d\\ude00"))
        assertTrue(actual.canonicalJson.contains("\\u003c/script\\u003e\\u0026"))
        assertTrue(actual.canonicalJson.indexOf("chunk-0") < actual.canonicalJson.indexOf("chunk-1"))
    }

    @Test fun dataUrisDecodeToTheirExactSourceClips() {
        val (wav, manifest) = fixture()
        val clips = FeedbackPcmClips.fromWav(wav, manifest)
        val actual = FeedbackInputAdapter.fromWav(wav, manifest, "read-abc")
        val uris = Regex("\"audio\":\"data:audio/flac;base64,([A-Za-z0-9+/]+=*)\"")
            .findAll(actual.canonicalJson).map { it.groupValues[1] }.toList()
        assertEquals(clips.size, uris.size)
        clips.forEachIndexed { i, clip ->
            assertArrayEquals("clip $i", clip.flac, java.util.Base64.getDecoder().decode(uris[i]))
        }
    }

    @Test fun chromiumAcceptsTheGeneratedInputAndItsDigest() {
        val chrome = listOf("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser")
            .firstOrNull { File(it).canExecute() }
        assumeTrue("Chromium required", chrome != null)
        var root = File(System.getProperty("user.dir") ?: error("missing user.dir")).absoluteFile
        while (!File(root, "feedback/page.html").isFile) root = root.parentFile ?: break
        val template = File(root, "feedback/page.html")
        assumeTrue("shared feedback template required", template.isFile)
        val (wav, manifest) = fixture()
        val input = FeedbackInputAdapter.fromWav(wav, manifest, "read-abc")
        val marker = "<!-- GGTTS_INPUT -->"
        val source = template.readText()
        assertEquals(1, source.split(marker).size - 1)
        val embedded = "<meta name=\"feedback-input-sha256\" content=\"" + input.sha256 +
            "\"><script id=\"feedback-input\" type=\"application/json\">" + input.canonicalJson + "</script>"
        val temp = kotlin.io.path.createTempDirectory("ggtts-feedback-browser-").toFile()
        try {
            val page = File(temp, "page.html").apply { writeText(source.replace(marker, embedded)) }
            val stderr = File(temp, "chrome.err")
            val p = ProcessBuilder(chrome!!, "--headless=new", "--no-sandbox", "--disable-gpu",
                "--disable-background-networking", "--disable-sync", "--no-first-run",
                "--host-resolver-rules=MAP * ~NOTFOUND", "--user-data-dir=" + File(temp, "profile").path,
                "--virtual-time-budget=10000", "--dump-dom", page.toURI().toString())
                .redirectError(stderr).start()
            val dom = p.inputStream.bufferedReader().readText()
            assertEquals(stderr.readText(), 0, p.waitFor())
            assertTrue(dom, dom.contains("<h2>Clip 1</h2>") && dom.contains("<h2>Clip 2</h2>"))
            assertTrue(dom, dom.contains("<p id=\"message\" role=\"status\" aria-live=\"polite\"></p>"))
            assertTrue(dom, dom.contains("aria-label=\"Play Clip 1\""))
        } finally { temp.deleteRecursively() }
    }

    @Test fun invalidPageIdentityOrChangedAudioIsRefused() {
        val (wav, manifest) = fixture()
        for (id in listOf("", "read:bad", "../x", "a".repeat(129))) {
            assertTrue(id, runCatching { FeedbackInputAdapter.fromWav(wav, manifest, id) }.isFailure)
        }
        val invalidChunk = manifest.chunks.first().copy(text = String(charArrayOf(0xd800.toChar())))
        assertTrue(runCatching {
            AudioBoundReadManifest.create("read", wav, 24000, 5, listOf(invalidChunk, manifest.chunks[1]))
        }.isFailure)
        val corrupt = wav.clone().also { it[44] = (it[44].toInt() xor 1).toByte() }
        assertTrue(runCatching { FeedbackInputAdapter.fromWav(corrupt, manifest, "read-ok") }.isFailure)
    }
}
