package ggtts.demo

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import java.util.Base64
import java.util.concurrent.TimeUnit

/** Real-time Chromium checks of the adapter's actual FLAC; independent of the Android WebView host. */
class FeedbackAdapterPlaybackTest {
    private fun fixedHeaderPlant(original: ByteArray): ByteArray {
        var p = 4
        while (true) {
            val last = original[p].toInt() and 0x80 != 0
            val length = ((original[p + 1].toInt() and 255) shl 16) or
                ((original[p + 2].toInt() and 255) shl 8) or (original[p + 3].toInt() and 255)
            p += 4 + length
            if (last) break
        }
        val output = java.io.ByteArrayOutputStream()
        output.write(original, 0, p)
        val sizes = intArrayOf(4096, 3104)
        var sampleOffset = 0
        sizes.forEachIndexed { frameIndex, n ->
            assertEquals(0xff, original[p].toInt() and 255)
            assertEquals(1, original[p + 1].toInt() and 1)
            val sizeCode = (original[p + 2].toInt() and 0xf0) shr 4
            assertEquals(if (n == 4096) 12 else 7, sizeCode)
            val offsetBytes = Flac.utf8Number(sampleOffset.toLong())
            assertArrayEquals(offsetBytes, original.copyOfRange(p + 4, p + 4 + offsetBytes.size))
            val extra = if (sizeCode == 7) 2 else 0
            val oldHeader = java.io.ByteArrayOutputStream()
            oldHeader.write(original, p, 1)
            oldHeader.write(original[p + 1].toInt() and 0xfe) // fixed-block strategy
            oldHeader.write(original, p + 2, 2)
            oldHeader.write(Flac.utf8Number(frameIndex.toLong())) // original frame index
            oldHeader.write(original, p + 4 + offsetBytes.size, extra)
            val header = oldHeader.toByteArray()
            var crc8 = 0
            for (byte in header) {
                crc8 = crc8 xor (byte.toInt() and 255)
                repeat(8) { crc8 = if (crc8 and 0x80 != 0) (crc8 shl 1) xor 0x07 else crc8 shl 1; crc8 = crc8 and 255 }
            }
            val frame = java.io.ByteArrayOutputStream()
            frame.write(header)
            frame.write(crc8)
            val subframe = p + 4 + offsetBytes.size + extra + 1
            assertEquals(2, original[subframe].toInt() and 255)
            frame.write(original, subframe, 1 + n * 2)
            val frameBytes = frame.toByteArray()
            var crc16 = 0
            for (byte in frameBytes) {
                crc16 = crc16 xor ((byte.toInt() and 255) shl 8)
                repeat(8) { crc16 = if (crc16 and 0x8000 != 0) (crc16 shl 1) xor 0x8005 else crc16 shl 1; crc16 = crc16 and 65535 }
            }
            output.write(frameBytes)
            output.write(crc16 shr 8)
            output.write(crc16 and 255)
            p = subframe + 1 + n * 2 + 2
            sampleOffset += n
        }
        assertEquals(original.size, p)
        return output.toByteArray()
    }

    private fun decodeFlac(bytes: ByteArray): ByteArray {
        val temp = kotlin.io.path.createTempDirectory("ggtts-old-flac-").toFile()
        try {
            val file = File(temp, "clip.flac").apply { writeBytes(bytes) }
            val raw = File(temp, "clip.raw")
            val process = ProcessBuilder("ffmpeg", "-v", "error", "-y", "-i", file.path,
                "-f", "s16le", "-acodec", "pcm_s16le", raw.path).redirectErrorStream(true).start()
            val output = process.inputStream.bufferedReader().readText()
            assertEquals(output, 0, process.waitFor())
            return raw.readBytes()
        } finally { temp.deleteRecursively() }
    }

    @Test fun realTimeBrowserPlaysAdapterFlacAndRejectsOldHeader() {
        val chrome = listOfNotNull(System.getenv("CHROME_BIN"), "/usr/bin/google-chrome", "/usr/bin/chromium",
            "/usr/bin/chromium-browser").firstOrNull { File(it).canExecute() }
        assumeTrue("Chromium required (set CHROME_BIN)", chrome != null)
        val python = runCatching { ProcessBuilder("python3", "--version").start().waitFor() == 0 }.getOrDefault(false)
        assumeTrue("python3 required", python)
        val ffmpeg = runCatching { ProcessBuilder("ffmpeg", "-version").start().waitFor() == 0 }.getOrDefault(false)
        assumeTrue("FFmpeg required for both PCM equality checks", ffmpeg)
        var root = File(System.getProperty("user.dir") ?: error("missing user.dir")).absoluteFile
        while (!File(root, "feedback/page.html").isFile) root = root.parentFile ?: break
        val template = File(root, "feedback/page.html")
        val driver = File(root, "scripts/feedback_browser_driver.py")
        assumeTrue("shared template and browser driver required", template.isFile && driver.isFile)

        val samples = FloatArray(7200) { i -> (kotlin.math.sin(i * 0.05) * 0.15).toFloat() }
        val wav = Wav.encode(samples, 24000)
        val evidence = VerifiedWavEvidence()
        evidence.add(0, "A中𠀀", 1, 0, 0, 0, samples.size.toLong(), samples, 24000)
        evidence.complete(true)
        val manifest = evidence.verify("real-time", wav)
        val clip = FeedbackPcmClips.fromWav(wav, manifest).single().flac
        val input = FeedbackInputAdapter.fromWav(wav, manifest, "browser-adapter")
        val originalUri = Base64.getEncoder().encodeToString(clip)
        assertTrue(input.canonicalJson.contains(originalUri))
        val planted = fixedHeaderPlant(clip)
        val savedPcm = wav.copyOfRange(44, wav.size)
        assertArrayEquals("corrected FLAC PCM", savedPcm, decodeFlac(clip))
        assertArrayEquals("old fixed-block plant PCM", savedPcm, decodeFlac(planted))
        val badJson = input.canonicalJson.replace(originalUri, Base64.getEncoder().encodeToString(planted))
        assertNotEquals(input.canonicalJson, badJson)
        val badHash = MessageDigest.getInstance("SHA-256").digest(badJson.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        val harness = """
<script>
(async()=>{
 const done=(ok,why)=>{document.body.dataset.hostTest=ok?'PASS':'FAIL';document.body.dataset.hostErrors=why;};
 try {
  const audio=document.querySelector('audio');
  if (!audio) return done(false,'no audio: input or digest rejected');
  let playing=false; audio.addEventListener('playing',()=>playing=true,{once:true});
  const ended=new Promise(resolve=>audio.addEventListener('ended',resolve,{once:true}));
  await audio.play();
  await Promise.race([ended,new Promise(resolve=>setTimeout(resolve,5000))]);
  done(playing && audio.ended && audio.currentTime>0, 'playback did not reach ended');
 } catch(e) { done(false,'FLAC play failed: '+e.name); }
})();
</script>
"""
        fun run(json: String, digest: String): Pair<Int, String> {
            val temp = kotlin.io.path.createTempDirectory("ggtts-adapter-playback-").toFile()
            try {
                val marker = "<!-- GGTTS_INPUT -->"
                val source = template.readText()
                assertEquals(1, source.split(marker).size - 1)
                val embedded = "<meta name=\"feedback-input-sha256\" content=\"" + digest +
                    "\"><script id=\"feedback-input\" type=\"application/json\">" + json + "</script>"
                val page = File(temp, "page.html").apply {
                    writeText(source.replace(marker, embedded).replace("</body>", harness + "</body>"))
                }
                val out = File(temp, "out")
                val err = File(temp, "err")
                val process = ProcessBuilder("python3", driver.path, chrome!!, page.path)
                    .redirectOutput(out).redirectError(err).start()
                if (!process.waitFor(35, TimeUnit.SECONDS)) { process.destroyForcibly(); fail("browser timed out") }
                return process.exitValue() to (out.readText() + err.readText())
            } finally { temp.deleteRecursively() }
        }
        val good = run(input.canonicalJson, input.sha256)
        assertEquals(good.second, 0, good.first)
        val bad = run(badJson, badHash)
        assertNotEquals("old fixed-block plant passed: " + bad.second, 0, bad.first)
        assertTrue("old fixed-block plant failed for wrong reason: " + bad.second,
            bad.second.contains("FLAC play failed: NotSupportedError"))
    }
}
