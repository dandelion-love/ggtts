// The import's checks, each with the input it must refuse: a file of the wrong type, an empty file, a GGUF name on
// other bytes, a default model file whose bytes are not the published ones, a same-named file already there, and a
// copy cut short. A refused import leaves nothing listed.
package ggtts.demo

import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class ModelImportTest {
    @Test
    fun oneRefusedExtraIsReportedSoTheIntentIsRefused() {
        assertEquals(emptyList<String>(), ModelImport.refusedNames(listOf("lm.gguf", null, "voice.json")))
        assertEquals(listOf("../x.gguf"), ModelImport.refusedNames(listOf("lm.gguf", "../x.gguf", null)))
        assertEquals(listOf("/a", ".."), ModelImport.refusedNames(listOf("/a", "s3gen.gguf", "..")))
    }

    @Test
    fun scriptedModelExtrasMustBePlainNamesInTheModelsDirectory() {
        for (ok in listOf("kitten-lm.gguf", "voice.json", "a b.gguf", ".hidden.gguf")) {
            assertTrue(ok, ModelImport.isPlainName(ok))
        }
        for (bad in listOf("", ".", "..", "/sdcard/x.gguf", "../x.gguf", "sub/x.gguf", "..\\x.gguf", "x\u0000.gguf")) {
            assertFalse(bad, ModelImport.isPlainName(bad))
        }
    }

    private val dir: File = Files.createTempDirectory("ggtts-import").toFile()
    private val gguf = "GGUF".toByteArray() + ByteArray(64) { it.toByte() }

    private fun sha256(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    private fun refused(imp: ModelImport, name: String, size: Long?, bytes: ByteArray, says: String) {
        try {
            imp.importFile(name, size, ByteArrayInputStream(bytes))
            fail("$name was imported")
        } catch (e: ImportException) {
            assertTrue("'${e.message}' says '$says'", e.message!!.contains(says))
        }
        assertEquals("nothing new is listed", emptyList<String>(), ModelImport.listed(dir).filter { it == File(name).name })
        assertFalse("no .part is left", File(dir, File(name).name + ModelImport.PART).exists())
    }

    @Test
    fun aGgufAndAVoiceJsonAreImported() {
        val imp = ModelImport(dir)
        imp.importFile("lm.gguf", gguf.size.toLong(), ByteArrayInputStream(gguf))
        imp.importFile("voice.json", null, ByteArrayInputStream("  {\"v\": 1}".toByteArray()))
        assertEquals(listOf("lm.gguf", "voice.json"), ModelImport.listed(dir))
        assertTrue(File(dir, "lm.gguf").readBytes().contentEquals(gguf))
    }

    @Test
    fun theWrongTypeIsRefused() = refused(ModelImport(dir), "photo.jpg", 10, ByteArray(10), "not a model file")

    @Test
    fun anEmptyFileIsRefused() {
        refused(ModelImport(dir), "lm.gguf", 0, ByteArray(0), "is empty")
        refused(ModelImport(dir), "lm.gguf", null, ByteArray(0), "is empty")
    }

    @Test
    fun aGgufNameOnOtherBytesIsRefused() =
        refused(ModelImport(dir), "lm.gguf", 8, "not gguf".toByteArray(), "not a GGUF file")

    @Test
    fun aJsonNameOnOtherBytesIsRefused() =
        refused(ModelImport(dir), "voice.json", 4, "GGUF".toByteArray(), "not a voice JSON file")

    @Test
    fun aDefaultModelFileWithOtherBytesIsRefused() {
        refused(ModelImport(dir), "qwen-tokenizer-12hz-q8_0.gguf", gguf.size.toLong(), gguf, "published sha256")
    }

    @Test
    fun aKnownFileWithItsPublishedBytesIsImported() {
        val imp = ModelImport(dir, mapOf("codec.gguf" to sha256(gguf)))
        imp.importFile("codec.gguf", gguf.size.toLong(), ByteArrayInputStream(gguf))
        assertEquals(listOf("codec.gguf"), ModelImport.listed(dir))
        val other = gguf.copyOf().also { it[10] = 99 }
        refused(ModelImport(dir, mapOf("codec2.gguf" to sha256(gguf))), "codec2.gguf", other.size.toLong(), other,
            "published sha256")
    }

    @Test
    fun aSameNamedFileIsNotReplaced() {
        File(dir, "lm.gguf").writeBytes(gguf)
        try {
            ModelImport(dir).importFile("lm.gguf", gguf.size.toLong(), ByteArrayInputStream(gguf + gguf))
            fail("replaced")
        } catch (e: ImportException) {
            assertTrue(e.message!!.contains("already imported"))
        }
        assertTrue("the old file is untouched", File(dir, "lm.gguf").readBytes().contentEquals(gguf))
    }

    @Test
    fun aCopyCutShortIsNotListed() {
        // The source fails partway, as a cancelled picker or a pulled card would.
        val failing = object : InputStream() {
            var n = 0
            override fun read(): Int = if (n < gguf.size) gguf[n++].toInt() and 0xff else throw IOException("gone")
        }
        try {
            ModelImport(dir).importFile("lm.gguf", 1000, failing)
            fail("imported")
        } catch (e: ImportException) {
            assertTrue(e.message!!, e.message!!.contains("gone"))
        }
        assertEquals(emptyList<String>(), ModelImport.listed(dir))
        assertFalse(File(dir, "lm.gguf.part").exists())
    }

    @Test
    fun midCopyOnlyThePartFileExists() {
        // What .part is for: a copy killed partway (no catch block runs) must leave nothing under the real name. So
        // look from inside the copy: partway through reading, NAME.part exists and NAME does not.
        val big = gguf + ByteArray(4 shl 20) { (it % 251).toByte() }
        val seen = mutableListOf<Pair<Boolean, Boolean>>()
        val watching = object : InputStream() {
            var n = 0
            override fun read(): Int = if (n < big.size) big[n++].toInt() and 0xff else -1
            override fun read(b: ByteArray, off: Int, len: Int): Int {
                if (n >= big.size) return -1
                if (n > 2 shl 20 && seen.isEmpty()) {
                    seen += File(dir, "lm.gguf.part").exists() to File(dir, "lm.gguf").exists()
                }
                val k = minOf(len, big.size - n)
                System.arraycopy(big, n, b, off, k)
                n += k
                return k
            }
        }
        ModelImport(dir).importFile("lm.gguf", big.size.toLong(), watching)
        assertEquals("looked once, partway through", 1, seen.size)
        assertEquals("partway: NAME.part exists, NAME does not", true to false, seen.single())
        assertEquals(listOf("lm.gguf"), ModelImport.listed(dir))
        assertFalse(File(dir, "lm.gguf.part").exists())
    }

    @Test
    fun aVoiceJsonWithLongLeadingWhitespaceIsImported() {
        ModelImport(dir).importFile("voice.json", null, ByteArrayInputStream("\n\n    \t {\"v\": 1}".toByteArray()))
        assertEquals(listOf("voice.json"), ModelImport.listed(dir))
    }

    @Test
    fun aCopyShorterThanTheReportedSizeIsRefused() =
        refused(ModelImport(dir), "lm.gguf", gguf.size + 10L, gguf, "incomplete")

    @Test
    fun aPartFileIsNeverListed() {
        File(dir, "lm.gguf.part").writeBytes(gguf)
        assertEquals(emptyList<String>(), ModelImport.listed(dir))
    }
}
