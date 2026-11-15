// Storage's parts apart from Android: which files the screen lists from the app's directory and the picked folder
// (and from where), the descriptor path a picked file reaches the engine as, what a saved read is called, which voice
// samples still go to Music/ggtts/, and that the voices are read through the same opener whichever place a file is in.
package ggtts.demo

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class StorageTest {
    private fun uri(name: String) = "content://com.android.externalstorage.documents/tree/primary%3ADownload%2Fggtts/document/primary%3ADownload%2Fggtts%2F$name"

    @Test
    fun aNameInBothPlacesIsTheAppDirectorysAndTheFolderAddsTheRest() {
        val app = listOf(Entry("talker.gguf", "/data/models/talker.gguf"))
        val folder = listOf(Entry("talker.gguf", uri("talker.gguf")), Entry("codec.gguf", uri("codec.gguf")))
        val c = Catalog.merge(app, folder)
        assertEquals(listOf("codec.gguf", "talker.gguf"), c.names)
        assertEquals("/data/models/talker.gguf", c.location("talker.gguf"))
        assertEquals(uri("codec.gguf"), c.location("codec.gguf"))
        assertTrue(Storage.isContent(c.location("codec.gguf")!!))
        assertFalse(Storage.isContent(c.location("talker.gguf")!!))
        assertNull(c.location("missing.gguf"))
    }

    @Test
    fun aCopyInProgressAndANameThatIsNotAPlainFileNameAreNeverListed() {
        val folder = listOf(
            Entry("lm.gguf.part", uri("lm.gguf.part")),
            Entry("../escape.gguf", uri("x")),
            Entry("sub/x.gguf", uri("y")),
            Entry("", uri("z")),
            Entry("lm.gguf", uri("lm.gguf")),
        )
        assertEquals(listOf("lm.gguf"), Catalog.merge(emptyList(), folder).names)
    }

    @Test
    fun theAppDirectoryIsListedAsPathsWithNothingPicked() {
        val dir = Files.createTempDirectory("ggtts-storage").toFile()
        File(dir, "a.gguf").writeBytes(byteArrayOf(1))
        File(dir, "b.json.part").writeBytes(byteArrayOf(1))
        val c = Catalog.merge(Catalog.appEntries(dir), emptyList())
        assertEquals(listOf("a.gguf"), c.names)
        assertEquals(File(dir, "a.gguf").path, c.location("a.gguf"))
    }

    @Test
    fun voicesAreReadThroughTheOpenerWhicheverPlaceAFileIsIn() {
        // A talker in the app directory, its codec only in the picked folder: one voice per speaker, with both.
        val bytes = mapOf(
            "/app/talker.gguf" to gguf("qwen3-tts", listOf("hk_voice_343")),
            uri("codec.gguf") to gguf("qwen3-tts-tokenizer", emptyList()),
        )
        val c = Catalog.merge(listOf(Entry("talker.gguf", "/app/talker.gguf")), listOf(Entry("codec.gguf", uri("codec.gguf"))))
        val opened = mutableListOf<String>()
        val files = Voices.scan(c.names) { name -> c.open(name) { loc -> opened += loc; ByteArrayInputStream(bytes.getValue(loc)) } }
        assertEquals(setOf("/app/talker.gguf", uri("codec.gguf")), opened.toSet())
        val found = Voices.group(files)
        assertEquals(listOf("hk_voice_343"), found.voices.map { it.name })
        assertEquals("codec.gguf", found.voices.single().codec)
    }

    @Test
    fun openingANameNotListedFails() {
        try {
            Catalog.EMPTY.open("x.gguf") { ByteArrayInputStream(ByteArray(0)) }
            fail("opened a name that is not listed")
        } catch (e: FileNotFoundException) {
            assertEquals("x.gguf", e.message)
        }
    }

    @Test
    fun aSavedReadIsNamedByItsEngineAndTimeSoSavesNeverCollide() {
        val a = Storage.readName("qwen3-tts", 1_700_000_000_000L)
        val b = Storage.readName("qwen3-tts", 1_700_000_001_000L)
        assertTrue(a, Regex("ggtts-qwen3-tts-\\d{8}-\\d{6}\\.wav").matches(a))
        assertTrue(a != b)
        assertEquals("Music/ggtts/", Storage.MUSIC_PATH)
    }

    @Test
    fun onlySamplesNotAlreadyInMusicAreExported() {
        val samples = listOf("hk_voice_777.sample.wav", "talker.gguf", "hk_voice_343.sample.wav")
        assertEquals(listOf("hk_voice_343.sample.wav", "hk_voice_777.sample.wav"), Storage.samplesToExport(samples, emptySet()))
        assertEquals(listOf("hk_voice_343.sample.wav"), Storage.samplesToExport(samples, setOf("hk_voice_777.sample.wav")))
    }

    @Test
    fun aPickedFileReachesTheEngineAsItsDescriptorsPathAndAPathPassesThrough() {
        // The exact form ggtts_core::fdpath takes as a descriptor; any other form would be reopened by path and refused.
        assertEquals("/proc/self/fd/42", Storage.fdPath(42))
        assertEquals("/sdcard/x.gguf", PathResolver.PLAIN.open("/sdcard/x.gguf").path)
        // The folder's model files are listed where they are, not copied: their location is the document URI.
        val c = Catalog.merge(emptyList(), listOf(Entry("talker.gguf", uri("talker.gguf"))))
        assertTrue(Storage.isContent(c.location("talker.gguf")!!))
    }

    /** A GGUF header with general.architecture and, if any, the talker's speaker names. */
    private fun gguf(arch: String, speakers: List<String>): ByteArray {
        val o = ByteArrayOutputStream()
        fun u32(v: Long) = (0 until 4).forEach { o.write(((v shr (8 * it)) and 0xff).toInt()) }
        fun u64(v: Long) = (0 until 8).forEach { o.write(((v shr (8 * it)) and 0xff).toInt()) }
        fun str(s: String) { val b = s.toByteArray(); u64(b.size.toLong()); o.write(b) }
        o.write("GGUF".toByteArray()); u32(3); u64(0); u64(if (speakers.isEmpty()) 1 else 2)
        str("general.architecture"); u32(8); str(arch)
        if (speakers.isNotEmpty()) {
            str(GgufInfo.SPEAKER_NAMES); u32(9); u32(8); u64(speakers.size.toLong()); speakers.forEach { str(it) }
        }
        return o.toByteArray()
    }
}
