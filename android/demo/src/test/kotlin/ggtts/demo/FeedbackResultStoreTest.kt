package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files

class FeedbackResultStoreTest {
    private fun input(): PreparedFeedbackInput {
        val audio = java.util.Base64.getEncoder().encodeToString(Flac.encodePcm16(shortArrayOf(100, -100), 24000))
        return PreparedFeedbackInput("""{"format":"ggtts-feedback-input/1","id":"r1","items":[{"audio":"data:audio/flac;base64,""" + audio + """","id":"c0","text":"A""" + 92.toChar().toString() + "u4e2d" + 92.toChar().toString() + "ud840" + 92.toChar().toString() + "udc00!" + """"}]}""")
    }
    private fun result(i: PreparedFeedbackInput, marks: String = "[0,1]", id: String = "c0") =
        """{"format":"ggtts-feedback-result/1","id":"r1","input_sha256":"${i.digest}","note":"","items":[{"id":"$id","marks":$marks,"rating":"unsure","note":""}]}"""

    @Test fun validatesSavesAndRestoresExactInput() {
        val dir = Files.createTempDirectory("feedback").toFile()
        try {
            val i = input(); val store = FeedbackResultStore(dir)
            store.save(i, result(i))
            assertEquals(result(i), FeedbackResultStore(dir).load(i))
            File(dir, i.digest + ".json").writeText(result(i, "[2]"))
            assertNull(store.load(i))
        } finally { dir.deleteRecursively() }
    }

    @Test fun refusesWrongIdentityIndicesAndDuplicateKeys() {
        val i = input()
        for (bad in listOf(result(i, "[2]"), result(i, "[0,0]"), result(i, "[0]", "other"),
            result(i).replace(i.digest, "0".repeat(64)), result(i).replace("\"note\":\"\"", "\"note\":\"\",\"note\":\"\""))) {
            assertTrue(runCatching { i.validateResult(bad) }.isFailure)
        }
        assertTrue(runCatching { PreparedFeedbackInput(i.canonicalJson.replace("r1", "r:1")) }.isFailure)
    }

    @Test fun refusesNoncanonicalAndMalformedFlac() {
        val source = input().canonicalJson
        assertTrue(runCatching { PreparedFeedbackInput(source.replaceFirst(":", ": ")) }.isFailure)
        assertTrue(runCatching { PreparedFeedbackInput(source.replace("data:audio/flac;base64,", "data:audio/flac;base64,AAAA")) }.isFailure)
        assertTrue(runCatching { PreparedFeedbackInput(source.replace("u4e2d", "u4E2D")) }.isFailure)
        assertTrue(feedbackMediaAllowed("data:audio/flac;base64,ZkxhQw=="))
        assertFalse(feedbackMediaAllowed("https://example.com/audio.flac"))
        assertFalse(feedbackMediaAllowed("file:///tmp/audio.flac"))
        assertFalse(feedbackMediaAllowed("data:text/html,hello"))
    }

    @Test fun hostPageUsesValidatedSaveAndHidesImport() {
        val template = File("src/main/assets/feedback-page.html").readText()
        val page = feedbackHostPage(template, input())
        assertTrue(page.contains(">Save feedback</button>"))
        assertTrue(page.contains("<input id=\"import-file\" type=\"file\" hidden disabled>"))
        assertFalse(page.contains("Restore feedback from a JSON file"))
        assertFalse(page.contains("localStorage.setItem"))
        assertFalse(page.contains("localStorage.getItem"))
        assertTrue(page.contains("FeedbackHost.save(JSON.stringify(state))"))
        assertTrue(page.contains("FeedbackHost.load()"))
        assertFalse(page.contains("new Blob("))
        assertTrue(runCatching { feedbackHostPage(template.replace("    function persist() {", ""), input()) }.isFailure)
    }

    @Test fun bundledTemplateMatchesSharedPage() {
        val shared = File("../../feedback/page.html")
        val asset = File("src/main/assets/feedback-page.html")
        assertEquals(shared.readText(), asset.readText())
    }
    @Test fun exportRestoreNeverOverwritesDifferentOrCorruptPrivateResult() {
        val dir = Files.createTempDirectory("feedback-conflict").toFile()
        try {
            val i = input()
            val exported = result(i)
            val different = exported.replace("\"rating\":\"unsure\"", "\"rating\":\"accept\"")
            val store = FeedbackResultStore(dir)
            assertTrue(store.restoreExport(i, exported))
            assertTrue(store.restoreExport(i, exported))
            assertFalse(store.restoreExport(i, different))
            assertEquals(exported, store.load(i))
            val target = File(dir, i.digest + ".json")
            target.writeText("{")
            assertFalse(store.restoreExport(i, exported))
            assertEquals("{", target.readText())
        } finally { dir.deleteRecursively() }
    }
}
