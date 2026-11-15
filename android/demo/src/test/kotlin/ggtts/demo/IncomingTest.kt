// Text from other apps: which intents fill the text card, which are refused and why, the stated size limits, and that
// nothing here can start a read.
package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File

class IncomingTest {
    private fun refusedWhy(i: Incoming): String = (i as? Incoming.Refused)?.why ?: error("not refused: $i")

    /** MainActivity's onCreate, from its signature to its closing brace. */
    private fun onCreate(): String {
        val f = File("src/main/kotlin/ggtts/demo/MainActivity.kt")
        assertTrue("MainActivity.kt not found from ${File(".").absolutePath}", f.isFile)
        val src = f.readText()
        val start = src.indexOf("override fun onCreate(savedInstanceState: Bundle?)")
        assertTrue("no onCreate in MainActivity.kt", start >= 0)
        return src.substring(start, src.indexOf("\n    }\n", start))
    }

    /**
     * A screen takes its intent on a fresh start only: not when it is re-created from saved state, and not when its
     * task is started again from Recents with its old intent (no saved state, but the history flag).
     */
    @Test
    fun onlyAFreshStartTakesItsIntent() {
        assertTrue(Launch.takesIntent(null, 0))
        assertTrue("other flags do not matter", Launch.takesIntent(null, 0x10000000))
        assertFalse("a re-created screen would put the old share back over the user's edits", Launch.takesIntent(Any(), 0))
        assertFalse(
            "a task started again from Recents carries its old share",
            Launch.takesIntent(null, Launch.FLAG_LAUNCHED_FROM_HISTORY or 0x10000000),
        )
        assertEquals("Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY", 0x00100000, Launch.FLAG_LAUNCHED_FROM_HISTORY)
    }

    /** The saved state carries the card's text up to its cap, and no further; longer text goes to a file. */
    @Test
    fun theSavedStateCarriesTheCardsTextOnlyUpToItsCap() {
        assertTrue(Launch.fitsSavedState("a".repeat(Launch.MAX_SAVED_CHARS)))
        assertFalse(Launch.fitsSavedState("a".repeat(Launch.MAX_SAVED_CHARS + 1)))
        assertTrue("the cap stays well under the binder's 1 MB (2 bytes a unit)", Launch.MAX_SAVED_CHARS * 2 <= 256 * 1024)
        val src = File("src/main/kotlin/ggtts/demo/MainActivity.kt").readText()
        val save = src.substring(src.indexOf("override fun onSaveInstanceState"), src.indexOf("override fun onTrimMemory"))
        val check = save.indexOf("if (Launch.fitsSavedState(s))")
        val put = save.indexOf("outState.putString(STATE_TEXT, s)")
        assertTrue("onSaveInstanceState checks the cap before it saves the text:\n$save", check in 0 until put)
        assertTrue("a longer card goes to a file:\n$save", "File(cacheDir, CARD_FILE).writeText(s)" in save)
    }

    /**
     * The review finding on PR 404: onCreate took its intent on every creation, so a re-created screen put the original
     * share back over the user's edits. It now takes it only under [Launch.takesIntent], and restores the card's text
     * from the saved state. Deleting the guard (a bare `applyIntent(intent)`) fails this test.
     */
    @Test
    fun aRecreatedScreenKeepsTheUsersTextAndDoesNotRetakeTheShare() {
        val body = onCreate()
        val calls = Regex("""applyIntent\(""").findAll(body).count()
        assertEquals("onCreate calls applyIntent once, under the guard:\n$body", 1, calls)
        assertTrue(
            "onCreate takes its intent only on a fresh start:\n$body",
            "if (Launch.takesIntent(savedInstanceState, intent.flags)) applyIntent(intent)" in body,
        )
        assertTrue("onCreate restores the card's text:\n$body", "saved.getString(STATE_TEXT)" in body)
        assertTrue("onCreate restores a long card from its file:\n$body", "saved.getString(STATE_TEXT_FILE)" in body)
    }

    @Test
    fun charactersAreCountedAsAReaderCountsThem() {
        assertEquals("an emoji is one character, not two UTF-16 units", 3, IncomingText.characters("a😀字"))
        val atLimit = "😀".repeat(IncomingText.MAX_CHARS)
        assertEquals(Incoming.Text(atLimit, "shared"), IncomingText.parse(IncomingText.SEND, "text/plain", sendText = atLimit))
    }

    @Test
    fun theStatusSaysWhenSharedTextReplacedTheCard() {
        assertEquals(
            "Text shared from another app: 15 characters. Press Read aloud to hear it.",
            IncomingText.status("shared", 15, replaced = false),
        )
        assertTrue(IncomingText.status("selected", 9, replaced = true).contains("in place of what the card held"))
    }

    @Test
    fun sharedPlainTextFillsTheCard() {
        assertEquals(
            Incoming.Text("今日天氣好好。", "shared"),
            IncomingText.parse(IncomingText.SEND, "text/plain", sendText = "  今日天氣好好。\r\n"),
        )
        assertEquals("a charset parameter is still plain text", Incoming.Text("Hi.", "shared"),
            IncomingText.parse(IncomingText.SEND, "text/plain; charset=utf-8", sendText = "Hi."))
    }

    @Test
    fun theSelectionMenusTextFillsTheCard() {
        assertEquals(
            Incoming.Text("行完之後，去食飯。", "selected"),
            IncomingText.parse(IncomingText.PROCESS_TEXT, "text/plain", processText = "行完之後，去食飯。"),
        )
    }

    @Test
    fun aSharedOrOpenedFileIsReadByItsContentUri() {
        assertEquals(Incoming.File("content://p/notes.txt"),
            IncomingText.parse(IncomingText.SEND, "text/plain", stream = "content://p/notes.txt"))
        assertEquals(Incoming.File("content://p/notes.txt"),
            IncomingText.parse(IncomingText.VIEW, "text/plain", data = "content://p/notes.txt"))
        assertTrue(refusedWhy(IncomingText.parse(IncomingText.VIEW, "text/plain", data = "file:///sdcard/a.txt"))
            .contains("not a path"))
    }

    @Test
    fun aLaunchOrOtherIntentIsNotAHandOver() {
        assertEquals(Incoming.None, IncomingText.parse("android.intent.action.MAIN", null))
        assertEquals(Incoming.None, IncomingText.parse(null, null, sendText = "ignored without SEND"))
        assertEquals("VIEW with no data", Incoming.None, IncomingText.parse(IncomingText.VIEW, null))
    }

    @Test
    fun whatIsNotPlainTextOrHasNoTextIsRefused() {
        assertTrue(refusedWhy(IncomingText.parse(IncomingText.SEND, "image/png", stream = "content://p/a.png"))
            .contains("image/png"))
        assertTrue(refusedWhy(IncomingText.parse(IncomingText.SEND, "text/plain")).contains("no text"))
        assertTrue(refusedWhy(IncomingText.parse(IncomingText.SEND, "text/plain", sendText = " \n\t ")).contains("no text"))
        assertTrue(refusedWhy(IncomingText.parse(IncomingText.VIEW, "application/pdf", data = "content://p/a.pdf"))
            .contains("application/pdf"))
    }

    @Test
    fun theCharacterLimitIsStatedAndEnforced() {
        val atLimit = "字".repeat(IncomingText.MAX_CHARS)
        assertEquals(Incoming.Text(atLimit, "shared"), IncomingText.parse(IncomingText.SEND, "text/plain", sendText = atLimit))
        val why = refusedWhy(IncomingText.parse(IncomingText.SEND, "text/plain", sendText = atLimit + "字"))
        assertTrue(why, why.contains("${IncomingText.MAX_CHARS + 1} characters") && why.contains("${IncomingText.MAX_CHARS}"))
    }

    @Test
    fun aTextFileIsReadAsUtf8WithinItsLimits() {
        val bom = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte())
        assertEquals(Incoming.Text("今日天氣好好。", "opened"),
            IncomingText.readFile(ByteArrayInputStream(bom + "今日天氣好好。\n".toByteArray())))
        val notUtf8 = IncomingText.readFile(ByteArrayInputStream(byteArrayOf(0xC3.toByte(), 0x28)))
        assertTrue(refusedWhy(notUtf8).contains("not UTF-8"))
        val atLimit = ByteArray(IncomingText.MAX_FILE_BYTES) { 'a'.code.toByte() }
        // At the byte limit it is read; it is then over the character limit, which refuses it by that limit instead.
        assertTrue(refusedWhy(IncomingText.readFile(ByteArrayInputStream(atLimit))).contains("characters"))
        val over = IncomingText.readFile(ByteArrayInputStream(ByteArray(IncomingText.MAX_FILE_BYTES + 1) { 'a'.code.toByte() }))
        assertTrue(refusedWhy(over).contains("over ${IncomingText.MAX_FILE_BYTES / 1024} KB"))
    }

    @Test
    fun controlCharactersAreDroppedAndLineEndsKept() {
        assertEquals("a\nb\tc", IncomingText.clean("\u0000a\r\nb\tc\u0007"))
    }
}
