package dev.ggtts.ggtts_app

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class FeedbackPageTest {
    private val root = File(System.getProperty("ggtts.repoRoot") ?: error("missing ggtts.repoRoot"))
    private val asset = File(root, "apps/flutter/assets/feedback-page.html")
    private val template = asset.readText()

    @Test fun assetMatchesSharedTemplateAndAdaptationRemainsLocal() {
        assertArrayEquals(File(root, "feedback/page.html").readBytes(), asset.readBytes())
        val page = FeedbackPage.adapt(template, "{}", "a".repeat(64))
        assertTrue(page.contains("FeedbackHost.save(JSON.stringify(state))"))
        assertTrue(page.contains("FeedbackHost.submit(JSON.stringify(state))"))
        assertTrue(page.contains("FeedbackHost.load()"))
        assertFalse(page.contains("localStorage.setItem(storageKey"))
        assertFalse(page.contains("const blob = new Blob("))
        assertFalse(page.contains("<!-- GGTTS_INPUT -->"))
    }

    @Test fun malformedOrDuplicatedTemplateAnchorsFailClosed() {
        for (bad in listOf(template.replace("<!-- GGTTS_INPUT -->", ""),
            template + "<!-- GGTTS_INPUT -->",
            template.replace("        const blob = new Blob(", "        const altered = new Blob("),
            template.replace("    function persist() {", "    function altered() {"))) {
            try { FeedbackPage.adapt(bad, "{}", "a".repeat(64)); fail("changed template accepted") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun onlyInlineFlacMediaIsAllowed() {
        assertTrue(FeedbackPage.mediaAllowed("data:audio/flac;base64,AA=="))
        for (url in listOf("https://example.test/audio", "file:///x", "content://x",
            "data:text/html,hi", "DATA:audio/flac;base64,AA==")) {
            assertFalse(url, FeedbackPage.mediaAllowed(url))
        }
    }

    @Test fun draftPersistsAndOversizeAndWriteFailureAreRefused() {
        val dir = kotlin.io.path.createTempDirectory("feedback-draft-test").toFile()
        try {
            val draft = FeedbackDraft(File(dir, "draft"))
            assertEquals("", draft.load())
            assertTrue(draft.save("{\"note\":\"kept\"}"))
            assertEquals("{\"note\":\"kept\"}", FeedbackDraft(File(dir, "draft")).load())
            assertFalse(draft.save("x".repeat(1048577)))
            assertEquals("{\"note\":\"kept\"}", draft.load())
            assertFalse(FeedbackDraft(File(dir, "missing/child")).save("x"))
        } finally { dir.deleteRecursively() }
    }
}
