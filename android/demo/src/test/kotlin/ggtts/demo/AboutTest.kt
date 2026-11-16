package ggtts.demo

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AboutTest {
    private fun manifest(): String {
        // Gradle runs unit tests from the module directory.
        val f = File("src/main/AndroidManifest.xml")
        assertTrue("manifest not found from ${File(".").absolutePath}", f.isFile)
        return f.readText()
    }

    @Test
    fun theTaglineFitsAStoreShortDescription() {
        assertTrue("${About.TAGLINE.length} characters", About.TAGLINE.length <= 80)
    }

    @Test
    fun theNoInternetClaimHoldsForTheManifest() {
        // The screen tells the user nothing leaves the phone because the app cannot reach the network. Adding the
        // permission must fail here, so the sentence is changed with it.
        assertTrue(About.PURPOSE.any { "no permission to use the internet" in it })
        assertFalse("the manifest asks for INTERNET; change About.PURPOSE", "android.permission.INTERNET" in manifest())
    }

    @Test
    fun theStatedWaitMatchesTheMeasuredCosts() {
        // About states each engine's measured first-audio range (EngineCosts), the range the player shows for a voice
        // of that engine ("First sound in about 30 to 50 s"), never a span across engines that the player does not.
        val wait = About.PURPOSE.single { "first sound takes" in it }
        for ((engine, cost) in EngineCosts.BY_ENGINE) {
            assertTrue("$engine's ${cost.firstAudioS} s is not in: $wait", "${cost.firstAudioS} " in wait)
        }
        assertEquals("30 to 50", EngineCosts.BY_ENGINE.getValue("qwen3-tts").firstAudioS)
        assertTrue("the wait says 30 to 60, which no voice shows: $wait", "30 to 60" !in wait)
    }

    /**
     * The screen opens saying what the app is for: the header's subtitle is the tagline, and an empty text box names
     * the use by other apps that About states. The layout is not built on the JVM, so its source is held to the names.
     */
    @Test
    fun theReadingScreenOpensWithTheTaglineAndTheOtherAppsHint() {
        assertEquals(About.TAGLINE, About.HEADER_SUBTITLE)
        // The screen shows them through the string resources, in the phone's language; in English they are these.
        assertEquals(About.TAGLINE, About.tagline(EnglishWords))
        assertEquals(About.EMPTY_TEXT_HINT, About.emptyTextHint(EnglishWords))
        val screen = File("src/main/kotlin/ggtts/demo/MainActivity.kt").readText()
        assertTrue("the header's subtitle is not the tagline", "subtitle = About.tagline(tr)" in screen)
        assertTrue("the empty text box does not show the other-apps hint", "About.emptyTextHint(tr)" in screen)
        assertTrue("text-to-speech engine" in About.EMPTY_TEXT_HINT)
        assertTrue(About.PURPOSE.any { "text-to-speech engine" in it })
    }

    @Test
    fun gettingStartedShowsOnlyWithoutModelFiles() {
        assertEquals(3, About.gettingStarted(0).size)
        assertTrue(About.gettingStarted(1).isEmpty())
    }
}
