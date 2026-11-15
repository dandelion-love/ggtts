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
        // "30 to 60 seconds" spans the measured first-audio ranges of every engine (EngineCosts).
        val bounds = EngineCosts.BY_ENGINE.values.flatMap { c -> c.firstAudioS.split(" to ").map { it.trim().toInt() } }
        assertEquals(30, bounds.min())
        assertEquals(60, bounds.max())
        assertTrue(About.PURPOSE.any { "30 to 60 seconds" in it })
    }

    @Test
    fun gettingStartedShowsOnlyWithoutModelFiles() {
        assertEquals(3, About.gettingStarted(0).size)
        assertTrue(About.gettingStarted(1).isEmpty())
    }
}
