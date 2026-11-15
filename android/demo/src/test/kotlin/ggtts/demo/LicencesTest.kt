// The licences and privacy screen's content: the notices are the ones the binding's AAR carries (read from its
// resources here, from the APK on a phone), a missing one is said, and the privacy statement is held to the manifest.
package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class LicencesTest {
    private val loader = javaClass.classLoader!!

    /** The paths under META-INF/ggtts/ on the test classpath: the AAR's notices, as the APK carries them. */
    private fun aarEntries(): List<String> {
        val url = loader.getResource(Licences.DIR + "LICENSE")
        assertTrue("META-INF/ggtts/LICENSE is not on the test classpath (build with scripts/android_build.sh)", url != null)
        val dir = File(url!!.toURI()).parentFile
        return dir.walkTopDown().filter { it.isFile }.map { Licences.DIR + it.relativeTo(dir).invariantSeparatorsPath }.toList()
    }

    /** #413 AC1: the screen lists the notices the AAR carries, ggtts's own LICENSE first, and no build record. */
    @Test
    fun theNoticesAreTheOnesTheAarCarries() {
        val entries = aarEntries()
        val notices = Licences.notices(entries) { loader.getResourceAsStream(it) }
        assertEquals("LICENSE", notices.first().name)
        val names = notices.map { it.name }.toSet()
        val expected = entries.map { it.removePrefix(Licences.DIR) }.filter { it != "BUILDINFO.json" }.toSet()
        assertEquals("every notice file in the AAR is listed", expected, names)
        assertTrue("THIRD_PARTY.md" in names && names.any { it.startsWith("licenses/") })
        assertFalse("the build record is not a notice", "BUILDINFO.json" in names)
        assertTrue("every notice was read", notices.all { !it.text.isNullOrBlank() })
    }

    /** #413 AC4: a notice the APK lacks is listed as missing, by name, and the rest still read (planted). */
    @Test
    fun aMissingNoticeIsSaidAndTheRestStillRead() {
        val entries = listOf("META-INF/ggtts/LICENSE", "META-INF/ggtts/licenses/gone.txt", "META-INF/ggtts/THIRD_PARTY.md", "other/file")
        val notices = Licences.notices(entries) { path ->
            if (path.endsWith("gone.txt")) null else "text of $path".byteInputStream()
        }
        assertEquals(listOf("LICENSE", "THIRD_PARTY.md", "licenses/gone.txt"), notices.map { it.name })
        assertNull(notices.single { it.name == "licenses/gone.txt" }.text)
        assertEquals("text of META-INF/ggtts/LICENSE", notices.first().text)
        assertEquals("licenses/gone.txt is missing from this build.", EnglishWords.get("notice_missing", "licenses/gone.txt"))
    }

    /** #413 AC3: the privacy statement says the app collects no data, and the manifest asks for no internet. */
    @Test
    fun thePrivacyStatementIsHeldToTheManifest() {
        val statement = EnglishWords.get("privacy_statement")
        assertTrue(statement, "collects no data" in statement && "no permission to use the internet" in statement)
        val manifest = File("src/main/AndroidManifest.xml").readText()
        assertTrue("the manifest asks for INTERNET; the privacy statement is wrong", Licences.hasNoInternetPermission(manifest))
        // Planted: the same manifest with the permission added fails the check.
        val planted = manifest.replace("<application", "<uses-permission android:name=\"android.permission.INTERNET\" />\n    <application")
        assertFalse(Licences.hasNoInternetPermission(planted))
        // A comment that names the permission is not a request for it.
        val comment = manifest.replace("<application", "<!-- not android.permission.INTERNET -->\n    <application")
        assertTrue(Licences.hasNoInternetPermission(comment))
    }

    /** #413 AC1: the ⋮ menu opens the screen. */
    @Test
    fun theMenuOpensTheScreen() {
        val src = File("src/main/kotlin/ggtts/demo/MainActivity.kt").readText()
        assertTrue("header.menu.add(0, 5, 4, getString(R.string.menu_licences))" in src)
        assertTrue("5 -> showLicences()" in src)
    }
}
