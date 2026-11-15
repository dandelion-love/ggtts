// TalkBack labels the screen gives beyond visible text, held to the screen's source (the layout is built in code and
// not on the JVM). The rest is checked on a phone with scripts/android_a11y_check.py over uiautomator dumps.
package ggtts.demo

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class TalkBackTest {
    private val src by lazy {
        val f = File("src/main/kotlin/ggtts/demo/MainActivity.kt")
        assertTrue("MainActivity.kt not found from ${File(".").absolutePath}", f.isFile)
        f.readText()
    }

    /** #412 AC1: each Advanced spinner is named by its label (labelFor), so TalkBack says what it chooses. */
    @Test
    fun everySpinnerIsNamedByItsLabel() {
        for (s in listOf("model", "codec", "voice", "threads")) {
            assertTrue("the $s spinner has no labelling label", Regex("""labelling\([^)]*\), $s\)\); adv\.addView\($s\)""").containsMatchIn(src))
        }
        assertTrue("labelling sets labelFor", "labelFor = control.id" in src)
    }

    /** #412 AC1: controls whose text carries a symbol, or the sheet's outside, have a spoken label of their own. */
    @Test
    fun symbolButtonsAndTheSheetsOutsideHaveSpokenLabels() {
        assertTrue("contentDescription = getString(R.string.read_aloud_label)" in src)
        assertTrue("contentDescription = getString(R.string.stop_label)" in src)
        assertTrue("contentDescription = getString(R.string.play_sample, v.name)" in src)
        assertTrue("dialog.findViewById<View>(M.id.touch_outside)?.contentDescription = getString(R.string.close_sheet)" in src)
    }

    /** #412 AC3: the check fails when a spinner loses its label (planted on the source text). */
    @Test
    fun aSpinnerWithoutItsLabelIsCaught() {
        val planted = src.replace("adv.addView(labelling(getString(R.string.codec_label), codec)); adv.addView(codec)",
            "adv.addView(label(getString(R.string.codec_label))); adv.addView(codec)")
        assertTrue("the plant did not apply", planted != src)
        assertTrue(!Regex("""labelling\([^)]*\), codec\)\); adv\.addView\(codec\)""").containsMatchIn(planted))
    }
}
