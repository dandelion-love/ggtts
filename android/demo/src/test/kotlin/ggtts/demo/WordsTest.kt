// The demo's words: the English resources and EnglishWords say the same, every key has a Traditional Chinese (Hong
// Kong) translation with the same format arguments, and the screen's code passes no English literal to a label.
package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory

class WordsTest {
    /** A strings.xml as name to its text, with Android's escapes (\' \" \n \\) undone. */
    private fun strings(path: String): Map<String, String> {
        val f = File(path)
        assertTrue("$path not found from ${File(".").absolutePath}", f.isFile)
        val doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(f)
        val nodes = doc.getElementsByTagName("string")
        return (0 until nodes.length).associate { i ->
            val e = nodes.item(i)
            e.attributes.getNamedItem("name").nodeValue to unescape(e.textContent)
        }
    }

    private fun unescape(s: String): String {
        val b = StringBuilder()
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (c == '\\' && i + 1 < s.length) {
                b.append(when (val n = s[i + 1]) { 'n' -> '\n'; 't' -> '\t'; else -> n })
                i += 2
            } else {
                b.append(c)
                i++
            }
        }
        return b.toString()
    }

    private val english by lazy { strings("src/main/res/values/strings.xml") }
    private val hongKong by lazy { strings("src/main/res/values-zh-rHK/strings.xml") }

    /** The format arguments a string takes, as `%1$s`, `%2$.1f`: a translation must take the same. */
    private fun arguments(s: String): Set<String> = Regex("""%\d+\$[-#+ 0,(]*\d*(\.\d+)?[a-zA-Z]""").findAll(s).map { it.value }.toSet()

    /** What [translation] lacks or gets wrong against [base]: missing keys, and keys whose format arguments differ. */
    private fun problems(base: Map<String, String>, translation: Map<String, String>): List<String> =
        base.mapNotNull { (k, v) ->
            when (val t = translation[k]) {
                null -> "$k has no translation"
                else -> if (arguments(t) != arguments(v)) "$k takes ${arguments(t)}, not ${arguments(v)}" else null
            }
        } + (translation.keys - base.keys).map { "$it is not an English key" }

    /** #411 AC1: EnglishWords, which the plain-Kotlin classes use, is exactly the English resources. */
    @Test
    fun englishWordsIsTheEnglishResources() {
        assertEquals(english, EnglishWords.TEXT)
    }

    /** #411 AC2: every English key has a zh-HK entry, with the same format arguments, and no stray keys. */
    @Test
    fun everyKeyIsTranslatedIntoChineseHongKong() {
        val found = problems(english, hongKong)
        assertTrue(found.joinToString("\n"), found.isEmpty())
        assertTrue("the translation is not the English", hongKong.getValue("app_title") != english.getValue("app_title"))
    }

    /** #411 AC2 and AC4: the check sees a missing translation and a changed format argument (planted). */
    @Test
    fun aMissingOrMismatchedTranslationIsCaught() {
        val missing = problems(english, hongKong - "app_title")
        assertEquals(listOf("app_title has no translation"), missing)
        val mismatched = problems(english, hongKong + ("characters" to "%1\$s 個字"))
        assertEquals(1, mismatched.size)
        assertTrue(mismatched.single(), mismatched.single().startsWith("characters takes"))
    }

    /**
     * The quoted literals of [src] that hold a letter once their `$name` / `${...}` templates are taken out, on lines
     * that are not comments, logs or annotations, and that are not in [ALLOWED]: each would be words on the screen in
     * one language. Every such literal is reported, whatever it is passed to (a label, a conditional, a dialog).
     */
    private fun screenLiterals(src: String): List<String> {
        val literal = Regex(""""((?:[^"\\\n]|\\.)*)"""")
        val template = Regex("""\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*""")
        return src.lines().withIndex().flatMap { (i, line) ->
            val t = line.trimStart()
            if (t.startsWith("//") || t.startsWith("*") || t.startsWith("/*") || "Log." in line || t.startsWith("@")) {
                emptyList()
            } else {
                literal.findAll(line).map { it.groupValues[1] }
                    .filter { it !in ALLOWED }
                    .filter { Regex("[A-Za-z]").containsMatchIn(unescape(template.replace(it, ""))) }
                    .map { "line ${i + 1}: \"$it\" in ${t.take(100)}" }.toList()
            }
        }
    }

    /** #411 AC1: MainActivity holds no words of its own: every literal with a letter is a key, a name or an id. */
    @Test
    fun theScreenHoldsNoWordsOfItsOwn() {
        val found = screenLiterals(File("src/main/kotlin/ggtts/demo/MainActivity.kt").readText())
        assertTrue(found.joinToString("\n"), found.isEmpty())
    }

    /** The check reports words in a conditional, a template and a dialog (planted), and not an allowed key. */
    @Test
    fun wordsInAnyFormAreCaught() {
        val planted = """
            t.text = if (x) "Stop now" else "Go on"
            v.showStatus("${'$'}n chunks done")
            AlertDialog.Builder(this).setPositiveButton("OK", null).setMessage("Delete this voice?")
            val n = "${'$'}{count}"
            extras.getString("model")
        """.trimIndent()
        val found = screenLiterals(planted)
        for (w in listOf("Stop now", "Go on", "\$n chunks done", "OK", "Delete this voice?")) {
            assertTrue("$w was not reported: $found", found.any { "\"$w\"" in it })
        }
        assertTrue("an allowed key or a bare template was reported: $found", found.size == 5)
    }

    /**
     * #411 AC1, beyond the screen: no source file passes a quoted English literal to a toast, a notification (title,
     * text, channel, action) or a chooser. Each would reach the user untranslated.
     */
    @Test
    fun noToastOrNotificationTakesAnEnglishLiteral() {
        val literal = Regex(
            """(Toast\.makeText\([^,]+,\s*"[A-Za-z])|""" +
                """((setContentTitle|setContentText|setTicker|createChooser|setSubText)\("[A-Za-z])|""" +
                """(NotificationChannel\([^,]+,\s*"[A-Za-z])|""" +
                """(Action\.Builder\([^,]+,\s*"[A-Za-z])""",
        )
        val found = File("src/main/kotlin/ggtts/demo").listFiles { f -> f.name.endsWith(".kt") }!!.flatMap { f ->
            f.readLines().withIndex()
                .filter { (_, line) -> !line.trimStart().startsWith("//") && literal.containsMatchIn(line) }
                .map { (i, line) -> "${f.name}:${i + 1}: ${line.trim()}" }
        }
        assertTrue(found.joinToString("\n"), found.isEmpty())
        // The check sees the literal it is for (planted).
        assertTrue(literal.containsMatchIn("""Notification.Action.Builder(null, "Stop", stop)"""))
        assertTrue(literal.containsMatchIn("""Toast.makeText(this, "Saved", Toast.LENGTH_SHORT)"""))
    }

    /** #411: Android 13 and later list the app's languages in its settings (localeConfig), English and zh-HK. */
    @Test
    fun theAppDeclaresItsLanguages() {
        val manifest = File("src/main/AndroidManifest.xml").readText()
        assertTrue("the manifest names no localeConfig", "android:localeConfig=\"@xml/locales_config\"" in manifest)
        val config = File("src/main/res/xml/locales_config.xml").readText()
        for (tag in listOf("en", "zh-HK")) assertTrue("$tag is not in locales_config.xml", "android:name=\"$tag\"" in config)
    }

    companion object {
        /**
         * Literals in MainActivity that are not words on the screen: intent-extra keys (the debug build's adb extras),
         * engine ids, directory and file names, WAV chunk tags, the language identifiers Voices uses, the language
         * badge, a TextToSpeech utterance id, and the English sample voice's spoken line (content to read, like the
         * Cantonese one). A new literal is either added here, with the reason being one of these, or moved into the
         * string resources.
         */
        val ALLOWED = setOf(
            "ggtts-demo", "qwen3-tts", "kitten-tts-2", "Hello from ggtts, running on this phone.",
            "Cantonese", "English", "EN", "models", "ggtts-check", "\\n", "unnamed", "read", "\$engine.wav",
            "RIFF", "WAVEfmt ", "data",
            "model", "codec", "voice", "engine", "speaker", "threads", "stop", "speak_system", "text", "speak",
        )
    }

    /** About's English constants, which its own tests check, are the English resources. */
    @Test
    fun aboutsEnglishIsTheResources() {
        assertEquals(english.getValue("tagline"), About.TAGLINE)
        assertEquals(english.getValue("empty_text_hint"), About.EMPTY_TEXT_HINT)
        assertEquals(EnglishWords.get("seconds_range", 30, 50), EngineCosts.BY_ENGINE.getValue("qwen3-tts").firstAudioS)
    }
}
