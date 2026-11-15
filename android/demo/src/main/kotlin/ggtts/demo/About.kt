// What the app is for, said at the top of its screen before anything technical. A store listing leads with the same
// words (android/README.md, "What the app is for"). Kept here, not in the layout code, so a test can hold the claims to
// what the app actually does.
package ggtts.demo

object About {
    /** One line, the length of a store's short description (80 characters at most). English; [tagline] translates. */
    const val TAGLINE = "Reads text aloud in Cantonese or English, entirely on this phone."

    /** The reading screen's header subtitle: the store line, so the screen opens saying what the app is for. */
    const val HEADER_SUBTITLE = TAGLINE

    /**
     * Under the text box while it is empty: the use a first-time user would not guess. The wait is on the player
     * ("First sound in about…", from [EngineCosts]), and the full sentences stay under About ([PURPOSE]).
     */
    const val EMPTY_TEXT_HINT =
        "Other apps can read aloud with it too: choose ggtts as the text-to-speech engine in Android's settings."

    /**
     * What a person gets, what it costs them, and what it is not for. Every claim here is checked: "no internet
     * permission" by [AboutTest] against the manifest, and the wait by [EngineCosts], whose figures are measured: each
     * voice's range is read from there, so it says what the player's "First sound in about…" says for that voice.
     */
    val PURPOSE: List<String> get() = purpose(EnglishWords)

    /** [TAGLINE] in [w]'s language. */
    fun tagline(w: Words): String = w.get("tagline")

    /** [EMPTY_TEXT_HINT] in [w]'s language. */
    fun emptyTextHint(w: Words): String = w.get("empty_text_hint")

    /** [PURPOSE] in [w]'s language, with each voice's wait from [EngineCosts]. */
    fun purpose(w: Words): List<String> = listOf(
        w.get("purpose_1"),
        w.get("purpose_2", EngineCosts.firstAudio("qwen3-tts", w), EngineCosts.firstAudio("kitten-tts-2", w)),
        w.get("purpose_3"),
    )

    /**
     * The steps still ahead of a new user, given what the models directory holds: the first thing a store reviewer
     * meets on a fresh install is an app with nothing to speak with. Empty when model files are present.
     */
    fun gettingStarted(modelFiles: Int, w: Words = EnglishWords): List<String> =
        if (modelFiles > 0) emptyList()
        else listOf(w.get("start_1"), w.get("start_2"), w.get("start_3"))
}
