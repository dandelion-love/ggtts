// What the app is for, said at the top of its screen before anything technical. A store listing leads with the same
// words (android/README.md, "What the app is for"). Kept here, not in the layout code, so a test can hold the claims to
// what the app actually does.
package ggtts.demo

object About {
    /** One line, the length of a store's short description (80 characters at most). */
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
    val PURPOSE = listOf(
        "Type or paste a passage and listen to it read in a natural voice. Nothing you type leaves the phone: the app has no permission to use the internet.",
        "It is slow. The first sound takes ${EngineCosts.BY_ENGINE.getValue("qwen3-tts").firstAudioS} seconds with a Cantonese voice and ${EngineCosts.BY_ENGINE.getValue("kitten-tts-2").firstAudioS} with an English one, so it suits listening to a passage you have chosen, not reading the screen as you go.",
        "Other apps can use it too: it is a text-to-speech engine in Android's settings.",
    )

    /**
     * The steps still ahead of a new user, given what the models directory holds: the first thing a store reviewer
     * meets on a fresh install is an app with nothing to speak with. Empty when model files are present.
     */
    fun gettingStarted(modelFiles: Int): List<String> =
        if (modelFiles > 0) emptyList()
        else listOf(
            "1. Get a voice's model files on this phone (about 2.5 GB; android/README.md in the ggtts repository says where).",
            "2. Press \"Import a model file\" and pick each one.",
            "3. Choose the files below and press Speak.",
        )
}
