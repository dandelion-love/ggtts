// What the app is for, said at the top of its screen before anything technical. A store listing leads with the same
// words (android/README.md, "What the app is for"). Kept here, not in the layout code, so a test can hold the claims to
// what the app actually does.
package ggtts.demo

object About {
    /** One line, the length of a store's short description (80 characters at most). */
    const val TAGLINE = "Reads text aloud in Cantonese or English, entirely on this phone."

    /**
     * What a person gets, what it costs them, and what it is not for. Every claim here is checked: "no internet
     * permission" by [AboutTest] against the manifest, and the wait by [EngineCosts], whose figures are measured.
     */
    val PURPOSE = listOf(
        "Type or paste a passage and listen to it read in a natural voice. Nothing you type leaves the phone: the app has no permission to use the internet.",
        "It is slow. A sentence takes 30 to 60 seconds before it starts, so it suits listening to a passage you have chosen, not reading the screen as you go.",
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
