// Text another app hands the demo: shared to it (ACTION_SEND, text/plain), picked from the text-selection menu's
// "Read aloud" (ACTION_PROCESS_TEXT), or a .txt file opened with it (ACTION_VIEW, or a shared file). Plain Kotlin, apart
// from Android, so the JVM tests read every case; MainActivity hands it the intent's fields and shows what comes back.
//
// What comes in only fills the text card. It never starts a read: the user presses Read aloud. Another app chooses
// what it sends, so a read it could start would be that app speaking through this one.
package ggtts.demo

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction

/** What an intent brought. */
sealed class Incoming {
    /** Not a hand-over: a launch, or the debug build's adb extras. */
    object None : Incoming()

    /** Text to put in the card, from [source] ("shared", "selected"). */
    data class Text(val text: String, val source: String) : Incoming()

    /** A text file to read, by its content URI. */
    data class File(val uri: String) : Incoming()

    /** A hand-over the app will not take, and why, to show the user. */
    data class Refused(val why: String) : Incoming()
}

object IncomingText {
    const val SEND = "android.intent.action.SEND"
    const val VIEW = "android.intent.action.VIEW"
    const val PROCESS_TEXT = "android.intent.action.PROCESS_TEXT"
    const val TEXT_PLAIN = "text/plain"

    /**
     * The most characters the card takes from another app: about an hour and a half of speech, many hours to render.
     * Counted as a reader counts them ([characters]): an emoji or a rare CJK character is one, not two.
     */
    const val MAX_CHARS = 20_000

    /** The largest .txt file read: enough for [MAX_CHARS] characters of CJK text (3 bytes each in UTF-8) and more. */
    const val MAX_FILE_BYTES = 256 * 1024

    /**
     * What an intent brought. [action] and [type] are the intent's; [sendText] is EXTRA_TEXT, [processText]
     * EXTRA_PROCESS_TEXT, [stream] EXTRA_STREAM as a string, and [data] the intent's data URI as a string.
     */
    fun parse(
        action: String?,
        type: String?,
        sendText: CharSequence? = null,
        processText: CharSequence? = null,
        stream: String? = null,
        data: String? = null,
    ): Incoming = when (action) {
        PROCESS_TEXT -> text(processText, "selected")
        SEND -> when {
            !isPlainText(type) -> Incoming.Refused("Only plain text can be shared to this app, not $type.")
            sendText != null -> text(sendText, "shared")
            stream != null -> file(stream)
            else -> Incoming.Refused("The shared item had no text in it.")
        }
        VIEW -> when {
            data == null -> Incoming.None
            !isPlainText(type) -> Incoming.Refused("Only a plain text (.txt) file can be opened in this app, not $type.")
            else -> file(data)
        }
        else -> Incoming.None
    }

    /**
     * The text of a .txt file read from [input]: UTF-8 (a byte order mark is dropped), at most [MAX_FILE_BYTES] bytes
     * and [MAX_CHARS] characters. A file over either limit, or one that is not UTF-8, is refused, not cut short.
     */
    fun readFile(input: InputStream): Incoming {
        val bytes = input.use { s ->
            val buf = ByteArray(MAX_FILE_BYTES + 1)
            var n = 0
            while (n < buf.size) {
                val r = s.read(buf, n, buf.size - n)
                if (r < 0) break
                n += r
            }
            if (n > MAX_FILE_BYTES) {
                return Incoming.Refused(
                    "That file is over ${MAX_FILE_BYTES / 1024} KB. This app opens text files up to " +
                        "${MAX_FILE_BYTES / 1024} KB and $MAX_CHARS characters.",
                )
            }
            buf.copyOf(n)
        }
        val decoded = try {
            Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (e: CharacterCodingException) {
            return Incoming.Refused("That file is not UTF-8 text, so it was not opened.")
        }
        return text(decoded.removePrefix("﻿"), "opened")
    }

    private fun text(raw: CharSequence?, source: String): Incoming {
        val s = clean(raw?.toString() ?: "")
        val n = characters(s)
        return when {
            s.isEmpty() -> Incoming.Refused("There was no text to read in what was ${source}.")
            n > MAX_CHARS -> Incoming.Refused(
                "That text is $n characters. This app takes up to $MAX_CHARS at a time; " +
                    "share a shorter part.",
            )
            else -> Incoming.Text(s, source)
        }
    }

    /** Only a content URI: a file:// URI is another app's path, which this app has no permission to read. */
    private fun file(uri: String): Incoming =
        if (uri.startsWith("content://")) Incoming.File(uri)
        else Incoming.Refused("This app opens text files shared through Android, not a path (${uri.substringBefore(':')}).")

    private fun isPlainText(type: String?): Boolean = type?.substringBefore(';')?.trim()?.lowercase() == TEXT_PLAIN

    /** Characters as a reader counts them: Unicode code points, so a character outside the BMP is one, not two. */
    fun characters(s: String): Int = s.codePointCount(0, s.length)

    /**
     * What the screen says when text arrives: where it came from, how long it is, whether it replaced what the card
     * held (never silently), and that Read aloud starts it.
     */
    fun status(source: String, characters: Int, replaced: Boolean): String =
        "Text $source from another app: $characters characters" +
            (if (replaced) ", in place of what the card held" else "") + ". Press Read aloud to hear it."

    /** Line ends as \n, no NUL or other control characters but tab and newline, no edge whitespace. */
    fun clean(s: String): String =
        s.replace("\r\n", "\n").replace('\r', '\n')
            .filter { it == '\n' || it == '\t' || !it.isISOControl() }
            .trim()
}

/**
 * What a screen being created takes from its intent, and what it keeps across a re-creation.
 */
object Launch {
    /** `Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY`: the task was started again from Recents, with its old intent. */
    const val FLAG_LAUNCHED_FROM_HISTORY = 0x00100000

    /**
     * The most of the card's text the saved state carries itself: 100,000 UTF-16 units, about 200 KB in the Parcel.
     * The saved state crosses the binder, whose limit is about 1 MB per transaction, and a Bundle over it crashes the
     * app when the screen stops (TransactionTooLargeException). Pasted or typed text has no other limit, so a longer
     * card is written to a file in the app's cache instead ([MainActivity]'s onSaveInstanceState).
     */
    const val MAX_SAVED_CHARS = 100_000

    /**
     * Whether a screen being created takes its intent. Only a fresh start does. A re-created screen (dark mode or the
     * language changed, or the process restored) still holds the original intent; so does a task started again from
     * Recents, which has no saved state but carries [FLAG_LAUNCHED_FROM_HISTORY]. Taking it again would put an old
     * share back over the user's edits, or press Speak again for the debug build's adb extras. [savedState] is
     * onCreate's savedInstanceState (null on a fresh start) and [flags] the intent's flags.
     */
    fun takesIntent(savedState: Any?, flags: Int): Boolean =
        savedState == null && (flags and FLAG_LAUNCHED_FROM_HISTORY) == 0

    /** Whether [text] fits in the saved state itself ([MAX_SAVED_CHARS]); longer text goes to a file. */
    fun fitsSavedState(text: String): Boolean = text.length <= MAX_SAVED_CHARS
}
