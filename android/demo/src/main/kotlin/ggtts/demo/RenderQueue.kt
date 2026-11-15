// Reads queued to render for later, apart from Android: what is waiting, what is rendering, what is done and what
// failed, kept in one small file so the queue outlives the screen and the process. RenderService works it on a phone;
// the JVM tests drive it here.
//
// An entry is taken (WAITING to RENDERING) before its render starts and settled (DONE, or FAILED with the reason)
// after. A process killed mid-render leaves its entry RENDERING; [RenderQueue.recover], run when the queue is opened,
// puts it back to WAITING, so it renders again from the start: never lost, and never saved twice, since a render
// that did not finish saved nothing (its file is written pending and removed unless it completes).
package ggtts.demo

/** Where the queue is kept: a file in app storage on a phone, a string in a test. */
interface QueueStore {
    fun read(): String?
    fun write(text: String)
}

class RenderQueue(private val store: QueueStore) {
    enum class State { WAITING, RENDERING, DONE, FAILED }

    /** One queued read: the voice and options as the screen's request document, the text, and a title to save it by. */
    data class Entry(
        val id: Long,
        val request: LoadRequest,
        val text: String,
        val title: String,
        val state: State = State.WAITING,
        /** Why it failed, or where it was saved when done. */
        val note: String = "",
    )

    private val entries = mutableListOf<Entry>()

    init {
        entries += parse(store.read())
        recover()
    }

    /** Every entry, in queue order. */
    val all: List<Entry> get() = synchronized(this) { entries.toList() }

    /** How many wait to render, the one rendering included. */
    val pending: Int get() = synchronized(this) { entries.count { it.state == State.WAITING || it.state == State.RENDERING } }

    /** Queue [text] to render with [request]; returns the entry. */
    fun add(request: LoadRequest, text: String): Entry = synchronized(this) {
        val e = Entry((entries.maxOfOrNull { it.id } ?: 0) + 1, request, text, titleOf(text))
        entries += e
        save()
        e
    }

    /** The next entry to render, marked RENDERING, or null if none waits. One renders at a time. */
    fun take(): Entry? = synchronized(this) {
        if (entries.any { it.state == State.RENDERING }) return null
        val i = entries.indexOfFirst { it.state == State.WAITING }
        if (i < 0) return null
        entries[i] = entries[i].copy(state = State.RENDERING)
        save()
        entries[i]
    }

    /** [id] rendered and saved, at [where]. */
    fun done(id: Long, where: String) = settle(id, State.DONE, where)

    /** [id] could not render, for [reason]; the queue moves on to the next. */
    fun failed(id: Long, reason: String) = settle(id, State.FAILED, reason)

    /** [id] stopped before it finished (a Stop, low memory, the service's time running out): it waits again. */
    fun requeue(id: Long) = settle(id, State.WAITING, "")

    /** Drop every entry that waits (the notification's Clear queue); done and failed entries stay. Returns how many. */
    fun clearWaiting(): Int = synchronized(this) {
        val n = entries.count { it.state == State.WAITING }
        entries.removeAll { it.state == State.WAITING }
        if (n > 0) save()
        n
    }

    /**
     * An entry left RENDERING by a process that died mid-render waits again, to render from the start. Run when the
     * queue is opened.
     */
    fun recover() = synchronized(this) {
        var changed = false
        for (i in entries.indices) {
            if (entries[i].state == State.RENDERING) {
                entries[i] = entries[i].copy(state = State.WAITING)
                changed = true
            }
        }
        if (changed) save()
    }

    private fun settle(id: Long, state: State, note: String) = synchronized(this) {
        val i = entries.indexOfFirst { it.id == id }
        if (i >= 0) {
            entries[i] = entries[i].copy(state = state, note = note)
            save()
        }
    }

    private fun save() = store.write(entries.joinToString("\n") { line(it) })

    companion object {
        /** A saved read's title: the passage's first words, at most 40 characters, on one line. */
        fun titleOf(text: String): String {
            val one = text.trim().replace(Regex("\\s+"), " ")
            return if (one.length <= 40) one else one.take(39).trimEnd() + "…"
        }

        // Each field as hex of its UTF-8, so a tab or newline in a passage never breaks a line (java.util.Base64 is API 26,
        // and this app supports 24).
        private fun b64(s: String) = s.toByteArray(Charsets.UTF_8).joinToString("") { "%02x".format(it) }
        private fun un64(s: String) = String(ByteArray(s.length / 2) { s.substring(2 * it, 2 * it + 2).toInt(16).toByte() }, Charsets.UTF_8)

        private fun line(e: Entry) =
            listOf(e.id.toString(), e.state.name, b64(e.request.options.toJson()), b64(e.text), b64(e.title), b64(e.note))
                .joinToString("\t")

        /** The stored queue; a line that cannot be read is skipped rather than losing the rest. */
        fun parse(text: String?): List<Entry> = text.orEmpty().lines().filter { it.isNotBlank() }.mapNotNull { l ->
            runCatching {
                val f = l.split("\t")
                Entry(f[0].toLong(), LoadRequest(Options.parse(un64(f[2]))), un64(f[3]), un64(f[4]), State.valueOf(f[1]), un64(f[5]))
            }.getOrNull()
        }
    }
}
