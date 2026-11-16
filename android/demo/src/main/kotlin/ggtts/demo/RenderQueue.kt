// Reads queued to render for later, apart from Android: what is waiting, what is rendering, what is done and what
// failed, kept in one small file so the queue outlives the screen and the process. RenderService works it on a phone;
// the JVM tests drive it here.
//
// An entry is taken (WAITING to RENDERING) before its render starts and settled (DONE, or FAILED with the reason)
// after. A process killed mid-render returns it to WAITING. Before MediaStore insertion, a durable PUBLISHING record
// stores the unique row identity and expected bytes; recovery verifies that exact row before deciding DONE, retry, or
// hold for an unknown provider result.
package ggtts.demo

/** Where the queue is kept: a file in app storage on a phone, a string in a test. */
interface QueueStore {
    fun read(): String?
    fun write(text: String)
}

/** Durable identity for the one MediaStore row a queue entry is publishing. */
data class Publication(val name: String, val token: String, val sha256: String, val byteCount: Long) {
    init {
        require(token.matches(Regex("[0-9a-f]{32}")) && name.endsWith("-q$token.flac"))
        require(sha256.matches(Regex("[0-9a-f]{64}")) && byteCount > 0)
    }
}

/** What recovery could establish about a journaled publication. Unknown is deliberately not Missing. */
sealed class PublicationRecovery {
    data class Published(val uri: String) : PublicationRecovery()
    data class Pending(val uri: String) : PublicationRecovery()
    object Missing : PublicationRecovery()
    data class Unknown(val reason: String) : PublicationRecovery()
}

/** A sink that can reconcile a durable publication intent after process death. */
interface JournaledReadSink : ReadSink {
    fun saveJournaled(name: String, title: String, flac: ByteArray, publication: Publication,
                      publish: (() -> Unit) -> Boolean): String
    fun locate(publication: Publication): PublicationRecovery
    /** Verify and publish a complete pending row, or remove only that exact owned incomplete row. */
    fun recoverPending(publication: Publication, uri: String): PublicationRecovery
}

class RenderQueue(private val store: QueueStore) {
    enum class State { WAITING, RENDERING, PUBLISHING, DONE, FAILED }

    /** One queued read: the voice and options as the screen's request document, the text, and a title to save it by. */
    data class Entry(
        val id: Long,
        val request: LoadRequest,
        val text: String,
        val title: String,
        val state: State = State.WAITING,
        /** Why it failed, or where it was saved when done. */
        val note: String = "",
        /** Present before MediaStore insertion and retained until recovery settles publication. */
        val publication: Publication? = null,
    )

    private val entries = mutableListOf<Entry>()

    init {
        entries += parse(store.read())
        recover()
    }

    /** Every entry, in queue order. */
    val all: List<Entry> get() = synchronized(this) { entries.toList() }

    /** How many wait to render, the one rendering included. */
    val pending: Int get() = synchronized(this) { entries.count { it.state == State.WAITING || it.state == State.RENDERING || it.state == State.PUBLISHING } }

    /** Queue [text] to render with [request]; returns the entry. */
    fun add(request: LoadRequest, text: String): Entry = synchronized(this) {
        val e = Entry((entries.maxOfOrNull { it.id } ?: 0) + 1, request, text, titleOf(text))
        entries += e
        try { save() } catch (t: Throwable) { entries.remove(e); throw t }
        e
    }

    /** The next entry to render, marked RENDERING, or null if none waits. One renders at a time. */
    fun take(): Entry? = synchronized(this) {
        if (entries.any { it.state == State.RENDERING || it.state == State.PUBLISHING }) return null
        val i = entries.indexOfFirst { it.state == State.WAITING }
        if (i < 0) return null
        val old = entries[i]
        entries[i] = old.copy(state = State.RENDERING)
        try { save() } catch (t: Throwable) { entries[i] = old; throw t }
        entries[i]
    }

    /** [id] rendered and saved, at [where]. */
    fun done(id: Long, where: String) = settle(id, State.DONE, where)

    /** Persist the exact identity before the sink can create a MediaStore row. */
    fun preparePublication(id: Long, publication: Publication) = synchronized(this) {
        val i = entries.indexOfFirst { it.id == id }
        check(i >= 0 && entries[i].state == State.RENDERING && entries[i].publication == null) { "entry is not rendering" }
        val old = entries[i]
        entries[i] = old.copy(state = State.PUBLISHING, publication = publication, note = "")
        try { save() } catch (t: Throwable) { entries[i] = old; throw t }
    }

    /** Entries whose durable publication intent survived a process restart. */
    fun publications(): List<Entry> = synchronized(this) { entries.filter { it.state == State.PUBLISHING } }

    /** Resolve a journal after the sink has established the row state. Unknown/Pending stay blocked. */
    fun reconcilePublication(id: Long, result: PublicationRecovery) = synchronized(this) {
        val i = entries.indexOfFirst { it.id == id }
        if (i < 0 || entries[i].state != State.PUBLISHING) return@synchronized
        val old = entries[i]
        when (result) {
            is PublicationRecovery.Published -> entries[i] = old.copy(state = State.DONE, note = result.uri, publication = null)
            PublicationRecovery.Missing -> entries[i] = old.copy(state = State.WAITING, note = "", publication = null)
            is PublicationRecovery.Pending, is PublicationRecovery.Unknown -> return@synchronized
        }
        try { save() } catch (t: Throwable) { entries[i] = old; throw t }
    }

    /** [id] could not render, for [reason]; the queue moves on to the next. */
    fun failed(id: Long, reason: String) = settle(id, State.FAILED, reason)

    /** [id] stopped before it finished (a Stop, low memory, the service's time running out): it waits again. */
    fun requeue(id: Long) = settle(id, State.WAITING, "")

    /** Drop every entry that waits (the notification's Clear queue); done and failed entries stay. Returns how many. */
    fun clearWaiting(): Int = synchronized(this) {
        val n = entries.count { it.state == State.WAITING }
        val old = entries.toList()
        entries.removeAll { it.state == State.WAITING }
        if (n > 0) try { save() } catch (t: Throwable) { entries.clear(); entries.addAll(old); throw t }
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
                if (entries[i].publication == null) entries[i] = entries[i].copy(state = State.WAITING)
                else continue // Publication may already be visible; reconcile its exact identity before retrying.
                changed = true
            }
        }
        if (changed) {
            // If writing fails, the old RENDERING state is safe too: next launch repeats this recovery.
            runCatching { save() }
        }
    }

    private fun settle(id: Long, state: State, note: String) = synchronized(this) {
        val i = entries.indexOfFirst { it.id == id }
        if (i >= 0) {
            val old = entries[i]
            entries[i] = old.copy(state = state, note = note,
                publication = if (state == State.WAITING || state == State.FAILED || state == State.DONE) null else old.publication)
            try { save() } catch (t: Throwable) { entries[i] = old; throw t }
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

        private fun line(e: Entry): String {
            val p = e.publication
            return (listOf(e.id.toString(), e.state.name, b64(e.request.options.toJson()), b64(e.text), b64(e.title), b64(e.note)) +
                if (p == null) listOf("") else listOf(b64(p.name), b64(p.token), p.sha256, p.byteCount.toString())).joinToString("\t")
        }

        /** The stored queue; a line that cannot be read is skipped rather than losing the rest. */
        fun parse(text: String?): List<Entry> = text.orEmpty().lines().filter { it.isNotBlank() }.mapNotNull { l ->
            try {
                val f = l.split("\t")
                val state = State.valueOf(f[1])
                val publication = if (f.size >= 10 && f[6].isNotEmpty()) runCatching {
                    Publication(un64(f[6]), un64(f[7]), f[8], f[9].toLong())
                }.getOrNull() else null
                Entry(f[0].toLong(), LoadRequest(Options.parse(un64(f[2]))), un64(f[3]), un64(f[4]), state, un64(f[5]), publication)
            } catch (t: Throwable) {
                if (l.split("\t").getOrNull(1) == State.PUBLISHING.name) {
                    throw IllegalStateException("cannot read a publication journal; refusing to lose a possibly published read", t)
                }
                null
            }
        }
    }
}
