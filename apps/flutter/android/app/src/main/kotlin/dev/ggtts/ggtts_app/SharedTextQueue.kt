package dev.ggtts.ggtts_app

/** In-memory, bounded FIFO. A rejected share never replaces an earlier draft. */
internal class SharedTextQueue(
    private val maxChars: Int = 100_000,
    private val maxItems: Int = 8,
) {
    private val pending = ArrayDeque<String>()

    fun offer(action: String?, mime: String?, text: CharSequence?): Boolean {
        if (action != "android.intent.action.SEND" || mime != "text/plain" ||
            text == null || text.isEmpty() || text.length > maxChars || pending.size >= maxItems) return false
        val value = text.toString()
        var i = 0
        while (i < value.length) {
            val c = value[i]
            if (Character.isHighSurrogate(c)) {
                if (i + 1 >= value.length || !Character.isLowSurrogate(value[i + 1])) return false
                i += 2
            } else {
                if (Character.isLowSurrogate(c) || c == '\u0000') return false
                i++
            }
        }
        pending.addLast(value)
        return true
    }

    fun take(): String? = if (pending.isEmpty()) null else pending.removeFirst()
}
