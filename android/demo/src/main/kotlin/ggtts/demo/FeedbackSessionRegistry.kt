package ggtts.demo

/** One editable page per exact input within this process. */
internal object FeedbackSessionRegistry {
    private val active = mutableSetOf<String>()

    fun acquire(digest: String): Lease? = synchronized(active) {
        if (!active.add(digest)) null else Lease(digest)
    }

    class Lease internal constructor(private val digest: String) : AutoCloseable {
        private var closed = false
        private var running = 0

        fun <T> runIfActive(block: () -> T): T? {
            synchronized(active) {
                if (closed) return null
                running++
            }
            return try { block() } finally {
                synchronized(active) {
                    running--
                    if (closed && running == 0) active.remove(digest)
                }
            }
        }

        override fun close() = synchronized(active) {
            if (!closed) {
                closed = true
                if (running == 0) active.remove(digest)
            }
        }
    }
}
