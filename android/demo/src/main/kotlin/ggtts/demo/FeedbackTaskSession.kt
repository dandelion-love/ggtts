package ggtts.demo

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean

/** All sheets share one worker; closing a session cancels its queued work and drops posted results. */
internal class FeedbackTaskSession(
    private val worker: ExecutorService = SHARED,
    private val post: (() -> Unit) -> Unit,
) : AutoCloseable {
    private val active = AtomicBoolean(true)
    private val lock = Any()
    private val futures = mutableListOf<Future<*>>()
    fun <T> submit(work: () -> T, deliver: (T) -> Unit) {
        synchronized(lock) {
            if (!active.get()) return
            futures.add(worker.submit {
                if (!active.get()) return@submit
                val result = work()
                if (active.get()) post { if (active.get()) deliver(result) }
            })
        }
    }
    override fun close() {
        synchronized(lock) {
            active.set(false)
            futures.forEach { it.cancel(true) }
            futures.clear()
        }
    }
    companion object {
        private val SHARED = Executors.newSingleThreadExecutor { task ->
            Thread(task, "ggtts-feedback").apply { isDaemon = true }
        }
    }
}

/** Prevent duplicate opens, and release the row on every terminal outcome. Main thread only. */
internal class FeedbackOpenGate(private val setEnabled: (Boolean) -> Unit) {
    private var opening = false
    fun begin(): Boolean {
        if (opening) return false
        opening = true
        setEnabled(false)
        return true
    }
    fun finish() {
        if (!opening) return
        opening = false
        setEnabled(true)
    }
}
