package dev.ggtts.ggtts_app

/** One fresh mono PCM output; all methods are serialized by [ControlledPcmSession]. */
internal interface ControlledPcmSink {
    fun play()
    fun pause()
    fun write(samples: FloatArray, offset: Int, count: Int): Int
    /** Unsigned head for this track. A changed or unknown output route must throw. */
    fun head(): Long
    fun flush()
    fun release()
}

/**
 * A single-owner output primitive for a future Flutter transport.
 *
 * This class is NOT wired to the rendering path and never claims completion.
 * IDs are not reused, writes are nonblocking and return accepted mono frames,
 * and a snapshot is valid only while this exact sink and route remain active.
 * Callers must serialize snapshot accounting with writes and apply their own
 * deadline. A failed or replaced route invalidates the session.
 */
internal class ControlledPcmSession(private val factory: (Int) -> ControlledPcmSink) {
    internal data class Snapshot(val submitted: Long, val head: Long, val paused: Boolean)
    private var next = 1L
    private var id = 0L
    private var sink: ControlledPcmSink? = null
    private var submitted = 0L
    private var head = 0L
    private var paused = false

    @Synchronized fun open(rate: Int): Long {
        check(id == 0L && next > 0 && rate in 1..192000)
        val created = factory(rate)
        try {
            check(created.head() == 0L)
            created.play()
        } catch (e: Throwable) {
            runCatching { created.release() }
            throw e
        }
        id = next
        next = if (next == Long.MAX_VALUE) 0 else next + 1
        sink = created
        submitted = 0
        head = 0
        paused = false
        return id
    }

    @Synchronized fun write(session: Long, samples: FloatArray): Int {
        val target = requireSession(session)
        check(samples.size <= 2048)
        if (paused) return 0
        return try {
            val count = target.write(samples, 0, samples.size)
            check(count in 0..samples.size && submitted + count <= 0xffffffffL)
            submitted += count
            count
        } catch (e: Throwable) { invalidate(); throw e }
    }

    @Synchronized fun snapshot(session: Long): Snapshot {
        val target = requireSession(session)
        return try {
            val now = target.head()
            check(now in head..submitted)
            head = now
            Snapshot(submitted, head, paused)
        } catch (e: Throwable) { invalidate(); throw e }
    }

    @Synchronized fun pause(session: Long) {
        val target = requireSession(session)
        try { if (!paused) { target.pause(); paused = true } }
        catch (e: Throwable) { invalidate(); throw e }
    }

    @Synchronized fun resume(session: Long) {
        val target = requireSession(session)
        try { if (paused) { target.play(); paused = false } }
        catch (e: Throwable) { invalidate(); throw e }
    }

    /** A stop discards remaining audio; it never confirms presentation. */
    @Synchronized fun stop(session: Long) {
        requireSession(session)
        val old = sink!!
        sink = null
        id = 0
        try { old.pause(); old.flush() } finally { old.release() }
    }

    private fun requireSession(session: Long): ControlledPcmSink {
        check(session != 0L && session == id)
        return checkNotNull(sink)
    }

    private fun invalidate() {
        val old = sink
        sink = null
        id = 0
        runCatching { old?.pause() }
        runCatching { old?.flush() }
        runCatching { old?.release() }
    }
}
