package dev.ggtts.ggtts_app

/** Focus callbacks and route callbacks must report every loss for this session. */
internal interface PlaybackFocus {
    fun acquire(onLoss: () -> Unit): Boolean
    fun release()
}

/** Serialized lifecycle shared by synchronous JNI entry points and interrupt callbacks. */
internal class PlaybackCoordinator(
    private val focus: PlaybackFocus,
    private val factory: (Int, () -> Unit) -> ControlledPcmSink,
) {
    private var bridge: ControlledPcmSession? = null
    private var next = 1L
    private var id = 0L
    private var inner = 0L
    private var state = 0
    private var failed = false

    @Synchronized fun reserve(): Long {
        if (id != 0L || next == 0L) return 0
        id = next
        next = if (next == Long.MAX_VALUE) 0 else next + 1
        state = 0
        failed = false
        return id
    }

    @Synchronized fun init(session: Long, rate: Int): Int {
        if (session != id || id == 0L || state != 0) return -1
        if (!focus.acquire { invalidate(session) }) return -1
        return try {
            val created = ControlledPcmSession { r -> factory(r) { invalidate(session) } }
            inner = created.open(rate)
            bridge = created
            state = 1
            if (failed) { invalidate(session); -1 } else 0
        } catch (_: Throwable) {
            failed = true
            focus.release()
            -1
        }
    }

    @Synchronized fun push(session: Long, samples: FloatArray): Int {
        if (!valid(session)) return -3
        if (state == 2) return -2
        return try { bridge!!.write(inner, samples) } catch (_: Throwable) { failed = true; -3 }
    }

    @Synchronized fun submitted(session: Long): Long = snapshot(session)?.submitted ?: -1
    @Synchronized fun head(session: Long): Long = snapshot(session)?.head ?: -1
    private fun snapshot(session: Long): ControlledPcmSession.Snapshot? {
        if (!valid(session)) return null
        return try { bridge!!.snapshot(inner) } catch (_: Throwable) { failed = true; null }
    }

    @Synchronized fun status(session: Long): Int =
        if (session != id || id == 0L) -1 else if (failed) -2 else state

    @Synchronized fun pause(session: Long): Int {
        if (!valid(session)) return -1
        return try { bridge!!.pause(inner); state = 2; 0 } catch (_: Throwable) { failed = true; -1 }
    }

    @Synchronized fun resume(session: Long): Int {
        if (!valid(session)) return -1
        return try { bridge!!.resume(inner); state = 1; 0 } catch (_: Throwable) { failed = true; -1 }
    }

    @Synchronized fun stop(session: Long): Int {
        if (session != id || id == 0L) return -1
        if (state == 3) return if (failed) -1 else 0
        val result = try { if (bridge != null) bridge!!.stop(inner); 0 } catch (_: Throwable) { -1 }
        bridge = null
        state = 3
        focus.release()
        return if (failed) -1 else result
    }

    @Synchronized fun release(session: Long): Int {
        if (session != id || id == 0L) return -1
        val wasFailed = failed
        val result = if (state != 3) stop(session) else 0
        id = 0
        return if (wasFailed) -1 else result
    }

    @Synchronized fun interruptActive() { if (id != 0L) invalidate(id) }

    private fun valid(session: Long) = session != 0L && session == id && !failed && (state == 1 || state == 2)

    @Synchronized private fun invalidate(session: Long) {
        if (session != id || id == 0L) return
        failed = true
        try { if (bridge != null) bridge!!.stop(inner) } catch (_: Throwable) {}
        bridge = null
        focus.release()
    }
}
