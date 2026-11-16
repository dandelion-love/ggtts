package dev.ggtts.ggtts_app

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class PlaybackCoordinatorTest {
    private class Focus : PlaybackFocus {
        var callback: (() -> Unit)? = null
        var releases = 0
        override fun acquire(onLoss: () -> Unit): Boolean { callback = onLoss; return true }
        override fun release() { releases++ }
    }
    private class Sink : ControlledPcmSink {
        var released = 0
        var position = 0L
        override fun play() {}
        override fun pause() {}
        override fun write(samples: FloatArray, offset: Int, count: Int) = count
        override fun head() = position
        override fun flush() {}
        override fun release() { released++ }
    }
    @Test fun routeChangeInvalidatesAndStaleEventCannotAffectNextRead() {
        val focus = Focus()
        val sinks = mutableListOf<Sink>()
        val routes = mutableListOf<() -> Unit>()
        val c = PlaybackCoordinator(focus) { _, route ->
            routes.add(route); Sink().also(sinks::add)
        }
        val first = c.reserve()
        assertEquals(0, c.init(first, 24000))
        assertEquals(0, c.push(first, FloatArray(2)))
        routes[0]()
        assertEquals(2, c.push(first, FloatArray(2)))
        routes[0]()
        assertEquals(-2, c.status(first))
        assertEquals(-1, c.release(first))
        assertEquals(1, sinks[0].released)
        val second = c.reserve()
        assertEquals(0, c.init(second, 24000))
        routes[1]()
        routes[0]()
        assertEquals(1, c.status(second))
        assertEquals(0, c.release(second))
    }
    @Test fun focusLossAfterHeadBeforeCleanupNeverSucceeds() {
        val focus = Focus()
        val sink = Sink()
        lateinit var route: () -> Unit
        val c = PlaybackCoordinator(focus) { _, listener -> route = listener; sink }
        val id = c.reserve()
        assertEquals(0, c.init(id, 24000))
        route()
        assertEquals(2, c.push(id, FloatArray(2)))
        sink.position = 2
        assertEquals(2L, c.head(id))
        focus.callback!!()
        assertEquals(-1, c.release(id))
        assertEquals(1, sink.released)
    }
    @Test fun initialRouteNotificationEnablesWrites() {
        val focus = Focus()
        lateinit var route: () -> Unit
        val c = PlaybackCoordinator(focus) { _, listener -> route = listener; Sink() }
        val id = c.reserve()
        assertEquals(0, c.init(id, 24000))
        assertEquals(0, c.push(id, FloatArray(1)))
        route()
        assertEquals(1, c.push(id, FloatArray(1)))
        assertEquals(1, c.status(id))
        assertEquals(0, c.release(id))
    }
    @Test fun routeChangeDuringInitializationReleasesCreatedTrack() {
        val focus = Focus()
        val sink = Sink()
        val c = PlaybackCoordinator(focus) { _, route -> route(); route(); sink }
        val id = c.reserve()
        assertEquals(-1, c.init(id, 24000))
        assertEquals(-2, c.status(id))
        assertEquals(1, sink.released)
        assertEquals(-1, c.release(id))
    }

    @Test fun lifecycleInterruptionFailsBeforeCleanup() {
        val sink = Sink()
        val c = PlaybackCoordinator(Focus()) { _, _ -> sink }
        val id = c.reserve()
        assertEquals(0, c.init(id, 24000))
        c.interruptActive()
        assertEquals(-2, c.status(id))
        assertEquals(-1, c.release(id))
        assertEquals(1, sink.released)
    }

    @Test fun delayedFocusLossWinsBeforeReleaseAcrossThreads() {
        val focus = Focus()
        val sink = Sink()
        lateinit var route: () -> Unit
        val c = PlaybackCoordinator(focus) { _, listener -> route = listener; sink }
        val id = c.reserve()
        assertEquals(0, c.init(id, 24000))
        route()
        assertEquals(2, c.push(id, FloatArray(2)))
        sink.position = 2
        val observed = CountDownLatch(1)
        val invalidated = CountDownLatch(1)
        val result = AtomicInteger(0)
        val worker = Thread {
            assertEquals(2L, c.head(id))
            observed.countDown()
            check(invalidated.await(2, TimeUnit.SECONDS))
            result.set(c.release(id))
        }
        worker.start()
        assertTrue(observed.await(2, TimeUnit.SECONDS))
        focus.callback!!()
        invalidated.countDown()
        worker.join(2000)
        assertFalse(worker.isAlive)
        assertEquals(-1, result.get())
        assertEquals(1, sink.released)
    }

    @Test fun noInitialRouteKeepsWritesBoundedByCaller() {
        val c = PlaybackCoordinator(Focus()) { _, _ -> Sink() }
        val id = c.reserve()
        assertEquals(0, c.init(id, 24000))
        repeat(3) { assertEquals(0, c.push(id, FloatArray(1))) }
        assertEquals(0, c.release(id))
    }

}
