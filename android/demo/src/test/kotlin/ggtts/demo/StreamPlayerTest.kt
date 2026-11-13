// The player on the host JVM, with a fake device in place of AudioTrack. The fake refuses any call after it is released
// (as AudioTrack does, with an IllegalStateException) and records which thread made each call, so these tests see the
// races the old DeviceAudio had: a Stop releasing the track under a writing thread, and a Stop joining on the UI thread.
package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class StreamPlayerTest {
    private inner class FakeSink : PcmSink {
        val calls: MutableList<String> = Collections.synchronizedList(mutableListOf())
        val misuse: MutableList<String> = Collections.synchronizedList(mutableListOf())
        val paused = CountDownLatch(1)
        val released = CountDownLatch(1)
        @Volatile var isReleased = false
        /** Writes block until the sink is paused, as a full AudioTrack buffer does. */
        var blockWrites = false
        /** Runs inside release(), on the releasing thread. */
        var onRelease: () -> Unit = {}

        private fun call(name: String) {
            calls += "$name@${Thread.currentThread().name}"
            if (isReleased) misuse += "$name after release"
        }

        override fun play() = call("play")
        /** Runs inside write(), on the player thread, before the write takes anything. */
        var beforeWrite: () -> Unit = {}
        @Volatile var isPaused = false
        override fun write(samples: FloatArray, offset: Int, count: Int): Int {
            call("write")
            beforeWrite()
            // A paused device takes nothing, as a full AudioTrack buffer does; with blockWrites a running one is full too.
            return if (isPaused || blockWrites) 0 else count
        }
        override fun pause() { call("pause"); isPaused = true; paused.countDown() }
        override fun flush() = call("flush")
        override fun stop() = call("stop")
        override fun release() {
            call("release")
            onRelease()
            isReleased = true
            released.countDown()
        }
    }

    private val sinks = mutableListOf<FakeSink>()
    private val player = StreamPlayer({ FakeSink().also { sinks += it } }, drainMs = 0)

    private fun awaitIgnoringInterrupts(latch: CountDownLatch) {
        while (true) {
            try {
                latch.await(5, TimeUnit.SECONDS)
                return
            } catch (_: InterruptedException) {
            }
        }
    }

    /** Runs [block] on a thread standing in for the main thread; false if it did not return within [ms]. */
    private fun returnsWithin(ms: Long, block: () -> Unit): Boolean {
        val done = CountDownLatch(1)
        Thread({ block(); done.countDown() }, "main-stand-in").start()
        return done.await(ms, TimeUnit.MILLISECONDS)
    }

    @Test
    fun stopReturnsWithoutWaitingForThePlayerThread() {
        player.begin(24000)
        val sink = sinks.single()
        sink.blockWrites = true
        val gate = CountDownLatch(1)
        sink.onRelease = { awaitIgnoringInterrupts(gate) } // the player thread is held in release (stop interrupts it)
        player.play(FloatArray(10))
        assertTrue("stop returned while the player thread was still finishing", returnsWithin(1000) { player.stop() })
        assertTrue("the pause ended the blocking write", sink.paused.await(1, TimeUnit.SECONDS))
        gate.countDown()
        assertTrue(sink.released.await(2, TimeUnit.SECONDS))
    }

    @Test
    fun onlyThePlayerThreadWritesStopsAndReleasesTheSink() {
        player.begin(24000)
        val sink = sinks.single()
        sink.blockWrites = true
        player.play(FloatArray(10))
        player.stop()
        assertTrue(sink.released.await(2, TimeUnit.SECONDS))
        val owned = sink.calls.filter { it.substringBefore('@') in setOf("write", "stop", "release") }
        assertTrue(owned.toString(), owned.isNotEmpty() && owned.all { it.endsWith("@ggtts-player") })
        assertEquals(1, sink.calls.count { it.startsWith("release@") })
        assertEquals(emptyList<String>(), sink.misuse)
    }

    @Test
    fun aStopAsTheStreamEndsNeverTouchesAReleasedSink() {
        player.begin(24000)
        val sink = sinks.single()
        lateinit var stopper: Thread
        var stopWaited = false
        // The stream is ending on its own: a Stop arrives while the player thread is releasing the sink.
        // The release waits until that Stop is blocked on the stream's lock, so it is sure to arrive in this window.
        sink.onRelease = {
            stopper = Thread({ player.stop() }, "main-stand-in").apply { start() }
            val deadline = System.nanoTime() + 2_000_000_000L
            while (stopper.state != Thread.State.BLOCKED && System.nanoTime() < deadline) Thread.onSpinWait()
            stopWaited = stopper.state == Thread.State.BLOCKED
        }
        player.play(FloatArray(10))
        player.finish()
        stopper.join(2000)
        assertTrue("the Stop arrived while the sink was being released", stopWaited)
        assertEquals(emptyList<String>(), sink.misuse)
    }

    @Test
    fun aStopLandingJustBeforeAWriteStillEndsThePlayerThread() {
        player.begin(24000)
        val sink = sinks.single()
        var once = true
        // The Stop runs whole between the player's check for a Stop and its write. Its interrupt is swallowed here, so
        // only the player's own check of the stream can end the loop on a device that will take nothing more.
        sink.beforeWrite = {
            if (once) {
                once = false
                val stopper = Thread({ player.stop() }, "main-stand-in").apply { start() }
                while (stopper.isAlive) {
                    try {
                        stopper.join()
                    } catch (_: InterruptedException) {
                    }
                }
                Thread.interrupted()
            }
        }
        player.play(FloatArray(10))
        assertTrue("the player thread released the device", sink.released.await(2, TimeUnit.SECONDS))
        assertEquals(emptyList<String>(), sink.misuse)
    }

    @Test
    fun finishPlaysEverythingThenReleases() {
        player.begin(24000)
        val sink = sinks.single()
        repeat(3) { player.play(FloatArray(10)) }
        player.finish()
        assertEquals(3, sink.calls.count { it.startsWith("write@") })
        assertTrue(sink.isReleased)
        player.stop() // nothing playing: does nothing
        assertEquals(emptyList<String>(), sink.misuse)
    }

    @Test
    fun aNewStreamAfterAStopGetsItsOwnSink() {
        player.begin(24000)
        player.play(FloatArray(10))
        player.stop()
        player.begin(22050)
        player.play(FloatArray(10))
        player.finish()
        assertEquals(2, sinks.size)
        assertTrue(sinks[1].calls.any { it.startsWith("write@") })
        sinks.forEach { assertTrue(it.released.await(2, TimeUnit.SECONDS)); assertEquals(emptyList<String>(), it.misuse) }
    }
}
