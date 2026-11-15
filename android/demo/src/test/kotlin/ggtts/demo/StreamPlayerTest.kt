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

        /** play() throws, as AudioTrack.play() does on a track that failed to initialise. */
        var playThrows = false
        /** What every write returns when not paused; null takes everything. */
        var writeResult: Int? = null
        override fun play() {
            call("play")
            if (playThrows) throw IllegalStateException("play() called on an uninitialized AudioTrack")
            isPaused = false
        }
        /** Samples the device has taken, in all. */
        val taken = java.util.concurrent.atomic.AtomicInteger()
        /** Runs inside write(), on the player thread, before the write takes anything. */
        var beforeWrite: () -> Unit = {}
        @Volatile var isPaused = false
        override fun write(samples: FloatArray, offset: Int, count: Int): Int {
            call("write")
            beforeWrite()
            // A paused device takes nothing, as a full AudioTrack buffer does; with blockWrites a running one is full too.
            val n = if (isPaused || blockWrites) 0 else writeResult ?: count
            if (n > 0) taken.addAndGet(n)
            return n
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
    /** Applied to each new sink as begin() creates it, before its play(). */
    private var setUp: (FakeSink) -> Unit = {}
    private val player = StreamPlayer({ FakeSink().also { setUp(it); sinks += it } }, drainMs = 0)

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
    fun aDeviceThatFailsToStartIsReleasedAndTheNextStreamWorks() {
        setUp = { it.playThrows = true }
        val e = runCatching { player.begin(24000) }.exceptionOrNull()
        assertTrue("the failure reaches the caller: $e", e is IllegalStateException)
        val failed = sinks.single()
        assertEquals("released once", 1, failed.calls.count { it.startsWith("release@") })
        assertEquals(emptyList<String>(), failed.misuse)
        player.play(FloatArray(10)) // nothing is playing: dropped
        player.stop() // nothing is playing: does nothing
        assertEquals(1, failed.calls.count { it.startsWith("release@") })
        setUp = {}
        player.begin(24000)
        player.play(FloatArray(10))
        player.finish()
        assertTrue(sinks[1].calls.any { it.startsWith("write@") })
    }

    @Test
    fun aRefusedWriteIsReportedByFinishNotTakenForTheEnd() {
        setUp = { it.writeResult = -6 }
        player.begin(24000)
        player.play(FloatArray(10))
        player.play(FloatArray(10))
        val e = runCatching { player.finish() }.exceptionOrNull()
        assertTrue("finish reports the refusal: $e", e is PlaybackFailed && e.code == -6)
        val sink = sinks.single()
        assertTrue(sink.isReleased)
        assertEquals("the stream ended at the refused write", 1, sink.calls.count { it.startsWith("write@") })
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

    // ---- Pause and resume (#324). Each test names the planted fault it catches.

    /** Plant: pause() that does not pause the device. The device must stop taking audio at once. */
    @Test
    fun pauseHoldsTheDeviceAndResumePlaysOn() {
        player.begin(24000)
        val sink = sinks.single()
        player.pause()
        assertTrue("the device was not paused", sink.isPaused)
        assertTrue(player.paused)
        player.resume()
        assertTrue("the device was not played again", !sink.isPaused)
        assertTrue(!player.paused)
        player.stop()
    }

    /** Plant: a drain that ignores the pause. The stream must not end, nor finish() return, while it is held. */
    @Test
    fun aHeldStreamDoesNotEndUntilResumedAndPlaysEverything() {
        player.begin(24000)
        val sink = sinks.single()
        player.pause()
        // Rendering goes on while paused: pieces arrive and wait.
        player.play(FloatArray(100))
        player.play(FloatArray(50))
        assertTrue("finish() returned while the stream was paused", !returnsWithin(300) { player.finish() })
        assertEquals("the paused device took audio", 0, sink.taken.get())
        player.resume()
        assertTrue("the stream did not end after Resume", sink.released.await(2, TimeUnit.SECONDS))
        assertEquals("a piece held during the pause was lost", 150, sink.taken.get())
        assertEquals(emptyList<String>(), sink.misuse)
    }

    /**
     * Plant: a drain that ignores the pause. Everything has been written and sits in the device's buffer when Pause
     * comes; the stream must not end (and release the device with that audio unplayed) until Resume.
     */
    @Test
    fun aPauseAfterTheLastWriteStillHoldsTheEndOfTheStream() {
        player.begin(24000)
        val sink = sinks.single()
        player.play(FloatArray(80))
        val deadline = System.nanoTime() + 2_000_000_000L
        while (sink.taken.get() < 80 && System.nanoTime() < deadline) Thread.sleep(1)
        assertEquals(80, sink.taken.get())
        player.pause()
        assertTrue("finish() returned while the buffered end was paused", !returnsWithin(300) { player.finish() })
        assertEquals("the device was released with paused audio in it", 1L, sink.released.count)
        player.resume()
        assertTrue(sink.released.await(2, TimeUnit.SECONDS))
    }

    /** Plant: begin() that forgets a pause made before the first piece. The new stream must start held. */
    @Test
    fun aPauseBeforeTheFirstAudioHoldsTheStreamThatBegins() {
        player.pause()
        player.begin(24000)
        val sink = sinks.single()
        assertTrue("the stream began playing though Pause was pressed first", sink.isPaused)
        player.play(FloatArray(10))
        Thread.sleep(50)
        assertEquals(0, sink.taken.get())
        player.resume()
        player.finish()
        assertEquals(10, sink.taken.get())
    }

    /** Plant: stop() that leaves the pause set. Stop ends a held stream, and the next read is not held. */
    @Test
    fun stopEndsAHeldStreamAndTheNextOneIsNotHeld() {
        player.begin(24000)
        val first = sinks.single()
        player.pause()
        player.play(FloatArray(10))
        assertTrue(returnsWithin(1000) { player.stop() })
        assertTrue(first.released.await(2, TimeUnit.SECONDS))
        assertTrue(!player.paused)
        player.begin(24000)
        assertTrue("the next read began held", !sinks[1].isPaused)
        player.stop()
    }
}
