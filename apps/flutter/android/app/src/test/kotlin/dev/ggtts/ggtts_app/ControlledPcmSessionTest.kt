package dev.ggtts.ggtts_app

import org.junit.Assert.*
import org.junit.Test

class ControlledPcmSessionTest {
    private class Fake : ControlledPcmSink {
        var pos = 0L
        var count = 0
        var released = 0
        var flushed = 0
        var fail = false
        override fun play() { check(!fail) }
        override fun pause() { check(!fail) }
        override fun write(samples: FloatArray, offset: Int, count: Int): Int {
            check(!fail)
            this.count++
            return minOf(count, 2)
        }
        override fun head(): Long { check(!fail); return pos }
        override fun flush() { flushed++ }
        override fun release() { released++ }
    }

    @Test fun partialPauseStopAndStaleSession() {
        val sinks = mutableListOf<Fake>()
        val bridge = ControlledPcmSession { Fake().also(sinks::add) }
        val first = bridge.open(24000)
        assertEquals(2, bridge.write(first, FloatArray(4)))
        sinks[0].pos = 1
        assertEquals(ControlledPcmSession.Snapshot(2, 1, false), bridge.snapshot(first))
        bridge.pause(first)
        assertEquals(0, bridge.write(first, FloatArray(1)))
        assertEquals(1, sinks[0].count)
        bridge.resume(first)
        bridge.stop(first)
        assertEquals(1, sinks[0].released)
        assertEquals(1, sinks[0].flushed)
        val second = bridge.open(24000)
        assertTrue(second > first)
        assertThrows(IllegalStateException::class.java) { bridge.stop(first) }
        assertEquals(2, bridge.write(second, FloatArray(3)))
        bridge.stop(second)
    }

    @Test fun invalidHeadAndWriteFailureReleaseExactlyOnce() {
        val sink = Fake()
        val bridge = ControlledPcmSession { sink }
        val id = bridge.open(24000)
        bridge.write(id, FloatArray(4))
        sink.pos = 3
        assertThrows(IllegalStateException::class.java) { bridge.snapshot(id) }
        assertEquals(1, sink.released)
        assertThrows(IllegalStateException::class.java) { bridge.stop(id) }
        val retry = Fake()
        val next = ControlledPcmSession { retry }
        val nextId = next.open(24000)
        retry.fail = true
        assertThrows(IllegalStateException::class.java) { next.write(nextId, FloatArray(1)) }
        assertEquals(1, retry.released)
    }

    @Test fun rejectsSecondOpenAndNonzeroInitialHead() {
        val sink = Fake().apply { pos = 1 }
        val bridge = ControlledPcmSession { sink }
        assertThrows(IllegalStateException::class.java) { bridge.open(24000) }
        assertEquals(1, sink.released)
        val healthy = ControlledPcmSession { Fake() }
        val id = healthy.open(24000)
        assertThrows(IllegalStateException::class.java) { healthy.open(24000) }
        healthy.stop(id)
    }
    @Test fun headResetAndStopFailureAreTerminal() {
        val sink = Fake()
        val bridge = ControlledPcmSession { sink }
        val id = bridge.open(24000)
        bridge.write(id, FloatArray(4))
        sink.pos = 1
        bridge.snapshot(id)
        sink.pos = 0
        assertThrows(IllegalStateException::class.java) { bridge.snapshot(id) }
        assertEquals(1, sink.released)

        val stopSink = Fake()
        val stopped = ControlledPcmSession { stopSink }
        val stoppedId = stopped.open(24000)
        stopSink.fail = true
        assertThrows(IllegalStateException::class.java) { stopped.stop(stoppedId) }
        assertEquals(1, stopSink.released)
        assertThrows(IllegalStateException::class.java) { stopped.stop(stoppedId) }
    }

}
