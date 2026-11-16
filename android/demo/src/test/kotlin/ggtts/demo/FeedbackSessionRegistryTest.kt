package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class FeedbackSessionRegistryTest {
    @Test fun privateAndExportedViewsCannotShareInput() {
        val first = FeedbackSessionRegistry.acquire("same-digest")!!
        assertNull(FeedbackSessionRegistry.acquire("same-digest"))
        val other = FeedbackSessionRegistry.acquire("other-digest")!!
        first.close()
        val next = FeedbackSessionRegistry.acquire("same-digest")!!
        first.close()
        assertNull(FeedbackSessionRegistry.acquire("same-digest"))
        next.close(); other.close()
        assertNotNull(FeedbackSessionRegistry.acquire("same-digest")!!.also { it.close() })
    }

    @Test fun concurrentOpenHasOnlyOneWinner() {
        val start = CountDownLatch(1)
        val acquired = CountDownLatch(12)
        val release = CountDownLatch(1)
        val winners = AtomicInteger()
        val threads = (0 until 12).map {
            Thread {
                start.await()
                val lease = FeedbackSessionRegistry.acquire("contended")
                if (lease != null) winners.incrementAndGet()
                acquired.countDown()
                release.await()
                lease?.close()
            }.also { it.start() }
        }
        start.countDown()
        assertTrue(acquired.await(2, TimeUnit.SECONDS))
        assertEquals(1, winners.get())
        release.countDown()
        threads.forEach { it.join(2000) }
        assertNotNull(FeedbackSessionRegistry.acquire("contended")!!.also { it.close() })
    }

    @Test fun exportedRowGateRejectsRepeatedTapsUntilTerminalOutcome() {
        var enabled = true
        val gate = FeedbackOpenGate { enabled = it }
        assertTrue(gate.begin())
        assertFalse(enabled)
        assertFalse(gate.begin())
        gate.finish()
        assertTrue(enabled)
        assertTrue(gate.begin())
        gate.finish()
    }
    @Test fun closingRejectsNewSavesButWaitsForInFlightSaveBeforeRelease() {
        val lease = FeedbackSessionRegistry.acquire("in-flight")!!
        val started = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val ended = CountDownLatch(1)
        val worker = Thread {
            lease.runIfActive {
                started.countDown()
                finish.await()
            }
            ended.countDown()
        }
        worker.start()
        assertTrue(started.await(2, TimeUnit.SECONDS))
        lease.close()
        assertNull(lease.runIfActive { "late" })
        assertNull(FeedbackSessionRegistry.acquire("in-flight"))
        finish.countDown()
        assertTrue(ended.await(2, TimeUnit.SECONDS))
        val next = FeedbackSessionRegistry.acquire("in-flight")
        assertNotNull(next)
        next!!.close()
        worker.join(2000)
    }
}
