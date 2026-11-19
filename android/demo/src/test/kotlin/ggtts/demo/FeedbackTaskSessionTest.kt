package ggtts.demo

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class FeedbackTaskSessionTest {
    @Test fun dismissalSuppressesAlreadyPostedResult() {
        val posted = CountDownLatch(1)
        val pending = AtomicReference<() -> Unit>()
        val delivered = AtomicBoolean(false)
        val worker = Executors.newSingleThreadExecutor()
        val session = FeedbackTaskSession(worker) {
            pending.set(it); posted.countDown()
        }
        session.submit({ 7 }) { delivered.set(true) }
        assertTrue(posted.await(2, TimeUnit.SECONDS))
        session.close()
        pending.get().invoke()
        assertFalse(delivered.get())
        worker.shutdownNow()
    }

    @Test fun dismissalSuppressesRunningResult() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val ended = CountDownLatch(1)
        val delivered = AtomicBoolean(false)
        val worker = Executors.newSingleThreadExecutor()
        val session = FeedbackTaskSession(worker) { it() }
        session.submit({
            started.countDown()
            try { release.await() } catch (_: InterruptedException) {}
            ended.countDown()
            1
        }) { delivered.set(true) }
        assertTrue(started.await(2, TimeUnit.SECONDS))
        session.close()
        release.countDown()
        assertTrue(ended.await(2, TimeUnit.SECONDS))
        assertFalse(delivered.get())
        worker.shutdownNow()
    }
    @Test fun openGateRestoresAfterDismissalPreparationFailureOrHostError() {
        for (outcome in listOf("dismiss", "preparation", "host")) {
            var visible = true
            var enabled = true
            val gate = FeedbackOpenGate { if (visible) enabled = it }
            assertTrue(outcome, gate.begin())
            assertFalse(outcome, enabled)
            assertFalse(outcome, gate.begin())
            gate.finish()
            assertTrue(outcome, enabled)
            assertTrue(outcome, gate.begin())
            visible = false
            gate.finish()
            assertFalse("dismissed sheet must not be updated", enabled)
        }
    }
}
