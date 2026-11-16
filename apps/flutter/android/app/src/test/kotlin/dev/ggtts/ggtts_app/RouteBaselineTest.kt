package dev.ggtts.ggtts_app

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class RouteBaselineTest {
    @Test fun baselineBeforeWriteIsAllowedButLaterEventsFailClosed() {
        val route = RouteBaseline()
        assertTrue(route.initialEvent(2))
        route.beforeWrite()
        assertFalse(route.initialEvent(2))
    }

    @Test fun writesDoNotWaitForInitialCallbackAndDelayedEventFailsClosed() {
        val route = RouteBaseline()
        route.beforeWrite()
        assertFalse(route.initialEvent(7))
        assertFalse(RouteBaseline().initialEvent(null))
        val duplicate = RouteBaseline()
        assertTrue(duplicate.initialEvent(7))
        assertFalse(duplicate.initialEvent(7))
    }

    @Test fun eventAfterWriteBeginsCannotBeClassifiedAsInitial() {
        val route = RouteBaseline()
        val began = CountDownLatch(1)
        val done = CountDownLatch(1)
        val accepted = AtomicBoolean(true)
        val worker = Thread {
            check(began.await(2, TimeUnit.SECONDS))
            accepted.set(route.initialEvent(4))
            done.countDown()
        }
        worker.start()
        route.beforeWrite()
        began.countDown()
        assertTrue(done.await(2, TimeUnit.SECONDS))
        worker.join(2000)
        assertFalse(worker.isAlive)
        assertFalse(accepted.get())
    }
}
