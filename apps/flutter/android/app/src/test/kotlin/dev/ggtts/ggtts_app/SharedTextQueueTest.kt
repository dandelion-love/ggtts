package dev.ggtts.ggtts_app

import org.junit.Assert.*
import org.junit.Test

class SharedTextQueueTest {
    @Test fun unicodeFifoAndBoundedRefusal() {
        val queue = SharedTextQueue(maxChars = 12, maxItems = 2)
        val first = "\u4eca\u65e5\ud840\udc00"
        val second = "\u9999\u6e2f"
        assertTrue(queue.offer("android.intent.action.SEND", "text/plain", first))
        assertTrue(queue.offer("android.intent.action.SEND", "text/plain", second))
        assertFalse(queue.offer("android.intent.action.SEND", "text/plain", "overflow"))
        assertEquals(first, queue.take())
        assertEquals(second, queue.take())
        assertNull(queue.take())
    }

    @Test fun malformedAndOtherActionsCannotAlterQueue() {
        val queue = SharedTextQueue(maxChars = 3)
        assertFalse(queue.offer("android.intent.action.VIEW", "text/plain", "ok"))
        assertFalse(queue.offer("android.intent.action.SEND", "text/html", "ok"))
        assertFalse(queue.offer("android.intent.action.SEND", "text/plain", ""))
        assertFalse(queue.offer("android.intent.action.SEND", "text/plain", "long"))
        assertFalse(queue.offer("android.intent.action.SEND", "text/plain", "\ud800"))
        assertFalse(queue.offer("android.intent.action.SEND", "text/plain", "\u0000"))
        assertNull(queue.take())
    }
}
