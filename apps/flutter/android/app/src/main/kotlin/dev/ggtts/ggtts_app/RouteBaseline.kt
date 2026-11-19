package dev.ggtts.ggtts_app

/**
 * Classifies an initial route event without blocking the first write. A route
 * callback after any write attempt is treated as an interruption, even if it
 * may only be a delayed initial event. Event delivery may itself be delayed;
 * this class cannot prove that no interruption is pending at completion.
 */
internal class RouteBaseline {
    private var attemptedWrite = false
    private var initialObserved = false

    @Synchronized fun beforeWrite() { attemptedWrite = true }

    /** True means the event is an initial baseline before any write attempt. */
    @Synchronized fun initialEvent(routeId: Int?): Boolean {
        if (routeId == null || attemptedWrite || initialObserved) return false
        initialObserved = true
        return true
    }
}
