package ggtts.demo

/**
 * Linearizes notification Clear queue requests with the render service's worker shutdown.
 *
 * A true request is consumed by [finish]. Once finish has crossed the boundary, requests return false and the caller
 * must clear the waiting queue itself; no notification can be lost between the worker's last check and stopSelf.
 */
internal class QueueClearHandoff {
    private var finishing = false
    private var clearRequested = false

    @Synchronized
    fun begin() {
        finishing = false
        clearRequested = false
    }

    /** Returns true when the worker's shutdown path will consume this clear request. */
    @Synchronized
    fun requestClear(): Boolean {
        if (finishing) return false
        clearRequested = true
        return true
    }

    /** Closes the handoff and returns whether the worker must clear before stopping. */
    @Synchronized
    fun finish(): Boolean {
        finishing = true
        val clear = clearRequested
        clearRequested = false
        return clear
    }
}
