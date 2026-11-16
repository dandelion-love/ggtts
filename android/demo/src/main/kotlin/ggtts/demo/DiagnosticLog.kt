package ggtts.demo

import java.util.concurrent.atomic.AtomicLong

/** Fixed vocabulary only: diagnostics cannot carry text, paths, options or exception messages. */
enum class DiagnosticStage { EVIDENCE, PLAYBACK, FEEDBACK_PREPARE, HOST_SETUP, FEEDBACK_REOPEN, BRIDGE_SAVE, BRIDGE_RESTORE, EXPORT_PREPARE, EXPORT_STAGE, EXPORT_PUBLISH, EXPORT_REOPEN }
enum class DiagnosticOutcome { OK, UNAVAILABLE, INCOMPLETE, INVALID, INTERRUPTED, UNSUPPORTED, PROVIDER_FAILURE, FAILURE }

class DiagnosticLog(
    private val sink: (String) -> Unit = {},
    private val clock: () -> Long = System::nanoTime,
) {
    fun start(stage: DiagnosticStage): Event = Event(stage, sequence.incrementAndGet(), clock())
    inner class Event internal constructor(private val stage: DiagnosticStage, private val id: Long, private val started: Long) {
        fun finish(outcome: DiagnosticOutcome) {
            val ms = ((clock() - started).coerceAtLeast(0) / 1_000_000)
            sink("stage=${stage.name.lowercase()} outcome=${outcome.name.lowercase()} operation=$id elapsed_ms=$ms")
        }
    }
    companion object { private val sequence = AtomicLong() }
}

/** One success and one failure per host, even when the page saves on every edit. */
internal class BridgeDiagnosticLimiter {
    private val success = java.util.concurrent.atomic.AtomicBoolean()
    private val failure = java.util.concurrent.atomic.AtomicBoolean()
    fun report(event: DiagnosticLog.Event, ok: Boolean) {
        val first = if (ok) success.compareAndSet(false, true) else failure.compareAndSet(false, true)
        if (first) event.finish(if (ok) DiagnosticOutcome.OK else DiagnosticOutcome.FAILURE)
    }
}
