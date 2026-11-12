// What each engine costs on a phone, shown before the user waits for it, and a free-memory check before loading.
// The figures are measured, not estimated: a Galaxy Z Fold5 (Snapdragon 8 Gen 2), 4 threads, 2026-11-12; see
// docs/android/knowledge/research/limitations.md. Another phone will differ; the screen says which phone they are
// from.
package ggtts.demo

data class EngineCost(
    /** The model files together. */
    val downloadGb: Double,
    /** Peak resident memory while rendering (it counts the mapped model files, which Android can reclaim). */
    val peakGb: Double,
    /** Seconds of rendering per second of audio. */
    val rtf: String,
    /** Seconds until the first chunk plays, for a sentence or two. */
    val firstAudioS: String,
)

object EngineCosts {
    const val MEASURED_ON = "a Galaxy Z Fold5 (Snapdragon 8 Gen 2), 4 threads"

    val BY_ENGINE = mapOf(
        "qwen3-tts" to EngineCost(downloadGb = 2.33, peakGb = 2.93, rtf = "about 7", firstAudioS = "30 to 50"),
        "kitten-tts-2" to EngineCost(downloadGb = 2.54, peakGb = 3.70, rtf = "about 12", firstAudioS = "50 to 70"),
    )

    /** The line the screen shows for [engine]: what it needs and how long it takes, and what it is not for. */
    fun describe(engine: String): String {
        val c = BY_ENGINE[engine] ?: return ""
        return "Needs %.1f GB of model files and about %.1f GB of memory. On %s, the first audio takes %s s, then each second of speech takes %s s to render. Not suitable for screen readers or anything interactive.".format(
            c.downloadGb, c.peakGb, MEASURED_ON, c.firstAudioS, c.rtf,
        )
    }

    /**
     * A warning when [availableBytes] is below what [engine] peaked at, or when the system says memory is low; null
     * when there is room. The peak counts mapped model files, which the system can drop, so this warns, it does not
     * forbid: the controller loads anyway on a second press.
     */
    fun memoryWarning(engine: String, availableBytes: Long, lowMemory: Boolean): String? {
        val c = BY_ENGINE[engine] ?: return null
        val availGb = availableBytes / 1e9
        if (!lowMemory && availGb >= c.peakGb) return null
        return "Low memory: %s used up to %.1f GB on the test phone, and %.1f GB is available now%s. Close other apps, or press Speak again to load anyway (the system may stop this app).".format(
            engine, c.peakGb, availGb, if (lowMemory) ", and the system reports memory is low" else "",
        )
    }
}
