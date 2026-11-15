// What each engine costs on a phone, shown before the user waits for it, and a free-memory check before loading.
// The figures are measured, not estimated: a Galaxy Z Fold5 (Snapdragon 8 Gen 2), 4 threads, 2026-11-12; see
// docs/android/knowledge/research/limitations.md. Another phone will differ; the screen says which phone they are
// from.
package ggtts.demo

data class EngineCost(
    /** The model files together. */
    val downloadGb: Double,
    /**
     * Peak resident memory, in GB of 10^9 bytes (`VmHWM` is KiB). For Kitten-TTS 2 the peak is a load-time spike; a
     * render holds about 2.9 GB (docs/android/knowledge/research/kitten-memory-and-tq1.md).
     */
    val peakGb: Double,
    /** Seconds of rendering per second of audio, about. */
    val rtfAbout: Int,
    /**
     * Seconds until the first chunk plays, for a sentence or two, at 4 threads, as a range: Qwen3-TTS 32.2 s (thread
     * sweep) to 49.9 s (a three-sentence line), Kitten-TTS 2 47.4 s (budget run) to 55.7 s (thread sweep).
     */
    val firstAudioMinS: Int,
    val firstAudioMaxS: Int,
) {
    /** The first-audio range in English ("30 to 50"); [EngineCosts.firstAudio] gives it in another language. */
    val firstAudioS: String get() = EnglishWords.get("seconds_range", firstAudioMinS, firstAudioMaxS)
}

object EngineCosts {
    val MEASURED_ON: String get() = EnglishWords.get("measured_on")

    val BY_ENGINE = mapOf(
        "qwen3-tts" to EngineCost(downloadGb = 2.33, peakGb = 3.00, rtfAbout = 7, firstAudioMinS = 30, firstAudioMaxS = 50),
        "kitten-tts-2" to EngineCost(downloadGb = 2.54, peakGb = 3.78, rtfAbout = 12, firstAudioMinS = 45, firstAudioMaxS = 60),
    )

    /** The line the screen shows for [engine]: what it needs and how long it takes, and what it is not for. */
    fun describe(engine: String, w: Words = EnglishWords): String {
        val c = BY_ENGINE[engine] ?: return ""
        return w.get(
            "engine_costs", c.downloadGb, c.peakGb, w.get("measured_on"), firstAudio(engine, w), w.get("about_n", c.rtfAbout),
        )
    }

    /** [engine]'s first-audio range in [w]'s language ("30 to 50"), or "" for an engine with no figures. */
    fun firstAudio(engine: String, w: Words = EnglishWords): String =
        BY_ENGINE[engine]?.let { w.get("seconds_range", it.firstAudioMinS, it.firstAudioMaxS) } ?: ""

    /**
     * A warning when [availableBytes] is below what [engine] peaked at, or when the system says memory is low; null
     * when there is room. The peak is a few seconds of load (the model files are read whole and copied); a render holds
     * less, so this warns, it does not forbid: the controller loads anyway on a second press.
     */
    fun memoryWarning(engine: String, availableBytes: Long, lowMemory: Boolean, w: Words = EnglishWords): String? {
        val c = BY_ENGINE[engine] ?: return null
        val availGb = availableBytes / 1e9
        if (!lowMemory && availGb >= c.peakGb) return null
        return w.get("low_memory", engine, c.peakGb, availGb, if (lowMemory) w.get("low_memory_system") else "")
    }
}
