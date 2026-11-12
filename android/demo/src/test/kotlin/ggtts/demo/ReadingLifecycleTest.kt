// The reading service's lifecycle and the shared engine, on the host JVM: the service always goes foreground before
// it is stopped (a read that fails fast must not crash the app), it stops when a read ends, the notification's Stop
// ends a read the way the screen's does, and the screen and the system engine never evict, release or double the
// one loaded engine while either reads.
package ggtts.demo

import ggtts.Ggtts
import ggtts.GgttsException
import ggtts.Pcm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReadingLifecycleTest {
    // ---- the service's start and stop

    private val events = mutableListOf<String>()
    private val gate = ForegroundGate(startService = { events += "start" }, stopService = { events += "stop" })
    private fun command(action: String? = null, stopRead: () -> Unit = { events += "stopRead" }) =
        gate.onCommand(action, stopRead) { events += "foreground" }

    @Test
    fun aReadThatFailsFastStopsTheServiceOnlyAfterItWentForeground() {
        gate.readStarted()
        gate.readEnded() // before Android ran onStartCommand
        assertEquals("not stopped yet: Android would crash the app", listOf("start"), events)
        command()
        assertEquals(listOf("start", "foreground", "stop"), events)
    }

    @Test
    fun theServiceStopsWhenTheReadEnds() {
        gate.readStarted()
        command()
        gate.readEnded()
        assertEquals(listOf("start", "foreground", "stop"), events)
        gate.onDestroy()
        gate.readStarted()
        command()
        assertEquals("a second read starts it again", listOf("start", "foreground", "stop", "start", "foreground"), events)
    }

    @Test
    fun aSecondStartWhileStartingIsNotAnotherService() {
        gate.readStarted()
        gate.readEnded()
        gate.readStarted()
        command()
        assertEquals("one start, foreground, and it stays: a read runs", listOf("start", "foreground"), events)
    }

    @Test
    fun aStopThatReachesAServiceWithNoReadStopsItAgain() {
        command(ForegroundGate.ACTION_STOP)
        assertEquals("the read is asked to stop, and the service it started goes", listOf("stopRead", "stop"), events)
    }

    @Test
    fun theNotificationsStopEndsTheReadAsTheScreensDoes() {
        val statuses = mutableListOf<String>()
        val view = object : DemoView {
            override fun setBusy(busy: Boolean) {}
            override fun showStatus(text: String) { statuses += text }
            override fun showError(text: String) { statuses += "error: $text" }
        }
        val queued = mutableListOf<Runnable>()
        val c = DemoController(TtsBinding { Engine() }, view, Audio(), { queued += it }, { it.run() })
        gate.readStarted()
        command()
        c.speak(qwen, "今日天氣好好。", Ggtts.DEFAULT_SEED)
        command(ForegroundGate.ACTION_STOP, stopRead = { c.stop() })
        queued.single().run()
        assertEquals("Stopped after 0 chunk(s).", statuses.last())
        assertEquals("the Stop command never goes foreground again", 1, events.count { it == "foreground" })
    }

    // ---- the shared engine

    private val qwen = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343", 4)
    private val loads = mutableListOf<LoadRequest>()
    private val engines = mutableListOf<Engine>()

    private inner class Engine(val during: () -> Unit = {}) : TtsEngine {
        var closed = false
        override fun render(text: String, seed: Long): Pcm = error("streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean {
            check(!closed) { "rendered on a closed engine" }
            during()
            check(!closed) { "closed during its own read" }
            return onPiece(FloatArray(100), 24000)
        }
        override fun close() { closed = true }
    }

    private class Audio : AudioOut {
        override fun begin(rate: Int) {}
        override fun play(samples: FloatArray) {}
        override fun finish() {}
        override fun stop() {}
        override fun save(samples: FloatArray, rate: Int, engine: String) = "/x.wav"
    }

    private class Out : SynthesisOut {
        override val maxBytes = 4096
        val events = mutableListOf<String>()
        override fun start(sampleRate: Int) { events += "start" }
        override fun audio(pcm16: ByteArray, offset: Int, length: Int) {}
        override fun done() { events += "done" }
        override fun error(code: Int) { events += "error $code" }
    }

    private val voices = VoiceStore(object : KeyValue {
        val m = mutableMapOf<String, String>()
        override fun get(key: String) = m[key]
        override fun put(key: String, value: String) { m[key] = value }
    })

    /** A system-engine read whose render runs [during] in the middle, with the screen sharing the cache. */
    private fun systemReadWith(during: (EngineCache, DemoController, MutableList<String>) -> Unit): Triple<EngineCache, Out, List<String>> {
        lateinit var cache: EngineCache
        lateinit var screen: DemoController
        val shown = mutableListOf<String>()
        val view = object : DemoView {
            override fun setBusy(busy: Boolean) {}
            override fun showStatus(text: String) { shown += text }
            override fun showError(text: String) { shown += "error: $text" }
        }
        val binding = TtsBinding { r ->
            loads += r
            Engine { if (loads.size == 1) during(cache, screen, shown) }.also { engines += it }
        }
        cache = EngineCache(binding)
        screen = DemoController(binding, view, Audio(), { it.run() }, { it.run() }, cache = cache)
        voices.save(qwen)
        val out = Out()
        SynthesisSession(binding, voices, cache = cache).synthesize("yue", "HKG", "今日天氣好好。", Ggtts.DEFAULT_SEED, out)
        return Triple(cache, out, shown)
    }

    @Test
    fun aMemoryTrimDuringASystemReadKeepsTheEngine() {
        var released = true
        val (cache, out, _) = systemReadWith { c, _, _ -> released = TrimPolicy.trim(TrimPolicy.RUNNING_CRITICAL, false, c) }
        assertFalse("the screen is idle, but the system engine is reading", released)
        assertEquals(listOf("start", "done"), out.events)
        assertTrue(cache.holds(qwen))
        assertTrue("idle again: the trim releases it", TrimPolicy.trim(TrimPolicy.RUNNING_CRITICAL, false, cache))
        assertTrue(engines.single().closed)
    }

    @Test
    fun theScreenAskingForTheSameFilesMidReadIsRefusedAndReleasesNothing() {
        val (cache, out, shown) = systemReadWith { _, screen, _ -> screen.speak(qwen, "x", Ggtts.DEFAULT_SEED) }
        assertTrue("the screen says why", shown.last().startsWith("error: Not started: the qwen3-tts engine is busy"))
        assertEquals("the system read finished on its engine", listOf("start", "done"), out.events)
        assertEquals(1, loads.size)
        assertFalse(engines.single().closed)
        assertTrue(cache.holds(qwen))
    }

    @Test
    fun theScreenAskingForOtherFilesMidReadLoadsNoSecondModel() {
        val (_, out, shown) = systemReadWith { _, screen, _ ->
            screen.speak(qwen.copy(model = "/m/other.gguf"), "x", Ggtts.DEFAULT_SEED)
        }
        assertTrue(shown.last().startsWith("error: Not started"))
        assertEquals("one model resident, never two", 1, loads.size)
        assertEquals(listOf("start", "done"), out.events)
    }

    @Test
    fun theSystemEngineIsRefusedWhileTheScreenReads() {
        lateinit var cache: EngineCache
        val out = Out()
        val binding = TtsBinding { r ->
            loads += r
            Engine {
                if (loads.size == 1) {
                    SynthesisSession(TtsBinding { error("never loads") }, voices, cache = cache)
                        .synthesize("yue", "HKG", "x", Ggtts.DEFAULT_SEED, out)
                }
            }.also { engines += it }
        }
        cache = EngineCache(binding)
        voices.save(qwen)
        val view = object : DemoView {
            override fun setBusy(busy: Boolean) {}
            override fun showStatus(text: String) {}
            override fun showError(text: String) { throw AssertionError(text) }
        }
        DemoController(binding, view, Audio(), { it.run() }, { it.run() }, cache = cache).speak(qwen, "y", Ggtts.DEFAULT_SEED)
        assertEquals("busy: ERROR_SERVICE", listOf("error ${TtsCodes.ERROR_SERVICE}"), out.events)
        assertEquals(1, loads.size)
        assertFalse(engines.single().closed)
    }

    @Test
    fun releaseRefusesWhileAReadHoldsTheEngine() {
        val cache = EngineCache(TtsBinding { r -> loads += r; Engine().also { engines += it } })
        cache.acquire(qwen)
        assertFalse("a read holds it: refused", cache.release())
        assertFalse("and nothing closed", engines.single().closed)
        assertTrue(cache.holds(qwen))
        cache.done()
        assertTrue("idle: released", cache.release())
        assertTrue(engines.single().closed)
    }

    @Test
    fun aTrimDuringAReadDoesNotEvenTryToRelease() {
        // The trim's own busy check, apart from release()'s guard: during a read it must not call release() at all,
        // which the cache would log as a refusal.
        val log = mutableListOf<String>()
        val cache = EngineCache(TtsBinding { r -> loads += r; Engine().also { engines += it } }) { log += it }
        cache.acquire(qwen)
        assertFalse(TrimPolicy.trim(TrimPolicy.RUNNING_CRITICAL, false, cache))
        assertFalse("not tried: $log", log.any { it.startsWith("release refused") })
        assertFalse(engines.single().closed)
    }

    @Test
    fun aFailedReadClosesTheEngineSoTheNextLoadsAfresh() {
        val cache = EngineCache(TtsBinding { r ->
            loads += r
            object : TtsEngine {
                override fun render(text: String, seed: Long): Pcm = error("streams")
                override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
                    throw GgttsException(-5, "planted")
                override fun close() {}
            }
        })
        voices.save(qwen)
        val out = Out()
        SynthesisSession(TtsBinding { error("unused") }, voices, cache = cache).synthesize("yue", "HKG", "x", 1, out)
        assertEquals(listOf("error ${TtsCodes.ERROR_SYNTHESIS}"), out.events)
        assertFalse("closed after the failure", cache.holds(qwen))
        assertFalse("and not left marked busy", cache.busy)
    }
}
