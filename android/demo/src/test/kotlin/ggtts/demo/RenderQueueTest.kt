// The render-for-later queue and its worker, and the FLAC they save, on the host JVM. Each test names the planted fault
// it catches.
package ggtts.demo

import ggtts.Pcm
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RenderQueueTest {
    private class Store(var text: String? = null) : QueueStore {
        override fun read() = text
        override fun write(text: String) { this.text = text }
    }

    private val request = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343")

    // ---- the queue

    /** Plant: recover() that leaves an entry RENDERING. A process killed mid-render must not lose the entry. */
    @Test
    fun anEntryCutOffByAKilledProcessRendersAgainAndNoneIsLostOrDoneTwice() {
        val store = Store()
        val q = RenderQueue(store)
        val a = q.add(request, "第一段")
        val b = q.add(request, "第二段")
        assertEquals(a.id, q.take()!!.id) // rendering a when the process dies
        val reopened = RenderQueue(store)
        assertEquals(listOf(RenderQueue.State.WAITING, RenderQueue.State.WAITING), reopened.all.map { it.state })
        assertEquals("a renders again, first", a.id, reopened.take()!!.id)
        reopened.done(a.id, "content://a")
        assertEquals(b.id, reopened.take()!!.id)
        reopened.done(b.id, "content://b")
        assertNull("a done entry is never taken again", reopened.take())
        assertEquals(listOf("content://a", "content://b"), RenderQueue(store).all.map { it.note })
    }

    /** Plant: take() that hands out a second entry while one renders. One renders at a time. */
    @Test
    fun oneEntryRendersAtATime() {
        val q = RenderQueue(Store())
        q.add(request, "one"); q.add(request, "two")
        val first = q.take()!!
        assertNull(q.take())
        q.requeue(first.id)
        assertEquals("a stopped entry waits again, at its place", first.id, q.take()!!.id)
    }

    @Test
    fun anEntryKeepsItsRequestAndTextThroughTheFileWhateverTheyHold() {
        val store = Store()
        val q = RenderQueue(store)
        val text = "一行\t有 tab\n另一行「引號」"
        q.add(request.with(LoadRequest.RENDER, "instruct", "慢慢講"), text)
        val back = RenderQueue(store).all.single()
        assertEquals(text, back.text)
        assertEquals(request.with(LoadRequest.RENDER, "instruct", "慢慢講"), back.request)
        assertEquals("一行 有 tab 另一行「引號」", back.title)
        assertTrue(RenderQueue.titleOf("x".repeat(100)).length <= 40)
    }

    // ---- the worker

    private class Engine(val pieces: Int, val onPiece: (Int) -> Unit = {}) : TtsEngine {
        override fun render(text: String, seed: Long): Pcm = error("streams")
        override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean {
            for (i in 0 until pieces) {
                this.onPiece(i)
                if (!onPiece(FloatArray(1000) { 0.25f }, 24000)) return false
            }
            return true
        }
        override fun close() {}
    }

    private class Sink : ReadSink {
        val saved = mutableListOf<Triple<String, String, ByteArray>>()
        override fun save(name: String, title: String, flac: ByteArray, publish: (() -> Unit) -> Boolean): String {
            if (!publish { saved += Triple(name, title, flac) }) throw SaveCancelled()
            return "content://media/${saved.size}"
        }
    }

    private fun worker(q: RenderQueue, sink: Sink, binding: TtsBinding) =
        QueueWorker(q, EngineCache(binding), sink, seed = 1, now = { 1_700_000_000_000L })

    @Test
    fun aQueuedReadRendersToOneFlacWithItsTitleAndIsDone() {
        val q = RenderQueue(Store())
        val sink = Sink()
        q.add(request, "今日天氣好好。")
        val o = worker(q, sink, TtsBinding { Engine(3) }).step()
        assertTrue(o.toString(), o is QueueWorker.Outcome.Done)
        val (name, title, flac) = sink.saved.single()
        assertTrue(name, name.startsWith("ggtts-qwen3-tts-") && name.endsWith(".flac"))
        assertEquals("今日天氣好好。", title)
        assertEquals("今日天氣好好。", Flac.title(flac))
        assertEquals(RenderQueue.State.DONE, q.all.single().state)
        assertTrue(worker(q, sink, TtsBinding { Engine(3) }).step() is QueueWorker.Outcome.Idle)
    }

    /**
     * Plant: a failure left waiting (or saved anyway). An entry that cannot render is marked failed with the reason,
     * nothing is saved for it, and the queue moves on to the next.
     */
    @Test
    fun anEntryThatCannotRenderFailsWithItsReasonSavesNothingAndTheQueueMovesOn() {
        val q = RenderQueue(Store())
        val sink = Sink()
        q.add(request, "missing model")
        q.add(request, "fine")
        var first = true
        val binding = TtsBinding {
            if (first) { first = false; throw IllegalStateException("model /m/talker.gguf: No such file or directory") }
            Engine(2)
        }
        val w = worker(q, sink, binding)
        val o = w.step()
        assertTrue(o.toString(), o is QueueWorker.Outcome.Failed)
        assertEquals(RenderQueue.State.FAILED, q.all[0].state)
        assertTrue(q.all[0].note, "No such file" in q.all[0].note)
        assertTrue("a failed entry saved a file", sink.saved.isEmpty())
        assertTrue(w.step() is QueueWorker.Outcome.Done)
        assertEquals(1, sink.saved.size)
    }

    /**
     * AC3's other half. Plant: a save that fails (storage full) left RENDERING or retried forever. The failure comes
     * after the render, from the sink; the entry is FAILED with that reason, nothing is counted as saved, and the queue
     * moves on. (That MediaStore removes the pending item when its write throws is MusicFolder.save, read, not run.)
     */
    @Test
    fun aSaveThatFailsForStorageFailsTheEntryWithItsReasonAndTheQueueMovesOn() {
        val q = RenderQueue(Store())
        q.add(request, "storage full")
        q.add(request, "fine")
        var first = true
        val saved = mutableListOf<String>()
        val sink = ReadSink { name, _, _, publish ->
            if (first) { first = false; throw java.io.IOException("No space left on device") }
            if (!publish { saved += name }) throw SaveCancelled()
            "content://media/1"
        }
        val w = QueueWorker(q, EngineCache(TtsBinding { Engine(2) }), sink, seed = 1, now = { 1_700_000_000_000L })
        val o = w.step()
        assertTrue(o.toString(), o is QueueWorker.Outcome.Failed)
        assertEquals(RenderQueue.State.FAILED, q.all[0].state)
        assertTrue(q.all[0].note, "No space left on device" in q.all[0].note)
        assertTrue(saved.isEmpty())
        assertTrue(w.step() is QueueWorker.Outcome.Done)
        assertEquals(1, saved.size)
        assertEquals(RenderQueue.State.DONE, q.all[1].state)
    }

    /** Plant: a stop that saves what was rendered so far. A stopped render saves nothing, and its entry waits. */
    @Test
    fun aStopSavesNothingAndTheEntryWaits() {
        val q = RenderQueue(Store())
        val sink = Sink()
        q.add(request, "long")
        lateinit var w: QueueWorker
        w = worker(q, sink, TtsBinding { Engine(5) { i -> if (i == 2) w.stop() } })
        val o = w.step()
        assertTrue(o.toString(), o is QueueWorker.Outcome.Stopped)
        assertEquals(RenderStops.Reason.PERSON, (o as QueueWorker.Outcome.Stopped).reason)
        assertTrue(sink.saved.isEmpty())
        assertEquals(RenderQueue.State.WAITING, q.all.single().state)
        assertTrue(RenderStops.stopped(o.reason, q.pending), "Nothing half-made was saved" in RenderStops.stopped(o.reason, q.pending))
    }

    @Test
    fun stopMessagesUseWordsAndPassTheWaitingCount() {
        val words = Words { key, args -> "$key:${args.joinToString()}" }
        assertEquals("stopping:", RenderStops.stopping(RenderStops.Reason.PERSON, words))
        assertEquals("render_stopping_memory:", RenderStops.stopping(RenderStops.Reason.MEMORY, words))
        assertEquals("render_stopped_person_empty:", RenderStops.stopped(RenderStops.Reason.PERSON, 0, words))
        assertEquals("render_stopped_person_waiting:2", RenderStops.stopped(RenderStops.Reason.PERSON, 2, words))
        assertEquals("render_stopped_memory_empty:", RenderStops.stopped(RenderStops.Reason.MEMORY, 0, words))
        assertEquals("render_stopped_memory_waiting:3", RenderStops.stopped(RenderStops.Reason.MEMORY, 3, words))
    }

    // ---- #418: stopping cleanly

    /**
     * Plant: a memory stop reported as the person's, or saving what it had. A low-memory stop ends the render after its
     * chunk with nothing saved, keeps the entry, and says it was memory.
     */
    @Test
    fun aLowMemoryStopSavesNothingKeepsTheEntryAndSaysWhy() {
        val q = RenderQueue(Store())
        val sink = Sink()
        q.add(request, "long"); q.add(request, "next")
        lateinit var w: QueueWorker
        var trims = 1 // one trim: the next render runs in full
        w = worker(q, sink, TtsBinding { Engine(5) { i -> if (i == 1 && trims-- > 0) w.stop(RenderStops.Reason.MEMORY) } })
        val o = w.step() as QueueWorker.Outcome.Stopped
        assertEquals(RenderStops.Reason.MEMORY, o.reason)
        assertTrue(sink.saved.isEmpty())
        assertEquals(listOf(RenderQueue.State.WAITING, RenderQueue.State.WAITING), q.all.map { it.state })
        val said = RenderStops.stopped(o.reason, q.pending)
        assertTrue(said, "low on memory" in said && "2 waiting" in said)
        assertTrue("low on memory" in RenderStops.stopping(RenderStops.Reason.MEMORY))
        // A fresh worker resumes the entry: a stop is never a failure.
        assertTrue(worker(q, sink, TtsBinding { Engine(5) }).step() is QueueWorker.Outcome.Done)
    }

    @Test
    fun aStopBeforeOrBetweenStepsDoesNotStartAnotherEntry() {
        for (between in listOf(false, true)) {
            val q = RenderQueue(Store())
            val sink = Sink()
            q.add(request, "first")
            if (between) q.add(request, "second")
            val w = worker(q, sink, TtsBinding { Engine(1) })
            if (between) assertTrue(w.step() is QueueWorker.Outcome.Done)
            w.stop(RenderStops.Reason.MEMORY)
            assertEquals(RenderStops.Reason.MEMORY, (w.step() as QueueWorker.Outcome.Halted).reason)
            assertEquals(1, q.pending)
            assertEquals(if (between) 1 else 0, sink.saved.size)
        }
    }

    @Test
    fun aStopDuringSaveCancelsPublicationAndRequeues() {
        val q = RenderQueue(Store())
        q.add(request, "first")
        var published = false
        lateinit var w: QueueWorker
        val sink = ReadSink { _, _, _, publish ->
            w.stop()
            if (!publish { published = true }) throw SaveCancelled()
            "content://media/1"
        }
        w = QueueWorker(q, EngineCache(TtsBinding { Engine(1) }), sink, seed = 1)
        assertTrue(w.step() is QueueWorker.Outcome.Stopped)
        assertTrue(!published)
        assertEquals(RenderQueue.State.WAITING, q.all.single().state)
    }

    /**
     * Plant: every trim stops a render. Leaving the app trims at 20 (UI hidden) and 40 (background), and the queue must
     * go on then; only critical shortage (15, or 80 and up) stops it.
     */
    @Test
    fun onlyACriticalMemoryTrimStopsARender() {
        assertTrue(RenderStops.stopsFor(15))
        assertTrue(RenderStops.stopsFor(80))
        for (level in listOf(5, 10, 20, 40, 60)) assertTrue("level $level stopped a render", !RenderStops.stopsFor(level))
    }

    /** Plant: Clear queue dropping finished entries too, or none. It drops what waits; done and failed stay listed. */
    @Test
    fun clearQueueDropsWhatWaitsAndKeepsWhatIsSettled() {
        val store = Store()
        val q = RenderQueue(store)
        val a = q.add(request, "done"); val b = q.add(request, "failed"); q.add(request, "waits"); q.add(request, "waits too")
        q.take(); q.done(a.id, "content://a")
        q.take(); q.failed(b.id, "gone")
        assertEquals(2, q.clearWaiting())
        assertEquals(listOf(RenderQueue.State.DONE, RenderQueue.State.FAILED), RenderQueue(store).all.map { it.state })
        assertEquals(0, q.pending)
    }

    /** Plant: an engine in use counted as a failure. The entry waits until the engine is free. */
    @Test
    fun anEngineInUseLeavesTheEntryWaiting() {
        val q = RenderQueue(Store())
        q.add(request, "later")
        val cache = EngineCache(TtsBinding { Engine(1) })
        cache.acquire(LoadRequest("kitten-tts-2", "/lm.gguf", "/s3.gguf", "/v.json")) // the screen is reading
        val o = QueueWorker(q, cache, Sink(), seed = 1).step()
        assertTrue(o.toString(), o is QueueWorker.Outcome.Busy)
        assertEquals(RenderQueue.State.WAITING, q.all.single().state)
    }

    // ---- FLAC

    /** Plant: a CRC with the wrong polynomial. Decoders refuse such frames; FLAC's reference values hold them. */
    @Test
    fun theFlacCrcsAndFrameNumbersMatchTheFormat() {
        assertEquals(0xf4.toByte(), Flac.crc8("123456789".toByteArray()))
        assertEquals(0xfee8, Flac.crc16("123456789".toByteArray()))
        assertArrayEquals(byteArrayOf(0x7f), Flac.utf8Number(0x7f))
        assertArrayEquals(byteArrayOf(0xc2.toByte(), 0x80.toByte()), Flac.utf8Number(0x80))
        assertArrayEquals(byteArrayOf(0xe0.toByte(), 0xa0.toByte(), 0x80.toByte()), Flac.utf8Number(0x800))
    }

    @Test
    fun theFlacCarriesItsRateLengthTitleAndEverySample() {
        val samples = FloatArray(Flac.BLOCK + 10) { if (it % 2 == 0) 0.5f else -1.5f }
        val f = Flac.encode(samples, 24000, "標題 title")
        assertEquals("fLaC", String(f, 0, 4))
        assertEquals("標題 title", Flac.title(f))
        assertNull(Flac.title(Flac.encode(samples, 24000)))
        // STREAMINFO: 24000 Hz in 20 bits from byte 18, 16-bit mono, and the sample count in 36 bits.
        val rate = ((f[18].toInt() and 0xff) shl 12) or ((f[19].toInt() and 0xff) shl 4) or ((f[20].toInt() and 0xff) shr 4)
        assertEquals(24000, rate)
        val count = ((f[21].toLong() and 0x0f) shl 32) or ((f[22].toLong() and 0xff) shl 24) or ((f[23].toLong() and 0xff) shl 16) or
            ((f[24].toLong() and 0xff) shl 8) or (f[25].toLong() and 0xff)
        assertEquals(samples.size.toLong(), count)
        // The last 2 bytes are the last frame's CRC-16; the 10 samples before it are the last block, verbatim.
        val end = f.size - 2
        val last = (0 until 10).map { ((f[end - 20 + 2 * it].toInt() shl 8) or (f[end - 19 + 2 * it].toInt() and 0xff)).toShort() }
        assertEquals((0 until 10).map { Flac.toI16(samples[Flac.BLOCK + it]) }, last)
        assertEquals("saturated", Short.MIN_VALUE, Flac.toI16(-1.5f))
    }
}
