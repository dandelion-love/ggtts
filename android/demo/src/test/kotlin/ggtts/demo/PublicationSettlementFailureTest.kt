package ggtts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** Plant a storage failure after MediaStore has published but before the queue can persist DONE. */
class PublicationSettlementFailureTest {
    private class Store : QueueStore {
        var saved: String? = null
        var failDoneOnce = false
        override fun read() = saved
        override fun write(text: String) {
            if (failDoneOnce && text.lines().any { it.split("\t").getOrNull(1) == RenderQueue.State.DONE.name }) {
                failDoneOnce = false
                throw java.io.IOException("planted queue fsync failure")
            }
            saved = text
        }
    }

    private class Sink : JournaledReadSink {
        var recovery: PublicationRecovery = PublicationRecovery.Missing
        var published = 0
        override fun save(name: String, title: String, flac: ByteArray, publish: (() -> Unit) -> Boolean): String =
            error("journaled save required")
        override fun saveJournaled(name: String, title: String, flac: ByteArray, publication: Publication,
                                   publish: (() -> Unit) -> Boolean): String {
            val uri = "content://owned/1"
            if (!publish {
                published++
                recovery = PublicationRecovery.Published(uri)
            }) throw SaveCancelled()
            return uri
        }
        override fun locate(publication: Publication) = recovery
        override fun recoverPending(publication: Publication, uri: String) = recovery
    }

    private fun cache(loads: () -> Unit) = EngineCache(TtsBinding {
        loads()
        object : TtsEngine {
            override fun render(text: String, seed: Long): ggtts.Pcm = error("stream only")
            override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
                onPiece(FloatArray(2000) { 0.25f }, 24000)
            override fun close() {}
        }
    })

    @Test fun serviceStopsInsteadOfBusyLoopingOnUnknownPublication() {
        val source = File("src/main/kotlin/ggtts/demo/RenderService.kt").readText()
        val loop = source.substringAfter("when (val o = w.step())").substringBefore("else -> Unit")
        val blockedBranch = loop.substringAfter("is QueueWorker.Outcome.RecoveryBlocked")
        assertTrue("ambiguous provider recovery must break the worker loop", "break" in blockedBranch)
        assertTrue("blocked recovery must leave a visible message", "done(" in blockedBranch)
    }

    @Test fun publishedReadKeepsItsJournalWhenDoneCannotBePersisted() {
        val store = Store()
        val request = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343")
        val queue = RenderQueue(store)
        queue.add(request, "one")
        val sink = Sink()
        var loads = 0
        store.failDoneOnce = true

        val first = QueueWorker(queue, cache { loads++ }, sink, seed = 1).step()
        assertTrue(first.toString(), first is QueueWorker.Outcome.RecoveryBlocked)
        assertEquals(RenderQueue.State.PUBLISHING, queue.all.single().state)
        assertEquals(RenderQueue.State.PUBLISHING, RenderQueue(store).all.single().state)
        assertEquals(1, sink.published)

        val resumed = QueueWorker(RenderQueue(store), cache { loads++ }, sink, seed = 1).step()
        assertTrue(resumed.toString(), resumed is QueueWorker.Outcome.Recovered)
        assertEquals(RenderQueue.State.DONE, RenderQueue(store).all.single().state)
        assertEquals("content://owned/1", RenderQueue(store).all.single().note)
        assertEquals("no duplicate render", 1, loads)
        assertEquals("no duplicate publication", 1, sink.published)
    }
}
