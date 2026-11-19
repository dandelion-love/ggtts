package ggtts.demo

import org.junit.Assert.*
import org.junit.Test

/** Simulates process death by reopening the durable store after each publication boundary. */
class PublicationRecoveryIndependentTest {
    private class Store(var saved: String? = null) : QueueStore {
        override fun read() = saved
        override fun write(text: String) { saved = text }
    }
    private val request = LoadRequest("qwen3-tts", "/m/talker.gguf", "/m/codec.gguf", null, "hk_voice_343")
    private val pub = Publication("same-name-q" + "1".repeat(32) + ".flac", "1".repeat(32), "a".repeat(64), 123L)

    @Test fun crashBeforeInsertionKeepsIntentUntilAbsenceIsProven() {
        val store = Store()
        val q = RenderQueue(store)
        val first = q.add(request, "one")
        q.add(request, "two")
        assertEquals(first.id, q.take()!!.id)
        q.preparePublication(first.id, pub)
        var reopened = RenderQueue(store)
        assertEquals(pub, reopened.publications().single().publication)
        assertNull("unreconciled publication blocks rerender", reopened.take())
        reopened.reconcilePublication(first.id, PublicationRecovery.Unknown("provider inaccessible"))
        reopened = RenderQueue(store)
        assertEquals(RenderQueue.State.PUBLISHING, reopened.all[0].state)
        assertNull(reopened.take())
        reopened.reconcilePublication(first.id, PublicationRecovery.Pending("content://owned"))
        assertNull(RenderQueue(store).take())
        reopened.reconcilePublication(first.id, PublicationRecovery.Missing)
        reopened = RenderQueue(store)
        assertEquals(RenderQueue.State.WAITING, reopened.all[0].state)
        assertNull(reopened.all[0].publication)
        assertEquals(first.id, reopened.take()!!.id)
    }

    @Test fun crashAfterPublicationSettlesOriginalWithoutRerender() {
        val store = Store()
        val q = RenderQueue(store)
        val first = q.add(request, "one")
        val second = q.add(request, "two")
        q.take()
        q.preparePublication(first.id, pub)
        val reopened = RenderQueue(store)
        reopened.reconcilePublication(first.id, PublicationRecovery.Published("content://owned"))
        val again = RenderQueue(store)
        assertEquals(RenderQueue.State.DONE, again.all[0].state)
        assertEquals("content://owned", again.all[0].note)
        assertNull(again.all[0].publication)
        assertEquals(second.id, again.take()!!.id)
        assertEquals(1, again.all.count { it.state == RenderQueue.State.DONE })
    }

    @Test fun legacyInterruptedRenderReturnsToWaitingWithoutLosingOtherEntries() {
        val store = Store()
        val q = RenderQueue(store)
        val first = q.add(request, "one")
        q.add(request, "two")
        q.take()
        val reopened = RenderQueue(store)
        assertEquals(2, reopened.all.size)
        assertEquals(first.id, reopened.take()!!.id)
    }

    private class Provider : JournaledReadSink {
        val rows = mutableMapOf<Publication, Pair<String, Boolean>>()
        var unavailable = false
        var renders = 0
        var failAfterPublish = false
        override fun save(name: String, title: String, flac: ByteArray, publish: (() -> Unit) -> Boolean): String = error("journal required")
        override fun saveJournaled(name: String, title: String, flac: ByteArray, publication: Publication,
                                   publish: (() -> Unit) -> Boolean): String {
            renders++
            val uri = "content://owned/$renders"
            if (!publish { rows[publication] = uri to true }) throw SaveCancelled()
            if (failAfterPublish) throw IllegalStateException("planted crash after publication")
            return uri
        }
        override fun locate(publication: Publication): PublicationRecovery {
            if (unavailable) return PublicationRecovery.Unknown("unavailable")
            val row = rows[publication] ?: return PublicationRecovery.Missing
            return if (row.second) PublicationRecovery.Published(row.first) else PublicationRecovery.Pending(row.first)
        }
        override fun recoverPending(publication: Publication, uri: String): PublicationRecovery {
            val row = rows[publication] ?: return PublicationRecovery.Missing
            check(row.first == uri)
            rows[publication] = uri to true
            return PublicationRecovery.Published(uri)
        }
    }

    @Test fun workerRecoversExactOwnedRowAndLeavesSameNamedUnrelatedMedia() {
        val store = Store()
        val q = RenderQueue(store)
        val first = q.add(request, "one")
        q.take()
        q.preparePublication(first.id, pub)
        val provider = Provider()
        val other = pub.copy(sha256 = "b".repeat(64))
        provider.rows[other] = "content://unrelated" to true
        provider.rows[pub] = "content://owned" to false
        var loads = 0
        val worker = QueueWorker(RenderQueue(store), EngineCache(TtsBinding { loads++; error("must not rerender") }), provider, seed = 1)
        provider.unavailable = true
        assertTrue(worker.step() is QueueWorker.Outcome.RecoveryBlocked)
        assertEquals(RenderQueue.State.PUBLISHING, RenderQueue(store).all.single().state)
        provider.unavailable = false
        assertTrue(worker.step() is QueueWorker.Outcome.Recovered)
        assertEquals(RenderQueue.State.DONE, RenderQueue(store).all.single().state)
        assertEquals("content://owned", RenderQueue(store).all.single().note)
        assertEquals("content://unrelated", provider.rows[other]!!.first)
        assertEquals(0, provider.renders)
        assertEquals(0, loads)
    }

    @Test fun exceptionAfterPublicationDoesNotPublishTwiceOnRestart() {
        val store = Store()
        val q = RenderQueue(store)
        q.add(request, "one")
        val provider = Provider()
        provider.failAfterPublish = true
        val binding = TtsBinding {
            object : TtsEngine {
                override fun render(text: String, seed: Long): ggtts.Pcm = error("stream only")
                override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
                    onPiece(FloatArray(2000) { 0.25f }, 24000)
                override fun close() {}
            }
        }
        val first = QueueWorker(q, EngineCache(binding), provider, seed = 1)
        assertTrue(first.step() is QueueWorker.Outcome.RecoveryBlocked)
        assertEquals(1, provider.rows.size)
        assertEquals(RenderQueue.State.PUBLISHING, RenderQueue(store).all.single().state)
        val resumed = QueueWorker(RenderQueue(store), EngineCache(binding), provider, seed = 1)
        assertTrue(resumed.step() is QueueWorker.Outcome.Recovered)
        assertEquals(RenderQueue.State.DONE, RenderQueue(store).all.single().state)
        assertEquals(1, provider.rows.size)
        assertEquals(1, provider.renders)
    }
}
