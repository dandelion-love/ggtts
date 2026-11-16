package ggtts.demo

import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

/**
 * Two-invocation MediaStore crash test. Run prepare, force-stop the isolated ggtts.demo.recovery target, then recover and cleanup.
 * All audio is a generated tone; queue journals and media rows have random names and are deleted by exact URI.
 */
@RunWith(AndroidJUnit4::class)
class MediaStoreRecoveryInstrumentationTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val runId: String get() = InstrumentationRegistry.getArguments().getString("fixture")
        ?.takeIf { it.matches(Regex("[0-9a-f]{32}")) }
        ?: throw AssertionError("the host runner must provide a random fixture id")
    private val fixture: File get() = File(instrumentation.context.filesDir, "queue-recovery-fixture-$runId.json")

    private fun writeFixture(document: JSONObject) {
        val part = File(fixture.path + ".part")
        FileOutputStream(part).use { stream ->
            stream.write(document.toString().toByteArray(Charsets.UTF_8))
            stream.fd.sync()
        }
        android.system.Os.rename(part.absolutePath, fixture.absolutePath)
        val dirFd = android.system.Os.open(fixture.parentFile!!.absolutePath, android.system.OsConstants.O_RDONLY, 0)
        try { android.system.Os.fsync(dirFd) } finally { android.system.Os.close(dirFd) }
    }

    /** Creates PUBLISHING journals with a published row, a complete pending row, a missing row, and an unrelated row. */
    @Test fun preparePublishedPendingMissingAndUnrelatedPending() {
        assertTrue("pending-row inclusion requires Android 11+", Build.VERSION.SDK_INT >= 30)
        assertFalse("refusing to overwrite an existing test fixture", fixture.exists())
        val target = instrumentation.targetContext.applicationContext
        val activityManager = target.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        assertFalse("isolated recovery app must have no active render service before the test",
            activityManager.getRunningServices(Int.MAX_VALUE).any { it.service?.className == RenderService::class.java.name })
        val resolver = target.contentResolver
        val collection = MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val entries = JSONArray()
        val createdRows = JSONArray()
        val token = UUID.randomUUID().toString().replace("-", "")
        val samples = FloatArray(24_000 / 8) { i -> (0.15 * kotlin.math.sin(2.0 * Math.PI * 440.0 * i / 24_000.0)).toFloat() }
        val bytes = Flac.encode(samples, 24_000, "Synthetic queue recovery probe")
        val hash = sha256(bytes)
        val request = LoadRequest("qwen3-tts", "/synthetic/talker.gguf", "/synthetic/codec.gguf", null, "synthetic")
        val music = MusicFolder(target)
        writeFixture(JSONObject().put("rows", createdRows).put("entries", entries))

        fun checkpoint() {
            writeFixture(JSONObject().put("rows", createdRows).put("entries", entries))
        }

        fun addJournal(rowState: String) {
            val id = UUID.randomUUID().toString().replace("-", "")
            val name = "ggtts-recovery-$id-q$id.flac"
            val publication = Publication(name, id, hash, bytes.size.toLong())
            val queueFile = "queue-recovery-$id.txt"
            val isolatedQueueFile = File(target.filesDir, queueFile)
            assertFalse("refusing to overwrite an existing isolated journal", isolatedQueueFile.exists() || File(isolatedQueueFile.path + ".part").exists())
            val queue = RenderQueue(FileQueueStore(isolatedQueueFile))
            val entry = queue.add(request, "Synthetic recovery probe $rowState")
            assertEquals(entry.id, queue.take()!!.id)
            queue.preparePublication(entry.id, publication)
            entries.put(JSONObject().put("queueFile", queueFile).put("state", rowState).put("name", name)
                .put("token", id).put("sha256", hash).put("bytes", bytes.size).put("title", entry.title))
            checkpoint()
            var uri: Uri? = null
            when (rowState) {
                "published" -> uri = music.save(name, Flac.MIME, entry.title, publication = publication) { it.write(bytes) }
                "pending" -> {
                    val values = ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                        put(MediaStore.MediaColumns.MIME_TYPE, Flac.MIME)
                        put(MediaStore.MediaColumns.RELATIVE_PATH, Storage.MUSIC_PATH)
                        put(MediaStore.MediaColumns.TITLE, entry.title)
                        put(MediaStore.MediaColumns.IS_PENDING, 1)
                    }
                    val pendingUri = resolver.insert(collection, values)
                        ?: throw AssertionError("could not create a pending synthetic row")
                    uri = pendingUri
                    resolver.openOutputStream(pendingUri).use { output ->
                        val stream = output ?: throw AssertionError("could not open the pending synthetic row")
                        stream.write(bytes)
                    }
                }
                "missing" -> Unit
                else -> fail("unknown fixture row state")
            }
            if (uri != null) {
                createdRows.put(JSONObject().put("uri", uri.toString()).put("name", name).put("owner", target.packageName))
                checkpoint()
                assertEquals(target.packageName, rowOwner(resolver, uri!!))
            }
        }

        try {
            addJournal("published")
            addJournal("pending")
            addJournal("missing")
            val unrelatedName = "ggtts-recovery-unrelated-$token.flac"
            val sentinelRecord = JSONObject().put("uri", "").put("name", unrelatedName).put("owner", target.packageName)
            createdRows.put(sentinelRecord)
            checkpoint() // Persist the exact name/path/owner intent before MediaStore insertion.
            val unrelated = resolver.insert(collection, ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, unrelatedName)
                put(MediaStore.MediaColumns.MIME_TYPE, Flac.MIME)
                put(MediaStore.MediaColumns.RELATIVE_PATH, Storage.MUSIC_PATH)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            })
                ?: throw AssertionError("could not create unrelated pending sentinel")
            sentinelRecord.put("uri", unrelated.toString())
            checkpoint()
            assertEquals(target.packageName, rowOwner(resolver, unrelated))
        } catch (t: Throwable) {
            // Keep fixture and journals unless exact row cleanup is proven.
            val cleanup = runCatching {
                cleanupRows(target, createdRows)
                cleanupJournalRows(target, entries)
                cleanupJournals(target, entries)
                assertTrue("test fixture cleanup failed", fixture.delete() || !fixture.exists())
            }.exceptionOrNull()
            if (cleanup != null) {
                t.addSuppressed(cleanup)
                throw AssertionError("prepare failed; retained recovery fixture at ${fixture.absolutePath}", t)
            }
            throw t
        }
    }

    /** Run only after testPrepare... returned and the host force-stopped the isolated recovery target. */
    @Test fun recoverAfterForceStopWithoutDuplicates() {
        assertTrue("pending-row inclusion requires Android 11+", Build.VERSION.SDK_INT >= 30)
        assertTrue("run the preparation phase first", fixture.isFile)
        val target = instrumentation.targetContext.applicationContext
        val music = MusicFolder(target)
        val document = JSONObject(fixture.readText())
        val entries = document.getJSONArray("entries")
        var renders = 0
        val outputRows = document.getJSONArray("rows")
        val sink = object : JournaledReadSink {
            override fun save(name: String, title: String, flac: ByteArray, publish: (() -> Unit) -> Boolean): String =
                error("journaled save required")
            override fun saveJournaled(name: String, title: String, flac: ByteArray, publication: Publication,
                                       publish: (() -> Unit) -> Boolean): String {
                val uri = music.save(name, Flac.MIME, title, publish = publish, publication = publication) { it.write(flac) }
                outputRows.put(JSONObject().put("uri", uri.toString()).put("name", name).put("owner", target.packageName))
                try {
                    writeFixture(document.put("rows", outputRows))
                } catch (t: Throwable) {
                    runCatching { cleanupRows(target, JSONArray().put(JSONObject()
                        .put("uri", uri.toString()).put("name", name).put("owner", target.packageName))) }
                    throw t
                }
                return uri.toString()
            }
            override fun locate(publication: Publication) = music.locate(publication)
            override fun recoverPending(publication: Publication, uri: String) = music.recoverPending(publication, uri)
        }
        val binding = TtsBinding {
            renders++
            object : TtsEngine {
                override fun render(text: String, seed: Long): ggtts.Pcm = error("stream only")
                override fun renderStream(text: String, seed: Long, onPiece: (FloatArray, Int) -> Boolean): Boolean =
                    onPiece(FloatArray(24_000 / 8) { 0.15f }, 24_000)
                override fun close() {}
            }
        }
        for (i in 0 until entries.length()) {
            val item = entries.getJSONObject(i)
            val publication = Publication(item.getString("name"), item.getString("token"),
                item.getString("sha256"), item.getLong("bytes"))
            val queue = RenderQueue(FileQueueStore(File(target.filesDir, item.getString("queueFile"))))
            assertEquals(RenderQueue.State.PUBLISHING, queue.all.single().state)
            val worker = QueueWorker(queue, EngineCache(binding), sink, seed = 1)
            val outcome = worker.step()
            when (item.getString("state")) {
                "published", "pending" -> assertTrue(outcome.toString(), outcome is QueueWorker.Outcome.Recovered)
                "missing" -> assertTrue(outcome.toString(), outcome is QueueWorker.Outcome.Done)
            }
            assertEquals("queue entry settled exactly once", RenderQueue.State.DONE, queue.all.single().state)
            assertTrue("the queue did not save the recovered URI", queue.all.single().note.startsWith("content://"))
            if (item.getString("state") == "published" || item.getString("state") == "pending") {
                assertTrue(music.locate(publication).toString(), music.locate(publication) is PublicationRecovery.Published)
            }
        }
        assertEquals("only the missing publication was rendered", 1, renders)
        assertEquals("the unrelated pending row survived recovery", 1, (0 until document.getJSONArray("rows").length())
            .count { row -> document.getJSONArray("rows").getJSONObject(row).getString("name").contains("unrelated") })
        writeFixture(document.put("rows", outputRows))
        val sentinel = (0 until outputRows.length()).map { outputRows.getJSONObject(it) }
            .single { it.getString("name").contains("unrelated") }
        assertTrue("sentinel must remain pending", rowPending(target.contentResolver, Uri.parse(sentinel.getString("uri"))))
    }

    /** Safe cleanup phase: only delete URIs created by this test after checking owner, path and exact display name. */
    @Test fun cleanupExactRowsAndJournals() {
        if (!fixture.isFile) return
        val target = instrumentation.targetContext.applicationContext
        val document = JSONObject(fixture.readText())
        val entries = document.getJSONArray("entries")
        cleanupRows(target, document.getJSONArray("rows"))
        cleanupJournalRows(target, entries)
        cleanupJournals(target, entries)
        assertTrue("test fixture cleanup failed", fixture.delete() || !fixture.exists())
    }

    private fun cleanupRows(target: android.content.Context, rows: JSONArray) {
        val resolver = target.contentResolver
        for (i in 0 until rows.length()) {
            val row = rows.getJSONObject(i)
            val recordedUri = row.optString("uri")
            val candidates = if (recordedUri.isNotEmpty()) listOf(Uri.parse(recordedUri))
                else exactRowsByName(resolver, row.getString("name"))
            for (uri in candidates) {
                val cursor = queryIncludingPending(resolver, uri, arrayOf(
                    MediaStore.MediaColumns.DISPLAY_NAME,
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    MediaStore.MediaColumns.OWNER_PACKAGE_NAME,
                )) ?: throw AssertionError("cannot verify exact cleanup row $uri")
                val owner = cursor.use {
                    if (!it.moveToFirst()) null
                    else {
                        if (it.getString(0) != row.getString("name") || it.getString(1) != Storage.MUSIC_PATH) {
                            throw AssertionError("refusing to delete row with mismatched name/path: $uri")
                        }
                        it.getString(2)
                    }
                } ?: continue
                val expectedOwner = row.optString("owner")
                val allowedOwners = setOf(target.packageName, instrumentation.context.packageName)
                if (expectedOwner.isNotEmpty() && owner != expectedOwner) {
                    throw AssertionError("refusing to delete row with mismatched owner: $uri")
                }
                if (owner !in allowedOwners) throw AssertionError("refusing to delete media not owned by this test/app: $uri")
                assertEquals("failed to delete exact test row $uri", 1, resolver.delete(uri, null, null))
            }
        }
    }

    /** Recover URIs not yet written to the cleanup ledger from their isolated per-test journal files. */
    private fun cleanupJournalRows(target: android.content.Context, entries: JSONArray) {
        val music = MusicFolder(target)
        for (i in 0 until entries.length()) {
            val queueFile = File(target.filesDir, entries.getJSONObject(i).getString("queueFile"))
            val parsed = RenderQueue.parse(FileQueueStore(queueFile).read())
            for (entry in parsed) {
                val publication = entry.publication ?: continue
                val uri = when (val found = music.locate(publication)) {
                    is PublicationRecovery.Published -> found.uri
                    is PublicationRecovery.Pending -> found.uri
                    PublicationRecovery.Missing -> continue
                    is PublicationRecovery.Unknown -> throw AssertionError("cannot safely clean publication: ${found.reason}")
                }
                cleanupRows(target, JSONArray().put(JSONObject().put("uri", uri)
                    .put("name", publication.name).put("owner", target.packageName)))
            }
        }
    }

    private fun exactRowsByName(resolver: android.content.ContentResolver, name: String): List<Uri> {
        val collection = MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val selection = "${MediaStore.MediaColumns.DISPLAY_NAME} = ? AND ${MediaStore.MediaColumns.RELATIVE_PATH} = ?"
        val query = Bundle().apply {
            putString(android.content.ContentResolver.QUERY_ARG_SQL_SELECTION, selection)
            putStringArray(android.content.ContentResolver.QUERY_ARG_SQL_SELECTION_ARGS, arrayOf(name, Storage.MUSIC_PATH))
            putInt(MediaStore.QUERY_ARG_MATCH_PENDING, MediaStore.MATCH_INCLUDE)
        }
        val cursor = resolver.query(collection, arrayOf(MediaStore.MediaColumns._ID), query, null)
            ?: throw AssertionError("cannot find an unrecorded test row by exact name")
        val out = mutableListOf<Uri>()
        cursor.use {
            val id = it.getColumnIndexOrThrow(MediaStore.MediaColumns._ID)
            while (it.moveToNext()) out += android.content.ContentUris.withAppendedId(collection, it.getLong(id))
        }
        return out
    }

    private fun queryIncludingPending(
        resolver: android.content.ContentResolver,
        uri: Uri,
        projection: Array<String>,
    ) = resolver.query(uri, projection, Bundle().apply {
        putInt(MediaStore.QUERY_ARG_MATCH_PENDING, MediaStore.MATCH_INCLUDE)
    }, null)

    private fun cleanupJournals(target: android.content.Context, entries: JSONArray) {
        for (i in 0 until entries.length()) {
            val name = entries.getJSONObject(i).getString("queueFile")
            if (name.matches(Regex("""queue-recovery-[0-9a-f]{32}\.txt"""))) {
                val file = File(target.filesDir, name)
                val part = File(file.path + ".part")
                if (file.exists() && !file.delete()) throw AssertionError("could not delete exact test journal ${file.absolutePath}")
                if (part.exists() && !part.delete()) throw AssertionError("could not delete exact test journal part ${part.absolutePath}")
            }
        }
    }

    private fun rowOwner(resolver: android.content.ContentResolver, uri: Uri): String {
        val cursor = queryIncludingPending(resolver, uri, arrayOf(MediaStore.MediaColumns.OWNER_PACKAGE_NAME))
            ?: throw AssertionError("could not inspect MediaStore owner")
        cursor.use { assertTrue(it.moveToFirst()); return it.getString(0) }
        error("unreachable")
    }

    private fun rowPending(resolver: android.content.ContentResolver, uri: Uri): Boolean {
        val cursor = queryIncludingPending(resolver, uri, arrayOf(MediaStore.MediaColumns.IS_PENDING))
            ?: throw AssertionError("could not inspect pending sentinel")
        cursor.use { assertTrue(it.moveToFirst()); return it.getInt(0) != 0 }
        error("unreachable")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it) }
}
