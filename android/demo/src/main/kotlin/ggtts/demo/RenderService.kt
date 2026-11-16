// Rendering queued reads in the background: no playback, one entry at a time, each saved as FLAC to Music/ggtts/ with
// its title. QueueWorker is the work apart from Android (the JVM tests drive it); RenderService runs it in a foreground
// service so it goes on with the screen off.
//
// The service type. A render that does not play is not mediaPlayback, and Android 14 and later require the declared
// type to match the work. On Android 15 and later it is mediaProcessing (made for work like transcoding media); below,
// dataSync. Both are capped by Android 15 at 6 hours a day per app, and Android calls onTimeout when that runs out: the
// entry rendering then stops cleanly and waits again, and the queue resumes when the app is next opened. At 7 to 13 s
// of rendering per second of speech on a phone, 6 hours renders about 28 to 50 minutes of speech a day.
package ggtts.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicReference
import java.security.MessageDigest
import java.util.UUID

/** Why a queued render stops, which levels of a memory trim stop one, and what the person is told. */
object RenderStops {
    enum class Reason { PERSON, MEMORY }

    /**
     * Whether a memory trim at [level] stops a queued render: only when Android is critically short
     * (TRIM_MEMORY_RUNNING_CRITICAL, 15, or TRIM_MEMORY_COMPLETE, 80 and up). Leaving the app (TRIM_MEMORY_UI_HIDDEN,
     * 20, and TRIM_MEMORY_BACKGROUND, 40) does not: rendering with the app in the background is the whole point.
     */
    fun stopsFor(level: Int): Boolean = level == 15 || level >= 80

    /** While the chunk in progress finishes: a chunk can take most of a minute, so Stop must not look ignored. */
    fun stopping(reason: Reason, words: Words = EnglishWords): String = words.get(
        if (reason == Reason.PERSON) "stopping" else "render_stopping_memory"
    )

    /** After the render stopped: what happened, that nothing half-made was saved, and what waits. */
    fun stopped(reason: Reason, waiting: Int, words: Words = EnglishWords): String {
        val key = when (reason) {
            Reason.PERSON -> if (waiting == 0) "render_stopped_person_empty" else "render_stopped_person_waiting"
            Reason.MEMORY -> if (waiting == 0) "render_stopped_memory_empty" else "render_stopped_memory_waiting"
        }
        return if (waiting == 0) words.get(key) else words.get(key, waiting)
    }
}

/** Where a finished render goes: Music/ggtts/ on a phone. Writes nothing it does not finish. Returns where it went. */
fun interface ReadSink {
    fun save(name: String, title: String, flac: ByteArray, publish: (() -> Unit) -> Boolean): String
}

class QueueWorker(
    private val queue: RenderQueue,
    private val cache: EngineCache,
    private val sink: ReadSink,
    private val seed: Long,
    log: (String) -> Unit = {},
    private val clock: () -> Long = System::nanoTime,
    /** Wall-clock milliseconds, for the saved file's name. */
    private val now: () -> Long = System::currentTimeMillis,
) {
    private val log = bestEffortLog(log)

    /** What one [step] did. */
    sealed class Outcome {
        object Idle : Outcome()
        /** The engine is reading for the screen or the system engine; try again later. The entry waits. */
        object Busy : Outcome()
        /** A stop arrived between entries, before another entry was taken. */
        data class Halted(val reason: RenderStops.Reason) : Outcome()
        data class Done(val entry: RenderQueue.Entry, val where: String, val seconds: Double, val renderMs: Long) : Outcome()
        data class Recovered(val entry: RenderQueue.Entry, val where: String) : Outcome()
        data class RecoveryBlocked(val entry: RenderQueue.Entry, val reason: String) : Outcome()
        data class Failed(val entry: RenderQueue.Entry, val reason: String) : Outcome()
        data class Stopped(val entry: RenderQueue.Entry, val reason: RenderStops.Reason) : Outcome()
    }

    private val stopRequested = AtomicReference<RenderStops.Reason?>(null)

    /** Stop the render in progress after its current chunk, for [reason]; its entry waits again. */
    fun stop(reason: RenderStops.Reason = RenderStops.Reason.PERSON) {
        stopRequested.compareAndSet(null, reason)
    }

    /** Render the next waiting entry, if any, to the end, and settle it. */
    fun step(): Outcome {
        stopRequested.get()?.let { return Outcome.Halted(it) }
        queue.publications().firstOrNull()?.let { pending ->
            val journal = sink as? JournaledReadSink
                ?: return Outcome.RecoveryBlocked(pending, "the saved read needs publication recovery, but this sink cannot reconcile it")
            val publication = pending.publication
                ?: return Outcome.RecoveryBlocked(pending, "the saved publication journal is incomplete; refusing to render a possible duplicate")
            val recovery = try {
                when (val located = journal.locate(publication)) {
                    is PublicationRecovery.Pending -> journal.recoverPending(publication, located.uri)
                    else -> located
                }
            } catch (t: Throwable) {
                PublicationRecovery.Unknown(t.message ?: t.javaClass.simpleName)
            }
            try {
                queue.reconcilePublication(pending.id, recovery)
            } catch (t: Throwable) {
                return Outcome.RecoveryBlocked(pending, "could not persist publication recovery: ${t.message ?: t.javaClass.simpleName}")
            }
            when (recovery) {
                is PublicationRecovery.Published -> return Outcome.Recovered(pending, recovery.uri)
                is PublicationRecovery.Unknown -> return Outcome.RecoveryBlocked(pending, recovery.reason)
                is PublicationRecovery.Pending -> return Outcome.RecoveryBlocked(pending, "the pending media row could not be safely reconciled")
                PublicationRecovery.Missing -> Unit // Safe to render again; the exact journaled row is absent.
            }
        }
        stopRequested.get()?.let { return Outcome.Halted(it) }
        val e = queue.take() ?: return Outcome.Idle
        stopRequested.get()?.let {
            queue.requeue(e.id)
            return Outcome.Stopped(e, it)
        }
        val tts = try {
            cache.acquire(e.request)
        } catch (busy: EngineInUse) {
            queue.requeue(e.id)
            return Outcome.Busy
        } catch (t: Throwable) {
            return fail(e, "the voice did not load: ${t.message ?: t.javaClass.simpleName}")
        }
        var ok = false
        try {
            val t0 = clock()
            val pieces = mutableListOf<FloatArray>()
            var rate = 0
            val finished = tts.renderStream(e.text, seed) { samples, r ->
                if (stopRequested.get() != null) return@renderStream false
                rate = r
                pieces += samples
                true
            }
            val renderMs = (clock() - t0) / 1_000_000
            if (!finished || stopRequested.get() != null) {
                val reason = stopRequested.get() ?: RenderStops.Reason.PERSON
                queue.requeue(e.id)
                ok = true
                log("queue_render outcome=interrupted elapsed_ms=$renderMs chunks=${pieces.size}")
                return Outcome.Stopped(e, reason)
            }
            val all = FloatArray(pieces.sumOf { it.size })
            var at = 0
            for (p in pieces) { p.copyInto(all, at); at += p.size }
            if (all.isEmpty() || rate <= 0) return fail(e, "the render made no audio")
            val flac = Flac.encode(all, rate, e.title)
            val gate: (() -> Unit) -> Boolean = { publish ->
                // This atomic read is the publication gate. A Stop observed first cancels the pending item; once the
                // gate is passed, publication owns the item and its result is settled as DONE or FAILED. In
                // particular, Stop never waits for a potentially slow MediaStore provider call.
                if (stopRequested.get() != null) false else { publish(); true }
            }
            val where = if (sink is JournaledReadSink) {
                val token = UUID.randomUUID().toString().replace("-", "")
                val name = Storage.queuedName(e.request.engine, now()).removeSuffix(".flac") + "-q$token.flac"
                val publication = Publication(name, token, sha256(flac), flac.size.toLong())
                queue.preparePublication(e.id, publication)
                sink.saveJournaled(name, e.title, flac, publication, gate)
            } else {
                sink.save(Storage.queuedName(e.request.engine, now()), e.title, flac, gate)
            }
            queue.done(e.id, where)
            ok = true
            val seconds = all.size.toDouble() / rate
            log("queue_render outcome=ok elapsed_ms=$renderMs")
            return Outcome.Done(e, where, seconds, renderMs)
        } catch (t: Throwable) {
            if (t is SaveCancelled) {
                val reason = stopRequested.get() ?: RenderStops.Reason.PERSON
                queue.requeue(e.id)
                ok = true
                return Outcome.Stopped(e, reason)
            }
            if (queue.all.firstOrNull { it.id == e.id }?.state == RenderQueue.State.PUBLISHING) {
                // The provider may have inserted or published before throwing. Keep the durable identity and let a
                // later step establish the exact row state instead of creating a duplicate.
                queue.reconcilePublication(e.id, PublicationRecovery.Unknown(t.message ?: t.javaClass.simpleName))
                ok = true
                return Outcome.RecoveryBlocked(e, t.message ?: t.javaClass.simpleName)
            }
            return fail(e, t.message ?: t.javaClass.simpleName)
        } finally {
            cache.done(failed = !ok)
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it) }

    private fun fail(e: RenderQueue.Entry, reason: String): Outcome {
        queue.failed(e.id, reason)
        log("queue_render outcome=failure")
        return Outcome.Failed(e, reason)
    }
}

/** A pending file was discarded because the queue stopped before publication. */
class SaveCancelled : Exception()

/** The queue kept in app storage. */
class FileQueueStore(private val file: File) : QueueStore {
    override fun read(): String? = if (file.isFile) file.readText() else null
    override fun write(text: String) {
        file.parentFile?.mkdirs()
        val part = File(file.path + ".part")
        java.io.FileOutputStream(part).use { stream ->
            stream.write(text.toByteArray(Charsets.UTF_8))
            stream.fd.sync()
        }
        // Same-directory rename atomically replaces the prior journal without deleting the last known-good copy.
        android.system.Os.rename(part.absolutePath, file.absolutePath)
        // Sync the directory entry too: PUBLISHING must be durable before MediaStore insertion starts.
        val dirFd = android.system.Os.open(
            file.parentFile!!.absolutePath,
            android.system.OsConstants.O_RDONLY,
            0,
        )
        try { android.system.Os.fsync(dirFd) } finally { android.system.Os.close(dirFd) }
    }
}

/** The process's one queue and worker. */
object Queue {
    private var queue: RenderQueue? = null

    fun get(context: Context): RenderQueue = synchronized(this) {
        queue ?: RenderQueue(FileQueueStore(File(context.applicationContext.filesDir, "render-queue.txt"))).also { queue = it }
    }
}

/** Works the render queue in the background, with a notification saying how many wait. */
class RenderService : Service() {
    private var worker: QueueWorker? = null
    private var thread: Thread? = null

    override fun onBind(intent: Intent?): IBinder? = null

    /** Linearizes a notification's Clear queue with the worker's final shutdown handoff. */
    private val clearHandoff = QueueClearHandoff()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // The notification's Stop and Clear queue: a control, never a start.
        when (intent?.action) {
            ACTION_STOP, ACTION_CLEAR -> {
                val clearing = intent.action == ACTION_CLEAR
                val running = thread?.isAlive == true
                val handedToWorker = clearing && running && clearHandoff.requestClear()
                Log.i(MainActivity.TAG, "queue: notification ${if (clearing) "Clear queue" else "Stop"}")
                if (clearing && !handedToWorker) {
                    // The worker already crossed its shutdown handoff (or is not running); no future worker step will
                    // consume this request, so this caller owns clearing the waiting entries now.
                    Queue.get(this).clearWaiting()
                }
                if (running && (!clearing || handedToWorker)) {
                    worker?.stop(RenderStops.Reason.PERSON)
                    notify(RenderStops.stopping(RenderStops.Reason.PERSON, ResourceWords(this)))
                } else {
                    stopSelf()
                }
                return START_NOT_STICKY
            }
        }
        goForeground(getString(R.string.render_preparing))
        if (thread?.isAlive == true) return START_NOT_STICKY
        val queue = Queue.get(this)
        val music = MusicFolder(this)
        val w = QueueWorker(
            queue, Reader.shared(this),
            sink = object : JournaledReadSink {
                override fun save(name: String, title: String, flac: ByteArray, publish: (() -> Unit) -> Boolean): String =
                    music.save(name, Flac.MIME, title, publish = publish) { it.write(flac) }.toString()
                override fun saveJournaled(name: String, title: String, flac: ByteArray, publication: Publication,
                                           publish: (() -> Unit) -> Boolean): String =
                    music.save(name, Flac.MIME, title, publish = publish, publication = publication) { it.write(flac) }.toString()
                override fun locate(publication: Publication): PublicationRecovery = music.locate(publication)
                override fun recoverPending(publication: Publication, uri: String): PublicationRecovery =
                    music.recoverPending(publication, uri)
            },
            seed = ggtts.Ggtts.DEFAULT_SEED,
            log = { Log.i(MainActivity.TAG, it) },
        )
        worker = w
        clearHandoff.begin()
        thread = Thread({
            var stopped: RenderStops.Reason? = null
            while (true) {
                notify(getString(R.string.render_waiting, queue.pending))
                when (val o = w.step()) {
                    QueueWorker.Outcome.Idle -> break
                    QueueWorker.Outcome.Busy -> Thread.sleep(15_000) // the engine is reading; try again
                    is QueueWorker.Outcome.Stopped -> { stopped = o.reason; break }
                    is QueueWorker.Outcome.Halted -> { stopped = o.reason; break }
                    is QueueWorker.Outcome.RecoveryBlocked -> {
                        Log.e(MainActivity.TAG, "queue: publication recovery blocked")
                        done("A saved read could not be verified. It remains queued; no duplicate was created.")
                        break
                    }
                    else -> Unit
                }
            }
            if (clearHandoff.finish()) queue.clearWaiting()
            stopForeground(STOP_FOREGROUND_REMOVE)
            // What happened stays in the shade after the service has gone.
            stopped?.let { done(RenderStops.stopped(it, queue.pending, ResourceWords(this))) }
            stopSelf()
        }, "ggtts-render-queue").apply { start() }
        return START_NOT_STICKY
    }

    /**
     * Android 15 and later: the service type's daily time ran out. Android allows only a few seconds to stop, and a
     * chunk can take far longer, so the service stops now: the worker is told to stop, and the entry it was rendering
     * is requeued when its chunk ends (or by recover() if the process goes first). A cut-off render saves nothing.
     */
    override fun onTimeout(startId: Int, fgsType: Int) {
        Log.i(MainActivity.TAG, "queue: the service's daily time ran out; stopping now, and the entry waits")
        worker?.stop()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    /** Android is critically short of memory: stop the render after its chunk; its entry waits. */
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (RenderStops.stopsFor(level) && thread?.isAlive == true) {
            Log.i(MainActivity.TAG, "queue: memory trim level $level; stopping the render")
            worker?.stop(RenderStops.Reason.MEMORY)
            notify(RenderStops.stopping(RenderStops.Reason.MEMORY, ResourceWords(this)))
        }
    }

    override fun onDestroy() {
        worker?.stop()
        super.onDestroy()
    }

    private fun goForeground(text: String) {
        val n = notification(text)
        when {
            Build.VERSION.SDK_INT >= 35 -> startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING)
            Build.VERSION.SDK_INT >= 29 -> startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            else -> startForeground(ID, n)
        }
    }

    private fun notify(text: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        runCatching { nm.notify(ID, notification(text)) }
    }

    /** A notification that stays after the service has stopped, saying why it stopped. */
    private fun done(text: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        runCatching { nm.notify(DONE_ID, notification(text, ongoing = false)) }
    }

    private fun notification(text: String, ongoing: Boolean = true): Notification {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.render_channel), NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            this, 2, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL) else
            @Suppress("DEPRECATION") Notification.Builder(this)
        b.setSmallIcon(android.R.drawable.ic_menu_save)
            .setContentTitle(getString(if (ongoing) R.string.render_notification_title else R.string.render_channel))
            .setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setOngoing(ongoing)
            .setAutoCancel(!ongoing)
            .setContentIntent(open)
        if (ongoing) {
            // Stop keeps the queue (it resumes when the app opens); Clear queue also drops what waits.
            fun action(label: String, act: String, code: Int) = Notification.Action.Builder(null, label,
                PendingIntent.getService(this, code, Intent(this, RenderService::class.java).setAction(act),
                    PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)).build()
            b.addAction(action(getString(R.string.notification_stop), ACTION_STOP, 10))
            b.addAction(action(getString(R.string.render_clear_queue), ACTION_CLEAR, 11))
        }
        return b.build()
    }

    companion object {
        private const val ID = 2
        private const val DONE_ID = 3
        private const val CHANNEL = "render-queue"
        const val ACTION_STOP = "ggtts.demo.QUEUE_STOP"
        const val ACTION_CLEAR = "ggtts.demo.QUEUE_CLEAR"

        /** Start working the queue, if anything waits. */
        fun start(context: Context) {
            if (Queue.get(context).pending == 0) return
            val i = Intent(context, RenderService::class.java)
            runCatching { if (Build.VERSION.SDK_INT >= 26) context.startForegroundService(i) else context.startService(i) }
                .onFailure { Log.w(MainActivity.TAG, "queue_service outcome=start_failure") }
        }
    }
}
