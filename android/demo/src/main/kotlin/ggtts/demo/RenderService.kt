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

/** Where a finished render goes: Music/ggtts/ on a phone. Writes nothing it does not finish. Returns where it went. */
fun interface ReadSink {
    fun save(name: String, title: String, flac: ByteArray): String
}

class QueueWorker(
    private val queue: RenderQueue,
    private val cache: EngineCache,
    private val sink: ReadSink,
    private val seed: Long,
    private val log: (String) -> Unit = {},
    private val clock: () -> Long = System::nanoTime,
    /** Wall-clock milliseconds, for the saved file's name. */
    private val now: () -> Long = System::currentTimeMillis,
) {
    /** What one [step] did. */
    sealed class Outcome {
        object Idle : Outcome()
        /** The engine is reading for the screen or the system engine; try again later. The entry waits. */
        object Busy : Outcome()
        data class Done(val entry: RenderQueue.Entry, val where: String, val seconds: Double, val renderMs: Long) : Outcome()
        data class Failed(val entry: RenderQueue.Entry, val reason: String) : Outcome()
        data class Stopped(val entry: RenderQueue.Entry) : Outcome()
    }

    @Volatile
    private var stopRequested = false

    /** Stop the render in progress after its current chunk; its entry waits again. */
    fun stop() {
        stopRequested = true
    }

    /** Render the next waiting entry, if any, to the end, and settle it. */
    fun step(): Outcome {
        stopRequested = false
        val e = queue.take() ?: return Outcome.Idle
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
                if (stopRequested) return@renderStream false
                rate = r
                pieces += samples
                true
            }
            val renderMs = (clock() - t0) / 1_000_000
            if (!finished || stopRequested) {
                queue.requeue(e.id)
                ok = true
                log("queue: entry ${e.id} stopped after ${pieces.size} chunks, $renderMs ms; it waits again")
                return Outcome.Stopped(e)
            }
            val all = FloatArray(pieces.sumOf { it.size })
            var at = 0
            for (p in pieces) { p.copyInto(all, at); at += p.size }
            if (all.isEmpty() || rate <= 0) return fail(e, "the render made no audio")
            val where = sink.save(Storage.queuedName(e.request.engine, now()), e.title, Flac.encode(all, rate, e.title))
            queue.done(e.id, where)
            ok = true
            val seconds = all.size.toDouble() / rate
            log("queue: entry ${e.id} rendered %.2f s of audio in $renderMs ms (RTF %.2f), saved to $where".format(seconds, renderMs / 1000.0 / seconds))
            return Outcome.Done(e, where, seconds, renderMs)
        } catch (t: Throwable) {
            return fail(e, t.message ?: t.javaClass.simpleName)
        } finally {
            cache.done(failed = !ok)
        }
    }

    private fun fail(e: RenderQueue.Entry, reason: String): Outcome {
        queue.failed(e.id, reason)
        log("queue: entry ${e.id} failed: $reason")
        return Outcome.Failed(e, reason)
    }
}

/** The queue kept in app storage. */
class FileQueueStore(private val file: File) : QueueStore {
    override fun read(): String? = if (file.isFile) file.readText() else null
    override fun write(text: String) {
        file.parentFile?.mkdirs()
        val part = File(file.path + ".part")
        part.writeText(text)
        if (!part.renameTo(file)) { file.delete(); part.renameTo(file) }
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

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        goForeground("Preparing...")
        if (thread?.isAlive == true) return START_NOT_STICKY
        val queue = Queue.get(this)
        val music = MusicFolder(this)
        val w = QueueWorker(
            queue, Reader.shared(this),
            sink = { name, title, flac -> music.save(name, Flac.MIME, title) { it.write(flac) }.toString() },
            seed = ggtts.Ggtts.DEFAULT_SEED,
            log = { Log.i(MainActivity.TAG, it) },
        )
        worker = w
        thread = Thread({
            while (true) {
                notify("Rendering for later: ${queue.pending} waiting")
                when (val o = w.step()) {
                    QueueWorker.Outcome.Idle -> break
                    QueueWorker.Outcome.Busy -> Thread.sleep(15_000) // the engine is reading; try again
                    is QueueWorker.Outcome.Stopped -> break
                    else -> Unit
                }
            }
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }, "ggtts-render-queue").apply { start() }
        return START_NOT_STICKY
    }

    /** Android 15 and later: the service type's daily time ran out. Stop cleanly; the entry waits again. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        Log.i(MainActivity.TAG, "queue: the service's daily time ran out; stopping, and the entry waits")
        worker?.stop()
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

    private fun notification(text: String): Notification {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Rendering for later", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            this, 2, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL) else
            @Suppress("DEPRECATION") Notification.Builder(this)
        return b.setSmallIcon(android.R.drawable.ic_menu_save)
            .setContentTitle("ggtts is rendering for later")
            .setContentText(text)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    companion object {
        private const val ID = 2
        private const val CHANNEL = "render-queue"

        /** Start working the queue, if anything waits. */
        fun start(context: Context) {
            if (Queue.get(context).pending == 0) return
            val i = Intent(context, RenderService::class.java)
            runCatching { if (Build.VERSION.SDK_INT >= 26) context.startForegroundService(i) else context.startService(i) }
                .onFailure { Log.w(MainActivity.TAG, "queue: could not start the render service: $it") }
        }
    }
}
