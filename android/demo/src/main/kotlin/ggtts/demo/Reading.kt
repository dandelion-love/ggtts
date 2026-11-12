// A read outlives the screen. One controller per process (Reader), whose view is relayed to whichever screen is
// open, and a foreground service (ReadingService) that keeps the process alive while a read runs, with a notification
// that shows progress and has a Stop action. The loaded engine is the app's shared one (SharedEngines), kept between
// reads and released when Android asks for memory back while nothing is being read.
package ggtts.demo

import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import java.io.File
import java.util.concurrent.Executors

/** When to give a loaded engine back: never during a read; when Android is critically short, or the app is in the background. */
object TrimPolicy {
    const val RUNNING_CRITICAL = 15 // ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL
    const val BACKGROUND = 40 // ComponentCallbacks2.TRIM_MEMORY_BACKGROUND

    fun shouldRelease(level: Int, busy: Boolean): Boolean = !busy && (level == RUNNING_CRITICAL || level >= BACKGROUND)
}

/** The process's one reader. Main thread only. */
object Reader {
    private var controller: DemoController? = null
    private var appContext: Context? = null

    /** The open screen, if any; the read reports to it, and to the notification. */
    var screen: DemoView? = null
        set(value) {
            field = value
            value?.let {
                it.setBusy(busy)
                if (lastError) it.showError(lastStatus) else if (lastStatus.isNotEmpty()) it.showStatus(lastStatus)
            }
        }

    var lastStatus = ""
        private set
    private var lastError = false
    val busy get() = controller?.busy == true

    private val relay = object : DemoView {
        override fun setBusy(busy: Boolean) {
            screen?.setBusy(busy)
            appContext?.let { if (busy) ReadingService.start(it) else ReadingService.finish(it) }
        }

        override fun showStatus(text: String) {
            lastStatus = text
            lastError = false
            screen?.showStatus(text)
            appContext?.let { ReadingService.update(it, text) }
        }

        override fun showError(text: String) {
            lastStatus = text
            lastError = true
            screen?.showError(text)
            appContext?.let { ReadingService.update(it, text) }
        }
    }

    /** The controller, created on first use with the application context. */
    fun controller(context: Context): DemoController {
        controller?.let { return it }
        val app = context.applicationContext
        appContext = app
        val worker = Executors.newSingleThreadExecutor()
        val handler = Handler(Looper.getMainLooper())
        val log: (String) -> Unit = { Log.i(MainActivity.TAG, it) }
        val c = DemoController(
            binding = GgttsBinding(File(app.filesDir, "ggtts-cache").path),
            view = relay,
            audio = DeviceAudio(File(app.getExternalFilesDir(null) ?: app.filesDir, "out")),
            background = { worker.execute(it) },
            main = { handler.post(it) },
            log = log,
            memory = {
                val info = ActivityManager.MemoryInfo()
                (app.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
                info.availMem to info.lowMemory
            },
            onSpoke = { VoiceStore(GgttsTtsService.prefs(app)).save(it) },
            cache = shared(app),
        )
        controller = c
        return c
    }

    /** The engine cache the screen and the system engine share. */
    fun shared(context: Context): EngineCache {
        val app = context.applicationContext
        return SharedEngines.get { EngineCache(GgttsBinding(File(app.filesDir, "ggtts-cache").path)) { Log.i(MainActivity.TAG, it) } }
    }

    /** Android asks for memory back: release the loaded engine if nothing is being read. */
    fun trim(context: Context, level: Int) {
        if (TrimPolicy.shouldRelease(level, busy)) {
            Log.i(MainActivity.TAG, "memory trim level $level: releasing the loaded engine")
            shared(context).release()
        }
    }
}

/**
 * Foreground while a read runs, so it goes on when the screen is left; its notification shows the read's progress
 * and has Stop. The app does not ask for the notification permission (Android 13 and later): without it the service
 * still runs, and the notification shows once the user allows notifications for the app in its settings.
 */
class ReadingService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            Log.i(MainActivity.TAG, "notification: Stop")
            Reader.controller(this).stop()
            return START_NOT_STICKY
        }
        val n = notification(this, Reader.lastStatus.ifEmpty { "Starting..." })
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        } else {
            startForeground(ID, n)
        }
        Log.i(MainActivity.TAG, "reading service: foreground")
        return START_NOT_STICKY
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Reader.trim(this, level)
    }

    override fun onDestroy() {
        Log.i(MainActivity.TAG, "reading service: stopped")
        super.onDestroy()
    }

    companion object {
        private const val ID = 1
        private const val CHANNEL = "reading"
        const val ACTION_STOP = "ggtts.demo.STOP"
        private var running = false

        fun start(context: Context) {
            running = true
            val i = Intent(context, ReadingService::class.java)
            if (Build.VERSION.SDK_INT >= 26) context.startForegroundService(i) else context.startService(i)
        }

        fun update(context: Context, text: String) {
            if (!running) return
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            runCatching { nm.notify(ID, notification(context, text)) }
        }

        fun finish(context: Context) {
            running = false
            context.stopService(Intent(context, ReadingService::class.java))
        }

        private fun notification(context: Context, text: String): Notification {
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
                nm.createNotificationChannel(NotificationChannel(CHANNEL, "Reading", NotificationManager.IMPORTANCE_LOW))
            }
            val stop = PendingIntent.getService(
                context, 0, Intent(context, ReadingService::class.java).setAction(ACTION_STOP),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            val open = PendingIntent.getActivity(
                context, 0, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(context, CHANNEL) else
                @Suppress("DEPRECATION") Notification.Builder(context)
            val full = "$text\nOff screen, Android runs the read on the phone's slow cores: it continues, several times slower. Open the app to speed it up."
            return b.setSmallIcon(android.R.drawable.ic_media_play)
                .setContentTitle("ggtts is reading")
                .setContentText(text)
                .setStyle(Notification.BigTextStyle().bigText(full))
                .setOngoing(true)
                .setContentIntent(open)
                .addAction(Notification.Action.Builder(null, "Stop", stop).build())
                .build()
        }
    }
}
