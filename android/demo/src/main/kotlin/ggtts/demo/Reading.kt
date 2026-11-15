// A read outlives the screen. One controller per process (Reader), whose view is relayed to whichever screen is
// open, and a foreground service (ReadingService) that keeps the process alive while a read runs, with a media session
// and a media notification: Pause or Play, and Stop, from the notification, the lock screen and a headset button
// (MediaControls). The loaded engine is the app's shared one (SharedEngines), kept between reads and released when
// Android asks for memory back while nothing is being read.
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
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
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

    /**
     * A memory trim at [level]: release [cache]'s engine if the level asks for it and nothing is reading, neither the
     * screen ([screenBusy]) nor anyone holding the engine (a system-engine read sets [EngineCache.busy] too). Returns
     * whether it was released.
     */
    fun trim(level: Int, screenBusy: Boolean, cache: EngineCache): Boolean =
        shouldRelease(level, screenBusy || cache.busy) && cache.release()
}

/**
 * When the reading service starts, goes foreground and stops, apart from Android, so the order can be tested. Main
 * thread only (setBusy and onStartCommand both run there).
 *
 * Android requires a service started with startForegroundService to call startForeground, and stopping it before
 * then crashes the app on Android 9 and later ("did not then call Service.startForeground()"). A read that fails fast
 * ends before the service's onStartCommand has run, so [readEnded] never stops a service that has not yet gone
 * foreground: [onCommand] always goes foreground first, then stops at once if the read has already ended.
 */
class ForegroundGate(private val startService: () -> Unit, private val stopService: () -> Unit) {
    private var wanted = false
    private var pending = false
    private var foreground = false

    fun readStarted() {
        wanted = true
        if (!pending && !foreground) {
            pending = true
            startService()
        }
    }

    fun readEnded() {
        wanted = false
        if (foreground) {
            foreground = false
            stopService()
        }
    }

    /**
     * The service's onStartCommand. The notification's Stop arrives as [ACTION_STOP] and ends the read through
     * [stopRead], the same call as the screen's Stop; any other command is the start, which always goes foreground.
     */
    fun onCommand(action: String?, stopRead: () -> Unit, goForeground: () -> Unit, control: (String) -> Unit = {}) {
        if (action in CONTROLS) {
            if (action == ACTION_STOP) stopRead() else control(action!!)
            // A Stop that reached a service with no read behind it (a stale notification) started it just for this:
            // stop it again, or it would stay.
            if (!foreground && !pending) stopService()
            return
        }
        pending = false
        goForeground()
        foreground = true
        // The read already ended (it failed fast): now that the service is foreground, it may stop.
        if (!wanted) readEnded()
    }

    /** The service was destroyed (stopped, or by the system). */
    fun onDestroy() {
        pending = false
        foreground = false
    }

    companion object {
        const val ACTION_STOP = "ggtts.demo.STOP"
        const val ACTION_PAUSE = "ggtts.demo.PAUSE"
        const val ACTION_PLAY = "ggtts.demo.PLAY"

        /** The commands that act on the read rather than start the service: Stop, Pause and Play. */
        val CONTROLS = setOf(ACTION_STOP, ACTION_PAUSE, ACTION_PLAY)
    }
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
            appContext?.let { val g = ReadingService.gate(it); if (busy) g.readStarted() else g.readEnded() }
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

    /**
     * Android asks for memory back: release the loaded engine if nothing is being read, by this screen or by the
     * system engine (the shared engine's own busy flag, which a system read sets too).
     */
    fun trim(context: Context, level: Int) {
        if (TrimPolicy.trim(level, busy, shared(context))) {
            Log.i(MainActivity.TAG, "memory trim level $level: released the loaded engine")
        }
    }
}

/**
 * Foreground while a read runs, so it goes on when the screen is left; its notification shows the read's progress
 * and has Stop. On Android 13 and later the screen asks for the notification permission at the first Speak; if it is
 * refused, the service still runs and the read goes on, but the notification (and its Stop) is hidden.
 */
class ReadingService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        gate(this).onCommand(
            intent?.action,
            stopRead = {
                Log.i(MainActivity.TAG, "notification: Stop")
                Reader.controller(this).stop()
            },
            goForeground = {
                val n = notification(this, Reader.lastStatus.ifEmpty { "Starting..." })
                if (Build.VERSION.SDK_INT >= 29) {
                    startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
                } else {
                    startForeground(ID, n)
                }
                Log.i(MainActivity.TAG, "reading service: foreground")
            },
            control = { action ->
                Log.i(MainActivity.TAG, "notification: ${MediaControls.apply(action, Reader.controller(this))}")
            },
        )
        return START_NOT_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        // The read's media session: the lock screen and the notification show its state, and a headset or Bluetooth
        // play/pause key (or `adb shell cmd media_session dispatch pause`) reaches it while the read is the active one.
        session = MediaSession(this, "ggtts-reading").apply {
            setCallback(object : MediaSession.Callback() {
                override fun onPause() = act(ForegroundGate.ACTION_PAUSE)
                override fun onPlay() = act(ForegroundGate.ACTION_PLAY)
                override fun onStop() = act(ForegroundGate.ACTION_STOP)
                private fun act(action: String) {
                    Log.i(MainActivity.TAG, "media session: ${MediaControls.apply(action, Reader.controller(this@ReadingService))}")
                }
            })
            setMetadata(MediaMetadata.Builder().putString(MediaMetadata.METADATA_KEY_TITLE, "ggtts is reading").build())
            isActive = true
        }
        publish(this)
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Reader.trim(this, level)
    }

    override fun onDestroy() {
        Log.i(MainActivity.TAG, "reading service: stopped")
        gate(this).onDestroy()
        session?.let { it.isActive = false; it.release() }
        session = null
        super.onDestroy()
    }

    companion object {
        private const val ID = 1
        private const val CHANNEL = "reading"
        private var gate: ForegroundGate? = null
        private var running = false
        /** The running service's media session, if it runs. Main thread only. */
        private var session: MediaSession? = null

        /** The read's state on the media session, from the controller (MediaControls.state). */
        private fun state(context: Context): MediaControls.State {
            val c = Reader.controller(context)
            return MediaControls.state(c.busy, c.paused)
        }

        /** Publishes the read's state to the media session, so the lock screen and a headset button see it. */
        private fun publish(context: Context) {
            val s = session ?: return
            val code = when (state(context)) {
                MediaControls.State.PLAYING -> PlaybackState.STATE_PLAYING
                MediaControls.State.PAUSED -> PlaybackState.STATE_PAUSED
                MediaControls.State.NONE -> PlaybackState.STATE_STOPPED
            }
            s.setPlaybackState(
                PlaybackState.Builder()
                    .setActions(PlaybackState.ACTION_PLAY or PlaybackState.ACTION_PAUSE or PlaybackState.ACTION_PLAY_PAUSE or PlaybackState.ACTION_STOP)
                    .setState(code, PlaybackState.PLAYBACK_POSITION_UNKNOWN, if (code == PlaybackState.STATE_PLAYING) 1f else 0f)
                    .build(),
            )
        }

        /** The process's one gate, with the application context. Main thread only. */
        fun gate(context: Context): ForegroundGate {
            gate?.let { return it }
            val app = context.applicationContext
            val i = Intent(app, ReadingService::class.java)
            return ForegroundGate(
                startService = {
                    running = true
                    if (Build.VERSION.SDK_INT >= 26) app.startForegroundService(i) else app.startService(i)
                },
                stopService = {
                    running = false
                    app.stopService(i)
                },
            ).also { gate = it }
        }

        fun update(context: Context, text: String) {
            if (!running) return
            publish(context)
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            runCatching { nm.notify(ID, notification(context, text)) }
        }

        private fun command(context: Context, action: String, code: Int) = PendingIntent.getService(
            context, code, Intent(context, ReadingService::class.java).setAction(action),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )

        private fun notification(context: Context, text: String): Notification {
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
                nm.createNotificationChannel(NotificationChannel(CHANNEL, "Reading", NotificationManager.IMPORTANCE_LOW))
            }
            val stop = command(context, ForegroundGate.ACTION_STOP, 0)
            val open = PendingIntent.getActivity(
                context, 0, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(context, CHANNEL) else
                @Suppress("DEPRECATION") Notification.Builder(context)
            // A media notification: Pause or Play, then Stop, on the shade and the lock screen. The slow-cores note is
            // in the text, since a media style shows no expanded text.
            val state = state(context)
            val toggle = MediaControls.toggleAction(state)
            b.setSmallIcon(android.R.drawable.ic_media_play)
                .setContentTitle("ggtts is reading")
                .setContentText("$text Off screen it continues several times slower; open the app to speed it up.")
                .setOngoing(true)
                .setVisibility(Notification.VISIBILITY_PUBLIC)
                .setContentIntent(open)
            val compact = mutableListOf<Int>()
            if (toggle != null) {
                val icon = if (state == MediaControls.State.PLAYING) android.R.drawable.ic_media_pause else android.R.drawable.ic_media_play
                b.addAction(Notification.Action.Builder(
                    android.graphics.drawable.Icon.createWithResource(context, icon), MediaControls.toggleLabel(state), command(context, toggle, 1),
                ).build())
                compact += 0
            }
            b.addAction(Notification.Action.Builder(
                android.graphics.drawable.Icon.createWithResource(context, android.R.drawable.ic_menu_close_clear_cancel), "Stop", stop,
            ).build())
            compact += compact.size
            val style = Notification.MediaStyle().setShowActionsInCompactView(*compact.toIntArray())
            session?.let { style.setMediaSession(it.sessionToken) }
            return b.setStyle(style).build()
        }
    }
}
