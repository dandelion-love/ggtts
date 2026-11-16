package dev.ggtts.ggtts_app

import android.content.Context
import android.media.AudioManager

/** Synchronous JNI facade; all failures are fail-closed, never confirmation. */
internal object AndroidPlaybackBridge {
    @Volatile private var coordinator: PlaybackCoordinator? = null

    @JvmStatic @Synchronized fun configure(context: Context) {
        if (coordinator != null) return
        val audio = context.applicationContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val focus = object : PlaybackFocus {
            private var listener: AudioManager.OnAudioFocusChangeListener? = null
            override fun acquire(onLoss: () -> Unit): Boolean {
                val l = AudioManager.OnAudioFocusChangeListener { if (it != AudioManager.AUDIOFOCUS_GAIN) onLoss() }
                @Suppress("DEPRECATION")
                val result = audio.requestAudioFocus(l, AudioManager.STREAM_MUSIC, AudioManager.AUDIOFOCUS_GAIN)
                if (result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED) listener = l
                return result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
            }
            override fun release() {
                val l = listener
                listener = null
                if (l != null) {
                    @Suppress("DEPRECATION")
                    audio.abandonAudioFocus(l)
                }
            }
        }
        coordinator = PlaybackCoordinator(focus) { rate, onRoute -> AndroidPcmSink(rate, onRoute) }
    }

    @JvmStatic fun interruptActive() { coordinator?.interruptActive() }
    @JvmStatic fun reserve(): Long = coordinator?.reserve() ?: 0
    @JvmStatic fun init(id: Long, rate: Int): Int = coordinator?.init(id, rate) ?: -1
    @JvmStatic fun push(id: Long, samples: FloatArray): Int = coordinator?.push(id, samples) ?: -3
    @JvmStatic fun submitted(id: Long): Long = coordinator?.submitted(id) ?: -1
    @JvmStatic fun head(id: Long): Long = coordinator?.head(id) ?: -1
    @JvmStatic fun status(id: Long): Int = coordinator?.status(id) ?: -1
    @JvmStatic fun pause(id: Long): Int = coordinator?.pause(id) ?: -1
    @JvmStatic fun resume(id: Long): Int = coordinator?.resume(id) ?: -1
    @JvmStatic fun stop(id: Long): Int = coordinator?.stop(id) ?: -1
    @JvmStatic fun release(id: Long): Int = coordinator?.release(id) ?: -1
}
