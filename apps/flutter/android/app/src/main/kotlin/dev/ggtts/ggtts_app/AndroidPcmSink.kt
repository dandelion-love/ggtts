package dev.ggtts.ggtts_app

import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRouting
import android.os.Handler
import android.os.Looper
import android.media.AudioTrack

/**
 * Real AudioTrack adapter; not yet connected to Flutter's rendering path.
 * Route checks sample the active route only when queried: they do not detect
 * transient reroutes between calls, focus loss or prove acoustic output.
 */
internal class AndroidPcmSink(rate: Int, onRouteChange: () -> Unit = {}) : ControlledPcmSink {
    private val track: AudioTrack
    private var route: Int? = null
    private val routeBaseline = RouteBaseline()
    private val routeListener: AudioRouting.OnRoutingChangedListener

    init {
        val minimum = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        require(minimum > 0)
        @Suppress("DEPRECATION")
        track = AudioTrack(
            AudioManager.STREAM_MUSIC, rate, AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT, minimum.coerceAtLeast(8192), AudioTrack.MODE_STREAM
        )
        if (track.state != AudioTrack.STATE_INITIALIZED) {
            track.release()
            error("AudioTrack initialization failed")
        }
        routeListener = AudioRouting.OnRoutingChangedListener {
            if (!routeBaseline.initialEvent(track.routedDevice?.id)) onRouteChange()
        }
        track.addOnRoutingChangedListener(routeListener, Handler(Looper.getMainLooper()))
    }

    override fun play() { track.play() }
    override fun pause() { track.pause() }
    override fun write(samples: FloatArray, offset: Int, count: Int): Int {
        routeBaseline.beforeWrite()
        return track.write(samples, offset, count, AudioTrack.WRITE_NON_BLOCKING)
    }

    override fun head(): Long {
        val current = track.routedDevice?.id
        // A route is unavailable before play. Callers cannot accept a later
        // route change as proof that earlier frames reached the same device.
        if (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
            check(current != null)
            if (route == null) route = current else check(route == current)
        }
        return track.playbackHeadPosition.toLong() and 0xffffffffL
    }

    override fun flush() { track.flush() }
    override fun release() { track.removeOnRoutingChangedListener(routeListener); track.release() }
}
