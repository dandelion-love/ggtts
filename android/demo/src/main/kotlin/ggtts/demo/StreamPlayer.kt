// Streaming playback apart from Android: one output device per stream, owned by a player thread, so rendering the
// next chunk never waits for playback and Stop is safe at any moment from any thread.
package ggtts.demo

import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicReference

/** The device refused a write with [code] (an AudioTrack error), so the stream did not play to its end. */
class PlaybackFailed(val code: Int) : RuntimeException("the audio device refused a write (error $code)")

/** One stream's output device, as the player sees it: an AudioTrack on a phone, a fake in a test. */
interface PcmSink {
    fun play()

    /**
     * Queue what fits of `samples[offset, offset + count)` without blocking, and return how many were taken (0 when the
     * device's buffer is full, as it stays once the device is paused), or a negative error code.
     */
    fun write(samples: FloatArray, offset: Int, count: Int): Int

    fun pause()
    fun flush()
    fun stop()
    fun release()
}

/**
 * Plays a stream of pieces through one [PcmSink] per stream, on that stream's player thread.
 *
 * The player thread owns its sink: it is the only thread that writes to it, stops it or releases it. [stop] touches the
 * sink only to pause and flush it, under the stream's lock and only while the sink is not yet released; it then tells
 * the thread to finish and returns without waiting for it, so it is safe on the main thread.
 *
 * Writes never block: a paused device takes nothing, and a blocking write begun just after a Stop's pause would wait
 * for good (pause ends only a write already in progress). The player writes what fits, checks for a Stop, and waits a
 * few milliseconds when the buffer is full, so a Stop at any moment ends the thread and releases the device.
 */
class StreamPlayer(
    private val newSink: (rate: Int) -> PcmSink,
    /** How long the last buffer is given to play out after the final write. */
    private val drainMs: Long = 250,
    /** How long the player waits before trying again when the device's buffer is full. */
    private val waitMs: Long = 5,
) {
    private class Stream(val sink: PcmSink) {
        val queue = LinkedBlockingQueue<FloatArray>()
        val lock = Any()
        @Volatile var cut = false
        var released = false // guarded by lock
        /** The code of a write the device refused, if one was; the stream ended there. */
        @Volatile var refused: Int? = null
        lateinit var thread: Thread
    }

    private val end = FloatArray(0)
    private val current = AtomicReference<Stream?>(null)

    /**
     * A new stream at [rate] Hz begins; a stream still playing is stopped first. If the device cannot start (its
     * `play` throws, or the player thread cannot start), the device is released and the throw passes to the caller.
     */
    fun begin(rate: Int) {
        stop()
        val s = Stream(newSink(rate))
        try {
            s.sink.play()
            s.thread = Thread({ run(s) }, "ggtts-player")
            current.set(s)
            s.thread.start()
        } catch (e: Throwable) {
            current.compareAndSet(s, null)
            release(s)
            throw e
        }
    }

    /** Stop and release [s]'s device, once. */
    private fun release(s: Stream) {
        synchronized(s.lock) {
            if (s.released) return
            s.released = true
            runCatching { s.sink.stop() }
            s.sink.release()
        }
    }

    private fun run(s: Stream) {
        try {
            while (!s.cut) {
                val piece = s.queue.take()
                if (piece === end || s.cut) break
                var at = 0
                while (at < piece.size && !s.cut) {
                    val n = s.sink.write(piece, at, piece.size - at)
                    if (n < 0) {
                        s.refused = n // the device refused the write: end the stream here, and finish() reports it
                        return
                    }
                    if (n == 0) Thread.sleep(waitMs) else at += n
                }
            }
            if (!s.cut) Thread.sleep(drainMs)
        } catch (_: InterruptedException) {
        } finally {
            release(s)
        }
    }

    /** Queue [samples] and return at once; dropped if no stream is playing. */
    fun play(samples: FloatArray) {
        if (samples.isNotEmpty()) current.get()?.queue?.put(samples)
    }

    /**
     * Wait until everything queued has played, or until the stream is stopped. Never on the main thread. Throws
     * [PlaybackFailed] if the device refused a write, since the stream then did not play to its end.
     */
    fun finish() {
        val s = current.get() ?: return
        s.queue.put(end)
        s.thread.join()
        current.compareAndSet(s, null)
        s.refused?.let { throw PlaybackFailed(it) }
    }

    /** Drop what is queued and stop playing now, without waiting for the player thread. Safe from any thread. */
    fun stop() {
        val s = current.getAndSet(null) ?: return
        s.cut = true
        s.queue.clear()
        s.queue.offer(end)
        synchronized(s.lock) {
            if (!s.released) {
                runCatching {
                    s.sink.pause()
                    s.sink.flush()
                }
            }
        }
        s.thread.interrupt()
    }
}
