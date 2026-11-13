// Streaming playback apart from Android: one output device per stream, owned by a player thread, so rendering the
// next chunk never waits for playback and Stop is safe at any moment from any thread.
package ggtts.demo

import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicReference

/** One stream's output device, as the player sees it: an AudioTrack on a phone, a fake in a test. */
interface PcmSink {
    fun play()

    /** Blocks until [samples] are queued on the device, or returns early once [pause] is called. */
    fun write(samples: FloatArray)

    fun pause()
    fun flush()
    fun stop()
    fun release()
}

/**
 * Plays a stream of pieces through one [PcmSink] per stream, on that stream's player thread.
 *
 * The player thread owns its sink: it is the only thread that writes to it, stops it or releases it. [stop] touches the
 * sink only to pause and flush it (pausing also ends a blocking write early), under the stream's lock and only while
 * the sink is not yet released; it then tells the thread to finish and returns without waiting for it, so it is safe
 * on the main thread.
 */
class StreamPlayer(
    private val newSink: (rate: Int) -> PcmSink,
    /** How long the last buffer is given to play out after the final write. */
    private val drainMs: Long = 250,
) {
    private class Stream(val sink: PcmSink) {
        val queue = LinkedBlockingQueue<FloatArray>()
        val lock = Any()
        @Volatile var cut = false
        var released = false // guarded by lock
        lateinit var thread: Thread
    }

    private val end = FloatArray(0)
    private val current = AtomicReference<Stream?>(null)

    /** A new stream at [rate] Hz begins; a stream still playing is stopped first. */
    fun begin(rate: Int) {
        stop()
        val s = Stream(newSink(rate))
        s.sink.play()
        s.thread = Thread({ run(s) }, "ggtts-player")
        current.set(s)
        s.thread.start()
    }

    private fun run(s: Stream) {
        try {
            while (!s.cut) {
                val piece = s.queue.take()
                if (piece === end || s.cut) break
                s.sink.write(piece)
            }
            if (!s.cut) Thread.sleep(drainMs)
        } catch (_: InterruptedException) {
        } finally {
            synchronized(s.lock) {
                s.released = true
                runCatching { s.sink.stop() }
                s.sink.release()
            }
        }
    }

    /** Queue [samples] and return at once; dropped if no stream is playing. */
    fun play(samples: FloatArray) {
        if (samples.isNotEmpty()) current.get()?.queue?.put(samples)
    }

    /** Wait until everything queued has played, or until the stream is stopped. Never on the main thread. */
    fun finish() {
        val s = current.get() ?: return
        s.queue.put(end)
        s.thread.join()
        current.compareAndSet(s, null)
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
