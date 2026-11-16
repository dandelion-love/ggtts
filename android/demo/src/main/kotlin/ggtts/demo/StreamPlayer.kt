// Streaming playback apart from Android: one output device per stream, owned by a player thread, so rendering the
// next chunk never waits for playback and Stop is safe at any moment from any thread. Pause holds the device with what
// it has buffered and keeps every queued piece; Resume plays on from there.
package ggtts.demo

import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicReference

/** The device refused a write with [code] (an AudioTrack error), so the stream did not play to its end. */
class PlaybackFailed(val code: Int) : RuntimeException("the audio device refused a write (error $code)")
class PlaybackUnconfirmed : RuntimeException("the audio device did not confirm playback completion")

/** One stream's output device, as the player sees it: an AudioTrack on a phone, a fake in a test. */
interface PcmSink {
    fun play()

    /**
     * Queue what fits of `samples[offset, offset + count)` without blocking, and return how many were taken (0 when the
     * device's buffer is full, as it stays once the device is paused), or a negative error code.
     */
    fun write(samples: FloatArray, offset: Int, count: Int): Int

    /** Unsigned 32-bit playback-head frame counter, before stop or flush. */
    fun playbackHead(): Long
    fun pause()
    fun flush()
    fun stop()
    fun release()
}

/** Tracks a fresh AudioTrack's unsigned 32-bit head without treating a reset as progress. */
internal class PlaybackHeadCounter {
    private var previous = 0L
    var played = 0L
        private set

    fun observe(head: Long, submitted: Long) {
        check(head in 0..0xffffffffL)
        val delta = (head - previous) and 0xffffffffL
        check(delta <= submitted - played)
        played += delta
        previous = head
    }
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
    /** How long the player waits before trying again when the device's buffer is full. */
    private val waitMs: Long = 5,
    /** A non-paused device that stops advancing cannot confirm playback. */
    private val stallMs: Long = 30_000,
) {
    private class Stream(val sink: PcmSink) {
        val queue = LinkedBlockingQueue<FloatArray>()
        val lock = Any()
        @Volatile var cut = false
        var released = false // guarded by lock
        var closing = false // guarded by lock
        /** The code of a write the device refused, if one was; the stream ended there. */
        @Volatile var refused: Int? = null
        @Volatile var failed = false
        @Volatile var completed = false
        lateinit var thread: Thread
    }

    private val end = FloatArray(0)
    private val current = AtomicReference<Stream?>(null)

    /** Whether playback is held ([pause]). It outlives a stream: a pause before the first piece holds the next one. */
    @Volatile
    var paused = false
        private set

    /**
     * A new stream at [rate] Hz begins; a stream still playing is stopped first. If the device cannot start (its
     * `play` throws, or the player thread cannot start), the device is released and the throw passes to the caller.
     */
    fun begin(rate: Int) {
        cut()
        val s = Stream(newSink(rate))
        try {
            s.sink.play()
            if (paused) s.sink.pause()
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
        var submitted = 0L
        val head = PlaybackHeadCounter()
        fun observe() = head.observe(s.sink.playbackHead(), submitted)
        try {
            check(s.sink.playbackHead() == 0L)
            while (!s.cut && !s.failed) {
                val piece = s.queue.take()
                if (piece === end || s.cut) break
                var at = 0
                var lastWriteProgress = System.nanoTime()
                while (at < piece.size && !s.cut && !s.failed) {
                    val n = s.sink.write(piece, at, piece.size - at)
                    if (n < 0) {
                        s.refused = n
                        return
                    }
                    check(n <= piece.size - at)
                    submitted += n.toLong()
                    at += n
                    observe()
                    if (n > 0 || paused) {
                        lastWriteProgress = System.nanoTime()
                    } else if (System.nanoTime() - lastWriteProgress >= stallMs * 1_000_000L) {
                        s.failed = true
                    }
                    if (n == 0 && !s.failed) Thread.sleep(waitMs)
                }
            }
            var lastProgress = System.nanoTime()
            var previousPlayed = head.played
            while (!s.cut && !s.failed) {
                synchronized(s.lock) {
                    if (!paused && !s.cut && !s.failed) {
                        observe()
                        if (head.played == submitted && submitted > 0) s.completed = true
                        if (head.played != previousPlayed) {
                            previousPlayed = head.played
                            lastProgress = System.nanoTime()
                        } else if (System.nanoTime() - lastProgress >= stallMs * 1_000_000L) {
                            s.failed = true
                        }
                    } else if (paused) {
                        lastProgress = System.nanoTime()
                    }
                }
                if (s.completed) break
                Thread.sleep(waitMs)
            }
        } catch (_: InterruptedException) {
            s.failed = true
        } catch (_: Throwable) {
            s.failed = true
        } finally {
            try { release(s) } catch (_: Throwable) { s.failed = true }
        }
    }

    /** Queue [samples] and return at once; dropped if no stream is playing. */
    fun play(samples: FloatArray) {
        if (samples.isEmpty()) return
        val s = current.get() ?: return
        synchronized(s.lock) {
            if (s.closing) {
                s.failed = true
                s.thread.interrupt()
            } else if (!s.cut && !s.failed) {
                s.queue.put(samples)
            }
        }
    }

    /**
     * Wait until everything queued has played, or until the stream is stopped. Never on the main thread. Throws
     * [PlaybackFailed] if the device refused a write, since the stream then did not play to its end.
     */
    fun finish() { if (!finishConfirmed()) throw PlaybackUnconfirmed() }

    /**
     * True only when this stream's playback head reached every submitted frame.
     * This says nothing about whether a person heard the output. False on stop,
     * replacement, head failure, or no stream. Call off the main thread.
     */
    fun finishConfirmed(): Boolean {
        val s = current.get() ?: return false
        synchronized(s.lock) {
            if (s.closing) return false
            s.closing = true
            s.queue.put(end)
        }
        s.thread.join()
        val stillCurrent = current.compareAndSet(s, null)
        s.refused?.let { throw PlaybackFailed(it) }
        return stillCurrent && !s.cut && !s.failed && s.completed
    }

    /**
     * Hold playback: the device pauses with what it has buffered, pieces still arrive and wait in the queue (rendering
     * goes on), and the stream does not end until [resume] or [stop]. Safe from any thread.
     */
    fun pause() {
        paused = true
        val s = current.get() ?: return
        synchronized(s.lock) { if (!s.released) try { s.sink.pause() } catch (_: Throwable) { s.failed = true; s.thread.interrupt() } }
    }

    /** Play on from where [pause] held it, nothing skipped. Safe from any thread. */
    fun resume() {
        paused = false
        val s = current.get() ?: return
        synchronized(s.lock) { if (!s.released) try { s.sink.play() } catch (_: Throwable) { s.failed = true; s.thread.interrupt() } }
    }

    /**
     * Drop what is queued and stop playing now, without waiting for the player thread, and end a pause. Safe from any
     * thread.
     */
    fun stop() {
        paused = false
        cut()
    }

    /** The stream playing now, if any, cut off; a pause stays (a new stream begun after it starts held). */
    private fun cut() {
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
