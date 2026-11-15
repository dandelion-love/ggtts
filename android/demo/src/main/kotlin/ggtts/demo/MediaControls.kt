// What the read's media controls show and do, apart from Android: the state a media session publishes, the action the
// notification's first button sends, and what a play/pause key (a headset, a Bluetooth remote, `cmd media_session
// dispatch`) does. ReadingService maps these onto MediaSession and Notification.MediaStyle; the tests drive them here.
package ggtts.demo

object MediaControls {
    /** The read as a media session states it. */
    enum class State { NONE, PLAYING, PAUSED }

    /** No read: none. A read whose playback is held: paused. Otherwise playing (preparing counts: Pause holds it). */
    fun state(busy: Boolean, paused: Boolean): State = when {
        !busy -> State.NONE
        paused -> State.PAUSED
        else -> State.PLAYING
    }

    /** The notification's first button: Pause while playing, Play while paused, none without a read. */
    fun toggleAction(state: State): String? = when (state) {
        State.PLAYING -> ForegroundGate.ACTION_PAUSE
        State.PAUSED -> ForegroundGate.ACTION_PLAY
        State.NONE -> null
    }

    /** The button's label, for [toggleAction]. */
    fun toggleLabel(state: State): String? = when (state) {
        State.PLAYING -> "Pause"
        State.PAUSED -> "Play"
        State.NONE -> null
    }

    /**
     * A command for the read: Pause or Play reaches [DemoController.pause] or [DemoController.resume], Stop reaches
     * [DemoController.stop]. Anything else does nothing. Returns what it did, for the log.
     */
    fun apply(action: String, controller: DemoController): String? = when (action) {
        ForegroundGate.ACTION_PAUSE -> { controller.pause(); "pause" }
        ForegroundGate.ACTION_PLAY -> { controller.resume(); "play" }
        ForegroundGate.ACTION_STOP -> { controller.stop(); "stop" }
        else -> null
    }
}
