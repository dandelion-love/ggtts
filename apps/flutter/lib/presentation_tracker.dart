/// Result of accounting for a controlled output stream.
enum PresentationState {
  pending,
  presented,
  stopped,
  stalled,
  timedOut,
  invalid,
}

/// Accounts for a *fresh* mono AudioTrack playback head and accepted writes.
///
/// This is not connected to any playback backend. A caller must serialize calls,
/// provide elapsed monotonic time, report only frames actually accepted by that
/// same track, observe head position without resetting/flush/replacing the
/// track, and call [seal] after the final write. A backend must stop and release
/// on every terminal non-success. Presentation does not establish audibility.
///
/// Sessions are capped below 2^32 frames: with an initially zero head this
/// makes a backwards counter unambiguously a reset rather than a wrap.
class PresentationTracker {
  PresentationTracker({
    required int initialHead,
    required this.stallAfter,
    required this.timeoutAfter,
  }) {
    if (initialHead != 0 ||
        stallAfter <= Duration.zero ||
        timeoutAfter <= Duration.zero) {
      _state = PresentationState.invalid;
    }
  }

  final Duration stallAfter;
  final Duration timeoutAfter;
  static const int maxFrames = 0xffffffff;
  int _submitted = 0;
  int _presented = 0;
  Duration _lastTime = Duration.zero;
  Duration _lastProgress = Duration.zero;
  Duration? _sealedAt;
  PresentationState _state = PresentationState.pending;

  int get submitted => _submitted;
  int get presented => _presented;
  PresentationState get state => _state;

  /// Record an accepted write before observing a head that includes it.
  void accepted(int frames) {
    if (_state != PresentationState.pending ||
        _sealedAt != null ||
        frames < 0 ||
        frames > maxFrames - _submitted) {
      _state = PresentationState.invalid;
      return;
    }
    _submitted += frames;
  }

  void seal(Duration now) {
    if (!_advanceTime(now) || _sealedAt != null || _submitted == 0) {
      _state = PresentationState.invalid;
      return;
    }
    _sealedAt = now;
    _lastProgress = now;
  }

  /// Head is an unsigned 32-bit frame count from the fresh output track.
  /// Pausing suppresses the progress-stall timer, not the absolute deadline.
  PresentationState observe(int head, Duration now, {bool paused = false}) {
    if (_state != PresentationState.pending) return _state;
    if (!_advanceTime(now) ||
        head < _presented ||
        head > _submitted ||
        head < 0 ||
        head > maxFrames) {
      return _state = PresentationState.invalid;
    }
    if (head > _presented || paused) _lastProgress = now;
    _presented = head;
    final sealed = _sealedAt;
    if (sealed == null) return _state;
    if (now - sealed >= timeoutAfter) {
      return _state = PresentationState.timedOut;
    }
    if (!paused && now - _lastProgress >= stallAfter) {
      return _state = PresentationState.stalled;
    }
    if (!paused && _presented == _submitted) {
      return _state = PresentationState.presented;
    }
    return _state;
  }

  void stop() {
    if (_state == PresentationState.pending) _state = PresentationState.stopped;
  }

  bool _advanceTime(Duration now) {
    if (_state != PresentationState.pending || now < _lastTime) return false;
    _lastTime = now;
    return true;
  }
}
