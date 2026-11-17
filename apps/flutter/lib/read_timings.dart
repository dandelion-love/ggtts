import 'dart:math' as math;

/// Bounded, text-free timing of software stages. A callback gap is elapsed time
/// between callbacks, not an estimate of audible silence or hardware playback.
class ReadTimings {
  ReadTimings({
    int Function()? nowMicros,
    void Function(String)? sink,
    this.maxPieces = 128,
  }) : _clock = Stopwatch()..start(),
       _nowOverride = nowMicros,
       _sink = sink ?? print;

  final Stopwatch _clock;
  final int Function()? _nowOverride;
  final void Function(String) _sink;
  final int maxPieces;
  int _pieces = 0;
  int? _lastCallbackEnd;

  int _now() => _nowOverride?.call() ?? _clock.elapsedMicroseconds;
  int _ms(int us) => math.min(2147483647, math.max(0, us ~/ 1000));

  void _emit(String text) {
    try {
      _sink(text);
    } catch (_) {
      /* Diagnostics must never break reading. */
    }
  }

  T load<T>(T Function() action) {
    final start = _now();
    try {
      return action();
    } finally {
      _emit('ggtts-read event=load elapsed_ms=${_ms(_now() - start)}');
    }
  }

  void beginStream() {
    _lastCallbackEnd = _now();
  }

  void piece({
    required int samples,
    required int rate,
    required void Function() setup,
    required void Function() save,
    required void Function() handoff,
  }) {
    final entry = _now();
    final wait = entry - (_lastCallbackEnd ?? entry);
    var setupUs = 0, saveUs = 0, handoffUs = 0;
    var outcome = 'failed';
    try {
      var start = _now();
      try {
        setup();
      } finally {
        setupUs = _now() - start;
      }
      start = _now();
      try {
        save();
      } finally {
        saveUs = _now() - start;
      }
      start = _now();
      try {
        handoff();
      } finally {
        handoffUs = _now() - start;
      }
      outcome = 'ok';
    } finally {
      _pieces++;
      if (_pieces <= maxPieces) {
        final audioMs = rate > 0 ? _ms(samples * 1000000 ~/ rate) : 0;
        _emit(
          'ggtts-read event=piece index=$_pieces outcome=$outcome '
          'wait_ms=${_ms(wait)} setup_ms=${_ms(setupUs)} '
          'save_ms=${_ms(saveUs)} handoff_ms=${_ms(handoffUs)} audio_ms=$audioMs',
        );
      } else if (_pieces == maxPieces + 1) {
        _emit('ggtts-read event=piece_log_limit');
      }
      _lastCallbackEnd = _now();
    }
  }

  void streamEnded() => _emit(
    'ggtts-read event=stream_end pieces=${math.min(_pieces, 2147483647)}',
  );

  T drain<T>(T Function() action) {
    final start = _now();
    try {
      return action();
    } finally {
      _emit('ggtts-read event=drain elapsed_ms=${_ms(_now() - start)}');
    }
  }
}
