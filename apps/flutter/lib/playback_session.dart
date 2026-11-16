import 'dart:io';
import 'dart:typed_data';

import 'package:jni/jni.dart';
import 'package:mp_audio_stream/ggtts_audio_session.dart';

/// Synchronous per-read output. Android uses JNI; other platforms keep miniaudio.
class PlaybackSession {
  PlaybackSession._(this.id, this._native);
  final int id;
  final AudioSession? _native;
  static JClass get _host =>
      JClass.forName('dev/ggtts/ggtts_app/AndroidPlaybackBridge');

  static int _call(String method, String signature, List<Object?> args) {
    final host = _host;
    try {
      return host
          .staticMethodId(method, signature)
          .call(host, jlong.type, args);
    } finally {
      host.release();
    }
  }

  static int _callInt(String method, String signature, List<Object?> args) {
    final host = _host;
    try {
      return host.staticMethodId(method, signature).call(host, jint.type, args);
    } finally {
      host.release();
    }
  }

  static PlaybackSession? reserve() {
    if (!Platform.isAndroid) {
      final native = AudioSession.reserve();
      return native == null ? null : PlaybackSession._(native.id, native);
    }
    final id = _call('reserve', '()J', []);
    return id == 0 ? null : PlaybackSession._(id, null);
  }

  static PlaybackSession attach(int id) => Platform.isAndroid
      ? PlaybackSession._(id, null)
      : PlaybackSession._(id, AudioSession.attach(id));

  int init({required int sampleRate}) =>
      _native?.init(sampleRate: sampleRate) ??
      _callInt('init', '(JI)I', [id, sampleRate]);
  int pause() => _native?.pause() ?? _callInt('pause', '(J)I', [id]);
  int resume() => _native?.resume() ?? _callInt('resume', '(J)I', [id]);
  int stop() => _native?.stop() ?? _callInt('stop', '(J)I', [id]);
  int release() => _native?.release() ?? _callInt('release', '(J)I', [id]);
  int state() {
    final value = _native?.state() ?? _callInt('status', '(J)I', [id]);
    if (value == -2) throw StateError('Android audio was interrupted.');
    return value;
  }

  int softwareEmpty() => _native!.softwareEmpty();
  int submitted() => _call('submitted', '(J)J', [id]);
  int head() => _call('head', '(J)J', [id]);

  /// Bounded nonblocking writes, preserving partial-write offsets.
  int push(Float32List samples) {
    if (_native != null) return _native.push(samples);
    final deadline = Stopwatch()..start();
    var offset = 0;
    while (offset < samples.length) {
      if (deadline.elapsed > const Duration(seconds: 30)) {
        throw StateError('Android audio output stalled.');
      }
      final status = state();
      if (status == 3 || status < 0) return -3;
      if (status == 2) {
        sleep(const Duration(milliseconds: 10));
        continue;
      }
      final buffer = JFloatArray.of(samples.sublist(offset));
      int written;
      try {
        written = _callInt('push', '(J[F)I', [id, buffer]);
      } finally {
        buffer.release();
      }
      if (written == -3) throw StateError('Android audio was interrupted.');
      if (written < 0) throw StateError('Android audio write failed.');
      if (written == 0) {
        sleep(const Duration(milliseconds: 10));
        continue;
      }
      offset += written;
      deadline.reset();
    }
    return 0;
  }
}
