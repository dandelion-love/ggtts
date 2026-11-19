import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

typedef _NewC = Uint64 Function();
typedef _NewD = int Function();
typedef _OpC = Int64 Function(Uint64);
typedef _OpD = int Function(int);
typedef _InitC = Int64 Function(Uint64, Int64, Int64, Int64, Int64, Int64);
typedef _InitD = int Function(int, int, int, int, int, int);
typedef _PushC = Int64 Function(Uint64, Pointer<Float>, Int64);
typedef _PushD = int Function(int, Pointer<Float>, int);

/// One reservation for one read. Release even after stop-before-init.
/// Device transitions may block; invoke away from the UI isolate.
class AudioSession {
  AudioSession._(DynamicLibrary lib, this.id)
      : _init = lib.lookupFunction<_InitC, _InitD>('ma_stream_session_init'),
        _pause = lib.lookupFunction<_OpC, _OpD>('ma_stream_session_pause'),
        _resume = lib.lookupFunction<_OpC, _OpD>('ma_stream_session_resume'),
        _stop = lib.lookupFunction<_OpC, _OpD>('ma_stream_session_stop'),
        _release = lib.lookupFunction<_OpC, _OpD>('ma_stream_session_uninit'),
        _empty =
            lib.lookupFunction<_OpC, _OpD>('ma_stream_session_software_empty'),
        _state = lib.lookupFunction<_OpC, _OpD>("ma_stream_session_state"),
        _queued =
            lib.lookupFunction<_OpC, _OpD>('ma_stream_session_queued_frames'),
        _push = lib.lookupFunction<_PushC, _PushD>('ma_stream_session_push');

  static DynamicLibrary _library() {
    if (Platform.isLinux || Platform.isAndroid) {
      return DynamicLibrary.open('libmp_audio_stream.so');
    }
    if (Platform.isWindows) return DynamicLibrary.open('mp_audio_stream.dll');
    throw UnsupportedError(
        'Controlled audio is not supported on this platform.');
  }

  static AudioSession? reserve() {
    final lib = _library();
    final id = lib.lookupFunction<_NewC, _NewD>('ma_stream_session_new')();
    return id == 0 ? null : AudioSession._(lib, id);
  }

  static AudioSession attach(int id) {
    if (id <= 0) throw ArgumentError.value(id, 'id');
    return AudioSession._(_library(), id);
  }

  final int id;
  final _InitD _init;
  final _OpD _pause, _resume, _stop, _release, _empty, _state, _queued;
  final _PushD _push;

  int init({required int sampleRate, int channels = 1, int bufferMs = 5000}) =>
      _init(id, channels * bufferMs * sampleRate ~/ 1000, 0, channels,
          sampleRate, 0);
  int pause() => _pause(id);
  int resume() => _resume(id);
  int stop() => _stop(id);
  int release() => _release(id);

  /// 1 = software ring empty, 0 = nonempty, -1 = invalid session.
  int softwareEmpty() => _empty(id);

  /// Queued software frames, not frames presented by hardware; -1 if invalid.
  int queuedFrames() => _queued(id);

  /// -1 invalid, 0 reserved, 1 running, 2 paused, 3 stopped.
  int state() => _state(id);
  int push(Float32List samples) {
    final ptr = calloc<Float>(samples.length);
    try {
      ptr.asTypedList(samples.length).setAll(0, samples);
      return _push(id, ptr, samples.length);
    } finally {
      calloc.free(ptr);
    }
  }
}
