import 'dart:ffi';
import 'dart:io';

typedef _ControlC = Int64 Function();
typedef _ControlD = int Function();

/// Native controls for the vendored playback backend.
///
/// softwareEmpty only means the software ring handed its frames to the device;
/// it does not confirm that the listener heard them. Calls may block while the
/// native device changes state; invoke off the Flutter UI isolate.
class AudioControls {
  AudioControls._(DynamicLibrary lib)
      : _pause = lib.lookupFunction<_ControlC, _ControlD>('ma_stream_pause'),
        _resume = lib.lookupFunction<_ControlC, _ControlD>('ma_stream_resume_controlled'),
        _stop = lib.lookupFunction<_ControlC, _ControlD>('ma_stream_stop_controlled'),
        _empty = lib.lookupFunction<_ControlC, _ControlD>('ma_stream_software_empty'),
        _state = lib.lookupFunction<_ControlC, _ControlD>('ma_stream_control_state');

  factory AudioControls.open() {
    if (Platform.isLinux || Platform.isAndroid) {
      return AudioControls._(DynamicLibrary.open('libmp_audio_stream.so'));
    }
    if (Platform.isWindows) {
      return AudioControls._(DynamicLibrary.open('mp_audio_stream.dll'));
    }
    throw UnsupportedError('Controlled audio is not supported on this platform.');
  }

  final _ControlD _pause, _resume, _stop, _empty, _state;
  bool pause() => _pause() == 0;
  bool resume() => _resume() == 0;
  bool stop() => _stop() == 0;
  bool get softwareEmpty => _empty() == 1;
  int get state => _state();
}
