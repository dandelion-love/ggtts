import 'dart:io';
import 'dart:typed_data';

/// Write bounded PCM pieces, applying backpressure when the output buffer is full.
class PcmQueue {
  PcmQueue(this._push, {void Function(Duration)? pause, this.maxRetries = 1000})
    : _pause = pause ?? sleep;
  final int Function(Float32List) _push;
  final void Function(Duration) _pause;
  final int maxRetries;

  void add(Float32List samples) {
    for (var offset = 0; offset < samples.length; offset += 2048) {
      final end = (offset + 2048).clamp(0, samples.length);
      final chunk = Float32List.sublistView(samples, offset, end);
      var attempts = 0;
      while (_push(chunk) != 0) {
        if (++attempts >= maxRetries) {
          throw StateError('Audio output stayed full.');
        }
        _pause(const Duration(milliseconds: 30));
      }
    }
  }
}
