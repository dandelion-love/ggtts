// Incremental mono 16-bit FLAC with verbatim subframes, matching
// ggtts-core/src/flac.rs. Buffers at most one 4096-sample block.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'ggtts.dart';

class FlacWriter {
  FlacWriter(File file) : _file = file.openSync(mode: FileMode.write);
  final RandomAccessFile _file;
  final Float32List _buffer = Float32List(4096);
  int _buffered = 0, _total = 0;
  int? _rate;
  bool _closed = false;

  /// Synchronous: safe to call from the synchronous native piece callback.
  void append(Pcm pcm) {
    if (_closed) throw StateError('FLAC writer is closed');
    if (pcm.rate <= 0 || pcm.rate >= (1 << 20)) {
      throw ArgumentError('invalid FLAC sample rate');
    }
    if (_rate != null && _rate != pcm.rate) {
      throw ArgumentError('sample rate changed');
    }
    if (_total + pcm.samples.length >= (1 << 36)) {
      throw ArgumentError('too many samples');
    }
    for (final x in pcm.samples) {
      if (!x.isFinite) throw ArgumentError('non-finite PCM sample');
    }
    if (_rate == null) {
      _rate = pcm.rate;
      _file.writeFromSync([0x66, 0x4c, 0x61, 0x43]);
      final info = _Bits()
        ..put(1, 1)
        ..put(0, 7)
        ..put(34, 24)
        ..put(4096, 16)
        ..put(4096, 16)
        ..put(0, 24)
        ..put(0, 24)
        ..put(pcm.rate, 20)
        ..put(0, 3)
        ..put(15, 5)
        ..put(0, 36)
        ..put(0, 64)
        ..put(0, 64);
      _file.writeFromSync(info.take());
    }
    var at = 0;
    while (at < pcm.samples.length) {
      final n = math.min(4096 - _buffered, pcm.samples.length - at);
      _buffer.setRange(_buffered, _buffered + n, pcm.samples, at);
      _buffered += n;
      _total += n;
      at += n;
      if (_buffered == 4096) _writeFrame();
    }
  }

  void _writeFrame() {
    final count = _buffered;
    final header = _Bits()
      ..put(0x3ffe, 14)
      ..put(0, 1)
      ..put(1, 1)
      ..put(count == 4096 ? 12 : 7, 4)
      ..put(0, 4)
      ..put(0, 4)
      ..put(4, 3)
      ..put(0, 1);
    final frame = BytesBuilder(copy: false)
      ..add(header.take())
      ..add(_utf8Number(_total - count));
    if (count != 4096) frame.add([(count - 1) >> 8, (count - 1) & 255]);
    frame.add([_crc(frame.toBytes(), 8, 0x07), 2]);
    for (var i = 0; i < count; i++) {
      final value = (_buffer[i].clamp(-1.0, 1.0) * 32767).round();
      frame.add([(value >> 8) & 255, value & 255]);
    }
    final sum = _crc(frame.toBytes(), 16, 0x8005);
    frame.add([sum >> 8, sum & 255]);
    _file.writeFromSync(frame.takeBytes());
    _buffered = 0;
  }

  /// Finish stream info and close. Empty reads are not valid saved audio.
  void finish() {
    if (_closed) throw StateError('FLAC writer is closed');
    if (_rate == null || _total == 0) throw StateError('no audio was rendered');
    if (_buffered > 0) _writeFrame();
    final minBlockSize = _total % 4096 == 0 ? 4096 : _total % 4096;
    final blockSizes = _Bits()
      ..put(minBlockSize, 16)
      ..put(4096, 16);
    _file.setPositionSync(8);
    _file.writeFromSync(blockSizes.take());
    final summary = _Bits()
      ..put(_rate!, 20)
      ..put(0, 3)
      ..put(15, 5)
      ..put(_total, 36);
    _file.setPositionSync(18);
    _file.writeFromSync(summary.take());
    _file.flushSync();
    _file.closeSync();
    _closed = true;
  }

  void close() {
    if (!_closed) {
      _file.closeSync();
      _closed = true;
    }
  }
}

int _crc(List<int> bytes, int bits, int polynomial) {
  final mask = (1 << bits) - 1, high = 1 << (bits - 1);
  var crc = 0;
  for (final byte in bytes) {
    crc ^= byte << (bits - 8);
    for (var i = 0; i < 8; i++) {
      crc = ((crc << 1) ^ ((crc & high) != 0 ? polynomial : 0)) & mask;
    }
  }
  return crc;
}

List<int> _utf8Number(int number) {
  if (number < 0x80) return [number];
  var n = 2;
  while (n < 7 && number >= (1 << (5 * n + 1))) {
    n++;
  }
  final out = List<int>.filled(n, 0);
  var rest = number;
  for (var i = n - 1; i > 0; i--) {
    out[i] = 0x80 | (rest & 0x3f);
    rest >>= 6;
  }
  out[0] = ((0xff00 >> n) | rest) & 255;
  return out;
}

class _Bits {
  final _out = BytesBuilder(copy: false);
  int _acc = 0, _count = 0;
  void put(int value, int bits) {
    for (var i = bits - 1; i >= 0; i--) {
      _acc = (_acc << 1) | ((value >> i) & 1);
      if (++_count == 8) {
        _out.add([_acc]);
        _acc = 0;
        _count = 0;
      }
    }
  }

  Uint8List take() {
    if (_count != 0) throw StateError('FLAC field is not byte aligned');
    return _out.takeBytes();
  }
}
