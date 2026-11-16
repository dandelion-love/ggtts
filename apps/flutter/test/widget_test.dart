import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/ggtts.dart';
import 'package:ggtts_app/main.dart';

void main() {
  test('wavBytes writes a 16-bit mono WAV header and clamps samples', () {
    final wav = wavBytes(Pcm(Float32List.fromList([0.0, 1.0, -2.0]), 24000));
    final b = ByteData.sublistView(wav);
    expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
    expect(b.getUint32(24, Endian.little), 24000);
    expect(b.getUint32(40, Endian.little), 6);
    expect(b.getInt16(46, Endian.little), 32767);
    expect(b.getInt16(48, Endian.little), -32767);
  });

  test('the load document carries only load options, downloads off', () {
    final doc = loadDocument('/cache');
    expect(doc.keys, ['load']);
    final load = doc['load'] as Map;
    expect(load['offline'], true);
    expect(load['cache_dir'], '/cache');
  });
}
