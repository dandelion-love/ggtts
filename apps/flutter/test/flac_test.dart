import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

import 'package:ggtts_app/flac.dart';
import 'package:ggtts_app/ggtts.dart';

class _Decoded {
  const _Decoded(this.sampleStart, this.samples);
  final int sampleStart;
  final List<int> samples;
}

int _crc(List<int> bytes, int width, int polynomial) {
  final mask = (1 << width) - 1;
  final high = 1 << (width - 1);
  var crc = 0;
  for (final byte in bytes) {
    crc ^= byte << (width - 8);
    for (var i = 0; i < 8; i++) {
      crc = ((crc << 1) ^ ((crc & high) != 0 ? polynomial : 0)) & mask;
    }
  }
  return crc;
}

(int, int) _utf8Number(Uint8List bytes, int start) {
  final lead = bytes[start];
  if (lead < 0x80) return (lead, 1);
  var width = 0;
  while ((lead & (0x80 >> width)) != 0) {
    width++;
  }
  var value = lead & (0x7f >> width);
  for (var i = 1; i < width; i++) {
    value = (value << 6) | (bytes[start + i] & 0x3f);
  }
  return (value, width);
}

List<_Decoded> _decodeVerbatimFrames(Uint8List bytes) {
  expect(bytes.sublist(0, 4), [0x66, 0x4c, 0x61, 0x43]);
  final info = ByteData.sublistView(bytes, 8);
  final minBlock = info.getUint16(0, Endian.big);
  final maxBlock = info.getUint16(2, Endian.big);
  final packed = info.getUint64(10, Endian.big);
  final total = packed & ((1 << 36) - 1);
  expect(maxBlock, 4096);
  expect(minBlock, 16);

  final frames = <_Decoded>[];
  var offset = 42;
  var samplesBefore = 0;
  while (samplesBefore < total) {
    final frameStart = offset;
    expect(bytes[offset], 0xff);
    expect((bytes[offset + 1] & 1), 1, reason: 'variable-block strategy bit');
    final blockCode = (bytes[offset + 2] >> 4) & 0x0f;
    final (sampleStart, numberBytes) = _utf8Number(bytes, offset + 4);
    offset += 4 + numberBytes;
    final count = switch (blockCode) {
      12 => 4096,
      7 => (() {
        final size = (bytes[offset] << 8) | bytes[offset + 1];
        offset += 2;
        return size + 1;
      })(),
      _ => fail('unexpected FLAC block size code $blockCode'),
    };
    expect(
      sampleStart,
      samplesBefore,
      reason: 'frame must name first sample position',
    );
    expect(count, total - samplesBefore < 4096 ? total - samplesBefore : 4096);
    final headerCrc = bytes[offset];
    expect(headerCrc, _crc(bytes.sublist(frameStart, offset), 8, 0x07));
    offset++;
    expect(bytes[offset], 2, reason: 'verbatim subframe header');
    offset++;
    final samples = <int>[];
    for (var i = 0; i < count; i++) {
      samples.add(
        ByteData.sublistView(bytes, offset, offset + 2).getInt16(0, Endian.big),
      );
      offset += 2;
    }
    final recordedCrc = (bytes[offset] << 8) | bytes[offset + 1];
    expect(recordedCrc, _crc(bytes.sublist(frameStart, offset), 16, 0x8005));
    offset += 2;
    frames.add(_Decoded(sampleStart, samples));
    samplesBefore += count;
  }
  expect(offset, bytes.length, reason: 'no trailing or missing frame data');
  return frames;
}

void main() {
  test(
    'short final frames use valid variable-block bounds and preserve PCM',
    () async {
      final directory = await Directory.systemTemp.createTemp(
        'ggtts-flac-writer-',
      );
      addTearDown(() => directory.delete(recursive: true));
      for (final count in [1, 15, 16, 17, 4097, 4111, 23760]) {
        final file = File('${directory.path}/$count.flac');
        final pcm = Float32List.fromList(
          List<double>.generate(
            count,
            (i) => ((i * 37) % 20001 - 10000) / 12000.0,
          ),
        );
        final expected = pcm
            .map((sample) => (sample.clamp(-1.0, 1.0) * 32767).round())
            .toList();

        final writer = FlacWriter(file);
        final first = count ~/ 3;
        final second = (count * 2) ~/ 3;
        if (first > 0) {
          writer.append(
            Pcm(Float32List.fromList(pcm.sublist(0, first)), 24000),
          );
        }
        if (second > first) {
          writer.append(
            Pcm(Float32List.fromList(pcm.sublist(first, second)), 24000),
          );
        }
        if (count > second) {
          writer.append(Pcm(Float32List.fromList(pcm.sublist(second)), 24000));
        }
        writer.finish();

        final bytes = await file.readAsBytes();
        final frames = _decodeVerbatimFrames(bytes);
        expect(
          frames.map((frame) => frame.sampleStart).toList(),
          List<int>.generate((count + 4095) ~/ 4096, (i) => i * 4096),
          reason: 'sample-number headers for $count samples',
        );
        expect(
          frames.expand((frame) => frame.samples).toList(),
          expected,
          reason: 'PCM for $count samples',
        );
      }
    },
  );
}
