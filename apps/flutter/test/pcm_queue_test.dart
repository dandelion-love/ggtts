import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/pcm_queue.dart';
import 'package:ggtts_app/playback_session.dart';

void main() {
  test('20-second bounded ring accepts the observed-size first piece without drain', () {
    const rate = 24000;
    final samples = Float32List.fromList(
      List.generate(342480, (i) => (i % 13) / 16.0),
    );
    int deliver(int capacity, {required bool expectFull}) {
      var accepted = 0;
      final output = Float32List(samples.length);
      final queue = PcmQueue(
        (piece) {
          if (accepted + piece.length > capacity) return -1;
          output.setRange(accepted, accepted + piece.length, piece);
          accepted += piece.length;
          return 0;
        },
        pause: (_) {},
        maxRetries: 2,
      );
      if (expectFull) {
        expect(() => queue.add(samples), throwsStateError);
      } else {
        queue.add(samples);
        expect(accepted, samples.length);
        expect(output, samples);
      }
      return accepted;
    }

    expect(deliver(rate * 5, expectFull: true), lessThan(samples.length));
    expect(
      deliver(rate * playbackBufferMs(rate) ~/ 1000, expectFull: false),
      samples.length,
    );
  });

  test('ring duration is capped by sample count and rejects invalid rates', () {
    expect(playbackBufferMs(24000), 20000);
    expect(playbackBufferMs(48000), 20000);
    expect(playbackBufferMs(192000), 5000);
    for (final rate in [0, -1, 192001]) {
      expect(() => playbackBufferMs(rate), throwsArgumentError);
    }
    for (final rate in [8000, 24000, 44100, 48000, 96000, 192000]) {
      expect(
        rate * playbackBufferMs(rate) ~/ 1000,
        lessThanOrEqualTo(maxPlaybackBufferSamples),
      );
    }
  });

  test('a full output retries without dropping or reordering samples', () {
    var blocked = true;
    final received = <double>[];
    final queue = PcmQueue(
      (chunk) {
        if (blocked) return -1;
        received.addAll(chunk);
        return 0;
      },
      pause: (_) {
        blocked = false;
      },
    );
    final input = Float32List.fromList(
      List.generate(5000, (i) => i.toDouble()),
    );
    queue.add(input);
    expect(received, input);
  });

  test('a permanently full output fails rather than retrying forever', () {
    var calls = 0;
    final queue = PcmQueue(
      (_) {
        calls++;
        return -1;
      },
      pause: (_) {},
      maxRetries: 3,
    );
    expect(() => queue.add(Float32List.fromList([1])), throwsStateError);
    expect(calls, 3);
  });

  test('pause does not consume retries and resume preserves samples', () {
    var calls = 0;
    final received = <double>[];
    final queue = PcmQueue(
      (chunk) {
        if (++calls <= 5) return -2;
        received.addAll(chunk);
        return 0;
      },
      pause: (_) {},
      maxRetries: 2,
    );
    final samples = Float32List.fromList([0.1, 0.2]);
    queue.add(samples);
    expect(received, samples);
  });

  test('stopped session exits immediately rather than retrying as full', () {
    var calls = 0;
    final queue = PcmQueue((_) {
      calls++;
      return -3;
    }, pause: (_) {});
    expect(
      () => queue.add(Float32List.fromList([1])),
      throwsA(isA<ReadStopped>()),
    );
    expect(calls, 1);
  });
}
