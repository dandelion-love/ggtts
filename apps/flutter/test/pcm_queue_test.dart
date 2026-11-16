import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/pcm_queue.dart';

void main() {
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
