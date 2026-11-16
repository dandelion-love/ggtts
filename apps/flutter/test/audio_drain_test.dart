import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/audio_drain.dart';
import 'package:ggtts_app/pcm_queue.dart';

void main() {
  test('waits for software handoff and does not count paused time', () {
    var tick = 0;
    final states = [1, 1, 2, 2, 1, 1];
    drainAudio(
      state: () => states[tick],
      softwareEmpty: () => tick == 0 ? 0 : 1,
      wait: (_) => tick++,
      graceTicks: 2,
    );
    expect(tick, 5);
  });

  test('stop interrupts drain while paused', () {
    var tick = 0;
    expect(
      () => drainAudio(
        state: () => tick == 0 ? 2 : 3,
        softwareEmpty: () => fail('paused session must not be drained'),
        wait: (_) => tick++,
      ),
      throwsA(isA<ReadStopped>()),
    );
  });
}
