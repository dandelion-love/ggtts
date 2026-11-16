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
  test('presentation drain waits beyond old grace for the downstream head', () {
    var now = Duration.zero;
    var polls = 0;
    drainPresentation(
      state: () => 1,
      submitted: () => 24000,
      head: () {
        polls++;
        return now < const Duration(seconds: 6) ? 0 : 24000;
      },
      now: () => now,
      wait: (duration) {
        now += duration;
      },
    );
    expect(now, greaterThanOrEqualTo(const Duration(seconds: 6)));
    expect(polls, greaterThan(1));
  });

  test('presentation drain refuses interruption and stalled output', () {
    var now = Duration.zero;
    expect(
      () => drainPresentation(
        state: () => 1,
        submitted: () => 10,
        head: () => 0,
        now: () => now,
        wait: (d) {
          now += d;
        },
      ),
      throwsA(isA<StateError>()),
    );
    expect(
      () => drainPresentation(
        state: () => -2,
        submitted: () => 10,
        head: () => 0,
      ),
      throwsA(isA<StateError>()),
    );
    expect(
      () =>
          drainPresentation(state: () => 3, submitted: () => 10, head: () => 0),
      throwsA(isA<ReadStopped>()),
    );
  });
}
