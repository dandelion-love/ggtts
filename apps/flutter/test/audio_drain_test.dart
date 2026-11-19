import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/audio_drain.dart';
import 'package:ggtts_app/pcm_queue.dart';
import 'package:ggtts_app/playback_session.dart';

void main() {
  test(
    'Android defaults to miniaudio and prototype requires explicit opt-in',
    () {
      expect(useAndroidPlaybackBridge(isAndroid: true), isFalse);
      expect(
        useAndroidPlaybackBridge(isAndroid: false, experimental: true),
        isFalse,
      );
      expect(
        useAndroidPlaybackBridge(isAndroid: true, experimental: true),
        isTrue,
      );
    },
  );

  test('waits for software handoff and does not count paused time', () {
    var tick = 0;
    final states = [1, 1, 2, 2, 1, 1];
    drainAudio(
      state: () => states[tick],
      queuedFrames: () => tick == 0 ? 1 : 0,
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
        queuedFrames: () => fail('paused session must not be drained'),
        wait: (_) => tick++,
      ),
      throwsA(isA<ReadStopped>()),
    );
  });
  test('nonempty ring that stops progressing fails after the deadline', () {
    var now = Duration.zero;
    expect(
      () => drainAudio(
        state: () => 1,
        queuedFrames: () => 42,
        now: () => now,
        wait: (d) => now += d,
      ),
      throwsA(isA<StateError>()),
    );
    expect(now, greaterThanOrEqualTo(const Duration(seconds: 30)));
    expect(now, lessThan(const Duration(seconds: 31)));
  });

  test(
    'slow consumption resets the deadline and pause time does not count',
    () {
      var now = Duration.zero;
      drainAudio(
        state: () =>
            now >= const Duration(seconds: 15) &&
                now < const Duration(seconds: 55)
            ? 2
            : 1,
        queuedFrames: () {
          if (now < const Duration(seconds: 60)) return 3;
          if (now < const Duration(seconds: 85)) return 2;
          if (now < const Duration(seconds: 110)) return 1;
          return 0;
        },
        now: () => now,
        wait: (d) => now += d,
        graceTicks: 1,
      );
      expect(now, greaterThanOrEqualTo(const Duration(seconds: 110)));
    },
  );

  test('a stopped session cannot be mistaken for an empty drain', () {
    var now = Duration.zero;
    expect(
      () => drainAudio(
        state: () => now >= const Duration(seconds: 1) ? 3 : 1,
        queuedFrames: () => 50,
        now: () => now,
        wait: (d) => now += d,
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
