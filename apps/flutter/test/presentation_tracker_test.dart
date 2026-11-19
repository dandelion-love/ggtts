import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/presentation_tracker.dart';

void main() {
  PresentationTracker tracker() => PresentationTracker(
    initialHead: 0,
    stallAfter: const Duration(seconds: 10),
    timeoutAfter: const Duration(seconds: 20),
  );

  test('downstream delay exceeds old grace; completion follows head', () {
    final p = tracker()
      ..accepted(24000)
      ..seal(Duration.zero);
    expect(
      p.observe(0, const Duration(milliseconds: 5200)),
      PresentationState.pending,
    );
    expect(
      p.observe(23999, const Duration(seconds: 6)),
      PresentationState.pending,
    );
    expect(
      p.observe(24000, const Duration(seconds: 7)),
      PresentationState.presented,
    );
  });

  test('stop, pause timeout and no progress fail closed', () {
    final stopped = tracker()
      ..accepted(4)
      ..seal(Duration.zero);
    stopped.stop();
    expect(
      stopped.observe(4, const Duration(seconds: 1)),
      PresentationState.stopped,
    );
    final paused = tracker()
      ..accepted(4)
      ..seal(Duration.zero);
    expect(
      paused.observe(0, const Duration(seconds: 19), paused: true),
      PresentationState.pending,
    );
    expect(
      paused.observe(0, const Duration(seconds: 20), paused: true),
      PresentationState.timedOut,
    );
    final stalled = tracker()
      ..accepted(4)
      ..seal(Duration.zero);
    expect(
      stalled.observe(0, const Duration(seconds: 10)),
      PresentationState.stalled,
    );
  });

  test('invalid head, reset, late writes and clock regression fail closed', () {
    final ahead = tracker()
      ..accepted(4)
      ..seal(Duration.zero);
    expect(
      ahead.observe(5, const Duration(seconds: 1)),
      PresentationState.invalid,
    );
    final reset = tracker()
      ..accepted(4)
      ..seal(Duration.zero);
    expect(
      reset.observe(2, const Duration(seconds: 1)),
      PresentationState.pending,
    );
    expect(
      reset.observe(1, const Duration(seconds: 2)),
      PresentationState.invalid,
    );
    final late = tracker()
      ..accepted(4)
      ..seal(Duration.zero)
      ..accepted(1);
    expect(late.state, PresentationState.invalid);
    final time = tracker()
      ..accepted(4)
      ..seal(const Duration(seconds: 1));
    expect(time.observe(0, Duration.zero), PresentationState.invalid);
  });

  test('a head at the submitted count before sealing is not complete', () {
    final p = tracker()..accepted(4);
    expect(p.observe(4, const Duration(seconds: 1)), PresentationState.pending);
    p.seal(const Duration(seconds: 2));
    expect(
      p.observe(4, const Duration(seconds: 3)),
      PresentationState.presented,
    );
  });
}
