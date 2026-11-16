import 'dart:io';

import 'pcm_queue.dart';
import 'presentation_tracker.dart';

/// Wait for the software ring, then retain the old output grace interval.
/// This is not a hardware playback completion signal.
void drainAudio({
  required int Function() state,
  required int Function() softwareEmpty,
  void Function(Duration) wait = sleep,
  int graceTicks = 170,
}) {
  const tick = Duration(milliseconds: 30);
  var remaining = graceTicks;
  while (true) {
    final current = state();
    if (current == 3 || current < 0) throw ReadStopped();
    if (current == 2) {
      wait(tick);
      continue;
    }
    final empty = softwareEmpty();
    if (empty < 0) throw ReadStopped();
    if (empty == 1) {
      if (remaining == 0) return;
      remaining--;
    }
    wait(tick);
  }
}

/// Wait for Android's track head on the same session, with bounded failure.
void drainPresentation({
  required int Function() state,
  required int Function() submitted,
  required int Function() head,
  Duration Function()? now,
  void Function(Duration) wait = sleep,
}) {
  final clock = Stopwatch()..start();
  final readTime = now ?? () => clock.elapsed;
  final count = submitted();
  if (count <= 0) throw StateError('Android audio submission failed.');
  // The session validates a fresh zero-based head before accepting frames.
  final tracker =
      PresentationTracker(
          initialHead: 0,
          stallAfter: const Duration(seconds: 30),
          timeoutAfter: const Duration(seconds: 90),
        )
        ..accepted(count)
        ..seal(readTime());
  while (true) {
    final current = state();
    if (current == 3 || current == -1) throw ReadStopped();
    if (current != 1 && current != 2) {
      throw StateError('Android audio interrupted.');
    }
    final position = head();
    if (position < 0) throw StateError('Android audio position unavailable.');
    switch (tracker.observe(position, readTime(), paused: current == 2)) {
      case PresentationState.presented:
        return;
      case PresentationState.pending:
        wait(const Duration(milliseconds: 30));
      case PresentationState.stopped:
        throw ReadStopped();
      case PresentationState.stalled:
      case PresentationState.timedOut:
      case PresentationState.invalid:
        throw StateError('Android audio presentation was not confirmed.');
    }
  }
}
