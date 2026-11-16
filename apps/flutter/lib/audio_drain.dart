import 'dart:io';

import 'pcm_queue.dart';

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
