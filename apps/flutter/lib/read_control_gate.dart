/// UI-isolate admission for one read. A terminal event closes control before
/// asynchronous cleanup, so no command can be queued behind close.
class ReadControlGate {
  bool stopRequested = false;
  bool terminal = false;

  bool requestStop() {
    if (terminal || stopRequested) return false;
    stopRequested = true;
    return true;
  }

  void finish() {
    terminal = true;
  }
}
