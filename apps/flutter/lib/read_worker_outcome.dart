import 'dart:async';

/// Runs a read to one terminal event. Cleanup must succeed before a pending
/// read is marked ready; a failure always attempts to discard partial audio.
Future<void> runReadOutcome({
  required Future<void> Function() generate,
  required FutureOr<void> Function() cleanup,
  required String Function() finish,
  required Future<void> Function() discard,
  required void Function(Map<String, Object?>) send,
  bool Function(Object)? isStopped,
}) async {
  Object? failure;
  var stopped = false;
  try {
    await generate();
  } catch (error) {
    if (isStopped?.call(error) == true) {
      stopped = true;
    } else {
      failure = error;
    }
  }
  try {
    await cleanup();
  } catch (error) {
    failure ??= error;
  }
  String? token;
  if (failure == null && !stopped) {
    try {
      token = finish();
    } catch (error) {
      failure = error;
    }
  }
  if (failure != null || stopped) {
    try {
      await discard();
    } catch (_) {
      // Keep the original failure and still notify the caller.
    }
    if (failure != null) {
      send({'error': failure.toString()});
    } else {
      send({'stopped': true});
    }
  } else {
    send({'done': true, 'token': token});
  }
}
