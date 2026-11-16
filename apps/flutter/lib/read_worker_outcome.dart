import 'dart:async';

/// Runs a read to one terminal event. Cleanup must succeed before a pending
/// read is marked ready; a failure always attempts to discard partial audio.
Future<void> runReadOutcome({
  required Future<void> Function() generate,
  required FutureOr<void> Function() cleanup,
  required String Function() finish,
  required Future<void> Function() discard,
  required void Function(Map<String, Object?>) send,
}) async {
  Object? failure;
  try {
    await generate();
  } catch (error) {
    failure = error;
  }
  try {
    await cleanup();
  } catch (error) {
    failure ??= error;
  }
  String? token;
  if (failure == null) {
    try {
      token = finish();
    } catch (error) {
      failure = error;
    }
  }
  if (failure != null) {
    try {
      await discard();
    } catch (_) {
      // Keep the original failure and still notify the caller.
    }
    send({'error': failure.toString()});
  } else {
    send({'done': true, 'token': token});
  }
}
