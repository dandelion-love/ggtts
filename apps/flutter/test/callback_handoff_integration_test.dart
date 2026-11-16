import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/main.dart';
import 'package:ggtts_app/saved_reads.dart';

typedef _VoidC = Void Function();
typedef _VoidD = void Function();
typedef _IntC = Int32 Function();
typedef _IntD = int Function();
typedef _SampleC = Float Function(Int32);
typedef _SampleD = double Function(int);
typedef _IdC = Uint64 Function();
typedef _IdD = int Function();

void main() {
  final enabled =
      Platform.isLinux && Platform.environment['GGTTS_HANDOFF_FIXTURE'] == '1';
  test(
    'stream callbacks reach audio in order and failures discard',
    () async {
      final lib = DynamicLibrary.open(Platform.environment['GGTTS_LIB']!);
      final reset = lib.lookupFunction<_VoidC, _VoidD>('fixture_reset');
      final count = lib.lookupFunction<_IntC, _IntD>('fixture_count');
      final sample = lib.lookupFunction<_SampleC, _SampleD>('fixture_sample');
      final first = lib.lookupFunction<_IntC, _IntD>(
        'fixture_first_handed_off',
      );
      final inits = lib.lookupFunction<_IntC, _IntD>('fixture_init_count');
      final releases = lib.lookupFunction<_IntC, _IntD>(
        'fixture_release_count',
      );
      final newSession = lib.lookupFunction<_IdC, _IdD>(
        'ma_stream_session_new',
      );
      final root = await Directory.systemTemp.createTemp('ggtts-handoff-');
      try {
        reset();
        final success = await runWorker(root, newSession(), false);
        expect(success['done'], isTrue, reason: '$success');
        expect(
          first(),
          1,
          reason: 'first callback must hand off before the next',
        );
        expect(inits(), 1);
        expect(releases(), 1);
        expect(List.generate(count(), sample), [
          0.25,
          -0.5,
          0.75,
          1.0,
        ], reason: 'callback samples dropped or reordered');
        await SavedReads(root).discardPending(success['token'] as String);
        expect(root.listSync(), isEmpty);

        reset();
        final failed = await runWorker(root, newSession(), true);
        expect(failed['error'], contains('sample rate changed'));
        expect(failed['done'], isNull);
        expect(List.generate(count(), sample), [0.25, -0.5]);
        expect(inits(), 1);
        expect(releases(), 1);
        expect(root.listSync(), isEmpty);
      } finally {
        await root.delete(recursive: true);
      }
    },
    timeout: const Timeout(Duration(seconds: 30)),
    skip: enabled ? false : 'Run tool/test_callback_handoff.sh on Linux',
  );
}

Future<Map> runWorker(Directory root, int audio, bool badRate) async {
  final port = ReceivePort();
  final events = StreamIterator<dynamic>(port);
  await Isolate.spawn(
    renderWorker,
    <String, Object?>{
      'port': port.sendPort,
      'audio': audio,
      'cancellable': false,
      'library': root.path,
      'text': 'fixture',
      'options': <String, Object?>{},
      'load': <String, Object?>{},
      'render': badRate
          ? <String, Object?>{'fixture': 'bad_rate'}
          : <String, Object?>{},
    },
    onError: port.sendPort,
    onExit: port.sendPort,
  );
  Map? terminal;
  try {
    while (await events.moveNext().timeout(const Duration(seconds: 20))) {
      final event = events.current;
      if (event == null) break;
      if (event is! Map) throw StateError('worker crashed: $event');
      if (event['error'] != null ||
          event['done'] == true ||
          event['stopped'] == true) {
        terminal = event;
      }
    }
    return terminal ?? (throw StateError('worker exited without a result'));
  } finally {
    await events.cancel();
    port.close();
  }
}
