import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/main.dart';
import 'package:ggtts_app/playback_session.dart';
import 'package:ggtts_app/saved_reads.dart';
import 'package:mp_audio_stream/ggtts_audio_session.dart';

typedef _SetC = Void Function(Int32);
typedef _SetD = void Function(int);
typedef _GetC = Int32 Function();
typedef _GetD = int Function();

Future<void> waitUntil(bool Function() predicate) async {
  final deadline = DateTime.now().add(const Duration(seconds: 5));
  while (!predicate()) {
    if (DateTime.now().isAfter(deadline)) {
      throw TimeoutException('fixture wait');
    }
    await Future<void>.delayed(const Duration(milliseconds: 10));
  }
}

void main() {
  final enabled =
      Platform.isLinux &&
      Platform.environment['GGTTS_READ_NATIVE_FIXTURE'] == '1';
  test(
    'real FFI isolates stop blocked generation and paused full output',
    () async {
      final native = DynamicLibrary.open(Platform.environment['GGTTS_LIB']!);
      final mode = native.lookupFunction<_SetC, _SetD>('fake_mode');
      final stage = native.lookupFunction<_GetC, _GetD>('fake_stage');
      final pushes = native.lookupFunction<_GetC, _GetD>('fake_pushes');
      final pausedPushes = native.lookupFunction<_GetC, _GetD>(
        'fake_paused_pushes',
      );
      final root = await Directory.systemTemp.createTemp('ggtts-native-read-');
      try {
        mode(0);
        final first = await Run.start(root);
        await waitUntil(() => stage() == 1);
        expect(
          await controlCommand(
            first.control,
            'stop',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        expect((await first.terminal())['stopped'], isTrue);
        expect(
          await controlCommand(
            first.control,
            'close',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        first.close();
        expect(root.listSync(), isEmpty);
        expect(AudioSession.attach(first.audio).state(), -1);

        mode(1);
        final second = await Run.start(root);
        expect(second.audio, isNot(first.audio));
        expect(AudioSession.attach(first.audio).stop(), -1);
        await waitUntil(() => pushes() > 0);
        expect(
          await controlCommand(
            second.control,
            'pause',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        await waitUntil(() => pausedPushes() > 0);
        expect(
          await controlCommand(
            second.control,
            'stop',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        expect((await second.terminal())['stopped'], isTrue);
        expect(
          await controlCommand(
            second.control,
            'close',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        second.close();
        expect(root.listSync(), isEmpty);
        expect(AudioSession.attach(second.audio).state(), -1);

        // An older/unsupported capability statement omits cancellable. The
        // ordinary renderer still works and does not create a cancellation token.
        mode(2);
        final ordinary = await Run.start(root, cancellable: false);
        final done = await ordinary.terminal();
        expect(done['done'], isTrue);
        expect(
          await controlCommand(
            ordinary.control,
            'close',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        ordinary.close();
        await SavedReads(root).discardPending(done['token'] as String);
        expect(root.listSync(), isEmpty);

        mode(3);
        final invalid = await Run.start(root, cancellable: false);
        final failure = await invalid.terminal();
        expect(failure['error'], contains('fixture invalid options'));
        expect(failure['stopped'], isNull);
        expect(
          await controlCommand(
            invalid.control,
            'close',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        invalid.close();
        expect(root.listSync(), isEmpty);
      } finally {
        await root.delete(recursive: true);
      }
    },
    skip: enabled && Platform.environment['GGTTS_FIXTURE_ABI'] != '5'
        ? false
        : 'Requires ABI 6 fixture via tool/test_read_native_fixture.sh',
  );
  test(
    'Android default selects the existing native session',
    () async {
      final session = PlaybackSession.reserve(isAndroid: true);
      expect(session, isNotNull);
      expect(session!.usesPresentationTracking, isFalse);
      final attached = PlaybackSession.attach(session.id, isAndroid: true);
      expect(attached.usesPresentationTracking, isFalse);
      expect(attached.state(), 0);
      expect(attached.stop(), 0);
      expect(session.release(), 0);
    },
    skip: enabled
        ? false
        : 'Requires the native fake via tool/test_read_native_fixture.sh',
  );
  test(
    'ABI 5 library renders without cancellation symbols being required',
    () async {
      final native = DynamicLibrary.open(Platform.environment['GGTTS_LIB']!);
      native.lookupFunction<_SetC, _SetD>('fake_mode')(2);
      final root = await Directory.systemTemp.createTemp('ggtts-abi5-');
      try {
        final run = await Run.start(root, cancellable: false);
        final done = await run.terminal();
        expect(done['done'], isTrue);
        expect(
          await controlCommand(
            run.control,
            'close',
          ).timeout(const Duration(seconds: 5)),
          isNull,
        );
        run.close();
        await SavedReads(root).discardPending(done['token'] as String);
        expect(root.listSync(), isEmpty);
      } finally {
        await root.delete(recursive: true);
      }
    },
    skip: enabled && Platform.environment['GGTTS_FIXTURE_ABI'] == '5'
        ? false
        : 'Requires ABI 5 fixture via tool/test_read_native_fixture.sh',
  );
}

class Run {
  Run(this.control, this.audio, this.port, this.events);
  final SendPort control;
  final int audio;
  final ReceivePort port;
  final StreamIterator<dynamic> events;

  static Future<Run> start(Directory root, {bool cancellable = true}) async {
    final setup = ReceivePort();
    await Isolate.spawn(
      cancellable ? controlWorker : controlWorkerWithoutCancellation,
      setup.sendPort,
      onError: setup.sendPort,
      onExit: setup.sendPort,
    );
    final initial =
        await setup.first.timeout(const Duration(seconds: 5)) as Map;
    setup.close();
    if (initial['error'] != null) throw StateError(initial['error']);
    final port = ReceivePort();
    final events = StreamIterator<dynamic>(port);
    await Isolate.spawn(
      renderWorker,
      <String, Object?>{
        'port': port.sendPort,
        'audio': initial['audio'],
        'token': initial['token'],
        'cancellable': cancellable,
        'library': root.path,
        'text': 'fixture',
        'options': <String, Object?>{},
        'load': <String, Object?>{},
        'render': <String, Object?>{},
      },
      onError: port.sendPort,
      onExit: port.sendPort,
    );
    return Run(
      initial['port'] as SendPort,
      initial['audio'] as int,
      port,
      events,
    );
  }

  Future<Map> terminal() async {
    while (await events.moveNext().timeout(const Duration(seconds: 10))) {
      final event = events.current;
      if (event is! Map) throw StateError('worker exited: $event');
      if (event['stopped'] == true ||
          event['done'] == true ||
          event['error'] != null) {
        return event;
      }
    }
    throw StateError('worker closed before terminal event');
  }

  void close() {
    port.close();
    events.cancel();
  }
}
