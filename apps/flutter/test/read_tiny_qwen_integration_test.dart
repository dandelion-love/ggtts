// CPU-only production Qwen/FFI and real Dart isolates; audio is a fake sink.
import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/main.dart';
import 'package:ggtts_app/saved_reads.dart';
import 'package:mp_audio_stream/ggtts_audio_session.dart';

typedef _SetC = Void Function(Int32);
typedef _SetD = void Function(int);
typedef _GetC = Int32 Function();
typedef _GetD = int Function();

void main() {
  final enabled =
      Platform.isLinux && Platform.environment['GGTTS_TINY_READ_PROBE'] == '1';
  test(
    'production tiny Qwen delivers then stop and next read succeeds',
    () async {
      final fake = DynamicLibrary.open(
        Platform.environment['GGTTS_FAKE_AUDIO']!,
      );
      final mode = fake.lookupFunction<_SetC, _SetD>('fake_mode');
      final pushes = fake.lookupFunction<_GetC, _GetD>('fake_pushes');
      final root = await Directory.systemTemp.createTemp('ggtts-tiny-read-');
      final timer = Stopwatch()..start();
      try {
        mode(1);
        final first = await TinyRead.start(root);
        await first.awaitAudio();
        stderr.writeln(
          'first real Qwen piece at ${timer.elapsedMilliseconds} ms',
        );
        final deadline = DateTime.now().add(const Duration(seconds: 10));
        while (pushes() == 0 && DateTime.now().isBefore(deadline)) {
          await Future<void>.delayed(const Duration(milliseconds: 10));
        }
        expect(
          pushes(),
          greaterThan(0),
          reason: 'fake output did not report full',
        );
        final stopAt = timer.elapsedMilliseconds;
        expect(
          await controlCommand(
            first.control,
            'stop',
          ).timeout(const Duration(seconds: 10)),
          isNull,
        );
        expect((await first.terminal())['stopped'], isTrue);
        stderr.writeln(
          'stop after fake full output took ${timer.elapsedMilliseconds - stopAt} ms',
        );
        expect(
          await controlCommand(
            first.control,
            'close',
          ).timeout(const Duration(seconds: 10)),
          isNull,
        );
        first.close();
        expect(root.listSync(), isEmpty);
        expect(AudioSession.attach(first.id).state(), -1);

        mode(2);
        final second = await TinyRead.start(root);
        expect(second.id, isNot(first.id));
        expect(AudioSession.attach(first.id).stop(), -1);
        await second.awaitAudio();
        stderr.writeln(
          'second real Qwen piece at ${timer.elapsedMilliseconds} ms',
        );
        final done = await second.terminal();
        expect(done['done'], isTrue, reason: '$done');
        expect(
          await controlCommand(
            second.control,
            'close',
          ).timeout(const Duration(seconds: 10)),
          isNull,
        );
        second.close();
        await SavedReads(root).discardPending(done['token'] as String);
        expect(root.listSync(), isEmpty);
      } finally {
        await root.delete(recursive: true);
      }
    },
    timeout: const Timeout(Duration(seconds: 100)),
    skip: enabled
        ? false
        : 'Run tool/test_read_tiny_qwen.sh with pinned CPU fixture',
  );
}

class TinyRead {
  TinyRead(this.control, this.id, this.port, this.events);
  final SendPort control;
  final int id;
  final ReceivePort port;
  final StreamIterator<dynamic> events;

  static Future<TinyRead> start(Directory root) async {
    final setup = ReceivePort();
    await Isolate.spawn(
      controlWorker,
      setup.sendPort,
      onError: setup.sendPort,
      onExit: setup.sendPort,
    );
    final initial =
        await setup.first.timeout(const Duration(seconds: 10)) as Map;
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
        'cancellable': true,
        'library': root.path,
        'text': 'Hello.',
        'options': <String, Object?>{},
        'load': <String, Object?>{
          'load': <String, Object?>{
            'engine': 'qwen3-tts',
            'model': Platform.environment['GGTTS_TINY_TALKER'],
            'codec': Platform.environment['GGTTS_CODEC'],
            'device': 'cpu',
            'offline': true,
            'threads': 2,
            'max_frames': 2,
          },
        },
        'render': <String, Object?>{},
      },
      onError: port.sendPort,
      onExit: port.sendPort,
    );
    return TinyRead(
      initial['port'] as SendPort,
      initial['audio'] as int,
      port,
      events,
    );
  }

  Future<Map> next() async {
    if (!await events.moveNext().timeout(const Duration(seconds: 40))) {
      throw StateError('render worker exited');
    }
    final event = events.current;
    if (event is! Map) throw StateError('worker failed: $event');
    return event;
  }

  Future<void> awaitAudio() async {
    while (true) {
      final event = await next();
      if (event['audioReady'] == true) return;
      if (event['error'] != null ||
          event['done'] == true ||
          event['stopped'] == true) {
        throw StateError('no audio piece: $event');
      }
    }
  }

  Future<Map> terminal() async {
    while (true) {
      final event = await next();
      if (event['error'] != null ||
          event['done'] == true ||
          event['stopped'] == true) {
        return event;
      }
    }
  }

  void close() {
    port.close();
    events.cancel();
  }
}
