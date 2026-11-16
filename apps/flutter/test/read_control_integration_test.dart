import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/main.dart';

void main() {
  testWidgets('stop before first audio and start a second read', (
    tester,
  ) async {
    final harness = await Harness.start(tester);
    await harness.speak();
    await harness.waitFor('worker');
    await harness.tap('Stop');
    await harness.waitFor('stop');
    harness.worker!.send('stopped');
    await harness.waitText('Read stopped.');
    await harness.waitSpeakReady();
    expect(harness.events.where((e) => e == 'stop'), hasLength(1));
    await harness.speak();
    await harness.waitFor('worker', count: 2);
    harness.worker!.send('done');
    await harness.waitText('Audio submitted. Save the read or discard it.');
    await harness.close();
  });

  testWidgets('pause then stop while rendering is stalled', (tester) async {
    final harness = await Harness.start(tester);
    await harness.speak();
    await harness.waitFor('worker');
    harness.worker!.send('audio');
    await harness.waitButton('Pause');
    await harness.tap('Pause');
    await harness.waitFor('pause');
    await harness.waitText('Resume');
    await harness.tap('Stop');
    await harness.waitFor('stop');
    harness.worker!.send('stopped');
    await harness.waitText('Read stopped.');
    await harness.waitFor('close');
    await harness.waitSpeakReady();
    expect(
      harness.events.indexOf('pause'),
      lessThan(harness.events.indexOf('stop')),
    );
    await harness.close();
  });

  testWidgets('terminal done closes control before delayed close reply', (
    tester,
  ) async {
    final harness = await Harness.start(tester, holdClose: true);
    await harness.speak();
    await harness.waitFor('worker');
    harness.worker!.send('done');
    await harness.waitFor('close');
    expect(find.text('Stop'), findsNothing);
    harness.control!.send({'releaseClose': true});
    await harness.waitText('Audio submitted. Save the read or discard it.');
    expect(harness.events.where((e) => e == 'stop'), isEmpty);
    await harness.waitButton('Discard read');
    await harness.tap('Discard read');
    await harness.waitText('Read discarded.');
    await harness.waitSpeakReady();
    await harness.speak();
    await harness.waitFor('worker', count: 2);
    harness.worker!.send('stopped');
    await harness.waitFor('close', count: 2);
    harness.control!.send({'releaseClose': true});
    await harness.close();
  });

  testWidgets('legacy or unsupported engine hides stop and can render', (
    tester,
  ) async {
    final harness = await Harness.start(tester, cancellable: false);
    await harness.speak();
    await harness.waitFor('worker');
    expect(find.text('Stop'), findsNothing);
    expect(find.text('Stop is unavailable for this engine.'), findsOneWidget);
    harness.worker!.send('done');
    await harness.waitFor('close');
    await harness.waitButton('Discard read');
    expect(
      find.text('Audio submitted. Save the read or discard it.'),
      findsOneWidget,
    );
    await harness.close();
  });

  testWidgets('unsupported cancellable backend is shown and releases control', (
    tester,
  ) async {
    final harness = await Harness.start(tester);
    await harness.speak();
    await harness.waitFor('worker');
    harness.worker!.send('unsupported');
    await harness.waitFor('close');
    await harness.waitText('Read failed.');
    expect(find.textContaining('cancellation unsupported'), findsOneWidget);
    await harness.close();
  });
}

class Harness {
  Harness(this.tester, this.root, this.inbox, this.subscription, this.events);
  final WidgetTester tester;
  final Directory root;
  final ReceivePort inbox;
  final StreamSubscription<dynamic> subscription;
  final List<Object?> events;
  SendPort? worker;
  SendPort? control;

  static Future<Harness> start(
    WidgetTester tester, {
    bool holdClose = false,
    bool cancellable = true,
  }) async {
    final root = Directory.systemTemp.createTempSync('ggtts-read-test-');
    final inbox = ReceivePort();
    final events = <Object?>[];
    late Harness harness;
    final sub = inbox.listen((dynamic event) {
      if (event is Map && event['kind'] == 'worker') {
        harness.worker = event['port'] as SendPort;
        events.add('worker');
      } else if (event is Map && event['kind'] == 'control') {
        harness.control = event['port'] as SendPort;
        events.add('control');
      } else {
        events.add(event);
      }
    });
    harness = Harness(tester, root, inbox, sub, events);
    final eventPort = inbox.sendPort;
    final fakeControl = makeFakeControl(eventPort, holdClose);
    final fakeRender = makeFakeRender(eventPort);
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          prepare: (_) async => {
            'schema': <String, Object?>{},
            'capabilities': <String, Object?>{'cancellable': cancellable},
          },
          loadSettings: () async => {},
          readDirectory: () async => root,
          controlEntry: fakeControl,
          renderEntry: fakeRender,
        ),
      ),
    );
    await harness.tap('Prepare model (download if needed)');
    await harness.waitText('Model ready.');
    return harness;
  }

  Future<void> tick() async {
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 10)),
    );
    await tester.pump();
  }

  Future<void> waitFor(String event, {int count = 1}) async {
    for (
      var i = 0;
      i < 100 && events.where((e) => e == event).length < count;
      i++
    ) {
      await tick();
    }
    expect(
      events.where((e) => e == event).length,
      greaterThanOrEqualTo(count),
      reason:
          '$events / ${find.byKey(const Key('error')).evaluate().map((e) => (e.widget as Text).data).toList()}',
    );
  }

  Future<void> waitText(String text) async {
    for (var i = 0; i < 100 && find.text(text).evaluate().isEmpty; i++) {
      await tick();
    }
    expect(find.text(text), findsOneWidget);
  }

  Future<void> tap(String text) async {
    await tester.ensureVisible(find.text(text));
    await tester.tap(find.text(text));
    await tester.pump();
  }

  Future<void> waitButton(String text) async {
    for (var i = 0; i < 150; i++) {
      final finder = find.widgetWithText(TextButton, text);
      if (finder.evaluate().isNotEmpty &&
          tester.widget<TextButton>(finder).onPressed != null) {
        return;
      }
      await tick();
    }
    fail('$text did not become available');
  }

  Future<void> waitSpeakReady() async {
    for (var i = 0; i < 100; i++) {
      final button = tester.widget<FilledButton>(
        find.widgetWithText(FilledButton, 'Speak'),
      );
      if (button.onPressed != null) return;
      await tick();
    }
    fail('Speak did not become available');
  }

  Future<void> speak() async {
    await tap('Speak');
  }

  Future<void> close() async {
    await tester.pumpWidget(const SizedBox());
    inbox.close();
    root.deleteSync(recursive: true);
  }
}

void Function(SendPort) makeFakeControl(SendPort eventPort, bool holdClose) {
  return (SendPort parent) {
    final commands = ReceivePort();
    SendPort? closeReply;
    eventPort.send({'kind': 'control', 'port': commands.sendPort});
    parent.send({'port': commands.sendPort, 'audio': 1, 'token': 1});
    commands.listen((dynamic raw) {
      final command = raw as Map;
      if (command['releaseClose'] == true) {
        closeReply?.send(null);
        commands.close();
        return;
      }
      final op = command['op'] as String;
      eventPort.send(op);
      final reply = command['reply'] as SendPort;
      if (op == 'close' && holdClose) {
        closeReply = reply;
      } else {
        reply.send(null);
        if (op == 'close') commands.close();
      }
    });
  };
}

void Function(Map<String, Object?>) makeFakeRender(SendPort eventPort) {
  return (Map<String, Object?> request) {
    final commands = ReceivePort();
    final parent = request['port'] as SendPort;
    eventPort.send({'kind': 'worker', 'port': commands.sendPort});
    commands.listen((dynamic command) {
      switch (command) {
        case 'audio':
          parent.send({'audioReady': true});
        case 'stopped':
          parent.send({'stopped': true});
          commands.close();
        case 'done':
          parent.send({
            'done': true,
            'token': '0123456789abcdef0123456789abcdef',
          });
          commands.close();
        case 'unsupported':
          parent.send({'error': 'cancellation unsupported'});
          commands.close();
      }
    });
  };
}
