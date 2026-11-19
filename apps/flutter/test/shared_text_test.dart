import 'dart:async';
import 'dart:collection';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/main.dart';
import 'package:ggtts_app/shared_text.dart';

class FakeSharedTextInbox implements SharedTextInbox {
  final queue = Queue<String>();
  void Function()? listener;
  bool closed = false;
  void send(String text) {
    queue.add(text);
    listener?.call();
  }

  @override
  Future<String?> take() async => queue.isEmpty ? null : queue.removeFirst();
  @override
  void onAvailable(void Function()? callback) {
    listener = callback;
  }

  @override
  void close() {
    listener = null;
    closed = true;
  }
}

class RacingInbox implements SharedTextInbox {
  final first = Completer<String?>();
  void Function()? listener;
  var takes = 0;
  @override
  Future<String?> take() async => ++takes == 1 ? first.future : 'late share';
  @override
  void onAvailable(void Function()? callback) {
    listener = callback;
  }

  @override
  void close() {
    listener = null;
  }
}

void main() {
  testWidgets('notification during initial empty poll is not lost', (
    tester,
  ) async {
    final inbox = RacingInbox();
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          sharedTextInbox: inbox,
          loadSettings: () async => {},
          prepare: (_) async => {},
        ),
      ),
    );
    await tester.pump();
    inbox.listener!();
    inbox.first.complete(null);
    await tester.pumpAndSettle();
    expect(inbox.takes, 2);
    expect(find.text('Shared text: late share'), findsOneWidget);
  });

  testWidgets('method channel delivers notification and pulls Unicode text', (
    tester,
  ) async {
    const channel = MethodChannel('ggtts/shared_text');
    final text = String.fromCharCodes([0x4eca, 0x65e5, 0x20000]);
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(channel, (call) async {
      expect(call.method, 'take');
      return text;
    });
    final inbox = AndroidSharedTextInbox();
    var notified = 0;
    inbox.onAvailable(() => notified++);
    expect(await inbox.take(), text);
    final completed = Completer<void>();
    messenger.handlePlatformMessage(
      'ggtts/shared_text',
      const StandardMethodCodec().encodeMethodCall(
        const MethodCall('available'),
      ),
      (_) => completed.complete(),
    );
    await completed.future;
    expect(notified, 1);
    inbox.close();
    messenger.setMockMethodCallHandler(channel, null);
  });

  testWidgets('queued warm shares are shown in order', (tester) async {
    final inbox = FakeSharedTextInbox();
    inbox.send('first');
    inbox.send('second');
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          sharedTextInbox: inbox,
          loadSettings: () async => {},
          prepare: (_) async => {},
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('Shared text: first'), findsOneWidget);
    await tester.tap(find.text('Discard shared text'));
    await tester.pumpAndSettle();
    expect(find.text('Shared text: second'), findsOneWidget);
    await tester.tap(find.text('Use shared text'));
    await tester.pumpAndSettle();
    expect(
      tester.widget<TextField>(find.byType(TextField).first).controller!.text,
      'second',
    );
  });

  testWidgets('cold and warm shares preserve draft until chosen', (
    tester,
  ) async {
    final inbox = FakeSharedTextInbox();
    final first = String.fromCharCodes([0x4eca, 0x65e5, 0x20000]);
    final second = String.fromCharCodes([0x9999, 0x6e2f, 0x1f600]);
    inbox.send(first);
    var preparations = 0;
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          sharedTextInbox: inbox,
          loadSettings: () async => {},
          prepare: (_) async {
            preparations++;
            return {};
          },
        ),
      ),
    );
    await tester.pumpAndSettle();
    TextField field() => tester.widget<TextField>(find.byType(TextField).first);
    final original = field().controller!.text;
    expect(original, isNot(first));
    expect(find.textContaining(first), findsOneWidget);
    await tester.tap(find.text('Discard shared text'));
    await tester.pumpAndSettle();
    expect(field().controller!.text, original);

    inbox.send(second);
    await tester.pumpAndSettle();
    expect(field().controller!.text, original);
    await tester.tap(find.text('Use shared text'));
    await tester.pumpAndSettle();
    expect(field().controller!.text, second);
    expect(preparations, 0);
    await tester.pumpWidget(const SizedBox());
    expect(inbox.closed, isTrue);
  });

  testWidgets('shared text cannot replace a draft while preparation is busy', (
    tester,
  ) async {
    final inbox = FakeSharedTextInbox();
    final done = Completer<Map<String, Object?>>();
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          sharedTextInbox: inbox,
          loadSettings: () async => {},
          prepare: (_) => done.future,
        ),
      ),
    );
    await tester.tap(find.text('Prepare model (download if needed)'));
    await tester.pump();
    inbox.send('new text');
    await tester.pump();
    final use = tester.widget<TextButton>(
      find.widgetWithText(TextButton, 'Use shared text'),
    );
    expect(use.onPressed, isNull);
    done.complete({
      'schema': <String, Object?>{},
      'capabilities': <String, Object?>{},
    });
    await tester.pumpAndSettle();
    expect(
      tester
          .widget<TextButton>(
            find.widgetWithText(TextButton, 'Use shared text'),
          )
          .onPressed,
      isNotNull,
    );
  });
}
