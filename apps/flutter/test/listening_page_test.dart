import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/listening_page.dart';
import 'package:ggtts_app/listening_source.dart';

class FakeSource implements ListeningSource {
  @override
  String get identity => 'https://listener.example';
  bool fail = true;
  final keys = <String>[];
  final bodies = <Map<String, Object?>>[];
  @override
  Future<void> signIn(String user, String password) async {
    expect(user, 'listener');
    expect(password, 'secret');
  }

  @override
  Future<List<ListeningSession>> sessions() async => [
    ListeningSession('s', 'Assigned', [
      ListeningClip(
        'c',
        'A\u{1F600}B',
        Uri.parse('https://audio.example/signed?secret=x'),
      ),
    ]),
  ];
  @override
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip) async =>
      ListeningAudio(Uint8List.fromList([1, 2]), refreshed: true);
  @override
  Future<void> submit(
    String sessionId,
    Map<String, Object?> body,
    String key,
  ) async {
    keys.add(key);
    bodies.add(body);
    if (fail) {
      fail = false;
      throw ListeningException('offline');
    }
  }
}

class MemoryQueue extends ListeningQueue {
  MemoryQueue() : super(File('unused'));
  List<Map<String, dynamic>> items = [];
  @override
  Future<List<Map<String, dynamic>>> read() async => List.of(items);
  @override
  Future<void> write(List<Map<String, dynamic>> pending) async {
    items = List.of(pending);
  }
}

void main() {
  testWidgets(
    'listener marks scalar offsets and retries the same queued submission',
    (tester) async {
      final source = FakeSource();
      final queue = MemoryQueue();
      var played = false;
      await tester.pumpWidget(
        MaterialApp(
          home: ListeningPage(
            sourceFactory: (_) => source,
            queue: queue,
            play: (bytes) async {
              played = bytes.length == 2;
            },
          ),
        ),
      );
      await tester.enterText(
        find.byKey(const Key('listening-source')),
        'https://listener.example',
      );
      await tester.enterText(
        find.byKey(const Key('listening-user')),
        'listener',
      );
      await tester.enterText(
        find.byKey(const Key('listening-password')),
        'secret',
      );
      await tester.tap(find.text('Sign in'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Assigned'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('A\u{1F600}B'));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('character-1')));
      await tester.enterText(find.byType(TextField).last, 'keep this note');
      await tester.tap(find.text('All clips'));
      await tester.pumpAndSettle();
      expect(find.text('Discard unsent marks?'), findsOneWidget);
      await tester.tap(find.text('Keep editing'));
      await tester.pumpAndSettle();
      expect(find.text('keep this note'), findsOneWidget);
      await tester.tap(find.byTooltip('Sign out'));
      await tester.pumpAndSettle();
      expect(find.text('Discard unsent marks?'), findsOneWidget);
      await tester.tap(find.text('Keep editing'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Play clip'));
      await tester.pumpAndSettle();
      expect(played, isTrue);
      expect(
        find.text('Audio link expired; refreshed and playing.'),
        findsOneWidget,
      );
      await tester.ensureVisible(find.text('Submit marks'));
      await tester.tap(find.text('Submit marks'));
      await tester.pumpAndSettle();
      expect(
        find.text('offline Marks are queued; use Retry pending when ready.'),
        findsOneWidget,
      );
      final stored = jsonEncode(await queue.read());
      expect(stored, isNot(contains('secret')));
      expect(stored, isNot(contains('signed?')));
      final mark = (source.bodies.single['marks'] as List).single as Map;
      expect(mark['start'], 1);
      expect(mark['end'], 2);
      await tester.tap(find.byTooltip('Retry pending'));
      await tester.pumpAndSettle();
      expect(source.keys.length, 2);
      expect(source.keys.first, source.keys.last);
      expect(await queue.read(), isEmpty);
    },
  );
}
