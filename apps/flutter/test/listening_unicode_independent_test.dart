import 'dart:io';
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/listening_page.dart';
import 'package:ggtts_app/listening_source.dart';

class Source implements ListeningSource {
  final bodies = <Map<String, Object?>>[];
  @override
  String get identity => 'https://example.test';
  @override
  Future<void> signIn(String user, String password) async {}
  @override
  Future<List<ListeningSession>> sessions() async => [
    ListeningSession('s', 'Assigned', [
      ListeningClip('c', 'A😀é', Uri.parse('https://example.test/a')),
    ]),
  ];
  @override
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip) async =>
      ListeningAudio(Uint8List(0));
  @override
  Future<void> submit(
    String sessionId,
    Map<String, Object?> body,
    String key,
  ) async {
    bodies.add(body);
  }
}

class MemoryQueue extends ListeningQueue {
  MemoryQueue() : super(File('unused'));
  final items = <Map<String, dynamic>>[];
  @override
  Future<List<Map<String, dynamic>>> read() async => List.of(items);
  @override
  Future<void> enqueue(Map<String, dynamic> item) async {
    items.add(item);
  }

  @override
  Future<int> retry(
    ListeningSource source,
    String user, {
    bool automatic = false,
  }) async {
    for (final item in List<Map<String, dynamic>>.of(items)) {
      await source.submit(
        item['session'] as String,
        Map<String, Object?>.from(item['body'] as Map),
        item['key'] as String,
      );
      items.remove(item);
    }
    return 1;
  }
}

void main() {
  testWidgets(
    'marks Unicode scalar offsets including supplementary and combining characters',
    (tester) async {
      final source = Source();
      await tester.pumpWidget(
        MaterialApp(
          home: ListeningPage(
            sourceFactory: (_) => source,
            queue: MemoryQueue(),
            play: (_) async {},
          ),
        ),
      );
      await tester.enterText(
        find.byKey(const Key('listening-source')),
        'https://example.test',
      );
      await tester.enterText(find.byKey(const Key('listening-user')), 'alice');
      await tester.enterText(
        find.byKey(const Key('listening-password')),
        'password',
      );
      await tester.tap(find.text('Sign in'));
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 150)),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.text('Assigned'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('A😀é'));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('character-1')));
      await tester.tap(find.byKey(const Key('character-3')));
      await tester.ensureVisible(find.text('Submit marks'));
      await tester.tap(find.text('Submit marks'));
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 150)),
      );
      await tester.pumpAndSettle();
      final marks = source.bodies.single['marks'] as List;
      expect(marks.map((m) => [m['start'], m['end']]).toList(), [
        [1, 2],
        [3, 4],
      ]);
    },
  );
}
