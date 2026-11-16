import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
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

class SharedSource extends FakeSource {
  @override
  Future<List<ListeningSession>> sessions() async => [
    ListeningSession('s', 'Assigned', [
      ListeningClip(
        'c',
        'A中😀𠀀，文',
        Uri.parse('https://audio.example/signed?secret=x'),
      ),
    ]),
  ];
  @override
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip) async =>
      ListeningAudio(
        Uint8List.fromList(
          base64Decode(
            'ZkxhQ4AAACIAAAAAAAAAAAAAAQAAAAAAAAEAAAAAAAAAAAAAAAAAAAAA',
          ),
        ),
        refreshed: false,
      );
}

void main() {
  testWidgets('shared page result queues and retries with same key', (
    tester,
  ) async {
    final dir = Directory.systemTemp.createTempSync('feedback-widget-');
    addTearDown(() => dir.deleteSync(recursive: true));
    final source = SharedSource();
    final queue = MemoryQueue();
    await tester.pumpWidget(
      MaterialApp(
        home: ListeningPage(
          sourceFactory: (_) => source,
          queue: queue,
          feedbackDirectory: () async => dir,
        ),
      ),
    );
    await tester.enterText(
      find.byKey(const Key('listening-source')),
      'https://listener.example',
    );
    await tester.enterText(find.byKey(const Key('listening-user')), 'listener');
    await tester.enterText(
      find.byKey(const Key('listening-password')),
      'secret',
    );
    await tester.tap(find.text('Sign in'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Assigned'));
    await tester.pumpAndSettle();
    await tester.runAsync(() async {
      await tester.tap(find.text('A中😀𠀀，文'));
      await Future<void>.delayed(const Duration(milliseconds: 100));
    });
    await tester.pumpAndSettle();
    await tester.runAsync(() async {
      await tester.tap(find.text('Prepare shared feedback page'));
      await Future<void>.delayed(const Duration(milliseconds: 300));
    });
    await tester.pumpAndSettle();
    final state = dir.listSync().whereType<File>().singleWhere(
      (f) => f.path.endsWith('.json'),
    );
    final saved = jsonDecode(state.readAsStringSync()) as Map;
    final canonical = saved['canonical'] as String;
    final input = jsonDecode(canonical) as Map;
    final html = File(saved['page_path']).readAsStringSync();
    expect(html, isNot(contains('signed?secret')));
    expect(html, isNot(contains('secret')));
    final id = input['id'] as String;
    final meta = RegExp('content="([0-9a-f]{64})"').firstMatch(html)!.group(1)!;
    await tester.tap(find.byTooltip('Sign out'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Leave'));
    await tester.pumpAndSettle();
    await tester.enterText(
      find.byKey(const Key('listening-password')),
      'secret',
    );
    await tester.tap(find.text('Sign in'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Assigned'));
    await tester.pumpAndSettle();
    await tester.runAsync(() async {
      await tester.tap(find.text('A中😀𠀀，文'));
      await Future<void>.delayed(const Duration(milliseconds: 100));
    });
    await tester.pumpAndSettle();
    expect(
      find.text('Reopened the saved feedback page binding.'),
      findsOneWidget,
    );
    final result = jsonEncode({
      'format': 'ggtts-feedback-result/1',
      'id': id,
      'input_sha256': meta,
      'note': 'page',
      'items': [
        {
          'id': 'clip',
          'marks': [0, 1, 2],
          'rating': 'accept',
          'note': 'item',
        },
      ],
    });
    await tester.enterText(
      find.byKey(const Key('shared-feedback-result')),
      result,
    );
    await tester.ensureVisible(find.text('Submit feedback'));
    await tester.tap(find.text('Submit feedback'));
    await tester.pumpAndSettle();
    expect(queue.items, hasLength(1));
    expect(source.bodies.single['marks'], [
      {'start': 1, 'end': 2, 'verdict': 'reject', 'note': null},
      {'start': 3, 'end': 4, 'verdict': 'reject', 'note': null},
      {'start': 5, 'end': 6, 'verdict': 'reject', 'note': null},
    ]);
    await tester.tap(find.byTooltip('Retry pending'));
    await tester.pumpAndSettle();
    expect(queue.items, isEmpty);
    expect(source.keys[0], source.keys[1]);
  });
}
