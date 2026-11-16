import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/settings_page.dart';
import 'package:ggtts_app/settings_store.dart';
import 'package:ggtts_app/read_options.dart';

class FakeStore extends SettingsStore {
  FakeStore() : super(File('/unused'));
  Map<String, Object?> document = {};
  @override
  Future<Map<String, Object?>> load() async => document;
  @override
  Future<void> save(Map<String, Object?> value) async {
    validateSettings(value);
    document = value;
  }
}

void main() {
  testWidgets('edits persist settings in a scrollable form', (tester) async {
    final store = FakeStore();
    await tester.pumpWidget(
      MaterialApp(
        home: SettingsPage(store: store, devices: const ['cpu']),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.byType(SingleChildScrollView), findsWidgets);
    await tester.enterText(
      find.widgetWithText(TextField, 'Threads (blank for default)'),
      '3',
    );
    await tester.enterText(
      find.widgetWithText(TextField, 'Cache folder (blank for default)'),
      '/cache',
    );
    await tester.ensureVisible(find.text('Save settings'));
    await tester.tap(find.text('Save settings'));
    await tester.pumpAndSettle();
    expect(store.document['load'], {'threads': 3, 'cache_dir': '/cache'});
  });
  testWidgets(
    'model selections persist, preserve other fields, and can be cleared',
    (tester) async {
      final store = FakeStore();
      store.document = {
        'load': {'threads': 2},
        'render': {'instruction': 'keep'},
      };
      await tester.pumpWidget(MaterialApp(home: SettingsPage(store: store)));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('engine')));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Kitten-TTS 2').last);
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const Key('model_path')),
        '/models/lm.gguf',
      );
      await tester.enterText(
        find.byKey(const Key('codec_path')),
        '/models/codec.gguf',
      );
      await tester.enterText(
        find.byKey(const Key('voice_path')),
        '/models/voice.json',
      );
      await tester.ensureVisible(find.text('Save settings'));
      await tester.tap(find.text('Save settings'));
      await tester.pumpAndSettle();
      expect(store.document, {
        'load': {
          'threads': 2,
          'engine': 'kitten-tts-2',
          'model': '/models/lm.gguf',
          'codec': '/models/codec.gguf',
          'voice': '/models/voice.json',
        },
        'render': {'instruction': 'keep'},
      });

      final forRead = composeLoadDocument(
        store.document,
        '/cache',
        offline: true,
      );
      expect((forRead['load'] as Map)['model'], '/models/lm.gguf');
      expect((forRead['load'] as Map)['codec'], '/models/codec.gguf');
      expect((forRead['load'] as Map)['voice'], '/models/voice.json');
      expect((forRead['load'] as Map)['engine'], 'kitten-tts-2');
      expect((forRead['load'] as Map)['offline'], true);

      await tester.pumpWidget(const SizedBox());
      await tester.pumpWidget(MaterialApp(home: SettingsPage(store: store)));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('engine')));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Automatic').last);
      await tester.pumpAndSettle();
      for (final key in ['model_path', 'codec_path', 'voice_path']) {
        await tester.enterText(find.byKey(Key(key)), '');
      }
      await tester.ensureVisible(find.text('Save settings'));
      await tester.tap(find.text('Save settings'));
      await tester.pumpAndSettle();
      expect(store.document, {
        'load': {'threads': 2},
        'render': {'instruction': 'keep'},
      });
    },
  );

  testWidgets('document-provider URI is refused without replacing settings', (
    tester,
  ) async {
    final store = FakeStore();
    store.document = {
      'load': {'model': '/safe/model.gguf'},
    };
    await tester.pumpWidget(MaterialApp(home: SettingsPage(store: store)));
    await tester.pumpAndSettle();
    await tester.enterText(
      find.byKey(const Key('model_path')),
      'content://provider/model',
    );
    await tester.ensureVisible(find.text('Save settings'));
    await tester.tap(find.text('Save settings'));
    await tester.pumpAndSettle();
    expect(
      find.textContaining('cannot be a document-provider link'),
      findsOneWidget,
    );
    expect(store.document, {
      'load': {'model': '/safe/model.gguf'},
    });
  });
}
