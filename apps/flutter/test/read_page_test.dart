import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/main.dart';

class FakePrepareCancellation implements PrepareCancellation {
  @override
  int get token => 1;
  int signals = 0;
  int closes = 0;
  @override
  void signal() {
    signals++;
  }

  @override
  void close() {
    closes++;
  }
}

void main() {
  testWidgets('offline preparation uses selected paths and can read', (
    tester,
  ) async {
    Map<String, Object?>? observed;
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          loadSettings: () async => {
            'load': {
              'model': '/chosen/model.gguf',
              'codec': '/chosen/codec.gguf',
              'offline': false,
            },
          },
          prepare: (doc) async {
            observed = doc;
            return {
              'schema': <String, Object?>{},
              'capabilities': <String, Object?>{},
            };
          },
        ),
      ),
    );
    final offline = find.text('Prepare offline (selected or cached files)');
    await tester.ensureVisible(offline);
    await tester.tap(offline);
    await tester.pumpAndSettle();
    final load = observed!['load'] as Map;
    expect(load['offline'], true);
    expect(load['model'], '/chosen/model.gguf');
    expect(load['codec'], '/chosen/codec.gguf');
    expect(find.text('Model ready.'), findsOneWidget);
    expect(
      find.text('Prepare model again (download if needed)'),
      findsOneWidget,
    );
    expect(
      tester
          .widget<FilledButton>(find.widgetWithText(FilledButton, 'Speak'))
          .onPressed,
      isNotNull,
    );
  });

  testWidgets(
    'offline missing files report failure and explicit download retries',
    (tester) async {
      final seen = <bool>[];
      await tester.pumpWidget(
        MaterialApp(
          home: ReadPage(
            loadSettings: () async => {},
            prepare: (doc) async {
              final offline = (doc['load'] as Map)['offline'] as bool;
              seen.add(offline);
              if (offline) throw StateError('file unavailable offline');
              return {
                'schema': <String, Object?>{},
                'capabilities': <String, Object?>{},
              };
            },
          ),
        ),
      );
      final offline = find.text('Prepare offline (selected or cached files)');
      await tester.ensureVisible(offline);
      await tester.tap(offline);
      await tester.pumpAndSettle();
      expect(
        find.textContaining('Offline preparation failed.'),
        findsOneWidget,
      );
      expect(find.textContaining('file unavailable offline'), findsOneWidget);
      expect(
        tester
            .widget<FilledButton>(find.widgetWithText(FilledButton, 'Speak'))
            .onPressed,
        isNull,
      );
      final download = find.text('Prepare model (download if needed)');
      await tester.ensureVisible(download);
      await tester.tap(download);
      await tester.pumpAndSettle();
      expect(seen, [true, false]);
      expect(find.text('Model ready.'), findsOneWidget);
    },
  );

  testWidgets(
    'stop preparation waits for worker and rejects late success; retry works',
    (tester) async {
      final first = Completer<Map<String, Object?>>();
      final controls = <FakePrepareCancellation>[];
      var attempts = 0;
      Future<Map<String, Object?>> prepare(
        Map<String, Object?> _,
        void Function(String) notice,
      ) {
        attempts++;
        return attempts == 1
            ? first.future
            : Future.value({
                'schema': <String, Object?>{},
                'capabilities': <String, Object?>{},
              });
      }

      await tester.pumpWidget(
        MaterialApp(
          home: ReadPage(
            loadSettings: () async => {},
            prepareWithProgress: prepare,
            createPrepareCancellation: () {
              final control = FakePrepareCancellation();
              controls.add(control);
              return control;
            },
          ),
        ),
      );
      await tester.tap(find.text('Prepare model (download if needed)'));
      await tester.pump();
      await tester.tap(find.text('Stop preparation'));
      await tester.pump();
      expect(controls.single.signals, 1);
      expect(controls.single.closes, 0);
      expect(find.text('Stopping model preparation...'), findsOneWidget);
      first.complete({
        'schema': <String, Object?>{},
        'capabilities': <String, Object?>{},
      });
      await tester.pumpAndSettle();
      expect(find.text('Model preparation stopped.'), findsOneWidget);
      expect(find.text('Model ready.'), findsNothing);
      expect(controls.single.closes, 1);
      await tester.tap(find.text('Prepare model (download if needed)'));
      await tester.pumpAndSettle();
      expect(find.text('Model ready.'), findsOneWidget);
      expect(controls.length, 2);
      expect(controls.last.closes, 1);
    },
  );

  testWidgets('shows numeric download progress without private notices', (
    tester,
  ) async {
    final done = Completer<Map<String, Object?>>();
    late void Function(String) notice;
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          loadSettings: () async => {},
          prepareWithProgress: (_, onProgress) {
            notice = onProgress;
            return done.future;
          },
        ),
      ),
    );
    await tester.tap(find.text('Prepare model (download if needed)'));
    await tester.pump();
    notice('downloading https://secret.invalid into /private/path');
    await tester.pump();
    expect(find.textContaining('secret.invalid'), findsNothing);
    expect(find.textContaining('/private/path'), findsNothing);
    notice('model.gguf: 50 of 100 MB (50%), 1.0 MB/s');
    await tester.pump();
    expect(
      find.text('Downloading: 50 of 100 MB (50%), 1.0 MB/s'),
      findsOneWidget,
    );
    done.complete({
      'schema': <String, Object?>{},
      'capabilities': <String, Object?>{},
    });
    await tester.pumpAndSettle();
    expect(find.text('Model ready.'), findsOneWidget);
  });

  testWidgets(
    'model capabilities populate voices and disable unavailable steering',
    (tester) async {
      Future<Map<String, Object?>> prepare(Map<String, Object?> _) async => {
        'schema': {
          'properties': {
            'load': {
              'properties': {'speaker': {}},
            },
            'render': {
              'properties': {'steer': {}},
            },
          },
        },
        'capabilities': {
          'speaker_names': ['voice-a', 'voice-b'],
          'steering': false,
        },
      };
      await tester.pumpWidget(
        MaterialApp(
          home: ReadPage(prepare: prepare, loadSettings: () async => {}),
        ),
      );
      expect(
        tester.widget<TextField>(find.byKey(const Key('steering'))).enabled,
        false,
      );
      await tester.tap(find.text('Prepare model (download if needed)'));
      await tester.pumpAndSettle();
      expect(find.text('Model ready.'), findsOneWidget);
      expect(find.text('voice-a'), findsOneWidget);
      expect(
        tester.widget<TextField>(find.byKey(const Key('steering'))).enabled,
        false,
      );
      expect(
        tester
            .widget<DropdownButtonFormField<String>>(
              find.byKey(const Key('voice')),
            )
            .onChanged,
        isNotNull,
      );
    },
  );

  testWidgets('preparation error is shown and can be retried', (tester) async {
    var attempts = 0;
    Future<Map<String, Object?>> prepare(Map<String, Object?> _) async {
      attempts++;
      if (attempts == 1) throw StateError('offline');
      return {
        'schema': <String, Object?>{},
        'capabilities': <String, Object?>{},
      };
    }

    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(prepare: prepare, loadSettings: () async => {}),
      ),
    );
    await tester.tap(find.text('Prepare model (download if needed)'));
    await tester.pumpAndSettle();
    expect(
      tester.widget<Text>(find.byKey(const Key('error'))).data,
      contains('offline'),
    );
    await tester.tap(find.text('Prepare model (download if needed)'));
    await tester.pumpAndSettle();
    expect(find.text('Model ready.'), findsOneWidget);
    expect(attempts, 2);
  });
  testWidgets('saved settings are sent in the load options document', (
    tester,
  ) async {
    Map<String, Object?>? observed;
    Future<Map<String, Object?>> prepare(Map<String, Object?> doc) async {
      observed = doc;
      return {
        'schema': <String, Object?>{},
        'capabilities': <String, Object?>{},
      };
    }

    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          prepare: prepare,
          loadSettings: () async => {
            'load': {'device': 'cpu', 'threads': 3, 'cache_dir': '/chosen'},
          },
        ),
      ),
    );
    await tester.tap(find.text('Prepare model (download if needed)'));
    await tester.pumpAndSettle();
    expect(observed!['load'], {
      'device': 'cpu',
      'threads': 3,
      'cache_dir': '/chosen',
      'offline': false,
    });
  });
  testWidgets('prepare uses saved model and voice paths', (tester) async {
    Map<String, Object?>? observed;
    final settings = <String, Object?>{
      'load': {
        'engine': 'kitten-tts-2',
        'model': '/models/lm.gguf',
        'codec': '/models/codec.gguf',
        'voice': '/models/voice.json',
      },
    };
    await tester.pumpWidget(
      MaterialApp(
        home: ReadPage(
          prepare: (doc) async {
            observed = doc;
            return {
              'schema': <String, Object?>{},
              'capabilities': <String, Object?>{},
            };
          },
          loadSettings: () async => settings,
        ),
      ),
    );
    await tester.tap(find.text('Prepare model (download if needed)'));
    await tester.pumpAndSettle();
    final load = observed!['load'] as Map;
    expect(load['engine'], 'kitten-tts-2');
    expect(load['model'], '/models/lm.gguf');
    expect(load['codec'], '/models/codec.gguf');
    expect(load['voice'], '/models/voice.json');
    expect(load['offline'], false);
  });
}
