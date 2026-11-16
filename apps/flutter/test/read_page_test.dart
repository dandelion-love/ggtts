import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/main.dart';

void main() {
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
    expect(find.textContaining('offline'), findsOneWidget);
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
}
