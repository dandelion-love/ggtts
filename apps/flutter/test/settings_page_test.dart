import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/settings_page.dart';
import 'package:ggtts_app/settings_store.dart';

void main() {
  testWidgets('edits persist the engine settings document', (tester) async {
    await tester.runAsync(() async {
      final root = await Directory.systemTemp.createTemp(
        'ggtts-settings-page-',
      );
      addTearDown(() async {
        if (await root.exists()) await root.delete(recursive: true);
      });
      final store = SettingsStore(File('${root.path}/options.json'));
      await tester.pumpWidget(
        MaterialApp(
          home: SettingsPage(store: store, devices: const ['cpu']),
        ),
      );
      await Future<void>.delayed(const Duration(milliseconds: 200));
      await tester.pumpAndSettle();
      await tester.enterText(
        find.widgetWithText(TextField, 'Threads (blank for default)'),
        '3',
      );
      await tester.enterText(
        find.widgetWithText(TextField, 'Cache folder (blank for default)'),
        '/cache',
      );
      await tester.tap(find.text('Save settings'));
      await Future<void>.delayed(const Duration(milliseconds: 200));
      await tester.pumpAndSettle();
      expect((await store.load())['load'], {
        'threads': 3,
        'cache_dir': '/cache',
      });
    });
  });
}
