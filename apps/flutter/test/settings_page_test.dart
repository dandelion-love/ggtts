import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/settings_page.dart';
import 'package:ggtts_app/settings_store.dart';

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
}
