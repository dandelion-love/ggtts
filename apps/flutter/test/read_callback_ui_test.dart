import 'dart:ffi';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/main.dart';

typedef _SetC = Void Function(Int32);
typedef _SetD = void Function(int);
typedef _GetC = Int32 Function();
typedef _GetD = int Function();

void main() {
  testWidgets(
    'callback failure is visible, discards partial output, and next read works',
    (tester) async {
      final lib = DynamicLibrary.open(Platform.environment['GGTTS_LIB']!);
      final mode = lib.lookupFunction<_SetC, _SetD>('fake_mode');
      final pushes = lib.lookupFunction<_GetC, _GetD>('fake_pushes');
      final root = Directory.systemTemp.createTempSync('ggtts-callback-ui-');
      Future<void> waitFor(Finder finder) async {
        for (var i = 0; i < 500 && finder.evaluate().isEmpty; i++) {
          await tester.runAsync(
            () => Future<void>.delayed(const Duration(milliseconds: 10)),
          );
          await tester.pump();
        }
        expect(finder, findsOneWidget);
      }

      Future<void> tap(String label) async {
        await tester.ensureVisible(find.text(label));
        await tester.tap(find.text(label));
        await tester.pump();
      }

      try {
        mode(4);
        await tester.pumpWidget(
          MaterialApp(
            home: ReadPage(
              prepare: (_) async => {
                'schema': <String, Object?>{},
                'capabilities': <String, Object?>{},
              },
              loadSettings: () async => {},
              readDirectory: () async => root,
            ),
          ),
        );
        await tap('Prepare model (download if needed)');
        await waitFor(find.text('Model ready.'));
        await tap('Speak');
        await waitFor(find.text('Read failed.'));
        expect(
          find.textContaining('Audio sample rate changed during the read.'),
          findsOneWidget,
        );
        expect(find.text('Save read'), findsNothing);
        expect(
          pushes(),
          greaterThan(0),
          reason: 'first callback reached audio',
        );
        expect(root.listSync(), isEmpty);
        mode(2);
        await tap('Speak');
        await waitFor(
          find.text('Audio submitted. Save the read or discard it.'),
        );
        await waitFor(find.text('Discard read'));
        await tap('Discard read');
        await waitFor(find.text('Read discarded.'));
      } finally {
        await tester.pumpWidget(const SizedBox());
        root.deleteSync(recursive: true);
      }
    },
    skip:
        !(Platform.isLinux &&
            Platform.environment["GGTTS_READ_NATIVE_FIXTURE"] == "1"),
  );
}
