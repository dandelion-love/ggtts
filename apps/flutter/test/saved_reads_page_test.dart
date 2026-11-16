import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/ggtts.dart';
import 'package:ggtts_app/saved_reads.dart';
import 'package:ggtts_app/saved_reads_page.dart';

void main() {
  testWidgets('a missing audio file is marked and can be removed', (
    tester,
  ) async {
    await tester.runAsync(() async {
      final root = await Directory.systemTemp.createTemp('ggtts-read-page-');
      addTearDown(() async {
        if (await root.exists()) await root.delete(recursive: true);
      });
      final library = SavedReads(root);
      final pending = await library.begin(
        text: 'sample',
        options: {'load_document': {}},
      );
      pending.append(Pcm(Float32List.fromList([0.2, -0.2]), 24000));
      final read = await library.publish(pending.finish());
      await File(read.audioPath).delete();
      await tester.pumpWidget(
        MaterialApp(home: SavedReadsPage(library: library)),
      );
      await Future<void>.delayed(const Duration(milliseconds: 200));
      await tester.pumpAndSettle();
      expect(find.text('Audio missing'), findsOneWidget);
      expect(
        tester
            .widget<IconButton>(
              find.byWidgetPredicate(
                (w) => w is IconButton && w.tooltip == 'Play',
              ),
            )
            .onPressed,
        isNull,
      );
      await tester.tap(find.byTooltip('Remove missing read'));
      await Future<void>.delayed(const Duration(milliseconds: 200));
      await tester.pumpAndSettle();
      expect(find.text('No saved reads yet.'), findsOneWidget);
    });
  });
}
