import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/saved_reads.dart';
import 'package:ggtts_app/saved_reads_page.dart';

import 'recovery_page_test.dart' show FakeReads;

void main() {
  testWidgets('missing audio is marked and can be removed', (tester) async {
    final library = FakeReads();
    library.reads.add(
      SavedRead(
        id: 'missing',
        text: 'sample',
        options: {},
        audioPath: '',
        missing: true,
        savedAt: DateTime.utc(2020),
      ),
    );
    await tester.pumpWidget(
      MaterialApp(home: SavedReadsPage(library: library)),
    );
    await tester.pumpAndSettle();
    expect(find.text('Audio missing'), findsOneWidget);
    final tile = find.ancestor(
      of: find.text('sample'),
      matching: find.byType(ListTile),
    );
    final play = find.descendant(
      of: tile,
      matching: find.widgetWithIcon(IconButton, Icons.play_arrow),
    );
    expect(tester.widget<IconButton>(play).onPressed, isNull);
    await tester.tap(find.byTooltip('Remove missing read'));
    await tester.pumpAndSettle();
    expect(find.text('sample'), findsNothing);
  });
}
