import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:ggtts_app/saved_reads.dart';
import 'package:ggtts_app/saved_reads_page.dart';

class FakeReads extends SavedReads {
  FakeReads() : super(Directory('/unused'));
  final pending = <PendingReadInfo>[
    PendingReadInfo(token: 'one', text: 'recover me', options: {}),
    PendingReadInfo(token: 'two', text: 'discard me', options: {}),
    PendingReadInfo(
      token: 'bad',
      text: '',
      options: {},
      problem: 'bad metadata',
    ),
  ];
  final reads = <SavedRead>[
    SavedRead(
      id: 'bad',
      text: '',
      options: {},
      audioPath: '',
      missing: false,
      savedAt: DateTime.utc(2020),
      problem: 'bad metadata',
    ),
  ];

  @override
  Future<List<SavedRead>> list() async => List.of(reads);
  @override
  Future<List<PendingReadInfo>> completedPending() async => List.of(pending);
  @override
  Future<SavedRead> publish(String token) async {
    final item = pending.firstWhere((p) => p.token == token);
    pending.remove(item);
    final saved = SavedRead(
      id: token,
      text: item.text,
      options: item.options,
      audioPath: '',
      missing: true,
      savedAt: DateTime.utc(2020),
    );
    reads.add(saved);
    return saved;
  }

  @override
  Future<void> discardPending(String token) async =>
      pending.removeWhere((p) => p.token == token);
  @override
  Future<void> delete(SavedRead read) async => reads.remove(read);
}

void main() {
  testWidgets('recover, discard and remove invalid entries', (tester) async {
    final library = FakeReads();
    await tester.pumpWidget(
      MaterialApp(home: SavedReadsPage(library: library)),
    );
    await tester.pumpAndSettle();
    expect(find.text('recover me'), findsOneWidget);
    expect(find.text('Unreadable saved read'), findsOneWidget);
    expect(find.text('Unreadable pending read'), findsOneWidget);

    final invalid = find.ancestor(
      of: find.text('Unreadable pending read'),
      matching: find.byType(ListTile),
    );
    expect(
      tester
          .widget<IconButton>(
            find.descendant(of: invalid, matching: find.widgetWithIcon(IconButton, Icons.save)),
          )
          .onPressed,
      isNull,
    );
    final first = find.ancestor(
      of: find.text('recover me'),
      matching: find.byType(ListTile),
    );
    await tester.tap(
      find.descendant(of: first, matching: find.widgetWithIcon(IconButton, Icons.save)),
    );
    await tester.pumpAndSettle();
    expect(library.reads.where((r) => r.text == 'recover me'), hasLength(1));
    expect(library.pending.where((r) => r.text == 'recover me'), isEmpty);

    final second = find.ancestor(
      of: find.text('discard me'),
      matching: find.byType(ListTile),
    );
    await tester.tap(
      find.descendant(
        of: second,
        matching: find.byTooltip('Discard pending read'),
      ),
    );
    await tester.pumpAndSettle();
    expect(library.pending.where((r) => r.text == 'discard me'), isEmpty);
    await tester.tap(find.byTooltip('Remove invalid read'));
    await tester.pumpAndSettle();
    expect(library.reads.where((r) => r.invalid), isEmpty);
  });
}
