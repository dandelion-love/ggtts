import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/epub_import.dart';

class InspectorProcess implements Process {
  InspectorProcess(this.out, {this.err = const []});
  final List<List<int>> out;
  final List<List<int>> err;
  bool killed = false;
  @override
  Stream<List<int>> get stdout => Stream.fromIterable(out);
  @override
  Stream<List<int>> get stderr => Stream.fromIterable(err);
  @override
  Future<int> get exitCode async => 0;
  @override
  bool kill([ProcessSignal signal = ProcessSignal.sigterm]) {
    killed = true;
    return true;
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  test('reads planned blocks in chapter order and skips front matter', () {
    final book = ImportedBook.fromJson(
      jsonEncode({
        'format': 'ggtts-book-chapters/1',
        'chapters': [
          {
            'title': 'Front',
            'skippable': true,
            'blocks': [
              {'text': 'Do not read'},
            ],
            'chunks': [
              {'paragraph': 0},
            ],
          },
          {
            'title': 'One',
            'skippable': false,
            'blocks': [
              {'text': '甲乙。'},
              {'text': '***'},
              {'text': '丙丁。'},
            ],
            'chunks': [
              {'paragraph': 0},
              {'paragraph': 2},
            ],
          },
          {
            'title': 'Two',
            'skippable': false,
            'blocks': [
              {'text': '第二章。'},
            ],
            'chunks': [
              {'paragraph': 0},
            ],
          },
        ],
      }),
    );
    expect(book.text, '甲乙。\n丙丁。\n第二章。');
    expect(book.chapterTitles, ['One', 'Two']);
    expect(book.chapterChunks, [2, 1]);
  });

  test('rejects an invalid chapter plan', () {
    final doc = jsonEncode({
      'format': 'ggtts-book-chapters/1',
      'chapters': [
        {
          'skippable': false,
          'blocks': [
            {'text': 'a'},
          ],
          'chunks': [
            {'paragraph': 3},
          ],
        },
      ],
    });
    expect(() => ImportedBook.fromJson(doc), throwsFormatException);
    expect(() => ImportedBook.fromJson('{}'), throwsFormatException);
  });

  test(
    'shared import bounds input and helper output, and cleans the snapshot',
    () async {
      final dir = await Directory.systemTemp.createTemp('epub-limits-test-');
      addTearDown(() => dir.delete(recursive: true));
      final file = File('${dir.path}${Platform.pathSeparator}input.epub');
      final helper = File('${dir.path}${Platform.pathSeparator}helper');
      await file.writeAsBytes([1, 2, 3]);
      await helper.writeAsString('fixture');
      var starts = 0;
      InspectorProcess? child;
      String? snapshot;
      Future<Process> start(String executable, List<String> args) async {
        starts++;
        snapshot = args[1];
        expect(await File(snapshot!).readAsBytes(), [1, 2, 3]);
        return child!;
      }

      Future<ImportedBook> run(EpubImportLimits limits) => inspectEpub(
        file.path,
        limits: limits,
        helperPath: helper.path,
        start: start,
      );
      await expectLater(
        run(const EpubImportLimits(inputBytes: 2)),
        throwsFormatException,
      );
      expect(
        starts,
        0,
        reason: 'an oversized source must not start the helper',
      );
      child = InspectorProcess([List.filled(9, 65)]);
      await expectLater(
        run(const EpubImportLimits(outputBytes: 8)),
        throwsFormatException,
      );
      expect(child.killed, isTrue);
      expect(await File(snapshot!).exists(), isFalse);
      child = InspectorProcess([utf8.encode('{}')], err: [List.filled(9, 65)]);
      await expectLater(
        run(const EpubImportLimits(errorBytes: 8)),
        throwsFormatException,
      );
      expect(child.killed, isTrue);
      expect(await File(snapshot!).exists(), isFalse);
      final data = jsonEncode({
        'format': 'ggtts-book-chapters/1',
        'chapters': [
          {
            'title': 'One',
            'skippable': false,
            'blocks': [
              {'text': 'abc'},
            ],
            'chunks': [
              {'paragraph': 0},
            ],
          },
        ],
      });
      child = InspectorProcess([utf8.encode(data)]);
      await expectLater(
        run(const EpubImportLimits(textCodeUnits: 3)),
        throwsFormatException,
      );
      expect(await File(snapshot!).exists(), isFalse);
      child = InspectorProcess([utf8.encode(data)]);
      final result = await run(const EpubImportLimits(textCodeUnits: 4));
      expect(result.text, 'abc');
      expect(await File(snapshot!).exists(), isFalse);
      expect(await file.readAsBytes(), [1, 2, 3]);
    },
  );

  test('oversized output terminates a real inspector process', () async {
    if (!Platform.isWindows) return;
    final dir = await Directory.systemTemp.createTemp('epub-child-test-');
    addTearDown(() => dir.delete(recursive: true));
    final file = File('${dir.path}${Platform.pathSeparator}input.epub');
    final helper = File('${dir.path}${Platform.pathSeparator}helper');
    await file.writeAsBytes([1]);
    await helper.writeAsString('fixture');
    Process? child;
    String? snapshot;
    Future<Process> start(String executable, List<String> args) async {
      snapshot = args[1];
      child = await Process.start('powershell.exe', [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        "[Console]::Out.Write('x' * 2048); Start-Sleep -Seconds 30",
      ]);
      return child!;
    }

    await expectLater(
      inspectEpub(
        file.path,
        limits: const EpubImportLimits(outputBytes: 128),
        helperPath: helper.path,
        start: start,
      ).timeout(const Duration(seconds: 10)),
      throwsFormatException,
    );
    expect(await child!.exitCode.timeout(const Duration(seconds: 3)), isNot(0));
    expect(await File(snapshot!).exists(), isFalse);
  });
}
