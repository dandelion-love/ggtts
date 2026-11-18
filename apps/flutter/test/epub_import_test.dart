import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/epub_import.dart';

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
}
