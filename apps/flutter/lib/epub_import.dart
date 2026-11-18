import 'dart:convert';
import 'dart:io';

/// Read EPUBs with the existing ggtts-book parser, preserving chapter order.
class ImportedBook {
  const ImportedBook(this.text, this.chapterTitles, this.chapterChunks);
  final String text;
  final List<String> chapterTitles;
  final List<int> chapterChunks;

  static ImportedBook fromJson(String source) {
    final doc = jsonDecode(source);
    if (doc is! Map ||
        doc['format'] != 'ggtts-book-chapters/1' ||
        doc['chapters'] is! List) {
      throw const FormatException('Invalid EPUB inspection result');
    }
    final texts = <String>[], names = <String>[];
    final counts = <int>[];
    for (final chapter in doc['chapters'] as List) {
      if (chapter is! Map ||
          chapter['skippable'] is! bool ||
          chapter['blocks'] is! List ||
          chapter['chunks'] is! List) {
        throw const FormatException('Invalid EPUB chapter');
      }
      if (chapter['skippable'] == true || (chapter['chunks'] as List).isEmpty)
        continue;
      final blocks = <String>[];
      final spoken = <int>{};
      for (final chunk in chapter['chunks'] as List) {
        if (chunk is! Map || chunk['paragraph'] is! int)
          throw const FormatException('Invalid EPUB plan');
        spoken.add(chunk['paragraph'] as int);
      }
      final allBlocks = chapter['blocks'] as List;
      if (spoken.any((i) => i < 0 || i >= allBlocks.length))
        throw const FormatException('Invalid EPUB plan');
      for (var i = 0; i < allBlocks.length; i++) {
        final block = allBlocks[i];
        if (block is! Map || block['text'] is! String)
          throw const FormatException('Invalid EPUB text');
        final text = (block['text'] as String).trim();
        if (spoken.contains(i) && text.isNotEmpty) blocks.add(text);
      }
      if (blocks.isEmpty) continue;
      names.add(
        chapter['title'] is String
            ? chapter['title'] as String
            : 'Chapter ${names.length + 1}',
      );
      counts.add((chapter['chunks'] as List).length);
      texts.addAll(blocks);
    }
    if (texts.isEmpty)
      throw const FormatException('The EPUB has no readable chapters');
    return ImportedBook(texts.join('\n'), names, counts);
  }
}

Future<ImportedBook> inspectEpub(String file) async {
  if (!Platform.isWindows)
    throw UnsupportedError('EPUB import is available on Windows');
  final helper = File(
    '${File(Platform.resolvedExecutable).parent.path}${Platform.pathSeparator}ggtts-book.exe',
  );
  if (!await helper.exists())
    throw StateError('The EPUB reader is missing from this Windows build');
  final result = await Process.run(
    helper.path,
    ['inspect', file],
    stdoutEncoding: utf8,
    stderrEncoding: utf8,
  );
  if (result.exitCode != 0)
    throw const FormatException('This EPUB could not be opened');
  return ImportedBook.fromJson(result.stdout as String);
}
