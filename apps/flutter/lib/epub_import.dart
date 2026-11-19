import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Conservative app limits, separate from the book parser's inflate limits.
class EpubImportLimits {
  const EpubImportLimits({
    this.inputBytes = 32 * 1024 * 1024,
    this.outputBytes = 16 * 1024 * 1024,
    this.errorBytes = 64 * 1024,
    this.textCodeUnits = 1000000,
  });
  final int inputBytes;
  final int outputBytes;
  final int errorBytes;
  final int textCodeUnits;
}

typedef InspectorStarter = Future<Process> Function(String, List<String>);

/// Read EPUBs with the existing ggtts-book parser, preserving chapter order.
class ImportedBook {
  const ImportedBook(this.text, this.chapterTitles, this.chapterChunks);
  final String text;
  final List<String> chapterTitles;
  final List<int> chapterChunks;

  static ImportedBook fromJson(
    String source, {
    int maxTextCodeUnits = 1000000,
  }) {
    final doc = jsonDecode(source);
    if (doc is! Map ||
        doc['format'] != 'ggtts-book-chapters/1' ||
        doc['chapters'] is! List) {
      throw const FormatException('Invalid EPUB inspection result');
    }
    final texts = <String>[], names = <String>[];
    final counts = <int>[];
    var textUnits = 0;
    for (final chapter in doc['chapters'] as List) {
      if (chapter is! Map ||
          chapter['skippable'] is! bool ||
          chapter['blocks'] is! List ||
          chapter['chunks'] is! List) {
        throw const FormatException('Invalid EPUB chapter');
      }
      if (chapter['skippable'] == true || (chapter['chunks'] as List).isEmpty) {
        continue;
      }
      final blocks = <String>[];
      final spoken = <int>{};
      for (final chunk in chapter['chunks'] as List) {
        if (chunk is! Map || chunk['paragraph'] is! int) {
          throw const FormatException('Invalid EPUB plan');
        }
        spoken.add(chunk['paragraph'] as int);
      }
      final allBlocks = chapter['blocks'] as List;
      if (spoken.any((i) => i < 0 || i >= allBlocks.length)) {
        throw const FormatException('Invalid EPUB plan');
      }
      for (var i = 0; i < allBlocks.length; i++) {
        final block = allBlocks[i];
        if (block is! Map || block['text'] is! String) {
          throw const FormatException('Invalid EPUB text');
        }
        final text = (block['text'] as String).trim();
        if (spoken.contains(i) && text.isNotEmpty) {
          textUnits += text.length + 1;
          if (textUnits > maxTextCodeUnits) {
            throw const FormatException('EPUB text exceeds the app limit');
          }
          blocks.add(text);
        }
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
    if (texts.isEmpty) {
      throw const FormatException('The EPUB has no readable chapters');
    }
    return ImportedBook(texts.join('\n'), names, counts);
  }
}

Future<ImportedBook> inspectEpub(
  String file, {
  EpubImportLimits limits = const EpubImportLimits(),
  InspectorStarter? start,
  String? helperPath,
}) async {
  if (!Platform.isWindows && start == null) {
    throw UnsupportedError('EPUB import is available on Windows');
  }
  final helper = File(
    helperPath ??
        '${File(Platform.resolvedExecutable).parent.path}${Platform.pathSeparator}ggtts-book.exe',
  );
  if (!await helper.exists()) {
    throw StateError('The EPUB reader is missing from this Windows build');
  }
  final source = File(file);
  if (await source.length() > limits.inputBytes) {
    throw const FormatException('EPUB file exceeds the app limit');
  }
  // Give the helper a bounded snapshot: an input that grows after the length
  // check cannot make it read an unbounded file, and the original stays intact.
  final temp = await Directory.systemTemp.createTemp('ggtts-epub-');
  Process? process;
  try {
    final snapshot = File('${temp.path}${Platform.pathSeparator}book.epub');
    final sink = snapshot.openWrite();
    var count = 0;
    try {
      await for (final bytes in source.openRead()) {
        count += bytes.length;
        if (count > limits.inputBytes) {
          throw const FormatException('EPUB file exceeds the app limit');
        }
        sink.add(bytes);
      }
    } finally {
      await sink.close();
    }
    process = await (start ?? Process.start)(helper.path, [
      'inspect',
      snapshot.path,
    ]);
    final child = process;
    Future<Uint8List> collect(Stream<List<int>> stream, int limit) async {
      final buffer = BytesBuilder(copy: false);
      var size = 0;
      try {
        await for (final bytes in stream) {
          size += bytes.length;
          if (size > limit) {
            throw const FormatException(
              'EPUB inspector output exceeds the app limit',
            );
          }
          buffer.add(bytes);
        }
      } catch (_) {
        child.kill();
        rethrow;
      }
      return buffer.takeBytes();
    }

    final output = await Future.wait<Object>([
      collect(child.stdout, limits.outputBytes),
      collect(child.stderr, limits.errorBytes),
      child.exitCode,
    ]);
    if (output[2] != 0) {
      throw const FormatException('This EPUB could not be opened');
    }
    return ImportedBook.fromJson(
      utf8.decode(output[0] as Uint8List),
      maxTextCodeUnits: limits.textCodeUnits,
    );
  } finally {
    process?.kill();
    await temp.delete(recursive: true);
  }
}
