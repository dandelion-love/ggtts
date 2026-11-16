import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'flac.dart';
import 'ggtts.dart';

class SavedRead {
  SavedRead({
    required this.id,
    required this.text,
    required this.options,
    required this.audioPath,
    required this.missing,
    required this.savedAt,
  });
  final String id, text, audioPath;
  final Map<String, Object?> options;
  final bool missing;
  final DateTime savedAt;
}

/// An app-private library; callers supply a durable application directory.
class SavedReads {
  SavedReads(this.root);
  final Directory root;
  static final _idPattern = RegExp(r'^[0-9a-f]{32}$');
  String _path(String name) => '${root.path}${Platform.pathSeparator}$name';
  String _audio(String id) => '${_path(id)}${Platform.pathSeparator}audio.flac';

  /// Create a hidden staging directory; it is not a saved read until save().
  Future<PendingRead> begin({
    required String text,
    required Map<String, Object?> options,
  }) async {
    final optionsJson = jsonEncode(options); // snapshot before the first await
    final random = Random.secure();
    final id = List.generate(
      16,
      (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0'),
    ).join();
    await root.create(recursive: true);
    final staging = Directory(_path('.$id.tmp'));
    if (await staging.exists() || await Directory(_path(id)).exists()) {
      throw FileSystemException('saved-read ID collision', id);
    }
    await staging.create();
    try {
      await File('${staging.path}${Platform.pathSeparator}pending.json')
          .writeAsString(
            jsonEncode({
              'version': 1,
              'id': id,
              'text': text,
              'options': jsonDecode(optionsJson),
            }),
            flush: true,
          );
      return PendingRead._(staging, id);
    } catch (_) {
      await staging.delete(recursive: true);
      rethrow;
    }
  }

  /// Publish a completed worker's token only after the user selects Save.
  Future<SavedRead> publish(String token) async {
    if (!_idPattern.hasMatch(token)) {
      throw ArgumentError('invalid pending-read token');
    }
    final staging = Directory(_path('.$token.tmp'));
    if (!await File('${staging.path}${Platform.pathSeparator}ready').exists()) {
      throw StateError('pending read is not complete');
    }
    final metadata = jsonDecode(
      await File('${staging.path}${Platform.pathSeparator}pending.json')
          .readAsString(),
    ) as Map<String, dynamic>;
    if (metadata['version'] != 1 ||
        metadata['id'] != token ||
        !await File('${staging.path}${Platform.pathSeparator}audio.flac')
            .exists()) {
      throw FormatException('invalid pending read');
    }
    final savedAt = DateTime.now().toUtc();
    final options = (metadata['options'] as Map).cast<String, Object?>();
    final text = metadata['text'] as String;
    await File('${staging.path}${Platform.pathSeparator}read.json')
        .writeAsString(
          jsonEncode({
            'version': 1,
            'id': token,
            'text': text,
            'options': options,
            'saved_at': savedAt.toIso8601String(),
          }),
          flush: true,
        );
    if (await Directory(_path(token)).exists()) {
      throw StateError('saved-read ID collision');
    }
    await staging.rename(_path(token));
    return SavedRead(
      id: token,
      text: text,
      options: options,
      audioPath: _audio(token),
      missing: false,
      savedAt: savedAt,
    );
  }

  /// Discard a completed staging token; never follows a caller-supplied path.
  Future<void> discardPending(String token) async {
    if (!_idPattern.hasMatch(token)) {
      throw ArgumentError('invalid pending-read token');
    }
    final staging = Directory(_path('.$token.tmp'));
    if (await staging.exists()) await staging.delete(recursive: true);
  }

  Future<List<SavedRead>> list() async {
    if (!await root.exists()) return [];
    final reads = <SavedRead>[];
    await for (final entity in root.list(followLinks: false)) {
      if (entity is! Directory) continue;
      final id = entity.uri.pathSegments.where((s) => s.isNotEmpty).last;
      if (!_idPattern.hasMatch(id)) continue;
      final metadata = jsonDecode(
        await File('${entity.path}${Platform.pathSeparator}read.json')
            .readAsString(),
      ) as Map<String, dynamic>;
      if (metadata['version'] != 1 || metadata['id'] != id) {
        throw FormatException('invalid saved read metadata: $id');
      }
      final path = _audio(id);
      reads.add(
        SavedRead(
          id: id,
          text: metadata['text'] as String,
          options: (metadata['options'] as Map).cast<String, Object?>(),
          audioPath: path,
          missing: !await File(path).exists(),
          savedAt: DateTime.parse(metadata['saved_at'] as String),
        ),
      );
    }
    reads.sort((a, b) => b.savedAt.compareTo(a.savedAt));
    return reads;
  }

  Future<void> delete(SavedRead read) async {
    if (!_idPattern.hasMatch(read.id)) {
      throw ArgumentError('invalid saved-read ID');
    }
    final dir = Directory(_path(read.id));
    if (await dir.exists()) await dir.delete(recursive: true);
  }
}

class PendingRead {
  PendingRead._(this._staging, this._id)
    : _writer = FlacWriter(
        File('${_staging.path}${Platform.pathSeparator}audio.flac'),
      );
  final Directory _staging;
  final String _id;
  final FlacWriter _writer;
  bool _finished = false, _done = false;

  /// Write synchronously from the render callback on the worker isolate.
  void append(Pcm piece) {
    if (_finished || _done) throw StateError('read is not accepting audio');
    _writer.append(piece);
  }

  /// Close the file and return a sendable token for the UI isolate.
  String finish() {
    if (_finished || _done) throw StateError('read is already finished');
    _writer.finish();
    File('${_staging.path}${Platform.pathSeparator}ready')
        .writeAsStringSync('ready', flush: true);
    _finished = true;
    return _id;
  }

  Future<void> discard() async {
    if (_done) return;
    _writer.close();
    if (await _staging.exists()) await _staging.delete(recursive: true);
    _done = true;
  }
}
