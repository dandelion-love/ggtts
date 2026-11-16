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
    this.problem,
  });
  final String id, text, audioPath;
  final Map<String, Object?> options;
  final bool missing;
  final DateTime savedAt;

  /// A metadata failure; invalid entries can still be explicitly removed.
  final String? problem;
  bool get invalid => problem != null;
}

/// A completed stage recoverable after the rendering isolate or app exits.
class PendingReadInfo {
  PendingReadInfo({
    required this.token,
    required this.text,
    required this.options,
    this.problem,
  });
  final String token, text;
  final Map<String, Object?> options;
  final String? problem;
  bool get invalid => problem != null;
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

  static Future<FileSystemEntityType> _type(String path) =>
      FileSystemEntity.type(path, followLinks: false);

  Future<void> _requireDirectory(String path) async {
    if (await _type(path) != FileSystemEntityType.directory) {
      throw FileSystemException('expected a real directory', path);
    }
  }

  Future<bool> _regularFile(String path) async =>
      await _type(path) == FileSystemEntityType.file;

  Future<Map<String, dynamic>> _metadata(String path, String id) async {
    if (!await _regularFile(path)) {
      throw FormatException('metadata unavailable');
    }
    final value = jsonDecode(await File(path).readAsString());
    if (value is! Map<String, dynamic> ||
        value['version'] != 1 ||
        value['id'] != id ||
        value['text'] is! String ||
        value['options'] is! Map<String, dynamic>) {
      throw FormatException('invalid metadata');
    }
    return value;
  }

  /// Completed stages only. Never removes or assumes ownership of live stages.
  Future<List<PendingReadInfo>> completedPending() async {
    if (!await root.exists()) return [];
    final result = <PendingReadInfo>[];
    final pattern = RegExp(r'^\.([0-9a-f]{32})\.tmp$');
    await for (final entity in root.list(followLinks: false)) {
      if (entity is! Directory) continue;
      final name = entity.uri.pathSegments.where((s) => s.isNotEmpty).last;
      final match = pattern.firstMatch(name);
      if (match == null) continue;
      final token = match.group(1)!;
      if (!await _regularFile('${entity.path}${Platform.pathSeparator}ready')) {
        continue;
      }
      try {
        final metadata = await _metadata(
          '${entity.path}${Platform.pathSeparator}pending.json',
          token,
        );
        if (!await _regularFile(
          '${entity.path}${Platform.pathSeparator}audio.flac',
        )) {
          throw FormatException('pending audio unavailable');
        }
        result.add(
          PendingReadInfo(
            token: token,
            text: metadata['text'] as String,
            options: (metadata['options'] as Map).cast<String, Object?>(),
          ),
        );
      } catch (e) {
        result.add(
          PendingReadInfo(
            token: token,
            text: '',
            options: {},
            problem: e.toString(),
          ),
        );
      }
    }
    return result;
  }

  /// Publish a completed worker's token only after the user selects Save.
  Future<SavedRead> publish(String token) async {
    if (!_idPattern.hasMatch(token)) {
      throw ArgumentError('invalid pending-read token');
    }
    final staging = Directory(_path('.$token.tmp'));
    if (await _type(staging.path) == FileSystemEntityType.notFound) {
      throw StateError('pending read is not complete');
    }
    await _requireDirectory(staging.path);
    if (!await _regularFile('${staging.path}${Platform.pathSeparator}ready')) {
      throw StateError('pending read is not complete');
    }
    final metadata = await _metadata(
      '${staging.path}${Platform.pathSeparator}pending.json',
      token,
    );
    if (!await _regularFile(
      '${staging.path}${Platform.pathSeparator}audio.flac',
    )) {
      throw FormatException('pending audio unavailable');
    }
    final outputMetadata = '${staging.path}${Platform.pathSeparator}read.json';
    if (await _type(outputMetadata) != FileSystemEntityType.notFound &&
        !await _regularFile(outputMetadata)) {
      throw FileSystemException('unsafe metadata path', outputMetadata);
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
    if (await _type(_path(token)) != FileSystemEntityType.notFound) {
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
    final type = await _type(staging.path);
    if (type == FileSystemEntityType.notFound) return;
    await _requireDirectory(staging.path);
    await staging.delete(recursive: true);
  }

  Future<List<SavedRead>> list() async {
    if (!await root.exists()) return [];
    final reads = <SavedRead>[];
    await for (final entity in root.list(followLinks: false)) {
      if (entity is! Directory) continue;
      final id = entity.uri.pathSegments.where((s) => s.isNotEmpty).last;
      if (!_idPattern.hasMatch(id)) continue;
      final path = _audio(id);
      final missing = !await _regularFile(path);
      try {
        final metadata = await _metadata(
          '${entity.path}${Platform.pathSeparator}read.json',
          id,
        );
        final savedAt = DateTime.parse(metadata['saved_at'] as String);
        reads.add(
          SavedRead(
            id: id,
            text: metadata['text'] as String,
            options: (metadata['options'] as Map).cast<String, Object?>(),
            audioPath: path,
            missing: missing,
            savedAt: savedAt,
          ),
        );
      } catch (e) {
        reads.add(
          SavedRead(
            id: id,
            text: '',
            options: {},
            audioPath: path,
            missing: missing,
            savedAt: DateTime.fromMillisecondsSinceEpoch(0, isUtc: true),
            problem: e.toString(),
          ),
        );
      }
    }
    reads.sort((a, b) => b.savedAt.compareTo(a.savedAt));
    return reads;
  }

  Future<void> delete(SavedRead read) async {
    if (!_idPattern.hasMatch(read.id)) {
      throw ArgumentError('invalid saved-read ID');
    }
    final dir = Directory(_path(read.id));
    final type = await _type(dir.path);
    if (type == FileSystemEntityType.notFound) return;
    await _requireDirectory(dir.path);
    await dir.delete(recursive: true);
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
