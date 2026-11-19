import 'dart:convert';
import 'dart:io';

/// Persists the options document that is passed to the engine. The caller supplies
/// any defaults and edits this document itself; there is no second settings copy.
class SettingsStore {
  SettingsStore(this.file);

  final File file;

  /// An absent file is a new empty document. Corruption is reported, not reset.
  Future<Map<String, Object?>> load() async {
    String text;
    try {
      text = await file.readAsString();
    } on PathNotFoundException {
      return <String, Object?>{};
    }
    final Object? decoded = jsonDecode(text);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Options document must be an object.');
    }
    final document = Map<String, Object?>.from(decoded);
    validateSettings(document);
    return document;
  }

  /// Validate before touching the previous document, then replace it atomically
  /// with a complete file on the same filesystem.
  Future<void> save(Map<String, Object?> document) async {
    validateSettings(document);
    await validateReadableFiles(document);
    final encoded = jsonEncode(document);
    final parent = file.absolute.parent;
    await parent.create(recursive: true);
    final tempDir = await parent.createTemp('.settings-');
    try {
      final temp = File('${tempDir.path}/options.json');
      await temp.writeAsString(encoded, flush: true);
      await temp.rename(file.absolute.path);
    } finally {
      await tempDir.delete(recursive: true);
    }
  }
}

/// Check paths while the old settings file is still intact. This checks access,
/// not model format or future availability; the engine validates those on load.
Future<void> validateReadableFiles(
  Map<String, Object?> document, {
  Future<void> Function(String path)? checkFile,
}) async {
  final load = document['load'];
  if (load is! Map<String, dynamic>) return;
  for (final key in ['model', 'codec', 'voice']) {
    final path = load[key];
    if (path is! String) continue;
    try {
      await (checkFile ?? _checkReadableFile)(path);
    } on FileSystemException {
      throw FormatException('$key must be an app-readable file.');
    }
  }
}

Future<void> _checkReadableFile(String path) async {
  if (await FileSystemEntity.type(path) != FileSystemEntityType.file) {
    throw const FileSystemException('Not a regular file');
  }
  final file = await File(path).open(mode: FileMode.read);
  try {
    await file.read(1);
  } finally {
    await file.close();
  }
}

/// Check settings this screen edits against the shared options schema. Other
/// options stay intact and are checked by the engine when the document is used.
void validateSettings(Map<String, Object?> document) {
  if (!document.containsKey('load')) return;
  final load = document['load'];
  if (load is! Map<String, dynamic>) {
    throw const FormatException('load must be an object.');
  }
  if (load.containsKey('engine') &&
      !const [
        'auto',
        'qwen3-tts',
        'kitten-tts-2',
        'fake',
      ].contains(load['engine'])) {
    throw const FormatException(
      'engine must be auto, qwen3-tts, kitten-tts-2 or fake.',
    );
  }
  for (final key in ['model', 'codec', 'voice']) {
    if (!load.containsKey(key)) continue;
    final value = load[key];
    if (value is! String || value.trim().isEmpty || value.contains('\u0000')) {
      throw FormatException('$key must be a nonempty file path without NUL.');
    }
    if (value.startsWith('content://')) {
      throw FormatException(
        '$key cannot be a document-provider link; use an app-readable file path.',
      );
    }
  }
  if (load.containsKey('device') &&
      load['device'] != 'cpu' &&
      load['device'] != 'cuda') {
    throw const FormatException('device must be cpu or cuda.');
  }
  if (load.containsKey('threads')) {
    final value = load['threads'];
    if (value is! int || value < 1 || value > 2147483647) {
      throw const FormatException(
        'threads must be an integer from 1 to 2147483647.',
      );
    }
  }
  if (load.containsKey('cache_dir')) {
    final value = load['cache_dir'];
    if (value is! String || value.isEmpty || value.contains('\u0000')) {
      throw const FormatException(
        'cache_dir must be a nonempty path without NUL.',
      );
    }
  }
}
