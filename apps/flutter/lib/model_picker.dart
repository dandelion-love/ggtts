import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/services.dart';

/// Select a model file. Android imports its content into app-private storage;
/// desktop platforms return the selected filesystem path.
Future<String?> pickModelFile(String kind) async {
  if (!const ['model', 'codec', 'voice'].contains(kind)) {
    throw ArgumentError.value(kind, 'kind');
  }
  if (Platform.isAndroid) {
    return const MethodChannel('ggtts/model_picker')
        .invokeMethod<String>('pick', {'kind': kind});
  }
  if (Platform.isLinux || Platform.isWindows || Platform.isMacOS) {
    final file = await openFile(
      acceptedTypeGroups: [
        XTypeGroup(
          label: kind == 'voice' ? 'Voice JSON' : 'GGUF model',
          extensions: [kind == 'voice' ? 'json' : 'gguf'],
        ),
      ],
    );
    return file?.path;
  }
  throw UnsupportedError('File selection is unavailable on this platform.');
}
