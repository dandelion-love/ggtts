import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/main.dart';

void main() {
  test(
    'preparation isolate delivers native load notices and result',
    () async {
      final notices = <String>[];
      final result = await inspectModel({}, notices.add);
      expect(notices, ['fixture: 1 of 2 MB (50%), 1.0 MB/s']);
      expect((result['capabilities'] as Map)['speaker_names'], ['fixture']);
    },
    skip: Platform.environment['GGTTS_LIB'] == null
        ? 'Run with GGTTS_LIB pointing at the compiled progress fixture'
        : false,
  );
}
