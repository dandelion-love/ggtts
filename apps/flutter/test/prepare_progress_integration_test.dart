import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/main.dart';
import 'package:ggtts_app/ggtts.dart';

void main() {
  test(
    'native stop reaches preparation isolate and a fresh token succeeds',
    () async {
      final api = CancellationApi(openGgtts());
      final token = api.create();
      final notices = <String>[];
      final entered = Completer<void>();
      final result = inspectModel({}, (notice) {
        notices.add(notice);
        if (!entered.isCompleted) entered.complete();
      }, token);
      await entered.future.timeout(const Duration(seconds: 5));
      api.signal(token);
      await expectLater(result, throwsA(isA<StateError>()));
      api.free(token);
      expect(notices, isNotEmpty);
      final next = api.create();
      final ready = await inspectModel({}, (_) {}, next);
      expect((ready['capabilities'] as Map)['speaker_names'], ['fixture']);
      api.free(next);
    },
    skip: Platform.environment['GGTTS_LIB'] == null
        ? 'Run with GGTTS_LIB pointing at the compiled progress fixture'
        : false,
  );

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
