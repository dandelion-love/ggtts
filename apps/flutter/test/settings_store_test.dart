import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/settings_store.dart';

void main() {
  late Directory directory;
  late SettingsStore store;

  setUp(() async {
    directory = await Directory.systemTemp.createTemp('ggtts-settings-test-');
    store = SettingsStore(File('${directory.path}/settings/options.json'));
  });

  tearDown(() async => directory.delete(recursive: true));

  test('missing file is an empty options document', () async {
    expect(await store.load(), isEmpty);
  });

  test(
    'round trip preserves the exact document supplied to the engine',
    () async {
      final document = <String, Object?>{
        'load': {
          'device': 'cpu',
          'threads': 3,
          'cache_dir': '/models',
          'offline': true,
        },
        'render': {'seed': 42},
      };
      await store.save(document);
      expect(await store.load(), document);
      (document['load'] as Map)['threads'] = 7;
      await store.save(document);
      expect(await store.load(), document);
    },
  );

  test('invalid settings cannot replace a valid document', () async {
    final valid = <String, Object?>{
      'load': {'threads': 2},
    };
    await store.save(valid);
    for (final invalid in <Map<String, Object?>>[
      {'load': 'cpu'},
      {
        'load': {'device': 'gpu'},
      },
      {
        'load': {'threads': 0},
      },
      {
        'load': {'threads': 2147483648},
      },
      {
        'load': {'threads': 2.5},
      },
      {
        'load': {'cache_dir': ''},
      },
      {
        'load': {'cache_dir': 'a\u0000b'},
      },
    ]) {
      await expectLater(store.save(invalid), throwsFormatException);
      expect(await store.load(), valid);
    }
  });

  test(
    'invalid or corrupt files are reported without being overwritten',
    () async {
      await store.file.parent.create(recursive: true);
      for (final content in ['{', '[]', '{"load":{"threads":0}}']) {
        await store.file.writeAsString(content);
        await expectLater(store.load(), throwsFormatException);
        expect(await store.file.readAsString(), content);
      }
    },
  );
}
