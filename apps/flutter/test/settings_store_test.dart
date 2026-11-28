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
    'unreadable model, codec and voice leave the saved document intact',
    () async {
      final files = <String, String>{};
      for (final key in ['model', 'codec', 'voice']) {
        final file = File('${directory.path}/$key');
        await file.writeAsString('fixture');
        files[key] = file.path;
      }
      final valid = <String, Object?>{
        'load': {...files, 'threads': 2},
        'render': {'seed': 42},
      };
      await store.save(valid);
      final savedBytes = await store.file.readAsBytes();
      for (final key in files.keys) {
        for (final badPath in ['${directory.path}/missing', directory.path]) {
          final invalid = <String, Object?>{
            'load': {...files, key: badPath, 'threads': 3},
            'render': {'seed': 99},
          };
          await expectLater(store.save(invalid), throwsFormatException);
          expect(await store.file.readAsBytes(), savedBytes);
          expect(await store.load(), valid);
        }
      }
      // Clearing optional paths still uses the defaults.
      await store.save({
        'load': {'threads': 2},
      });
      expect(await store.load(), {
        'load': {'threads': 2},
      });
    },
  );

  test('a released model name is saved for the engine to resolve', () async {
    // The engine downloads a known name and refuses an unknown one, listing
    // the known names; the settings check holds only file paths to a file.
    for (final name in ['hk-cantonese-1.7b', 'hk-cantonese-0.6b']) {
      final document = <String, Object?>{
        'load': {'model': name},
      };
      await store.save(document);
      expect(await store.load(), document);
    }
    // Paths are still paths: a missing one is refused as before.
    for (final path in [
      '${directory.path}/hk-cantonese-1.7b',
      'missing.gguf',
      'voice.json',
    ]) {
      await expectLater(
        store.save({
          'load': {'model': path},
        }),
        throwsFormatException,
      );
    }
    // A codec or voice is never a model name.
    await expectLater(
      store.save({
        'load': {'codec': 'hk-cantonese-1.7b'},
      }),
      throwsFormatException,
    );
    expect(readsAsModelName('hk-cantonese-1.7b'), isTrue);
    for (final path in ['./m', 'a\\b', 'C:m', 'm.GGUF', 'v.json', '']) {
      expect(readsAsModelName(path), isFalse, reason: path);
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
