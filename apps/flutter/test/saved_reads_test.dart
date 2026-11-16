import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/ggtts.dart';
import 'package:ggtts_app/saved_reads.dart';

void main() {
  late Directory temp;
  late SavedReads library;
  setUp(() {
    temp = Directory.systemTemp.createTempSync('ggtts-reads-');
    library = SavedReads(temp);
  });
  tearDown(() => temp.deleteSync(recursive: true));

  test(
    'stages chunks, publishes a snapshot and detects missing audio',
    () async {
      final options = <String, Object?>{
        'load': {'threads': 2},
        'render': {'speaker': 'voice'},
      };
      final pending = await library.begin(text: 'hello', options: options);
      (options['load'] as Map)['threads'] = 99;
      pending.append(Pcm(Float32List.fromList([0.1, -0.2]), 24000));
      expect(await library.list(), isEmpty);
      final token = pending.finish();
      expect(await library.list(), isEmpty);
      final saved = await SavedReads(temp).publish(token);
      final loaded = (await SavedReads(temp).list()).single;
      expect(loaded.text, 'hello');
      expect((loaded.options['load'] as Map)['threads'], 2);
      expect(loaded.audioPath, saved.audioPath);
      expect(loaded.missing, isFalse);
      await File(loaded.audioPath).delete();
      final missing = (await library.list()).single;
      expect(missing.missing, isTrue);
      await library.delete(missing);
      expect(await library.list(), isEmpty);
    },
  );

  test(
    'cannot publish unfinished or malformed tokens; discard removes stage',
    () async {
      final pending = await library.begin(text: '', options: {});
      await expectLater(library.publish('../outside'), throwsArgumentError);
      await pending.discard();
      expect(await library.list(), isEmpty);
    },
  );

  test('rejects invalid audio and rate changes; worker can discard', () async {
    final pending = await library.begin(text: 'x', options: {});
    expect(
      () => pending.append(Pcm(Float32List.fromList([double.nan]), 24000)),
      throwsArgumentError,
    );
    pending.append(Pcm(Float32List.fromList([0.1]), 24000));
    expect(
      () => pending.append(Pcm(Float32List.fromList([0.1]), 16000)),
      throwsArgumentError,
    );
    await pending.discard();
    expect(await library.list(), isEmpty);
  });

  test('finished token can be discarded on the UI side', () async {
    final pending = await library.begin(text: 'x', options: {});
    pending.append(Pcm(Float32List.fromList([0.1]), 24000));
    final token = pending.finish();
    await SavedReads(temp).discardPending(token);
    await expectLater(library.publish(token), throwsStateError);
  });

  test(
    'worker closes its file and sends only a token to the UI isolate',
    () async {
      final root = temp.path;
      final token = await Isolate.run(() async {
        final pending = await SavedReads(Directory(root))
            .begin(text: 'isolate', options: {});
        pending.append(Pcm(Float32List.fromList([0.25, -0.25]), 24000));
        return pending.finish();
      });
      final saved = await library.publish(token);
      expect(saved.text, 'isolate');
      expect(await File(saved.audioPath).exists(), isTrue);
    },
  );

  test(
    'a corrupt record remains visible without hiding healthy reads',
    () async {
      final first = await library.begin(text: 'healthy', options: {});
      first.append(Pcm(Float32List.fromList([0.1]), 24000));
      final healthy = await library.publish(first.finish());
      final second = await library.begin(text: 'damaged', options: {});
      second.append(Pcm(Float32List.fromList([0.2]), 24000));
      final damaged = await library.publish(second.finish());
      await File('${temp.path}/${damaged.id}/read.json')
          .writeAsString('{bad json');
      final found = await library.list();
      expect(found.length, 2);
      expect(found.singleWhere((r) => r.id == healthy.id).text, 'healthy');
      final bad = found.singleWhere((r) => r.id == damaged.id);
      expect(bad.invalid, isTrue);
      expect(bad.problem, isNotEmpty);
      await library.delete(bad);
      expect((await library.list()).single.id, healthy.id);
    },
  );

  test(
    'recover completed stage across instances without touching live stages',
    () async {
      final live = await library.begin(text: 'live', options: {});
      final completed = await library.begin(
        text: 'recover me',
        options: {
          'load': {'threads': 3},
        },
      );
      completed.append(Pcm(Float32List.fromList([0.1]), 24000));
      final token = completed.finish();
      final found = await SavedReads(temp).completedPending();
      expect(found.length, 1);
      expect(found.single.token, token);
      expect(found.single.text, 'recover me');
      expect(found.single.invalid, isFalse);
      expect((await SavedReads(temp).publish(token)).text, 'recover me');
      live.append(Pcm(Float32List.fromList([0.3]), 24000));
      await live.discard();
    },
  );

  test(
    'corrupt completed stage is reported and can be explicitly discarded',
    () async {
      final pending = await library.begin(text: 'x', options: {});
      pending.append(Pcm(Float32List.fromList([0.1]), 24000));
      final token = pending.finish();
      await File('${temp.path}/.$token.tmp/pending.json')
          .writeAsString('broken');
      final entry = (await library.completedPending()).single;
      expect(entry.invalid, isTrue);
      expect(entry.token, token);
      await expectLater(
        library.publish(token),
        throwsA(isA<FormatException>()),
      );
      await library.discardPending(token);
      expect(await library.completedPending(), isEmpty);
    },
  );

  test('recovery and deletion refuse symlinked entries', () async {
    final outside = await Directory.systemTemp.createTemp('ggtts-outside-');
    try {
      await File('${outside.path}/keep').writeAsString('safe');
      const token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
      await Link('${temp.path}/.$token.tmp').create(outside.path);
      expect(await library.completedPending(), isEmpty);
      await expectLater(
        library.discardPending(token),
        throwsA(isA<FileSystemException>()),
      );
      await expectLater(
        library.publish(token),
        throwsA(isA<FileSystemException>()),
      );
      await Link('${temp.path}/$token').create(outside.path);
      final forged = SavedRead(
        id: token,
        text: '',
        options: {},
        audioPath: '',
        missing: true,
        savedAt: DateTime.utc(1970),
      );
      await expectLater(
        library.delete(forged),
        throwsA(isA<FileSystemException>()),
      );
      expect(await File('${outside.path}/keep').readAsString(), 'safe');
    } finally {
      await outside.delete(recursive: true);
    }
  }, skip: !Platform.isLinux ? 'symlink test runs on Linux' : false);

  test('publishing refuses a symlinked metadata output', () async {
    final outside = await Directory.systemTemp.createTemp('ggtts-target-');
    try {
      final target = File('${outside.path}/target');
      await target.writeAsString('safe');
      final pending = await library.begin(text: 'x', options: {});
      pending.append(Pcm(Float32List.fromList([0.1]), 24000));
      final token = pending.finish();
      await Link('${temp.path}/.$token.tmp/read.json').create(target.path);
      await expectLater(
        library.publish(token),
        throwsA(isA<FileSystemException>()),
      );
      expect(await target.readAsString(), 'safe');
      await library.discardPending(token);
    } finally {
      await outside.delete(recursive: true);
    }
  }, skip: !Platform.isLinux ? 'symlink test runs on Linux' : false);

  test('incremental FLAC decodes independently with ffmpeg', () async {
    final pending = await library.begin(text: 'audio', options: {});
    final source = Float32List.fromList(
      List.generate(10003, (i) => ((i % 200) - 100) / 100),
    );
    pending.append(Pcm(Float32List.sublistView(source, 0, 19), 24000));
    pending.append(Pcm(Float32List.sublistView(source, 19, 9000), 24000));
    pending.append(Pcm(Float32List.sublistView(source, 9000), 24000));
    final saved = await library.publish(pending.finish());
    final out = File('${temp.path}/decoded.raw');
    final result = await Process.run('ffmpeg', [
      '-v',
      'error',
      '-y',
      '-i',
      saved.audioPath,
      '-f',
      's16le',
      '-acodec',
      'pcm_s16le',
      out.path,
    ]);
    expect(result.exitCode, 0, reason: '${result.stderr}');
    final decoded = ByteData.sublistView(await out.readAsBytes());
    expect(decoded.lengthInBytes, source.length * 2);
    for (var i = 0; i < source.length; i++) {
      expect(
        decoded.getInt16(i * 2, Endian.little),
        (source[i].clamp(-1.0, 1.0) * 32767).round(),
      );
    }
  }, skip: !Platform.isLinux ? 'ffmpeg verification runs on Linux' : false);
}
