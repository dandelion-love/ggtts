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
