import 'dart:ffi';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/ggtts.dart';

void main() {
  if (!Platform.isLinux) {
    test('native fixture requires Linux and a C compiler', () {}, skip: true);
    return;
  }
  late Directory temp;
  late Ggtts ggtts;
  late CancellationApi cancel;
  setUpAll(() {
    temp = Directory.systemTemp.createTempSync('ggtts-bindings-');
    final lib = '${temp.path}/fixture.so';
    final result = Process.runSync('cc', [
      '-shared',
      '-fPIC',
      'test/fixtures/ggtts_fixture.c',
      '-o',
      lib,
    ]);
    if (result.exitCode != 0) {
      throw StateError('fixture build failed: ${result.stderr}');
    }
    final native = DynamicLibrary.open(lib);
    ggtts = Ggtts(native);
    cancel = CancellationApi(native);
  });
  tearDownAll(() => temp.deleteSync(recursive: true));

  test('schema and capabilities are decoded from the library', () {
    expect(ggtts.optionsSchema(), {'schema': 'fixture'});
    expect(ggtts.capabilities(7), {
      'speakers': true,
      'speaker_names': ['fixture'],
    });
  });

  test(
    'load progress copies transient native messages and cannot alter load',
    () {
      final messages = <String>[];
      final handle = ggtts.loadWithProgress({}, (notice) {
        messages.add(notice);
        throw StateError('sink failed');
      });
      expect(handle, 7);
      expect(messages, ['fixture: 1 of 2 MB (50%), 1.0 MB/s']);
    },
  );

  test('old native ABI refuses progress before symbol lookup', () {
    final source = File('test/fixtures/ggtts_fixture.c')
        .readAsStringSync()
        .replaceFirst('return 8;', 'return 6;');
    final path = '${temp.path}/old.c';
    final lib = '${temp.path}/old.so';
    File(path).writeAsStringSync(source);
    expect(
      Process.runSync('cc', ['-shared', '-fPIC', path, '-o', lib]).exitCode,
      0,
    );
    final old = Ggtts(DynamicLibrary.open(lib));
    expect(() => old.loadWithProgress({}, (_) {}), throwsUnsupportedError);
  });

  test('copies each synchronous piece before native buffer reuse', () {
    final pieces = <Pcm>[];
    ggtts.renderStream(7, 'text', null, (piece) {
      pieces.add(piece);
      return true;
    });
    expect(pieces.map((p) => p.rate), [24000, 24000]);
    expect(pieces[0].samples, [0.25, -0.5]);
    expect(pieces[1].samples, [0.75, 1.0]);
  });

  test('false stops native render', () {
    var calls = 0;
    expect(
      () => ggtts.renderStream(7, 'text', null, (_) {
        calls++;
        return false;
      }),
      throwsA(isA<GgttsException>().having((e) => e.code, 'code', -5)),
    );
    expect(calls, 1);
  });

  test('cancellable render keeps one-shot native token semantics', () {
    final id = cancel.create();
    var calls = 0;
    ggtts.renderStream(7, 'text', null, (_) {
      calls++;
      return true;
    }, cancellationToken: id);
    expect(calls, 2);
    cancel.signal(id);
    expect(
      () => ggtts.renderStream(
        7,
        'text',
        null,
        (_) => true,
        cancellationToken: id,
      ),
      throwsA(isA<GgttsException>().having((e) => e.code, 'code', -5)),
    );
    cancel.free(id);
  });

  test('callback exceptions stop native render and are rethrown', () {
    final error = StateError('sink failed');
    expect(
      () => ggtts.renderStream(7, 'text', null, (_) => throw error),
      throwsA(same(error)),
    );
  });

  for (final mode in ['bad_rate', 'bad_length', 'bad_huge']) {
    test('refuses invalid native piece: $mode', () {
      expect(
        () => ggtts.renderStream(7, 'text', {'fixture': mode}, (_) => true),
        throwsA(isA<StateError>()),
      );
    });
  }
}
