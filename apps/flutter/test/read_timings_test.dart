import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/read_timings.dart';

void main() {
  test('separates planted inter-piece wait, save and handoff delays', () {
    var now = 0;
    final lines = <String>[];
    final timings = ReadTimings(nowMicros: () => now, sink: lines.add);
    timings.load(() {
      now += 13000;
    });
    timings.beginStream();
    now += 200000;
    timings.piece(
      samples: 24000,
      rate: 24000,
      setup: () {
        now += 11000;
      },
      save: () {
        now += 37000;
      },
      handoff: () {
        now += 83000;
      },
    );
    now += 510000;
    timings.piece(
      samples: 12000,
      rate: 24000,
      setup: () {},
      save: () {
        now += 7000;
      },
      handoff: () {
        now += 3000;
      },
    );
    timings.streamEnded();
    timings.drain(() {
      now += 40000;
    });
    expect(lines, [
      'ggtts-read event=load elapsed_ms=13',
      'ggtts-read event=piece index=1 outcome=ok wait_ms=200 setup_ms=11 save_ms=37 handoff_ms=83 audio_ms=1000',
      'ggtts-read event=piece index=2 outcome=ok wait_ms=510 setup_ms=0 save_ms=7 handoff_ms=3 audio_ms=500',
      'ggtts-read event=stream_end pieces=2',
      'ggtts-read event=drain elapsed_ms=40',
    ]);
  });

  test('failure text is never logged and the original failure propagates', () {
    var now = 0;
    final lines = <String>[];
    final timings = ReadTimings(nowMicros: () => now, sink: lines.add);
    timings.beginStream();
    final failure = StateError('secret text /private/file');
    expect(
      () => timings.piece(
        samples: 1,
        rate: 1,
        setup: () {},
        save: () {
          now += 5000;
          throw failure;
        },
        handoff: () => fail('handoff must not run'),
      ),
      throwsA(same(failure)),
    );
    expect(lines.single, contains('outcome=failed'));
    expect(lines.single, contains('save_ms=5'));
    expect(lines.single, isNot(contains('secret')));
    expect(lines.single, isNot(contains('/private')));
  });

  test('logging is bounded and sink failure cannot change read outcome', () {
    final lines = <String>[];
    final timings = ReadTimings(
      nowMicros: () => 0,
      sink: lines.add,
      maxPieces: 2,
    );
    timings.beginStream();
    for (var i = 0; i < 5; i++) {
      timings.piece(
        samples: 1,
        rate: 1,
        setup: () {},
        save: () {},
        handoff: () {},
      );
    }
    expect(lines.length, 3);
    expect(lines.last, 'ggtts-read event=piece_log_limit');
    final failing = ReadTimings(
      nowMicros: () => 0,
      sink: (_) => throw StateError('sink'),
    );
    failing.beginStream();
    var delivered = false;
    failing.piece(
      samples: 1,
      rate: 1,
      setup: () {},
      save: () {},
      handoff: () {
        delivered = true;
      },
    );
    expect(delivered, true);
  });
}
