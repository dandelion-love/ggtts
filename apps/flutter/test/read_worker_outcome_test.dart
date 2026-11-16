import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/read_worker_outcome.dart';

void main() {
  test(
    'failing audio uninit emits one error and discards without finishing',
    () async {
      final events = <Map<String, Object?>>[];
      var finished = false;
      var discarded = false;
      await runReadOutcome(
        generate: () async {},
        cleanup: () => throw StateError('planted uninit failure'),
        finish: () {
          finished = true;
          return 'token';
        },
        discard: () async {
          discarded = true;
        },
        send: events.add,
      );
      expect(finished, isFalse);
      expect(discarded, isTrue);
      expect(events, hasLength(1));
      expect(events.single['error'], contains('planted uninit failure'));
      expect(events.single.containsKey('done'), isFalse);
    },
  );

  test(
    'generation, cleanup and discard failures still emit only one error',
    () async {
      final events = <Map<String, Object?>>[];
      await runReadOutcome(
        generate: () async => throw StateError('generation failed'),
        cleanup: () => throw StateError('uninit failed'),
        finish: () => throw StateError('must not finish'),
        discard: () async => throw StateError('discard failed'),
        send: events.add,
      );
      expect(events, hasLength(1));
      expect(events.single['error'], contains('generation failed'));
    },
  );

  test('stopped read discards only after successful cleanup', () async {
    final order = <String>[];
    await runReadOutcome(
      generate: () async {
        throw const _Stop();
      },
      cleanup: () {
        order.add('cleanup');
      },
      finish: () => throw StateError('must not finish'),
      discard: () async {
        order.add('discard');
      },
      isStopped: (e) => e is _Stop,
      send: (event) {
        expect(event, {'stopped': true});
        order.add('send');
      },
    );
    expect(order, ['cleanup', 'discard', 'send']);
  });

  test('cleanup failure after stop remains an error', () async {
    final events = <Map<String, Object?>>[];
    await runReadOutcome(
      generate: () async {
        throw const _Stop();
      },
      cleanup: () => throw StateError('cleanup failure'),
      finish: () => throw StateError('must not finish'),
      discard: () async {},
      isStopped: (e) => e is _Stop,
      send: events.add,
    );
    expect(events.single['error'], contains('cleanup failure'));
  });

  test('successful cleanup precedes ready marker and completion', () async {
    final order = <String>[];
    await runReadOutcome(
      generate: () async {
        order.add('generate');
      },
      cleanup: () {
        order.add('cleanup');
      },
      finish: () {
        order.add('finish');
        return 'token';
      },
      discard: () async {
        fail('unexpected discard');
      },
      send: (event) {
        order.add('send');
        expect(event, {'done': true, 'token': 'token'});
      },
    );
    expect(order, ['generate', 'cleanup', 'finish', 'send']);
  });
}

class _Stop implements Exception {
  const _Stop();
}
