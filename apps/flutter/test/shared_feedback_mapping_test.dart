import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/shared_feedback_mapping.dart';

void main() {
  final canonical =
      r'{"format":"ggtts-feedback-input/1","id":"page","items":[{"audio":"data:audio/flac;base64,ZkxhQ4AAACIAAAAAAAAAAAAAAQAAAAAAAAEAAAAAAAAAAAAAAAAAAAAA","id":"item1","text":"A\u4e2d\ud83d\ude00\ud840\udc00\uff0c\u6587"},{"audio":"data:audio/flac;base64,ZkxhQ4AAACIAAAAAAAAAAAAAAQAAAAAAAAEAAAAAAAAAAAAAAAAAAAAA","id":"item2","text":"\u6587"}]}';
  const audioHash =
      '2bbd61d9d7aa8b6a20403f5dfa6735e05cb6d7aa18f39908adef384c84b96ace';
  final binding = FeedbackBinding.fromCanonicalInput(utf8.encode(canonical), {
    'item1': const FeedbackClip('opaque:/source', 'A中😀𠀀，文', audioHash),
    'item2': const FeedbackClip('source two', '文', audioHash),
  });
  final digest = binding.inputSha256;
  Map<String, Object?> result() => {
    'format': 'ggtts-feedback-result/1',
    'id': 'page',
    'input_sha256': digest,
    'note': 'page note',
    'items': [
      {
        'id': 'item1',
        'marks': [0, 1, 2],
        'rating': 'accept',
        'note': 'item note',
      },
      {'id': 'item2', 'marks': <int>[], 'rating': 'unsure', 'note': ''},
    ],
  };
  test('maps scalar indices, opaque IDs and both notes', () {
    final mapped = mapSharedFeedback(binding, result());
    expect(mapped[0]['clip_id'], 'opaque:/source');
    expect(mapped[0]['verdict'], 'accept');
    expect(mapped[0]['marks'], [
      {'start': 1, 'end': 2, 'verdict': 'reject', 'note': null},
      {'start': 3, 'end': 4, 'verdict': 'reject', 'note': null},
      {'start': 5, 'end': 6, 'verdict': 'reject', 'note': null},
    ]);
    expect(jsonDecode(mapped[0]['note'] as String), {
      'format': 'ggtts-shared-notes/1',
      'page_note': 'page note',
      'item_note': 'item note',
    });
  });
  test('rejects tampering, duplicate or missing items and invalid marks', () {
    final wrong = result()..['input_sha256'] = 'b' * 64;
    expect(() => mapSharedFeedback(binding, wrong), throwsFormatException);
    final duplicate = result();
    (duplicate['items'] as List)[1]['id'] = 'item1';
    expect(() => mapSharedFeedback(binding, duplicate), throwsFormatException);
    for (final marks in [
      [0, 0],
      [3],
      [-1],
      [1.0],
    ]) {
      final bad = result();
      (bad['items'] as List)[0]['marks'] = marks;
      expect(() => mapSharedFeedback(binding, bad), throwsFormatException);
    }
    final extra = result()..['extra'] = true;
    expect(() => mapSharedFeedback(binding, extra), throwsFormatException);
  });
  test('rejects unrepresentable notes before producing submissions', () {
    final bad = result()..['note'] = 'x' * 4001;
    expect(() => mapSharedFeedback(binding, bad), throwsFormatException);
    final short = result()..['note'] = 'x' * 1000;
    expect(mapSharedFeedback(binding, short), hasLength(2));
  });
  test('rejects noncanonical input, changed text, long source IDs and malformed Unicode', () {
    Map<String, FeedbackClip> clips(String id) => {
      'item1': FeedbackClip(id, 'A中😀𠀀，文', audioHash),
      'item2': const FeedbackClip('other', '文', audioHash),
    };
    expect(
      () => FeedbackBinding.fromCanonicalInput(
        utf8.encode('$canonical '),
        clips('ok'),
      ),
      throwsFormatException,
    );
    expect(
      () => FeedbackBinding.fromCanonicalInput(
        utf8.encode(canonical),
        clips('x' * 257),
      ),
      throwsFormatException,
    );
    expect(
      FeedbackBinding.fromCanonicalInput(
        utf8.encode(canonical),
        clips('x' * 256),
      ).clips,
      hasLength(2),
    );
    expect(
      () => FeedbackBinding.fromCanonicalInput(utf8.encode(canonical), {
        'item1': const FeedbackClip('ok', 'changed', audioHash),
        'item2': const FeedbackClip('other', '文', audioHash),
      }),
      throwsFormatException,
    );
    final bad = result()..['note'] = String.fromCharCode(0xd800);
    expect(() => mapSharedFeedback(binding, bad), throwsFormatException);
  });
  test('rejects swapping same-text clips with different audio', () {
    final sameText = canonical.replaceFirst(
      r'A\u4e2d\ud83d\ude00\ud840\udc00\uff0c\u6587',
      r'\u6587',
    );
    final distinct = sameText.replaceFirst(
      'ZkxhQ4AAACIAAAAAAAAAAAAAAQAAAAAAAAEAAAAAAAAAAAAAAAAAAAAA',
      'ZkxhQ4AAACIAAAAAAAAAAAAAAQAAAAAAAAEAAAAAAAAAAAAAAAAAAAAB',
    );
    expect(
      () => FeedbackBinding.fromCanonicalInput(utf8.encode(distinct), {
        'item1': const FeedbackClip('source two', '文', audioHash),
        'item2': const FeedbackClip('source one', '文', audioHash),
      }),
      throwsFormatException,
    );
  });
}
