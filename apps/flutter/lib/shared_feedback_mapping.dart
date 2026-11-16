import 'dart:convert';

import 'package:crypto/crypto.dart';

class FeedbackBinding {
  FeedbackBinding._(this.pageId, this.inputSha256, this.clips);
  final String pageId;
  final String inputSha256;
  final Map<String, FeedbackClip> clips;

  factory FeedbackBinding.fromCanonicalInput(
    List<int> bytes,
    Map<String, FeedbackClip> source,
  ) {
    Never invalid() => throw const FormatException('Invalid feedback input');
    final text = utf8.decode(bytes);
    final doc = jsonDecode(text);
    if (doc is! Map ||
        !_keys(doc, {'format', 'id', 'items'}) ||
        doc['format'] != 'ggtts-feedback-input/1' ||
        doc['id'] is! String ||
        !_validId(doc['id']) ||
        doc['items'] is! List ||
        (doc['items'] as List).isEmpty) {
      invalid();
    }
    final clips = <String, FeedbackClip>{};
    for (final item in doc['items'] as List) {
      if (item is! Map ||
          !_keys(item, {'id', 'text', 'audio'}) ||
          item['id'] is! String ||
          !_validId(item['id']) ||
          item['text'] is! String ||
          (item['text'] as String).isEmpty ||
          !_scalarText(item['text']) ||
          item['audio'] is! String ||
          !_validAudio(item['audio'])) {
        invalid();
      }
      final id = item['id'] as String;
      final clip = source[id];
      if (clip == null ||
          clip.text != item['text'] ||
          !RegExp(r'^[0-9a-f]{64}$').hasMatch(clip.audioSha256) ||
          sha256
                  .convert(
                    base64.decode(
                      (item['audio'] as String).substring(
                        'data:audio/flac;base64,'.length,
                      ),
                    ),
                  )
                  .toString() !=
              clip.audioSha256 ||
          clips.containsKey(id) ||
          clip.sourceId.isEmpty ||
          clip.sourceId.runes.length > 256 ||
          !_scalarText(clip.sourceId)) {
        invalid();
      }
      clips[id] = clip;
    }
    if (clips.length != source.length ||
        clips.values.map((c) => c.sourceId).toSet().length != clips.length ||
        _canonical(doc) != text) {
      invalid();
    }
    return FeedbackBinding._(
      doc['id'],
      sha256.convert(bytes).toString(),
      Map.unmodifiable(clips),
    );
  }
}

bool _validId(String id) =>
    RegExp(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$').hasMatch(id);

bool _scalarText(String s) {
  final units = s.codeUnits;
  for (var i = 0; i < units.length; i++) {
    final c = units[i];
    if (c >= 0xd800 && c <= 0xdbff) {
      if (++i >= units.length || units[i] < 0xdc00 || units[i] > 0xdfff) {
        return false;
      }
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      return false;
    }
  }
  return true;
}

bool _validAudio(String audio) {
  const prefix = 'data:audio/flac;base64,';
  if (!audio.startsWith(prefix)) return false;
  try {
    final encoded = audio.substring(prefix.length);
    final data = base64.decode(encoded);
    if (base64.encode(data) != encoded ||
        data.length < 42 ||
        data[0] != 102 ||
        data[1] != 76 ||
        data[2] != 97 ||
        data[3] != 67 ||
        data[4] & 127 != 0 ||
        data[5] != 0 ||
        data[6] != 0 ||
        data[7] != 34) {
      return false;
    }
    final rate = (data[18] << 12) | (data[19] << 4) | (data[20] >> 4);
    final count =
        ((data[21] & 15) << 32) |
        (data[22] << 24) |
        (data[23] << 16) |
        (data[24] << 8) |
        data[25];
    return rate != 0 && count != 0;
  } on FormatException {
    return false;
  }
}

String _canonical(Object? value) {
  if (value is String) {
    final out = StringBuffer('"');
    for (final unit in value.codeUnits) {
      switch (unit) {
        case 34:
          out.write(r'\"');
          break;
        case 92:
          out.write(r'\\');
          break;
        case 8:
          out.write(r'\b');
          break;
        case 9:
          out.write(r'\t');
          break;
        case 10:
          out.write(r'\n');
          break;
        case 12:
          out.write(r'\f');
          break;
        case 13:
          out.write(r'\r');
          break;
        default:
          if (unit < 32 ||
              unit >= 127 ||
              unit == 60 ||
              unit == 62 ||
              unit == 38) {
            out.write('\\u${unit.toRadixString(16).padLeft(4, '0')}');
          } else {
            out.writeCharCode(unit);
          }
      }
    }
    return '$out"';
  }
  if (value is List) return '[${value.map(_canonical).join(',')}]';
  if (value is Map) {
    final keys = value.keys.cast<String>().toList()..sort();
    return '{${keys.map((key) => '${_canonical(key)}:${_canonical(value[key])}').join(',')}}';
  }
  throw const FormatException('Invalid canonical input');
}

class FeedbackClip {
  const FeedbackClip(this.sourceId, this.text, this.audioSha256);
  final String sourceId;
  final String text;

  /// SHA-256 of the audio fetched for this exact source clip.
  final String audioSha256;
}

bool _cjk(int c) =>
    (c >= 0x3400 && c <= 0x4dbf) ||
    (c >= 0x4e00 && c <= 0x9fff) ||
    (c >= 0xf900 && c <= 0xfaff) ||
    (c >= 0x20000 && c <= 0x2fa1f) ||
    (c >= 0x30000 && c <= 0x323af);

bool _keys(Map<dynamic, dynamic> value, Set<String> expected) =>
    value.length == expected.length && value.keys.every(expected.contains);

/// Validates the complete result before returning submissions. The host must
/// bind the digest and texts to the exact canonical input embedded in the page.
/// Notes use the lossless ggtts-shared-notes/1 JSON envelope.
List<Map<String, Object?>> mapSharedFeedback(
  FeedbackBinding binding,
  Object? result,
) {
  final idPattern = RegExp(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$');
  final digestPattern = RegExp(r'^[0-9a-f]{64}$');
  Never invalid() => throw const FormatException('Invalid shared feedback');
  if (!idPattern.hasMatch(binding.pageId) ||
      !digestPattern.hasMatch(binding.inputSha256) ||
      binding.clips.isEmpty ||
      binding.clips.keys.any((id) => !idPattern.hasMatch(id)) ||
      binding.clips.values.any((c) => c.sourceId.isEmpty || c.text.isEmpty) ||
      binding.clips.values.map((c) => c.sourceId).toSet().length !=
          binding.clips.length) {
    invalid();
  }
  if (result is! Map ||
      !_keys(result, {'format', 'id', 'input_sha256', 'items', 'note'}) ||
      result['format'] != 'ggtts-feedback-result/1' ||
      result['id'] != binding.pageId ||
      result['input_sha256'] != binding.inputSha256 ||
      result['note'] is! String ||
      !_scalarText(result['note']) ||
      result['items'] is! List) {
    invalid();
  }
  final pageNote = result['note'] as String;
  final items = result['items'] as List;
  if (items.length != binding.clips.length) invalid();
  final seen = <String>{};
  final submissions = <Map<String, Object?>>[];
  for (final item in items) {
    if (item is! Map ||
        !_keys(item, {'id', 'marks', 'rating', 'note'}) ||
        item['id'] is! String ||
        item['note'] is! String ||
        !_scalarText(item['note']) ||
        item['marks'] is! List ||
        !{'accept', 'reject', 'unsure'}.contains(item['rating'])) {
      invalid();
    }
    final id = item['id'] as String;
    final clip = binding.clips[id];
    if (clip == null || !seen.add(id)) invalid();
    final scalarOffsets = <int>[];
    final scalars = clip.text.runes.toList();
    for (var i = 0; i < scalars.length; i++) {
      if (_cjk(scalars[i])) scalarOffsets.add(i);
    }
    final marks = <Map<String, Object?>>[];
    final marked = <int>{};
    for (final ordinal in item['marks'] as List) {
      if (ordinal is! int ||
          ordinal < 0 ||
          ordinal >= scalarOffsets.length ||
          !marked.add(ordinal)) {
        invalid();
      }
      final start = scalarOffsets[ordinal];
      marks.add({
        'start': start,
        'end': start + 1,
        'verdict': 'reject',
        'note': null,
      });
    }
    final note = jsonEncode({
      'format': 'ggtts-shared-notes/1',
      'page_note': pageNote,
      'item_note': item['note'],
    });
    if (note.runes.length > 4000) invalid();
    submissions.add({
      'clip_id': clip.sourceId,
      'verdict': item['rating'],
      'marks': marks,
      'note': note,
    });
  }
  return submissions;
}
