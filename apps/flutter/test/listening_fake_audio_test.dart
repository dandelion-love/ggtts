import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'support/listening_source_fake.dart';

void main() {
  test('fake serves a complete synthetic FLAC stream with 250 ms of mono audio', () async {
    final fake = await ListeningSourceFake.start(expireFirstAudio: false);
    addTearDown(fake.close);
    final client = HttpClient();
    addTearDown(() => client.close(force: true));
    final response = await (await client.getUrl(
      fake.baseUrl.replace(path: '/audio/fresh'),
    )).close();
    final bytes = await response.fold<List<int>>(
      [],
      (all, part) => all..addAll(part),
    );
    expect(response.statusCode, 200);
    expect(bytes, fake.audioBytes);
    expect(ascii.decode(bytes.take(4).toList()), 'fLaC');
    expect(bytes.length, greaterThan(100));
    // STREAMINFO sample rate, channel count, bit depth and total-sample count.
    var packed = 0;
    for (final b in bytes.sublist(18, 26)) {
      packed = (packed << 8) | b;
    }
    expect(packed >> 44, 24000);
    expect(((packed >> 41) & 7) + 1, 1);
    expect(((packed >> 36) & 31) + 1, 16);
    expect(packed & ((1 << 36) - 1), 6000);
  });
}
