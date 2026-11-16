import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/listening_source.dart';

void main() {
  test(
    'permanent 403 is retained and skipped after restart until explicit retry',
    () async {
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      final dir = await Directory.systemTemp.createTemp('ggtts-retry-');
      addTearDown(() async {
        await server.close(force: true);
        await dir.delete(recursive: true);
      });
      var posts = 0;
      var status = 403;
      server.listen((request) async {
        if (request.uri.path == '/v1/auth/token') {
          request.response.write('{"access_token":"token"}');
        } else {
          posts++;
          request.response.statusCode = status;
          if (status == 200) {
            request.response.write('{"accepted":true,"clip_id":"c"}');
          }
        }
        await request.response.close();
      });
      final source = HttpListeningSource(
        Uri.parse('http://127.0.0.1:${server.port}'),
      );
      await source.signIn('user', 'pass');
      final file = File('${dir.path}/queue.json');
      final queue = ListeningQueue(file);
      await queue.enqueue({
        'source': source.identity,
        'user': 'user',
        'session': 's',
        'key': 'same-key',
        'body': {
          'clip_id': 'c',
          'verdict': 'unsure',
          'marks': [],
          'note': null,
        },
      });
      expect(await queue.retry(source, 'user'), 0);
      expect(queue.lastError, contains('403'));
      expect((await queue.read()).single['manual_only'], true);
      final restarted = ListeningQueue(file);
      expect(await restarted.retry(source, 'user', automatic: true), 0);
      expect(posts, 1);
      expect(restarted.lastError, contains('rejected'));
      status = 200;
      expect(await restarted.retry(source, 'user'), 1);
      expect(posts, 2);
      expect(await restarted.read(), isEmpty);
    },
  );

  test('status classes distinguish reauthentication, transient and permanent errors', () async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    addTearDown(() => server.close(force: true));
    var status = 500;
    server.listen((request) async {
      if (request.uri.path == '/v1/auth/token') {
        request.response.write('{"access_token":"token"}');
      } else {
        request.response.statusCode = status;
      }
      await request.response.close();
    });
    final source = HttpListeningSource(
      Uri.parse('http://127.0.0.1:${server.port}'),
    );
    await source.signIn('user', 'pass');
    for (final pair in [
      (401, true),
      (403, false),
      (409, false),
      (429, true),
      (500, true),
    ]) {
      status = pair.$1;
      try {
        await source.submit('s', {'clip_id': 'c'}, 'key');
        fail('expected HTTP ${pair.$1} to fail');
      } on ListeningException catch (e) {
        expect(e.automaticRetry, pair.$2, reason: 'HTTP ${pair.$1}');
      }
    }
  });
}
