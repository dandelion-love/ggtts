import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/diagnostics_server.dart';

void main() {
  const token =
      '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
  test('rejects malformed token before opening listener', () async {
    await expectLater(
      DiagnosticsServer.start('short', (_, _) async => null),
      throwsArgumentError,
    );
  });

  test(
    'authorizes only bounded fixed operations and keeps stop available',
    () async {
      final began = Completer<void>();
      final release = Completer<void>();
      var called = 0;
      String? seenText;
      final server = await DiagnosticsServer.start(token, (op, input) async {
        called++;
        if (op == 'text') seenText = input['text'] as String;
        if (op == 'speak') {
          began.complete();
          await release.future;
        }
        return {'ok': true};
      });
      final client = HttpClient();
      addTearDown(() async {
        client.close(force: true);
        await server.close();
      });

      Future<(int, String)> request(
        String path, {
        String? auth = token,
        String? origin,
        String? host,
        String body = '{}',
      }) async {
        final req = await client.postUrl(
          Uri.parse('http://127.0.0.1:${server.port}/$path'),
        );
        if (auth != null) {
          req.headers.set(HttpHeaders.authorizationHeader, 'Bearer $auth');
        }
        if (origin != null) req.headers.set('Origin', origin);
        if (host != null) req.headers.set(HttpHeaders.hostHeader, host);
        req.add(utf8.encode(body));
        final response = await req.close();
        return (response.statusCode, await utf8.decoder.bind(response).join());
      }

      expect((await request('status', auth: null)).$1, 403);
      expect((await request('status', auth: 'wrong')).$1, 403);
      expect((await request('status', origin: 'https://example.test')).$1, 403);
      expect((await request('status', host: 'evil.test')).$1, 403);
      expect((await request('shell')).$1, 404);
      expect(
        (await request(
          'text',
          body: jsonEncode({'text': '\u4e2d\u{20000}'}),
        )).$1,
        200,
      );
      expect(seenText, '\u4e2d\u{20000}');
      expect(
        (await request('text', body: jsonEncode({'text': 'x' * 16001}))).$1,
        400,
      );
      expect(
        (await request(
          'text',
          body: jsonEncode({'text': 'x', 'path': '/etc/passwd'}),
        )).$1,
        400,
      );
      expect(
        (await request('audio', body: jsonEncode({'path': '/etc/passwd'}))).$1,
        400,
      );
      expect((await request('text', body: 'x' * 65537)).$1, 413);
      expect(
        (await request('reopen', body: jsonEncode({'id': '../x'}))).$1,
        400,
      );
      final speaking = request('speak');
      await began.future.timeout(const Duration(seconds: 3));
      expect((await request('save')).$1, 409);
      expect((await request('status')).$1, 200);
      expect((await request('screenshot')).$1, 200);
      expect((await request('stop')).$1, 200);
      release.complete();
      expect((await speaking).$1, 200);
      expect(called, 5);
    },
  );
}
