import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ggtts_app/listening_source.dart';

void main() {
  test('expired audio refreshes using API credentials only, then posts marks', () async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    addTearDown(() => server.close(force: true));
    final seen = <String>[];
    final auth = <String?>[];
    server.listen((r) async {
      seen.add('${r.method} ${r.uri.path}');
      auth.add(r.headers.value(HttpHeaders.authorizationHeader));
      r.response.headers.contentType = ContentType.json;
      switch (r.uri.path) {
        case '/v1/auth/token':
          r.response.write(jsonEncode({'access_token': 'bearer-secret'}));
        case '/v1/listening-sessions':
          r.response.write(
            jsonEncode({
              'sessions': [
                {
                  'id': 'session',
                  'title': 'Assigned',
                  'clips': [
                    {
                      'id': 'clip',
                      'text': 'A😀é',
                      'audio_url':
                          'http://127.0.0.1:${server.port}/expired?signature=secret',
                    },
                  ],
                },
              ],
            }),
          );
        case '/expired':
          r.response.statusCode = 403;
        case '/v1/listening-sessions/session/clips/clip/audio-url':
          r.response.write(
            jsonEncode({
              'audio_url':
                  'http://127.0.0.1:${server.port}/fresh?signature=secret',
            }),
          );
        case '/fresh':
          r.response.add([1, 2, 3]);
        case '/v1/listening-sessions/session/marks':
          expect(r.headers.value('Idempotency-Key'), 'stable-key');
          expect(
            jsonDecode(await utf8.decoder.bind(r).join())['marks'],
            isNotEmpty,
          );
          r.response.write(jsonEncode({'accepted': true, 'clip_id': 'clip'}));
        default:
          r.response.statusCode = 404;
      }
      await r.response.close();
    });
    final source = HttpListeningSource(
      Uri.parse('http://127.0.0.1:${server.port}'),
    );
    await source.signIn('user', 'password-secret');
    final session = (await source.sessions()).single;
    expect(session.clips.single.text.runes.length, 4);
    final audio = await source.audio(session.id, session.clips.single);
    expect(audio.refreshed, isTrue);
    expect(audio.bytes, [1, 2, 3]);
    await source.submit(session.id, {
      'clip_id': 'clip',
      'marks': [
        {'start': 1, 'end': 2},
      ],
    }, 'stable-key');
    expect(seen, [
      'POST /v1/auth/token',
      'GET /v1/listening-sessions',
      'GET /expired',
      'GET /v1/listening-sessions/session/clips/clip/audio-url',
      'GET /fresh',
      'POST /v1/listening-sessions/session/marks',
    ]);
    expect(auth, [
      null,
      'Bearer bearer-secret',
      null,
      'Bearer bearer-secret',
      null,
      'Bearer bearer-secret',
    ]);
  });

  test(
    'expired sign-in keeps queued marks for a newly authenticated client',
    () async {
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => server.close(force: true));
      final dir = await Directory.systemTemp.createTemp('listening-reauth-');
      addTearDown(() => dir.delete(recursive: true));
      var signins = 0;
      final keys = <String?>[];
      final tokens = <String?>[];
      server.listen((request) async {
        request.response.headers.contentType = ContentType.json;
        if (request.uri.path == '/v1/auth/token') {
          signins++;
          request.response.write(
            jsonEncode({'access_token': 'token-$signins'}),
          );
        } else {
          keys.add(request.headers.value('Idempotency-Key'));
          tokens.add(request.headers.value(HttpHeaders.authorizationHeader));
          if (tokens.last == 'Bearer token-1') {
            request.response.statusCode = 401;
          } else {
            request.response.write(
              jsonEncode({'accepted': true, 'clip_id': 'c'}),
            );
          }
        }
        await request.response.close();
      });
      final uri = Uri.parse('http://127.0.0.1:${server.port}');
      final first = HttpListeningSource(uri);
      await first.signIn('alice', 'password');
      final file = File('${dir.path}/queue.json');
      final queue = ListeningQueue(file);
      await queue.enqueue({
        'source': first.identity,
        'user': 'alice',
        'session': 's',
        'key': 'persisted-key',
        'body': {'clip_id': 'c', 'marks': []},
      });
      expect(await queue.retry(first, 'alice'), 0);
      expect((await queue.read()).length, 1);
      final restarted = HttpListeningSource(uri);
      await restarted.signIn('alice', 'password');
      expect(await ListeningQueue(file).retry(restarted, 'alice'), 1);
      expect(await ListeningQueue(file).read(), isEmpty);
      expect(keys, ['persisted-key', 'persisted-key']);
      expect(tokens, ['Bearer token-1', 'Bearer token-2']);
      expect(await file.readAsString(), isNot(contains('token-')));
    },
  );

  test(
    'queue persists and retries only matching source and user without secrets',
    () async {
      final dir = await Directory.systemTemp.createTemp('listening-queue-');
      addTearDown(() => dir.delete(recursive: true));
      final queue = ListeningQueue(File('${dir.path}/pending.json'));
      Map<String, dynamic> item(String user, String key) => {
        'source': 'https://example.test',
        'user': user,
        'session': 's',
        'key': key,
        'body': {
          'clip_id': 'c',
          'marks': [
            {'start': 1, 'end': 2, 'verdict': 'reject'},
          ],
          'note': null,
          'verdict': 'unsure',
        },
      };
      await queue.enqueue(item('alice', 'stable'));
      await queue.enqueue(item('bob', 'other'));
      final freshQueue = ListeningQueue(queue.file);
      final source = QueueSource();
      expect(await freshQueue.retry(source, 'alice'), 0);
      expect((await freshQueue.read()).length, 2);
      expect(await freshQueue.retry(source, 'alice'), 1);
      expect(source.keys, ['stable', 'stable']);
      expect((await freshQueue.read()).single['user'], 'bob');
      final persisted = await queue.file.readAsString();
      expect(persisted, isNot(contains('bearer-secret')));
      expect(persisted, isNot(contains('password-secret')));
      expect(persisted, isNot(contains('signature=')));
    },
  );
}

class QueueSource implements ListeningSource {
  bool fail = true;
  final keys = <String>[];
  @override
  String get identity => 'https://example.test';
  @override
  Future<void> submit(
    String sessionId,
    Map<String, Object?> body,
    String key,
  ) async {
    keys.add(key);
    if (fail) {
      fail = false;
      throw ListeningException('offline');
    }
  }

  @override
  Future<void> signIn(String user, String password) async {}
  @override
  Future<List<ListeningSession>> sessions() async => [];
  @override
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip) =>
      throw UnimplementedError();
}
