import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Loopback source for Flutter integration tests. It never accesses storage.
class ListeningSourceFake {
  ListeningSourceFake._(
    this._server,
    this.expireFirstAudio,
    this.failFirstSubmission,
    this.clipText,
  );

  final HttpServer _server;
  final bool expireFirstAudio;
  final bool failFirstSubmission;
  final String clipText;
  final List<FakeSourceRequest> requests = [];
  final List<Map<String, Object?>> submissions = [];
  final Map<
    String,
    ({String key, Map<String, Object?> body, Map<String, Object?> response})
  >
  _accepted = {};
  bool _expiredAudioServed = false;
  bool _submissionFailureServed = false;
  bool _closed = false;
  // Synthetic 440 Hz sine wave, 250 ms at 24 kHz mono; no speech or corpus content.
  final Uint8List audioBytes = base64Decode(
    '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',
  );

  Uri get baseUrl => Uri(scheme: 'http', host: '127.0.0.1', port: _server.port);

  static Future<ListeningSourceFake> start({
    bool expireFirstAudio = true,
    bool failFirstSubmission = true,
    String clipText = 'A \u{1F60A} cafe\u0301.',
  }) async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    final fake = ListeningSourceFake._(
      server,
      expireFirstAudio,
      failFirstSubmission,
      clipText,
    );
    unawaited(server.forEach(fake._handle));
    return fake;
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _server.close(force: true);
  }

  Future<void> _json(HttpResponse response, int status, Object value) async {
    response.statusCode = status;
    response.headers.contentType = ContentType.json;
    response.write(jsonEncode(value));
    await response.close();
  }

  Future<Map<String, Object?>> _readJson(HttpRequest request) async {
    final text = await utf8.decoder.bind(request).join();
    if (text.isEmpty) return <String, Object?>{};
    return Map<String, Object?>.from(jsonDecode(text) as Map);
  }

  bool _authenticated(HttpRequest request) =>
      request.headers.value(HttpHeaders.authorizationHeader) ==
      'Bearer fake-listener-token';

  String get _expiredAudioPath => '/audio/expired';
  String get _freshAudioPath => '/audio/fresh';

  Map<String, Object?> _sessionPayload() => {
    'sessions': [
      {
        'id': 'session-1',
        'title': 'Assigned test session',
        'clips': [
          {
            'id': 'clip-1',
            'text': clipText,
            'audio_url': baseUrl
                .replace(
                  path: _expiredAudioPath,
                  queryParameters: {'sig': 'expired'},
                )
                .toString(),
            'audio_url_expires_at': '2000-01-01T00:00:00Z',
          },
        ],
      },
    ],
  };

  Future<void> _handle(HttpRequest request) async {
    final rawPath = request.uri.path;
    final safePath = rawPath; // Query string is deliberately excluded from the fake request log.
    Map<String, Object?>? body;
    if (request.method == 'POST' && rawPath.endsWith('/marks')) {
      body = await _readJson(request);
    } else if (request.method == 'POST' && rawPath == '/v1/auth/token') {
      body = await _readJson(request);
    }
    requests.add(
      FakeSourceRequest(
        method: request.method,
        path: safePath,
        hasAuthorization:
            request.headers.value(HttpHeaders.authorizationHeader) != null,
        idempotencyKey: request.headers.value('Idempotency-Key'),
        body: rawPath == '/v1/auth/token' ? null : body,
      ),
    );

    if (request.method == 'POST' && rawPath == '/v1/auth/token') {
      await _json(request.response, 200, {
        'access_token': 'fake-listener-token',
        'token_type': 'Bearer',
        'expires_at': '2099-01-01T00:00:00Z',
      });
      return;
    }
    if (rawPath.startsWith('/audio/')) {
      if (request.headers.value(HttpHeaders.authorizationHeader) != null) {
        await _json(request.response, 400, {
          'error': {
            'code': 'audio_auth_forbidden',
            'message': 'Audio must be fetched anonymously.',
            'retryable': false,
          },
        });
        return;
      }
      if (rawPath == _expiredAudioPath &&
          expireFirstAudio &&
          !_expiredAudioServed) {
        _expiredAudioServed = true;
        await _json(request.response, 403, {
          'error': {
            'code': 'url_expired',
            'message': 'Audio link expired.',
            'retryable': false,
          },
        });
        return;
      }
      if (rawPath != _freshAudioPath && rawPath != _expiredAudioPath) {
        await _json(request.response, 404, {
          'error': {
            'code': 'not_found',
            'message': 'Audio not found.',
            'retryable': false,
          },
        });
        return;
      }
      request.response.statusCode = 200;
      request.response.headers.contentType = ContentType('audio', 'flac');
      request.response.add(audioBytes);
      await request.response.close();
      return;
    }

    if (!_authenticated(request)) {
      await _json(request.response, 401, {
        'error': {
          'code': 'token_expired',
          'message': 'Sign in again.',
          'retryable': false,
        },
      });
      return;
    }
    if (request.method == 'GET' && rawPath == '/v1/listening-sessions') {
      await _json(request.response, 200, _sessionPayload());
      return;
    }
    if (request.method == 'GET' &&
        rawPath == '/v1/listening-sessions/session-1/clips/clip-1/audio-url') {
      await _json(request.response, 200, {
        'audio_url': baseUrl
            .replace(path: _freshAudioPath, queryParameters: {'sig': 'fresh'})
            .toString(),
        'expires_at': '2099-01-01T00:00:00Z',
      });
      return;
    }
    if (request.method == 'POST' &&
        rawPath == '/v1/listening-sessions/session-1/marks') {
      final key = request.headers.value('Idempotency-Key');
      if (key == null || key.isEmpty || body == null) {
        await _json(request.response, 400, {
          'error': {
            'code': 'invalid_submission',
            'message': 'A key and JSON body are required.',
            'retryable': false,
          },
        });
        return;
      }
      final prior = _accepted[key];
      if (prior != null) {
        if (jsonEncode(prior.body) != jsonEncode(body)) {
          await _json(request.response, 409, {
            'error': {
              'code': 'idempotency_conflict',
              'message': 'Key was already used for another body.',
              'retryable': false,
            },
          });
        } else {
          await _json(request.response, 200, prior.response);
        }
        return;
      }
      if (failFirstSubmission && !_submissionFailureServed) {
        _submissionFailureServed = true;
        await _json(request.response, 503, {
          'error': {
            'code': 'temporary_failure',
            'message': 'Try again.',
            'retryable': true,
          },
        });
        return;
      }
      submissions.add(body);
      final response = <String, Object?>{
        'accepted': true,
        'clip_id': body['clip_id'],
      };
      _accepted[key] = (key: key, body: body, response: response);
      await _json(request.response, 200, response);
      return;
    }
    await _json(request.response, 404, {
      'error': {
        'code': 'not_found',
        'message': 'Route not found.',
        'retryable': false,
      },
    });
  }
}

class FakeSourceRequest {
  const FakeSourceRequest({
    required this.method,
    required this.path,
    required this.hasAuthorization,
    required this.idempotencyKey,
    required this.body,
  });
  final String method;
  final String path;
  final bool hasAuthorization;
  final String? idempotencyKey;
  final Map<String, Object?>? body;
}
