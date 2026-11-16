import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

/// Only the source API receives the bearer token. Audio URLs are fetched anonymously.
abstract class ListeningSource {
  String get identity;
  Future<void> signIn(String user, String password);
  Future<List<ListeningSession>> sessions();
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip);
  Future<void> submit(String sessionId, Map<String, Object?> body, String key);
}

class ListeningAudio {
  ListeningAudio(this.bytes, {this.refreshed = false});
  final Uint8List bytes;
  final bool refreshed;
}

class ListeningClip {
  ListeningClip(this.id, this.text, this.audioUrl);
  final String id, text;
  Uri audioUrl;
  factory ListeningClip.fromJson(Map<String, dynamic> j) => ListeningClip(
    j['id'] as String,
    j['text'] as String,
    Uri.parse(j['audio_url'] as String),
  );
}

class ListeningSession {
  ListeningSession(this.id, this.title, this.clips);
  final String id, title;
  final List<ListeningClip> clips;
  factory ListeningSession.fromJson(Map<String, dynamic> j) => ListeningSession(
    j['id'] as String,
    j['title'] as String,
    (j['clips'] as List)
        .map((c) => ListeningClip.fromJson(Map<String, dynamic>.from(c as Map)))
        .toList(),
  );
}

class ListeningException implements Exception {
  ListeningException(this.message, {this.automaticRetry = true});
  final String message;
  final bool automaticRetry;
  @override
  String toString() => message;
}

bool _safeSource(Uri u) =>
    u.scheme == 'https' ||
    (u.scheme == 'http' &&
        (u.host == 'localhost' || u.host == '127.0.0.1' || u.host == '::1'));

class HttpListeningSource implements ListeningSource {
  HttpListeningSource(this.base) {
    if (!_safeSource(base) ||
        base.host.isEmpty ||
        base.userInfo.isNotEmpty ||
        base.hasQuery ||
        base.hasFragment) {
      throw ListeningException(
        'Use an HTTPS source URL (HTTP is allowed for localhost).',
      );
    }
  }
  final Uri base;
  String? _token;
  @override
  String get identity => base.toString();

  Uri _api(String path) => base.resolve(path);
  Future<(int, Uint8List)> _request(
    Uri uri, {
    String method = 'GET',
    Map<String, Object?>? body,
    bool authenticated = false,
    String? key,
  }) async {
    if (!_safeSource(uri) || uri.userInfo.isNotEmpty) {
      throw ListeningException('The source returned an unsafe URL.');
    }
    final client = HttpClient();
    try {
      final request = await client.openUrl(method, uri);
      request.followRedirects = false;
      if (authenticated) {
        if (_token == null) throw ListeningException('Please sign in again.');
        request.headers.set(HttpHeaders.authorizationHeader, 'Bearer $_token');
      }
      if (key != null) request.headers.set('Idempotency-Key', key);
      if (body != null) {
        request.headers.contentType = ContentType.json;
        request.write(jsonEncode(body));
      }
      final response = await request.close();
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response) {
        builder.add(chunk);
      }
      return (response.statusCode, builder.takeBytes());
    } on ListeningException {
      rethrow;
    } catch (_) {
      throw ListeningException(
        'Could not contact the listening source. Try again.',
      );
    } finally {
      client.close(force: true);
    }
  }

  Map<String, dynamic> _json(Uint8List data) {
    try {
      return Map<String, dynamic>.from(jsonDecode(utf8.decode(data)) as Map);
    } catch (_) {
      throw ListeningException('The source returned an invalid response.');
    }
  }

  void _ok(int status, Uint8List bytes) {
    if (status == 401) {
      throw ListeningException('Sign-in expired. Please sign in again.');
    }
    if (status < 200 || status >= 300) {
      var retryable = status == 408 || status == 429 || status >= 500;
      try {
        final error = _json(bytes)['error'];
        if (error is Map && error['retryable'] is bool) {
          retryable = error['retryable'] as bool;
        }
      } catch (_) {
        // An absent or malformed error body falls back to the status class.
      }
      // Authorization failures are never retried automatically, even if mislabelled.
      if (status == 403) retryable = false;
      throw ListeningException(
        'The source could not complete the request (HTTP $status).',
        automaticRetry: retryable,
      );
    }
  }

  @override
  Future<void> signIn(String user, String password) async {
    _token = null;
    final (status, bytes) = await _request(
      _api('/v1/auth/token'),
      method: 'POST',
      body: {'username': user, 'password': password},
    );
    _ok(status, bytes);
    final token = _json(bytes)['access_token'];
    if (token is! String || token.isEmpty) {
      throw ListeningException(
        'The source returned an invalid sign-in response.',
      );
    }
    _token = token;
  }

  @override
  Future<List<ListeningSession>> sessions() async {
    final (status, bytes) = await _request(
      _api('/v1/listening-sessions'),
      authenticated: true,
    );
    _ok(status, bytes);
    try {
      return (_json(bytes)['sessions'] as List)
          .map(
            (s) =>
                ListeningSession.fromJson(Map<String, dynamic>.from(s as Map)),
          )
          .toList();
    } catch (_) {
      throw ListeningException('The source returned an invalid session list.');
    }
  }

  @override
  Future<ListeningAudio> audio(String sessionId, ListeningClip clip) async {
    var refreshed = false;
    var (status, bytes) = await _request(clip.audioUrl);
    if (status == 403) {
      refreshed = true;
      final path =
          '/v1/listening-sessions/${Uri.encodeComponent(sessionId)}/clips/${Uri.encodeComponent(clip.id)}/audio-url';
      final (refreshStatus, refreshBytes) = await _request(
        _api(path),
        authenticated: true,
      );
      _ok(refreshStatus, refreshBytes);
      try {
        clip.audioUrl = Uri.parse(_json(refreshBytes)['audio_url'] as String);
      } catch (_) {
        throw ListeningException('The source returned an invalid audio URL.');
      }
      (status, bytes) = await _request(clip.audioUrl);
    }
    _ok(status, bytes);
    return ListeningAudio(bytes, refreshed: refreshed);
  }

  @override
  Future<void> submit(
    String sessionId,
    Map<String, Object?> body,
    String key,
  ) async {
    final (status, bytes) = await _request(
      _api('/v1/listening-sessions/${Uri.encodeComponent(sessionId)}/marks'),
      method: 'POST',
      authenticated: true,
      body: body,
      key: key,
    );
    _ok(status, bytes);
    final result = _json(bytes);
    if (status != 200 ||
        result['accepted'] != true ||
        result['clip_id'] != body['clip_id']) {
      throw ListeningException(
        'The source did not confirm the submitted clip.',
      );
    }
  }
}

/// Durable queue contains marks and their scope, never passwords, bearer tokens or signed URLs.
class ListeningQueue {
  ListeningQueue(this.file);
  String? lastError;
  final File file;
  Future<List<Map<String, dynamic>>> read() async {
    if (!await file.exists()) return [];
    try {
      return (jsonDecode(await file.readAsString()) as List)
          .map((v) => Map<String, dynamic>.from(v as Map))
          .toList();
    } catch (_) {
      throw ListeningException('The pending marks file could not be read.');
    }
  }

  Future<void> write(List<Map<String, dynamic>> pending) async {
    await file.parent.create(recursive: true);
    final temp = File('${file.path}.tmp');
    await temp.writeAsString(jsonEncode(pending), flush: true);
    await temp.rename(file.path);
  }

  Future<void> enqueue(Map<String, dynamic> item) async {
    final items = await read();
    items.add(item);
    await write(items);
  }

  Future<int> retry(
    ListeningSource source,
    String user, {
    bool automatic = false,
  }) async {
    final items = await read();
    var sent = 0;
    lastError = null;
    for (final item in List<Map<String, dynamic>>.from(items)) {
      if (item['source'] != source.identity || item['user'] != user) continue;
      if (automatic && item['manual_only'] == true) {
        lastError = 'The source rejected these marks. Review access or submission before retrying.';
        break;
      }
      try {
        await source.submit(
          item['session'] as String,
          Map<String, Object?>.from(item['body'] as Map),
          item['key'] as String,
        );
        items.remove(item);
        await write(items);
        sent++;
      } catch (e) {
        lastError = e is ListeningException
            ? e.message
            : 'Upload failed. Try again.';
        if (e is ListeningException && !e.automaticRetry) {
          item['manual_only'] = true;
          await write(items);
        }
        break;
      }
    }
    return sent;
  }
}

String newSubmissionKey() {
  final random = Random.secure();
  return List.generate(
    16,
    (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0'),
  ).join();
}
