import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Narrow developer-only loopback protocol. The caller owns all actual actions.
class DiagnosticsServer {
  DiagnosticsServer._(this._server, this._token, this._action);
  final HttpServer _server;
  final String _token;
  final Future<Object?> Function(String, Map<String, Object?>) _action;
  bool _closed = false;
  bool _busy = false;
  int _inflight = 0;
  int get port => _server.port;

  static Future<DiagnosticsServer> start(
    String token,
    Future<Object?> Function(String, Map<String, Object?>) action,
  ) async {
    if (!RegExp(r'^[0-9a-f]{64}$').hasMatch(token)) {
      throw ArgumentError(
        'diagnostics token must be 64 lowercase hexadecimal digits',
      );
    }
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    final result = DiagnosticsServer._(server, token, action);
    server.listen(result._handle);
    return result;
  }

  bool _authenticated(String? supplied) {
    final actual = supplied ?? '';
    final expected = 'Bearer $_token';
    var different = actual.length ^ expected.length;
    for (var i = 0; i < expected.length; i++) {
      different |=
          expected.codeUnitAt(i) ^
          (i < actual.length ? actual.codeUnitAt(i) : 0);
    }
    return different == 0;
  }

  Future<void> _handle(HttpRequest req) async {
    final response = req.response;
    // Early refusals may leave an unread request body.
    response.persistentConnection = false;
    if (_inflight >= 8) {
      response.statusCode = HttpStatus.serviceUnavailable;
      await response.close();
      return;
    }
    _inflight++;
    try {
      response.headers.set('Cache-Control', 'no-store');
      response.headers.set('X-Content-Type-Options', 'nosniff');
      final host = req.headers.value(HttpHeaders.hostHeader);
      if (req.connectionInfo?.remoteAddress.isLoopback != true ||
          host != '127.0.0.1:$port' ||
          req.headers.value('origin') != null ||
          !_authenticated(req.headers.value(HttpHeaders.authorizationHeader))) {
        response.statusCode = HttpStatus.forbidden;
        return;
      }
      if (req.method != 'POST' ||
          req.uri.hasQuery ||
          req.uri.hasFragment ||
          !const {
            '/status',
            '/text',
            '/prepare',
            '/speak',
            '/stop',
            '/save',
            '/replay',
            '/reopen',
            '/screenshot',
            '/audio',
          }.contains(req.uri.path)) {
        response.statusCode = HttpStatus.notFound;
        return;
      }
      const bodyLimit = 65536;
      // Drain only a small bounded excess before refusing. Closing a socket
      // while the peer is still sending can reset the response on Windows.
      const drainLimit = 131072;
      if (req.contentLength > drainLimit) {
        response.statusCode = HttpStatus.requestEntityTooLarge;
        return;
      }
      final bytes = BytesBuilder(copy: false);
      var received = 0;
      final bodyClock = Stopwatch()..start();
      await for (final chunk in req.timeout(const Duration(seconds: 5))) {
        if (bodyClock.elapsed > const Duration(seconds: 10)) {
          throw TimeoutException('request body deadline');
        }
        received += chunk.length;
        if (received > drainLimit) {
          response.statusCode = HttpStatus.requestEntityTooLarge;
          return;
        }
        if (received <= bodyLimit) bytes.add(chunk);
      }
      if (received > bodyLimit) {
        response.statusCode = HttpStatus.requestEntityTooLarge;
        return;
      }
      final raw = bytes.takeBytes();
      final dynamic decoded = raw.isEmpty
          ? <String, Object?>{}
          : jsonDecode(utf8.decode(raw));
      if (decoded is! Map<String, dynamic>) throw const FormatException();
      final operation = req.uri.path.substring(1);
      final input = Map<String, Object?>.from(decoded);
      if (operation == 'text') {
        if (input.length != 1 ||
            input['text'] is! String ||
            (input['text'] as String).isEmpty ||
            (input['text'] as String).length > 16000) {
          throw const FormatException();
        }
      } else if (operation == 'reopen') {
        if (input.length != 1 ||
            input['id'] is! String ||
            !RegExp(r'^[0-9a-f]{32}$').hasMatch(input['id'] as String)) {
          throw const FormatException();
        }
      } else if (input.isNotEmpty) {
        throw const FormatException();
      }
      final readonly =
          operation == 'status' ||
          operation == 'stop' ||
          operation == 'screenshot';
      if (!readonly && _busy) {
        response.statusCode = HttpStatus.conflict;
        return;
      }
      if (!readonly) _busy = true;
      try {
        final result = await _action(operation, input);
        if (result is Uint8List) {
          response.headers.contentType = operation == 'screenshot'
              ? ContentType('image', 'png')
              : ContentType('audio', 'flac');
          response.add(result);
        } else {
          response.headers.contentType = ContentType.json;
          response.write(jsonEncode(result));
        }
      } finally {
        if (!readonly) _busy = false;
      }
    } on FormatException {
      response.statusCode = HttpStatus.badRequest;
    } on TimeoutException {
      response.statusCode = HttpStatus.requestTimeout;
    } catch (_) {
      response.statusCode = HttpStatus.internalServerError;
    } finally {
      _inflight--;
      await response.close();
    }
  }

  Future<void> close() async {
    if (!_closed) {
      _closed = true;
      await _server.close(force: true);
    }
  }
}
