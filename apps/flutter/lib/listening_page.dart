import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';

import 'listening_source.dart';
import 'shared_feedback_mapping.dart';

typedef SourceFactory = ListeningSource Function(Uri uri);
typedef AudioPlay = Future<void> Function(Uint8List bytes);

class ListeningPage extends StatefulWidget {
  const ListeningPage({
    super.key,
    this.sourceFactory,
    this.queue,
    this.play,
    this.feedbackDirectory,
  });
  final SourceFactory? sourceFactory;
  final ListeningQueue? queue;
  final AudioPlay? play;
  final Future<Directory> Function()? feedbackDirectory;
  @override
  State<ListeningPage> createState() => _ListeningPageState();
}

class _ListeningPageState extends State<ListeningPage> {
  final _url = TextEditingController();
  final _user = TextEditingController();
  final _password = TextEditingController();
  final _result = TextEditingController();
  ListeningSource? _source;
  ListeningQueue? _queue;
  List<ListeningSession>? _sessions;
  ListeningSession? _session;
  ListeningClip? _clip;
  AudioPlayer? _player;
  FeedbackBinding? _binding;
  String? _pagePath;
  String? _username;
  String? _status;
  bool _busy = false;
  bool _imported = false;
  bool _allowPop = false;

  @override
  void dispose() {
    _url.dispose();
    _user.dispose();
    _password.dispose();
    _result.dispose();
    _player?.dispose();
    super.dispose();
  }

  Future<void> _run(Future<void> Function() f) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _status = null;
    });
    try {
      await f();
    } catch (e) {
      if (mounted) {
        setState(
          () => _status = e is ListeningException
              ? e.message
              : 'Could not complete the action. Please try again.',
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _signIn() => _run(() async {
    final uri = Uri.tryParse(_url.text.trim());
    final username = _user.text.trim();
    if (uri == null || username.isEmpty || _password.text.isEmpty) {
      throw ListeningException('Enter source URL, username and password.');
    }
    final source = (widget.sourceFactory ?? HttpListeningSource.new)(uri);
    await source.signIn(username, _password.text);
    _password.clear();
    final queue =
        widget.queue ??
        ListeningQueue(
          File(
            '${(await getApplicationSupportDirectory()).path}/listening-pending.json',
          ),
        );
    final sessions = await source.sessions();
    final sent = await queue.retry(source, username, automatic: true);
    if (!mounted) return;
    setState(() {
      _source = source;
      _queue = queue;
      _username = username;
      _sessions = sessions;
      _session = null;
      _clip = null;
      _status = sent > 0
          ? 'Uploaded $sent pending submissions.'
          : queue.lastError ?? 'Signed in.';
    });
  });

  Future<void> _retry() => _run(() async {
    final sent = await _queue!.retry(_source!, _username!);
    final pending = (await _queue!.read())
        .where(
          (item) =>
              item['source'] == _source!.identity && item['user'] == _username,
        )
        .length;
    if (mounted) {
      setState(
        () => _status = pending == 0
            ? 'Pending feedback uploaded ($sent sent).'
            : '${_queue!.lastError ?? 'Upload failed.'} $pending still queued.',
      );
    }
  });

  Future<void> _play() => _run(() async {
    final audio = await _source!.audio(_session!.id, _clip!);
    if (widget.play != null) {
      await widget.play!(audio.bytes);
    } else {
      await (_player ??= AudioPlayer()).play(BytesSource(audio.bytes));
    }
    if (mounted) {
      setState(
        () => _status = audio.refreshed
            ? 'Audio link expired; refreshed and playing.'
            : 'Playing clip.',
      );
    }
  });

  Future<Directory> _feedbackDir() =>
      (widget.feedbackDirectory ?? getApplicationSupportDirectory)();

  String _stateName(ListeningClip clip) => sha256
      .convert(
        utf8.encode(
          jsonEncode([_source!.identity, _username!, _session!.id, clip.id]),
        ),
      )
      .toString();

  Future<void> _selectClip(ListeningClip clip) => _run(() async {
    final dir = await _feedbackDir();
    final state = File('${dir.path}/feedback-${_stateName(clip)}.json');
    FeedbackBinding? binding;
    String? pagePath;
    if (await state.exists()) {
      try {
        final doc = jsonDecode(await state.readAsString()) as Map;
        if (doc['source'] != _source!.identity ||
            doc['user'] != _username ||
            doc['session'] != _session!.id ||
            doc['clip'] != clip.id ||
            doc['text'] != clip.text ||
            doc['canonical'] is! String ||
            doc['audio_sha256'] is! String ||
            doc['page_path'] is! String) {
          throw const FormatException();
        }
        binding = FeedbackBinding.fromCanonicalInput(
          utf8.encode(doc['canonical']),
          {'clip': FeedbackClip(clip.id, clip.text, doc['audio_sha256'])},
        );
        pagePath = doc['page_path'];
      } catch (_) {
        throw ListeningException(
          'Saved feedback binding is invalid; cannot reopen it.',
        );
      }
    }
    if (mounted) {
      setState(() {
        _clip = clip;
        _binding = binding;
        _pagePath = pagePath;
        _imported = false;
        if (binding != null) {
          _status = 'Reopened the saved feedback page binding.';
        }
      });
    }
  });

  Future<void> _prepare() => _run(() async {
    if (Platform.isAndroid || Platform.isIOS) {
      throw ListeningException(
        'Opening shared feedback files is not supported on this device yet.',
      );
    }
    final clip = _clip!;
    final audio = await _source!.audio(_session!.id, clip);
    final pageId = 'page-${newSubmissionKey()}';
    final input = <String, Object?>{
      'format': 'ggtts-feedback-input/1',
      'id': pageId,
      'items': [
        {
          'id': 'clip',
          'text': clip.text,
          'audio': 'data:audio/flac;base64,${base64Encode(audio.bytes)}',
        },
      ],
    };
    final canonical = canonicalFeedbackInput(input);
    FeedbackBinding binding;
    try {
      binding = FeedbackBinding.fromCanonicalInput(canonical, {
        'clip': FeedbackClip(
          clip.id,
          clip.text,
          sha256.convert(audio.bytes).toString(),
        ),
      });
    } on FormatException {
      throw ListeningException(
        'This clip is not a supported FLAC feedback input.',
      );
    }
    final template = await rootBundle.loadString('assets/feedback-page.html');
    const marker = '<!-- GGTTS_INPUT -->';
    if (marker.allMatches(template).length != 1) {
      throw ListeningException('The shared feedback page is unavailable.');
    }
    final embedded =
        '<meta name="feedback-input-sha256" content="${binding.inputSha256}">'
        '<script id="feedback-input" type="application/json">${utf8.decode(canonical)}</script>';
    final dir = await _feedbackDir();
    final file = File('${dir.path}/feedback-$pageId.html');
    await dir.create(recursive: true);
    await file.writeAsString(
      template.replaceFirst(marker, embedded),
      flush: true,
    );
    final stateFile = File('${dir.path}/feedback-${_stateName(clip)}.json');
    final tmp = File('${stateFile.path}.tmp');
    await tmp.writeAsString(
      jsonEncode({
        'source': _source!.identity,
        'user': _username!,
        'session': _session!.id,
        'clip': clip.id,
        'text': clip.text,
        'canonical': utf8.decode(canonical),
        'audio_sha256': sha256.convert(audio.bytes).toString(),
        'page_path': file.path,
      }),
      flush: true,
    );
    await tmp.rename(stateFile.path);
    if (mounted) {
      setState(() {
        _binding = binding;
        _pagePath = file.path;
        _imported = false;
        _result.clear();
        _status =
            'Open the saved page, export feedback, then paste the JSON below.';
      });
    }
  });

  Future<void> _submit() => _run(() async {
    if (_binding == null || _imported) {
      throw ListeningException('Prepare a page before importing feedback.');
    }
    List<Map<String, Object?>> bodies;
    try {
      bodies = mapSharedFeedback(_binding!, jsonDecode(_result.text));
    } on FormatException {
      throw ListeningException(
        'Feedback does not match this page or is invalid.',
      );
    }
    final key = sha256
        .convert(
          utf8.encode(jsonEncode([_binding!.inputSha256, bodies.single])),
        )
        .toString();
    final alreadyQueued = (await _queue!.read()).any(
      (item) =>
          item['source'] == _source!.identity &&
          item['user'] == _username &&
          item['session'] == _session!.id &&
          item['key'] == key,
    );
    if (!alreadyQueued) {
      await _queue!.enqueue({
        'source': _source!.identity,
        'user': _username!,
        'session': _session!.id,
        'key': key,
        'body': bodies.single,
      });
    }
    _imported = true;
    final sent = await _queue!.retry(_source!, _username!);
    final pending = (await _queue!.read()).any((item) => item['key'] == key);
    if (mounted) {
      setState(
        () => _status = pending
            ? '${_queue!.lastError ?? 'Upload failed.'} Feedback is queued; use Retry pending.'
            : 'Feedback uploaded ($sent submissions sent).',
      );
    }
  });

  Future<bool> _discard() async {
    if (_binding == null || _imported) return true;
    return await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Leave feedback page?'),
            content: const Text(
              'Unsubmitted feedback remains in the browser and cannot be imported after leaving.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text('Stay'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(context, true),
                child: const Text('Leave'),
              ),
            ],
          ),
        ) ??
        false;
  }

  void _clearPage() {
    _binding = null;
    _pagePath = null;
    _result.clear();
    _imported = false;
  }

  @override
  Widget build(BuildContext context) => PopScope(
    canPop: _allowPop || (!_busy && (_binding == null || _imported)),
    onPopInvokedWithResult: (popped, _) async {
      if (!popped && !_busy && await _discard() && context.mounted) {
        setState(() => _allowPop = true);
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (mounted) Navigator.of(context).pop();
        });
      }
    },
    child: Scaffold(
      appBar: AppBar(
        title: const Text('Listening sessions'),
        actions: [
          if (_source != null)
            IconButton(
              tooltip: 'Retry pending',
              onPressed: _busy ? null : _retry,
              icon: const Icon(Icons.sync),
            ),
          if (_source != null)
            IconButton(
              tooltip: 'Sign out',
              icon: const Icon(Icons.logout),
              onPressed: _busy
                  ? null
                  : () async {
                      if (await _discard() && mounted) {
                        setState(() {
                          _source = null;
                          _session = null;
                          _clip = null;
                          _username = null;
                          _clearPage();
                        });
                      }
                    },
            ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_source == null) ...[
                TextField(
                  key: const Key('listening-source'),
                  controller: _url,
                  decoration: const InputDecoration(labelText: 'Source URL'),
                ),
                TextField(
                  key: const Key('listening-user'),
                  controller: _user,
                  decoration: const InputDecoration(labelText: 'Username'),
                ),
                TextField(
                  key: const Key('listening-password'),
                  controller: _password,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Password'),
                ),
                FilledButton(
                  onPressed: _busy ? null : _signIn,
                  child: const Text('Sign in'),
                ),
              ] else if (_session == null) ...[
                const Text('Assigned sessions'),
                if (_sessions!.isEmpty) const Text('No sessions assigned.'),
                for (final session in _sessions!)
                  ListTile(
                    title: Text(session.title),
                    onTap: _busy
                        ? null
                        : () => setState(() => _session = session),
                  ),
              ] else if (_clip == null) ...[
                TextButton(
                  onPressed: _busy
                      ? null
                      : () => setState(() => _session = null),
                  child: const Text('All sessions'),
                ),
                Text(_session!.title),
                for (final clip in _session!.clips)
                  ListTile(
                    title: Text(clip.text),
                    onTap: _busy ? null : () => _selectClip(clip),
                  ),
              ] else ...[
                TextButton(
                  onPressed: _busy
                      ? null
                      : () async {
                          if (await _discard() && mounted) {
                            setState(() {
                              _clip = null;
                              _clearPage();
                            });
                          }
                        },
                  child: const Text('All clips'),
                ),
                FilledButton.icon(
                  onPressed: _busy ? null : _play,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Play clip'),
                ),
                FilledButton(
                  onPressed: _busy || _binding != null ? null : _prepare,
                  child: const Text('Prepare shared feedback page'),
                ),
                if (_pagePath != null) ...[
                  const Text(
                    'Open this file in a browser. Export feedback and paste the JSON below.',
                  ),
                  SelectableText(_pagePath!),
                  TextField(
                    key: const Key('shared-feedback-result'),
                    controller: _result,
                    maxLines: 8,
                    decoration: const InputDecoration(
                      labelText: 'Exported feedback JSON',
                    ),
                  ),
                  FilledButton(
                    onPressed: _busy || _imported ? null : _submit,
                    child: const Text('Submit feedback'),
                  ),
                ],
              ],
              if (_busy) const LinearProgressIndicator(),
              if (_status != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_status!),
                ),
            ],
          ),
        ),
      ),
    ),
  );
}
