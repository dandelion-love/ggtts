import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';

import 'listening_source.dart';

typedef SourceFactory = ListeningSource Function(Uri uri);
typedef AudioPlay = Future<void> Function(Uint8List bytes);

class ListeningPage extends StatefulWidget {
  const ListeningPage({super.key, this.sourceFactory, this.queue, this.play});
  final SourceFactory? sourceFactory;
  final ListeningQueue? queue;
  final AudioPlay? play;

  @override
  State<ListeningPage> createState() => _ListeningPageState();
}

class _ListeningPageState extends State<ListeningPage> {
  final _url = TextEditingController();
  final _user = TextEditingController();
  final _password = TextEditingController();
  final _note = TextEditingController();
  ListeningSource? _source;
  ListeningQueue? _queue;
  AudioPlayer? _player;
  List<ListeningSession>? _sessions;
  ListeningSession? _session;
  ListeningClip? _clip;
  final Set<int> _marks = {};
  String _verdict = 'unsure';
  String _markVerdict = 'reject';
  bool _busy = false;
  String? _status;
  String? _signedInUser;
  String? _pendingKey;
  bool _allowPop = false;

  @override
  void dispose() {
    _url.dispose();
    _user.dispose();
    _password.dispose();
    _note.dispose();
    _player?.dispose();
    super.dispose();
  }

  Future<void> _run(Future<void> Function() action) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _status = null;
    });
    try {
      await action();
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
    if (uri == null) throw ListeningException('Enter a valid source URL.');
    final source = (widget.sourceFactory ?? HttpListeningSource.new)(uri);
    final username = _user.text.trim();
    if (username.isEmpty || _password.text.isEmpty) {
      throw ListeningException('Enter your username and password.');
    }
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
    final retried = await queue.retry(source, username, automatic: true);
    final pending = (await queue.read())
        .where((x) => x['source'] == source.identity && x['user'] == username)
        .toList();
    if (!mounted) return;
    setState(() {
      _source = source;
      _queue = queue;
      _signedInUser = username;
      _sessions = sessions;
      _session = null;
      _clip = null;
      _pendingKey = pending.isEmpty ? null : pending.first['key'] as String;
      _status = pending.isNotEmpty
          ? '${queue.lastError ?? 'Marks are queued.'} Use Retry pending when ready.'
          : (retried > 0
                ? 'Uploaded $retried pending submission(s).'
                : 'Signed in.');
    });
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

  Future<void> _submit() => _run(() async {
    final marks = _marks.toList()..sort();
    final body = <String, Object?>{
      'clip_id': _clip!.id,
      'verdict': _verdict,
      'note': _note.text.isEmpty ? null : _note.text,
      'marks': [
        for (final i in marks)
          {'start': i, 'end': i + 1, 'verdict': _markVerdict, 'note': null},
      ],
    };
    final item = <String, dynamic>{
      'source': _source!.identity,
      'user': _signedInUser!,
      'session': _session!.id,
      'key': newSubmissionKey(),
      'body': body,
    };
    // Persist before the request, so interruption and ambiguous network failures cannot lose marks.
    await _queue!.enqueue(item);
    final sent = await _queue!.retry(_source!, _signedInUser!);
    final remains = (await _queue!.read()).any((x) => x['key'] == item['key']);
    if (!mounted) return;
    setState(() {
      _pendingKey = remains ? item['key'] as String : null;
      if (!remains) {
        _marks.clear();
        _note.clear();
        _verdict = 'unsure';
        _markVerdict = 'reject';
      }
      _status = remains
          ? (_queue!.lastError?.contains('Sign-in expired') == true
                ? 'Sign-in expired. Sign out and sign in again; marks remain queued.'
                : '${_queue!.lastError ?? 'Upload failed.'} Marks are queued; use Retry pending when ready.')
          : 'Marks uploaded ($sent pending submission(s) sent).';
    });
  });

  Future<void> _retry() => _run(() async {
    final sent = await _queue!.retry(_source!, _signedInUser!);
    final remaining = (await _queue!.read())
        .where(
          (x) => x['source'] == _source!.identity && x['user'] == _signedInUser,
        )
        .length;
    if (mounted) {
      setState(() {
        _pendingKey = remaining == 0 ? null : _pendingKey;
        _status = remaining == 0
            ? 'Pending marks uploaded. Any current draft is still here.'
            : (_queue!.lastError?.contains('Sign-in expired') == true
                  ? 'Sign-in expired. Sign out and sign in again; marks remain queued.'
                  : '${_queue!.lastError ?? 'Upload failed.'} Uploaded $sent; $remaining still queued. Retry when ready.');
      });
    }
  });

  bool get _hasDraft =>
      _clip != null &&
      (_marks.isNotEmpty ||
          _note.text.isNotEmpty ||
          _verdict != 'unsure' ||
          _markVerdict != 'reject');

  Future<bool> _confirmDiscard() async {
    if (!_hasDraft) return true;
    return await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Discard unsent marks?'),
            content: const Text(
              'Your current marks, verdict and note have not been submitted.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text('Keep editing'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(context, true),
                child: const Text('Discard'),
              ),
            ],
          ),
        ) ??
        false;
  }

  Future<void> _leaveClip() async {
    if (await _confirmDiscard() && mounted) {
      setState(() {
        _clip = null;
        _marks.clear();
        _note.clear();
        _verdict = 'unsure';
        _markVerdict = 'reject';
      });
    }
  }

  Future<void> _signOut() async {
    if (!await _confirmDiscard() || !mounted) return;
    setState(() {
      _source = null;
      _sessions = null;
      _session = null;
      _clip = null;
      _signedInUser = null;
      _pendingKey = null;
      _marks.clear();
      _note.clear();
      _verdict = 'unsure';
      _markVerdict = 'reject';
      _status = 'Signed out. Pending marks remain queued.';
    });
  }

  Future<void> _leavePage() async {
    if (_busy || !await _confirmDiscard() || !mounted) return;
    setState(() => _allowPop = true);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) Navigator.of(context).pop();
    });
  }

  Widget _selector(String label, String value, ValueChanged<String?> changed) =>
      DropdownButtonFormField<String>(
        initialValue: value,
        decoration: InputDecoration(labelText: label),
        items: const [
          'accept',
          'reject',
          'unsure',
        ].map((v) => DropdownMenuItem(value: v, child: Text(v))).toList(),
        onChanged: _busy ? null : changed,
      );

  @override
  Widget build(BuildContext context) => PopScope(
    canPop: _allowPop || (!_busy && !_hasDraft),
    onPopInvokedWithResult: (didPop, _) {
      if (!didPop) _leavePage();
    },
    child: Scaffold(
      appBar: AppBar(
        title: const Text('Listening sessions'),
        actions: [
          if (_source != null)
            IconButton(
              tooltip: 'Sign out',
              onPressed: _busy ? null : _signOut,
              icon: const Icon(Icons.logout),
            ),
          if (_source != null)
            IconButton(
              tooltip: 'Retry pending',
              onPressed: _busy ? null : _retry,
              icon: const Icon(Icons.sync),
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
                const Text(
                  'Assigned sessions',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                if (_sessions!.isEmpty) const Text('No sessions assigned.'),
                for (final session in _sessions!)
                  ListTile(
                    title: Text(session.title),
                    onTap: _busy
                        ? null
                        : () => setState(() {
                            _session = session;
                            _clip = null;
                          }),
                  ),
              ] else if (_clip == null) ...[
                TextButton(
                  onPressed: _busy
                      ? null
                      : () => setState(() => _session = null),
                  child: const Text('All sessions'),
                ),
                Text(
                  _session!.title,
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                for (final clip in _session!.clips)
                  ListTile(
                    title: Text(clip.text),
                    onTap: _busy
                        ? null
                        : () => setState(() {
                            _clip = clip;
                            _marks.clear();
                            _note.clear();
                            _verdict = 'unsure';
                          }),
                  ),
              ] else ...[
                TextButton(
                  onPressed: _busy ? null : _leaveClip,
                  child: const Text('All clips'),
                ),
                FilledButton.icon(
                  onPressed: _busy ? null : _play,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Play clip'),
                ),
                const Text('Tap characters to mark them.'),
                Wrap(
                  children: [
                    for (final entry
                        in _clip!.text.runes.toList().asMap().entries)
                      InkWell(
                        key: ValueKey('character-${entry.key}'),
                        onTap: _busy
                            ? null
                            : () => setState(() {
                                if (!_marks.add(entry.key)) {
                                  _marks.remove(entry.key);
                                }
                              }),
                        child: Container(
                          padding: const EdgeInsets.all(3),
                          color: _marks.contains(entry.key)
                              ? Theme.of(context).colorScheme.primaryContainer
                              : null,
                          child: Text(
                            String.fromCharCode(entry.value),
                            style: const TextStyle(fontSize: 22),
                          ),
                        ),
                      ),
                  ],
                ),
                _selector(
                  'Clip verdict',
                  _verdict,
                  (v) => setState(() => _verdict = v!),
                ),
                _selector(
                  'Marked character verdict',
                  _markVerdict,
                  (v) => setState(() => _markVerdict = v!),
                ),
                TextField(
                  controller: _note,
                  onChanged: (_) => setState(() {}),
                  maxLength: 4000,
                  decoration: const InputDecoration(
                    labelText: 'Note (optional)',
                  ),
                ),
                FilledButton(
                  onPressed: _busy || _pendingKey != null ? null : _submit,
                  child: const Text('Submit marks'),
                ),
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
