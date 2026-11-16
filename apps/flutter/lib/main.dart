// A thin read-aloud screen over the shared ggtts options-document ABI.
import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:material_ui/material_ui.dart';
import 'package:mp_audio_stream/ggtts_audio_session.dart';
import 'package:path_provider/path_provider.dart';

import 'ggtts.dart';
import 'audio_drain.dart';
import 'read_options.dart';
import 'read_control_gate.dart';
import 'read_worker_outcome.dart';
import 'pcm_queue.dart';
import 'settings_page.dart';
import 'settings_store.dart';
import 'saved_reads.dart';
import 'saved_reads_page.dart';

void main() => runApp(const ReadApp());

class ReadApp extends StatelessWidget {
  const ReadApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'ggtts',
    theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
    home: const ReadPage(),
  );
}

/// Kept as the offline default for callers and tests; downloads require an explicit preparation action.
Map<String, Object?> loadDocument(String cacheDir) => loadOptions(cacheDir);

Future<SettingsStore> settingsStore() async => SettingsStore(
  File('${(await getApplicationSupportDirectory()).path}/options.json'),
);

Future<Map<String, Object?>> loadSavedSettings() async =>
    (await settingsStore()).load();

Future<Map<String, Object?>> inspectModel(Map<String, Object?> document) =>
    Isolate.run(() {
      final engine = Ggtts(openGgtts());
      final schema = engine.optionsSchema();
      final handle = engine.load(document);
      try {
        return {'schema': schema, 'capabilities': engine.capabilities(handle)};
      } finally {
        engine.free(handle);
      }
    });

/// Controls are executed outside the UI and rendering isolates. The token and
/// reservation are unique to this read, so late messages cannot affect another.
void controlWorker(SendPort parent) {
  final commands = ReceivePort();
  AudioSession? audio;
  CancellationApi? cancellation;
  int? token;
  try {
    cancellation = CancellationApi(openGgtts());
    token = cancellation.create();
    audio = AudioSession.reserve();
    if (audio == null) throw StateError('Audio output is already in use.');
    parent.send({'port': commands.sendPort, 'audio': audio.id, 'token': token});
  } catch (error) {
    if (audio != null) audio.release();
    if (token != null) cancellation?.free(token);
    parent.send({'error': error.toString()});
    commands.close();
    return;
  }
  final session = audio;
  final cancel = cancellation;
  final cancelToken = token;
  commands.listen((dynamic message) {
    final command = message as Map;
    final reply = command['reply'] as SendPort;
    try {
      switch (command['op']) {
        case 'pause':
          if (session.pause() != 0) throw StateError('Could not pause audio.');
        case 'resume':
          if (session.resume() != 0) {
            throw StateError('Could not resume audio.');
          }
        case 'stop':
          Object? cancelError;
          try {
            cancel.signal(cancelToken);
          } catch (error) {
            cancelError = error;
          }
          final stopResult = session.stop();
          if (cancelError != null) throw cancelError;
          if (stopResult != 0 && session.state() != -1) {
            throw StateError('Could not stop audio.');
          }
        case 'close':
          Object? cleanupError;
          try {
            if (session.state() != -1 && session.release() != 0) {
              throw StateError('Audio cleanup failed.');
            }
          } catch (error) {
            cleanupError = error;
          }
          try {
            cancel.free(cancelToken);
          } catch (error) {
            cleanupError ??= error;
          }
          reply.send(cleanupError?.toString());
          commands.close();
          return;
        default:
          throw StateError('Unknown audio command.');
      }
      reply.send(null);
    } catch (error) {
      reply.send(error.toString());
    }
  });
}

Future<String?> controlCommand(SendPort port, String op) async {
  final reply = ReceivePort();
  try {
    port.send({'op': op, 'reply': reply.sendPort});
    return await reply.first as String?;
  } finally {
    reply.close();
  }
}

/// Native callbacks remain synchronous on the render isolate.
void renderWorker(Map<String, Object?> request) async {
  final port = request['port'] as SendPort;
  final audio = AudioSession.attach(request['audio'] as int);
  var audioReady = false;
  int? sampleRate;
  PcmQueue? queue;
  PendingRead? pending;
  await runReadOutcome(
    generate: () async {
      pending = await SavedReads(Directory(request['library'] as String)).begin(
        text: request['text'] as String,
        options: Map<String, Object?>.from(request['options'] as Map),
      );
      final engine = Ggtts(openGgtts());
      final handle = engine.load(
        Map<String, Object?>.from(request['load'] as Map),
      );
      try {
        engine.renderStream(
          handle,
          request['text'] as String,
          Map<String, Object?>.from(request['render'] as Map),
          (piece) {
            if (!audioReady) {
              if (audio.init(sampleRate: piece.rate) != 0) {
                if (audio.state() == 3) throw ReadStopped();
                throw StateError('Could not initialize audio playback.');
              }
              audioReady = true;
              sampleRate = piece.rate;
              queue = PcmQueue(audio.push);
              port.send({'audioReady': true});
            }
            if (sampleRate != piece.rate) {
              throw StateError('Audio sample rate changed during the read.');
            }
            pending!.append(piece);
            queue!.add(piece.samples);
            return true;
          },
          cancellationToken: request['token'] as int,
        );
        if (audioReady) {
          drainAudio(state: audio.state, softwareEmpty: audio.softwareEmpty);
        }
        if (audio.state() == 3) throw ReadStopped();
      } finally {
        engine.free(handle);
      }
    },
    cleanup: () {
      if (audio.release() != 0) throw StateError('Audio cleanup failed.');
    },
    finish: () => pending!.finish(),
    discard: () async {
      await pending?.discard();
    },
    send: port.send,
    isStopped: (error) =>
        error is ReadStopped || (error is GgttsException && error.code == -5),
  );
}

class ReadPage extends StatefulWidget {
  const ReadPage({
    super.key,
    this.prepare = inspectModel,
    this.loadSettings = loadSavedSettings,
    this.controlEntry = controlWorker,
    this.renderEntry = renderWorker,
    this.readDirectory,
  });
  final Future<Map<String, Object?>> Function(Map<String, Object?>) prepare;
  final Future<Map<String, Object?>> Function() loadSettings;
  final void Function(SendPort) controlEntry;
  final void Function(Map<String, Object?>) renderEntry;
  final Future<Directory> Function()? readDirectory;
  @override
  State<ReadPage> createState() => _ReadPageState();
}

class _ReadPageState extends State<ReadPage> {
  final _text = TextEditingController(text: '今日天氣幾好，我哋出去行下啦。');
  final _steering = TextEditingController();
  SendPort? _control;
  ReadControlGate _gate = ReadControlGate();
  bool _audioReady = false;
  bool _paused = false;
  bool _controlPending = false;
  bool _busy = false;
  String _status = 'Prepare the model to get started.';
  String? _error;
  String? _speaker;
  Map<String, Object?>? _schema;
  Map<String, Object?>? _caps;
  Map<String, Object?> _settings = {};
  SavedReads? _library;
  String? _pendingToken;

  @override
  void initState() {
    super.initState();
    if (const bool.fromEnvironment('GGTTS_AUTOSPEAK')) {
      Future.microtask(() async {
        await _prepare();
        if (mounted && _caps != null) await _speak();
        if (const bool.fromEnvironment('GGTTS_SMOKE_SAVE') &&
            mounted &&
            _pendingToken != null) {
          await _saveRead();
          debugPrint('ggtts-read: saved smoke read');
        }
        if (const bool.fromEnvironment('GGTTS_SMOKE_REPEAT')) {
          await Future<void>.delayed(const Duration(seconds: 5));
          if (mounted && _caps != null) await _speak();
        }
      });
    }
  }

  Future<String> _cacheDir() async {
    final override = Platform.environment['GGTTS_CACHE_DIR'];
    if (override != null && override.isNotEmpty) return override;
    if (Platform.isAndroid) {
      return '${(await getApplicationDocumentsDirectory()).path}/ggtts-cache';
    }
    if (Platform.isWindows) {
      return '${Platform.environment['LOCALAPPDATA']}/ggtts/cache';
    }
    return '${Platform.environment['HOME']}/.cache/ggtts';
  }

  Future<void> _prepare() async {
    setState(() {
      _busy = true;
      _error = null;
      _caps = null;
      _schema = null;
      _status = 'Preparing model. Download may take several minutes…';
    });
    try {
      final settings = await widget.loadSettings();
      final document = composeLoadDocument(
        settings,
        await _cacheDir(),
        offline: false,
      );
      final result = await widget.prepare(document);
      if (!mounted) return;
      final caps = Map<String, Object?>.from(result['capabilities'] as Map);
      final schema = Map<String, Object?>.from(result['schema'] as Map);
      final names =
          (caps['speaker_names'] as List?)?.whereType<String>().toList() ??
          <String>[];
      setState(() {
        _caps = caps;
        _schema = schema;
        _settings = settings;
        _speaker = names.isEmpty ? null : names.first;
        _status = 'Model ready.';
        debugPrint('ggtts-read: model ready');
      });
    } catch (error) {
      debugPrint('ggtts-read: model preparation failed: $error');
      if (mounted) {
        setState(() {
          _error = error.toString();
          _status = 'Model preparation failed. Retry to try again.';
        });
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _speak() async {
    setState(() {
      _busy = true;
      _error = null;
      _status = 'Loading and reading…';
    });
    final text = _text.text;
    final steering = _steering.text;
    final speaker = _speaker;
    try {
      final cache = await _cacheDir();
      if (!mounted) return;
      final readRoot = widget.readDirectory == null
          ? Directory('${(await getApplicationSupportDirectory()).path}/reads')
          : await widget.readDirectory!();
      final library = SavedReads(readRoot);
      if (!mounted) return;
      _library = library;
      final load = composeLoadDocument(_settings, cache, speaker: speaker);
      final render = renderOptions(steering: steering);
      final options = <String, Object?>{
        'load_document': load,
        'render_document': render,
      };
      _gate = ReadControlGate();
      _audioReady = false;
      _paused = false;
      final controlEvents = ReceivePort();
      await Isolate.spawn(
        widget.controlEntry,
        controlEvents.sendPort,
        onError: controlEvents.sendPort,
        onExit: controlEvents.sendPort,
      );
      final dynamic startup = await controlEvents.first;
      if (startup is! Map) throw StateError('Control worker exited: $startup');
      final initial = startup;
      controlEvents.close();
      if (initial['error'] != null) throw StateError(initial['error']);
      final control = initial['port'] as SendPort;
      _control = control;
      if (mounted) setState(() {});
      final port = ReceivePort();
      try {
        await Isolate.spawn(
          widget.renderEntry,
          <String, Object?>{
            'port': port.sendPort,
            'load': load,
            'render': render,
            'options': options,
            'library': library.root.path,
            'text': text,
            'audio': initial['audio'],
            'token': initial['token'],
          },
          onError: port.sendPort,
          onExit: port.sendPort,
        );
        if (_gate.stopRequested || !mounted) {
          _gate.requestStop();
          await controlCommand(control, 'stop');
        }
        var workerStopped = false;
        await for (final dynamic message in port) {
          if (message is! Map) {
            throw StateError('Read worker exited unexpectedly: $message');
          }
          if (message['audioReady'] == true) {
            if (mounted) setState(() => _audioReady = true);
            continue;
          }
          if (message['stopped'] == true) {
            workerStopped = true;
            _gate.finish();
            break;
          }
          if (message['error'] != null) {
            _gate.finish();
            throw StateError(message['error']);
          }
          if (message['done'] == true) {
            _gate.finish();
            final token = message['token'] as String;
            if (_gate.stopRequested || !mounted) {
              await library.discardPending(token);
            } else {
              _pendingToken = token;
            }
            break;
          }
        }
        if (mounted) {
          setState(
            () => _status = _gate.stopRequested || workerStopped
                ? 'Read stopped.'
                : 'Audio submitted. Save the read or discard it.',
          );
        }
      } finally {
        _gate.finish();
        port.close();
        _control = null;
        if (mounted) setState(() {});
        final closeError = await controlCommand(control, 'close');
        if ((!mounted || _gate.stopRequested) && _pendingToken != null) {
          final token = _pendingToken!;
          _pendingToken = null;
          await library.discardPending(token);
        }
        if (closeError != null) {
          final token = _pendingToken;
          _pendingToken = null;
          if (token != null) await library.discardPending(token);
          throw StateError(closeError);
        }
      }
    } catch (error) {
      if (mounted) {
        setState(() {
          _error = error.toString();
          _status = 'Read failed.';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _audioReady = false;
          _controlPending = false;
        });
      }
    }
  }

  void _stopRead({bool updateUi = true}) {
    if (!_gate.requestStop()) return;
    if (mounted && updateUi) setState(() => _status = 'Stopping...');
    final control = _control;
    if (control != null) {
      controlCommand(control, 'stop').then((error) {
        if (error != null && mounted) setState(() => _error = error);
      });
    }
  }

  Future<void> _togglePause() async {
    final control = _control;
    if (control == null ||
        _controlPending ||
        _gate.stopRequested ||
        _gate.terminal) {
      return;
    }
    setState(() => _controlPending = true);
    final error = await controlCommand(control, _paused ? 'resume' : 'pause');
    if (!mounted) return;
    setState(() {
      _controlPending = false;
      if (error != null) {
        _error = error;
      } else if (!_gate.stopRequested) {
        _paused = !_paused;
        _status = _paused ? 'Paused.' : 'Reading...';
      }
    });
  }

  Future<void> _saveRead() async {
    if (_busy || _pendingToken == null) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await _library!.publish(_pendingToken!);
      if (mounted) {
        setState(() {
          _pendingToken = null;
          _status = 'Read saved.';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _discardRead() async {
    if (_busy || _pendingToken == null) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await _library!.discardPending(_pendingToken!);
      if (mounted) {
        setState(() {
          _pendingToken = null;
          _status = 'Read discarded.';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _openReads() async {
    try {
      final library = SavedReads(
        Directory('${(await getApplicationSupportDirectory()).path}/reads'),
      );
      if (mounted) {
        await Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => SavedReadsPage(library: library)),
        );
      }
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  Future<void> _openSettings() async {
    try {
      final store = await settingsStore();
      if (!mounted) return;
      final changed = await Navigator.of(context).push<bool>(
        MaterialPageRoute(
          builder: (_) => SettingsPage(
            store: store,
            devices:
                (_caps?['devices'] as List?)?.whereType<String>().toList() ??
                [],
          ),
        ),
      );
      if (changed == true && mounted) {
        setState(() {
          _caps = null;
          _schema = null;
          _status = 'Prepare the model with the new settings.';
        });
      }
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  @override
  void dispose() {
    _stopRead(updateUi: false);
    _text.dispose();
    _steering.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final names =
        (_caps?['speaker_names'] as List?)?.whereType<String>().toList() ??
        <String>[];
    final speakerAvailable =
        _schema != null &&
        hasSchemaOption(_schema!, 'load', 'speaker') &&
        names.isNotEmpty;
    final steeringAvailable =
        _schema != null &&
        hasSchemaOption(_schema!, 'render', 'steer') &&
        _caps?['steering'] == true;
    return Scaffold(
      appBar: AppBar(
        title: const Text('ggtts'),
        actions: [
          IconButton(
            tooltip: 'Saved reads',
            onPressed: _busy || _pendingToken != null ? null : _openReads,
            icon: const Icon(Icons.library_music),
          ),
          IconButton(
            tooltip: 'Licences',
            onPressed: () => Navigator.of(context)
                .push(MaterialPageRoute(builder: (_) => const LicencePage())),
            icon: const Icon(Icons.info_outline),
          ),
          IconButton(
            tooltip: 'Settings',
            onPressed: _busy ? null : _openSettings,
            icon: const Icon(Icons.settings),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _text,
                maxLines: 5,
                onChanged: (_) => setState(() {}),
                decoration: const InputDecoration(labelText: 'Text'),
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                key: const Key('voice'),
                initialValue: speakerAvailable ? _speaker : null,
                decoration: const InputDecoration(labelText: 'Voice'),
                items: names
                    .map(
                      (name) =>
                          DropdownMenuItem(value: name, child: Text(name)),
                    )
                    .toList(),
                onChanged: !_busy && speakerAvailable
                    ? (value) => setState(() => _speaker = value)
                    : null,
              ),
              TextField(
                key: const Key('steering'),
                controller: _steering,
                enabled: !_busy && steeringAvailable,
                decoration: const InputDecoration(
                  labelText: 'Steering file or URL',
                  helperText: 'Optional; local file or cached URL',
                ),
              ),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: _busy ? null : _prepare,
                child: Text(
                  _caps == null
                      ? 'Prepare model (download if needed)'
                      : 'Prepare model again',
                ),
              ),
              FilledButton(
                onPressed:
                    !_busy &&
                        _pendingToken == null &&
                        _caps != null &&
                        _text.text.isNotEmpty
                    ? _speak
                    : null,
                child: const Text('Speak'),
              ),
              if (_busy && _control != null) ...[
                TextButton(
                  onPressed: _gate.stopRequested || _gate.terminal
                      ? null
                      : () => _stopRead(),
                  child: const Text('Stop'),
                ),
                TextButton(
                  onPressed:
                      _audioReady &&
                          !_controlPending &&
                          !_gate.stopRequested &&
                          !_gate.terminal
                      ? _togglePause
                      : null,
                  child: Text(_paused ? 'Resume' : 'Pause'),
                ),
              ],
              if (_pendingToken != null) ...[
                FilledButton(
                  onPressed: _busy ? null : _saveRead,
                  child: const Text('Save read'),
                ),
                TextButton(
                  onPressed: _busy ? null : _discardRead,
                  child: const Text('Discard read'),
                ),
              ],
              if (_busy) const LinearProgressIndicator(),
              Text(_status, key: const Key('status')),
              if (_error != null)
                Text(
                  _error!,
                  key: const Key('error'),
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class LicencePage extends StatelessWidget {
  const LicencePage({super.key});
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Licences')),
    body: const Padding(
      padding: EdgeInsets.all(16),
      child: Text(
        'Model and voice licence details are not available from this engine. Consult the source of each model or voice.',
      ),
    ),
  );
}
