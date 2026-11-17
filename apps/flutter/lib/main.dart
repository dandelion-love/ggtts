// A thin read-aloud screen over the shared ggtts options-document ABI.
import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:material_ui/material_ui.dart';

import 'playback_session.dart';

import 'package:path_provider/path_provider.dart';

import 'ggtts.dart';
import 'audio_drain.dart';
import 'read_options.dart';
import 'read_control_gate.dart';
import 'read_worker_outcome.dart';
import 'read_timings.dart';
import 'pcm_queue.dart';
import 'settings_page.dart';
import 'shared_text.dart';
import 'settings_store.dart';
import 'saved_reads.dart';
import 'saved_reads_page.dart';
import 'listening_page.dart';

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

void _inspectModelWorker(Map<String, Object?> request) {
  final port = request['port'] as SendPort;
  try {
    final engine = Ggtts(openGgtts());
    final schema = engine.optionsSchema();
    final handle = engine.loadWithProgress(
      Map<String, Object?>.from(request['document'] as Map),
      (notice) => port.send({'notice': notice}),
      cancellationToken: request['token'] as int?,
    );
    Map<String, Object?> caps;
    try {
      caps = engine.capabilities(handle);
    } finally {
      engine.free(handle);
    }
    port.send({
      'result': {'schema': schema, 'capabilities': caps},
    });
  } catch (error) {
    port.send({'error': error.toString()});
  }
}

Future<Map<String, Object?>> inspectModel(
  Map<String, Object?> document, [
  void Function(String)? onProgress,
  int? token,
]) async {
  final port = ReceivePort();
  try {
    await Isolate.spawn(
      _inspectModelWorker,
      {'port': port.sendPort, 'document': document, 'token': token},
      onError: port.sendPort,
      onExit: port.sendPort,
    );
    await for (final dynamic message in port) {
      if (message is! Map) {
        throw StateError('Model preparation worker exited unexpectedly.');
      }
      if (message['notice'] is String) {
        onProgress?.call(message['notice'] as String);
      } else if (message['error'] != null) {
        throw StateError(message['error']);
      } else if (message['result'] is Map) {
        return Map<String, Object?>.from(message['result'] as Map);
      }
    }
    throw StateError('Model preparation worker exited unexpectedly.');
  } finally {
    port.close();
  }
}

/// Controls are executed outside the UI and rendering isolates. The token and
/// reservation are unique to this read, so late messages cannot affect another.
void controlWorker(SendPort parent) => _controlWorker(parent, true);
void controlWorkerWithoutCancellation(SendPort parent) =>
    _controlWorker(parent, false);

void _controlWorker(SendPort parent, bool cancellable) {
  final commands = ReceivePort();
  PlaybackSession? audio;
  CancellationApi? cancellation;
  int? token;
  try {
    if (cancellable) {
      cancellation = CancellationApi(openGgtts());
      token = cancellation.create();
    }
    audio = PlaybackSession.reserve();
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
            if (cancel == null || cancelToken == null) {
              throw UnsupportedError(
                'Cancellation is unavailable for this engine.',
              );
            }
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
            if (cancel != null && cancelToken != null) cancel.free(cancelToken);
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
  final audio = PlaybackSession.attach(request['audio'] as int);
  var audioReady = false;
  int? sampleRate;
  PcmQueue? queue;
  PendingRead? pending;
  final timings = ReadTimings();
  await runReadOutcome(
    generate: () async {
      pending = await SavedReads(Directory(request['library'] as String)).begin(
        text: request['text'] as String,
        options: Map<String, Object?>.from(request['options'] as Map),
      );
      final engine = Ggtts(openGgtts());
      final handle = timings.load(
        () => engine.load(Map<String, Object?>.from(request['load'] as Map)),
      );
      try {
        final cancellable = request['cancellable'] == true;
        if (cancellable && engine.capabilities(handle)['cancellable'] != true) {
          throw UnsupportedError(
            'Loaded engine does not support cancellation.',
          );
        }
        timings.beginStream();
        try {
          engine.renderStream(
            handle,
            request['text'] as String,
            Map<String, Object?>.from(request['render'] as Map),
            (piece) {
              timings.piece(
                samples: piece.samples.length,
                rate: piece.rate,
                setup: () {
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
                    throw StateError(
                      'Audio sample rate changed during the read.',
                    );
                  }
                },
                save: () => pending!.append(piece),
                handoff: () => queue!.add(piece.samples),
              );
              return true;
            },
            cancellationToken: cancellable ? request['token'] as int : null,
          );
        } finally {
          timings.streamEnded();
        }
        if (audioReady) {
          timings.drain(() {
            if (audio.usesPresentationTracking) {
              drainPresentation(
                state: audio.state,
                submitted: audio.submitted,
                head: audio.head,
              );
            } else {
              drainAudio(
                state: audio.state,
                softwareEmpty: audio.softwareEmpty,
              );
            }
          });
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

abstract interface class PrepareCancellation {
  int get token;
  void signal();
  void close();
}

class _NativePrepareCancellation implements PrepareCancellation {
  _NativePrepareCancellation() : _api = CancellationApi(openGgtts()) {
    token = _api.create();
  }
  final CancellationApi _api;
  @override
  late final int token;
  @override
  void signal() => _api.signal(token);
  @override
  void close() => _api.free(token);
}

class ReadPage extends StatefulWidget {
  const ReadPage({
    super.key,
    this.prepare,
    this.prepareWithProgress,
    this.createPrepareCancellation,
    this.loadSettings = loadSavedSettings,
    this.controlEntry,
    this.renderEntry = renderWorker,
    this.readDirectory,
    this.sharedTextInbox,
  });
  final Future<Map<String, Object?>> Function(Map<String, Object?>)? prepare;
  final Future<Map<String, Object?>> Function(
    Map<String, Object?>,
    void Function(String),
  )?
  prepareWithProgress;
  final PrepareCancellation Function()? createPrepareCancellation;
  final Future<Map<String, Object?>> Function() loadSettings;
  final void Function(SendPort)? controlEntry;
  final void Function(Map<String, Object?>) renderEntry;
  final Future<Directory> Function()? readDirectory;
  final SharedTextInbox? sharedTextInbox;
  @override
  State<ReadPage> createState() => _ReadPageState();
}

class _ReadPageState extends State<ReadPage> {
  final _text = TextEditingController(text: '今日天氣幾好，我哋出去行下啦。');
  final _steering = TextEditingController();
  SendPort? _control;
  ReadControlGate _gate = ReadControlGate();
  bool _audioReady = false;
  bool _cancellable = false;
  bool _paused = false;
  bool _controlPending = false;
  bool _busy = false;
  bool _preparing = false;
  bool _prepareStopped = false;
  PrepareCancellation? _prepareCancellation;
  String _status = 'Prepare the model to get started.';
  String? _error;
  String? _speaker;
  Map<String, Object?>? _schema;
  Map<String, Object?>? _caps;
  Map<String, Object?> _settings = {};
  SavedReads? _library;
  String? _pendingToken;
  SharedTextInbox? _sharedInbox;
  String? _sharedPending;
  bool _sharedPolling = false;
  bool _sharedRepoll = false;

  @override
  void initState() {
    super.initState();
    _sharedInbox = widget.sharedTextInbox ?? platformSharedTextInbox();
    _sharedInbox?.onAvailable(_pollSharedText);
    _pollSharedText();
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

  Future<void> _pollSharedText() async {
    if (_sharedInbox == null || !mounted || _sharedPending != null) return;
    if (_sharedPolling) {
      _sharedRepoll = true;
      return;
    }
    _sharedPolling = true;
    try {
      final text = await _sharedInbox!.take();
      if (mounted && text != null) setState(() => _sharedPending = text);
    } catch (_) {
      if (mounted) setState(() => _error = 'Could not receive shared text.');
    } finally {
      _sharedPolling = false;
      if (_sharedRepoll) {
        _sharedRepoll = false;
        if (mounted && _sharedPending == null) _pollSharedText();
      }
    }
  }

  void _finishSharedText({required bool use}) {
    final text = _sharedPending;
    if (_busy || text == null) return;
    setState(() {
      if (use) {
        _text.value = TextEditingValue(
          text: text,
          selection: TextSelection.collapsed(offset: text.length),
        );
      }
      _sharedPending = null;
    });
    _pollSharedText();
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

  Future<void> _prepare({bool offline = false}) async {
    setState(() {
      _busy = true;
      _preparing = true;
      _prepareStopped = false;
      _error = null;
      _caps = null;
      _schema = null;
      _status = offline
          ? 'Preparing model from selected and cached files...'
          : 'Preparing model. Download may take several minutes...';
    });
    try {
      final settings = await widget.loadSettings();
      final document = composeLoadDocument(
        settings,
        await _cacheDir(),
        offline: offline,
      );
      void onProgress(String progress) {
        // Notices may contain URLs and private cache paths. Only display the
        // numeric progress generated by the pinned-model downloader.
        final match = RegExp(
          r'^[^:]*: ([0-9.]+ of [0-9.]+ MB \([0-9]+%\), [0-9.]+ MB/s)$',
        ).firstMatch(progress);
        if (mounted && !_prepareStopped && match != null) {
          setState(() => _status = 'Downloading: ${match.group(1)}');
        }
      }

      if (_prepareStopped || !mounted) {
        if (mounted) setState(() => _status = 'Model preparation stopped.');
        return;
      }
      late final Map<String, Object?> result;
      if (widget.prepare != null) {
        result = await widget.prepare!(document);
      } else if (widget.prepareWithProgress != null) {
        if (widget.createPrepareCancellation != null) {
          _prepareCancellation = widget.createPrepareCancellation!();
        }
        result = await widget.prepareWithProgress!(document, onProgress);
      } else {
        _prepareCancellation =
            (widget.createPrepareCancellation ??
            _NativePrepareCancellation.new)();
        result = await inspectModel(
          document,
          onProgress,
          _prepareCancellation!.token,
        );
      }
      if (!mounted) return;
      if (_prepareStopped) {
        setState(() => _status = 'Model preparation stopped.');
        return;
      }
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
      if (mounted) {
        setState(() {
          if (_prepareStopped) {
            _status = 'Model preparation stopped.';
          } else {
            _error = error.toString();
            _status = offline
                ? 'Offline preparation failed. Check selected files or prepare with download.'
                : 'Model preparation failed. Retry to try again.';
          }
        });
      }
    } finally {
      try {
        _prepareCancellation?.close();
      } catch (error) {
        if (mounted) {
          setState(() {
            _caps = null;
            _schema = null;
            _error = error.toString();
            _status = 'Model preparation cleanup failed.';
          });
        }
      }
      _prepareCancellation = null;
      _preparing = false;
      if (mounted) setState(() => _busy = false);
    }
  }

  void _stopPreparation({bool updateUi = true}) {
    if (!_preparing || _prepareStopped) return;
    try {
      _prepareCancellation?.signal();
      _prepareStopped = true;
      if (mounted && updateUi) {
        setState(() => _status = 'Stopping model preparation...');
      }
    } catch (error) {
      if (mounted && updateUi) {
        setState(() => _error = 'Could not request stop: $error');
      }
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
      _cancellable = _caps?['cancellable'] == true;
      _audioReady = false;
      _paused = false;
      final controlEvents = ReceivePort();
      await Isolate.spawn(
        widget.controlEntry ??
            (_cancellable ? controlWorker : controlWorkerWithoutCancellation),
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
            'cancellable': _cancellable,
          },
          onError: port.sendPort,
          onExit: port.sendPort,
        );
        if (_cancellable && (_gate.stopRequested || !mounted)) {
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
    if (!_cancellable || !_gate.requestStop()) return;
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
    _sharedInbox?.close();
    _stopPreparation(updateUi: false);
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
            tooltip: 'Listening sessions',
            onPressed: _busy
                ? null
                : () => Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const ListeningPage()),
                  ),
            icon: const Icon(Icons.headphones),
          ),
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
              if (_sharedPending != null) ...[
                Text(
                  'Shared text: ${String.fromCharCodes(_sharedPending!.runes.take(120))}${_sharedPending!.runes.length > 120 ? "..." : ""}',
                  key: const Key('shared_text_preview'),
                ),
                Row(
                  children: [
                    TextButton(
                      onPressed: _busy
                          ? null
                          : () => _finishSharedText(use: true),
                      child: const Text('Use shared text'),
                    ),
                    TextButton(
                      onPressed: _busy
                          ? null
                          : () => _finishSharedText(use: false),
                      child: const Text('Discard shared text'),
                    ),
                  ],
                ),
              ],
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
                      : 'Prepare model again (download if needed)',
                ),
              ),
              OutlinedButton(
                onPressed: _busy ? null : () => _prepare(offline: true),
                child: const Text('Prepare offline (selected or cached files)'),
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
              if (_preparing)
                TextButton(
                  onPressed: _prepareStopped ? null : _stopPreparation,
                  child: const Text('Stop preparation'),
                ),
              if (_busy && _control != null) ...[
                if (!_cancellable)
                  const Text('Stop is unavailable for this engine.'),
                if (_cancellable)
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
