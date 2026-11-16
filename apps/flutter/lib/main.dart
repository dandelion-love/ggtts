// A thin read-aloud screen over the shared ggtts options-document ABI.
import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:material_ui/material_ui.dart';
import 'package:mp_audio_stream/mp_audio_stream.dart';
import 'package:path_provider/path_provider.dart';

import 'ggtts.dart';
import 'read_options.dart';
import 'pcm_queue.dart';
import 'settings_page.dart';
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

/// The native callback is synchronous on this worker; each piece is copied by the binding before sending it.
void renderWorker(Map<String, Object?> request) async {
  final port = request['port'] as SendPort;
  final audio = getAudioStream();
  var audioReady = false;
  int? sampleRate;
  PcmQueue? queue;
  var completed = false;
  PendingRead? pending;
  String? token;
  try {
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
            if (audio.init(
                  sampleRate: piece.rate,
                  channels: 1,
                  bufferMilliSec: 5000,
                  waitingBufferMilliSec: 0,
                ) !=
                0) {
              throw StateError('Could not initialize audio playback.');
            }
            audioReady = true;
            sampleRate = piece.rate;
            queue = PcmQueue(audio.push);
            audio.resume();
          }
          if (sampleRate != piece.rate) {
            throw StateError('Audio sample rate changed during the read.');
          }
          pending!.append(piece);
          queue!.add(piece.samples);
          port.send({'samples': piece.samples.length, 'rate': piece.rate});
          return true;
        },
      );
      // At most five seconds remains in the native ring after the last write.
      if (audioReady) sleep(const Duration(milliseconds: 5100));
    } finally {
      engine.free(handle);
    }
    token = pending.finish();
    completed = true;
  } catch (error) {
    await pending?.discard();
    port.send({'error': error.toString()});
  } finally {
    if (audioReady) audio.uninit();
  }
  if (completed) port.send({'done': true, 'token': token});
}

class ReadPage extends StatefulWidget {
  const ReadPage({
    super.key,
    this.prepare = inspectModel,
    this.loadSettings = loadSavedSettings,
  });
  final Future<Map<String, Object?>> Function(Map<String, Object?>) prepare;
  final Future<Map<String, Object?>> Function() loadSettings;
  @override
  State<ReadPage> createState() => _ReadPageState();
}

class _ReadPageState extends State<ReadPage> {
  final _text = TextEditingController(text: '今日天氣幾好，我哋出去行下啦。');
  final _steering = TextEditingController();
  ReceivePort? _receive;
  Isolate? _worker;
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
      final library = SavedReads(
        Directory('${(await getApplicationSupportDirectory()).path}/reads'),
      );
      if (!mounted) return;
      _library = library;
      final load = composeLoadDocument(_settings, cache, speaker: speaker);
      final render = renderOptions(steering: steering);
      final options = <String, Object?>{
        'load_document': load,
        'render_document': render,
      };
      final port = ReceivePort();
      _receive = port;
      final worker = await Isolate.spawn(
        renderWorker,
        <String, Object?>{
          'port': port.sendPort,
          'load': load,
          'render': render,
          'options': options,
          'library': library.root.path,
          'text': text,
        },
        onError: port.sendPort,
        onExit: port.sendPort,
      );
      if (!mounted) {
        worker.kill(priority: Isolate.immediate);
        port.close();
        return;
      }
      _worker = worker;
      await for (final dynamic message in port) {
        if (!mounted) break;
        if (message is! Map) {
          throw StateError('Read worker exited unexpectedly: $message');
        }
        final event = message;
        if (event['error'] != null) throw StateError(event['error']);
        if (event['done'] == true) {
          _pendingToken = event['token'] as String;
          break;
        }
        debugPrint(
          'ggtts-read: received ${event['samples']} samples at ${event['rate']} Hz',
        );
      }
      if (mounted) {
        setState(() => _status = 'Read complete. Save it or discard it.');
      }
      debugPrint('ggtts-read: generation complete');
    } catch (error) {
      debugPrint('ggtts-read: read failed: $error');
      if (mounted) {
        setState(() {
          _error = error.toString();
          _status = 'Read failed.';
        });
      }
    } finally {
      _receive?.close();
      _receive = null;
      _worker?.kill(priority: Isolate.immediate);
      _worker = null;
      if (mounted) setState(() => _busy = false);
    }
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
    _receive?.close();
    _worker?.kill(priority: Isolate.immediate);
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
