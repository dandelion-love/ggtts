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

Future<Map<String, Object?>> inspectModel(String cacheDir) => Isolate.run(() {
  final engine = Ggtts(openGgtts());
  final schema = engine.optionsSchema();
  final handle = engine.load(loadOptions(cacheDir, offline: false));
  try {
    return {'schema': schema, 'capabilities': engine.capabilities(handle)};
  } finally {
    engine.free(handle);
  }
});

/// The native callback is synchronous on this worker; each piece is copied by the binding before sending it.
void renderWorker(Map<String, Object?> request) {
  final port = request['port'] as SendPort;
  final audio = getAudioStream();
  var audioReady = false;
  int? sampleRate;
  PcmQueue? queue;
  var completed = false;
  try {
    final engine = Ggtts(openGgtts());
    final handle = engine.load(
      loadOptions(
        request['cache'] as String,
        speaker: request['speaker'] as String?,
      ),
    );
    try {
      engine.renderStream(
        handle,
        request['text'] as String,
        renderOptions(steering: request['steering'] as String?),
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
    completed = true;
  } catch (error) {
    port.send({'error': error.toString()});
  } finally {
    if (audioReady) audio.uninit();
  }
  if (completed) port.send({'done': true});
}

class ReadPage extends StatefulWidget {
  const ReadPage({super.key, this.prepare = inspectModel});
  final Future<Map<String, Object?>> Function(String) prepare;
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

  @override
  void initState() {
    super.initState();
    if (const bool.fromEnvironment('GGTTS_AUTOSPEAK')) {
      Future.microtask(() async {
        await _prepare();
        if (mounted && _caps != null) await _speak();
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
      final result = await widget.prepare(await _cacheDir());
      if (!mounted) return;
      final caps = Map<String, Object?>.from(result['capabilities'] as Map);
      final schema = Map<String, Object?>.from(result['schema'] as Map);
      final names =
          (caps['speaker_names'] as List?)?.whereType<String>().toList() ??
          <String>[];
      setState(() {
        _caps = caps;
        _schema = schema;
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
      final port = ReceivePort();
      _receive = port;
      final worker = await Isolate.spawn(
        renderWorker,
        <String, Object?>{
          'port': port.sendPort,
          'cache': cache,
          'speaker': speaker,
          'text': text,
          'steering': steering,
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
        if (event['done'] == true) break;
        debugPrint(
          'ggtts-read: received ${event['samples']} samples at ${event['rate']} Hz',
        );
      }
      if (mounted) setState(() => _status = 'Read generated.');
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
      appBar: AppBar(title: const Text('ggtts')),
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
                onPressed: !_busy && _caps != null && _text.text.isNotEmpty
                    ? _speak
                    : null,
                child: const Text('Speak'),
              ),
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
