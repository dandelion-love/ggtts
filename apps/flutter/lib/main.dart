// The ggtts Flutter spike: load the C library, render one line through an options document, play it.
import 'dart:io';
import 'dart:isolate';

import 'package:audioplayers/audioplayers.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';

import 'ggtts.dart';

void main() => runApp(const SpikeApp());

class SpikeApp extends StatelessWidget {
  const SpikeApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'ggtts',
        theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
        home: const SpikePage(),
      );
}

/// What one Speak press returns from the worker isolate.
class _Result {
  _Result({this.wav, this.seconds = 0, this.rate = 0, this.capabilities, this.loadMs = 0, this.renderMs = 0, this.error});
  final String? wav;
  final double seconds;
  final int rate;
  final Map<String, Object?>? capabilities;
  final int loadMs;
  final int renderMs;
  final String? error;
}

/// The load document. The model files come from the engine's own cache (cache_dir): never downloaded here, so a
/// missing model is the error path the spike checks, not a 2 GB surprise.
Map<String, Object?> loadDocument(String cacheDir) => {
      'load': {'offline': true, 'cache_dir': cacheDir, 'threads': Platform.numberOfProcessors.clamp(1, 8)},
    };

/// Load, render one line, write it as WAV. Runs in its own isolate so the UI stays live during the render.
_Result _speak(String text, String cacheDir, String outDir) {
  final sw = Stopwatch()..start();
  try {
    final g = Ggtts(openGgtts());
    final abi = g.abiVersion();
    if (abi < requiredAbi) return _Result(error: 'library ABI $abi is older than the $requiredAbi this app needs');
    final h = g.load(loadDocument(cacheDir));
    final loadMs = sw.elapsedMilliseconds;
    try {
      final caps = g.capabilities(h);
      final pcm = g.render(h, text, {
        'render': {'seed_salt': 'spike'},
      });
      final renderMs = sw.elapsedMilliseconds - loadMs;
      final path = '$outDir/spike.wav';
      File(path).writeAsBytesSync(wavBytes(pcm));
      return _Result(wav: path, seconds: pcm.seconds, rate: pcm.rate, capabilities: caps, loadMs: loadMs, renderMs: renderMs);
    } finally {
      g.free(h);
    }
  } on GgttsException catch (e) {
    return _Result(error: e.toString());
  } catch (e) {
    return _Result(error: 'could not load the ggtts library: $e');
  }
}

/// Top level on purpose: a closure made inside the State would capture it (and its AudioPlayer), which cannot be sent
/// to another isolate.
Future<_Result> _speakInIsolate(String text, String cacheDir, String outDir) =>
    Isolate.run(() => _speak(text, cacheDir, outDir));

class SpikePage extends StatefulWidget {
  const SpikePage({super.key});
  @override
  State<SpikePage> createState() => _SpikePageState();
}

class _SpikePageState extends State<SpikePage> {
  final _text = TextEditingController(text: '今日天氣幾好，我哋出去行下啦。');
  final _player = AudioPlayer();
  String _status = '';
  String? _error;
  bool _busy = false;
  String _abi = '?';

  @override
  void initState() {
    super.initState();
    try {
      _abi = '${Ggtts(openGgtts()).abiVersion()}';
    } catch (e) {
      _error = 'could not load the ggtts library: $e';
    }
    debugPrint('ggtts-spike: abi $_abi${_error == null ? '' : ', $_error'}');
    // --dart-define=GGTTS_AUTOSPEAK=true speaks on launch, so a device check needs no tap (results go to the log).
    if (const bool.fromEnvironment('GGTTS_AUTOSPEAK') && _error == null) Future.microtask(_speakPressed);
  }

  Future<String> _cacheDir() async {
    final env = Platform.environment['GGTTS_CACHE_DIR'];
    if (env != null && env.isNotEmpty) return env;
    if (Platform.isAndroid) {
      // App-specific external storage: adb can place model files there, and the app reads them without a permission.
      final ext = await getExternalStorageDirectory();
      return '${ext!.path}/ggtts-cache';
    }
    return '${Platform.environment['HOME']}/.cache/ggtts';
  }

  Future<void> _speakPressed() async {
    setState(() {
      _busy = true;
      _error = null;
      _status = 'Loading and rendering…';
    });
    final cacheDir = await _cacheDir();
    final outDir = (await getTemporaryDirectory()).path;
    debugPrint('ggtts-spike: speaking, cache $cacheDir');
    final text = _text.text;
    final r = await _speakInIsolate(text, cacheDir, outDir);
    if (!mounted) return;
    if (r.error != null) {
      debugPrint('ggtts-spike: error ${r.error}');
      setState(() {
        _busy = false;
        _error = r.error;
        _status = '';
      });
      return;
    }
    final speakers = (r.capabilities?['speaker_names'] as List?)?.length ?? 0;
    final status = 'Rendered ${r.seconds.toStringAsFixed(2)} s at ${r.rate} Hz: load ${r.loadMs} ms, '
        'render ${r.renderMs} ms, ${r.capabilities?['family']} with $speakers speakers';
    debugPrint('ggtts-spike: $status');
    setState(() {
      _busy = false;
      _status = status;
    });
    await _player.play(DeviceFileSource(r.wav!));
    debugPrint('ggtts-spike: playing ${r.wav}');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('ggtts spike (ABI $_abi)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          TextField(controller: _text, maxLines: 3, decoration: const InputDecoration(labelText: 'Text')),
          const SizedBox(height: 16),
          FilledButton(onPressed: _busy ? null : _speakPressed, child: Text(_busy ? 'Working…' : 'Speak')),
          const SizedBox(height: 16),
          if (_busy) const LinearProgressIndicator(),
          if (_status.isNotEmpty) Text(_status, key: const Key('status')),
          if (_error != null)
            Text(_error!, key: const Key('error'), style: TextStyle(color: Theme.of(context).colorScheme.error)),
        ]),
      ),
    );
  }
}
