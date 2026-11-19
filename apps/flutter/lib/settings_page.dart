import 'dart:async';
import 'dart:io';

import 'package:material_ui/material_ui.dart';

import 'settings_store.dart';
import 'model_picker.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({
    super.key,
    required this.store,
    this.devices = const [],
    this.pickFile = pickModelFile,
    this.pickedFilesAreOwned,
  });
  final SettingsStore store;
  final List<String> devices;
  final Future<String?> Function(String) pickFile;

  /// Test seam: Android's picker returns freshly imported app-owned files.
  final bool? pickedFilesAreOwned;
  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  Map<String, Object?>? _document;
  final _threads = TextEditingController();
  final _cache = TextEditingController();
  final _model = TextEditingController();
  final _codec = TextEditingController();
  final _voice = TextEditingController();
  String? _engine;
  String? _device;
  String? _error;
  bool _picking = false;
  bool _saving = false;
  final Set<String> _uncommittedImports = {};
  bool get _ownsPicks => widget.pickedFilesAreOwned ?? Platform.isAndroid;

  void _discardImport(String path) {
    unawaited(() async {
      try {
        final file = File(path);
        await file.delete();
        await file.parent
            .delete(); // only an empty, newly created import directory
      } catch (_) {
        // The selected file remains private; cleanup is best-effort.
      }
    }());
  }

  void _discardUncommitted() {
    for (final path in _uncommittedImports) {
      _discardImport(path);
    }
    _uncommittedImports.clear();
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final doc = await widget.store.load();
      final load = (doc['load'] as Map?) ?? {};
      if (!mounted) return;
      setState(() {
        _document = doc;
        _device = load['device'] as String?;
        _engine = load['engine'] == 'auto' ? null : load['engine'] as String?;
        _model.text = load['model'] as String? ?? '';
        _codec.text = load['codec'] as String? ?? '';
        _voice.text = load['voice'] as String? ?? '';
        _threads.text = load['threads']?.toString() ?? '';
        _cache.text = load['cache_dir'] as String? ?? '';
      });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  Future<void> _pick(String kind, TextEditingController controller) async {
    if (_picking || _saving) return;
    setState(() {
      _picking = true;
      _error = null;
    });
    try {
      final selected = await widget.pickFile(kind);
      if (selected != null) {
        if (_ownsPicks) _uncommittedImports.add(selected);
        if (mounted) {
          setState(() => controller.text = selected);
        } else if (_uncommittedImports.remove(selected)) {
          _discardImport(selected);
        }
      }
    } catch (_) {
      if (mounted) {
        setState(
          () =>
              _error = 'Could not select or import the file. Please try again.',
        );
      }
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  Widget _fileField(
    String kind,
    String label,
    String key,
    TextEditingController controller,
  ) {
    return Row(
      children: [
        Expanded(
          child: TextField(
            key: Key(key),
            controller: controller,
            decoration: InputDecoration(labelText: label),
          ),
        ),
        IconButton(
          tooltip: 'Choose $kind file',
          onPressed: _picking || _saving ? null : () => _pick(kind, controller),
          icon: const Icon(Icons.folder_open),
        ),
      ],
    );
  }

  Future<void> _save() async {
    if (_saving || _picking) return;
    setState(() => _saving = true);
    try {
      final doc = Map<String, Object?>.from(_document!);
      final load = Map<String, Object?>.from((doc['load'] as Map?) ?? {});
      if (_engine == null || _engine == 'auto') {
        load.remove('engine');
      } else {
        load['engine'] = _engine;
      }
      for (final entry in {
        'model': _model.text,
        'codec': _codec.text,
        'voice': _voice.text,
      }.entries) {
        if (entry.value.trim().isEmpty) {
          load.remove(entry.key);
        } else {
          load[entry.key] = entry.value;
        }
      }
      if (_device == null) {
        load.remove('device');
      } else {
        load['device'] = _device;
      }
      if (_threads.text.trim().isEmpty) {
        load.remove('threads');
      } else {
        final value = int.tryParse(_threads.text.trim());
        if (value == null) {
          throw const FormatException('Threads must be a whole number.');
        }
        load['threads'] = value;
      }
      if (_cache.text.trim().isEmpty) {
        load.remove('cache_dir');
      } else {
        load['cache_dir'] = _cache.text.trim();
      }
      doc['load'] = load;
      final committedPaths = [
        load['model'],
        load['codec'],
        load['voice'],
      ].whereType<String>().toSet();
      await widget.store.save(doc);
      _uncommittedImports.removeAll(committedPaths);
      if (mounted) Navigator.of(context).pop(true);
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      _saving = false;
      if (mounted) {
        setState(() {});
      } else {
        _discardUncommitted();
      }
    }
  }

  @override
  void dispose() {
    if (!_saving) _discardUncommitted();
    _threads.dispose();
    _cache.dispose();
    _model.dispose();
    _codec.dispose();
    _voice.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Settings')),
    body: _document == null
        ? Center(child: Text(_error ?? 'Loading…'))
        : SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  DropdownButtonFormField<String>(
                    key: const Key('engine'),
                    initialValue: _engine,
                    decoration: const InputDecoration(labelText: 'Engine'),
                    items: [
                      const DropdownMenuItem(
                        value: null,
                        child: Text('Automatic'),
                      ),
                      const DropdownMenuItem(
                        value: 'qwen3-tts',
                        child: Text('Qwen3-TTS'),
                      ),
                      const DropdownMenuItem(
                        value: 'kitten-tts-2',
                        child: Text('Kitten-TTS 2'),
                      ),
                      if (_engine == 'fake')
                        const DropdownMenuItem(
                          value: 'fake',
                          child: Text('Fake test engine (saved)'),
                        ),
                    ],
                    onChanged: (value) => setState(() => _engine = value),
                  ),
                  const Text(
                    'Choose a file or enter an app-readable path. Android imports selected files into private storage.',
                  ),
                  _fileField(
                    'model',
                    'Model file path (blank for default)',
                    'model_path',
                    _model,
                  ),
                  _fileField(
                    'codec',
                    'Codec file path (blank for default)',
                    'codec_path',
                    _codec,
                  ),
                  _fileField(
                    'voice',
                    'Prepared voice file path (optional)',
                    'voice_path',
                    _voice,
                  ),
                  if (_picking) const Text('Selecting or importing a file&'),
                  DropdownButtonFormField<String>(
                    initialValue: _device,
                    decoration: const InputDecoration(labelText: 'Device'),
                    items: [
                      const DropdownMenuItem(
                        value: null,
                        child: Text('Default'),
                      ),
                      const DropdownMenuItem(value: 'cpu', child: Text('CPU')),
                      DropdownMenuItem(
                        value: 'cuda',
                        enabled: widget.devices.contains('cuda'),
                        child: const Text('CUDA'),
                      ),
                    ],
                    onChanged: (value) => setState(() => _device = value),
                  ),
                  TextField(
                    controller: _threads,
                    keyboardType: TextInputType.number,
                    decoration: const InputDecoration(
                      labelText: 'Threads (blank for default)',
                    ),
                  ),
                  TextField(
                    controller: _cache,
                    decoration: const InputDecoration(
                      labelText: 'Cache folder (blank for default)',
                    ),
                  ),
                  FilledButton(
                    onPressed: _picking || _saving ? null : _save,
                    child: const Text('Save settings'),
                  ),
                  if (_error != null)
                    Text(
                      _error!,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                      ),
                    ),
                ],
              ),
            ),
          ),
  );
}
