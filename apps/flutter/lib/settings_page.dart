import 'package:material_ui/material_ui.dart';

import 'settings_store.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.store, this.devices = const []});
  final SettingsStore store;
  final List<String> devices;
  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  Map<String, Object?>? _document;
  final _threads = TextEditingController();
  final _cache = TextEditingController();
  String? _device;
  String? _error;

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
        _threads.text = load['threads']?.toString() ?? '';
        _cache.text = load['cache_dir'] as String? ?? '';
      });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  Future<void> _save() async {
    try {
      final doc = Map<String, Object?>.from(_document!);
      final load = Map<String, Object?>.from((doc['load'] as Map?) ?? {});
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
      await widget.store.save(doc);
      if (mounted) Navigator.of(context).pop(true);
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  @override
  void dispose() {
    _threads.dispose();
    _cache.dispose();
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
                    onPressed: _save,
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
