import 'dart:io';

import 'package:audioplayers/audioplayers.dart';
import 'package:material_ui/material_ui.dart';

import 'saved_reads.dart';

class SavedReadsPage extends StatefulWidget {
  const SavedReadsPage({super.key, required this.library});
  final SavedReads library;
  @override
  State<SavedReadsPage> createState() => _SavedReadsPageState();
}

class _SavedReadsPageState extends State<SavedReadsPage> {
  AudioPlayer? _player;
  late Future<(List<SavedRead>, List<PendingReadInfo>)> _contents;
  String? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  void _refresh() {
    _contents = (() async => (
      await widget.library.list(),
      await widget.library.completedPending(),
    ))();
  }

  Future<void> _action(Future<void> Function() action) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await action();
      if (mounted) setState(_refresh);
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _play(SavedRead read) async {
    await _action(() async {
      if (!await File(read.audioPath).exists()) return;
      await (_player ??= AudioPlayer()).play(DeviceFileSource(read.audioPath));
    });
  }

  @override
  void dispose() {
    _player?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Saved reads')),
    body: FutureBuilder<(List<SavedRead>, List<PendingReadInfo>)>(
      future: _contents,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Center(child: Text('Could not load reads: ${snapshot.error}'));
        }
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }
        final (reads, pending) = snapshot.data!;
        return Column(
          children: [
            if (_error != null)
              Text(
                _error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            if (reads.isEmpty && pending.isEmpty)
              const Text('No saved reads yet.')
            else
              Expanded(
                child: ListView(
                  children: [
                    if (pending.isNotEmpty) ...[
                      const ListTile(title: Text('Completed reads to recover')),
                      for (final read in pending)
                        ListTile(
                          key: ValueKey('pending-${read.token}'),
                          title: Text(
                            read.invalid
                                ? 'Unreadable pending read'
                                : read.text,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: read.invalid
                              ? Text('Cannot recover: ${read.problem}')
                              : null,
                          leading: IconButton(
                            tooltip: 'Recover',
                            onPressed: _busy || read.invalid
                                ? null
                                : () => _action(() async {
                                    await widget.library.publish(read.token);
                                  }),
                            icon: const Icon(Icons.save),
                          ),
                          trailing: IconButton(
                            tooltip: 'Discard pending read',
                            onPressed: _busy
                                ? null
                                : () => _action(
                                    () => widget.library.discardPending(
                                      read.token,
                                    ),
                                  ),
                            icon: const Icon(Icons.delete),
                          ),
                        ),
                    ],
                    if (reads.isNotEmpty) ...[
                      const ListTile(title: Text('Saved reads')),
                      for (final read in reads)
                        ListTile(
                          key: ValueKey('saved-${read.id}'),
                          title: Text(
                            read.invalid ? 'Unreadable saved read' : read.text,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: Text(
                            read.invalid
                                ? 'Metadata unavailable: ${read.problem}'
                                : read.missing
                                ? 'Audio missing'
                                : read.savedAt.toLocal().toString(),
                          ),
                          leading: IconButton(
                            tooltip: 'Play',
                            onPressed: _busy || read.missing || read.invalid
                                ? null
                                : () => _play(read),
                            icon: const Icon(Icons.play_arrow),
                          ),
                          trailing: IconButton(
                            tooltip: read.invalid
                                ? 'Remove invalid read'
                                : read.missing
                                ? 'Remove missing read'
                                : 'Delete',
                            onPressed: _busy
                                ? null
                                : () => _action(
                                    () => widget.library.delete(read),
                                  ),
                            icon: const Icon(Icons.delete),
                          ),
                        ),
                    ],
                  ],
                ),
              ),
          ],
        );
      },
    ),
  );
}
