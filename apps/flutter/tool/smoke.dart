// Headless check of the app's bindings (lib/ggtts.dart) on the host: load, capabilities, render one line, write WAV,
// then the error path. Needs GGTTS_LIB (the built libggtts_ffi.so); the model comes from the engine's cache.
//
//   GGTTS_LIB=native/linux-x64/libggtts_ffi.so dart run tool/smoke.dart [out.wav]
import 'dart:io';

import 'package:ggtts_app/ggtts.dart';

void main(List<String> args) {
  final out = args.isEmpty ? 'smoke.wav' : args.first;
  final g = Ggtts(openGgtts());
  final abi = g.abiVersion();
  stdout.writeln('abi $abi');
  if (abi < requiredAbi) {
    stderr.writeln('library ABI $abi is older than $requiredAbi');
    exit(1);
  }
  final cache = Platform.environment['GGTTS_CACHE_DIR'] ?? '${Platform.environment['HOME']}/.cache/ggtts';
  final sw = Stopwatch()..start();
  final h = g.load({
    'load': {'offline': true, 'cache_dir': cache, 'threads': 8},
  });
  stdout.writeln('load ${sw.elapsedMilliseconds} ms');
  final caps = g.capabilities(h);
  stdout.writeln('capabilities family=${caps['family']} speakers=${(caps['speaker_names'] as List).length}');
  sw.reset();
  final pcm = g.render(h, '今日天氣幾好，我哋出去行下啦。', {
    'render': {'seed_salt': 'spike'},
  });
  stdout.writeln('render ${sw.elapsedMilliseconds} ms: ${pcm.seconds.toStringAsFixed(2)} s at ${pcm.rate} Hz');
  File(out).writeAsBytesSync(wavBytes(pcm));
  stdout.writeln('wrote $out');
  g.free(h);

  // The error path: a cache with no model and downloads off is refused with the library's message, not a crash.
  try {
    g.load({
      'load': {'offline': true, 'cache_dir': '${Directory.systemTemp.path}/ggtts-empty-cache'},
    });
    stderr.writeln('expected a refusal for an empty cache');
    exit(1);
  } on GgttsException catch (e) {
    stdout.writeln('refused as expected: $e');
  }
}
