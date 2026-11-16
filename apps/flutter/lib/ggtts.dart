// dart:ffi bindings to the ggtts C ABI (crates/ggtts-ffi/include/ggtts.h), options-document calls only.
//
// The app never fills ggtts_settings or calls ggtts_load_with: every setting travels in an options document
// (schemas/ggtts-options-1.schema.json), so the app carries whatever the shared table carries.
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

/// The ABI version this app is written against: ggtts_load_opts and ggtts_render_opts (4), ggtts_capabilities (5).
const int requiredAbi = 5;

const int _ok = 0;

typedef _AbiVersionC = Uint32 Function();
typedef _AbiVersionD = int Function();
typedef _LoadOptsC = Int32 Function(Pointer<Utf8>, Pointer<Uint64>);
typedef _LoadOptsD = int Function(Pointer<Utf8>, Pointer<Uint64>);
typedef _PieceC = Int32 Function(Pointer<Void>, Pointer<Float>, Size, Uint32);
typedef _RenderOptsC = Int32 Function(
  Uint64,
  Pointer<Utf8>,
  Pointer<Utf8>,
  Pointer<NativeFunction<_PieceC>>,
  Pointer<Void>,
  Pointer<Pointer<Float>>,
  Pointer<Size>,
  Pointer<Uint32>,
);
typedef _RenderOptsD = int Function(
  int,
  Pointer<Utf8>,
  Pointer<Utf8>,
  Pointer<NativeFunction<_PieceC>>,
  Pointer<Void>,
  Pointer<Pointer<Float>>,
  Pointer<Size>,
  Pointer<Uint32>,
);
typedef _PcmFreeC = Void Function(Pointer<Float>, Size);
typedef _PcmFreeD = void Function(Pointer<Float>, int);
typedef _FreeC = Int32 Function(Uint64);
typedef _FreeD = int Function(int);
typedef _LastErrorC = Pointer<Utf8> Function();
typedef _SchemaC = Pointer<Utf8> Function();
typedef _CapabilitiesC = Int32 Function(Uint64, Pointer<Pointer<Utf8>>);
typedef _CapabilitiesD = int Function(int, Pointer<Pointer<Utf8>>);
typedef _StringFreeC = Void Function(Pointer<Utf8>);
typedef _StringFreeD = void Function(Pointer<Utf8>);

/// A call the library refused: its code and ggtts_last_error()'s message.
class GgttsException implements Exception {
  GgttsException(this.code, this.message);
  final int code;
  final String message;
  @override
  String toString() => 'ggtts error $code: $message';
}

/// One read: mono float samples and their rate.
class Pcm {
  Pcm(this.samples, this.rate);
  final Float32List samples;
  final int rate;
  double get seconds => samples.length / rate;
}

/// Opens the library: on Android the jniLibs copy, on Linux the bundle's lib/ (or GGTTS_LIB).
DynamicLibrary openGgtts() {
  if (Platform.isAndroid) return DynamicLibrary.open('libggtts_ffi.so');
  final override = Platform.environment['GGTTS_LIB'];
  if (override != null && override.isNotEmpty) return DynamicLibrary.open(override);
  final exeDir = File(Platform.resolvedExecutable).parent.path;
  return DynamicLibrary.open('$exeDir/lib/libggtts_ffi.so');
}

class Ggtts {
  Ggtts(DynamicLibrary lib)
    : _abiVersion = lib.lookupFunction<_AbiVersionC, _AbiVersionD>(
        'ggtts_abi_version',
      ),
      _loadOpts = lib.lookupFunction<_LoadOptsC, _LoadOptsD>('ggtts_load_opts'),
      _renderOpts = lib.lookupFunction<_RenderOptsC, _RenderOptsD>(
        'ggtts_render_opts',
      ),
      _pcmFree = lib.lookupFunction<_PcmFreeC, _PcmFreeD>('ggtts_pcm_free'),
      _free = lib.lookupFunction<_FreeC, _FreeD>('ggtts_free'),
      _lastError = lib.lookupFunction<_LastErrorC, _LastErrorC>(
        'ggtts_last_error',
      ),
      _optionsSchema = lib.lookupFunction<_SchemaC, _SchemaC>(
        'ggtts_options_schema',
      ),
      _capabilities = lib.lookupFunction<_CapabilitiesC, _CapabilitiesD>(
        'ggtts_capabilities',
      ),
      _stringFree = lib.lookupFunction<_StringFreeC, _StringFreeD>(
        'ggtts_string_free',
      );

  final _AbiVersionD _abiVersion;
  final _LoadOptsD _loadOpts;
  final _RenderOptsD _renderOpts;
  final _PcmFreeD _pcmFree;
  final _FreeD _free;
  final Pointer<Utf8> Function() _lastError;
  final Pointer<Utf8> Function() _optionsSchema;
  final _CapabilitiesD _capabilities;
  final _StringFreeD _stringFree;

  int abiVersion() => _abiVersion();

  String _error() {
    final p = _lastError();
    return p == nullptr ? '' : p.toDartString();
  }

  void _check(int rc) {
    if (rc != _ok) throw GgttsException(rc, _error());
  }

  /// ggtts_load_opts with a load document, for example {"load": {"offline": true}}.
  int load(Map<String, Object?> options) {
    return using((arena) {
      final doc = jsonEncode(options).toNativeUtf8(allocator: arena);
      final handle = arena<Uint64>();
      _check(_loadOpts(doc, handle));
      return handle.value;
    });
  }

  /// The options schema is library-owned static JSON; never free its pointer.
  Map<String, Object?> optionsSchema() {
    final p = _optionsSchema();
    if (p == nullptr) throw StateError('ggtts_options_schema returned NULL');
    return jsonDecode(p.toDartString()) as Map<String, Object?>;
  }

  /// ggtts_capabilities of a loaded engine, decoded.
  Map<String, Object?> capabilities(int handle) {
    return using((arena) {
      final out = arena<Pointer<Utf8>>();
      _check(_capabilities(handle, out));
      try {
        return jsonDecode(out.value.toDartString()) as Map<String, Object?>;
      } finally {
        _stringFree(out.value);
      }
    });
  }

  /// A whole read through ggtts_render_opts with a render document (null: the load's render defaults).
  Pcm render(int handle, String text, [Map<String, Object?>? options]) {
    return using((arena) {
      final t = text.toNativeUtf8(allocator: arena);
      final doc = options == null
          ? nullptr.cast<Utf8>()
          : jsonEncode(options).toNativeUtf8(allocator: arena);
      final pcm = arena<Pointer<Float>>();
      final len = arena<Size>();
      final rate = arena<Uint32>();
      _check(_renderOpts(handle, t, doc, nullptr, nullptr, pcm, len, rate));
      try {
        return Pcm(
          Float32List.fromList(pcm.value.asTypedList(len.value)),
          rate.value,
        );
      } finally {
        _pcmFree(pcm.value, len.value);
      }
    });
  }

  /// Render on this isolate's thread. Each piece owns a copy of its samples.
  /// Return false from [onPiece] to stop; the engine reports GGTTS_ERR_STOPPED.
  /// The native engine must invoke the callback synchronously on this thread.
  void renderStream(
    int handle,
    String text,
    Map<String, Object?>? options,
    bool Function(Pcm) onPiece,
  ) {
    Object? callbackError;
    StackTrace? callbackStack;
    final callback = NativeCallable<_PieceC>.isolateLocal((
      Pointer<Void> user,
      Pointer<Float> samples,
      int len,
      int rate,
    ) {
      try {
        if (rate == 0 || len > 0x7fffffff || (samples == nullptr && len != 0)) {
          throw StateError(
            'invalid native PCM piece (length $len, rate $rate)',
          );
        }
        final copy = len == 0
            ? Float32List(0)
            : Float32List.fromList(samples.asTypedList(len));
        return onPiece(Pcm(copy, rate)) ? 0 : 1;
      } catch (e, stack) {
        callbackError = e;
        callbackStack = stack;
        return 1;
      }
    }, exceptionalReturn: 1);
    try {
      using((arena) {
        final t = text.toNativeUtf8(allocator: arena);
        final doc = options == null
            ? nullptr.cast<Utf8>()
            : jsonEncode(options).toNativeUtf8(allocator: arena);
        final rc = _renderOpts(
          handle,
          t,
          doc,
          callback.nativeFunction,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
        );
        if (callbackError != null) {
          Error.throwWithStackTrace(callbackError!, callbackStack!);
        }
        _check(rc);
      });
    } finally {
      callback.close();
    }
  }

  void free(int handle) => _check(_free(handle));
}

/// 16-bit PCM WAV bytes of a read, for a player that takes a file.
Uint8List wavBytes(Pcm pcm) {
  final n = pcm.samples.length;
  final b = ByteData(44 + n * 2);
  void ascii(int at, String s) {
    for (var i = 0; i < s.length; i++) {
      b.setUint8(at + i, s.codeUnitAt(i));
    }
  }

  ascii(0, 'RIFF');
  b.setUint32(4, 36 + n * 2, Endian.little);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  b.setUint32(16, 16, Endian.little);
  b.setUint16(20, 1, Endian.little);
  b.setUint16(22, 1, Endian.little);
  b.setUint32(24, pcm.rate, Endian.little);
  b.setUint32(28, pcm.rate * 2, Endian.little);
  b.setUint16(32, 2, Endian.little);
  b.setUint16(34, 16, Endian.little);
  ascii(36, 'data');
  b.setUint32(40, n * 2, Endian.little);
  for (var i = 0; i < n; i++) {
    final s = (pcm.samples[i].clamp(-1.0, 1.0) * 32767).round();
    b.setInt16(44 + i * 2, s, Endian.little);
  }
  return b.buffer.asUint8List();
}
