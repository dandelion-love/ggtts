# Finding: a Flutter app can drive the engine through the C ABI

Measured 2026-11-15 on `apps/flutter/` (Flutter 3.50.0 stable, Dart 3.14.0), the library built from this
repository's `rc` at the spike's base commit.

## The question

Can a thin Flutter shell call `ggtts_load_opts` and `ggtts_render_opts` with an options document on Linux and
Android, and what does it cost? The answer decides the stack for the cross-platform demo app.

## What was shown

| Check | Linux x86_64 (WSL2) | Android arm64 (Galaxy Z Fold5, Android 17) |
| --- | --- | --- |
| Library builds (`tool/build_native.sh`) | yes, 26 s incremental | yes, `libggtts_ffi.so` 6.8 MB plus `libc++_shared.so` 1.8 MB |
| App builds | headless Dart only (GUI needs packages, below) | debug APK 86.6 MB, release APK 23.0 MB (arm64 only) |
| `dart:ffi` loads the library, `ggtts_abi_version()` | 5 | 5 (logcat) |
| `ggtts_load_opts`, default model from the cache, offline | 3.5 s | started; not finished (below) |
| `ggtts_capabilities` | `qwen3-tts`, 2 speakers | not reached |
| `ggtts_render_opts`, one line of 15 characters | 3.89 s of audio in 6.8 s (8 threads) | not reached |
| Error path: empty cache, downloads off | refused with the library's message (`GgttsException -3`, naming the missing files and URLs); no crash | not run |

## What it cost and what broke

- **Bindings are small.** Eight functions, typed by hand from `ggtts.h` in `lib/ggtts.dart` (about 150 lines). An
  options document is a `Map` encoded to JSON; no struct crosses the boundary.
- **A wrong document is caught by the engine, not the app.** The first render passed `{"seed_salt": ...}` without its
  `render` section and came back as `ggtts error -1: seed_salt: is not an options section`. That is the one capability
  surface doing its job: the app needs no option logic of its own.
- **Rendering must leave the UI isolate,** and the closure handed to `Isolate.run` must be built outside the widget
  state: a closure made in a `State` captures it (here its `AudioPlayer`), which cannot cross isolates. The device
  found this; the headless Linux check could not.
- **Android builds need what the native Android build already needs:** NDK 27, cargo-ndk, a host libclang for
  bindgen (`ANDROID_NDK_ROOT` set for ggml's CMake, and the NDK's builtin headers passed to bindgen), and JDK 17 or
  newer for Gradle. `path_provider` pulls in a native plugin that needs the SDK's CMake and Ninja.
- **Linux desktop builds need system packages** (`clang`, `ninja-build`, `pkg-config`, `libgtk-3-dev`, GStreamer dev
  packages for playback), which need root; the build host did not have them, so the Linux GUI was not built.
- **Android freezes an app behind the lock screen.** On a locked phone the render started and the process then sat
  at 0% CPU; the on-device render and playback wait for a person to unlock the phone.

## Answer

**Go for Flutter, provisionally.** Every step that ran worked first time apart from the two app bugs above, and
nothing needed per-option wiring. Two things remain before the answer is final: the Linux GUI build with playback,
and the on-device render with playback, both of which need a person (root for packages; an unlocked phone).
The follow-up below supersedes this provisional state.


## Follow-up: Linux GUI and Android device

On 2026-11-15, a Linux debug GUI built with Flutter 3.50.0. With the GStreamer good and PulseAudio runtime plugins
installed, the auto-speak run logged ABI 5, a 3.89 s render at 24 kHz (load 3198 ms; render 7584 ms), then player
start without the earlier missing-plugin exception. This log does not establish that sound reached a speaker.

On an unlocked Galaxy Z Fold5, an external cache populated via adb failed with permission denied despite matching
the talker file's SHA-256. A debug build using app-private documents storage and the same files copied there by
`run-as` logged ABI 5 and a 3.65 s render at 24 kHz (load 12513 ms; render 78358 ms), then player start. The
private-cache test demonstrates a working route, not the precise cause of the external-storage failure. Player
start alone does not establish audible output. The error was displayed without an app crash.

These observations support Flutter for the spike. The adb staging procedure is developer-only; model provisioning
and playback verification remain necessary for a usable app. Times are single observations, not benchmarks.
