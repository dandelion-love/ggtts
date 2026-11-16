# ggtts Flutter app (spike)

A thin Flutter shell over the ggtts C ABI (`crates/ggtts-ffi/include/ggtts.h`) through `dart:ffi`. It loads the
library, renders one line through an options document (`ggtts_load_opts`, `ggtts_render_opts`), and plays it. It
never fills `ggtts_settings`: every setting travels in the options document, so the app carries what the shared
options table carries. This is the spike that decides the stack for the cross-platform demo app; see
[the finding](../../docs/app/flutter-spike.md).

## Build

Flutter stable (tested with 3.50.0) and the toolchains below. The native library is built first and is never
committed (`.gitignore`).

Linux x86_64 and Android arm64-v8a plus x86_64 use a single command each from the repository root:

```bash
apps/flutter/tool/build_app.sh linux
apps/flutter/tool/build_app.sh android
```

Linux needs Rust (the pinned toolchain), clang, ninja-build, pkg-config, GTK 3 and GStreamer development
packages, liblzma-dev, and the GStreamer good and PulseAudio runtime plugins. Android additionally needs NDK
27.2.12479018 (`ANDROID_NDK_HOME`), cargo-ndk, both Rust targets, a host libclang for bindgen, and JDK 17 or newer.
The APK includes both native ABIs.

On Windows x86_64, install Flutter's Windows desktop prerequisites, Rust's MSVC target, CMake and libclang, then:

```powershell
./apps/flutter/tool/build_windows.ps1
```

This builds the C library and bundles it beside the app executable. Set `LIBCLANG_PATH` if bindgen cannot find it.
If MSBuild reports a path-length or FTK1011 error, build from a short checkout path; the native build was verified
from a short drive mapping after a longer checkout failed.
Initialize the repository's ggml submodule before building on any platform. Hosted Flutter packaging is opt-in via
`GGTTS_FLUTTER_CI=enabled`; local builds are the evidence while hosted minutes are unavailable.

## Playback

Android uses the existing miniaudio output by default. Its software-drain wait reports handoff, not confirmed device
presentation or audibility. The Android AudioTrack/JNI path is an experimental build-time opt-in and is not yet
validated for routing and interruption ordering.

## Model files

The explicit **Prepare model** action may download the default model when it is missing; reading uses the cache
offline. When a pinned model is downloading, preparation shows its downloaded amount, percentage and rate.
This requires native ABI 7; rebuild the packaged engine when updating the app. Download cancellation is not yet available. In Settings, choose an engine and select model, codec and prepared-voice files, or enter app-readable paths.
On Linux and Windows the picker returns a filesystem path. On Android the system picker imports a private copy, with
an 8 GiB model/codec limit and a 64 MiB voice limit. It checks the GGUF signature or JSON object prefix; the engine
performs further validation. Picker cancellation leaves the previous setting unchanged. Unsaved copies are removed
on a best-effort basis; a process crash or a provider that blocks indefinitely can leave an incomplete private file.
This is not a catalogue, Wi-Fi-only policy or resumable download. On Linux the cache is `~/.cache/ggtts` (or
`GGTTS_CACHE_DIR`); on Android it is the app's private documents folder. For a debuggable trial build, stage the verified cache through the app's own identity
(install the APK first):

```bash
adb shell 'mkdir -p /sdcard/Android/data/dev.ggtts.ggtts_app/files/ggtts-cache'
adb push ~/.cache/ggtts/models /sdcard/Android/data/dev.ggtts.ggtts_app/files/ggtts-cache/
adb shell 'run-as dev.ggtts.ggtts_app mkdir -p app_flutter/ggtts-cache/models'
adb shell 'run-as dev.ggtts.ggtts_app cp -R /sdcard/Android/data/dev.ggtts.ggtts_app/files/ggtts-cache/models/. app_flutter/ggtts-cache/models/'
```

This is a developer-only staging procedure; a user-facing model provisioner remains needed.

## Checks

```bash
cd apps/flutter && flutter analyze && flutter test
# The bindings on the host, without a GUI: load, capabilities, one render, the error path.
GGTTS_LIB=native/linux-x64/libggtts_ffi.so dart run tool/smoke.dart /tmp/spike.wav
```

`--dart-define=GGTTS_AUTOSPEAK=true` makes a build speak on launch and log to `flutter` in logcat, so a device
check needs no tap. The app must be in the foreground: Android freezes it behind the lock screen.


The Android build runs the APK native gate tests, builds the native libraries,
and checks the packaged ARM64 and x86-64 ELF libraries for 16 KB LOAD alignment,
the Dart binding's required C exports, and equality with the current build after
stripping. The gate confirms that ggtts_abi_version is exported; it cannot
execute the cross-architecture library to verify its returned version. The app
checks that value at runtime when it loads the library.
