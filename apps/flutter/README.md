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

## Model files

The app loads the engine's default model from the engine's cache with downloads off (`"offline": true`), so a
missing model is reported, never fetched by surprise. On Linux the cache is `~/.cache/ggtts` (or `GGTTS_CACHE_DIR`).
On Android it is the app's private documents folder. For a debuggable trial build, stage the verified cache through
the app's own identity (install the APK first):

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
