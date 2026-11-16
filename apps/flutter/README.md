# ggtts Flutter app (spike)

A thin Flutter shell over the ggtts C ABI (`crates/ggtts-ffi/include/ggtts.h`) through `dart:ffi`. It loads the
library, renders one line through an options document (`ggtts_load_opts`, `ggtts_render_opts`), and plays it. It
never fills `ggtts_settings`: every setting travels in the options document, so the app carries what the shared
options table carries. This is the spike that decides the stack for the cross-platform demo app; see
[the finding](../../docs/app/flutter-spike.md).

## Build

Flutter stable (tested with 3.50.0) and the toolchains below. The native library is built first and is never
committed (`.gitignore`).

```bash
# Linux desktop: needs clang, ninja-build, pkg-config, libgtk-3-dev, liblzma-dev, the GStreamer dev packages, and the GStreamer good and PulseAudio runtime plugins.
apps/flutter/tool/build_native.sh linux
cd apps/flutter && flutter run -d linux

# Android arm64: NDK 27.2.12479018 (ANDROID_NDK_HOME), cargo-ndk, the aarch64-linux-android Rust target,
# a host libclang for bindgen (LIBCLANG_PATH), and JDK 17 or newer for Gradle.
apps/flutter/tool/build_native.sh android
cd apps/flutter && flutter build apk --target-platform android-arm64
```

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
