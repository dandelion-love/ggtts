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
27.2.12479018 (`ANDROID_NDK_HOME`), cargo-ndk, both Rust targets, a host libclang for bindgen, and JDK 21 or newer. Android lint uses Java 21 APIs, so Gradle checks its actual runtime and fails early on older JDKs. Set JAVA_HOME, the Flutter configured JDK, or org.gradle.java.home to JDK 21. JDK 21 is verified; newer versions are untested.
The APK includes both native ABIs.

On Windows x86_64, install Flutter's Windows desktop prerequisites, Rust's MSVC target, CMake and libclang, then:

```powershell
./apps/flutter/tool/build_windows.ps1
```

This builds the C library and bundles it beside the app executable. Set `LIBCLANG_PATH` if bindgen cannot find it.
For an experimental CUDA bundle, use a Visual C++ x64 developer environment with Ninja and CUDA 12 installed,
set `CUDA_PATH` to that toolkit, and run `./apps/flutter/tool/build_windows.ps1 -Cuda`. This builds a separate
native target and copies the CUDA runtime and cuBLAS DLLs beside the app. The default remains CPU-only. The local
CUDA build uses `GGML_CUDA_ARCH` (default 86); a build is not evidence of working device inference. Pass
`-Diagnostics` only for the authenticated local developer harness described below. `-TargetDir` can select an
existing native build directory.
If MSBuild reports a path-length or FTK1011 error, build from a short checkout path; the native build was verified
from a short drive mapping after a longer checkout failed.
Initialize the repository's ggml submodule before building on any platform. Hosted Flutter packaging is opt-in via
`GGTTS_FLUTTER_CI=enabled`; local builds are the evidence while hosted minutes are unavailable.

## Playback

Android uses the existing miniaudio output by default. Its software-drain wait reports handoff, not confirmed device
presentation or audibility. The Android AudioTrack/JNI path is an experimental build-time opt-in and is not yet
validated for routing and interruption ordering.

## Share text on Android

Share plain text to ggtts, then choose **Use shared text** to replace the current
draft or **Discard shared text** to keep it. Receiving text never starts
playback. Shares are held in memory (up to eight waiting, plus the one displayed; each at most 100,000 UTF-16
code units) and can be lost if the app process ends; empty, malformed or oversized
shares are ignored. A share arriving while a read or preparation is running waits
until the app is idle.

The native playback ring targets 20 seconds of mono audio, capped at 960,000
float samples (3.84 MB); sample rates above 192 kHz are refused. This lets
generation overlap playback of longer pieces, but it cannot prevent pauses
when generation is slower than playback. Stop still clears queued audio.

## Read timing diagnostics

The app logs bounded `ggtts-read event=...` records without text or paths.
`load` measures the model reload for a read. Each `piece` records callback
wait, audio setup, FLAC append, PCM handoff, and the piece's audio duration in
milliseconds. Callback wait includes engine work and scheduling; handoff may
include backpressure or a pause. Neither measures silence heard by a listener
or confirms hardware playback. At most 128 piece lines are emitted per read,
followed by a limit marker. `drain` includes the existing output grace period.

## Model files

The explicit **Prepare model** action may download the default model when it is missing. Choose **Prepare offline**
to use selected or cached files without downloading; missing files are reported and can be downloaded with the other
action. Reading uses the cache offline. When a pinned model is downloading, preparation shows its downloaded amount, percentage and rate.
This requires native ABI 8; rebuild the packaged engine when updating the app. Stop requests cooperative cancellation; a blocked network call or model load may delay stopping. A fully verified file may remain cached if publication wins a cancellation race. In Settings, choose an engine and select model, codec and prepared-voice files, or enter app-readable paths.
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

## Developer read diagnostics (Linux and Windows)

A Linux or Windows build can opt into a local diagnostics channel. Normal builds do not
start a listener. The opt-in build also requires
\`GGTTS_DIAGNOSTICS_TOKEN\` at launch: exactly 64 lowercase hexadecimal digits
generated with a cryptographically secure random source. Treat this value as a
password and pass it through the process environment, not a URL or a log.
The app prints only the ephemeral port as \`ggtts-diagnostics port=...\`.
If the token is absent or malformed the channel remains unavailable.

Build with \`flutter build linux --dart-define=GGTTS_DIAGNOSTICS=true\` or \`flutter build windows --dart-define=GGTTS_DIAGNOSTICS=true\`.
The server binds IPv4 loopback only. Requests are POSTs to
\`http://127.0.0.1:<port>/<operation>\` with the header
\`Authorization: Bearer <token>\` and a UTF-8 JSON object body. It rejects
browser Origin headers, unexpected Host headers, unknown operations, bodies
over 64 KiB, and more than eight concurrent requests. The token permits the
holder to control the running app and access its rendered text and audio; use a
private process environment and a separate application data directory for
private test runs.

Operations are deliberately fixed:

- \`text\`: \`{"text":"..."}\` sets the read editor, with a 16,000 UTF-16-unit limit.
- \`prepare\`: prepares using cached or selected model files only, with no download.
- \`speak\`: performs the normal read path and completes when that read ends.
  \`status\`, \`stop\` and \`screenshot\` remain available while it runs.
- \`stop\`: requests the existing cooperative stop; \`ok: false\` means it was unavailable.
- \`save\`: publishes the pending completed read and returns its opaque saved-read ID.
- \`reopen\`: \`{"id":"..."}\` selects that ID from the app's saved-read
  library after a restart. It accepts no paths.
- \`replay\`: requests playback through the same audio player as Saved reads.
- \`audio\`: returns the selected saved read's FLAC (maximum 64 MiB).
- \`screenshot\`: returns a PNG of the read page's own Flutter render boundary,
  not a desktop or another application (maximum 16 MiB).
- \`status\`: returns fixed state flags and bounded text-free software timing events.

All other operations take \`{}\`. The channel does not capture system audio.
The FLAC is generated PCM, and a successful replay request does not establish
that hardware presented it or that anyone heard it. Timing events describe
software work, not audible silence. The channel is intended for trusted local
developer sessions and should be disabled in ordinary builds.
