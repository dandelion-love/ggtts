# ggtts on Android: the library and the demo app

Two Gradle modules:

* **`ggtts`**, the Android library (AAR): the Kotlin binding [`Ggtts.kt`](ggtts/src/main/kotlin/ggtts/Ggtts.kt) and
  `libggtts_ffi.so` for `arm64-v8a` (the C-ABI with its JNI entry points, over ggml), with the NDK's
  `libc++_shared.so` beside it. How to call it: [ggtts from Kotlin and Android](../docs/bindings/kotlin-android.md).
* **`demo`**, a one-screen app (package `ggtts.demo`) that loads the model files you pick and speaks one line with
  Qwen3-TTS or Kitten-TTS 2, through the library.

**No model ships with the app.** The APK holds code only; you bring the model files. The build refuses an APK that
carries a `.gguf` (or any other weights file), and checks that refusal on a planted APK every time it runs.

## Build

```sh
git submodule update --init crates/ggtts-ggml-sys/ggml
rustup target add aarch64-linux-android
cargo install cargo-ndk --version 3.5.4 --locked
export ANDROID_HOME=/path/to/android-sdk                # with platforms;android-35
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/27.2.12479018
export JAVA_HOME=/path/to/jdk-17                        # 17 or later
scripts/android_build.sh app dist/android-arm64
```

This builds `libggtts_ffi.so` with cargo-ndk (ggml at its pin, NEON checked in the kernels), then runs Gradle (8.11.1,
downloaded once and checked against its sha256) to build the AAR and the demo's debug APK and to run the demo's JVM
tests. It writes `dist/android-arm64/ggtts.aar` and `dist/android-arm64/ggtts-demo-debug.apk`, and checks both. CI's
`android-arm64` job runs it on every pull request that touches `android/` and uploads both files as the job's
artifact.

The Android library is arm64-v8a only, API level 24 or later, and runs on the CPU.

## Bring your own model

The demo reads model files from its own directory on the device:

```
/sdcard/Android/data/ggtts.demo/files/models/
```

Put them there in either of two ways:

* **In the app:** "Import a model file..." opens the system file picker, and copies the file you choose into that
  directory. Copy each file once; a 2 GB file takes a while.
* **With adb**, from a computer, after the app has been opened once (opening it creates the directory):

  ```sh
  adb push qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf /sdcard/Android/data/ggtts.demo/files/models/
  ```

Then pick the files in the three lists, and press Speak.

| Engine | Model | Codec | Voice | Speaker |
|---|---|---|---|---|
| Qwen3-TTS | the talker GGUF | the 12 Hz codec GGUF | (none) | one the talker has |
| Kitten-TTS 2 | the LM GGUF | the S3Gen GGUF | the prepared voice JSON | (none) |

A Qwen3-TTS talker picked as a file uses the speaker `hk_cantonese_speaker` when the Speaker box is empty. The default
model's talker has `hk_voice_343` and `hk_voice_777` instead, so type one of those; with an unknown speaker the screen
shows the library's error, which lists the talker's speakers.

* **Qwen3-TTS.** The default model's two files are
  [`qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf` and `qwen-tokenizer-12hz-q8_0.gguf`](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF)
  (2.33 GB together; the README's [default model files](../README.md#default-model-files-what-ggtts-downloads-and-when)
  lists their sha256). Any Qwen3-TTS talker and codec that `ggtts-render` takes will do. The demo never downloads a
  model: with no model picked, the library looks for the default files in the app's cache, and says which file is
  missing and where it comes from.
* **Kitten-TTS 2.** No ggtts-ready Kitten files are published. Make the three files once on a computer with
  `ggtts-kitten-convert` and `ggtts-kitten prepare`, as [the Kitten README](../crates/ggtts-kitten/README.md) shows,
  then copy them to the device. Its licence is the model's: audio rendered with it must not be used to train other
  models.

Read each model's card before you use its voice; its licence is the model's, not this repository's.

**When a file is missing or wrong** (a path that does not exist, a file that is not a GGUF, a voice JSON picked as
the model), the screen shows the library's error, for example `ggtts error -3: ...`, and stays usable: pick again and
press Speak. A model the phone cannot hold in memory may still stop the app; Kitten-TTS 2 at Q4_0 is the smaller of
the two.

Each line is saved to `/sdcard/Android/data/ggtts.demo/files/out/<engine>.wav` (16-bit) and played. The app logs to
logcat under the tag `ggtts-demo`: what it loaded, how long the render took, and its real-time factor.

## Driving the demo from adb

For a scripted check, the activity takes its settings as extras; `speak` presses Speak. A file name is looked up in
the models directory, an absolute path is used as it is.

```sh
adb install -r dist/android-arm64/ggtts-demo-debug.apk
adb shell am start -n ggtts.demo/.MainActivity --es engine qwen3-tts \
    --es model qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf --es codec qwen-tokenizer-12hz-q8_0.gguf \
    --es speaker hk_voice_343 --ez speak true
adb logcat -s ggtts-demo
adb pull /sdcard/Android/data/ggtts.demo/files/out/qwen3-tts.wav
```

## Remove it

```sh
adb uninstall ggtts.demo
```

This removes the app and everything in its directory, the model files included.

## What the tests cover

`scripts/android_build.sh app` runs the demo's JVM tests (`android/demo/src/test`). They replace the native library
with a fake binding that throws the library's `GgttsException`, and check that a missing model, a file that is not a
GGUF, a failed render and an unexpected exception each show a message and leave the screen usable. They test the
app's error path, not the native library: a JVM on the build machine cannot load an `arm64-v8a` library, and CI has
no arm64 device. The native library's errors on a phone are checked by hand on a device.
