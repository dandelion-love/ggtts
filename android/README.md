# ggtts on Android: the library and the demo app

Two Gradle modules:

* **`ggtts`**, the Android library (AAR): the Kotlin binding [`Ggtts.kt`](ggtts/src/main/kotlin/ggtts/Ggtts.kt) and
  `libggtts_ffi.so` for `arm64-v8a` (the C-ABI with its JNI entry points, over ggml), with the NDK's
  `libc++_shared.so` beside it. How to call it: [ggtts from Kotlin and Android](../docs/bindings/kotlin-android.md).
* **`demo`**, the app "ggtts Text-to-Speech": a one-screen app (package `ggtts.demo`) that loads the model files you pick and speaks one line with
  Qwen3-TTS or Kitten-TTS 2, through the library.

**No model ships with the app.** The APK holds code only; you bring the model files. The build refuses an APK that
carries a weights file, by name (`.gguf`, `.safetensors`, `.bin`, `.pt`, `.onnx`) or by content (any entry starting
with GGUF's magic bytes), and checks both refusals on planted APKs every time it runs.

## What the app is for

**Reads text aloud in Cantonese or English, entirely on this phone.** You type or paste a passage and listen to it
in a neural voice; nothing you type leaves the phone, because the app has no permission to use the internet. It is
slow: on a Galaxy Z Fold5 the first sound takes 30 to 50 seconds with a Cantonese voice and 45 to 60 with an English
one, so it suits listening to a passage you have chosen, not reading the screen as you go, and it is not for screen
readers. Other apps can use it as a text-to-speech engine in Android's settings.

The reading screen opens with the first line as its subtitle, the wait on the player ("First sound in about…"), and,
while the text box is empty, a hint that other apps can use it; the rest is under ⋮ › About this app, with the steps
still ahead when no model files are present yet. They live in `About.kt`, and `AboutTest` holds them to the app:
adding the `INTERNET` permission, or a measured wait in `EngineCosts` that About does not state, fails the test until
the words change too.

## Build

```sh
git submodule update --init crates/ggtts-ggml-sys/ggml
rustup target add aarch64-linux-android
cargo install cargo-ndk --version 3.5.4 --locked
export ANDROID_HOME=/path/to/android-sdk                # with platforms;android-35
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/27.2.12479018
export JAVA_HOME=/path/to/jdk-17                        # 17 to 23 (Gradle 8.11's range)
scripts/android_build.sh app dist/android-arm64
```

This builds `libggtts_ffi.so` with cargo-ndk (ggml at its pin, NEON checked in the kernels, LOAD segments aligned to
16 KB), then runs Gradle (8.11.1, downloaded once and checked against its sha256) to build the AAR and the demo's
debug APK and to run the demo's JVM tests. `scripts/android_build.sh build`, which also builds `ggtts-render` for
Android, ends by running `app`, so it needs the same JDK and SDK. Both write into `dist/android-arm64/`:

| File | What it is |
|---|---|
| `ggtts.aar` | The library: `Ggtts.kt`'s classes, `jni/arm64-v8a/libggtts_ffi.so` and `libc++_shared.so`, and `META-INF/ggtts/` (below). |
| `ggtts-demo-debug.apk` | The demo app, debug-signed so it installs with `adb install`. |
| `symbols/arm64-v8a/libggtts_ffi.so` | The same library with its symbols, for reading a native crash (`ndk-stack -sym symbols/arm64-v8a`). |

CI's `android-arm64` job runs it on every pull request that touches `android/` and uploads that directory as the job's
artifact.

The Android library is arm64-v8a only, API level 24 or later, and runs on the CPU.

### What the AAR carries

* **Its version and build record**, `META-INF/ggtts/BUILDINFO.json`: the ggtts version and commit, the ggml pin, the
  Rust toolchain, the target and ABI, the minimum SDK, the NDK, and the features built in (the engines, `jni`, and the
  CPU extensions found in the kernels). The AAR file itself is not versioned by name; read the record.
* **Licence notices**, under `META-INF/ggtts/`: ggtts's `LICENSE` and `THIRD_PARTY.md`, and in `licenses/` the
  licences of ggml (`ggml-LICENSE.txt`), of every Rust crate linked into the library (`android-crates.txt`, generated
  by `scripts/third_party_notices.py` and checked against `Cargo.lock` before every build), and of the NDK's libc++
  (`libc++-LICENSE.txt`, Apache-2.0 with LLVM exceptions, taken from the NDK that built it). They are Java resources,
  so they travel into every APK that uses the AAR. An app that shows its users open-source licences should include
  them. The build refuses an AAR or an APK without them.
* **Stripped libraries.** Gradle strips debug symbols from both `.so` files when it packs them (`libggtts_ffi.so`
  goes from about 6.2 MB to 4.7 MB). The library with its symbols is `symbols/arm64-v8a/libggtts_ffi.so` in the same
  output; `libc++_shared.so`'s symbols come from the NDK.

## Bring your own model

The app asks for no storage permission. It reads model files from two places, and lists the voices both make:

* **A folder you choose once** (the easiest to fill): "Choose a models folder…" in the ⋮ menu opens the system's
  folder picker; pick or create one such as `Download/ggtts`, and copy model files into it with a file manager or
  over USB from a computer. The app keeps read access to that folder across restarts. When the folder is chosen,
  and each time the app opens, it copies the folder's model files that it does not have yet into its own directory,
  with the same checks as an import (below); a voice's sample in the folder plays from there. Choosing another
  folder replaces the first. The copy doubles the space a model takes, so delete the folder's copy once the app has
  it if space is short; the voice stays.

  Why a copy: the engine opens a model by path, and a picked file reaches it only as `/proc/self/fd/<n>`. Opening
  that path checks access to the file underneath, which is on shared storage the app may not read without a storage
  permission. On a Galaxy Z Fold5 (Android 17) all three loads that way failed with "Permission denied (os error
  13)". The copy itself is quick there: 0.55 s for the 291 MB codec and 3.5 s for the 2.04 GB talker. A load from the
  app's directory took 2.3 to 2.8 s, the same as before.
* **Its own directory**, which Android 11 and later hide from file managers and USB, so it is filled from the app or
  with adb:

  ```
  /sdcard/Android/data/ggtts.demo/files/models/
  ```

A file of the same name in both places is the app directory's.

Put files in the app's own directory in either of two ways:

* **In the app:** "Import a voice file…" in the ⋮ menu (or "Add a voice from a file" in the voice sheet) opens the
  system file picker, and copies the file you choose into that
  directory. Copy each file once; a 2 GB file takes a while. The import takes only a `.gguf` that starts with GGUF's
  magic bytes or a `.json` voice; it refuses an empty file, and it never replaces a file of the same name. It copies
  to `NAME.part` and renames the file when the copy is complete, so a cancelled or failed copy never shows in the lists.
  The default Qwen3-TTS model's two files are checked against their published sha256 while they copy, and a file with
  other bytes is refused.
* **With adb**, from a computer, after the app has been opened once (opening it creates the directory):

  ```sh
  adb push qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf /sdcard/Android/data/ggtts.demo/files/models/
  ```

Then tap the voice card to choose a voice, and press Read aloud. The app reads what each file is from the file itself, not from its name: a
GGUF's `general.architecture`, a Qwen3-TTS talker's speaker names, and a voice JSON's fields. It lists one voice for
each speaker a talker names (with a codec), and one for each prepared voice JSON (with a Kitten-TTS 2 language model
and decoder), and selects the first. Files that make no voice yet are named on the screen with what they lack. A
voice's sample is `<voice name>.sample.wav` beside the files, and the Sample button on its card in the voice sheet
plays it. `scripts/android_voice_samples.sh` makes one for each voice with the app itself, over adb, so a sample is
what that voice sounds like from those files on that phone.

A voice's language is the one its engine reads as Android's text-to-speech engine (Cantonese for Qwen3-TTS, English
for Kitten-TTS 2): no model file here declares a language, so a Kitten-TTS 2 voice made for Cantonese is still listed
as English.

"Advanced" (in the ⋮ menu, closed by default) holds what a voice sets: the engine, the three file lists, the speaker and the CPU
threads. A choice there is what Speak uses until another voice is chosen.

| Engine | Model | Codec | Voice | Speaker |
|---|---|---|---|---|
| Qwen3-TTS | the talker GGUF | the 12 Hz codec GGUF | (none) | one the talker has |
| Kitten-TTS 2 | the LM GGUF | the S3Gen GGUF | the prepared voice JSON | (none) |

A Qwen3-TTS talker picked as a file uses its default speaker when the Speaker box is empty: `hk_cantonese_speaker` if
it has one, otherwise the first of its speakers in alphabetical order (`hk_voice_343` for the released talkers, which
also have `hk_voice_777`). With an unknown speaker the screen shows the library's error, which lists the talker's
speakers.

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

**What each engine costs is on the screen before you press Speak:**
- the model files' size and the memory it peaked at;
- how long the first audio and each second of speech took, on the phone it was measured on;
- that it is not suitable for screen readers.

Before loading, the app checks free memory. Below the engine's measured peak, or when Android reports low memory,
the first press of Speak only shows a warning; pressing Speak again loads anyway. It warns rather than refuses
because the peak is a spike of a few seconds while the model loads (its files are read whole and copied); a render
then holds less, about 2.9 GB for Kitten-TTS 2, which rendered with 3.2 GB available against a 3.8 GB peak. If memory
runs out, it is during the load that Android stops the app.

**Speaking streams.** The line plays chunk by chunk: each chunk starts playing as soon as it is rendered, while the
next renders. On a phone, a chunk takes tens of seconds (the engines run at 7 to 13 times slower than real time on
a Snapdragon 8 Gen 2), so the first sound comes after the first chunk, not after the whole line. Stop silences
playback at once and ends the render after the chunk in progress. The status line shows which chunk is ready and
when the first audio came.

**CPU threads** can be 2, 4, 6 or 8 (default 4). On the phone this was measured on, 4 to 8 threads rendered at
about the same speed within thermal drift, 2 was slower, and the audio was bit-identical at every count.

When the line is done, the whole read is saved to `/sdcard/Android/data/ggtts.demo/files/out/<engine>.wav`
(16-bit). The streamed pieces are the render's samples before its final level step, so the saved file and what was
heard can differ slightly in loudness. That folder is hidden from file managers and USB on Android 11 and later, so a
finished read has **Save to Music** and **Share** under the player: Save adds it to `Music/ggtts/` as
`ggtts-<engine>-<date>-<time>.wav`, and Share sends that copy to the app you choose. Each voice's sample is copied to
`Music/ggtts/` too, once. Both go through Android's media store, which needs no permission on Android 10 and later;
on Android 7 to 9 the two buttons are hidden. The app logs to logcat under the tag `ggtts-demo`: what it loaded, the time
to first audio, how long the render took, and its real-time factor.

## Leaving the screen during a read

A read belongs to the app, not its screen. Leave the screen and the read goes on, in a foreground service. Its
notification shows the progress and has Stop, and reopening the app shows where the read is. The loaded model stays
loaded between reads, shared with the system engine below, and is released when Android asks for memory back while
nothing is being read.

- **Off screen it is much slower.** On the phone this was measured on, Android moved the app to its `moderate`
  cpuset once the screen was left: the four render threads ran only on the three Cortex-A510 little cores. An app
  cannot change that. So the screen stays on while a read runs and the app is open.
- **The notification needs permission.** On Android 13 and later the app asks for it the first time you press
  Speak. If you refuse, the read still runs, but the notification and its Stop are hidden until you allow
  notifications for the app in its settings. A Speak driven by adb extras never asks; grant it there with
  `adb shell pm grant ggtts.demo android.permission.POST_NOTIFICATIONS`.
- **One read at a time.** The screen and the system engine share one loaded engine, and one of them reads with it at
  a time. While one reads, the other is refused with a message (the screen) or `ERROR_SERVICE` (the system engine)
  rather than load a second model beside it. Android's memory trims never release the engine during either read.

## Text from other apps

Text can reach the app three ways, and none of them needs a permission:

- **Share:** share plain text to "ggtts Text-to-Speech" from another app's share sheet (`ACTION_SEND`, `text/plain`).
- **Selection menu:** select text in another app and pick **Read aloud** from the menu (`ACTION_PROCESS_TEXT`).
- **Open a file:** open a `.txt` file with the app, or share one to it. It must be UTF-8 and at most 256 KB.

Each fills the text card, keeps the voice as it is, and says where the text came from and whether it replaced what
the card held. **Nothing starts a read by itself**: you press Read aloud. Another app chooses what it sends, so a read
it could start would be that app speaking through this one. Text over 20,000 characters (counted as a reader counts
them: an emoji is one) is refused with that limit named, not cut short. A file that is not UTF-8 is refused too, and so
is anything that is not plain text. The text is taken once: if the screen is re-created later (dark mode or the
language changed), it keeps what you have in the card rather than taking the shared text again.

## ggtts as Android's text-to-speech engine

The demo app also provides a system TTS engine, a `TextToSpeechService` listed in the phone's settings as "ggtts Text-to-Speech",
the app's own name. It is slow: its voices declare very high latency, and the first audio takes tens of seconds. Any app
that speaks through Android's `TextToSpeech` API can use it, either by naming the engine (`ggtts.demo`) or after the
user picks it in the phone's text-to-speech settings. The app does not change that setting itself.

- **Voices:** Cantonese (Hong Kong), `yue-HK`, from Qwen3-TTS, and English (US) from Kitten-TTS 2. Each voice uses
  the model files of that engine's last finished Speak in the demo. Until there has been one, the engine reports the
  language as missing data, and a request gets `ERROR_NOT_INSTALLED_YET`. A voice is saved as its whole options
  document (the files, the speaker, the threads, and any other option the request carries). A saved voice the
  binding refuses (a key it does not take, say) is reported unusable with the binding's message. It is not listed,
  and a request gets `ERROR_INVALID_REQUEST` until the next finished Speak saves the voice again.
- **Speech rate and pitch** are not available: ggtts has no option for either. A request at a rate or pitch other than
  normal is spoken at normal, and the app logs one line naming what it ignored.
- **Latency:** every voice is declared `LATENCY_VERY_HIGH`, so a client that reads voice latency can see it before
  choosing it. It streams like the demo: the platform gets each chunk as it is rendered.
- **Not for screen readers.** The first audio takes tens of seconds.
- **"Speak through Android's TTS"** on the demo's screen speaks the line the way another app would: through
  `TextToSpeech`, with `ggtts.demo` named as the engine.

## Driving the demo from adb

For a scripted check, a **debug build** of the activity takes its settings as extras; `speak` presses Speak. The
`model`, `codec` and `voice` extras are file names in the models directory (put the files there first, with Import or
`adb push`); a path, or a name with `/` or `..`, is refused with a message. The activity is exported (it is the
launcher), so any app on the phone could send it these extras: limiting them to files already in the models
directory means another app cannot point the demo, or the system engine that remembers its files, at a file of its
own. One refused name refuses the whole intent, `speak` included. A release build ignores the extras. One gap remains
on Android 7 to 10: an app holding the legacy storage permission can write into the demo's models directory
(`Android/data/ggtts.demo/files/models`) and then name the file it planted, in a debug build only.

Pass `--activity-single-top`. Without it, when the app is in the background, Android may only bring its task to the
front ("its current task has been brought to the front") and drop the extras.

```sh
adb install -r dist/android-arm64/ggtts-demo-debug.apk
adb shell am start --activity-single-top -n ggtts.demo/.MainActivity --es engine qwen3-tts \
    --es model qwen-talker-hk-cantonese-cc0-v0.0.5-q8_0.gguf --es codec qwen-tokenizer-12hz-q8_0.gguf \
    --es speaker hk_voice_343 --ei threads 4 --ez speak true
adb shell am start --activity-single-top -n ggtts.demo/.MainActivity --ez stop true   # presses Stop
adb shell am start --activity-single-top -n ggtts.demo/.MainActivity --ez speak_system true   # through Android's TTS
adb logcat -s ggtts-demo
adb pull /sdcard/Android/data/ggtts.demo/files/out/qwen3-tts.wav
```

## Remove it

```sh
adb uninstall ggtts.demo
```

This removes the app and everything in its directory, the model files included.

## What the tests cover

`scripts/android_build.sh app` runs the demo's JVM tests (`android/demo/src/test`). `DemoControllerTest` replaces the
native library with a fake binding that throws the library's `GgttsException`, and checks that a missing model, a file
that is not a GGUF, a failed render and an unexpected exception each show a message and leave the screen usable.
`VoicesTest` checks how files make voices: one per talker speaker or voice JSON, in order, the first complete one
selected, and each incomplete bundle named with what it lacks; and it reads a GGUF header built in the test.
`ModelImportTest` gives the import each file it must refuse (the wrong type, empty, a `.gguf` name on other bytes, a
default model file with other bytes, a name already taken, a copy cut short) and checks that nothing is listed after.
Both test the
app's error path, not the native library: a JVM on the build machine cannot load an `arm64-v8a` library, and CI has
no arm64 device. The native library's errors on a phone are checked by hand on a device.
