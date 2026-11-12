---
okf_version: "0.2"
---
# ggtts on Android: what we know and what we decided

The Android track's knowledge, as an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge-format)
bundle. Plans and work items are GitHub issues, not concepts here. Each decision says how to undo it.

## Decisions

* [Android work lands on one integration branch](/decisions/integration-branch.md) - every Android change goes into `track/android` by reviewed PR, and reaches `rc` as one PR
* [Native libraries are aligned for 16 KB pages](/decisions/16kb-page-alignment.md) - link with max-page-size 16384 and refuse a library aligned below it
* [No model ships; imports are checked](/decisions/models-are-brought-not-bundled.md) - the user supplies model files; the app checks type, size and published hashes on the way in
* [The demo streams audio chunk by chunk](/decisions/stream-playback.md) - play each chunk as it is ready, with Stop, instead of rendering the whole read first
* [Measure on the device through ggtts-render](/decisions/measure-through-the-cli.md) - parity and budget runs use the command-line renderer over adb, not the app
* [Warn, do not refuse, on low memory](/decisions/warn-on-low-memory.md) - below the measured peak, the first Speak warns and the second loads

## Research

* [On-device parity and resource budget](/research/on-device-parity-and-budget.md) - a Galaxy Z Fold5 against desktop CPU: no exact parity, budget fails on memory, speed and first audio
* [Threads on the phone](/research/threads-on-the-phone.md) - what more CPU threads buy for each engine
* [Other on-device TTS on Android](/research/android-tts-landscape.md) - Google, Samsung, eSpeak NG, RHVoice, sherpa-onnx and the Kokoro apps, with Play ratings and speeds
* [Our limitations](/research/limitations.md) - where ggtts on a phone stands against those engines, stated plainly

## Patterns

* [UX patterns from other TTS apps](/patterns/ux-patterns-from-other-tts-apps.md) - the five worth copying for a slow, large-model engine, and what this track did with each
