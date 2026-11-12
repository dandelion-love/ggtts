---
type: Research Finding
title: Other on-device TTS on Android
description: The on-device text-to-speech engines people use on Android, with size, Cantonese support, system-engine status, speed, licence and Play ratings read on 2026-11-12.
tags: [android, research, competition]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:40:00-07:00 }
stale_after: 2027-05-12T00:00:00-07:00
sources:
  - { id: voiceping, resource: "https://voiceping-landingpage-prod.pages.dev/en/blog/research-offline-tts-eval/", title: "VoicePing offline TTS evaluation (Galaxy S10, 4 threads, Feb 2026)" }
  - { id: play-google, resource: "https://play.google.com/store/apps/details?id=com.google.android.tts", title: "Speech Recognition & Synthesis on Google Play" }
  - { id: samsung-yue, resource: "https://sysapps.bayton.org/packages/com.samsung.SMT.lang_zh_hk_f00", title: "SamsungTTS Cantonese Voice 1 package" }
  - { id: espeak-fdroid, resource: "https://f-droid.org/en/packages/com.reecedunn.espeak/", title: "eSpeak NG on F-Droid" }
  - { id: espeak-langs, resource: "https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md", title: "eSpeak NG languages" }
  - { id: rhvoice-play, resource: "https://play.google.com/store/apps/details?id=com.github.olga_yakovleva.rhvoice.android", title: "RHVoice on Google Play" }
  - { id: sherpa-apk, resource: "https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html", title: "sherpa-onnx TTS engine APKs" }
  - { id: sherpa-hf, resource: "https://huggingface.co/csukuangfj/sherpa-onnx-apk/tree/main/tts-engine-new", title: "sherpa-onnx engine APK sizes" }
  - { id: sherpatts, resource: "https://f-droid.org/en/packages/org.woheller69.ttsengine/", title: "SherpaTTS on F-Droid" }
  - { id: voxsherpa-play, resource: "https://play.google.com/store/apps/details?id=com.CodeBySonu.VoxSherpa", title: "VoxSherpa TTS on Google Play" }
  - { id: voxsherpa-gh, resource: "https://github.com/CodeBySonu95/VoxSherpa-TTS", title: "VoxSherpa TTS README" }
  - { id: marmalade, resource: "https://f-droid.org/en/packages/app.marmalade.tts/", title: "Marmalade TTS on F-Droid" }
  - { id: supertonic-play, resource: "https://play.google.com/store/apps/details?id=com.brahmadeo.supertonic.tts", title: "Supertonic TTS on Google Play" }
  - { id: tobesaid-play, resource: "https://play.google.com/store/apps/details?id=ai.lookbe.tts", title: "ToBe SAID on Google Play" }
  - { id: vocalizer-play, resource: "https://play.google.com/store/apps/details?id=es.codefactory.vocalizertts", title: "Vocalizer TTS on Google Play" }
  - { id: kokoro, resource: "https://huggingface.co/hexgrad/Kokoro-82M", title: "Kokoro-82M model card" }
---
# Method

A research pass on 2026-11-12, cross-checked where marked:
- Ratings and install counts come from each public Google Play listing (en-US), read that day.
- Sizes and licences come from F-Droid listings and project pages.
- Speeds come from one published benchmark on a Galaxy S10 (4 threads) and from developers' own tables.

The Play ratings for Google, RHVoice, VoxSherpa, Supertonic, ToBe SAID and Vocalizer were re-read from the saved
pages before this was written. "Unverified" means no source was found, not that the thing is absent.

The package id `com.k2fsa.sherpa.onnx.tts.engine` on Play belongs to an unrelated app. sherpa-onnx itself
distributes its engine APKs on GitHub and Hugging Face.

# Comparison

| Engine | Model or download | Cantonese offline | System TTS engine | Speed (RTF; lower is faster) | Licence | Rating (Play, 2026-11-12) |
|---|---|---|---|---|---|---|
| Google Speech Recognition & Synthesis | unverified | Cantonese (HK) is supported; an offline voice is unverified | yes, the default | 0.058, first audio 440 ms[^voiceping] | proprietary | 4.0, 4.37M reviews, 10B+ installs[^play-google] |
| Samsung TTS | Cantonese voice pack about 69.5 MB[^samsung-yue] | yes, Galaxy only | yes | unverified | proprietary | preinstalled; not on Play |
| eSpeak NG | 9 MiB APK[^espeak-fdroid] | lists `yue`, quality undocumented[^espeak-langs] | yes | unverified (formant synthesis, fast) | GPL-3.0+ | F-Droid only |
| RHVoice | 15 MiB plus voices | no | yes | "talks instantly"; no figure | GPL-3.0+ | 3.1, 1.84K reviews, 100K+ installs[^rhvoice-play] |
| sherpa-onnx engine APKs (Piper, VITS, Matcha, Kokoro) | 41 to 343 MiB per APK; Cantonese VITS 121 MB[^sherpa-hf] | yes, a VITS Cantonese APK[^sherpa-apk] (model licence unverified) | yes | Piper 0.077, Matcha 0.135, Kokoro 1.1 to 2.4[^voiceping] | Apache-2.0 code, per-model licences | GitHub and Hugging Face only |
| SherpaTTS | 86 MiB plus a model[^sherpatts] | unverified | yes | unverified | GPL-3.0 | F-Droid only |
| VoxSherpa TTS (Kokoro, Piper, Supertonic, cloning) | about 500 MB free recommended[^voxsherpa-gh] | no | yes | Kokoro 20 to 40 s per minute of audio on a flagship (RTF about 0.3 to 0.7)[^voxsherpa-gh] | GPL-3.0 | 4.0, 113 reviews, 10K+ installs[^voxsherpa-play] |
| Marmalade TTS (Kokoro, Pocket TTS) | 106 MiB plus on-demand engines[^marmalade] | no | yes | unverified | GPL-3.0+ | F-Droid only |
| Supertonic TTS | unverified | unverified | yes | "minimal latency" (no figure) | unverified | 4.7, 77 reviews, 5K+ installs[^supertonic-play] |
| ToBe SAID | English built in, more downloadable | unverified | yes | unverified | unverified | 4.5, 68 reviews, 5K+ installs[^tobesaid-play] |
| Code Factory Vocalizer | paid voice packs | yes, Cantonese (HK) for purchase | yes | unverified | proprietary | 2.8, 9.31K reviews, 1M+ installs[^vocalizer-play] |
| **ggtts (this track)** | **2.3 to 2.5 GB, user-supplied** | **yes: a Hong Kong Cantonese Qwen3-TTS voice** | **no (an app library and demo)** | **7 (Qwen3-TTS) and 12 (Kitten-TTS 2) on a Galaxy Z Fold5** | Apache-2.0 code; models' own licences | not published |

# What it shows

- **No shipping Android TTS runs a model of 1B parameters or more.** Kokoro-82M is the largest in common use, and
  on a Galaxy S10 it already runs slower than real time.[^voiceping][^kokoro] ggtts's engines are 1.4B (Kitten-TTS 2,
  ternary) and 1.7B (Qwen3-TTS) parameters.
- **Offline Cantonese is scarce.** It is confirmed only in Samsung's OEM pack, Code Factory's paid voice and
  sherpa-onnx's VITS APK. eSpeak NG lists `yue`, with undocumented quality.
- **Every engine with a published speed gives first audio in under a second.** ggtts on a phone takes tens of
  seconds ([limitations](/research/limitations.md)).
- **Almost every app is a system TTS engine** (an Android `TextToSpeechService`), so screen readers and other apps
  can use it. ggtts on Android is not one yet.
