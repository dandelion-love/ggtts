---
type: Pattern
title: UX patterns from other TTS apps
description: The five patterns from shipped Android TTS apps worth copying for a slow, large-model engine, and what this track did or filed for each.
tags: [android, ux]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:50:00-07:00 }
sources:
  - { id: callback, resource: "https://developer.android.com/reference/android/speech/tts/SynthesisCallback", title: "android.speech.tts.SynthesisCallback" }
  - { id: voice-api, resource: "https://developer.android.com/reference/android/speech/tts/Voice", title: "android.speech.tts.Voice" }
  - { id: sherpatts, resource: "https://github.com/woheller69/ttsEngine", title: "SherpaTTS README" }
  - { id: voxsherpa-gh, resource: "https://github.com/CodeBySonu95/VoxSherpa-TTS", title: "VoxSherpa TTS README" }
  - { id: rhvoice-play, resource: "https://play.google.com/store/apps/details?id=com.github.olga_yakovleva.rhvoice.android", title: "RHVoice on Google Play" }
  - { id: marmalade, resource: "https://f-droid.org/en/packages/app.marmalade.tts/", title: "Marmalade TTS on F-Droid" }
---
# The patterns, ranked for a slow engine

Drawn from the apps in [other on-device TTS](/research/android-tts-landscape.md). The ranking is for an engine
whose first audio takes tens of seconds ([limitations](/research/limitations.md)).

| # | Pattern | Who does it | This track |
|---|---|---|---|
| 1 | **Stream audio as it is ready, and tell the system you are slow.** Android's engine API streams by design: `start()`, `audioAvailable()` per piece, then `done()`.[^callback] Voices declare a latency level.[^voice-api] | the platform API; Marmalade's Pocket TTS[^marmalade] | **Done in the demo:** [chunk-by-chunk playback with Stop](/decisions/stream-playback.md). **Filed:** the system engine with `LATENCY_VERY_HIGH`, [#221](https://github.com/dandelion-love/ggtts/issues/221) |
| 2 | **A fast voice works at first launch; the big model is an opt-in download.** | ToBe SAID (English built in), Marmalade | **Not adopted.** ggtts has no small engine yet. Revisit when one exists |
| 3 | **Keep the model warm behind a foreground service, with a notification and Stop.** | SherpaTTS (preload after boot)[^sherpatts], VoxSherpa (media notification)[^voxsherpa-gh], RHVoice | **Filed:** [#222](https://github.com/dandelion-love/ggtts/issues/222) |
| 4 | **Try before committing storage, and import user models with clear rules.** | VoxSherpa (previews, `.onnx` import), SherpaTTS (sideloading) | **Done:** a checked import (type, size, published hash, `.part` then rename; [decision](/decisions/models-are-brought-not-bundled.md)). Previews are not adopted: no published samples to point at yet |
| 5 | **Say up front what it cannot do.** Name the missing languages, the speed class, and "not for screen readers". | RHVoice's listing names the languages it lacks[^rhvoice-play]; Vocalizer warns about apps that ignore third-party engines | **Done** ([#223](https://github.com/dandelion-love/ggtts/issues/223)): the screen states each engine's size, memory and wait on a named phone, and "not suitable for screen readers"; low memory [warns before a load](/decisions/warn-on-low-memory.md) |

# Also common, and lower in priority

- Speed and pitch come from the system TTS settings once an app is a system engine, so they follow from #221.
- An app shows the licences of its models. The AAR already carries its code and crate licences under
  `META-INF/ggtts/`. Model licences are the user's to read, because the user brings the model.
