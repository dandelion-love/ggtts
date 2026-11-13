---
type: Research Finding
title: Our limitations
description: Where ggtts on an Android phone stands against the engines people use, measured on one phone and stated plainly.
tags: [android, limitations]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T11:45:00-07:00 }
stale_after: 2027-05-12T00:00:00-07:00
sources:
  - { id: record, resource: "https://github.com/dandelion-love/ggtts/blob/track/android/docs/android/on-device-2026-11-12.md", title: "On a phone, parity and a resource budget (2026-11-12)" }
  - { id: voice-api, resource: "https://developer.android.com/reference/android/speech/tts/Voice", title: "android.speech.tts.Voice latency and quality levels" }
  - { id: voiceping, resource: "https://voiceping-landingpage-prod.pages.dev/en/blog/research-offline-tts-eval/", title: "VoicePing offline TTS evaluation (Galaxy S10)" }
  - { id: threads, resource: "/research/threads-on-the-phone.md", title: "Threads on the phone" }
---
# The numbers

All measured on a Galaxy Z Fold5 (Android 16, Snapdragon 8 Gen 2), CPU only, through `ggtts-render`.[^record] The
first-chunk times are for the short parity stories: Kitten-TTS 2's from the budget run, Qwen3-TTS's from the thread
sweep. A longer first chunk takes longer: a three-sentence line took 49.9 s in the demo
([streaming](/decisions/stream-playback.md)).
The comparison figures are in [other on-device TTS](/research/android-tts-landscape.md).

| | ggtts, Kitten-TTS 2 Q4_0 | ggtts, Qwen3-TTS q8_0 | What others ship |
|---|---|---|---|
| Real-time factor, 4 threads | 11.8 to 13.6 | 6.7 to 7.2 | 0.06 to 0.14 (the device's system TTS, Piper, Matcha); 1.1 to 2.4 (Kokoro); all on a Galaxy S10[^voiceping] |
| First chunk ready, 4 threads | 47.4 s;[^record] 104 s on `--stream`'s default 5 s level window | 32.2 s[^threads] | no time to first audio published; a whole utterance in 440 ms (system TTS) or 8.2 to 15.3 s (Kokoro), median[^voiceping] |
| Peak memory | 3.78 GB (3.59 GB with the released TQ1_0 LM) | 2.97 to 3.02 GB | not published; models of 40 to 350 MB |
| Download | 2.54 GB | 2.33 GB | 9 MB to 350 MB per voice |
| Heat over a 5-minute read | thermal status 0; battery +3.6 °C | not measured | not published |

For the effect of more threads, see [threads on the phone](/research/threads-on-the-phone.md).

# What follows from them

- **Not for screen readers or anything interactive.** Android's voice latency scale tops out at
  `LATENCY_VERY_HIGH` (over 200 ms).[^voice-api] ggtts's first chunk takes over a hundred times that.
- **Not for long reads in real time.** At an RTF of 7 to 13, a minute of speech takes 7 to 13 minutes to render.
  Streaming lets listening start after the first chunk, but playback then outruns rendering and waits.
- **It needs a phone with headroom.** It needs 3 to 4 GB of free memory, and 2.3 to 2.5 GB of storage per model.
  Beside a heavy app, the system may kill it.
- **No exact match with desktop.** The phone and a desktop CPU give different, equally valid draws for the same
  text and seed. Each machine repeats itself exactly.
- **What it has that others lack:** a neural Hong Kong Cantonese voice offline, voice cloning from a prepared
  reference (Kitten-TTS 2), and the same engines and output format as the desktop build.

# Rating, stated

On a 1 to 5 scale, judged against the shipped engines above:

| | Rating | Why |
|---|---|---|
| Speed | 1 | 50 to 200 times slower than real-time engines |
| First audio | 1 | 30 to 50 s before the first chunk plays; on a Galaxy S10 the system TTS renders a whole utterance in under half a second[^voiceping] |
| Memory and size | 1 | an order of magnitude above the rest |
| Cantonese offline | 4 | one of very few offline Cantonese voices; quality not yet judged by listeners on the phone |
| Integration | 2, and 3 once PR 235 lands | an AAR and a demo app; the system TTS engine that any app can name is in PR 235, not yet merged. Not the phone's default, and no voice-data check screen |

These ratings are this track's own judgement, not user reviews. They should be revisited after each speed or
integration change.
