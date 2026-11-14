---
type: Research Finding
title: What a store app would need that this one lacks
description: The demo read as a Play Store listing would be, gap by gap and ranked, with what each gap says about the voice models it runs.
tags: [android, store, ux, product]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-13T18:55:00-07:00 }
stale_after: 2027-05-13T00:00:00-07:00
sources:
  - { id: landscape, resource: "/research/android-tts-landscape.md", title: "Other on-device TTS on Android" }
  - { id: limits, resource: "/research/limitations.md", title: "Our limitations" }
  - { id: patterns, resource: "/patterns/ux-patterns-from-other-tts-apps.md", title: "UX patterns from other TTS apps" }
  - { id: process-text, resource: "https://developer.android.com/reference/android/content/Intent#ACTION_PROCESS_TEXT", title: "Intent.ACTION_PROCESS_TEXT" }
  - { id: media-session, resource: "https://developer.android.com/media/media3/session/control-playback", title: "Control and advertise playback using a MediaSession" }
  - { id: data-safety, resource: "https://support.google.com/googleplay/android-developer/answer/10787469", title: "Google Play Data safety section" }
---
# How it was read

The demo app on a Galaxy Z Fold5 (Android 17), opened as a new user would, and read against the apps in the
landscape[^landscape] and the patterns already taken from them.[^patterns] No submission is planned. The point of
reading it as a store app is that a real user's task decides what matters in the voice, which a one-line demo hides.

Before this finding the screen opened on an engine choice (`qwen3-tts`, `kitten-tts-2`), three file pickers and a
speaker id. Nothing said what the app does for a person. It now opens with one line and three sentences: what it
does, what it costs, and what it is not for (`About.kt`).

# The gaps, ranked

| # | Gap | What store apps do | What it would take here |
|---|---|---|---|
| 1 | **No stated purpose.** | One line under the name; the store's short description is the same line. | **Done:** `About.kt`, held to the manifest and the measured costs by `AboutTest`. [#319](https://github.com/dandelion-love/ggtts/issues/319) |
| 2 | **No way to get a voice inside the app.** A new user needs 2.3 to 2.5 GB of files from elsewhere, pushed or imported by hand. | RHVoice, SherpaTTS and Google list voices with their size and an Install button. | A voice list with size, a Wi-Fi-only resumable download, and the import's existing checks. It costs the "no internet permission" sentence, which must then say what the network is used for. [#320](https://github.com/dandelion-love/ggtts/issues/320) |
| 3 | **The choices are the engine's parts, not voices.** Engine, model, codec, voice file and speaker id are five choices; the pickers start at "(none)" even with files present. | One choice: a named voice, with a sample to play. | A voice is a bundle of files with a name, a language and a sample. Engine, threads and file pickers move under "Advanced". [#321](https://github.com/dandelion-love/ggtts/issues/321) |
| 4 | **Text has one way in: typing.** | Readers take shared text, a "Read aloud" item in the text-selection menu,[^process-text] and opened files. | `ACTION_SEND` for `text/plain`, `ACTION_PROCESS_TEXT`, and opening a `.txt` (and EPUB, which has its own track). [#322](https://github.com/dandelion-love/ggtts/issues/322) |
| 5 | **The result can only be heard once, live.** The read is saved to a file the user cannot reach. | Reader apps keep a library and resume where you stopped. | Given the speed, the honest product is "make audio to listen to later": a queue of reads rendered in the background, kept as files, shareable. [#323](https://github.com/dandelion-love/ggtts/issues/323) |
| 6 | **Playback has Stop only.** | Pause, resume, seek, lock-screen and headset controls through a media session.[^media-session] | A media session on the reading service, which already plays in the foreground. [#324](https://github.com/dandelion-love/ggtts/issues/324) |
| 7 | **Store hygiene.** | An icon, a theme with dark mode, text in the user's language, TalkBack labels, a licences screen, a privacy statement, a Data safety form.[^data-safety] | The default icon, unthemed views, English-only text, no licences screen. The package id `ggtts.demo` would be permanent on a store, so choose it before any listing. The Data safety answer is "no data collected" while the app has no internet permission. [#325](https://github.com/dandelion-love/ggtts/issues/325) |
| 8 | **Real text is untested.** | Not visible in a listing, but reviewers paste real text. | Numbers, dates, prices, English words inside Cantonese, and written Chinese against colloquial Cantonese, all read through each engine. [#326](https://github.com/dandelion-love/ggtts/issues/326) |

The speed itself is not a gap a screen can close: it is stated, measured, in the purpose text and the
limitations.[^limits]

# What the gaps say about the models

Modelling the real task changes which model properties count.

- **Long-form steadiness outranks first-audio latency.** At 7 to 13 seconds of rendering per second of
  speech,[^limits] the useful task is a passage rendered and listened to later (gaps 4 to 6). A voice that drifts,
  runs on, or changes character between chunks fails that task more than a slow start does.
- **Real text is mixed.** A passage someone shares from a news site or chat has numbers, dates, prices, English
  names and both written and colloquial forms (gap 8). How a model reads those decides whether a listener trusts it.
- **People choose a voice by hearing it.** A named voice with a sample (gap 3) makes the voice's character the thing
  a user picks or rejects in seconds.
- **Size is a model property.** A 2.5 GB download is the first thing a store user meets (gap 2), before any quality.
  A smaller model is worth more here than a small quality gain.

# Undo

This is a reading, not a decision. The purpose text is one file, `About.kt`, and its test.
