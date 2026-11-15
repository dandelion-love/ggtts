---
type: Research Finding
title: How read-aloud apps lay out their screens
description: Ten Android read-aloud apps and TTS engines, screen by screen, and the layout this app should take from them; the answer to "it looks like a list of widgets".
tags: [android, ux, design]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-14T20:20:00-07:00 }
stale_after: 2027-05-14T00:00:00-07:00
sources:
  - { id: speechify, resource: "https://help.speechify.com/en/collections/3001642-speechify-for-android", title: "Speechify for Android, help centre" }
  - { id: speechify-voice, resource: "https://help.speechify.com/en/articles/5336931-how-to-change-your-listening-voice-or-language", title: "Speechify: change your listening voice" }
  - { id: eleven, resource: "https://elevenlabs.io/docs/help-center/product/mobile-apps/elevenreader/how-to-guides-for-elevenreader/how-to-use-elevenreader-on-android", title: "ElevenReader on Android" }
  - { id: eleven-reader, resource: "https://elevenlabs.io/text-reader", title: "ElevenReader" }
  - { id: atvoice, resource: "https://hyperionics.com/atVoice/manual.html", title: "@Voice Aloud Reader manual" }
  - { id: chrome, resource: "https://support.google.com/chrome/answer/14768725?hl=en", title: "Chrome: listen to web pages" }
  - { id: chrome-9to5, resource: "https://9to5google.com/2024/06/16/chrome-listen-tts-android/", title: "9to5Google on Chrome's Listen to this page" }
  - { id: edge, resource: "https://www.makeuseof.com/how-to-set-up-microsoft-edge-read-aloud/", title: "Edge Read Aloud set-up" }
  - { id: rhvoice, resource: "https://rhvoice.com/android_support", title: "RHVoice Android guide" }
  - { id: sherpatts, resource: "https://github.com/woheller69/ttsEngine", title: "SherpaTTS" }
  - { id: sherpa-onnx, resource: "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/android/SherpaOnnxTtsEngine/app/src/main/java/com/k2fsa/sherpa/onnx/tts/engine/MainActivity.kt", title: "sherpa-onnx TTS engine app, MainActivity" }
  - { id: google-voice, resource: "https://www.nextpit.com/how-to-fix-speech-services-waiting-for-network-connection", title: "Speech Services voice data downloads" }
---
# What was looked at, and its limits

This survey is drawn from vendor help pages, F-Droid, source code and press coverage, read on 2026-11-14. The Play
listings returned no text and no screenshot captions. So this records what each app *does* and *where it puts it*,
not colours or exact styling. No other app was opened on the test phone.

# The apps

| App | Home and getting text in | While reading | Choosing a voice | Waiting |
|---|---|---|---|---|
| Speechify[^speechify] | A library with "+": paste, PDF, URL, camera, photos | A listening screen with themes and fonts | A voice icon in the player opens a searchable list with previews[^speechify-voice] | not documented |
| ElevenReader[^eleven] | "+": write, scan, link, file, web page | Words highlighted with the audio, speed up to 4x, sleep timer[^eleven-reader] | A library of voices, recommended by language | An offline download is a job: "about 2-5 minutes" |
| @Voice Aloud Reader[^atvoice] | Share menu, clipboard, files, reading list | Bottom bar: play/pause, previous/next sentence; current sentence highlighted; double-tap a sentence to start there | The system engine's voices | not documented |
| Chrome "Listen to this page"[^chrome] | One menu item or toolbar button | A docked mini-player (title, progress, play/pause) that expands to a scrubber, ±10 s and a "1.0x" chip[^chrome-9to5] | A sheet of voices described in words ("mid-pitch, warm"), each with Preview | not documented |
| Edge Read Aloud[^edge] | Menu item | A toolbar with a gear for speed and voice | A few offline voices, the rest online | not documented |
| Google / Samsung speech services[^google-voice] | Settings only | n/a | Language, then voices, with an install and a delete button per voice | Downloads can stall on "Waiting for network" |
| RHVoice[^rhvoice] | A list of languages | n/a | Language → voices, each with a sample to play and an install or uninstall button; a quality tier (standard, best quality, best speed) | n/a |
| SherpaTTS[^sherpatts] | "Manage languages" | n/a | Picking a language downloads its model; files can be installed by hand | Preloads the model after the phone starts |
| sherpa-onnx engine app[^sherpa-onnx] | A text box with example text, a speed slider, a speaker number | Start, Play, Stop, Save, Share buttons; then a line with threads, time and RTF | One model per app | **None**: Start greys out until the audio is made |

The last row is closest to this app today: stock widgets stacked in a column, numbers printed when a run ends.
It is the shape to move away from.

# What the reader apps share

They are built around **a player, not a form**. Ranked for an app whose first sound takes 12 to 60 s:

1. **The wait is a job with a stage and an estimate,** not a frozen button (ElevenReader). Show "Preparing the
   voice", then "Sentence 1 of 9", with an estimate taken from this phone's own past runs, and Stop.
2. **The model is loaded before Speak** (SherpaTTS preloads). Load when the app opens or a voice is chosen, and
   show "Voice ready".
3. **Reading goes sentence by sentence, with the current sentence highlighted** (@Voice, Chrome, ElevenReader). This
   app already streams by chunk, so the text box can become the reading view: highlight the chunk playing and dim
   the ones not yet made.
4. **A docked player, with a media notification when the app is left** (Chrome, @Voice). It has play/pause, the
   previous and next sentence, a progress line and a speed chip ("1.0x"), and it replaces the Speak and Stop pair.
5. **Getting text in is one step:** a "+" or the share sheet, with a paste button on an empty box. This is #320's
   "Read aloud" from other apps.
6. **The voice is picked from a sheet of cards,** each with its language, a short description and a
   **pre-rendered** sample, so a preview never costs a render (Chrome, RHVoice). Voices are named in words, not
   by model ids.
7. **A library of past texts,** with the audio kept so a replay is instant (Speechify, ElevenReader).
8. **Dark mode and reading themes,** which are expected but do nothing for the wait (done in PR 372).

What only the on-device engines do: a language → voices → install hierarchy with a sample per voice, and a
per-voice quality-or-speed tier (RHVoice). No app surveyed shows a voice's size, that it works offline, or its
time to the first sound. This app can show all three, because it measures them.

# What this means for this app's next screen

The order in [PR 372](https://github.com/dandelion-love/ggtts/pull/372) (voice, text, Speak, progress) is a form.
The next step is a reading screen: the text as the main surface, a docked player at the bottom (play/pause,
previous and next sentence, speed, progress with the stage), and the voice as a card that opens a sheet. About,
Import and Advanced go into an overflow menu. Items 3, 4 and 6 above are the ones to build first, and item 6 needs
the per-voice language and display name of [#366](https://github.com/dandelion-love/ggtts/issues/366).
