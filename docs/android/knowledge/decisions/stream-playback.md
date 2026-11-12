---
type: Decision
title: The demo streams audio chunk by chunk
description: The demo plays each chunk as soon as the binding hands it over, with a Stop button, instead of rendering the whole read before playing anything.
tags: [android, ux, demo]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:50:00-07:00 }
sources:
  - { id: callback, resource: "https://developer.android.com/reference/android/speech/tts/SynthesisCallback", title: "android.speech.tts.SynthesisCallback" }
---
# Decision

The demo renders through the binding's `renderStream`. Each chunk is queued to an AudioTrack on its own thread as
soon as it is ready, so playback never holds up rendering. Stop silences playback at once, and the render ends at
the next chunk. The screen shows the time to first audio and which chunk is ready. The whole read is still saved
as a WAV when it finishes.

# Why

On the phone, a chunk takes tens of seconds. Rendering the whole read before playing makes the first sound wait
for every chunk: 104 s for the 5-minute read, against 47 s for its first chunk
([measured](/research/on-device-parity-and-budget.md)). Every shipped engine streams, and Android's engine API is
built around it.[^callback] This is the first of the [UX patterns](/patterns/ux-patterns-from-other-tts-apps.md).

# Measured on the phone

Qwen3-TTS at 4 threads read a three-sentence Cantonese line that rendered in 2 chunks. The first audio came at
49.9 s and 50.7 s in two runs, against 78.2 s and 78.9 s for the whole line, so streaming cut the wait by about
28 s. Stop pressed 3 s after first audio ended the render when the chunk in progress finished, 22.8 s later. The
app stayed up.

# Trade-off

The streamed pieces are the render's samples before its final level step, so their loudness can differ a little
from the saved file's. A read whose rendering is slower than real time also pauses between chunks while the next
one renders. Both are accepted for a demo.

# Undo

`DemoController.run` used `render`, then `play`; reverting the streaming PR restores that.
