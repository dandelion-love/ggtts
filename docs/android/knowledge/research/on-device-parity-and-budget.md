---
type: Research Finding
title: On-device parity and resource budget
description: On a Galaxy Z Fold5, ggtts gives different draws from desktop CPU for the same request, and the Kitten-TTS 2 budget fails on memory, speed and first audio.
tags: [android, parity, performance]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T11:45:00-07:00 }
sources:
  - { id: record, resource: "https://github.com/dandelion-love/ggtts/blob/track/android/docs/android/on-device-2026-11-12.md", title: "On a phone, parity and a resource budget (2026-11-12)" }
---
# Finding

The full record, with method and per-run tables, is `docs/android/on-device-2026-11-12.md`.[^record] In short:

- **Exact parity fails.** Phone against desktop CPU differs on 7 of 7 draws for both engines. Each machine repeats
  itself bit for bit, so the difference is arithmetic. The stop probability already differs at frame 1
  (`7.0e-10` against `0.0`), and Qwen3-TTS's tokens part at frame 2 to 10.
- **Distribution parity on frame counts holds at six draws a side.** Exact permutation p is 0.22 for Qwen3-TTS
  and 1.0 for Kitten-TTS 2. Kitten-TTS 2's frames per chunk match within 2 frames, draw for draw.
- **The budget for Kitten-TTS 2 at four threads:**

| Line | Budget | Measured | Result |
|---|---|---|---|
| Peak RSS | 3.0 GB | 3.78 GB | fail |
| Real-time factor | 1.0 | 11.8 | fail |
| First audio | 2.0 s | 103.9 s | fail |
| Thermal status | at most 1 | 0 | pass |
| Battery temperature rise | 8 °C | +3.6 °C | pass |

# Consequence for the track

The parity level is distributional. A phone render is checked against desktop by its frame-count distribution, not
by its tokens. The budget fails are this track's [limitations](/research/limitations.md), and they set what to
work on next: first audio (streaming, [decided](/decisions/stream-playback.md)), threads
([measured](/research/threads-on-the-phone.md)), then the kernels themselves.
