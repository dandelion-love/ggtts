---
type: Research Finding
title: Threads on the phone
description: On a Galaxy Z Fold5, 4 to 8 CPU threads render at about the same speed within thermal drift, 2 is slower, and the output is bit-identical at every thread count.
tags: [android, performance]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T07:55:00-07:00 }
---
# What was run

On 2026-11-12, `ggtts-render` on the Galaxy Z Fold5 (Snapdragon 8 Gen 2: 1 Cortex-X3, 2 A715, 2 A710, 3 A510) ran
the two-paragraph story titled `Parity 1` at `--threads` 2, 4, 6 and 8, for both engines, back to back. Qwen3-TTS
then ran at 8 and then 4 threads again: the order reversed, on a warmer phone (thermal status 1, battery 36.9 °C).
The scripts are `scripts/android_on_device/threads.sh` and `threads_rev.sh`, beside the other on-device scripts.

# Results

| Engine | Threads | RTF | First chunk ready (s) | Peak RSS |
|---|---|---|---|---|
| Kitten-TTS 2 | 2 | 14.31 | 62.2 | 3.70 GB |
| | 4 | 12.71 | 55.7 | 3.70 GB |
| | 6 | 14.25 | 70.5 | 3.70 GB |
| | 8 | 17.98 | 105.1 | 3.70 GB |
| Qwen3-TTS | 2 | 9.37 | 40.1 | 2.93 GB |
| | 4 | 7.14 | 32.2 | 2.93 GB |
| | 6 | 8.39 | 40.0 | 2.93 GB |
| | 8 | 8.93 | 45.3 | 2.93 GB |
| Qwen3-TTS, reversed and warmer | 8 | 7.41 | | 2.93 GB |
| | 4 | 8.08 | | 2.93 GB |

- **The output is the same at every thread count.** The raw stream's sha256 is identical across 2, 4, 6 and 8
  threads for each engine, and in the reversed runs too. The thread count changes speed only, never the draw.
- **Between 4 and 8 threads, the ranking is within thermal drift.** In the first sweep, 4 threads was fastest and 8
  slowest, but the phone warmed as the sweep went on. Reversed, 8 beat 4. Two threads is slower in both orders
  that include it.

# Decision taken

Keep 4 threads as the default, in the binding and the demo. More threads show no reliable gain, and they keep more
cores busy (more power and heat). The demo lets the user choose 2, 4, 6 or 8 threads.

# What this does not show

One phone, one story, one run per point. A controlled test would let the phone cool between runs and repeat each
point. Pinning work to the big cores (affinity) was not tried. That, and the kernels themselves, are where speed
would come from: thread count is not the lever.
