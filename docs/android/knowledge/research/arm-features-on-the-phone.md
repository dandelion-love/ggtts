---
type: Research Finding
title: ggml's ARM dot-product and matrix features on a phone
description: Building ggml for armv8.6-a with dotprod, i8mm and fp16 makes Kitten-TTS 2's released TQ1_0 LM 1.23x faster on a Galaxy Z Fold5 with byte-identical audio; Q4_0 gains less and changes its draw; the decoder does not change.
tags: [android, kitten-tts-2, performance, arm, ggml]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T15:05:00-07:00 }
sources:
  - { id: tq1, resource: "/research/kitten-memory-and-tq1.md", title: "Kitten-TTS 2's memory, and the released TQ1_0 model on a phone" }
  - { id: survey, resource: "https://github.com/dandelion-love/ggtts/issues/251#issuecomment-6836164200", title: "A read-only survey of TQ1_0's speed on the phone (issue 251)" }
---
# What was compared

The Android build passes no `-march` to ggml, so ggml's ARM code is compiled without the dot-product (`sdot`), 8-bit
matrix (`smmla`, i8mm) and fp16 instructions, although this phone has all three. The survey on issue 251 ranked turning
them on first.[^survey] Two Android `ggtts-render` builds from the same commit (`1b7acba` on `track/android`), differing
only in `GGML_CPU_ARM_ARCH`, which `ggtts-ggml-sys`'s build already passes to ggml's CMake:

| Build | `GGML_CPU_ARM_ARCH` | `sdot` in the binary | `smmla` in the binary |
|---|---|---|---|
| base | unset (ggml's default) | 0 | 0 |
| armv86 | `armv8.6-a+dotprod+i8mm+fp16` | 923 | 244 |

Galaxy Z Fold5 (Android 16), CPU, 4 threads. Kitten-TTS 2 with the base Q4_0 LM and the released Cantonese TQ1_0 LM,
with the same S3Gen, voice, two-sentence Cantonese story and seed as the [TQ1_0 measurement](/research/kitten-memory-and-tq1.md),
and `--stream-raw`.[^tq1] Order per model: base, armv86, armv86, base, a minute apart (`armfeat-measure.sh`). Raw
files: `docs/android/data/armfeat-2026-11-12/`.

# Results

| Run (order) | Audio | Frames | First audio | RTF | LM frames/s | Decoder s per audio s | Raw stream sha256 |
|---|---|---|---|---|---|---|---|
| base, Q4_0 (1) | 9.44 s | 224 | 36.6 s | 9.15 | 3.59 | 2.54 | `db8b3118…` |
| armv86, Q4_0 (2) | 9.14 s | 216 | 32.1 s | 8.23 | 4.23 | 2.63 | `c2202c0f…` |
| armv86, Q4_0 (3) | 9.14 s | 216 | 45.9 s | 10.48 | 3.19 | 3.08 | `c2202c0f…` |
| base, Q4_0 (4) | 9.44 s | 224 | 39.1 s | 10.97 | 3.02 | 3.13 | `db8b3118…` |
| base, TQ1_0 (5) | 9.86 s | 235 | 53.5 s | 13.69 | 2.25 | 3.08 | `52072e62…` |
| armv86, TQ1_0 (6) | 9.86 s | 235 | 47.0 s | 11.97 | 2.70 | 3.15 | `52072e62…` |
| armv86, TQ1_0 (7) | 9.86 s | 235 | 49.8 s | 12.07 | 2.70 | 3.25 | `52072e62…` |
| base, TQ1_0 (8) | 9.86 s | 235 | 59.9 s | 14.34 | 2.16 | 3.32 | `52072e62…` |

Means of each pair of runs:

| | LM frames/s, base to armv86 | RTF | First audio |
|---|---|---|---|
| TQ1_0 (released) | 2.20 to 2.70, **1.23x** | 14.0 to 12.0 | 56.7 s to 48.4 s |
| Q4_0 | 3.31 to 3.71, 1.12x | 10.1 to 9.4 | 37.9 s to 39.0 s |

- **The released TQ1_0 model: 1.23x on the LM, and the same audio.** All four TQ1_0 runs give the same raw stream,
  byte for byte, from either build. Both armv86 runs are faster than both base runs, the first of which ran earlier
  and cooler, so this is not warming.
- **Q4_0: a smaller gain, and a different draw.** The armv86 build gives a different raw stream (216 frames against
  224), the same on both of its runs. Its dot-product kernel adds the per-block products in a different floating-point
  order, which changes a sampled token, so the read takes another valid path. The 1.12x is within this session's drift:
  the second armv86 run was slower than the first base run.
- **The decoder does not change** (about 2.5 to 3.3 s per second of audio either way), as expected: its weights are F32.
- **Peak memory does not change** (3.78 GB with Q4_0, 3.59 GB with TQ1_0).
- **Smaller than the survey's estimate** of 1.3x to 2x for the LM.[^survey] The LM step also includes the
  158,241 by 2048 bf16 output head, whose dot product has no NEON path in ggml at the pin, so these instructions do not
  touch it.

# What it means for a release

- An armv8.6-a build crashes, with an illegal instruction, on a CPU without i8mm. That rules out phones before about
  2022 (Cortex-A510, A710 and X2 onward have it). Dot product alone (armv8.2-a+dotprod) covers most phones in use, but
  was not measured here. So the library cannot simply be rebuilt with these flags: it needs variants chosen at run time
  (ggml can build its CPU backend as several variants and pick one by the CPU's features) or a check that refuses to
  load the wrong one.
- For TQ1_0 the audio is unchanged, so the parity record holds. For Q4_0 the read changes, as a different draw would;
  parity is distributional there already.

Not measured: dot product without i8mm; the repacked Q4_0 kernels (which need these flags, and weights in the repack
buffer type); the output head on its own.
