---
type: Research Finding
title: Where a Kitten-TTS 2 generation step goes, and a head that computes only the samplable rows
description: TQ1_0 is slower than Q4_0 on a phone because of its kernel, not its padding; the bf16 head costs about as much as all the layers there; computing the head only from the first end id to the end of the vocabulary is exact on 44 test lines (top-k margin at least 4.3) and about 1.2x faster on the phone and desktop.
tags: [kitten-tts-2, performance, profiling, tq1_0, android]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T17:10:00-07:00 }
sources:
  - { id: arm, resource: "/research/arm-features-on-the-phone.md", title: "ggml's ARM dot-product and matrix features on a phone" }
  - { id: tq1, resource: "/research/kitten-memory-and-tq1.md", title: "Kitten-TTS 2's memory, and the released TQ1_0 model on a phone" }
---
# Method

Tools in `crates/ggtts-kitten/examples/`, at 4 threads; raw outputs in `docs/android/data/kitten-step-profile-2026-11-12/`.

- `matmul_bench`: one-token products at the LM's shapes (28 per graph, as a step has them) per storage type, with
  random exact-ternary weights packed by the converter's own packer; and the output head in bf16 and f16.
- `op_profile`: times every node of the engine's own graph through ggml's scheduler evaluation callback. Observing
  makes each node its own split, which costs a fixed amount per node: about 0.1 ms on desktop, but about 1 ms on the
  phone, where it dominates (an observed step took about 1 s against about 0.4 s unobserved). So on the phone it is not
  a cost breakdown; `matmul_bench` is.
- `head_rows_check`: runs a 22-line test set (`fixtures/head_rows_lines.txt`, English and Cantonese) at the default seed
  with the whole head, and measures at every step the **margin**: the 50th-largest score (after the penalties and
  temperature, before top-k and top-p) minus the largest score of any id outside the head's rows.

Desktop: WSL Ubuntu on an i9-12900K, shared with other work, so every comparison that decides something was
interleaved. Phone: Galaxy Z Fold5 (Android 16), with the default ggml build and the armv8.6-a one
([ARM features](/research/arm-features-on-the-phone.md)).[^arm]

# Why TQ1_0 is slower: its kernel

One-token products for all 28 layers, per step, median of 30 (`matmul_bench`; two runs per build, base, armv86,
armv86, base):

| | Phone, default build | Phone, armv8.6-a build | Desktop |
|---|---|---|---|
| Padded TQ1_0 (the released LM) | 127 to 134 ms | 81 to 86 ms | 33 ms |
| Padded TQ2_0 | 84 to 88 ms | 53 to 54 ms | 31 ms |
| Q4_0 | 81 to 83 ms | 47 to 50 ms | 35 ms |
| Output head, bf16 | 82 to 93 ms | 85 to 88 ms | 23 ms |
| Output head, f16 | 30 to 33 ms | 17 to 18 ms | 22 ms |

- **The padding is not the cause.** Padded TQ2_0 does the same doubled work (one 128-group plus 128 zeros per block) and
  runs within 6% of Q4_0 on the phone. TQ1_0's products take 1.6 to 1.7 times Q4_0's there: its kernel unpacks base-3
  trits, which costs this ARM CPU far more than desktop, where the three types are level. That explains the 24 to 35%
  slower LM step measured on the phone.[^tq1]
- **On the phone the bf16 head costs about as much as all the layers** (82 to 93 ms a step). ggml has no NEON path for
  a bf16 dot product, so neither build helps it; the same table in f16 takes a third to a fifth of the time.
- What follows for the released model on Android: storing its ternary matrices as padded TQ2_0 (727 MB, against 595 MB
  as TQ1_0) would make them run at Q4_0's speed, at 132 MB more. Not done here.

# Where a desktop step goes (`op_profile`)

| Per one-token step | Q4_0 LM | Released TQ1_0 LM |
|---|---|---|
| Layer matrix products | 26.5 ms | 30.0 ms |
| Output head (bf16) | 16.2 ms | 16.1 ms |
| `CONT` | 26.3 ms | 27.6 ms |
| `PAD` (padding the input to 2 x 128 per group) | none | 2.5 ms |
| Everything else | 4.2 ms | 4.3 ms |

`CONT` is about a third of a step: two per layer, the large one the whole V cache transposed and copied on every token
(`ggml_cont(ggml_permute(vall, …))` in `Session::forward`). It grows with the context. Filed as issue 257.

# The head, restricted to the rows a step can need

A step keeps a frame for an audio id and stops on `speech_end` or `stop`, but the sampler draws over the whole
vocabulary, as both references do. So computing only some rows is exact for a step when no id outside them reaches the
top-k set.

**First range: the audio ids and the two end ids (6,566 rows). Not exact.** At 14 of 7,636 steps (5 with TQ1_0, 9 with
Q4_0) one id outside it was in the top-50, by up to 5.7: always id 158,238, `reference_speech_end_id`, the marker that
closes the reference's speech in the prompt, and always at the last step of a line, beside `speech_end`. Top-p then
removed it, so the tokens still matched, but that is luck, not exactness.

**Second range: from the first end id to the end of the vocabulary (151,670 to 158,241: the end ids, the audio ids, and
the five special ids after them; 6,571 rows). Exact on the test set.**

| | Released TQ1_0 LM | Q4_0 LM |
|---|---|---|
| Steps over the 22 lines | 3,923 | 3,713 |
| Smallest margin (50th score minus the best id outside the rows) | **+4.28** | **+6.77** |
| Steps with an id outside the rows in the top-50 | **0** | **0** |
| Lines with identical tokens, whole head against restricted | **22 of 22** | **22 of 22** |
| Desktop step, interleaved (whole, restricted, restricted, whole; 4 lines) | 108.3 to 94.6 ms, 1.14x | 87.3 to 71.9 ms, 1.21x |
| Phone LM, interleaved (released TQ1_0, default build) | 2.54 to 3.09 frames/s, about 1.2x (1.14x and 1.33x per pair) | not measured |

The desktop timings and the phone run used the first range; five more rows change neither. On the phone the raw stream
was byte-identical in all four runs (`52072e62…`, the same as every earlier TQ1_0 run), and first audio fell from 48.6
and 56.5 s to 43.9 and 44.0 s.

**The guard.** The engine computes the restricted head (`KittenEngine::head_rows`, a view of the rows, no copy; every
other id `f32::MIN`, finite and below any top-k cutoff) only inside the measured envelope: top-k on and at most 50, and
top-p below 1, the defaults. Outside it, or with `GGTTS_KITTEN_FULL_HEAD=1`, it computes the whole head. The parity
checks build their `Model` without it. `tests/head_rows.rs` (ignored without `GGTTS_KITTEN_LM`; run with `--ignored`)
requires the restricted logits to equal the whole head's bit for bit inside the rows and `f32::MIN` outside; a view
shifted one row turns it red.

**What this does not establish:** that no id outside the rows can ever reach the top 50. The smallest margin seen was
4.3 at the default settings and seed, on 22 lines; another voice, language or input could come closer.
`head_rows_check` is the instrument to re-run.
