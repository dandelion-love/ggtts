---
type: Research Finding
title: Where a Kitten-TTS 2 generation step goes, and a head that computes only the samplable rows
description: On desktop CPU a step is about a third layer products, a fifth output head and a third V-cache copying; TQ1_0's products cost 13% more than Q4_0's; computing the head for the 6,566 samplable ids only gives the same tokens on 44 of 44 lines and steps 1.14x to 1.21x faster.
tags: [kitten-tts-2, performance, profiling, tq1_0]
status: draft
generated: { by: claude-code/opus-5.5, at: 2026-11-12T16:40:00-07:00 }
sources:
  - { id: arm, resource: "/research/arm-features-on-the-phone.md", title: "ggml's ARM dot-product and matrix features on a phone" }
  - { id: tq1, resource: "/research/kitten-memory-and-tq1.md", title: "Kitten-TTS 2's memory, and the released TQ1_0 model on a phone" }
---
# Method

Three tools, all in `crates/ggtts-kitten/examples/`, run on desktop CPU (WSL Ubuntu, an i9-12900K) at 4 threads and
`nice -n 19`. The desktop was shared with other work, so absolute times move; comparisons were interleaved where they
decide anything. The phone half waits for the phone (it was off adb while this was measured).

- `op_profile`: the scheduler's evaluation callback observes every node of the engine's own graph, so each node's time
  is measured where it runs. Observing makes each node its own split, which costs a fixed amount per node: small ops
  read high, and a step reads about 75 to 80 ms observed against 70 to 110 ms unobserved.
- `matmul_bench`: one-token products at the LM's shapes, per storage type, with random exact-ternary weights packed by
  the converter's own packer, 28 per graph as a step has them.
- `head_rows_check`: runs a test set of 22 lines (`fixtures/head_rows_lines.txt`: English and Cantonese, from the
  repository's story fixtures and four more) at the default seed, and counts at every step the ids outside the audio
  ids and the two end ids that are still finite after the sampler's filters.

# Where a step goes (desktop, `op_profile`, 3 lines)

| Per one-token step | Q4_0 LM (508 steps) | Released TQ1_0 LM (601 steps) |
|---|---|---|
| Layer matrix products | 26.5 ms | 30.0 ms |
| Output head (158,241 by 2048, bf16) | 16.2 ms | 16.1 ms |
| `CONT` | 26.3 ms | 27.6 ms |
| `PAD` (padding the input to 2 x 128 per group) | none | 2.5 ms |
| Everything else (norms, RoPE, softmax, adds, copies) | 4.2 ms | 4.3 ms |

- **TQ1_0's products cost 13% more than Q4_0's here**, with twice the stored width (the padding). On the phone the
  whole LM step was 24 to 35% slower ([TQ1_0 on the phone](/research/kitten-memory-and-tq1.md));[^tq1] the same
  profile on the phone is still to run. `matmul_bench` on desktop puts the two types' products level (about 33 ms per
  step each, the head 23 ms), so the isolated products and the engine's do not fully agree: treat the engine's
  profile as the measurement.
- **`CONT` is about a third of a step.** There are two per layer, and the large one copies the whole V cache,
  transposed, on every token (`ggml_cont(ggml_permute(vall, …))` in `Session::forward`), so it grows with the context.
  Storing V transposed would remove it: filed as issue 257.
- **The head is about a fifth.** It multiplies all 158,241 rows, though a step can only keep the ids below.

# The head, restricted to the samplable rows

A step keeps a frame for an audio id and stops on `speech_end` or `stop`. Those 6,563 ids lie in one range,
151,670 to 158,236 (6,566 rows with three unused ids inside). The sampler does not restrict the vocabulary itself: it
draws from whatever stays finite after its penalties, temperature, top-k 50 and top-p 0.8. So computing only that
range is exact for a step exactly when no other id is still finite there.

| | Released TQ1_0 LM | Q4_0 LM |
|---|---|---|
| Steps over the 22 lines | 3,923 | 3,713 |
| Steps with any other id still finite | **0** | **0** |
| Ids still finite per step, on average | 11.8 to 15.5 | 4.3 to 6.0 |
| Lines with identical tokens, full head against restricted | **22 of 22** | **22 of 22** |
| Step time, full head against restricted (interleaved: full, restricted, restricted, full; 4 lines) | 108.3 to 94.6 ms, **1.14x** | 87.3 to 71.9 ms, **1.21x** |

The saving, 14 to 15 ms a step, is what the profile gave the head. A first, non-interleaved comparison read 1.38x for
TQ1_0; that was the shared desktop, not the change.

The engine now computes the head for that range only (`Model::head_rows`, set by `KittenEngine::new`; a view of the
rows, no copy) and gives every other id `f32::MIN`: finite, so the sampler's refusal of non-finite logits still
guards the computed ones, and always below the top-k cutoff. `GGTTS_KITTEN_FULL_HEAD=1` computes the whole head, to
compare. The parity checks build their `Model` without it and still compare every logit.

**Checked:** `tests/head_rows.rs` (needs `GGTTS_KITTEN_LM`) requires the restricted logits to equal the full head's bit
for bit inside the range and `f32::MIN` outside, on a prefill and three steps; it passes on both LMs, and a view
shifted by one row turns it red. In `head_rows_check`, a range that drops `speech_end` makes every line run to the
1,000-token cap: 0 of 3 identical.

**What this does not establish:** that no other id can ever reach the top 50. It held at every one of 7,636 steps here,
at the default sampler settings and seed. A setting that widens the candidates (a larger top-k, or no top-p) or an
unusual input could break it; `head_rows_check` is the instrument to re-run then.

# Not yet measured

- The same profile on the phone (both ggml builds), for issue 255's phone half.
- The restricted head's speed on the phone.
- `CONT` without the V copy (issue 257).
