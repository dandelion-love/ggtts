# Instrument hooks

This page lists every point where an instrument can observe a render, what each point carries, when it fires, and
what guarantees an instrument can rely on. The event stream that carries them is `ggtts-trace/2`, and its JSON
schema is [`schemas/ggtts-trace-2.schema.json`](../../schemas/ggtts-trace-2.schema.json). Readers also accept v1.

A hook is **available** when an engine emits it today, and **planned** when this page fixes its place and shape
but no code emits it yet. An instrument must not depend on a planned hook.

## Two tiers

| Tier | Language | Where it runs | What it sees | Guarantee |
|---|---|---|---|---|
| **Event instrument** | Any | Out of process | The `ggtts-trace/2` stream, live (a pipe) or replayed (a file), plus the run's artifacts (the rendered FLAC) | It cannot change the render's tokens or audio: it never shares the engine's memory, sampler or thread. A crash or bad output is its own failure. A live reader that falls behind can block the engine's writes until the runner (#30) bounds each instrument's buffer; see Timing and order. |
| **Tensor hook** | Rust | In process, behind a cargo feature | Values an event cannot carry cheaply: the talker's logits, chunk audio before trimming | Read-only by convention (shared references to values the engine already computed, no draw from the sampler's random stream), enforced by the identity check, not by the type system. It has no timeout or crash boundary: its failure is the render's. It is compiled only with its feature, so a default build has no tensor hook at all. |

Python never runs inside the decode loop. Anything that needs a value per frame and can be computed from what the
stream carries is an event instrument. A tensor hook is justified only by a value the stream cannot carry.

## The hook catalogue

| Level | Hook | Tier | Carried by | Data | Status |
|---|---|---|---|---|---|
| Per step | Stop probability | Event | `step.p_stop` | Engine-reported value; see the versioned meaning in the trace header | Available on Qwen and the fake engine; unavailable on Kitten |
| Per step | Sampler state | Event | `step.entropy`, `step.top1_p`, `step.rank` | Entropy in nats after temperature, before top-k and top-p; the most likely token's probability; the rank of the token drawn | Available (the fake engine) |
| Per step | Token | Event | `step.token`, and at level 2 `step.cb`, `step.cb_p` | Codebook 0's token; at level 2 the depth model's tokens (codebooks 1 onward) and their probabilities | Available (the fake engine) |
| Per step | Talker logits | Tensor | A tensor hook's `&[f32]` | The full logit vector for codebook 0 before temperature | Planned |
| Per chunk | Draw begins | Event | `chunk_begin` | Chunk index, draw index, the chunk's text, its seed, and the request's style instruction (`instruction`, absent when there is none) | Available |
| Per chunk | Draw ends | Event | `chunk_end` | Frames, stop reason (`eos`, `cap`, `abort`), the judge's seconds per character and verdict, wall-clock ms | Available |
| Per chunk | Tokens | Event | The draw's `step` events, in frame order | Every frame's codebook-0 token (all codebooks at level 2) | Available (the fake engine) |
| Per chunk | Trim | Event | `trim` | Lead and tail seconds cut from the kept draw, and the kept sample range | Available |
| Per chunk | Hand-off to the stream | Event | `emit` | The chunk's piece before stream levelling: its first sample and length. This is not the first stream write. | Available |
| Per chunk | Chunk audio | Tensor | A tensor hook's `&[f32]` and sample rate | The kept draw's decoded samples before trimming | Planned |
| Per render | Text | Event | `chunk_begin.text` of each chunk's draw 0, in chunk order | The text the engine was given, chunk by chunk | Available |
| Per render | Join | Event | `join` | Chunks joined, samples, the level gain in dB | Available |
| Per render | End | Event | `end` | Total frames generated (every draw), seconds of audio | Available |
| Per render | Final audio | Event (artifact) | The FLAC that `ggtts-render` writes | The read as delivered | Available as a file; the runner names it to instruments |
| Per render | Timings | Event | `timing.stage`, `timing.ms` | Per-draw generation, decoding and judging; on CLI story traces, render duration and first actual stream write when streaming | Available in v2 |
| Per run | All items | Event | One trace per item, listed by the run's manifest | Every render of the run | Planned (the runner) |

"Available (the fake engine)" means the stream defines the hook and the fake engine emits it; a real engine emits
it once its traced generation is wired. An engine that cannot supply a field leaves it out: a missing field is
never zero.

The header's `flags.p_stop_semantics` identifies the producer's meaning. `qwen-sampler-weights/1` is the EOS
weight divided by the sum of the actual f32 weights after suppression, repetition penalty, temperature, top-k,
top-p and the sampler's cutoff. It describes normalized weights, not a claim that the finite-precision random draw
has an exactly continuous probability distribution. `fake-synthetic/1` is a scripted test signal, not a model
probability. This metadata is a producer declaration, not independent proof of the implementation. Missing metadata
means the semantics are unknown: in particular, older Qwen traces used a separate
approximation that omitted the top-p cutoff, while older documentation incorrectly called it an unfiltered
temperature-1 probability. Do not reinterpret those traces as the new measurement. The standalone stop-hook JSONL
also adds `p_stop_semantics` when known; old records omit it. Strict readers must allow the new field.
Because values and headers can change, replaying a retained legacy trace against a corrected build can report a
semantic or value mismatch; preserve the original binary and trace when reproducing an old report.
Calculating this value repeats weight preparation on a copy of the logits. That work runs only when requested and is
included in the instrumented generation wall time; it is not free or isolated model-compute time.

Qwen emits a step for the EOS draw itself; the fake engine reports `stop: eos` without a separate EOS step, so that
terminal observation is not guaranteed by the current cross-engine trace contract. Measurement changes and the
impact on existing reports are tracked in [#104](https://github.com/dandelion-love/ggtts/issues/104).

## Timing and order

A render's stream is one JSON object per line, UTF-8, in this order:

1. `trace`, the header, once and first. It carries `format` (the version) and what made the trace.
2. For each chunk, in chunk order, for each draw of that chunk, in draw order:
   `chunk_begin`, then that draw's `step` events with `frame` rising from 0, then generation, decode and judge
   `timing` records and `chunk_end`.
   A chunk is redrawn only when the judge flags the draw, so most chunks have one draw.
3. `trim`, once per chunk, after that chunk's last `chunk_end`. It describes the kept draw (the last one).
4. `emit`, once per chunk, right after its `trim` and before the next chunk's `chunk_begin`: the chunk's piece
   was handed to the stream.
5. `join`, once, after every chunk.
6. CLI story traces add `first_audio` timing when streaming, then `render` timing, after the last stream write.
7. `end`, once, last. The CLI delays this marker until held streaming audio has been written.

Every event carries `t_ms`, non-decreasing through the stream. `t_ms`, `chunk_end.ms`, and `timing.ms` are fields that
may differ between two runs of the same render with the same seed: replay and every comparison ignore them, and an
instrument must not treat them as data about the audio.

An event instrument reading live sees each event once the engine has computed it. The engine never reads from the
stream. A live reader that falls behind can fill its pipe and block the engine's writes; the runner (#30) owns the
bound: a buffer per instrument, and on overflow the instrument is marked failed while the render's tokens and timings
are unchanged. Until the runner exists, read a written trace (replay) rather than a live pipe.

A stream without `end` is truncated: readers treat it as incomplete, never as a short render.

## The read-only guarantee, and how it is enforced

An instrument never changes output. This is checked, not assumed, and only as far as the check reaches: today on the
fake engine (`check_trace_identity`), and on a pinned real GGUF at a fixed seed once #34 lands. Each result reports
its backend, build and seed.

- **Event tier.** The engine computes an event's values from the distribution it has already built, and never
  draws from the sampler's random stream to compute them. `trace::check_trace_identity` renders the same requests
  untraced and traced and requires identical tokens. It goes red on a planted instrument that takes one extra draw
  (`crates/ggtts-core/tests/trace.rs`).
- **Tensor tier.** A hook receives shared references, so safe Rust cannot write through them; that is a convention,
  not a proof. What enforces read-only is the identity check: the same requests with and without the hook must give
  identical tokens and audio, and a planted hook that writes one value must turn it red (#32, #34). A tensor hook
  runs in the engine's process and thread, so it has no timeout or crash boundary: a panic, hang or abort in a hook
  is the render's (#30 documents and tests each).
- **Level 0 computes nothing.** With tracing off the sink is `NullEvents`, and neither the pipeline nor an engine
  computes an instrument value.

## The stream's version, and the compatibility rule

The header's `format` is the stream's version, `ggtts-trace/2` today. The schema is
[`schemas/ggtts-trace-2.schema.json`](../../schemas/ggtts-trace-2.schema.json); every line of every trace validates
against it, and a test checks that on every recorded fake-engine render (`crates/ggtts-core/tests/trace_schema.rs`).

- **Adding a field or an event is a new version.** The schema refuses fields and events it does not name, so one
  added without a schema change fails that test, and every new event kind lands with its schema entry and its
  row in this catalogue in the same change. `emit` is part of `ggtts-trace/1` as first published: streaming added
  it before this schema existed, and the schema test caught it.
- **Readers accept the current version and the one before it.** `trace::READS` lists the formats `trace::parse`
  accepts. `READS` is `["ggtts-trace/2", "ggtts-trace/1"]`, and
  `ggtts-trace/1` is dropped only at `/3`. A test holds `READS` to that shape.
- Writers write only the current version.
