# Instrument hooks

This page lists every point where an instrument can observe a render, what each point carries, when it fires, and
what guarantees an instrument can rely on. The event stream that carries them is `ggtts-trace/1`, and its JSON
schema is [`schemas/ggtts-trace-1.schema.json`](../../schemas/ggtts-trace-1.schema.json).

A hook is **available** when an engine emits it today, and **planned** when this page fixes its place and shape
but no code emits it yet. An instrument must not depend on a planned hook.

## Two tiers

| Tier | Language | Where it runs | What it sees | Guarantee |
|---|---|---|---|---|
| **Event instrument** | Any | Out of process | The `ggtts-trace/1` stream, live (a pipe) or replayed (a file), plus the run's artifacts (the rendered FLAC) | It cannot change the render: it never shares the engine's memory, sampler or thread. A crash, hang or bad output is its own failure, not the render's. |
| **Tensor hook** | Rust | In process, behind a cargo feature | Values an event cannot carry cheaply: the talker's logits, chunk audio before trimming | Read-only: it gets shared references to values the engine already computed, never a mutable one, and never draws from the sampler's random stream. It is compiled only with its feature, so a default build has no tensor hook at all. |

Python never runs inside the decode loop. Anything that needs a value per frame and can be computed from what the
stream carries is an event instrument. A tensor hook is justified only by a value the stream cannot carry.

## The hook catalogue

| Level | Hook | Tier | Carried by | Data | Status |
|---|---|---|---|---|---|
| Per step | Stop probability | Event | `step.p_stop` | Probability of end-of-speech at temperature 1, before filtering | Available (the fake engine) |
| Per step | Sampler state | Event | `step.entropy`, `step.top1_p`, `step.rank` | Entropy in nats after temperature, before top-k and top-p; the most likely token's probability; the rank of the token drawn | Available (the fake engine) |
| Per step | Token | Event | `step.token`, and at level 2 `step.cb`, `step.cb_p` | Codebook 0's token; at level 2 the depth model's tokens (codebooks 1 onward) and their probabilities | Available (the fake engine) |
| Per step | Talker logits | Tensor | A tensor hook's `&[f32]` | The full logit vector for codebook 0 before temperature | Planned |
| Per chunk | Draw begins | Event | `chunk_begin` | Chunk index, draw index, the chunk's text, its seed | Available |
| Per chunk | Draw ends | Event | `chunk_end` | Frames, stop reason (`eos`, `cap`, `abort`), the judge's seconds per character and verdict, wall-clock ms | Available |
| Per chunk | Tokens | Event | The draw's `step` events, in frame order | Every frame's codebook-0 token (all codebooks at level 2) | Available (the fake engine) |
| Per chunk | Trim | Event | `trim` | Lead and tail seconds cut from the kept draw, and the kept sample range | Available |
| Per chunk | Chunk audio | Tensor | A tensor hook's `&[f32]` and sample rate | The kept draw's decoded samples before trimming | Planned |
| Per render | Text | Event | `chunk_begin.text` of each chunk's draw 0, in chunk order | The text the engine was given, chunk by chunk | Available |
| Per render | Join | Event | `join` | Chunks joined, samples, the level gain in dB | Available |
| Per render | End | Event | `end` | Total frames generated (every draw), seconds of audio | Available |
| Per render | Final audio | Event (artifact) | The FLAC that `ggtts-render` writes | The read as delivered | Available as a file; the runner names it to instruments |
| Per render | Timings | Event | `t_ms` on every event, `chunk_end.ms` | Milliseconds since the trace began; the wall-clock of one draw (generate, decode and judge together) | Available |
| Per run | All items | Event | One trace per item, listed by the run's manifest | Every render of the run | Planned (the runner) |

"Available (the fake engine)" means the stream defines the hook and the fake engine emits it; a real engine emits
it once its traced generation is wired. An engine that cannot supply a field leaves it out: a missing field is
never zero.

## Timing and order

A render's stream is one JSON object per line, UTF-8, in this order:

1. `trace`, the header, once and first. It carries `format` (the version) and what made the trace.
2. For each chunk, in chunk order, for each draw of that chunk, in draw order:
   `chunk_begin`, then that draw's `step` events with `frame` rising from 0, then `chunk_end`.
   A chunk is redrawn only when the judge flags the draw, so most chunks have one draw.
3. `trim`, once per chunk, after that chunk's last `chunk_end`. It describes the kept draw (the last one).
4. `join`, once, after every chunk.
5. `end`, once, last.

Every event carries `t_ms`, non-decreasing through the stream. `t_ms` and `chunk_end.ms` are the only fields that
may differ between two runs of the same render with the same seed: replay and every comparison ignore them, and an
instrument must not treat them as data about the audio.

An event instrument reading live sees each event once the engine has computed it. Nothing waits for the
instrument: the engine writes the stream and never reads from it.

## The read-only guarantee, and how it is enforced

An instrument never changes output.

- **Event tier.** The engine computes an event's values from the distribution it has already built, and never
  draws from the sampler's random stream to compute them. `trace::check_trace_identity` renders the same requests
  untraced and traced and requires identical tokens. It goes red on a planted instrument that takes one extra draw
  (`crates/ggtts-core/tests/trace.rs`).
- **Tensor tier.** A hook receives shared references only, so the type system refuses a write. It runs on the
  engine's thread between computations the engine has finished, so it cannot reorder them. The same identity check
  applies to every tensor hook once one exists.
- **Level 0 computes nothing.** With tracing off the sink is `NullEvents`, and neither the pipeline nor an engine
  computes an instrument value.

## The stream's version, and the compatibility rule

The header's `format` is the stream's version, `ggtts-trace/1` today. The schema is
[`schemas/ggtts-trace-1.schema.json`](../../schemas/ggtts-trace-1.schema.json); every line of every trace validates
against it, and a test checks that on every recorded fake-engine render (`crates/ggtts-core/tests/trace_schema.rs`).

- **Adding a field or an event is a new version.** The schema refuses fields it does not name, so a field added
  without a schema change fails that test.
- **Readers accept the current version and the one before it.** `trace::READS` lists the formats `trace::parse`
  accepts. When `ggtts-trace/2` is published, `READS` becomes `["ggtts-trace/2", "ggtts-trace/1"]`, and
  `ggtts-trace/1` is dropped only at `/3`. A test holds `READS` to that shape.
- Writers write only the current version.
