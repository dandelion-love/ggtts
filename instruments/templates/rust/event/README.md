# Template: a Rust event instrument

An out-of-process instrument: a program that reads a render's event stream (`ggtts-trace/1`) and writes one envelope
(`ggtts-report/1`). The runner starts it after the render, so it cannot change the render. If it crashes, hangs or
writes nonsense, that is its own failure, not the render's.

This one reports **how long each draw ran**, from the `chunk_end` events:

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `draws` | 1 | neutral | Draws in the render |
| `longest_draw_frames` | 1 | lower | The most frames any draw generated; a runaway shows up here |
| `frames_total` | 1 | neutral | Frames over every draw |

A stream without `end` is reported `truncated`, never a short render.

```
cargo run -p ggtts-instrument-event-template -- --events trace.jsonl --envelope env.json
cargo test -p ggtts-instrument-event-template      # the planted defect, the no-op, truncation
```

It takes the runner's flags: `--events`, `--audio`, `--envelope`, `--name`, plus `--run` and `--render` for the
envelope's identity, and any other `--KEY VALUE`, which is recorded as configuration.

**Self-test** (`tests/selftest.rs`, on a fake-engine render):
- *Planted:* one `chunk_end` claims 5000 frames; `longest_draw_frames` reads 5000.
- *No-op:* every clock field moved; the envelope's metrics are unchanged.
- *Truncation:* the `end` line removed; the status is `truncated`.

To write your own, copy this folder, rename the package, and change `measure`.
