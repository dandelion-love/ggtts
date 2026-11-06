# Template: per-render hooks (`join`, `end`)

Reads the render's `join` and `end` events and reports **its totals**. Its real job is the completeness rule: a stream
that stops before `end` is a truncated measurement, never a short render, and this template shows how an instrument
says so.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `seconds` | s | neutral | Seconds of audio in the read (`end.seconds`) |
| `frames` | 1 | neutral | Frames generated, every draw (`end.frames`) |
| `chunks` | 1 | neutral | Chunks joined (`join.chunks`) |
| `gain_db` | dB | n/a | The level gain applied to the read, `null` when none was |

```
python instrument.py --events trace.jsonl --envelope env.json
python -m unittest test_instrument
```

**Self-test.** *Planted:* the `end` line removed; the envelope says `truncated` and the totals that `end` carries are
`null`. *No-op:* the clock moved; nothing changes.
