# Template: per-chunk hook (`chunk_begin`, `chunk_end`)

Reads every draw's `chunk_begin` and `chunk_end` and reports **how the draws ended**: redraws, stops at the frame cap,
and draws the judge flagged. Each chunk's draw count is scoped to its render and chunk, so it can be joined with other
per-chunk results.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `draws` | 1 | neutral | Draws in total |
| `redraws` | 1 | lower | Draws after a chunk's first |
| `cap_stops` | 1 | lower | Draws that stopped at the frame cap (`stop` = `cap`) |
| `flagged_draws` | 1 | lower | Draws the judge flagged (`verdict` not null) |
| `chunk_draws` (per chunk) | 1 | lower | That chunk's draws |

```
python instrument.py --events trace.jsonl --envelope env.json
python -m unittest test_instrument
```

**Self-test.** *Planted:* one `chunk_end` changed to `stop: cap`, `verdict: runaway`; `cap_stops` and `flagged_draws`
each rise by one. *No-op:* the clock moved; nothing changes.
