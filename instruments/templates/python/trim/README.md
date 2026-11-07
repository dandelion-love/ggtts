# Template: per-chunk trim hook (`trim`)

Reads every chunk's `trim` event and reports **how much edge silence was cut**. Long edges are dead air the engine
generated around the speech.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `lead_s_max` | s | lower | The longest lead cut from any chunk |
| `tail_s_max` | s | lower | The longest tail cut from any chunk |
| `trim_s_total` | s | neutral | Lead and tail cut, over every chunk |
| `edges_over_limit` | 1 | lower | Leads and tails longer than `--max-edge-s` (default 1.0) |

```
python instrument.py --events trace.jsonl --envelope env.json [--max-edge-s 1.0]
python -m unittest test_instrument
```

**Self-test.** *Planted:* one chunk's `lead_s` set to 5.0; `lead_s_max` reads 5.0 and `edges_over_limit` rises by one.
*No-op:* the clock moved; nothing changes.
