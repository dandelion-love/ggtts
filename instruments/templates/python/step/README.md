# Template: per-step hook (`step`)

Reads every `step` event and reports the **stop probability in the first half of each draw**. A well-behaved draw
keeps `p_stop` low until it is near its end, so a high value early is pressure to stop too soon.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `p_stop_early_max` | 1 | lower | The highest `p_stop` among the first half of any draw's frames |
| `early_stop_frames` | 1 | lower | Frames in a draw's first half with `p_stop` at or above `--threshold` (default 0.5) |

Both are `null` (missing, never 0) when every draw has only one frame, so there is no first-half sample. If a draw
has no steps or a step lacks `p_stop`, the instrument reports `skipped` with a reason; when required, the run is
INCOMPLETE. Kitten does not currently supply per-step events. A trace without `end` is reported `truncated`.

```
python instrument.py --events trace.jsonl --envelope env.json [--threshold 0.5]
python -m unittest test_instrument     # the planted defect, and the no-op
```

**Self-test.** *Planted:* `p_stop` set to 0.95 on one early frame; `early_stop_frames` rises by one and the maximum
reaches 0.95. *No-op:* the same trace with every clock field moved; every metric is unchanged.
