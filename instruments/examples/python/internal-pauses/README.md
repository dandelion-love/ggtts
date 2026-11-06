# Worked example: internal pauses

Finds **silences inside the speech** of a rendered read: stretches below a level floor that are longer than
`--min-pause-s`, between the first and the last sound. Edge silence is not counted, and the trim and the `trim`
template already report it.

The default `--min-pause-s` is **1.5 s**. That is longer than any pause the pipeline puts between chunks on purpose:
the longest is a paragraph break, about 1.2 s with the edge silence kept on both sides. So a pause found at the
default is one the engine generated, not one the pipeline designed. Lower it to study the designed pauses too.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `internal_pauses` | 1 | lower | Silences inside the speech longer than `--min-pause-s` |
| `longest_pause_s` | s | lower | The longest of them, `null` when there is none |
| `pause_s_total` | s | neutral | Their total length |

The level is measured in 10 ms frames. A frame is silent when its RMS is below `--floor-dbfs` (default -40 dBFS).

```
python instrument.py --audio out.flac --envelope env.json [--min-pause-s 1.5] [--floor-dbfs -40]
python -m unittest test_instrument
```

**Self-test.**
- *Planted:* 2.0 s of silence inserted into the middle of the speech. `internal_pauses` rises by one, and
  `longest_pause_s` reads at least 2.0.
- *Planted at the edge:* the same silence added before the first sound is not counted. An instrument that counted
  edges would fail this.
- *No-op:* the same samples rewritten as WAV; nothing changes.
