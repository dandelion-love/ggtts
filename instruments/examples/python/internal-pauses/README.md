# Worked example: internal pauses

Finds **silences inside the speech** of a rendered read: stretches below a level floor that are at least
`--min-pause-s`, between the first and the last sound. Edge silence is not counted, and the trim and the `trim`
template already report it.

The default `--min-pause-s` is **1.0 s**, inclusive. This can count intended pipeline pauses, including paragraph
breaks around 1.2 s; a count alone does not identify an engine defect. Set a higher threshold to study longer gaps.
Version 0.1.0 defaulted to strictly longer than 1.5 s, so results from the two versions are not directly comparable.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `internal_pauses` | 1 | lower | Silences inside the speech at least `--min-pause-s` |
| `longest_pause_s` | s | lower | The longest of them, `null` when there is none |
| `pause_s_total` | s | neutral | Their total length |

The level is measured in approximately 10 ms whole-sample frames. A frame is silent when its RMS is below
`--floor-dbfs` (default -40 dBFS); gap lengths are quantized to those frames. Pass a known trimmed render when
comparing the trimmed-render measure, and record its audio state separately; this instrument does not trim audio.

```
python instrument.py --audio out.flac --envelope env.json [--min-pause-s 1.0] [--floor-dbfs -40]
python -m unittest test_instrument
```

**Self-test.**
- *Planted:* 2.0 s of silence inserted into the middle of the speech. `internal_pauses` rises by one, and
  `longest_pause_s` reads at least 2.0.
- *Planted at the edge:* the same silence added before the first sound is not counted. An instrument that counted
  edges would fail this.
- *No-op:* the same samples rewritten as WAV; nothing changes.
- *Boundary:* 0.99 s is excluded, while 1.00 s and 1.01 s internal gaps are counted at the default.
