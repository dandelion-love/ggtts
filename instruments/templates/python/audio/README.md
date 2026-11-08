# Template: the render's audio (an artifact, not an event)

Reads the rendered audio the runner passes as `--audio` and reports **its levels**. Use it as the starting point for
anything measured on the sound itself.

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `peak_dbfs` | dBFS | n/a | The loudest sample |
| `rms_dbfs` | dBFS | neutral | RMS over the whole read |
| `clipped_samples` | 1 | lower | Samples at or above `--clip` in magnitude (default 0.999) |

A silent read has no level in dB: `peak_dbfs` and `rms_dbfs` are then `null`, never 0 or minus infinity. For a
pipeline story render, the runner records trimmed and joined audio; a single render is recorded as raw. For an existing file, `ggtts-measure`
records the supplied `--audio-state` (`raw`, `trimmed`, or `unknown`; default `unknown`). A standalone invocation
also reports `unknown`; this instrument does not determine or change the audio's trim state.

```
python instrument.py --audio out.flac --envelope env.json [--clip 0.999]
python -m unittest test_instrument
```

FLAC that ggtts wrote is read with the standard library. Any other FLAC needs `pip install soundfile`.

**Self-test.** *Planted:* 100 samples set to full scale; `clipped_samples` rises by 100 and the peak reads 0 dBFS.
*No-op:* the same samples rewritten as WAV, a different file with identical audio; nothing changes.
