# Audio measurement dogfood pilot

## Scope and provenance

This is a small shadow comparison, not a quality verdict or a replacement for the lab's baseline and owner-listening gates. It measures the same retained release-test FLACs with the ggtts instruments and existing lab functions, without generating new speech or committing audio. One short example from each engine is far too small to establish general parity.

The two inputs were retained from the v0.1.0-rc.7 CUDA release smoke exercises. Their exact model hashes and trim state were not preserved in the pilot inputs; treat both as **unknown**, not verified trimmed. Qwen's file has SHA-256 `aa1747a910576489674e75c8197766656c2f5b5dc4b7ea99a95b9be09b22920d` (77,760 samples); Kitten's has `55273a6ff22b7287c396a3e79bc44e833b1129fbe316c36c5910b44607806842` (75,840 samples). Both are mono 24 kHz. Only hashes and aggregate measurements appear here.

ggtts sources were at `2e7eca7259339062cb650a9c296a18ae7c975324` on `rc`; the `ggtts-measure` debug executable was built from the reviewed #90 head `d46e86c95dd57f00cc7571b0b8163fdf2c79e1f4`, SHA-256 `2bcc460b21908a49857f7930cfb3711320eb6ee79e4753ee0cbf9c7aeeb65284`. The retained rc.7 CUDA renderer's executable SHA-256 is `8f128663922970709f0f766aa1c3cbbf0d70c17c209482ddedc36303f74801e3`. The pilot uses the source-tree instruments, both version `0.1.0`. The lab checkout's HEAD was `bc4f20e304dade9d0bca6ae35e5c99476d957729`; the actual `room_stats` from `scripts/recording/serve.py` had file SHA-256 `709454affcb06e77e628507f390b565c9eef434447545bc9f544a3dde3ff9efb`, and `silence_report` from `scripts/common/audio_trim.py` had SHA-256 `df183b4131d73b66d8bf1af84303a311104a68bafe4c50af394118b5a5c8349e`. The interpreter was Python 3.10.12 with NumPy 1.24.4 and soundfile 0.13.1.

## Paired results

| Clip | ggtts peak / RMS dBFS | Lab peak / RMS dBFS | ggtts pauses / longest | Lab gaps / longest |
|---|---|---|---|---|
| Qwen original | -5.008 / -22.757 | -5.01 / -22.76 | 0 / null | 0 / 0 s |
| Kitten original | -0.088 / -18.806 | -0.09 / -18.81 | 0 / null | 0 / 0 s |
| Qwen, 2 s silence inserted | -5.008 / -24.845 | -5.01 / -24.84 | 1 / 2.00 s | 1 / 2.00 s |
| Kitten, 2 s silence inserted | -0.088 / -20.935 | -0.09 / -20.94 | 1 / 2.00 s | 1 / 2.00 s |

The lab level function rounds to 0.01 dB and adds a small numerical floor; on these nonzero clips its peak and whole-file RMS agree with ggtts to the displayed precision. This is `room_stats`' whole-file RMS, not the lab recording harness's speech-only RMS. Zero pauses is represented as `null` for ggtts's longest pause and `0.0` for the lab's; that difference is expected, not a discrepancy in count.

For this recorded run the lab's `gap_min_s` was set to 1.5 s, matching the then-current ggtts 0.1.0 default duration threshold. The lab's operational default is 0.25 s. In that run ggtts used an **absolute** -40 dBFS 10 ms frame threshold and counted runs strictly longer than 1.5 s; the lab uses 40 dB below the **loudest frame** and includes runs at least 1.5 s. Boundaries, short gaps and quiet audio can therefore disagree. These two examples do not establish equivalence.

The current pilot script explicitly requests a 1.5 s threshold. Instrument 0.2.0 includes gaps exactly at the threshold and defaults to 1.0 s; use the recorded 0.1.0 revision to reproduce the original strict boundary behavior. Neither version uses the lab's relative level threshold.

## Controls and limitations

- A byte-for-byte copy of each input gave identical hashes and reported metrics. This no-op floor is exact for a copy; it does not test re-encoding, resampling, timing, or an independent generation.
- Multiplying samples by 0.5 and writing PCM-16 FLAC reduced both peak and RMS by about 6.02 dB in both tools on each clip. The script checks a 0.02 dB tolerance; that tolerance was chosen after the first pilot run, so it is not a pre-registered acceptance bar.
- Inserting two seconds of zeros midway increased the pause count from zero to one in both tools on each clip, with a 2.00 s longest gap. It demonstrates detection of this coarse plant only.
- A missing `--audio` was rejected with exit 2 and no report directory. This is an input error, distinct from exit 3 for a rendered or supplied input whose required measurement is incomplete.
- The first run used system Python without `soundfile`. Both instruments rejected the lab-written predictive FLAC controls and the measurement returned incomplete. Selecting the lab environment via `GGTTS_PYTHON` made the same controls readable. Python with `soundfile` is required for those files; the built-in fallback only decodes ggtts's FLAC subset.
- The instrument envelopes from this run asserted `provenance.audio = trimmed` even though the files' trim state was not supplied. The inputs here have **unknown** trim state. [#91](https://github.com/dandelion-love/ggtts/issues/91) corrected that behavior for later runs; do not treat the old envelope fields as proof of trimming.

## Reproduce

Use a checkout with the two named lab files, the retained input files, a `ggtts-measure` binary, and a Python environment with NumPy and soundfile. Run from the ggtts checkout; the output directory must not already exist and must be outside both repositories. The script sets `GGTTS_PYTHON` to its own interpreter, writes its derived audio and individual reports only under that output directory, and produces `summary.json` there.

```sh
/path/to/lab/python scripts/dogfood_audio_pilot.py \
  --lab /path/to/qwen3-tts-cantonese-lab \
  --measure /path/to/ggtts-measure \
  --qwen /path/to/qwen.flac --kitten /path/to/kitten.flac \
  --output /tmp/ggtts-dogfood-new
```

The script checks exact no-op equality, the gain response, the planted pause and missing-input refusal. This run's reports remain in `/tmp/ggtts-dogfood-pilot-89c`; that temporary path is not a durable shared artifact. Input hashes allow a future operator to determine whether retained files match.

## Next integration

After #91 fixes trim provenance, run a shadow `ggtts-measure` alongside the lab's existing checks on a defined, permitted held-out set with explicit raw/trimmed state. Preserve the original files and compare like definitions on the same inputs. Record discrepancies and rejection cases before making any measurement a gate. Keep the lab's baseline-first protocol and owner's listening decision unchanged; use no metric here to close a quality claim.
