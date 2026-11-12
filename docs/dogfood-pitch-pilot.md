# Pitch measurement shadow check

The pitch instrument and a reference recording-booth analysis were run separately on the same two retained FLACs from the
[audio pilot](dogfood-audio-pilot.md). This is a small measurement comparison, not a quality verdict. Those clips'
model and trim provenance remain unresolved in [#89](https://github.com/dandelion-love/ggtts/issues/89); neither
the measurements nor the clips replace a baseline comparison and a listening test. No audio or source text is
committed here.

| Input SHA-256 | Reference range / median / voiced frames | ggtts range / median / voiced frames |
|---|---|---|
| `aa1747a910576489674e75c8197766656c2f5b5dc4b7ea99a95b9be09b22920d` (Qwen) | 7.0 st / 248 Hz / 58 | 7.01557 st / 247.727 Hz / 58 |
| `55273a6ff22b7287c396a3e79bc44e833b1129fbe316c36c5910b44607806842` (Kitten) | 10.3 st / 142 Hz / 99 | 10.34468 st / 142.297 Hz / 99 |

The booth rounds range to 0.1 semitone and median to whole Hz; ggtts reports unrounded values. The booth's analysis script (`page.js`) had
SHA-256 `358e2167ae693f0c504664231795f12f039449ea6486cb5c5cd8484a2b2be0c1`.
Both use Pitchy 4.1.0 SHA-256 `79b92333cde9ec16798757f73b624abd13522da31e8fab2993c19d319b3bb36e`
and fft.js 4.0.4 SHA-256 `4092b92cec888377666265b724d676628d377d4d5f56b03b76d9c2c27fd061ea`.
Both comparisons used Node 12.22.9. They share the detector, so agreement is a wiring and definition check, not
independent validation of the detector or evidence across different speakers and conditions.

This comparison is with the recording booth. The loud/soft-end analysis uses a different pYIN
voicing track and speech-core selection, so these results do not establish parity with that measure.

The synthetic controls include a copy no-op, an octave plant and silence. A 200 Hz tone with quiet boundaries
measured 200 Hz and a zero-semitone range in the reference. An uninterrupted constant-level tone produced no voiced
frames under the booth's relative gate; this behavior is important when interpreting null pitch. A missing Node
runtime makes the instrument skipped and a required run incomplete.

To repeat the reference calculation, pipe a JSON object containing `sample_rate` and mono `samples` to
`node scripts/reference_pitch.js /path/to/page.js PAGE_SHA256`. The script refuses a mismatched page or detector
hash and writes only aggregate pitch values to stdout. A local Python environment with `soundfile` can supply the
JSON without saving the audio or text:

```sh
python -c 'import json, soundfile as sf, sys; x, sr = sf.read(sys.argv[1]); print(json.dumps({"sample_rate": sr, "samples": x.tolist()}))' /path/to/input.flac |
  node scripts/reference_pitch.js /path/to/page.js 358e2167ae693f0c504664231795f12f039449ea6486cb5c5cd8484a2b2be0c1
```

For ggtts, run `ggtts-measure --audio /path/to/input.flac --audio-state unknown --report /new/report-dir
--instrument pitch-range --require pitch-range`. Supply a known audio state only when independently established.
