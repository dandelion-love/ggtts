# Pitch range

Measures voiced pitch with the lab recording booth's pinned Pitchy 4.1.0 detector. The 10 ms frame RMS gate is
`max(10th-percentile level + 10 dB, loudest frame - 35 dB)`. Pitch is measured in 1024-sample windows below
44.1 kHz, otherwise 2048 samples, at a 10 ms hop. A frame is voiced when the detector's clarity is strictly above
0.9 and its pitch is strictly between 60 and 500 Hz. With at least 10 voiced frames, the instrument reports the
median pitch and the 95th minus 5th percentile of voiced pitch in semitones. With fewer, the two pitch values are
null and `pitch_voiced_frames` records why; this is not a score of audio quality.

This is the recording-booth measure. The lab's checkpoint-card loud/soft-end analysis uses a different pYIN
voicing track and speech-core selection; this instrument's voiced frames or range must not be substituted for it.

Node.js is required. Set `GGTTS_NODE` to its executable if `node` is not on PATH. If Node is unavailable, the
instrument reports `skipped`; a required measurement is incomplete. The pinned MIT-licensed detector and FFT are
bundled in `instruments/vendor` and verified before use. The Python SDK's supported audio formats and optional
`soundfile` dependency still apply. The instrument measures the supplied audio as-is; it does not trim it or verify
the caller's audio-state claim.

An unchanged sample copy must give the same values; the self-test plants an octave change and checks the range rises.
Silence has zero voiced frames and null pitch values. These are measurement controls, not listening or quality gates.

```sh
python instrument.py --audio out.flac --envelope env.json
python -m unittest test_instrument
```
