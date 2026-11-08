# Loud and soft ends

This instrument computes the checkpoint-card loud and soft ends: 10 ms RMS levels of voiced
frames inside the speech core, relative to their median, then the 90th and 10th percentiles. It requires at least
20 voiced frames. It can now derive pYIN voicing and an edges/2 speech core from the audio in memory, with NumPy,
librosa and their dependencies. Missing dependencies or insufficient voiced frames result in an incomplete required
measurement. The core selector is derived from `cantonese-speech-analysis/scripts/analysis/canto/steps.py`
(Apache-2.0); it does not implement the complete edge-event detector.

No external voicing track is accepted. The recording-booth Pitchy track uses a different detector and eligibility
rule and is not a substitute for pYIN.

The comparison used Python 3.10 with NumPy 1.26.4, SciPy 1.15.3, librosa 0.11.0 and soundfile 0.12.1;
these direct dependencies are listed in `requirements.txt` (transitive dependencies are not locked). The instrument
refuses a different librosa version.
The runtime records its NumPy, SciPy and librosa versions in the envelope. Other versions have not been shown to
match; use a dedicated environment with those versions when reproducing the comparison. Promotion proof and broader
integration remain to be finished. Never commit user audio or derived per-frame tracks.

```sh
python3.10 -m venv /path/to/ggtts-measures
/path/to/ggtts-measures/bin/python -m pip install -r instruments/examples/python/loudness-ends/requirements.txt
```

On Windows, use `Scripts/python.exe` inside the environment. In a packaged release, the requirements file is under
the adjacent `instruments/examples/python/loudness-ends/` directory.

```sh
ggtts-measure --audio input.flac --audio-state unknown --report /new/report \
  --instrument loudness-ends --require loudness-ends
```

Select the Python environment with `GGTTS_PYTHON`. A missing pYIN dependency makes this required measurement
incomplete; the runner does not silently substitute a different pitch detector. The full pYIN test requires the
analysis stack and is not yet part of the default Python-only suite.
The real-instrument promotion proof requires every catalogued instrument; supply this environment through
`GGTTS_PYTHON` there too. A missing or incompatible runtime must leave that proof incomplete or failed, not passed.
