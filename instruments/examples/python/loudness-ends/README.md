# Loud and soft ends (work in progress)

This experimental instrument computes the checkpoint-card loud and soft ends: 10 ms RMS levels of voiced
frames inside the speech core, relative to their median, then the 90th and 10th percentiles. It requires at least
20 voiced frames. It can now derive pYIN voicing and an edges/2 speech core from the audio in memory, with NumPy,
librosa and their dependencies. Missing dependencies or insufficient voiced frames result in an incomplete required
measurement. The core selector is derived from `cantonese-speech-analysis/scripts/analysis/canto/steps.py`
(Apache-2.0); it does not implement the complete edge-event detector.

No external voicing track is accepted. The recording-booth Pitchy track uses a different detector and eligibility
rule and is not a substitute for pYIN.

The instrument is not yet registered in the shipped catalogue. Promotion proof and workflow integration for its
dependencies still need to be implemented before this is ready for review. Never commit user audio or derived
per-frame tracks.
