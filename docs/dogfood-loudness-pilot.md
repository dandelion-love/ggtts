# Loudness measurement shadow check

The `loudness-ends` instrument and the lab checkpoint-card calculation were run separately on the same two
retained FLACs described in [the audio pilot](dogfood-audio-pilot.md). The ggtts instrument derived pYIN and
the speech-core boundaries from the audio rather than consuming the lab's track. Both implementations use
librosa's pYIN, so this is a comparison of wiring and definitions, not independent validation of the detector.
The inputs' model and trim provenance remains unresolved in [#89](https://github.com/dandelion-love/ggtts/issues/89).

| Input | ggtts loud / soft / voiced frames | Lab loud / soft / voiced frames |
| --- | --- | --- |
| Qwen `aa1747a910576489674e75c8197766656c2f5b5dc4b7ea99a95b9be09b22920d` | +3.788319 / −19.282166 dB / 123 | +3.788319 / −19.282166 dB / 123 |
| Kitten `55273a6ff22b7287c396a3e79bc44e833b1129fbe316c36c5910b44607806842` | +4.837268 / −14.807365 dB / 162 | +4.837268 / −14.807365 dB / 162 |

The lab's selected core was 0.60–2.35 s for Qwen and 0.44–2.97 s for Kitten. The separate core implementation
also matched the reference on 1,003 deterministic synthetic level tracks. On the Qwen clip, a byte-for-byte copy
gave identical values, a half-gain PCM-16 control differed by at most 0.002 dB, and a planted dynamic compression
moved the loud and soft ends toward the median. These controls do not establish accuracy across speakers or models.

Both measurements used Python 3.10, NumPy 1.26.4, SciPy 1.15.3, librosa 0.11.0 and soundfile 0.12.1. The lab
reference's `steps.py` SHA-256 was `a0d3c0b265f51fb3e9d96fcf5ea6e746ed7862203bda87fb37033595f779ee41`;
`store.py` was `59619c36714fb00f3609c279da5dd877130f4e91ecd2494d032613f4ba6a75e7`. The packaged instrument was
also run through `ggtts-measure` outside the checkout. With the dependencies absent, the required instrument was
skipped and the measurement exited 3 as incomplete. No clip, source text, or per-frame track is stored here.

To repeat the core comparison with the pinned companion checkout, run
`python scripts/verify_loudness_core.py --companion /path/to/cantonese-speech-analysis`. The lab reference can be
rerun with `scripts/lab_loudness_reference.py --help`; it checks the supplied source hashes and outputs aggregate
results. Give `ggtts-measure` the same audio and the dependency environment described in the instrument README.

This shadow check does not replace the lab baseline or the owner's listening decision. See the instrument's
[README](../instruments/examples/python/loudness-ends/README.md) for its runtime dependencies and test coverage.
The default fake-engine no-change test explicitly excludes this pYIN-dependent instrument; it is not a passing
full-catalogue identity proof. The real-model promotion proof requires the full catalogue with pYIN installed.
The existing real-Qwen CPU no-change test was run locally with this instrument and the pYIN environment: every
catalogued instrument reported `ok`, and the test verified unchanged tokens, samples and input audio. This is a
CPU, single-model test; it does not establish GPU, timing or external-artifact identity.
