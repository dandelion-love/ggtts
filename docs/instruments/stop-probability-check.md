# Stop-probability correction: comparison

The previous Qwen helper did not always use the sampler's actual filtered weights. For logits `[0, -17, …, -17]`,
temperature 1, no top-k and top-p 0.99999994, it reported nonzero EOS mass even though the sampler's max-minus-16
cutoff zeroed that token. The corrected helper uses the same f32 weight preparation as the sampler on a copy;
the sampler's random stream is not used by the measurement. It reports normalized weights, rather than claiming an
exact continuous distribution for the finite-precision random draw. Trace headers and standalone hook records label
the new semantics; an unmarked older record remains unknown.
Existing `p-stop-early` reports derived from unmarked traces cannot establish which probability definition they
used. This change does not recompute those reports, and no same-input comparison with the lab's stop-probability
measure has yet been established; do not treat it as lab parity.

The sampler refactor was compared against the previous sampler for 100 seeds across greedy, temperature, top-k,
top-p, ties and the cutoff case; sampled tokens, weights and subsequent RNG state matched. One existing real-Qwen
CPU reference fixture produced identical tokens. The real-Qwen CPU no-change test also passed with tracing and the
available event instruments: full codec tokens, samples and resulting FLAC matched. These checks do not establish
cross-backend parity or external-artifact identity.

## Small timing comparison

On an i9-12900K / RTX 3080 Ti host, the old and corrected binaries each rendered the same synthetic Qwen story
three times with trace level 1 and four threads, after one warmup per binary; order alternated. The CPU and
CUDA-requested runs were separate. Values are median render milliseconds (minimum–maximum):

| Backend requested | Previous | Corrected |
| --- | ---: | ---: |
| CPU | 10,700 (10,489–10,763) | 10,645 (10,641–10,978) |
| CUDA | 4,597 (4,507–4,610) | 4,543 (4,441–4,555) |

FLAC hashes, draw seeds and emitted codebook-0 token sequences matched across each backend's runs. The ranges overlap;
three samples on one story do not establish a speed improvement or regression. Timing is host wall time, includes
synchronous trace emission and the repeated weight calculation, and does not isolate GPU compute or rule out CPU
fallback. This comparison does not measure the standalone stop-hook writer or compare all internal codebooks.

The story SHA-256 was `90e57e18382c982034cf5da44230689c89634576cb2daef2fb1d54b7679e5e3a`;
the talker and codec hashes were `33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67` and
`7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824`. Old and corrected executable hashes were
`b2144e305095756cee8255ea54c645457e9577093c5bc63c8edb3036081c7660` and
`0d401da1fdb5a262c9381a27779a8ccdcfcb9d19628440d4809e31b1dd9424d2` for CPU, and
`f3cb0a5a36b36e08e49a3f11a70a229660f3badd1a6bf495287a29f3727cd3e9` and
`d8d871cbf9ee765a679da1d8fcbca0cbce2f5bd3f3ecb5e441faac0132f83172` for CUDA.
Use `scripts/compare_stop_overhead.py --help` to repeat it. Its fresh scratch directory must be outside the checkout
and can contain audio and input text in traces. No audio, text or per-frame trace is committed here.
