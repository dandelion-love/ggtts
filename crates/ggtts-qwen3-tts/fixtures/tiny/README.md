# The tiny real-model fixture

**It checks identity, and nothing else.** This is the published v0.0.1 talker cut to one talker layer and one
code-predictor layer. A model cut that far cannot speak: it never emits the stop token, every request runs to its frame
cap, and its audio is noise. It exists so hosted CI can run the no-change check on a real engine: with and without an
instrument, at a fixed seed, the same codec tokens and the same decoded samples. Never judge audio, quality, timing or
termination from it.

## What it is

| | |
|---|---|
| Source | [`jmtl/hk-cantonese-cc0-qwen3-tts-GGUF`](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF) at commit `5a7660276b52d9efeed92d1c87f218d7a296110a` (its tag `v0.0.1`): the published files the CPU golden fixtures are recorded with, and that the promotion gate's instrument proof pins |
| Licence | Apache-2.0, as the model card at that commit declares (a fine-tune of the Apache-2.0 [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) on CC0 speech; see the card for its provenance) |
| Talker cut | `qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf` (2,042,834,112 bytes, 28 talker layers, 5 code-predictor layers, sha256 `33502b20…`) cut to 1 and 1: 63 of 404 tensors, **531,568,768 bytes** (507 MiB), sha256 `75b12656…` |
| Codec | `qwen-tokenizer-12hz-q8_0.gguf` as published (291,150,624 bytes, sha256 `7771c653…`), not cut |
| Pins | [`SOURCE`](SOURCE): the repository, the commit, both source files' sha256, the layer counts and the cut's sha256 |

The cut keeps every metadata key and every kept tensor byte for byte, and changes only the two layer counts. It
re-quantises nothing, so its weights are the published ones, under the published model's licence. The cut is
deterministic: the same input gives the same bytes, on any machine.

**Size.** The cut is 532 MB, more than "a few hundred MB". The floor is the text embedding: 315 MB, the whole
151,936-token vocabulary at Q8_0. The talker needs it for any prompt, and dropping rows would change token ids. Cutting
more layers saves little: one talker layer is 51 MB.

## What runs on it

The CI job `tiny-model` (`.github/workflows/ci.yml`), selected by `tiny_decision` in `scripts/select_specialized_ci.py`
when the engine, the core, ggml, the instruments, the CLI, the fixture or the workspace manifests change, and on every
promotion pull request. On CPU, it runs:

* `tests/tiny_model.rs`: the stop-probability trace changes no token and no sample, and a quiet frame hook changes no
  token, on 8 texts; the planted `FrameHookDraw` (a hook that draws from the sampler's stream) is red, and quiet
  without a hook.
* The three cases the promotion gate's CPU instrument proof (`scripts/instrument_promotion_gate.py`) runs on the full
  model: `frame_hook`'s stop-trace hook and hook-boundary logit plant, and `ggtts-cli`'s `no_change_real` (every event
  instrument in the catalogue, ASR excepted, leaves the tokens, the samples and the audio file as they were). The job
  sets `GGTTS_NO_CHANGE_MAX_FRAMES=120`, which lowers those cases' frame cap (1200 on the full model): the tiny model
  never stops, so at 1200 every request would generate and decode 100 s of noise. The gate leaves it unset.
* Two plants that must turn it red: a file that is not the pinned cut is refused, and an engine edited to sample
  differently whenever an instrument is attached fails the trace check, naming the texts.

Locally:

```text
eval "$(scripts/tiny_fixture.sh)"
cargo test --release -p ggtts-qwen3-tts --features ggml,plant --test tiny_model -- --ignored --nocapture
```

## Where it lives, and why not in the repository

**Nowhere new.** Nothing is uploaded and nothing is committed: the fixture is about 820 MB with the codec, over GitHub's
100 MB file limit and far too large to clone with the source. `scripts/tiny_fixture.sh` downloads the two published
files at the pinned commit, checks both against their sha256, cuts the talker with `scripts/cut_tiny_fixture.py`,
checks the cut's sha256, and prints the paths. CI caches the cut and the codec (`actions/cache`, keyed on the hash of
`SOURCE`), so a run after the first downloads nothing. The 2 GB source talker is deleted after the cut and never
cached.

It fails closed. A mismatch at any step stops the script (exit 1); a cached file is checked again on every run; and the
tests refuse any file that is not the pinned one, so a check never passes on the wrong bytes. What it depends on is
Hugging Face serving the published files on a cache miss: if that is down, the job is red, not skipped.

**To undo the decision** (host the cut instead of cutting it in CI): upload the cut as a release asset or to a Hugging
Face repository, point `scripts/tiny_fixture.sh`'s download of the cut at it by URL and sha256, and drop the source
talker's download. The pins in `SOURCE` and the tests need no change, because they check the cut's sha256 either way.
To remove the job, delete `tiny-model` from `ci.yml` and `tiny_decision` from the selector; the tests stay opt-in
(`#[ignore]`) and run locally as above.

## Regenerating it

The fixture is never edited by hand. After changing the layer counts (or the source): edit `SOURCE`, run
`scripts/tiny_fixture.sh` once (it stops at the cut's hash), and put the cut's printed sha256 in `CUT_SHA256`. The new
pins change the cache key, so CI cuts it afresh.
