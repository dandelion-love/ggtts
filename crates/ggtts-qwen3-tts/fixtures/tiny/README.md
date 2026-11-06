# The tiny real-model fixture

**It checks identity, and nothing else.** This is the published v0.0.1 talker cut to one talker layer and one
code-predictor layer. A model cut that far cannot speak, and its audio is noise. It exists so CI can run the
no-change check on a real engine: with and without an instrument, at a fixed seed, the same codec tokens and the
same decoded samples (`tests/tiny_model.rs`). Never judge audio, quality, timing or termination from it.

## What it is

| | |
|---|---|
| Source | [`jmtl/hk-cantonese-cc0-qwen3-tts-GGUF`](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF) at revision `5a766027` (tag `v0.0.1`), the same published files the CPU golden fixtures are recorded with |
| Talker cut | `qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf` (2.04 GB, 28 talker layers, 5 code-predictor layers) cut to 1 and 1: 63 of 404 tensors, 532 MB (507 MiB) |
| Codec | `qwen-tokenizer-12hz-q8_0.gguf` as published (291 MB), not cut |
| Pins | `SOURCE`: the repository, revision, both source hashes, the layer counts and the cut's sha256 |

The cut keeps every metadata key and every kept tensor byte for byte, and changes only the two layer counts. It
re-quantises nothing, so its weights are the published ones, under the published model's licence and provenance (see
its model card).

**Size.** The cut is 532 MB, more than the issue's "a few hundred MB". The floor is the text embedding: 315 MB, the
whole 151,936-token vocabulary at Q8_0. The talker needs it for any prompt, and dropping rows would change token ids.
Cutting more layers saves little: one talker layer is 51 MB.

## Where it lives

**Nowhere new.** The fixture is not uploaded anywhere. `scripts/tiny_fixture.sh` downloads the two published files at
the pinned revision, checks both hashes, cuts the talker with `scripts/cut_tiny_fixture.py`, checks the cut's hash,
and prints the paths. CI caches the result, keyed on `SOURCE`. A hash mismatch at any step stops the script, and the
test refuses a file that is not the pinned one, so the check fails closed and never passes on the wrong bytes.

Rebuilding the fixture after changing the layer counts (or the source): edit `SOURCE`, run the script once, and put
the cut's printed sha256 in `CUT_SHA256`. The fixture is never edited by hand.
