# ggtts
An instrumented text-to-speech engine on ggml, written in Rust: one engine per model architecture, a pluggable compute stack, and token-exact parity tests

> **Status: walking skeleton.** The long-text pipeline (chunking, a seed per chunk, the judge, trim, joins, level
> and FLAC output) runs against a fake engine and passes the reference's recorded pipeline cases in CI, on Linux
> and Windows. No model engine exists yet, so nothing real renders. `ggtts-render --version` reports the ggml pin.

## What it is

ggtts turns text into speech with neural codec language models (Qwen3-TTS first), running on
[ggml](https://github.com/ggml-org/ggml). It is built around three ideas.

**Two plug points, kept apart.**

* **The architecture.** Each model family is one engine behind a shared interface: text and a seed in, codec tokens
  out, and audio from the tokens. A new family is a new engine crate. It is not a change to the core.
* **The compute stack.** Each family's forward pass runs on a backend chosen at build time or at run time. A backend
  is the *model runner* for that family: it loads the weights, runs a step and returns logits. ggml is the first
  backend and the reference. Pluggability sits at that level, never at single tensor operations. The sampler and
  the seed stay above the backend, so a seed means the same thing on every backend.

**A pipeline for long text.** Long text is split into chunks with a deterministic rule. Each chunk gets its own seed,
derived from the text and its position. Each chunk is rendered, judged, trimmed and joined, and the read is levelled
and written as FLAC with a provenance manifest. The defaults are a fixed, versioned protocol: a render can be
reproduced from its command line and its manifest. The core holds the rules and no protocol number: the chunk
ceilings, the pause table, the judge bars and the level target are a profile the engine declares, named
`{protocol}/{family}`, for example `story-v1/qwen3-tts`.

**Instruments, opt in.** For a story, `--trace FILE` writes the complete `ggtts-trace/1` event stream as JSON
Lines, including per-step values the engine supplies (Qwen supplies the token and stop probability). Kitten does not
yet emit per-step events. For `--text`, Qwen's `--trace` remains its older step-only diagnostic and is not a complete
event stream for instruments. With tracing off nothing is computed for it. With it on, the output must not change: an
instrument never draws from the sampler's random stream. The hook points, the event order and the two instrument
tiers are in [docs/instruments/hooks.md](docs/instruments/hooks.md), and the stream's JSON schema is
[schemas/ggtts-trace-1.schema.json](schemas/ggtts-trace-1.schema.json).

**Running instruments.** `ggtts-render --instrument NAME` (or `--instruments FILE.yaml`) runs instruments on the
finished render, each in its own process with its own log, a timeout and a crash boundary; the template config
`instruments/instruments.yaml` lists every shipped one. The Python instruments are scripts, run under
`GGTTS_PYTHON` if set, else `python3` (`python` on Windows), and need Python 3.8 or later on `PATH`. They are found in the `instruments/` directory:
next to `ggtts-render` in a release archive, or the repository's own when run from a checkout. Set
`GGTTS_INSTRUMENTS_DIR` to use another. If none is found, the instrument is reported failed with that reason and
the render's audio is still written. Writing one: [instruments/README.md](instruments/README.md).
When measuring existing audio with `ggtts-measure`, choose a new `--report` directory: an existing directory is
refused so report files cannot overwrite an input through a path, symlink or hard link.

## Binaries

| Binary | Job |
|---|---|
| `ggtts-render` | Text in; FLAC and a provenance manifest out |
| `ggtts-server` | An OpenAI-style `POST /v1/audio/speech` |
| `ggtts-bench` | Speed, parity checks against golden fixtures, and trace replay |
| `ggtts-convert` | A Hugging Face checkpoint to GGUF |

Only `ggtts-render` exists so far, and it prints its version and the ggml pin. Every flag will have a `GGTTS_ARG_*`
environment variable, after llama.cpp's `LLAMA_ARG_*` convention.

### Streaming: which stream

- **To listen: `ggtts-render --stream FILE|-`.** The audio leaves chunk by chunk while the render runs, levelled with
  one gain set from its first 5 s of audio, which are held back until then. Its loudness is close to the FLAC's
  but not equal, because the FLAC's gain is computed over the whole read. `--stream-level-seconds S` trades the
  two: more seconds give loudness closer to the FLAC's, fewer give earlier first audio, and `0` takes the first
  chunk's gain alone, which adds no latency but goes wrong when the first chunk is unlike the rest.
- **For parity checks: `--stream FILE --stream-raw`.** The pieces before the level step, bit-exact to the batch read
  before its level step. The FLAC is this stream times the gain printed at the end. Tests and `stream_probe.sh` ask
  for it by name.

## How parity is tested

The tests compare outputs, not impressions. They form a ladder:

1. **Codec tokens are identical.** For the same GGUF, text, seed and hardware class, the ggml backend must produce
   exactly the tokens a reference runtime produced. A mismatch reports the first chunk, frame and codebook that
   differ. A fixture recorded on a different model, ggml commit or hardware class is refused rather than compared.
2. **Decoder audio is identical** for identical tokens, or within a tolerance that is stated and justified where a
   kernel's reduction order differs.
3. **The pipeline matches its specification case by case.** Chunk boundaries, seeds, judge verdicts, trim bounds,
   gaps and levels are checked against recorded cases, using a fake engine with scripted output. A planted change,
   such as moving the chunk ceiling by one, must turn the suite red.
4. **Distributions, only where identity cannot hold**: across GPU models, or on a backend other than ggml. Such a
   backend declares its parity level as "distribution", and every render records the backend and its level.

The reference runtime is [qwen3-tts.cpp](https://github.com/Danmoreng/qwen3-tts.cpp) (MIT), which builds on ggml
commit `af97976` (v0.16.0+7). ggtts pins the same commit, because identical tokens need identical kernels.

Fixture text in this repository is written by a language model for the purpose. Fixtures hold text, seeds, token ids
and hashes, never audio.

## Building

```sh
cargo build
cargo test
cargo test -p ggtts-core --test pipeline_cases -- --nocapture   # the pipeline cases, with their summary
cargo run -p ggtts-cli --bin ggtts-render -- --version
```

`rust-toolchain.toml` pins the Rust toolchain. CI runs `cargo fmt`, `clippy -D warnings`, the tests on Linux and
Windows, a planted change (the story chunk ceiling set to 44 must turn the pipeline cases red), and a compile-only
CUDA build of ggml at the pin for sm_86 and sm_89. A tag `v*` builds release archives named with the ggml pin, each
with a `BUILDINFO.json` and a build provenance attestation (`gh attestation verify <file> -R dandelion-love/ggtts`).

## The pipeline cases

`fixtures/pipeline/*.json` hold the reference implementation's recorded cases: text, numbers, decisions and hashes,
with synthetic signals described by their generator, never as samples. `out-of-scope.json` names every recorded
case the core does not implement and why, and the test summary lists them, so what is not covered is visible.

## The Kitten-TTS 2 engine

`crates/ggtts-kitten` runs [Kitten-TTS 2](https://huggingface.co/KittenML/kitten-tts-2) on ggml, on the CPU or CUDA:
a 1.7B speech LM with ternary (BitNet-style) transformer weights, Chatterbox-Turbo's S3Gen decoder, and voice
cloning from a reference clip, with no Python at run time. `ggtts-render` selects it from the GGUF's
architecture (`kitten-tts-2`) and takes `--codec` for the decoder GGUF and `--voice` for a prepared reference.

It reports the parity level `distribution` against Kitten's PyTorch path, with deterministic checks under it: the
ternary weights survive conversion exactly; LM logits are within stated bars of float32 PyTorch; the S3Gen decode
is at 89 dB SNR with recorded noise. It is `exact` against itself: the same inputs and seed give a byte-identical
FLAC. What it needs, how to convert and render, and the measurements: [`crates/ggtts-kitten/README.md`](crates/ggtts-kitten/README.md)
and [`DESIGN.md`](crates/ggtts-kitten/DESIGN.md).

**The model's licence is not this repository's.** Kitten-TTS 2's weights are under the Stellon Labs Community
License, and their outputs, including audio rendered here, must not be used to train any other model. No weights,
GGUFs or rendered audio are kept in this repository.

## Licence

MIT. Third-party sources and what is taken from each are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
