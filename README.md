# ggtts
An instrumented text-to-speech engine on ggml, written in Rust: one engine per model architecture, a pluggable compute stack, and token-exact parity tests

> **Status: skeleton.** The workspace builds and `ggtts-render --version` reports the ggml pin. Nothing renders
> yet. The design is under review before engine code is written.

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
reproduced from its command line and its manifest.

**Instruments, opt in.** `--trace` writes per-step events (the sampled token, the stop probability, entropy and
more) as JSON Lines. With tracing off nothing is computed for it. With it on, the output must not change: an
instrument never draws from the sampler's random stream.

## Binaries

| Binary | Job |
|---|---|
| `ggtts-render` | Text in; FLAC and a provenance manifest out |
| `ggtts-server` | An OpenAI-style `POST /v1/audio/speech` |
| `ggtts-bench` | Speed, parity checks against golden fixtures, and trace replay |
| `ggtts-convert` | A Hugging Face checkpoint to GGUF |

Only `ggtts-render` exists so far, and it prints its version and the ggml pin. Every flag will have a `GGTTS_ARG_*`
environment variable, after llama.cpp's `LLAMA_ARG_*` convention.

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
cargo run -p ggtts-cli --bin ggtts-render -- --version
```

## Licence

MIT. Third-party sources and what is taken from each will be listed in `THIRD_PARTY.md` as they are added.
