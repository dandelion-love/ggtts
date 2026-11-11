# ggtts-qwen3-tts

The Qwen3-TTS engine. Without features: the `story-v1/qwen3-tts` profile and the text tokenizer. With the `ggml`
feature: the model on ggml's CPU backend, ported op for op from
[qwen3-tts.cpp](https://github.com/Danmoreng/qwen3-tts.cpp) at `16bb5afcd06311031c72a8488f8d59660dc2fb46`, so that
its outputs are bit-identical to the reference's on CPU.

| Module | What it ports |
|---|---|
| `codec` | the codec decoder, tokens to 24 kHz samples (`src/decoder/`, the non-streaming path) |
| `talker` | the prompt, the talker, the code predictor, the generation loop and the sampler (`src/transformer/`) |
| `tokenizer` | the text tokenizer and the TTS prompt wrappers (`src/text_tokenizer.cpp`, `src/tokenizer_unicode.cpp`) |
| `engine` | `Qwen3Tts`, the core's `Engine` trait over the three |

Each function names the reference file and lines it follows.

## Parity, and how to check it

The golden fixtures in `fixtures/cpu/` are eight texts written for the purpose, recorded with the reference CLI on
CPU (`scripts/record_fixtures.sh`): speaker `hk_cantonese_speaker`, `-l zh`, seed 20260905, `--max-tokens 1200`, the
CLI's sampler defaults, 4 threads, each recorded twice and identical. Each holds the codec tokens and the sha256 of
the reference decoder's f32 samples (`reference/decode_ref.cpp`, which calls the reference library itself), and the
thread count it was recorded with, which the checks then run at. The model is the published
`jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` at `v0.0.1`; the checks refuse other files by hash.

```bash
git submodule update --init crates/ggtts-ggml-sys/ggml
export GGTTS_TALKER=.../qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf   # sha256 33502b20...0b67
export GGTTS_CODEC=.../qwen-tokenizer-12hz-q8_0.gguf                     # sha256 7771c653...7824
cargo test --release -p ggtts-qwen3-tts --features ggml,plant -- --ignored --nocapture
scripts/identity_runs.sh > records/identity-cpu.log   # the same, one test at a time, as a record
```

Each check has expected-failure tests beside it, `planted::*`, built only with the test-only `plant` feature: each
plants one defect and passes only when the check then goes red. Without `plant` there is no way to plant a defect,
and `codec::Plant`, `talker::Plant` and `load_planted` do not exist.

Measured on an i9-12900K, WSL2 Ubuntu 22.04, gcc 11.4, 4 threads, ggml built `-march=native`
([`records/identity-cpu.log`](records/identity-cpu.log)):

| Check | Result | Planted defects (`planted::*`, each must turn the check red) |
|---|---|---|
| `decoder_identity`: samples, sha256 of the f32 bytes | 8 of 8 identical | `rope-normal` 0 of 8; `snake-no-epsilon` 0 of 8; `window-off-by-one` 2 of 8 identical, the two fixtures shorter than the 72-frame window, which cannot see it |
| `token_identity`: every codec token of every frame, and the frame count | 8 of 8 identical | `rope-normal` 0 of 8 (first difference at frame 0, codebooks 1 to 5); `sampler-off-by-one` 0 of 8 (frame 0, codebook 0) |

Both hold whether ggml is built by `ggtts-ggml-sys` or the reference's own archives are linked
(`GGML_LIB_DIR=<reference build>/ggml/src`).

## What the identity rests on, and its limits

* **One machine.** The fixtures and the checks ran on one CPU with `-march=native`. A different CPU, a different
  instruction set or a different compiler may change ggml's kernels and so the bits; re-record with
  `scripts/record_fixtures.sh` on the machine that checks.
* **CPU proven, CUDA not yet run.** The `cuda` feature builds ggml's CUDA backend (`GGML_CUDA_ARCH`, default 86)
  and `Device::Cuda` follows the reference's CUDA-only branches: weights and caches on the device with a CPU
  fallback in the scheduler, packed Q/K/V projections in the talker and code predictor step graphs, and the
  decoder's per-codebook projection from 64 frames up. `scripts/cuda_probe.sh` records the fixtures with the
  reference built for CUDA and runs both checks and every plant's expected-failure test on CUDA. One open question
  it answers: the reference normalizes the decoder's codebooks by writing through the tensor's data pointer, which
  is host memory only on CPU; here the same arithmetic goes through the backend API. Until it has passed, the
  engine reports its parity on CUDA as `unmeasured` (`EngineInfo.parity`), and `exact` only on CPU.
* **Sampling, never greedy.** `Sampling::check` refuses a temperature at or below zero (and one that is not a
  finite number), at `Qwen3Tts::load` and at every `Talker::generate`: greedy decoding of the talker and code
  predictor does not terminate on this model, it runs to the frame cap. The reference CLI accepts `--temperature 0`;
  this engine does not.
* **The reference's CPU path.** It moves the talker's last hidden row to the code predictor through a device tensor;
  here it goes through the host as the same f32 bits. Its replay graphs, dedicated schedulers and CUDA-only greedy
  chains are not ported. Instructions, voice cloning (ICL) and the reference's streaming decoder are not ported; ggtts streams
  differently, by re-decoding the prefix (`src/stream.rs`).
* **top-p below 1.** The reference sorts with `std::sort`, which is not stable; ties in probability may order
  differently here. The fixtures use top-p 1.0, which skips that branch.
* **The sampler's `expf`** comes from the platform's C library on both sides.

## Rendering end to end

```bash
cargo build --release -p ggtts-cli --features ggml
target/release/ggtts-render --model "$GGTTS_TALKER" --codec "$GGTTS_CODEC" --out out.flac -j 4 \
  --text "今日天氣好好，我哋去公園行下"
# or --story story.txt: title line, one paragraph per line, rendered on the core's story path
```

`ggtts-render` picks the engine from the model's `general.architecture`; `--engine` overrides it.
Use `--speaker hk_voice_343` with the published Cantonese v0.0.5 talker, or select another speaker named in
the model card explicitly. Without `--speaker`, the legacy `hk_cantonese_speaker` default is retained for v0.0.1.
An unknown speaker fails, and the error lists the model's speaker names; the renderer does not choose an arbitrary speaker from a multi-speaker model.

The story path, on `fixtures/story/outing.txt` (five paragraphs written for the purpose): 8 chunks at the profile's
ceiling of 45, a seed per chunk from the story, no first-draw defects, joined, trimmed and levelled into 66.44 s of
audio in 110 s of rendering on 4 threads (RTF 1.66, 1.68 with model load). Rendered twice, the two FLAC files are
byte-identical (sha256 `bafda0ed…1645`).

The shipped `ggtts-render`, timed against the reference CLI by `scripts/time_render.sh`
([`records/render-timing-cpu.log`](records/render-timing-cpu.log)): the same machine, the same 4 threads, the `bus`
fixture text and seed, 11 rounds alternating which tool goes first, both under `nice -n 19` on a machine with other
work running. Medians, with the range over the 11 rounds:

| | `ggtts-render` | reference CLI | difference |
|---|---|---|---|
| wall clock, model load and file write included | 20,095 ms (19,466–20,540) | 20,029 ms (19,300–21,239) | +0.3% |
| generate | 11,242 ms (10,941–11,564) | 11,054 ms (10,800–11,893) | +1.7% |
| decode | 6,678 ms (6,557–6,812) | 6,680 ms (6,587–6,874) | −0.0% |

Every run of both produced 144 frames and 276,480 samples. The ranges overlap on every row, so these say the two
are within a few percent of each other here, not which is faster. `ggtts-render` writes FLAC and the CLI writes WAV,
which the wall clock includes.
