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
the reference decoder's f32 samples (`reference/decode_ref.cpp`, which calls the reference library itself). The
model is the published `jmtl/hk-cantonese-cc0-qwen3-tts-GGUF` at `v0.0.1`; the checks refuse other files by hash.

```bash
git submodule update --init crates/ggtts-ggml-sys/ggml
export GGTTS_TALKER=.../qwen-talker-hk-cantonese-cc0-v0.0.1-q8_0.gguf   # sha256 33502b20...0b67
export GGTTS_CODEC=.../qwen-tokenizer-12hz-q8_0.gguf                     # sha256 7771c653...7824
cargo test --release -p ggtts-qwen3-tts --features ggml -- --ignored --nocapture
```

Measured on an i9-12900K, WSL2 Ubuntu 22.04, gcc 11.4, 4 threads, ggml built `-march=native`:

| Check | Result | Planted defects (`GGTTS_PLANT`) |
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
  reference built for CUDA and runs both checks and every plant on CUDA. One open question it answers: the
  reference normalizes the decoder's codebooks by writing through the tensor's data pointer, which is host memory
  only on CPU; here the same arithmetic goes through the backend API.
* **The reference's CPU path.** It moves the talker's last hidden row to the code predictor through a device tensor;
  here it goes through the host as the same f32 bits. Its replay graphs, dedicated schedulers and CUDA-only greedy
  chains are not ported. Instructions, voice cloning (ICL) and the streaming decoder are not ported.
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

The story path, on `fixtures/story/outing.txt` (five paragraphs written for the purpose): 8 chunks at the profile's
ceiling of 45, a seed per chunk from the story, no first-draw defects, joined, trimmed and levelled into 66.44 s of
audio in 110 s of rendering on 4 threads (RTF 1.66, 1.68 with model load). Rendered twice, the two FLAC files are
byte-identical (sha256 `bafda0ed…1645`).

Timed against the reference CLI on the same machine, the same 4 threads and the `bus` fixture text (144 frames),
three alternating rounds, medians: generate 11,043 ms against 10,759 ms (+2.6%), decode 6,424 ms against 6,271 ms
(+2.4%), wall clock including model load 18,937 ms against 18,670 ms (+1.4%). The tokens were identical every
round. (Measured with the same render code as an example, before it moved into `ggtts-render`.)
