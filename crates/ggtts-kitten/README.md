# ggtts-kitten: the Kitten-TTS 2 engine

[Kitten-TTS 2](https://huggingface.co/KittenML/kitten-tts-2) on ggml. It is a 1.7B Qwen3 speech language model
whose 196 transformer linears are ternary (BitNet-style: one of `{-s, 0, +s}`, one scale per 128 weights), plus
Chatterbox-Turbo's S3Gen decoder. It clones a voice from a short reference clip and its transcript. The language
model, the decoder and the preparation of the reference clip all run on ggml, on the CPU or CUDA. No Python is
needed at run time.

How it was ported, from where, and the parity levels with their measurements: [`DESIGN.md`](https://github.com/dandelion-love/ggtts/blob/main/crates/ggtts-kitten/DESIGN.md).
A release archive carries this file as `README-kitten.md`.

## What it needs

- **The model.** `KittenML/kitten-tts-2` at `c93cb7c271baa51d9ae8204d5277988906f235a2`: `config.json`, `lm/` and
  `speaker/model.safetensors`. A QAT export in Kitten's key layout works in place of `lm/model.safetensors`.
- **The decoder.** `ResembleAI/chatterbox-turbo` at `749d1c1a46eb10492095d68fbcf55691ccf137cd`:
  `s3gen_meanflow.safetensors`.
- **The tools**: a [release archive](https://github.com/dandelion-love/ggtts/releases) ships `ggtts-kitten-convert`
  and `ggtts-kitten` beside `ggtts-render`, so nothing below needs a build, Python or PyTorch. From source, build
  with the `ggml` feature: `git submodule update --init crates/ggtts-ggml-sys/ggml`, cmake and a C/C++ compiler. For
  `--device cuda`, use the `cuda` feature with ggml built or linked with CUDA (`GGML_LIB_DIR`, see `ggtts-ggml-sys`).
- **A reference clip**, as a lossless FLAC or WAV file of about 5 to 30 seconds of one speaker, and its transcript.
  The model's `voices/` holds clips, with their transcripts in `voices/voices.json`.

No ggtts-ready GGUFs are published; the steps below make them once. A GGUF converted before the tokenizer was
embedded is refused by `ggtts-render` ("this GGUF has no embedded tokenizer"): convert it again with the first step.

## Convert, prepare, render

```bash
K=path/to/the/unpacked/archive    # or, from source: cargo build --release -p ggtts-kitten -p ggtts-cli \
                                  #   --features ggtts-cli/ggml && K=target/release

# The language model to a GGUF: Q4_0 by default (exact for ternary weights), and the readback check runs after it.
$K/ggtts-kitten-convert --repo $KITTEN --out kitten-lm.gguf
# The decoder, the tokenizer, CAMPPlus and the pyannote speaker model to one GGUF.
$K/ggtts-kitten-convert --s3gen $TURBO/s3gen_meanflow.safetensors --pyannote $KITTEN/speaker/model.safetensors \
    --out kitten-s3gen.gguf
# A reference voice, once per clip.
$K/ggtts-kitten prepare --s3gguf kitten-s3gen.gguf --wav voice.flac --transcript "What the clip says." --out voice.json
# Text to FLAC.
$K/ggtts-render --model kitten-lm.gguf --codec kitten-s3gen.gguf --voice voice.json --seed 1234 \
    --text "The little lighthouse kept its lamp burning all through the stormy night." --out line.flac
```

The ternary matrices can be stored three ways, each exact: every trit and every per-128 scale of the source, which
the readback check proves after writing.

| `--layout` | Stored as | Bits per ternary weight | The 1.41B ternary weights | On the CPU |
|---|---|---|---|---|
| `q4_0` (the default) | `Q4_0` | 4.5 | 793 MB | passes the ternary parity bar |
| `tq2_0` | padded `TQ2_0` | 4.125 | 727 MB | below it (`Q8_K` activations) |
| `tq1_0` | padded `TQ1_0` (`tq1_0-pad128`) | 3.375 | 595 MB | below it, bit-identical to `tq2_0` |

On CUDA all three are held as the same exact f16 matrices. A `TQ1_0` file with one scale per 256 weights
(`tq1_0-256-experimental`) still loads, but it is lossy for this model, whose groups are 128 weights, and the
converter does not write one.

`ggtts-render --story FILE` renders a story on the core's story path, with a draft profile (`draft/kitten-tts-2`).
`ggtts-kitten render --lines FILE` writes one FLAC per line, with a timing report. Add `--device cuda` to any of
them.

## Parity

This engine reports the parity level **`distribution`**. It shares no kernels with a reference runtime, and sampled
tokens cannot match PyTorch's. Under that level, deterministic checks with bars written before measuring
(`ggtts-kitten parity`, `decoder-parity`, `hift-parity`, `cond-parity`, and the converter's readback):

| Check | Result on CPU |
|---|---|
| Ternary weights in the GGUF (any `--layout`) | equal to the source, trit for trit and scale bit for bit, on all 196 matrices |
| LM logits against float32 PyTorch, weights dequantized | per-layer relative L2 at most `1.8e-6` |
| LM logits against float32 PyTorch, ternary at run time (Q4_0) | cosine `0.9999091` (bar 0.9999), argmax 32/32 |
| S3Gen decode, with the noise recorded | waveform SNR `89.3 dB` (bar 40 dB) |
| Reference preparation | S3 tokens 250/250; speaker vectors at cosine `1.000000` |

A planted fault turns each of these checks red. The engine is also `exact` against itself: the same GGUF, voice,
text, seed and thread count give a byte-identical FLAC.

## Licence

This crate is MIT, and the sources it ports are listed in
[`THIRD_PARTY.md`](https://github.com/dandelion-love/ggtts/blob/main/THIRD_PARTY.md). **The model is not MIT.** Kitten-TTS
2's weights are under the
[Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md), and a GGUF
converted from them carries the same terms; ggtts passes them through and neither grants nor changes them. In
summary, and the licence itself governs:

- it allows research, non-commercial and limited commercial use. Commercial users register with Stellon Labs, and
  the licence ends for a licensee above USD 1,000,000 in annual revenue or in total funding;
- whoever distributes the model or something built from it gives the recipient a copy of the licence, keeps its
  attribution notice in a notice file, and displays "Powered by KittenML";
- neither the model nor its outputs, including audio rendered with this engine, may be used to create or improve a
  foundational generative AI model, or to train or distill another model (fine-tunes and low-rank adaptations of
  the model itself excepted).

The S3Gen decoder weights (ResembleAI/chatterbox-turbo) and the pyannote speaker model are MIT. This repository and
its release archives hold no weights, no GGUFs, no recordings and no rendered audio.
