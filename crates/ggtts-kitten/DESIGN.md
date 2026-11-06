# ggtts-kitten: the Kitten-TTS 2 engine

Kitten-TTS 2 is the second architecture behind ggtts's `Engine` trait. It is a 1.7B Qwen3 language model over a
shared text and audio vocabulary, whose transformer linears are ternary (BitNet-style: each weight is one of
`{-s, 0, +s}` with one scale `s` per 128 weights). It reads text and writes S3 codec tokens (25 Hz, 6561 ids, one
codebook). Chatterbox-Turbo's S3Gen turns those tokens into 24 kHz audio. A voice reaches the model twice: as a
512-dim speaker embedding projected into the first prompt position, and as the reference clip's transcript and S3
tokens in the prompt.

This note is written before the measurements it promises. Where a result changes a statement here, the statement
is corrected in place with the date, never silently.

## Sources and pins

| Source | Pin | Licence | Used for |
|---|---|---|---|
| [KittenML/kitten-tts-2-cpp](https://github.com/KittenML/kitten-tts-2-cpp) | `2b9912e72245f5597c664d2be44a6e7ba0a3f293` | MIT | the C++ reference: prompt, sampler, TQ2_1, export rules |
| KittenML/kitten-tts-2 (Hugging Face) | `c93cb7c271baa51d9ae8204d5277988906f235a2` | Stellon Labs Community License | the weights, config, tokenizer, voices |
| `kittenml` Python package | 0.9.7 | Apache-2.0 | the PyTorch path the parity checks measure against |
| ResembleAI/chatterbox-turbo (Hugging Face) | `749d1c1a46eb10492095d68fbcf55691ccf137cd` | MIT | S3Gen, the decoder (`s3gen_meanflow.safetensors`) |
| ggml | `af97976c7810cdabb1863172f31c432dab767de7` (the ggtts pin) | MIT | the compute |

Citations below are `file:line` at those pins. `cpp/` means kitten-tts-2-cpp, `kittenml/` means
`kittenml/kittentts2/` in the Python package, `ggml/` means ggml at the ggtts pin.

**Nothing from a Kitten model enters this repository**: no weights, no GGUF, no rendered audio, no dumps. Kitten's
weights are under the Stellon Labs Community License, and its outputs may not be used to train any model. The
converter, the tests and the tools read a model from a path the user gives; tests that need one are skipped when
the path is not set.

## What is ported

| Part | Reference | Here |
|---|---|---|
| Token layout | `kittenml/tokens.py:22-60`, model `config.json` `token_map` | read from the model's `config.json` into the GGUF's metadata |
| Prompt | `cpp/tools/kitten-tts/sampling.h:22-35`, `kittenml/prompt.py:10-56` | `prompt.rs`, the same order: `[start]`, the reference transcript and S3 tokens between the reference-role markers, the optional emotion control, `text_start`, the target text, `final_seg`, `speech_start` |
| Speaker injection | `cpp/tools/kitten-tts/main.cpp:181-182` (a 2048-float embedding at position 0), `kittenml/speaker.py:54-104` (`spk_proj` = Linear 512->2048 + LayerNorm, written over position 0 at prefill only) | `spk_proj` is carried in the GGUF and run in the graph; its output replaces position 0's token embedding |
| LM | Qwen3, `lm/config.json`: 28 layers, hidden 2048, 16 query heads and 8 KV heads of 128, FFN 6144, RMSNorm eps 1e-6 with per-head q/k norms, NeoX RoPE theta 1e6, tied embedding of 158,241 rows | one ggml graph with a KV cache, prefill and single-token steps |
| Sampler | `kittenml/logits.py:24-117` (the PyTorch order), `cpp/tools/kitten-tts/sampling.h:41-90` | `sampler.rs`: token-run penalty, windowed repetition penalty exempting `speech_end` and `stop`, temperature, top-k, top-p, min-p, then a draw. One deliberate difference from the C++: it also exempts the silence token 4299 (`sampling.h:53`); `kittenml/logits.py:30-33` says silence is *not* exempt on purpose, and this engine follows the PyTorch path |
| Stop | `cpp/tools/kitten-tts/main.cpp:198` | `speech_end` or `stop`; anything outside the audio range is dropped from the codes, as `kittenml/model.py` `_generate_tokens` does |
| Tokenizer | the model's `lm/tokenizer.json` (Qwen2 byte-level BPE) | a byte-level BPE in Rust over `vocab.json`, `merges.txt` and the added tokens, checked against the Hugging Face tokenizer's ids |
| Decoder (S3Gen) | `kittenml/vocoder.py:65-170`, `kittenml/s3gen.py`; `cpp/tools/kitten-tts/decoder.cpp:12-27` | **not ported yet**, see below |

### The decoder, honestly

kitten-tts-2-cpp does not port S3Gen either: its decoder is a TorchScript export run through CPU LibTorch
(`cpp/tools/kitten-tts/decoder.cpp:6-27`, `cpp/README.md` "the S3 waveform decoder uses CPU LibTorch"), and its
voice cloning needs an offline Python step (`cpp/tools/kitten-tts/prepare_reference.py`). So there is no C++ S3Gen
to port from: a ggml S3Gen is a port from the PyTorch source, `kittenml/s3gen.py` (4,237 lines): the upsampling
conformer encoder, the meanflow conditional decoder (a 1-D U-Net with transformer blocks, two Euler steps), the HiFT
vocoder (an F0 predictor, a harmonic source, upsampling convolutions with Snake activations and an inverse STFT),
and for reference conditioning a mel front end, the CAMPPlus speaker encoder and the S3 tokenizer.

**Estimate: three to five working days** to port and hold to the decode parity below. Until then the engine's
`decode` hands the codes to Kitten's own PyTorch S3Gen on the CPU through `tools/kitten_ref.py decode`, the same
split kitten-tts-2-cpp makes with LibTorch, and reference conditioning is prepared once per clip by
`tools/kitten_ref.py reference`, as kitten-tts-2-cpp's `prepare_reference.py` does. That bridge is the smallest
honest alternative: the language model, the part that is ternary, runs entirely on ggml; the decoder is labelled
as not yet ggml in every render's report, never passed off as ported.

## The ternary type

**Kitten's rule** (`kittenml/ternary.py`; `cpp/ggml/src/ggml-common.h:289-295`; `cpp/ggml/src/ggml-quants.c:2382-2402`):
the 196 transformer linears (`q/k/v/o_proj`, `gate/up/down_proj` in 28 layers) are ternary; one scale per 128
consecutive weights along the input dimension of an output row; the scale is the group's absmax; a weight's code is
`round_half_away_from_zero(w / absmax)`, in `{-1, 0, +1}`. The token embedding (tied to the head), every norm and
`spk_proj` are not ternary (`cpp/tools/kitten-tts/export.py:67-73` forces the packed type on `model.layers.*` with
two dimensions only).

**What ggml has at the pin.** `TQ1_0` and `TQ2_0` are super-block types: 256 weights per block and one fp16 scale
(`ggml/src/ggml-common.h:276-288`, `QK_K` = 256 at `ggml/src/ggml-common.h:89`). Neither can hold two scales for
one 256-weight block, so stored directly they would change Kitten's weights wherever the two halves of a block have
different scales, which is almost everywhere. kitten-tts-2-cpp solves this with a new block type, `TQ2_1` (two fp16
scales per 256 weights, `cpp/ggml/src/ggml-common.h:289-295`, type 43), which stock ggml cannot load.

**What this engine uses: `TQ2_0` on zero-padded groups.** `TQ2_0`'s 64 code bytes are two halves of 32 bytes, and
each half codes exactly 128 consecutive weights (`ggml/src/ggml-quants.c:2399-2410`: byte `j + m`, bit pair `n`,
codes weight `j/32 * 128 + n*32 + m`). So a 256-weight block whose first half is one of Kitten's groups and whose
second half is zeros holds that group exactly, with the block's one fp16 scale equal to the group's scale. The
converter writes every ternary matrix of `in` columns as a `TQ2_0` matrix of `2*in` columns, group `g` at columns
`[256g, 256g+128)` and zeros after it; the graph pads each activation row the same way (`ggml_pad` on a
`[128, in/128]` view) before the matrix product. The product is unchanged: the padding contributes `0 * 0`.

Why this and not a port of `TQ2_1`:

* **It needs no change to ggml.** ggtts depends on ggml at a pin and nothing else; a new block type means carrying a
  patch on that pin across every kernel that dispatches on type (CPU, and CUDA later), and the patch is not
  upstream. Padding uses `TQ2_0`'s existing CPU and CUDA kernels.
* **It is exact.** The trits are Kitten's trits and the scales are Kitten's scales, provided each scale is exactly
  representable in fp16. Kitten's scales are bf16 values (the folded weights are bf16); a bf16 value is exact in
  fp16 when it is in fp16's normal range, and the converter refuses any scale that is not, rather than rounding it.
  kitten-tts-2-cpp's exporter makes the same refusal for `TQ2_1` (`cpp/tools/kitten-tts/README.md:74`).
* **What it costs.** The body is 4.125 bits per weight on disk and in memory instead of `TQ2_1`'s 2.125 (about 727
  MB instead of 375 MB for the 1.41B ternary weights), and the matrix products touch twice the bytes. The activation
  side is unchanged in effect: ggml quantizes activations to `Q8_K` per 256 values, and a padded block's absmax is
  its 128 real values' absmax, so each group's activations are quantized exactly as they would be per 128.

`TQ2_1` stays the better format for size and speed. If the padding's cost matters, the move is a `TQ2_1` patch on
the ggml pin with its upstreaming plan, recorded as a change to "depending only on ggml", not a quiet fork.

The GGUF also records `kitten.ternary.group_size` = 128 and `kitten.ternary.layout` = `tq2_0-pad128`; the loader
refuses a file whose layout it does not know.

## Generate and decode stay separate

`Engine::generate` takes text and a seed and returns `Codes` (one codebook, frame-major, ids 0..6560) and why it
stopped; `Engine::decode` takes `Codes` and returns PCM. Nothing in `generate` touches audio and nothing in
`decode` touches the language model, so the parity ladder can check tokens and audio independently: LM logits
against PyTorch with the decoder absent, and the decoder on fixed tokens with the LM absent. The reference voice is
state of the engine (set once, used by both halves: its prompt tokens and speaker embedding by `generate`, its
conditioning by `decode`), not an argument that couples them.

## The parity levels, stated before measuring

ggtts's parity ladder names two levels an engine can report: `exact` against golden fixtures, and `distribution`
where identity cannot hold. Against Kitten's PyTorch path this engine's level is **`distribution`**: there is no
reference runtime whose kernels it shares (kitten-tts-2-cpp is llama.cpp's kernels with `TQ2_1`), and sampled
tokens cannot match across runtimes (`cpp/README.md`, "Current scope"). Under that level sit component checks with
tolerances fixed here, before any measurement:

1. **Converter, exact.** For every one of the 196 ternary matrices, every trit read back from the GGUF equals the
   trit of the source weight and every block scale read back, widened to f32, has the same bits as the source
   group's absmax widened to f32. Every padding half reads back as zero codes. Non-ternary tensors read back equal
   to the source's values (the bf16 embedding bit for bit; norms and `spk_proj` as f32 widened from bf16). One
   caveat, stated rather than hidden: the bf16 source carries signed zeros (about half its zeros are `-0.0`), and an
   integer trit has no sign, so the readback is value-equal on zeros, as kitten-tts-2-cpp's `TQ2_1` and kittenml's
   own packer are.
   **Planted faults:** a group size of 256 instead of 128 (the scale becomes the absmax of two groups, so a weight
   from the smaller group no longer maps onto a trit) and a scale from the mean of absolute values instead of the
   max (a folded group shrinks by its nonzero fraction). Each must turn the check red.
2. **LM graph, teacher-forced logits, on fixed inputs** (a built-in voice's reference prefix, one fixed text, 24
   forced reference audio tokens; about 300 positions), against PyTorch in float32 on the CPU with the same weights.
   Two arms:
   * **Weights exact (`f32`)**: the GGUF's ternary matrices dequantized to f32 at load, everything else f32. This
     isolates the graph from the arithmetic. Bar: the hidden state after every layer within relative L2 error
     `1e-3` at every position, and the last 32 positions' logits within RMSE `0.01` with every argmax equal.
   * **Ternary runtime (`tq2_0`)**: the padded `TQ2_0` matrices with ggml's `Q8_K` activations, and the bf16
     embedding and head. Bar: logits RMSE at most `0.06`, cosine similarity at least `0.9999` on each of the 32
     positions, and argmax equal on at least 30 of 32. kitten-tts-2-cpp reports RMSE 0.022-0.029 and cosine above
     0.999998 for its `TQ2_1` path against float32 Python (`cpp/tools/kitten-tts/README.md:174`); these bars leave
     room for `Q8_K` versus `Q8_0` activations and a bf16 head, and no more.
   * The planted wrong group size must fail the ternary arm's bar, not merely the converter.
3. **Decoder, deterministic, on fixed tokens.** For the PyTorch bridge this is identity by construction (it is the
   reference). For a ggml S3Gen, when it exists: fixed tokens, fixed reference conditioning and a fixed noise draw,
   mel within relative L2 `1e-3` and waveform within SNR 40 dB of `S3Codec.decode`. Stated now so the port is held
   to a bar it did not choose.
4. **Sampled renders: distribution.** Token-exact parity with PyTorch is not claimed. ggtts holds this engine to
   `exact` against itself: the same GGUF, text, seed, thread count and build give the same tokens, checked by a
   golden fixture of token ids on text written for the purpose.

## Layout

| Path | What |
|---|---|
| `src/safetensors.rs` | a minimal safetensors reader (header, dtype, shape, bytes) |
| `src/gguf.rs` | a GGUF v3 writer and reader for the types used here |
| `src/ternary.rs` | Kitten's rule, `TQ2_0` padded packing and unpacking |
| `src/convert.rs` | Kitten's LM directory or a QAT export to a ggtts GGUF, and the readback check |
| `src/bin/ggtts-kitten-convert.rs` | the converter's command line |
| `tools/kitten_ref.py` | the PyTorch side: reference dumps, reference-clip preparation, the S3Gen bridge |

The QAT export (`ternary_bf16.safetensors`, the whole LM in Kitten's key layout with every ternary-born linear as
`quantize(master)` in bf16) is the same layout as Kitten's `lm/model.safetensors`, so one converter takes both; the
rest of the model directory (config, tokenizer, voices) comes from the base model.
