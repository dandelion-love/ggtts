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
| Sampler | `kittenml/logits.py:24-117` (the PyTorch order), `cpp/tools/kitten-tts/sampling.h:41-90` | `sampler.rs`: token-run penalty, windowed repetition penalty exempting `speech_end` and `stop`, temperature, top-k, top-p, min-p, then a draw. One deliberate difference from the C++, which also exempts the silence token 4299 from the repetition penalty (`sampling.h:53`): `kittenml/logits.py:30-33` says silence is *not* exempt, on purpose, and this engine follows the PyTorch path |
| Stop | `cpp/tools/kitten-tts/main.cpp:198` | `speech_end` or `stop`; anything outside the audio range is dropped from the codes, as `kittenml/model.py` `_generate_tokens` does |
| Tokenizer | the model's `lm/tokenizer.json` (Qwen2 byte-level BPE) | a byte-level BPE in Rust over `vocab.json`, `merges.txt` and the added tokens, checked against the Hugging Face tokenizer's ids |
| Decoder (S3Gen) | `kittenml/vocoder.py:65-170`, `kittenml/s3gen.py` (the flow, the meanflow estimator, HiFT); `cpp/tools/kitten-tts/decoder.cpp:12-27` | `flow.rs`, `hift.rs`, `dsp.rs`: ported to ggml (2026-11-06), see below; the PyTorch bridge stays as the labelled fallback |

### The decoder, honestly

kitten-tts-2-cpp does not port S3Gen either: its decoder is a TorchScript export run through CPU LibTorch
(`cpp/tools/kitten-tts/decoder.cpp:6-27`, `cpp/README.md` "the S3 waveform decoder uses CPU LibTorch"), and its
voice cloning needs an offline Python step (`cpp/tools/kitten-tts/prepare_reference.py`). So there is no C++ S3Gen
to port from: a ggml S3Gen is a port from the PyTorch source, `kittenml/s3gen.py` (4,237 lines): the upsampling
conformer encoder, the meanflow conditional decoder (a 1-D U-Net with transformer blocks, two Euler steps), the HiFT
vocoder (an F0 predictor, a harmonic source, upsampling convolutions with Snake activations and an inverse STFT),
and for reference conditioning a mel front end, the CAMPPlus speaker encoder and the S3 tokenizer.

**Estimate (as first written): three to five working days** to port and hold to the decode parity below. Until
the port matched, the engine's `decode` handed the codes to Kitten's own PyTorch S3Gen on the CPU through
`tools/kitten_ref.py decode`, the same split kitten-tts-2-cpp makes with LibTorch.

**2026-11-06: the decode is ported, and the estimate was too high.** The vocoder and the flow run on ggml and
meet level 3 (see Measured). PyTorch S3Gen stays available with `--decoder pytorch`, and every render's report names
the decoder that made it. **Reference conditioning is ported too (2026-11-06).** `ggtts-kitten prepare` computes the clip's 24 kHz mel, CAMPPlus
x-vector, S3 tokens and pyannote embedding without Python; `tools/kitten_ref.py reference` remains as the reference
it is measured against. One documented difference: resampling is torchaudio's sinc everywhere, where Python uses
soxr for the LM prompt's 16 kHz tokens (and for a clip not already at 24 kHz).

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

5. **Reference-clip preparation without Python** (bars written 2026-11-06, before the port was run). The parity
   inputs are a 24 kHz clip and PyTorch's own intermediate signals, so each stage is measured on the same input:
   * 24 kHz to 16 kHz resampling (torchaudio's sinc): relative L2 at most `1e-5`.
   * S3Gen's 24 kHz prompt mel, the S3 tokenizer's log-mel, and Kaldi fbank: relative L2 at most `1e-4` each.
   * S3 tokens on PyTorch's 16 kHz signal: at least 98% equal. On this port's own resampling against librosa's
     soxr path, which the LM prompt uses in Python: at least 90% equal, because the resamplers differ.
   * CAMPPlus x-vector and pyannote embedding: cosine at least `0.9999` each.
   * **Planted faults:** no rotary embedding in the tokenizer (token agreement below 98%), unfolded BatchNorm in
     CAMPPlus (cosine below the bar), no absolute value after pyannote's sinc filters (cosine below the bar), and no
     pre-emphasis in the fbank (relative L2 above the bar). Each must turn its check red.

## Measured (2026-11-06, CPU, 4 threads, WSL Ubuntu on x86-64)

The bars above were written before these numbers and are not moved by them.

* **Level 1, converter: PASS.** Kitten `c93cb7c` `lm/model.safetensors` and a QAT smoke export each give
  `READBACK PASS: 196 ternary matrices, 11010048 groups of 128 equal to the source trit for trit and scale bit for
  bit; 118 other tensors equal`. Every scale had an exact fp16. Planted on the real weights, `--plant group256`
  reads `READBACK FAIL: model.layers.0.mlp.down_proj.weight row 0: group 0: scale 0.20800781 read back, source
  absmax 0.15429688`, and `--plant absmean` reads the same row with scale `0.12902832`.
* **Level 2, f32 arm: PASS.** On 328 teacher-forced positions (a built-in voice's prefix, one fixed sentence, 24
  forced audio tokens), the worst hidden-state relative L2 over the 27 layers and the final norm is `1.755e-6`.
  The logits RMSE prints as `0.0000`, every one of 32 argmaxes is equal, and the speaker projection's relative L2
  is `3.05e-7`.
* **Level 2, ternary arm: FAIL on one of its three bars.** RMSE `0.0505` (bar 0.06, pass); argmax `32/32` (bar 30,
  pass); minimum cosine `0.9995754` (bar 0.9999, **fail**). The cause was measured, not guessed. The same body with
  an f32 head gives cosine `0.9995511` and RMSE `0.0452`, so the error is the body's `Q8_K` activation
  quantization and not the bf16 head. Per-layer hidden error grows from `6.9e-3` (layer 1) to `1.2e-2` (layer 27).
  kitten-tts-2-cpp's `TQ2_1` path quantizes activations per 32 (`Q8_0`). This one quantizes per 128 real values,
  through `Q8_K` on a padded block. The planted group of 256 through this arm reads RMSE `1.2213`, cosine
  `0.8298` and argmax `25/32`: far outside the bar, as level 2 requires.
* **Tokenizer:** 7 of 7 texts give the Hugging Face tokenizer's ids. The texts include contractions, digits, a
  blank line, emphasis and event markup, accents and the emotion control.

* **End to end, ternary LM on ggml plus the PyTorch S3Gen bridge (CPU, 4 threads).** On a QAT smoke checkpoint,
  with a built-in voice as the reference and three sentences written for the purpose, every line stopped on
  `speech_end`. The lines gave 116, 116 and 100 codes and 4.74 s, 4.74 s and 4.10 s of audio. The LM ran at
  15.8-16.3 tokens/s (25 are needed for real time), with a prefill of about 300 tokens in 3.2-3.3 s. That is an
  LM real-time factor of 2.22-2.28. The bridge's decode took 15.4-17.0 s per line, most of it loading S3Gen in a
  new process each time, for a total real-time factor of 5.5-6.1. With every line decoded in one bridge process,
  the same three lines from a reference prepared from a raw clip gave a real-time factor of 4.27 overall: 14.16 s
  of audio, 38.0 s of LM at 12.1-12.6 tokens/s on a busier machine, and a 22.4 s bridge, of which 5.0 s was
  loading S3Gen and 2.8-4.7 s per line decoding. Nobody has listened to these renders and no ASR has scored them.
* **Reference from a clip:** `kitten_ref.py reference --wav --transcript` on a built-in voice's own clip
  reproduces that voice's shipped S3 tokens (253 of 253) and speaker embedding (cosine 1.0000001).
* **Level 4, self-determinism:** the same GGUF, lines, seed and thread count gave identical codes on separate
  runs, three lines of three, and the clip render reproduced its codes again on its rerun.

* **Decoder port, step 1: the HiFT vocoder runs on ggml.** `src/hift.rs` covers the f0 predictor, the
  upsampling, the source fusion, the Snake resblocks and the post convolution. `src/dsp.rs` covers the host-side
  harmonic source, the 16-point STFT and its inverse. Both were checked on a fixed-token PyTorch decode with the
  random source recorded. The f0's relative L2 is `4.71e-7` (bar 1e-3), and the waveform is at SNR `106.2 dB`
  against PyTorch's (bar 40 dB): **PASS**. It vocodes 4.1 s of audio in 1.23 s on 4 threads.
* **Decoder port, step 2: the flow runs on ggml, so the whole S3Gen decode does.** `src/flow.rs` covers the
  conformer encoder with relative-position attention and the 2x upsampling, and the meanflow U-Net estimator (56
  transformer blocks) over two Euler steps. On a 103-token decode against a 248-token prompt, with PyTorch's noise
  and source recorded, every relative L2 was measured:

  | Stage | Relative L2 |
  |---|---|
  | speaker affine | `1.2e-7` |
  | encoder | `5.1e-7` |
  | `mu` | `3.4e-7` |
  | one estimator call | `4.5e-7` |
  | final mel | `2.9e-7` (bar 1e-3) |

  The waveform is at SNR `89.3 dB` (bar 40 dB): **level 3 PASS**. It decodes 4.10 s of audio in 3.63 s on 4
  threads (RTF 0.89). Reference conditioning (the clip's mel, CAMPPlus x-vector and S3 tokens) is still prepared
  by the PyTorch tool, as in kitten-tts-2-cpp.

* **End to end on ggml, LM and decoder (CPU, 4 threads, QAT smoke checkpoint).** The same three lines and seeds
  gave the same codes as before, as level 4 requires. Rendered with the ggml decoder, the RTF overall was 2.86
  (built-in voice) and 2.91 (reference from a clip). The LM ran at 16.9-18.7 tokens/s; decoding took 3.8-4.3 s
  per line of 4.1-5.0 s. The decoder's noise and source are this crate's own draws, so the audio is not sample-equal
  to PyTorch's. Against the PyTorch-decoded render of the same codes, it has the same length, RMS within 0.3 dB,
  and an amplitude-envelope correlation of 0.975-0.988 at 20 ms.

* **Reference-clip preparation without Python (level 5): PASS.** `src/refdsp.rs` covers WAV reading, torchaudio's
  sinc resampler, the Slaney mel filterbank, the 24 kHz prompt mel, the tokenizer's log-mel and Kaldi fbank.
  `src/condnet.rs` runs the S3 tokenizer v2, CAMPPlus and pyannote's SincNet x-vector on ggml. On a built-in
  voice's 10 s clip, measured against PyTorch's intermediate signals:

  | Stage | Result | Bar |
  |---|---|---|
  | resampling | relative L2 `8.9e-8` | 1e-5 |
  | prompt mel | relative L2 `2.9e-6` | 1e-4 |
  | tokenizer log-mel | relative L2 `1.5e-6` | 1e-4 |
  | fbank | relative L2 `5.1e-6` | 1e-4 |
  | S3 tokens | 250 of 250 equal | 98% |
  | LM-prompt tokens (own resampling against librosa's soxr) | 95.7% equal | 90% |
  | CAMPPlus x-vector | cosine `1.000000` | 0.9999 |
  | pyannote embedding | cosine `1.000000` | 0.9999 |

  Each plant turned its check red:

  | Plant | Result |
  |---|---|
  | no rotary embedding | tokens 32.0% equal |
  | unfolded BatchNorm | x-vector cosine `0.0017` |
  | no absolute value after the sinc filters | pyannote cosine `0.883` |
  | no pre-emphasis | fbank relative L2 `0.175` |

  `ggtts-kitten prepare` prepares the clip in 1.6 s. Against the Python-prepared reference, the S3Gen prompt tokens
  are equal 250 of 250 and the LM-prompt tokens 242 of 253, the speaker vectors have cosine `0.99999996`
  (pyannote) and `0.9999999998` (x-vector), and the prompt mel's relative L2 is `2.9e-6`. A render of the QAT
  smoke checkpoint from that reference ran with no Python process at all: 13.88 s of audio at RTF 3.51 on a
  busier machine.

**The ternary arm's cosine gap, closed by stock `Q4_0` (2026-11-06).** The measured cause was activation
quantization: `Q8_K` over 128 real values. kitten-tts-2-cpp's `TQ2_1`-on-AMX path quantizes per 32 (`Q8_0`) and
reports a cosine above 0.999998. `Q4_0` holds ternary weights exactly: each 32-weight block's `d` is the group's
scale, with codes 7, 8 and 9. It needs no padding and its activations are `Q8_0`. Measured on the same inputs and
against the unchanged bars:

| Layout | Readback | Logits RMSE | Min cosine | Argmax | Verdict | GGUF | Prefill (300 tokens) | Generation |
|---|---|---|---|---|---|---|---|---|
| padded `TQ2_0` | PASS | 0.0505 | 0.9995754 | 32/32 | **FAIL** (cosine) | 1380 MB | 93-100 tokens/s | 13.4-17.8 tokens/s |
| `Q4_0` | PASS | 0.0473 | **0.9999091** | 32/32 | **PASS** | 1446 MB | 79-82 tokens/s | 16.2-17.3 tokens/s |

Generation speed is the same within the noise of three alternating runs on a shared machine. Prefill is about 20%
slower on `Q4_0`, and its file is 5% larger. **`Q4_0` is now the converter's default**, because it is the only
layout that passes. `--layout tq2_0` keeps the padded form. A planted group of 256 reads back red on `Q4_0` too, and
through its parity arm gives RMSE `1.2201`, cosine `0.8327` and argmax `25/32`. Both are `ggml` stock types; nothing
is patched.

## Layout

| Path | What |
|---|---|
| `src/safetensors.rs` | a minimal safetensors reader (header, dtype, shape, bytes) |
| `src/gguf.rs` | a GGUF v3 writer and reader for the types used here |
| `src/ternary.rs` | Kitten's rule, `TQ2_0` padded packing and unpacking |
| `src/convert.rs` | Kitten's LM directory or a QAT export to a ggtts GGUF, and the readback check |
| `src/model.rs` | the ggml graph and KV-cache session (feature `ggml`) |
| `src/prompt.rs`, `src/sampler.rs`, `src/tokenizer.rs` | the prompt, the sampler in PyTorch's order, the BPE |
| `src/engine.rs` | `KittenEngine`, the `Engine` trait, and the PyTorch S3Gen bridge (feature `ggml`) |
| `src/parity.rs` | the parity measures and the bars above |
| `src/s3convert.rs` | S3Gen's safetensors to a GGUF, weight norm folded |
| `src/flow.rs` | the S3Gen flow on ggml: conformer encoder and meanflow estimator (feature `ggml`) |
| `src/refdsp.rs`, `src/condnet.rs` | reference-clip preparation: resampling, mels, fbank; the S3 tokenizer, CAMPPlus, pyannote (feature `ggml`) |
| `src/hift.rs`, `src/dsp.rs` | the HiFT vocoder on ggml (feature `ggml`) and its host-side source and STFT |
| `src/bin/ggtts-kitten-convert.rs` | the converter's command line |
| `src/bin/ggtts-kitten.rs` | `parity`, `tokcheck` and `render` (feature `ggml`) |
| `tools/kitten_ref.py` | the PyTorch side: reference dumps, reference-clip preparation, the S3Gen bridge |

The QAT export (`ternary_bf16.safetensors`, the whole LM in Kitten's key layout with every ternary-born linear as
`quantize(master)` in bf16) is the same layout as Kitten's `lm/model.safetensors`, so one converter takes both; the
rest of the model directory (config, tokenizer, voices) comes from the base model.
