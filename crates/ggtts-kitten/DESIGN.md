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
| [KittenML/kitten-tts-2-cpp](https://github.com/KittenML/kitten-tts-2-cpp) | `2b9912e72245f5597c664d2be44a6e7ba0a3f293` | Apache-2.0 (`tools/kitten-tts/LICENSE-KITTENTTS`; the repository root is llama.cpp's MIT) | the C++ reference: prompt, sampler, TQ2_1, export rules |
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

**`TQ1_0` on zero-padded groups (`--layout tq1_0`, layout `tq1_0-pad128`).** The same arrangement in ggml's other
ternary type. `TQ1_0` packs five trits a byte (54 bytes per 256 weights, `ggml/src/ggml-common.h:276-281`), and its
first 32 code bytes hold weights `m + 32n` for `n` in 0..5 (`ggml/src/ggml-quants.c`, `quantize_row_tq1_0_ref`), so a
group of 128 lies in the first half of a block and the second half is 128 zero trits. The block's scale is the
group's. The converter writes each matrix of `in` columns as a `TQ1_0` matrix of `2*in` columns, and the graph pads
the activations exactly as for `TQ2_0`.

Stored without padding, `TQ1_0` cannot hold Kitten's weights: a 256-weight block has one scale and two groups.
Measured on a 1.7B QAT export in Kitten's key layout (196 matrices, 11,010,048 groups of 128, every value on its
group's `trit * absmax` grid): of the 5,505,024 pairs of adjacent groups a 256-weight block would join, 103,112
(1.87%) have equal scales; the larger scale over the smaller has median 1.076, p99 1.505 and maximum 10.59; no row
and no matrix has a single scale. A 256-weight `TQ1_0` file of that export made outside ggtts differed from it in
496,172,378 of its 1,409,286,144 ternary values. The engine still reads such files (`tq1_0-256-experimental`,
[docs/tq1-cpu.md](../../docs/tq1-cpu.md)); the converter does not write them.

| Layout | `--layout` | Exact | Bits per ternary weight | The 1.41B ternary weights |
|---|---|---|---|---|
| `Q4_0` (the default) | `q4_0` | yes | 4.5 | 793 MB |
| padded `TQ2_0` | `tq2_0` | yes | 4.125 | 727 MB |
| padded `TQ1_0` | `tq1_0` | yes | 3.375 | 595 MB |
| `TQ1_0`, 256 weights a scale | not written | no (two groups share a scale) | 1.6875 | 297 MB |

On the CPU, padded `TQ1_0` and padded `TQ2_0` hold the same trits and scales and both take `Q8_K` activations, and
their products are bit-identical (`tests/tq1_padded.rs`, `tests/tq1_load.rs`). So padded `TQ1_0` inherits padded
`TQ2_0`'s ternary-arm result below: it misses the cosine bar that `Q4_0` passes, which is why `Q4_0` stays the
default. That result is inferred from the bit-identity on synthetic matrices; no parity run of padded `TQ1_0` on the
real weights has been made. On CUDA every layout is held as the same exact f16 matrices. The readback check covers padded `TQ1_0` as
it does the others: canonical code bytes, zero padding trits, every trit equal to the source's, every scale equal
bit for bit, every value equal through the runtime's decoder.

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
   * **Single-token path** (bar written 2026-11-06, after a CUDA render failed on it and before this check was
     first run). Both arms above run one batched forward, which never takes the path every generated token takes:
     one column at a time against the KV cache. So the same positions are also run both ways on the same model and
     device: a prefill, then the remaining positions as one batched forward in one session and one token at a time in
     another. Bar: every logit finite, and at each compared position the two logit vectors' cosine at least
     `0.99999` with the argmax equal. The paths share weights and inputs, so this is a check on the kernels and the
     cache, not on the model, and it is held tighter than the arms against PyTorch. `ggtts-kitten step-parity` and
     `tests/step_identity.rs` run it.
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

6. **CUDA** (bars written 2026-11-06, before any CUDA run; the first run's failures and the changes they led to are
   under "CUDA precision" below, and no bar moved). `--device cuda` runs every graph on the first CUDA
   device, with the CPU as the scheduler's fallback for ops CUDA lacks. It is held to the same bars as the CPU
   path: levels 1, 2, 3 and 5 against the same CPU PyTorch recordings, with the same plants red. Level 4 also holds
   on CUDA: the same seed must render a byte-identical FLAC twice. Matching the CPU render byte for byte is not
   claimed, because the kernels differ. `scripts/cuda_probe.sh` runs all of it in one command.

## Measured (2026-11-06, CPU, 4 threads, WSL Ubuntu on x86-64)

The bars above were written before these numbers and are not moved by them.

* **Level 1, converter: PASS.** Kitten `c93cb7c` `lm/model.safetensors` and a QAT smoke export each give
  `READBACK PASS: 196 ternary matrices, 11010048 groups of 128 equal to the source trit for trit and scale bit for
  bit; 118 other tensors equal`. Every scale had an exact fp16. Planted on the real weights, `--plant group256`
  reads `READBACK FAIL: model.layers.0.mlp.down_proj.weight row 0: group 0: scale 0.20800781 read back, source
  absmax 0.15429688`, and `--plant absmean` reads the same row with scale `0.12902832`.
* **Level 1 against Kitten's own packers (2026-11-06): PASS.** The readback above checks the GGUF against the bf16
  source and Kitten's rule as this crate implements it. That is not the same as Kitten's packed values. Kitten
  publishes no packed file: its packed form is whatever its exporter writes from the bf16 weights.
  `tools/kitten_packed_check.py` runs that exporter's own code from a kitten-tts-2-cpp checkout at the pin: the
  `TQ2_1` quantizer in its `gguf-py` (`gguf/quants.py`, class `TQ2_1`), which is the exporter's default, and
  `pack_ternary_q4` (`tools/kitten-tts/export.py:13-27`), its `ternary-q4_0` output. It compares both with the
  converted GGUF of Kitten `c93cb7c`:

  | Comparison | Result |
  |---|---|
  | trits against Kitten's `TQ2_1` | 1,409,286,144 of 1,409,286,144 equal |
  | per-128 fp16 scales against Kitten's `TQ2_1` | 11,010,048 of 11,010,048 bit-identical |
  | `Q4_0` blocks against Kitten's `pack_ternary_q4` | 44,040,191 of 44,040,192 byte-identical |

  The one `Q4_0` block that differs is a block of 32 zeros. Kitten's packer writes it a zero scale, and this
  converter writes the scale of its 128-group. Both decode to 32 zeros.
  The control: a GGUF converted with `--plant group256` reads `PACKED FAIL`. Its first finding is a trit of
  `model.layers.0.mlp.down_proj.weight` (row 22, column 3328: 0 where Kitten has 1). About half the `Q4_0` blocks
  differ.
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

* **Level 4, through `ggtts-render` on CPU: PASS.** The LM is Q4_0 and the decoder and reference preparation run on
  ggml. One line rendered twice with seed 1234 gave byte-identical FLACs (sha256 `9a71f26e...57b8`, 122 frames,
  5.0 s). The same file came out again after the move to the backend scheduler. A three-paragraph story on the
  story path also rendered twice byte-identical: 3 chunks, 27.18 s, sha256 `9026dc9f...b959`.

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

## CUDA precision (2026-11-06, written after the first CUDA probe and before the re-probe)

The first run of `scripts/cuda_probe.sh` (RTX 3080 Ti, sm_86) passed the converter, the LM f32 arm, the vocoder alone
and all of level 5, and failed three things. Its numbers, beside the CPU's:

| Check | CPU | CUDA, first probe | Bar |
|---|---|---|---|
| LM f32 arm, worst hidden rel L2 | `1.755e-6` | `3.668e-4` | 1e-3 |
| LM ternary arm (`Q4_0`): RMSE, min cosine, argmax | 0.0473, `0.9999091`, 32/32 | 0.0364, **`0.9998395`**, 32/32 | 0.06, 0.9999, 30 |
| flow: final mel rel L2 | `2.9e-7` | `2.747e-4` | 1e-3 |
| vocoder alone on PyTorch's mel: waveform SNR | 106.2 dB | 49.3 dB | 40 dB |
| flow then vocoder: waveform SNR | 89.3 dB | **30.7 dB** | 40 dB |
| render twice | byte-identical | not run: the probe's GGUF predated the embedded tokenizer | identical |

**No bar is moved by this section.** The CPU bars and the CUDA bars stay as level 6 states them. What changes is how
CUDA computes, and the predictions below are written before the re-probe that will test them.

### What loses the precision, read from ggml at the pin

Citations are `ggml/src/` at `af97976`. These were read, not measured op by op on the card.

* **TF32 in every large f32 product.** ggml creates each cuBLAS handle in TF32 tensor-op mode
  (`ggml-cuda/common.cuh:1478`, `cublasSetMathMode(..., CUBLAS_TF32_TENSOR_OP_MATH)`), and nothing at the pin turns it
  off: `GGML_CUDA_FORCE_CUBLAS` and `GGML_CUDA_FORCE_MMQ` are CMake options that choose kernels
  (`ggml-cuda/mmq.cu:268`, `:315`), and `GGML_CUDA_CUBLAS_COMPUTE_TYPE` chooses the compute type
  (`ggml-cuda/ggml-cuda.cu:1549`), not the math mode. An f32 product of more than 16 columns goes to cuBLAS
  (`ggml-cuda/ggml-cuda.cu:1752-1770`; `ggml-cuda/mmf.cu:176` refuses above 16), so both operands are rounded to a
  10-bit mantissa (unit roundoff `2^-11`, about `4.9e-4`). Every convolution here is an f32 `im2col` and such a
  product, and so is attention over a prompt. That is the size of the f32 arm's `3.7e-4` and the mel's `2.7e-4`,
  against the CPU's `1e-6` and `3e-7`.
* **The decoder chain multiplies the mel's error by about 118, on both backends.** On the CPU a mel at `2.9e-7`
  gave a waveform error of `10^(-89.3/20)` = `3.4e-5`: a gain of 118. Applied to CUDA's mel, `118 × 2.75e-4` =
  `3.2e-2`, or 29.8 dB; with the vocoder's own `3.4e-3` (49.3 dB) added, 29.7 dB. The probe measured 30.7 dB. So no
  single op in the chain loses precision beyond TF32 everywhere: the vocoder's gain turns a mel well inside its bar
  into a waveform outside its bar.
  That also says the two level-3 bars are not consistent with each other. At its own bar (`1e-3`), a mel would give
  about 18.6 dB end to end, and the 40 dB bar asks for a mel within about `8.5e-5`, twelve times inside the mel bar.
  The CPU passed both only because its f32 is far inside both. This is recorded, not acted on: the bars stay, because
  with TF32 off the chain should run near f32 on CUDA too. If the re-probe misses 40 dB with TF32 off, the bar question
  is opened in a dated entry here before any further measurement.
* **The ternary arm's activations are not coarser on CUDA for this check.** The 328-position forward sends each
  `Q4_0` product to MMQ (`ggml-cuda/ggml-cuda.cu:1766`; on sm_86 `ggml-cuda/mmq.cu:307` selects it), whose tensor-core
  path quantizes activations per 32 values to 8 bits with an fp16 scale (`ggml-cuda/quantize.cu:300-369`, layout
  `DS4` for `Q4_0` at `ggml-cuda/mmq.cuh:65-67`) and subtracts `Q4_0`'s offset of 8 from the weight codes in integers
  (`ggml-cuda/mmq.cuh:430-431`). That is the CPU's precision: `Q4_0` multiplies against `Q8_0`
  (`ggml-cpu/ggml-cpu.c:230`), per 32 with an fp16 scale and the offset subtracted in integers
  (`ggml-cpu/quants.c:247-248`). So "q8_1 against q8_0" is not the difference on this path. The arm's error is the
  same 8-bit activation rounding on both backends, which the CPU clears by `9e-6` on a `1e-4` allowance, plus CUDA's
  TF32 attention; CUDA's RMSE (0.0364) is in fact lower than the CPU's (0.0473), and its minimum cosine over 32
  positions falls on the other side of the bar.
* **Generation on CUDA has a worse path the check does not reach.** A product of at most 8 columns, which is every
  generated token, goes to MMVQ (`ggml-cuda/ggml-cuda.cu:1762`, `ggml-cuda/mmvq.cuh:3`). There `Q4_0` applies its
  offset as `d4 * (sumi * d8 - 8 * s8)` (`ggml-cuda/vecdotq.cuh:133`), where `s8` is the fp16-rounded sum of the
  activations before quantization (`ggml-cuda/quantize.cu:37-51`). For a ternary weight (codes 7, 8 and 9) the two
  terms nearly cancel, so each block's result carries eight times its summed activation-rounding error plus the
  fp16 rounding of `s8`.

### What changes on CUDA

* **The ternary matrices are held as f16 and multiplied at f32 precision** (`Model::ternary_f16`). Each weight is a
  trit times the block's fp16 scale, so f16 holds it exactly, and the loader checks the round trip row by row and
  refuses otherwise. The products carry `GGML_PREC_F32`, which makes the cuBLAS path compute in f32
  (`ggml-cuda/ggml-cuda.cu:1545`; without it an f16 matrix is computed in f16). A one-column product goes to the f32
  vector kernel (`ggml-cuda/mmvf.cu:708`), 2 to 16 columns to `mmf` (f16 activations, f32 accumulation), more to
  cuBLAS. The quantized kernels, MMVQ included, are no longer on the ternary path. The cost: the body takes 2.8 GB of
  device memory instead of 0.8 GB, and a generated token reads about 3.5 times the bytes. The GGUF is unchanged, and
  the CPU keeps `Q4_0` through ggml's quantized kernels.
* **TF32 is off.** `Backend::new` sets `NVIDIA_TF32_OVERRIDE=0` before creating the CUDA backend, unless the caller
  set it. That is NVIDIA's documented override of the cuBLAS math mode. ggml creates its cuBLAS handles lazily, at the
  first product. Whether cuBLAS honours a value set inside the process is measured, not assumed, by two diagnostic
  runs in the probe (below).
* **The probe converts the LM fresh** into its work directory on every run, with the binaries it built, and refuses a
  GGUF without the embedded tokenizer. The converter now refuses to convert when `lm/tokenizer.json` is missing,
  instead of writing a GGUF that `ggtts-render` refuses later. `DEVICE=cpu STEPS=render` runs the conversion and the
  render-twice step on the CPU, to check the probe's own render step without the card. With `TQ1_GGUF`, the probe
  also reports the CPU's packed TQ1 arm against the CPU f32 reference, clean and planted, with no bar on its numbers;
  `DEVICE=cpu STEPS=tq1` runs only that ([docs/tq1-cpu.md](../../docs/tq1-cpu.md)).

### Predictions for the re-probe

* LM f32 arm: worst hidden rel L2 well under `1e-4`, toward the CPU's `1e-6`.
* LM ternary arm: close to the f32 arm, with the bf16 head as the main remaining difference; min cosine at least
  `0.99999`, RMSE under `0.01`, argmax 32/32. The planted group of 256 stays red: it changes the weights, not the
  arithmetic.
* Flow: mel rel L2 under `1e-5`. Vocoder alone: above 80 dB. Flow then vocoder: above 60 dB.
* Diagnostics, which never decide the probe's exit status: with `NVIDIA_TF32_OVERRIDE=1` exported (TF32 allowed),
  flow then vocoder reads near the first probe's 30.7 dB, which shows the override is what changed it. With `=0`
  exported, the same as the in-process run. If the in-process run fails and the exported one passes, cuBLAS read the
  variable before the process set it, and the override moves to the process's start.
* Render: two CUDA renders byte-identical.

### The re-probe, and the render it broke (2026-11-06)

On `e732657` every parity check passed on CUDA. The LM f32 arm's worst hidden rel L2 was `2.7e-6`. The ternary arm
passed, and so did the vocoder and the flow-then-vocoder check, with all of level 5 and every plant red. Both TF32
diagnostics read as predicted. With TF32 allowed, flow then vocoder failed; with `=0` exported, it passed like the
in-process run. The ternary arm with TF32 allowed read cosine `0.9999987`, RMSE `0.019`.

**Both renders panicked in the sampler**: `sampler.rs:121`, an `unwrap()` on an empty list, because no logit was
finite. That is the single-token path, which no parity check took. What changes (written before the next probe):

* **The forward refuses non-finite logits.** It re-runs the same positions with every layer read back and names the
  first layer whose output is non-finite, and the engine adds the generation step and position. The sampler refuses
  non-finite logits too, rather than panicking.
* **The likely cause, read from ggml and not yet measured: a fusion drops `GGML_PREC_F32`.** For a one-column
  product followed by an add, ggml fuses the two into one mat-vec call whose destination is the add node
  (`ggml-cuda/ggml-cuda.cu:3657-3703`). The mat-vec kernel reads its precision from that destination
  (`ggml-cuda/mmvf.cu:650`). An add's parameters are zero, `GGML_PREC_DEFAULT`, so an f16 matrix is multiplied with
  f16 activations and a `half2` accumulator (`ggml-cuda/mmvf.cu:193-219`, selected at `:615`). The output and down
  projections both feed residual adds. The down projection sums 6144 products of inputs that can be
  large, so an f16 overflow to infinity is plausible. The next probe will show whether it happens. The batched forward is never fused this way.
  `Backend::new` therefore sets `GGML_CUDA_DISABLE_FUSION=1` on CUDA (`ggml-cuda/ggml-cuda.cu:3041`), unless the
  caller set it. This was not an issue before f16 weights: the `Q4_0` path fuses into MMVQ, which has no half
  accumulator.
* **The single-token check above** runs on the CPU in `tests/step_identity.rs` and on the card in the probe.

**Measured on the CPU (2026-11-06, after the bar was written).** The recorded prompt (328 positions) has a 312-position
prefill and 16 single-token steps. The bar does not move:

| Arm | Min cosine | Worst rel L2 | Argmax | Verdict |
|---|---|---|---|---|
| f32 | `1.0000000` | `1.3e-7` | 16/16 | PASS |
| ternary (`Q4_0`, bf16 head) | `0.9999995` | `1.4e-3` | **15/16** | **FAIL** (one argmax) |
| ternary, f32 head (diagnostic) | `0.9999998` | `5.9e-4` | 15/16 | FAIL |

`tests/step_identity.rs` runs 48 fixed pseudo-random ids, with 8 single-token steps. It passed for both arms: ternary
cosine `0.9999991`, rel L2 `1.3e-3`, argmax 8/8; f32 rel L2 `1.5e-7`.

So on the CPU the graph and the cache agree to f32 rounding, and the one-column quantized products do not match the
many-column ones. The quantized body differs at `5.9e-4` and the bf16 head adds the rest. (Corrected 2026-11-06:
this paragraph said ggml's CPU hands a many-column product to a different kernel, llamafile's sgemm. It does not.
`GGML_LLAMAFILE` is off in this build, and the products are bit-identical. The cause is under "The CPU
one-column gap, explained" below.) Two notes on size:
* The gap is a fraction of the ternary arm's own distance from PyTorch: RMSE 0.047 on the batched path.
* One near-tie argmax in 16 flipped.

What a generated token on the CPU gets is therefore within cosine `0.9999995` of what the parity check measured.
It is not identical. This is recorded as a failed bar on the CPU ternary arm, not explained away.
`cpu_one_column_ternary_product_differs_pinned` in `tests/step_identity.rs` pins it as an expected failure: argmax
15/16, cosine at least `0.999999`, and rel L2 between `1e-3` and `2e-3`. The test turns red if the gap closes or
grows. Its cause is tracked in ggtts#21.

Predictions for the next probe:
* The single-token check passes on CUDA for both arms. On CUDA both paths now multiply f16 matrices at f32
  precision, so they should agree as closely as the CPU's f32 arm does.
* With `GGML_CUDA_DISABLE_FUSION=0` (a diagnostic), it fails or errors, naming a layer.
* Two renders run, stop on their own and are byte-identical.

**The probe at `cd2f62f`, and a third kernel (2026-11-06).** Most of it passed:
* the f32 single-token check;
* the CUDA render, byte-identical twice (sha256 `a890c708...`), so the generation path runs;
* everything that passed before.

The ternary single-token check failed in its **batched** half. A 16-token forward gave 316,482 of 2,531,856 logits
non-finite, the first non-finite hidden state being after layer 28 of 28. That happened with the fusions on and off,
so the fused add is not its cause. It also means the fusions-on control never reached a single-token step, so it says
nothing about the fused add either.

The cause, read from ggml: a product of 2 to 16 columns goes to the small-batch `mmf` kernel
(`ggml-cuda/mmf.cu:176`, chosen at `ggml-cuda/ggml-cuda.cu:1758`). That kernel takes an f16 matrix on sm_86
(`mmf.cu:184-185`) and converts the f32 activations to f16 (`ggml-cuda/mmf.cuh:202-203`). It never reads the
product's precision, so `GGML_PREC_F32` cannot help. Only the last layer overflowing fits large activations there.

Real renders never send 2 to 16 tokens at once: a prompt is hundreds of tokens and generation is one at a time.
But a forward of any width should not produce infinities. So on CUDA the f16 ternary products of 2 to 16 columns
are now padded with zero columns to 17. That sends them to cuBLAS at f32 precision, and only the real columns are
kept (`CUDA_MMF_MAX_COLS` in `model.rs`). It costs at most 17 columns of work for a short batch.

Predictions for the probe after this change:
* Both single-token checks pass on CUDA.
* The fusions-on control now reaches the single-token steps. Whether it fails tells whether the fused add was the
  render's NaN or a bystander.
* The renders stay byte-identical, and their hash is unchanged, because they never take the padded path.

### The CPU one-column gap, explained (2026-11-06, ggtts#21)

On the CPU, the ternary arm's single-token forward differs from the batched one: argmax 15/16, rel L2 `1.4e-3`. The
f32 arm differs by only `1.3e-7`. The cause is not a kernel. It is `Q8_0` rounding amplifying f32-sized
differences.

* **Both paths run the same kernel, and it gives bit-identical results.** At the pin, ggml's CPU `mul_mat` quantizes
  every activation column to `Q8_0`, per 32 values, with the same `from_float` (`ggml-cpu/ggml-cpu.c:1322-1356`;
  `Q4_0`'s `vec_dot_type` is `Q8_0`, `:239-242`). It computes every output with the same `vec_dot`, one row at a
  time, whatever the column count (`:1243`, in `ggml_compute_forward_mul_mat_one_chunk`). The `llamafile_sgemm` branches (`:1295`, `:1366`) exist only under `GGML_LLAMAFILE`,
  which is off in this build (`GGML_LLAMAFILE_DEFAULT` OFF, `ggml/CMakeLists.txt:113-114`, and `build.rs` does not
  set it). `tests/q4_columns.rs` checks the result directly. 256 ternary rows of 2048 times 16 heavy-tailed columns,
  at 1 and at 4 threads: every output of the one-column products has the same bits as the 16-column product.
* **What differs is upstream, and `Q8_0` amplifies it.** The f32 arm shows how far the two paths' f32 arithmetic
  differs: `1.3e-7` (attention and softmax over a different number of positions, summed in a different order).
  `Q8_0` rounds each activation to one of 255 levels of its block's absmax, so an input that moves by `1e-7` leaves
  almost every output alone. Where an activation crosses a rounding boundary, the output jumps by a whole level. In
  `tests/q4_columns.rs`, a relative perturbation of `1e-6` on the activations moves the f32 product's output by
  1.03 times as much, and the `Q4_0` product's by 39.5 times as much. Over 28 layers and 16 positions the crossings
  compound into the `1.4e-3`. The bf16 head (`vec_dot_type` BF16) is a second rounding stage of the same kind, which
  is why an f32 head halves the gap.

**The levers, measured, and why none is taken.**

* **Exact f16 weights, the lever that fixed CUDA, does not carry over.** ggml's CPU multiplies an f16 matrix by
  rounding the activations to f16 (`ggml-cpu/ggml-cpu.c:221-224`, `vec_dot_type` F16). The CPU `mul_mat` never reads
  `GGML_PREC_F32`. So on the CPU it would be another rounding stage, finer than `Q8_0` but still discontinuous. It
  would also expose the f16 overflow that the last layer showed on CUDA.
* **Exact f32 weights** (`--weights f32`, the f32 arm) remove the rounding. They hold every bar: hidden rel L2
  `1.8e-6` against PyTorch, and single-token against batched `1.3e-7` with 16/16 argmax. They cost this much, on
  4 threads, alternating two runs each:

  | | `Q4_0` | f32 weights |
  |---|---|---|
  | prefill, 300 tokens | 89-90 tokens/s | 65-66 tokens/s |
  | generation, 64 steps | 19.2-20.3 tokens/s | 6.2-6.3 tokens/s |

  That is generation about 3.2 times slower. The weights take about 6.9 GB of memory instead of 1.5 GB: the body as
  f32, and the head widened from bf16.

**Decision:** keep `Q4_0` on the CPU. Generation is the CPU's bottleneck, already below the 25 tokens/s real time
needs, and the gap is not a defect. It is the expected behaviour of 8-bit activation rounding, whose effect on the
model is what the ternary arm's own bar against PyTorch measures. That bar passes. The bar for the single-token check
is not moved. The ternary arm still misses it on the CPU, and the pinned expected-failure test
(`cpu_one_column_ternary_product_differs_pinned`) stays. The f32 arm stays as the parity tool that shows the paths
agree when nothing is rounded.

## Streaming: chunk level only, and why (2026-11-06, ggtts#25)

Kitten-TTS 2 streams at **chunk level**: `--stream` hands out each chunk as soon as the pipeline keeps it
(`pipeline::render_story_streamed`). It does not stream frame by frame through S3Gen. This section gives the reason
from the code, then the measurement, made against a bar stated on the issue before any number existed.

### What S3Gen's stages see (from the code)
- **The flow encoder sees the whole sequence.** `Flow::encode` (`src/flow.rs:257-317`):
  - a `PreLookaheadLayer` with a 3-token right look-ahead (`flow.rs:283-287`);
  - 6 conformer layers, then 4 more after the ×2 upsample (`flow.rs:288-306`).

  Their attention has **no mask** (`ggml_soft_max_ext` with a null mask, `flow.rs:171`). The masks in the PyTorch
  source are all ones at batch 1 (`flow.rs:5`).
- **The flow decoder does too.** The meanflow estimator (`Flow::estimate`, `flow.rs:321-366`) has causal
  convolutions, but also 56 transformer blocks whose self-attention is unmasked (`flow.rs:192`). `Flow::mel` runs it
  twice over the prompt and the target together (`flow.rs:372-408`).
- **The vocoder's context is bounded.** HiFT is convolutional with symmetric padding (`src/hift.rs:323-325`, `364`,
  `388`), then an inverse STFT. Its receptive field is a fixed number of frames on each side.

So every mel frame depends on every token, the prompt's and the target's. **No bounded context reproduces the batch
decode's mel**, and an identical block decode is ruled out before measuring. The question left is how far a block
decode is from the batch decode, and the bar for that was stated first: the same bar the batch port meets, mel
relative L2 at most `1e-3` and waveform SNR at least 40 dB.

### Measured
**The method.** `ggtts-kitten block-parity` (CPU, 4 threads) works on the recorded S3Gen case: 103 target tokens,
248 prompt tokens, 4.10 s of audio.
- Each block of B tokens is decoded from its tokens plus C tokens of left context and a right look-ahead, with the
  prompt.
- The noise each frame has in the batch decode is reused, so only the context differs.
- The kept frames are stitched, vocoded with the recorded source, and compared with the batch decode of the same
  tokens.

The full log is `records/block-parity-cpu.log`.

**Every configuration fails the bar by orders of magnitude.** Waveform SNR in dB, mel relative L2 in brackets:

| Block (tokens) | Left context | Look-ahead 3 | Look-ahead 25 | Look-ahead 50 |
|---|---|---|---|---|
| 25 (1 s) | 1 token | −0.8 (6.0e-2) | −0.4 (5.3e-2) | 0.2 (5.1e-2) |
| 25 | 25 | 2.1 (3.3e-2) | 2.8 (2.4e-2) | 3.8 (2.3e-2) |
| 25 | 50 | 1.2 (3.2e-2) | 3.0 (2.0e-2) | 4.3 (1.8e-2) |
| 25 | all before | 2.4 (2.8e-2) | 5.1 (1.3e-2) | 7.4 (9.5e-3) |
| 50 (2 s) | 1 token | 0.1 (4.3e-2) | 1.2 (4.1e-2) | 1.6 (4.1e-2) |
| 50 | 25 | 1.3 (2.4e-2) | 2.9 (2.0e-2) | 3.5 (2.0e-2) |
| 50 | 50 | 3.4 (1.7e-2) | 6.3 (1.1e-2) | 7.8 (9.8e-3) |
| 50 | all before | 3.4 (1.6e-2) | 6.3 (1.0e-2) | 7.8 (9.5e-3) |

**The measure itself is checked both ways.** One block holding every token reads identical: relative L2 0, SNR
infinite. Each block's frames shifted by one (`--plant seam-shift`) is seen: SNR −2.9 dB.

**What the numbers say.** More context helps, and right context helps as much as left. That is what unmasked,
bidirectional attention predicts. But the best configuration measured is a 2 s block with all prior context and 2 s
of look-ahead, at 7.8 dB. That is 32 dB short of the bar, and its look-ahead already costs the latency streaming
was for. The stitched decode is also slower than the batch: the flow's time over the blocks is 5.4-14.6 s for 4.10 s
of audio (RTF 1.55-3.9 with the vocoder), against 2.5 s for the batch decode (RTF 0.85). A block decoder would run
behind real time on this CPU, so it is not offered as the CLI's stream.

**Chunk-level streaming today** (`crates/ggtts-qwen3-tts/scripts/stream_probe.sh`, `ENGINE=kitten-tts-2`, CPU,
4 threads, the `lighthouse` story; `records/stream-probe-kitten-cpu.log`):
- the first audio arrives at **28.90 s**;
- the whole read takes 74.93 s for 21.76 s of audio (RTF 3.44);
- the streamed and batch FLACs are byte-identical.

The first chunk's wait is the LM's generation of that chunk plus one S3Gen decode. A block decoder could only shorten
the decode part, and only at the parity loss above.

**What would change this.** S3Gen's attention masks. The upstream training code's chunk masks, if the weights were
trained with them, would give a bounded context by construction. That would be a new decoder mode with its own
parity cases against a PyTorch reference run with the same masks. It is not a stitching of the full-context model.
`block-parity` stays in the CLI to measure such a mode, or any other block decoder, against this bar.

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
| `src/backend.rs` | the device (CPU, or CUDA with a CPU fallback and TF32 off) and the scheduler every graph runs through |
| `scripts/cuda_probe.sh` | the CUDA parity probe: build, a fresh LM conversion, every check with `--device cuda`, the plants, TF32 diagnostics, a render twice; with a TQ1 model, the CPU packed arm reported against the CPU f32 arm |
| `src/parity.rs` | the parity measures and the bars above |
| `src/s3convert.rs` | S3Gen's safetensors to a GGUF, weight norm folded |
| `src/flow.rs` | the S3Gen flow on ggml: conformer encoder and meanflow estimator (feature `ggml`) |
| `src/refdsp.rs`, `src/condnet.rs` | reference-clip preparation: resampling, mels, fbank; the S3 tokenizer, CAMPPlus, pyannote (feature `ggml`) |
| `src/hift.rs`, `src/dsp.rs` | the HiFT vocoder on ggml (feature `ggml`) and its host-side source and STFT |
| `src/bin/ggtts-kitten-convert.rs` | the converter's command line |
| `src/bin/ggtts-kitten.rs` | `parity`, `tokcheck` and `render` (feature `ggml`) |
| `tools/kitten_ref.py` | the PyTorch side: reference dumps, reference-clip preparation, the S3Gen bridge |
| `tools/kitten_packed_check.py` | the converter's ternary matrices against Kitten's own packers (`TQ2_1` and `pack_ternary_q4`), numpy only |
| `tests/step_identity.rs` | the single-token path against the batched one, and the pinned CPU ternary gap |
| `records/` | measurement logs this document cites: `block-parity-cpu.log` and `stream-probe-kitten-cpu.log` (ggtts#25) |

The QAT export (`ternary_bf16.safetensors`, the whole LM in Kitten's key layout with every ternary-born linear as
`quantize(master)` in bf16) is the same layout as Kitten's `lm/model.safetensors`, so one converter takes both; the
rest of the model directory (config, tokenizer, voices) comes from the base model.
