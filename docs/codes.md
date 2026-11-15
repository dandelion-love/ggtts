# Codes files (`ggtts-codes/1`)

`ggtts-render --codes FILE` (the option `observe.codes`) writes the codec tokens of every chunk the render keeps,
one JSON object per line, on `--text` and `--story` alike and on every engine and device. These are the tokens the
engine generated and the decoder read, so two renders can be compared code for code: when they differ only in their
audio, the difference is in decoding, not in generation.

Writing them changes no sample: the file is filled from the codes the render already holds, after each chunk is
kept.

## Format

One line per kept chunk, in chunk order:

| Field | Meaning |
|---|---|
| `format` | `"ggtts-codes/1"` |
| `engine` | The engine's family, for example `qwen3-tts`. |
| `chunk` | The chunk's index in the plan (0 for `--text`). |
| `draw` | The kept draw's number: 0, or the redraw that was kept. |
| `seed` | The kept draw's seed. |
| `frames` | Codec frames in the draw. |
| `codebooks` | Tokens per frame: 16 on Qwen3-TTS (codebook 0 from the talker, 1 to 15 from the sub-talker). |
| `codes` | `frames x codebooks` tokens, frame-major: frame `f`, codebook `c` is `codes[f * codebooks + c]`. |

Only kept draws are written; a draw the judge rejected is not decoded into the output, so it has no place here. The
trace's `step` events at `--trace-level 2` (see [hooks.md](instruments/hooks.md)) also carry codebooks 1 onward for
an engine that emits steps, for every draw including rejected ones; the codes file is the record of the audio.

## Decoding a line again

A line holds the `frames`, `codebooks` and `codes` keys that the Qwen3-TTS decode example reads, so one chunk can be
decoded on its own and compared with the render:

```bash
sed -n 1p out.codes.jsonl > chunk0.json
cargo run --release -p ggtts-qwen3-tts --features ggml --example decode -- codec.gguf chunk0.json chunk0.f32 4
```

`chunk0.f32` is the decoded samples as little-endian f32. For a `--text` render, which has no trim and no level
step, they are the FLAC's samples before 16-bit quantisation.

`scripts/codes_decode_check.sh [cpu|cuda]` runs that check end to end on the default model from the cache: it
renders one `--text` line with and without `--codes` (the FLACs must match), decodes the written codes on the same
device, and compares them with the render's samples.
