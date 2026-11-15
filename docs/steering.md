# Steering

Steering adds a vector to the Qwen3-TTS talker's hidden state at the output of chosen layers, so activation-steering
research can render through ggtts instead of a PyTorch forward hook.

```text
ggtts-render --model talker.gguf --codec codec.gguf --text "..." --out steered.flac \
    --steer vectors.safetensors [--steer-scale 0.5] [--steer-layers 23,24,25]
```

## The file

A [safetensors](https://github.com/huggingface/safetensors) file with one tensor per steered talker layer:

- named `talker.layer.<i>`, with `<i>` the layer index in plain decimal (`talker.layer.14`, not `talker.layer.014`);
- dtype `F32`, one dimension, of the talker's hidden size (2048 for the 1.7B talker).

PyTorch writes it directly, so a reference and ggtts read the same bytes, with no conversion and no text rounding:

```python
from safetensors.torch import save_file
save_file({"talker.layer.14": v14, "talker.layer.15": v15}, "vectors.safetensors")
```

The file's sha256 identifies it. Nothing else in the file is read; only a `__metadata__` entry may sit beside the
vectors.

## The flags

| Flag | Meaning |
|---|---|
| `--steer FILE` | The steering file: a path, or an `https://` URL (see below). Without it the render is exactly the render without steering. |
| `--steer-scale X` | One scale for every layer, default 1.0. Each vector is multiplied by it, in f32. |
| `--steer-layers LIST` | The layers of the file to apply, for example `23,24,25`. Default: every layer in the file. One file can then serve a comparison of layer subsets. |

A different scale per layer is not a flag: write the vectors pre-scaled.

### A steering file named by URL

`--steer` (and `render.steer` in an options document, so every binding too) takes an `https://` URL, or the Hub
shorthand `hf://<owner>/<repo>[@<revision>]/<path>` (the Hub's `resolve` URL, at `main` unless a revision is given),
for example a published preset:

```bash
ggtts-render --steer https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-emotion-steering/resolve/main/presets/angry.safetensors \
  --steer-scale 0.1875 --steer-layers 21,22,23,24,25 --text "夠喇！呢個爛攤子你自己即刻執返佢！" --out angry.flac
```

* **Downloaded once.** The file goes into the same per-user cache as the model files (`GGTTS_CACHE_DIR` overrides
  it; `load.cache_dir` in a document), at `urls/<sha256 of the URL>/<file name>`, and later runs reuse it without
  asking the network. The log line names the URL and the file's sha256 every time, as it does for the codec.
  ggtts-render and ggtts-book print it; the bindings write no log, so their download is silent.
* **The shorthand.** The same preset is
  `--steer hf://jmtl/hk-cantonese-cc0-qwen3-tts-emotion-steering/presets/angry.safetensors`; `...-steering@<commit>/...`
  pins a revision.
* **One URL, one entry, never re-checked.** Once a URL is cached it is reused without asking the network, so a
  URL at a moving revision (a `resolve/main` link, or `hf://...` with no `@revision`) keeps the copy first
  downloaded even after `main` moves. Pin a revision (`@<commit>`) or a `#sha256=` to know which file you render
  with; delete the cache entry to fetch a newer one. A cached copy that changed on disk since its download is
  refused.
* **A revision is one segment**: a commit, a tag, or a branch with no `/`. The shorthand is split at its first two
  slashes, so a revision such as `refs/pr/3` needs the full `https://huggingface.co/.../resolve/refs%2Fpr%2F3/...`
  URL.
* **A pin.** `URL#sha256=<64 hex>` refuses a file whose sha256 differs. A download that differs is removed; a cached
  copy that differs is refused, naming where it is.
* **Offline.** `--offline` (`load.offline`) refuses a URL that is not cached yet, naming it and its cache path.
  A binding resolves with the settings its engine was loaded with (`offline`, `cache_dir`, or the Kotlin
  `allowDownload = false`), so an engine loaded offline never downloads one.
* **Limits.** HTTPS only, every redirect too; at most 256 MB. A local path behaves exactly as before.

## Where it is applied

At the output of each selected talker layer, after the layer's residual add, at every position: every row of the
prompt, and every generated frame. That is what a PyTorch forward hook returning `output + scale * vector` on the
layer does. The code predictor and the codec are not steered.

In the graph it is one add node per selected layer, in the prompt graph and in every step graph. With no steering
the graph has no such node: the unsteered graph is the graph as it was before steering existed, not that graph plus
a zero. A scale of 0 keeps the node and adds zeros, which gives the unsteered tokens.

## What is refused

Each of these stops the render before anything is generated, with a message naming the problem:

- a file that cannot be read, or that is not safetensors;
- a tensor that is not named `talker.layer.<i>`, is not `F32`, is not one-dimensional, or holds a value that is not
  finite, and a header that names one tensor twice;
- a vector that is not the talker's hidden size, or a layer the talker does not have;
- `--steer-layers` naming a layer the file does not hold, or naming one twice;
- `--steer-scale` that is not a finite number in f32 (the vectors are scaled and added in f32, so `1e39` is refused),
  a scale that makes any scaled value overflow f32, and `--steer-scale` or `--steer-layers` without `--steer`;
- `--steer` with an engine that cannot steer (Kitten-TTS 2, the fake engine). The engines refuse a steered request
  themselves too, so a library caller cannot have one ignored.

## What is recorded

A steered render says so wherever it records itself: the trace header, the timing and duration reports, and the
instruments' `report.json` carry

```json
"steer": {"sha256": "<the file's sha256>", "layers": [23, 24, 25], "scale": 0.5}
```

and an unsteered render carries no `steer` at all. A steered render can therefore never be taken for a plain one.
Replaying a steered trace needs the same steering (`trace::replay_steered`); `trace::replay` refuses it.

## Steering is not an instrument

An instrument never changes the output ([hooks](instruments/hooks.md)); steering changes it by design. So steering
is a field of the request (`ChunkRequest::steer`), like the style instruction, and never goes through the hooks or
the trace. Instruments stay read-only on a steered render: the trace-identity check compares a steered untraced
render with the same steered render traced, and still goes red on its planted instrument.

## How it is checked

`crates/ggtts-qwen3-tts/tests/steer.rs`, on the tiny fixture (`crates/ggtts-qwen3-tts/fixtures/tiny/`) and, for the
`full_model` checks, on the published talker it is cut from:

- an unsteered render gives the tokens and samples recorded before steering existed, and its graph has no steering
  node;
- scale 0 gives the unsteered tokens;
- a vector changes the tokens and the same request gives the same tokens twice; with the add node planted out
  (`Plant::SteerAddMissing`) that check is red;
- the final hidden state agrees with a PyTorch forward-hook reference. `scripts/steer_reference.py` builds the
  reference from the talker GGUF's own weights, dequantised into `transformers`' Qwen3 decoder, and writes a fixture
  of a five-row prefill: `crates/ggtts-qwen3-tts/fixtures/steer/` for the tiny talker at layer 0, and
  `fixtures/steer/full/` for the published talker at layer 14 of 28, neither the first layer nor the last. The check
  compares the steering's effect (steered minus unsteered hidden state) with the reference's, as a relative
  root-mean-square difference, within twice the engine's own unsteered error on the same prefill. That error
  grows with depth: about 1% after one layer, about 16% after 28. Its cause is not yet attributed (the reference
  already uses the GGUF's dequantised weights, so weight quantisation is not it); see #204.
  Taking the difference cancels most of it: the effect is off by 0.3% on the tiny talker and 20% on the full one.
  Three planted defects turn the full-talker check red: the vector one layer late (`Plant::SteerLayerShift`, effect
  error 84%), at its layer and the next (`Plant::SteerNextLayerToo`, 71%), and no add at all
  (`Plant::SteerAddMissing`, 100%), against a tolerance of 31%;
- the trace-identity check passes on steered requests, and its planted drawing instrument is red;
- a layer subset of a file gives the tokens of a file holding only those layers.

The recorded unsteered hashes are of an x86_64 CPU build; another CPU's kernels may round differently.

## Not in scope

Steering the code predictor or the codec, a scale per layer or per frame, steering only some positions, and computing
vectors. The C, Kotlin and Python bindings do not take steering yet.
