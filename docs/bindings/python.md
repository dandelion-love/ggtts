# ggtts from Python

`import ggtts` loads a text-to-speech engine, renders text to audio, and writes FLAC. It is the same engine as the C
library, with the same settings and error codes, in a Pythonic shape. [All bindings](README.md) compares it with C
and Kotlin.

## Get it

ggtts is not on PyPI. Each release from the first after v0.1.0-rc.10 ships two wheels on the
[releases page](https://github.com/dandelion-love/ggtts/releases), beside the archives:

| Wheel | For |
|---|---|
| `ggtts-<version>-cp39-abi3-manylinux_2_35_x86_64.whl` | Linux x86-64 with glibc 2.35 or newer (Ubuntu 22.04 or later) |
| `ggtts-<version>-cp39-abi3-win_amd64.whl` | Windows x86-64 |

One wheel serves every Python from 3.9. numpy is optional. Check a wheel, then install it:

```sh
gh attestation verify ggtts-<version>-cp39-abi3-manylinux_2_35_x86_64.whl -R dandelion-love/ggtts
pip install ggtts-<version>-cp39-abi3-manylinux_2_35_x86_64.whl
```

The wheels carry the Qwen3-TTS and Kitten-TTS 2 engines and render on the CPU. The Linux wheel bundles the GNU OpenMP
runtime (libgomp), so a minimal system needs nothing added. The wheels' ggml is built portable, as the archives' is,
so they need the archives' CPU: x86-64 with AVX2, FMA, F16C and BMI2. The Windows wheel is built with MSVC like the
Windows archive; if `import ggtts` fails for a missing DLL, install the Microsoft Visual C++ Redistributable for x64,
which the archive needs too.

**From source**, with the Rust toolchain that `rust-toolchain.toml` names, Python 3.9 or later, CMake and a C++
compiler:

```sh
git clone https://github.com/dandelion-love/ggtts && cd ggtts
git submodule update --init crates/ggtts-ggml-sys/ggml
pip install maturin
cd crates/ggtts-py
maturin build --release --features python,pyo3/extension-module,ggml   # Windows: bindgen in place of ggml (needs libclang)
pip install target/wheels/ggtts-*.whl
```

`--features` replaces the crate's default list, so name all of them. Without `ggml` (or `bindgen`) only the `fake`
test engine loads. Add `cuda` for `device="cuda"` (see [CUDA](#cuda)).

## A first render

This script loads Qwen3-TTS, renders one line, and writes `first_render.flac`. Run it with no arguments. The first run
downloads the default model (2.33 GB) into the cache, checks its sha256, and prints its progress to stderr. Later runs
download nothing.

<!-- example: examples/python/first_render.py -->
```python
"""first_render.py: load an engine, render one line, write it to a FLAC file.

    python first_render.py                                   Qwen3-TTS with its default model (downloaded once, 2.33 GB)
    python first_render.py qwen3-tts talker.gguf codec.gguf  Qwen3-TTS with model files you already have
    python first_render.py kitten-tts-2 lm.gguf s3gen.gguf voice.json
    python first_render.py fake                              the test engine: no files, a tone instead of speech
"""
import sys

import ggtts

engine, model, codec, voice = (sys.argv[1:] + [None] * 4)[:4]
engine = engine or "qwen3-tts"

with ggtts.load(engine, model=model, codec=codec, voice=voice) as tts:
    audio = tts.render("今日天氣好好。")       # one paragraph per line; seed=20260905 unless you pass one
    path = audio.write_flac("first_render.flac")
    print(f"{len(audio)} samples at {audio.sample_rate} Hz ({audio.duration:.2f} s), written to {path}")
```

## Loading

```text
ggtts.load(engine="qwen3-tts", model=None, codec=None, voice=None, speaker=None, instruction=None,
           device="cpu", threads=None, offline=False, progress=None) -> ggtts.TTS
```

| Argument | Meaning |
|---|---|
| `engine` | `"qwen3-tts"` (the default), `"kitten-tts-2"`, or `"fake"` (a test engine that needs no files). `ggtts.ENGINES` lists what this build has. |
| `model` | The model GGUF: Qwen3-TTS's talker, or Kitten-TTS 2's LM. A `str` or a path. None for Qwen3-TTS means the default model (below). |
| `codec` | The audio decoder GGUF: Qwen3-TTS's 12 Hz codec, or Kitten-TTS 2's S3Gen GGUF. None for Qwen3-TTS means the default codec. |
| `voice` | Kitten-TTS 2's prepared reference voice (a JSON file). |
| `speaker` | A Qwen3-TTS speaker the model has, such as `"hk_voice_777"`. Case is ignored. |
| `instruction` | A Qwen3-TTS style instruction for every chunk, such as `"Speak slowly and softly."` |
| `device` | `"cpu"` or `"cuda"`. See [CUDA](#cuda). |
| `threads` | CPU threads. None means 4. |
| `offline` | True: never download. A default model file missing from the cache is then an error naming it and its URL. |
| `progress` | A function called with each download message. By default they go to stderr. |

A setting the engine does not take is refused (`ERR_ARG`), never ignored. An empty string is no setting. A file that
is missing or is not a GGUF is refused before any engine loads. The settings hold for the engine's life.

What was loaded is on the `TTS`: `tts.engine`, `tts.speaker` (None for the model's own), `tts.instruction`,
`tts.device`, `tts.info` (the engine's family, backend and parity level) and `tts.model_files` (each file's role,
path, and source: `path` if you gave it, `cache` for a default file).

### The default model (Qwen3-TTS)

With engine `"qwen3-tts"` and no `model` or no `codec`, ggtts uses the same default files as `ggtts-render`: the
v0.0.5 talker and codec of
[jmtl/hk-cantonese-cc0-qwen3-tts-GGUF](https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts-GGUF), 2.33 GB in all,
each checked against a pinned sha256. A path you give is used as it is, and you may give one and not the other. Read
the model's card before you use its voice; its licence is the model's, not this repository's.

* **Python downloads by default.** A default file missing from the cache is downloaded inside `ggtts.load`, which
  blocks until it is verified. `offline=True` turns that off. (The C and Kotlin bindings are the other way round:
  they download only when asked.)
* **The cache.** `$XDG_CACHE_HOME/ggtts` (else `~/.cache/ggtts`) on Linux, `~/Library/Caches/ggtts` on macOS, and
  `%LOCALAPPDATA%\ggtts\cache` on Windows. Set the environment variable `GGTTS_CACHE_DIR` to move it; `load` has no
  argument for it. The README's
  [default model files](../../README.md#default-model-files-what-ggtts-downloads-and-when) section lists the files,
  their hashes and how a download is checked.
* **The speaker.** The default model's speakers are `hk_voice_343` (used when you give none) and `hk_voice_777`. A
  model you give by path uses `hk_cantonese_speaker` when you give none, so a talker without that speaker needs
  `speaker=`. An unknown speaker is `ERR_ENGINE`, and the message lists the model's speakers.

### Kitten-TTS 2

Kitten-TTS 2 needs `model` (the LM GGUF), `codec` (the S3Gen GGUF) and `voice` (a reference voice made by
`ggtts-kitten prepare`). It has no default model, and no ggtts-ready Kitten files are published: make them once with
`ggtts-kitten-convert` and `ggtts-kitten` from a release archive, as [the Kitten README](../../crates/ggtts-kitten/README.md)
shows. It takes no speaker and no instruction (`ERR_ARG`). Its licence is the model's: audio rendered with it must not
be used to train other models.

```sh
python first_render.py kitten-tts-2 kitten-lm.gguf kitten-s3gen.gguf prepared-voice.json
```

### CUDA

`device="cuda"` renders on an NVIDIA GPU, but only in a build with the `cuda` feature
(`--features python,pyo3/extension-module,cuda`). **The release wheels are CPU builds**: they refuse `"cuda"` with
`ERR_ENGINE` before reading any file. Any other device name is `ERR_ARG`, and the fake engine takes `"cpu"` only.

## Rendering

`tts.render(text, seed=ggtts.DEFAULT_SEED)` renders the whole text and returns an `ggtts.Audio`.

* **Text** is one paragraph per line. Empty lines are skipped. Text with nothing to say (empty, or only punctuation)
  is `ERR_ARG`, refused before any rendering.
* **The render** splits the text into chunks, renders each, checks it, trims it, joins them and levels the whole read,
  as `ggtts-render --story` does. It is the C library's render: the same text and seed give the same samples.
* **The seed** fixes the random draws. `ggtts.DEFAULT_SEED` is 20260905.

An `Audio` holds mono float32 samples:

| Member | What it is |
|---|---|
| `audio.samples` | The samples: a float32 numpy array when numpy is installed, else an `array.array('f')`. |
| `audio.numpy()` | The samples as a numpy array (needs numpy). |
| `audio.pcm`, `audio.to_bytes()` | The samples as native-endian float32 bytes. |
| `audio.sample_rate` | Samples per second (24000 for the models above). |
| `len(audio)`, `audio.duration` | The number of samples, and seconds. |
| `audio.write_flac(path)` | Write 16-bit mono FLAC. The suffix is forced to `.flac`; returns the path written. |

`ggtts.write_flac(path, samples, sample_rate=None)` writes any float32 samples (an `Audio`, a numpy array, an
`array.array`, float32 bytes or a list of floats) the same way.

## Streaming and stopping

Two ways to get each chunk as soon as it is ready, so playback can start before the render ends:

* **`tts.stream(text, seed=...)`** is a generator of `Audio` chunks, in order. `break` stops the render after the
  chunk in progress. It renders on a background thread, and at most one finished chunk waits for you.
* **`tts.render_stream(text, on_chunk, seed=...)`** calls `on_chunk(audio)` with each chunk, on the calling thread.
  Return `False` to stop. An exception in `on_chunk` stops the render and is raised from `render_stream`. It returns
  True when the whole text was rendered, False when stopped.

Stopping is not an error in Python: neither raises `ERR_STOPPED`. The chunks are the render's samples *before* its
final level step. Joined, they have the render's length, but not its exact values.

This script streams, stops early both ways, tries a render from inside the callback, and shows three errors:

<!-- example: examples/python/stream.py -->
```python
"""stream.py: render piece by piece, stop early, and see the threading rules and error codes at work.

    python stream.py                                   Qwen3-TTS with its default model, offline (it must be in the cache)
    python stream.py fake                              the test engine
    python stream.py qwen3-tts talker.gguf codec.gguf
"""
import sys

import ggtts

engine, model, codec = (sys.argv[1:] + [None] * 3)[:3]
engine = engine or "qwen3-tts"
TEXT = "今日天氣好好。\n我哋去海邊行下。\n行完之後，去食飯。"

try:
    tts = ggtts.load(engine, model=model, codec=codec, offline=True)  # offline: never download
except ggtts.GgttsError as e:
    # e.code is one of ggtts.ERR_*; an ERR_ENGINE here names a missing default file and its URL.
    sys.exit(f"load failed: {e.message} (code {e.code})")

with tts:
    # 1. A generator: each chunk as soon as it is ready. `break` stops the render after the chunk in progress.
    for n, chunk in enumerate(tts.stream(TEXT), start=1):
        print(f"chunk {n}: {len(chunk)} samples at {chunk.sample_rate} Hz")
        if n == 2:
            break

    # 2. A callback: return False to stop. render_stream returns False when stopped, True when the text was done.
    busy = []

    def on_chunk(chunk):
        try:
            tts.render("x")                    # one render at a time per engine: this is refused
        except ggtts.GgttsError as e:
            busy.append(e.code)
        return False                           # stop after the first chunk

    finished = tts.render_stream(TEXT, on_chunk)
    print(f"render_stream finished: {finished}; a render from the callback raised code {busy[0]} (ERR_BUSY)")
    assert finished is False and busy == [ggtts.ERR_BUSY]

    # 3. Text with nothing to say is refused before rendering.
    try:
        tts.render("？！")
    except ggtts.GgttsError as e:
        print(f"punctuation only: code {e.code} (ERR_ARG): {e.message}")
        assert e.code == ggtts.ERR_ARG

# 4. After close (here, the end of the with block) every call raises ERR_HANDLE.
try:
    tts.render(TEXT)
except ggtts.GgttsError as e:
    print(f"after close: code {e.code} (ERR_HANDLE)")
    assert e.code == ggtts.ERR_HANDLE
```

## Errors

Every failure is a `ggtts.GgttsError` with `code` (one of the constants below, the C library's codes) and `message`.

| Code | Value | What happened | What to do |
|---|---|---|---|
| `ggtts.ERR_ARG` | -1 | An argument is wrong: empty or punctuation-only text, a missing `model`, `codec` or `voice` the engine needs, a speaker or instruction for an engine that takes none, or an unknown device. | Fix the argument the message names. |
| `ggtts.ERR_HANDLE` | -2 | The engine was closed, or dropped after a panic. | Load again. Do not use a `TTS` after `close()`. |
| `ggtts.ERR_ENGINE` | -3 | The engine refused: an unknown engine name, a missing file or one that is not a GGUF, an unknown speaker, an instruction the checkpoint takes none of, a default model file missing while `offline`, a failed or mismatched download, `"cuda"` in a build without CUDA, an engine this build does not have, or a failed render. | Read the message. For a missing default file, allow the download, put the file in the cache, or pass paths. For an unknown speaker, pick one the message lists. For `"cuda"`, use `"cpu"` or a CUDA build. |
| `ggtts.ERR_PANIC` | -4 | A bug inside ggtts, caught before it reached Python. The engine is dropped. | Load again. Please report it as an issue, with the message. |
| `ggtts.ERR_STOPPED` | -5 | Never raised in Python: a stopped stream returns instead. It exists so the codes match C. | Nothing. |
| `ggtts.ERR_BUSY` | -6 | The engine is already rendering, on another thread or in its own stream callback. | Wait for that render, or load a second engine. |

## Threads and the GIL

* **The GIL is released while an engine loads (and downloads) and while it renders**, so your other Python threads
  keep running. `render_stream` takes the GIL back only to call your callback.
* **One render at a time per engine.** A second call on an engine that is rendering raises `ERR_BUSY` at once. It
  never waits and never deadlocks, including a call from the engine's own callback.
* **Different engines render at the same time**, on different threads. A second engine is a second loaded model, with
  its own memory.
* **`close()` never waits.** Called while another thread renders, it returns at once; that render finishes, and the
  engine goes when it returns.

## Lifetime

* Use the `TTS` as a context manager (`with ggtts.load() as tts:`), or call `tts.close()`. After that, every call
  raises `ERR_HANDLE`. `tts.closed` says which.
* An `Audio` holds its own copy of the samples. It stays valid after the engine is closed.

## What it does not do

* Per-render speakers or instructions: the settings are per engine. Load another engine to change them.
* A cache directory argument: use `GGTTS_CACHE_DIR`.
* Play audio. Pass `audio.samples` and `audio.sample_rate` to the audio library you use.
