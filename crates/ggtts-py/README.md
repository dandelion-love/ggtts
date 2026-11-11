# ggtts for Python

`import ggtts`: load a ggtts engine and render text to speech from Python. The engines and settings are those of
ggtts's C-ABI (`ggtts_load_with`): Qwen3-TTS with a speaker and a style instruction, and Kitten-TTS 2.

```python
import ggtts

with ggtts.load() as tts:                     # Qwen3-TTS, its default model, downloaded once into the cache
    audio = tts.render("今日天氣好好，我哋去海邊行下。")
    audio.write_flac("out.flac")              # 16-bit mono FLAC
    print(audio.sample_rate, audio.samples)   # float32: a numpy array when numpy is installed

    for chunk in tts.stream("第一段。\n第二段。"):   # chunks in order, as each is ready; `break` stops it
        play(chunk.samples, chunk.sample_rate)
```

`ggtts.load(engine="qwen3-tts", model=None, codec=None, voice=None, speaker=None, instruction=None, device="cpu",
threads=None, offline=False, progress=None)`. A setting an engine does not take is refused, never ignored. Errors
are `ggtts.GgttsError` with `code` (`ggtts.ERR_ARG`, `ERR_ENGINE`, ... the C-ABI's codes) and `message`; a panic
inside ggtts is a `GgttsError` with `ERR_PANIC`, never a crash of the interpreter. The GIL is released while an
engine loads and renders.

Build from a checkout (Rust 1.88 and Python 3.9 or later; the ggml engines need CMake and a C++ compiler, and on
Windows libclang with `--features bindgen` in place of `ggml`):

```bash
pip install maturin
cd crates/ggtts-py
maturin build --release --features python,pyo3/extension-module,ggml   # wheel in target/wheels
pip install target/wheels/ggtts-*.whl
```

Without the `ggml` feature only the `fake` test engine loads. Licence: MIT; see the repository's `THIRD_PARTY.md`.
