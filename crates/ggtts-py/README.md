# ggtts for Python

`import ggtts`: load a ggtts text-to-speech engine (Qwen3-TTS or Kitten-TTS 2) and render text to speech from Python.

```python
import ggtts

with ggtts.load() as tts:                     # Qwen3-TTS, its default model, downloaded once into the cache
    audio = tts.render("今日天氣好好，我哋去海邊行下。")
    audio.write_flac("out.flac")              # 16-bit mono FLAC
```

How to install it, every argument, streaming, the error codes, threads and the GIL:
[ggtts from Python](https://github.com/dandelion-love/ggtts/blob/main/docs/bindings/python.md). Licence: MIT; see the
repository's `THIRD_PARTY.md`.
