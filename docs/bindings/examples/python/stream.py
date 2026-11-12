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
