"""options.py: every option through one argument, `options`, at load and per render; a refused option names its key.

    python options.py                                   Qwen3-TTS with its default model (downloaded once, 2.33 GB)
    python options.py qwen3-tts talker.gguf codec.gguf  Qwen3-TTS with model files you already have
    python options.py fake                              the test engine: no files, a tone instead of speech
"""
import sys

import ggtts

engine, model, codec = (sys.argv[1:] + [None] * 3)[:3]
load = {"engine": engine or "qwen3-tts"}
if model:
    load.update(model=model, codec=codec)

# The load document: the load section, and render defaults every render on this engine starts from.
with ggtts.load(options={"load": load, "render": {"edge_s": 0.25}}) as tts:
    # A render document, laid over those defaults. seed= is the same value as "render": {"seed_salt": "7"}.
    audio = tts.render("今日天氣好好。", seed=7, options={"render": {"redraw": 1}})
    print(f"{audio.duration:.2f} s at {audio.sample_rate} Hz")

    # A stream through a document is levelled as ggtts-render --stream levels it: one gain from the first 5 s.
    pieces = list(tts.stream("今日天氣好好。\n我哋去海邊行下。", options={}))
    # "raw" asks for the pieces before the level step, as a stream with no document hands them.
    raw = list(tts.stream("今日天氣好好。\n我哋去海邊行下。", options={"output": {"stream": "raw"}}))
    print(f"{len(pieces)} levelled pieces, {len(raw)} raw")

    # A refused option is an OptionError: a ValueError and a GgttsError, whose key is the option's schema key.
    try:
        tts.render("今日天氣好好。", options={"render": {"nonesuch": 1}})
    except ValueError as e:
        print(f"refused {e.key}: {e.message}")
        assert e.key == "render.nonesuch"
