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
