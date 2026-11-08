#!/usr/bin/env python3
"""Template: the render's audio (an artifact, not an event). See README.md."""
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:  # run from the repository: the SDK is instruments/python
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

import math

from ggtts_instrument import read_audio, run

NAME, VERSION = "audio-levels", "0.1.0"


def db(x):
    return 20 * math.log10(x) if x > 0 else None


def measure(ctx, env):
    clip = ctx.number("clip", 0.999)
    if not ctx.audio:
        raise ValueError("--audio is required")
    a = read_audio(ctx.audio)
    env.input(ctx.audio)
    peak = max((abs(x) for x in a.samples), default=0.0)
    rms = math.sqrt(sum(x * x for x in a.samples) / len(a.samples)) if a.samples else 0.0
    env.metric("peak_dbfs", db(peak), "dBFS", "n/a", render=ctx.render)
    env.metric("rms_dbfs", db(rms), "dBFS", "neutral", render=ctx.render)
    env.metric("clipped_samples", sum(1 for x in a.samples if abs(x) >= clip), "1", "lower", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["artifact.audio"], audio_state="unknown"))
