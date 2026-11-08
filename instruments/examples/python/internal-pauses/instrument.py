#!/usr/bin/env python3
"""Worked example: internal pauses. See README.md."""
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:  # run from the repository: the SDK is instruments/python
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

import math

from ggtts_instrument import read_audio, run

NAME, VERSION = "internal-pauses", "0.1.0"


def silent_frames(samples, rate, floor_dbfs, frame_s=0.01):
    """One flag per 10 ms frame: True when its RMS is below the floor."""
    n = max(1, int(rate * frame_s))
    floor = 10 ** (floor_dbfs / 20.0)
    out = []
    for i in range(0, len(samples) - n + 1, n):
        f = samples[i:i + n]
        out.append(math.sqrt(sum(x * x for x in f) / n) < floor)
    return out, frame_s


def pauses(samples, rate, floor_dbfs, min_pause_s):
    """The internal silences longer than `min_pause_s`, in seconds, in order."""
    silent, frame_s = silent_frames(samples, rate, floor_dbfs)
    voiced = [i for i, s in enumerate(silent) if not s]
    if not voiced:
        return []
    found, run_len = [], 0
    for s in silent[voiced[0]:voiced[-1] + 1]:
        if s:
            run_len += 1
        else:
            if run_len * frame_s > min_pause_s:
                found.append(run_len * frame_s)
            run_len = 0
    return found


def measure(ctx, env):
    min_pause_s = ctx.number("min_pause_s", 1.5)
    floor = ctx.number("floor_dbfs", -40.0)
    if not ctx.audio:
        raise ValueError("--audio is required")
    a = read_audio(ctx.audio)
    env.input(ctx.audio)
    found = pauses(a.samples, a.rate, floor, min_pause_s)
    env.metric("internal_pauses", len(found), "1", "lower", render=ctx.render)
    env.metric("longest_pause_s", round(max(found), 3) if found else None, "s", "lower", render=ctx.render)
    env.metric("pause_s_total", round(sum(found), 3), "s", "neutral", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["artifact.audio"], audio_state="unknown"))
