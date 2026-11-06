#!/usr/bin/env python3
"""Template: per-render hooks (`join`, `end`). See README.md."""
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:  # run from the repository: the SDK is instruments/python
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_trace, run

NAME, VERSION = "render-totals", "0.1.0"


def measure(ctx, env):
    trace = read_trace(ctx.events)
    env.input(ctx.events).from_trace(trace, ctx.render)
    join = (trace.of("join") or [{}])[-1]
    end = (trace.of("end") or [{}])[-1]
    env.metric("seconds", end.get("seconds"), "s", "neutral", render=ctx.render)
    env.metric("frames", end.get("frames"), "1", "neutral", render=ctx.render)
    env.metric("chunks", join.get("chunks"), "1", "neutral", render=ctx.render)
    env.metric("gain_db", join.get("gain_db"), "dB", "n/a", render=ctx.render)
    if not trace.complete:
        env.status("truncated", "the stream has no end event, so the render's totals are unknown")


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["join", "end"]))
