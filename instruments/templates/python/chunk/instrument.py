#!/usr/bin/env python3
"""Template: per-chunk hook (`chunk_begin`, `chunk_end`). See README.md."""
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:  # run from the repository: the SDK is instruments/python
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_trace, run

NAME, VERSION = "draws", "0.1.0"


def measure(ctx, env):
    trace = read_trace(ctx.events)
    env.input(ctx.events).from_trace(trace, ctx.render)
    ends = trace.of("chunk_end")
    env.metric("draws", len(ends), "1", "neutral", render=ctx.render)
    env.metric("redraws", sum(1 for e in ends if e["draw"] > 0), "1", "lower", render=ctx.render)
    env.metric("cap_stops", sum(1 for e in ends if e["stop"] == "cap"), "1", "lower", render=ctx.render)
    env.metric("flagged_draws", sum(1 for e in ends if e.get("verdict")), "1", "lower", render=ctx.render)
    per = {}
    for e in ends:
        per[e["chunk"]] = per.get(e["chunk"], 0) + 1
    for chunk in sorted(per):
        env.metric("chunk_draws", per[chunk], "1", "lower", render=ctx.render, chunk=chunk)
    if not trace.complete:
        env.status("truncated", "the stream has no end event")


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["chunk_begin", "chunk_end"]))
