#!/usr/bin/env python3
"""Template: per-chunk trim hook (`trim`). See README.md."""
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:  # run from the repository: the SDK is instruments/python
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_trace, run

NAME, VERSION = "edge-trim", "0.1.0"


def measure(ctx, env):
    limit = ctx.number("max_edge_s", 1.0)
    trace = read_trace(ctx.events)
    env.input(ctx.events).from_trace(trace, ctx.render)
    trims = trace.of("trim")
    leads, tails = [t["lead_s"] for t in trims], [t["tail_s"] for t in trims]
    env.metric("lead_s_max", max(leads) if leads else None, "s", "lower", render=ctx.render)
    env.metric("tail_s_max", max(tails) if tails else None, "s", "lower", render=ctx.render)
    env.metric("trim_s_total", sum(leads) + sum(tails), "s", "neutral", render=ctx.render)
    env.metric("edges_over_limit", sum(1 for x in leads + tails if x > limit), "1", "lower", render=ctx.render)
    if not trace.complete:
        env.status("truncated", "the stream has no end event")


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["trim"]))
