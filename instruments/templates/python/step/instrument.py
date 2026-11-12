#!/usr/bin/env python3
"""Template: per-step hook (`step`). See README.md."""
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:  # run from the repository: the SDK is instruments/python
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_trace, run

NAME, VERSION = "p-stop-early", "0.1.0"


def measure(ctx, env):
    threshold = ctx.number("threshold", 0.5)
    trace = read_trace(ctx.events)
    env.input(ctx.events).from_trace(trace, ctx.render)
    flags = trace.header.get("flags")
    semantics = flags.get("p_stop_semantics") if isinstance(flags, dict) else None
    ctx.config["p_stop_semantics"] = semantics if isinstance(semantics, str) and semantics else "unknown"
    draws = trace.draws()
    if not trace.complete:
        env.status("truncated", "the stream has no end event")
        return
    if not draws or any(not steps or any("p_stop" not in step for step in steps)
                        for _begin, steps, _end in draws):
        env.status("skipped", "step p_stop data is unavailable for one or more draws")
        return
    early, hits, seen = 0.0, 0, False
    for _begin, steps, _end in draws:
        half = len(steps) // 2
        for s in steps[:half]:
            if "p_stop" in s:
                seen = True
                early = max(early, s["p_stop"])
                hits += s["p_stop"] >= threshold
    env.metric("p_stop_early_max", early if seen else None, "1", "lower", render=ctx.render)
    env.metric("early_stop_frames", hits if seen else None, "1", "lower", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["step.p_stop"]))
