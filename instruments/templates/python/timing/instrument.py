#!/usr/bin/env python3
"""Measure only timings actually recorded by a v2 streaming story trace."""
import os
import math
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_trace, run

NAME, VERSION = "render-timing", "0.1.0"


def measure(ctx, env):
    trace = read_trace(ctx.events)
    env.input(ctx.events).from_trace(trace, ctx.render)
    if not trace.complete:
        env.status("truncated", "the trace has no end event")
        return
    if trace.header.get("format") != "ggtts-trace/2":
        env.status("skipped", "timing requires a v2 trace")
        return
    draws = trace.draws()
    if not draws or any(end is None for _, _, end in draws):
        env.status("skipped", "complete draw timings are unavailable")
        return
    expected = {(begin["chunk"], begin["draw"]) for begin, _, _ in draws}
    if len(expected) != len(draws):
        env.status("skipped", "duplicate draws cannot be attributed")
        return
    timing = trace.of("timing")
    stages = {}
    for stage in ("generate", "decode", "judge"):
        rows = [e for e in timing if e.get("stage") == stage]
        keys = [(e.get("chunk"), e.get("draw")) for e in rows]
        if len(keys) != len(expected) or set(keys) != expected or any(
                not isinstance(e.get("ms"), int) or isinstance(e.get("ms"), bool) or e["ms"] < 0 for e in rows):
            env.status("skipped", "missing or invalid per-draw " + stage + " timing")
            return
        stages[stage] = sum(e["ms"] for e in rows)
    stage_only = ctx.config.get("stage_only", "false")
    if stage_only not in ("true", "false"):
        raise ValueError("stage_only must be true or false")
    ctx.config["stage_only"] = stage_only
    totals = {}
    if stage_only == "false":
        for stage in ("first_audio", "render"):
            rows = [e for e in timing if e.get("stage") == stage]
            if len(rows) != 1 or not isinstance(rows[0].get("ms"), int) or isinstance(rows[0].get("ms"), bool) or rows[0]["ms"] < 0:
                env.status("skipped", "missing or invalid " + stage + " timing; a streaming v2 trace is required")
                return
            totals[stage] = rows[0]["ms"]
    end = trace.of("end")[-1]
    seconds = end.get("seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        env.status("skipped", "positive audio duration is required for real-time factor")
        return
    if totals and totals["first_audio"] > totals["render"]:
        env.status("skipped", "first audio cannot occur after render completion")
        return
    env.metric("ttfa_ms", totals.get("first_audio"), "ms", "lower", render=ctx.render)
    env.metric("render_ms", totals.get("render"), "ms", "lower", render=ctx.render)
    env.metric("render_rtf", totals["render"] / 1000.0 / seconds if totals else None, "1", "lower", render=ctx.render)
    for stage, ms in stages.items():
        env.metric(stage + "_ms_total", ms, "ms", "lower", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["timing", "end"]))
