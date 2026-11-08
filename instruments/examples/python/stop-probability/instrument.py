#!/usr/bin/env python3
"""Export Qwen's versioned stop weight for each observed draw step."""
import json
import math
import os
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_trace, run
from ggtts_instrument.audio import sha256_of

NAME, VERSION = "stop-probability", "0.1.0"
SEMANTICS = "qwen-sampler-weights/1"


def collect(trace):
    """Validate draw boundaries and return data only when EOS semantics are known."""
    if not trace.complete:
        return None, "truncated", "the trace has no end event"
    flags = trace.header.get("flags")
    if (trace.header.get("engine") != "qwen3-tts" or not isinstance(flags, dict)
            or flags.get("p_stop_semantics") != SEMANTICS):
        return None, "skipped", "Qwen sampler-weight semantics are unavailable or unmarked"
    eos = flags.get("codec_eos_token")
    if not isinstance(eos, int) or isinstance(eos, bool) or not 0 <= eos <= 65535:
        return None, "skipped", "the producer did not declare its codec EOS token"
    rows, current, seen = [], None, set()
    for event in trace.events:
        kind = event.get("ev")
        if kind == "chunk_begin":
            key = (event.get("chunk"), event.get("draw"))
            seed = event.get("seed")
            if current is not None or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= 4294967295
                                          for x in key):
                return None, "skipped", "invalid or overlapping draw boundaries"
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 18446744073709551615:
                return None, "skipped", "draw seed is missing or invalid"
            if key in seen:
                return None, "skipped", "duplicate draw"
            seen.add(key)
            current = {"chunk": key[0], "draw": key[1], "steps": []}
        elif kind == "step":
            if current is None or event.get("chunk") != current["chunk"]:
                return None, "skipped", "step outside its draw"
            frame, p, token = event.get("frame"), event.get("p_stop"), event.get("token")
            if (not isinstance(frame, int) or isinstance(frame, bool) or frame > 4294967295
                    or frame != len(current["steps"])
                    or not isinstance(token, int) or isinstance(token, bool) or not 0 <= token <= 65535
                    or not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p)
                    or not 0 <= p <= 1):
                return None, "skipped", "missing, invalid or non-contiguous step probability"
            current["steps"].append({"chunk": current["chunk"], "draw": current["draw"],
                                     "step": frame, "token": token, "p_stop": p, "terminal_eos": False})
        elif kind == "chunk_end":
            if (current is None or event.get("chunk") != current["chunk"]
                    or event.get("draw") != current["draw"] or not current["steps"]):
                return None, "skipped", "missing step data or mismatched draw end"
            stop = event.get("stop")
            frames = event.get("frames")
            if not isinstance(frames, int) or isinstance(frames, bool) or not 0 <= frames <= 4294967295:
                return None, "skipped", "generated frame count is missing or invalid"
            if stop not in ("eos", "cap"):
                return None, "skipped", "draw did not finish with EOS or a frame cap"
            eos_steps = [i for i, step in enumerate(current["steps"]) if step["token"] == eos]
            if (stop == "eos" and eos_steps != [len(current["steps"]) - 1]) or (stop == "cap" and eos_steps):
                return None, "skipped", "terminal EOS step is missing or inconsistent with draw end"
            if stop == "eos":
                # Qwen's declared producer contract emits the EOS token as its final step.
                current["steps"][-1]["terminal_eos"] = True
            for step in current["steps"]:
                step["draw_stop"] = stop
            rows.extend(current["steps"])
            current = None
    if current is not None or not rows:
        return None, "skipped", "a draw is incomplete or no steps are available"
    return rows, "ok", None


def measure(ctx, env):
    if not ctx.events or ctx.events == "-":
        env.status("skipped", "a retained event trace file is required for provenance")
        return
    before = sha256_of(ctx.events)
    trace = read_trace(ctx.events)
    env.input(ctx.events)
    if env.doc["provenance"]["inputs"][-1]["sha256"] != before:
        env.status("skipped", "the trace changed while being read")
        return
    rows, status, error = collect(trace)
    if rows is None:
        env.status(status, error)
        return
    env.from_trace(trace, ctx.render)
    ctx.config["p_stop_semantics"] = SEMANTICS
    path = os.path.join(os.path.dirname(os.path.abspath(ctx.envelope)), "stop-probability.jsonl")
    with open(path, "x", encoding="utf8") as out:
        for row in rows:
            out.write(json.dumps({"format": "ggtts-stop-probability/1", "p_stop_semantics": SEMANTICS, **row},
                                 allow_nan=False) + "\n")
    env.artifact(path, "application/x-ndjson")
    env.metric("stop_steps", len(rows), "1", "neutral", render=ctx.render)
    env.metric("terminal_eos_draws", sum(row["terminal_eos"] for row in rows), "1", "neutral", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["step.p_stop"]))
