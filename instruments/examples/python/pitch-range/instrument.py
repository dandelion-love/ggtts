#!/usr/bin/env python3
"""Pitch range from the same pinned detector and frame rules as the reference recording booth."""
import array
import json
import os
import shutil
import struct
import subprocess
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_audio, run

NAME, VERSION = "pitch-range", "0.1.0"


def measure(ctx, env):
    if not ctx.audio:
        raise ValueError("--audio is required")
    node = os.environ.get("GGTTS_NODE") or shutil.which("node")
    if not node:
        env.status("skipped", "Node.js is required for the bundled Pitchy detector")
        return
    audio = read_audio(ctx.audio)
    env.input(ctx.audio)
    pcm = array.array("f", audio.samples)
    if sys.byteorder != "little":
        pcm.byteswap()
    command = [node, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitch.js")]
    proc = subprocess.run(command, input=struct.pack("<I", audio.rate) + pcm.tobytes(), capture_output=True,
                          timeout=45)
    if proc.returncode:
        raise ValueError("pitch detector failed: " + proc.stderr.decode("utf8", errors="replace")[-1000:])
    result = json.loads(proc.stdout)
    env.metric("pitch_voiced_frames", result["voiced_frames"], "1", "neutral", render=ctx.render)
    env.metric("pitch_median_hz", result["pitch_median_hz"], "Hz", "neutral", render=ctx.render)
    env.metric("pitch_range_st", result["pitch_range_st"], "semitones", "neutral", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["artifact.audio"], audio_state="unknown"))
