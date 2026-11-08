#!/usr/bin/env python3
"""Checkpoint-card loud and soft ends, from pYIN and a speech-core boundary."""
import math
import os
import statistics
import struct
import sys

try:
    import ggtts_instrument  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

from ggtts_instrument import read_audio, run
from core_selection import speech_core

NAME, VERSION = "loudness-ends", "0.1.0"


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def derive(audio):
    """Calculate, in memory, the pYIN track and the lab's edges/2 core selection."""
    import librosa
    import numpy as np

    samples = np.asarray(audio.samples, dtype=np.float32)
    hop = round(0.01 * audio.rate)
    if hop < 1:
        return None
    n = len(samples) // hop
    if not n:
        return None
    frames = np.asarray(samples[:n * hop], dtype=np.float64).reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    core = speech_core(db.tolist())
    if core is None:
        return None
    f0, voiced, _ = librosa.pyin(samples, sr=audio.rate, fmin=65.0, fmax=500.0,
                                  frame_length=1024, hop_length=hop)
    return {"f0_hz": [float(v) if keep and np.isfinite(v) else None for v, keep in zip(f0, voiced)],
            "core_start_s": core[0], "core_end_s": core[1]}


def measure(ctx, env):
    if not ctx.audio:
        env.status("skipped", "audio is required")
        return
    audio = read_audio(ctx.audio)
    try:
        track = derive(audio)
    except ImportError:
        env.status("skipped", "numpy and librosa are required to derive pYIN voicing")
        return
    if track is None:
        env.status("skipped", "no speech core found")
        return
    import librosa
    import numpy
    import scipy
    env.doc["provenance"]["config"].update({
        "voicing_method": "librosa.pyin", "speech_core_method": "edges/2",
        "numpy_version": numpy.__version__, "scipy_version": scipy.__version__,
        "librosa_version": librosa.__version__, "hop_s": 0.01,
        "fmin_hz": 65.0, "fmax_hz": 500.0, "frame_length": 1024,
    })
    env.input(ctx.audio)
    hop = round(0.01 * audio.rate)
    f0 = track.get("f0_hz")
    start, end = track.get("core_start_s"), track.get("core_end_s")
    if (not hop or not isinstance(f0, list) or not all(x is None or
            (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0) for x in f0)
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                       for x in (start, end)) or not 0 <= start < end <= len(audio.samples) / audio.rate):
        env.status("skipped", "invalid voicing values or speech-core boundaries")
        return
    n = len(audio.samples) // hop
    if abs(len(f0) - n) > 1:
        env.status("skipped", "voicing and audio frame counts do not match")
        return
    a, b = int(start / 0.01), int(end / 0.01)
    levels = []
    for i in range(a, min(b, n, len(f0))):
        if f0[i] is not None:
            frame = audio.samples[i * hop:(i + 1) * hop]
            rms = math.sqrt(sum(f32(x) ** 2 for x in frame) / hop)
            levels.append(f32(20 * math.log10(rms + 1e-12)))
    if len(levels) < 20:
        env.status("skipped", "fewer than 20 voiced frames in the speech core")
        return
    centre = f32(statistics.median(levels))
    relative = [f32(x - centre) for x in levels]
    env.metric("voiced_frames", len(levels), "1", "neutral", render=ctx.render)
    env.metric("loud_p90_db", percentile(relative, 0.9), "dB", "neutral", render=ctx.render)
    env.metric("soft_p10_db", percentile(relative, 0.1), "dB", "neutral", render=ctx.render)


if __name__ == "__main__":
    sys.exit(run(measure, NAME, VERSION, hooks=["artifact.audio"], audio_state="unknown"))
