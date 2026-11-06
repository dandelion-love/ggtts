"""Helpers for an instrument's self-test: a render to measure, ways to plant a defect in it, and running the
instrument exactly as the runner does (a child process with the runner's flags).

The render comes from `GGTTS_FIXTURE_DIR` when it is set: `cargo test -p ggtts-core --test python_instruments`
renders the recorded pipeline cases on the fake engine and points it there. It holds `trace.jsonl` and `audio.flac`,
as the runner would hand them over. Without it, `synthetic()` writes a small render of the same shape, so a template's
self-test runs with Python alone.
"""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile

from .envelope import check

HERE = os.path.dirname(os.path.abspath(__file__))


class Render:
    def __init__(self, trace, audio, source):
        self.trace = trace
        self.audio = audio
        self.source = source


def write_wav(path, samples, rate):
    """Mono 16-bit PCM WAV, scaled by 32768 to mirror `read_audio`: a sample read from 16-bit audio round-trips."""
    pcm = struct.pack("<%dh" % len(samples), *(max(-32768, min(32767, int(round(x * 32768)))) for x in samples))
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", len(pcm)) + pcm)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")


def synthetic(directory):
    """A two-chunk render: a trace shaped like the fake engine's, and its audio with one join gap."""
    rate = 24000
    lines = [{"ev": "trace", "format": "ggtts-trace/1", "ggtts": "0.0.1", "ggml": "af97976", "engine": "fake",
              "backend": "cpu", "parity": "exact", "protocol": "story-v1", "flags": {}, "trace_level": 1, "t_ms": 0}]
    audio, t = [], 1
    for chunk, n in enumerate((30, 40)):
        lines.append({"ev": "chunk_begin", "chunk": chunk, "draw": 0, "text": "synthetic", "seed": 100 + chunk,
                      "t_ms": t})
        for f in range(n):
            p = 0.001 + 0.989 * ((f + 1) / float(n)) ** 8
            lines.append({"ev": "step", "chunk": chunk, "frame": f, "token": (f * 7 + chunk) % 2048, "p_stop": p,
                          "entropy": 1.0, "top1_p": 0.5, "rank": 0, "t_ms": t})
        lines.append({"ev": "chunk_end", "chunk": chunk, "draw": 0, "frames": n, "stop": "eos", "spc": 0.3,
                      "verdict": None, "ms": 2, "t_ms": t + 1})
        lines.append({"ev": "trim", "chunk": chunk, "lead_s": 0.2, "tail_s": 0.3, "start": 4800, "end": 4800 + n * 960,
                      "t_ms": t + 1})
        t += 2
        tone = [0.3 * math.sin(2 * math.pi * 220 * i / rate) for i in range(int(rate * n / 25.0))]
        audio += ([0.0] * int(0.12 * rate)) + tone + ([0.0] * int(0.12 * rate))
        if chunk == 0:
            audio += [0.0] * int(0.3 * rate)
    lines.append({"ev": "join", "chunks": 2, "samples": len(audio), "gain_db": -1.5, "t_ms": t})
    lines.append({"ev": "end", "frames": 70, "seconds": len(audio) / float(rate), "t_ms": t})
    write_lines(os.path.join(directory, "trace.jsonl"), lines)
    write_wav(os.path.join(directory, "audio.wav"), audio, rate)
    return Render(os.path.join(directory, "trace.jsonl"), os.path.join(directory, "audio.wav"), "synthetic")


def fixture(directory):
    """The render to measure: the fake engine's when `GGTTS_FIXTURE_DIR` is set, else a synthetic one."""
    d = os.environ.get("GGTTS_FIXTURE_DIR")
    if d:
        return Render(os.path.join(d, "trace.jsonl"), os.path.join(d, "audio.flac"), "fake engine")
    return synthetic(directory)


def run_instrument(script, directory, events=None, audio=None, extra=()):
    """Run `script` as the runner would; its envelope as a dict, after checking it is valid."""
    out = os.path.join(directory, "envelope-%d.json" % len(os.listdir(directory)))
    argv = [sys.executable, script, "--envelope", out, "--run", "selftest"]
    if events:
        argv += ["--events", events]
    if audio:
        argv += ["--audio", audio]
    argv += list(extra)
    p = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if not os.path.exists(out):
        raise AssertionError("no envelope written; exit %d\n%s" % (p.returncode, p.stderr))
    with open(out, encoding="utf-8") as f:
        env = json.load(f)
    why = check(env)
    if why:
        raise AssertionError("invalid envelope: %s" % why)
    return env


def metrics(env):
    """{(name, render, chunk): value} for an envelope's metrics."""
    return {(m["name"], m.get("render"), m.get("chunk")): m["value"] for m in env["metrics"]}


def shift_clock(lines, by=1000):
    """The same trace with every wall-clock field moved: a no-op for any instrument."""
    out = []
    for l in lines:
        l = dict(l)
        l["t_ms"] = l["t_ms"] + by
        if "ms" in l:
            l["ms"] = l["ms"] + by
        out.append(l)
    return out


class Workdir:
    """A temporary directory for one test."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ggtts-instrument-")
        return self.tmp.name

    def __exit__(self, *a):
        self.tmp.cleanup()
