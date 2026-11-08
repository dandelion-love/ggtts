#!/usr/bin/env python3
"""Measure logging and trace overhead on repeated same-input renders; write only under a fresh scratch directory."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = (("error_no_trace", "error", None), ("info_no_trace", "info", None),
         ("debug_no_trace", "debug", None), ("info_trace_1", "info", 1),
         ("info_trace_2", "info", 2))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("render", "story", "work"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--engine", choices=("fake", "qwen3-tts", "kitten-tts-2"), required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    for name in ("model", "codec", "voice"):
        ap.add_argument("--" + name, type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.repeats < 3 or args.threads < 1:
        ap.error("at least three repeats and one thread are required")
    if args.engine != "fake" and (not args.model or not args.codec):
        ap.error("a real engine needs model and codec")
    if args.engine == "kitten-tts-2" and not args.voice:
        ap.error("kitten needs a prepared voice")
    work = args.work.resolve()
    inputs = [p.resolve() for p in (args.render, args.story, args.model, args.codec, args.voice) if p]
    if work == ROOT or ROOT in work.parents or any(p == work or work in p.parents for p in inputs):
        ap.error("work must be outside the checkout and must not contain an input")
    for path in inputs:
        if not path.is_file():
            ap.error("an input is missing: " + str(path))
    work.mkdir(parents=True, exist_ok=False)
    common = [str(args.render.resolve()), "--engine", args.engine, "--device", args.device,
              "--threads", str(args.threads), "--story", str(args.story.resolve()), "--log-format", "json"]
    for flag in ("model", "codec", "voice"):
        path = getattr(args, flag)
        if path:
            common += ["--" + flag, str(path.resolve())]

    def run(label, level, trace_level):
        base = work / label
        cmd = common + ["--out", str(base) + ".flac", "--stream", str(base) + ".wav",
                        "--timing-report", str(base) + ".timing.json", "--log-level", level]
        if trace_level:
            cmd += ["--trace", str(base) + ".trace.jsonl", "--trace-level", str(trace_level)]
        start = time.perf_counter_ns()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        elapsed = time.perf_counter_ns() - start
        if proc.returncode:
            raise RuntimeError(label + ": render failed, exit " + str(proc.returncode))
        try:
            logs = [json.loads(line) for line in proc.stderr.decode("utf8").splitlines()]
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(label + ": non-JSON stderr from the process") from error
        if any(item.get("format") != "ggtts-log/1" for item in logs):
            raise RuntimeError(label + ": unexpected log record")
        timing = json.loads(Path(str(base) + ".timing.json").read_text(encoding="utf8"))
        if timing.get("format") != "ggtts-timing/1" or not timing.get("stages"):
            raise RuntimeError(label + ": incomplete timing report")
        trace_tokens, trace_seeds, trace_bytes = None, None, None
        if trace_level:
            trace_path = Path(str(base) + ".trace.jsonl")
            trace_bytes = trace_path.stat().st_size
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf8").splitlines()]
            tokens = [(e["chunk"], e["frame"], e["token"]) for e in events if e.get("ev") == "step"]
            trace_seeds = [e["seed"] for e in events if e.get("ev") == "chunk_begin"]
            if not trace_seeds or events[-1].get("ev") != "end":
                raise RuntimeError(label + ": trace lacks required seed or end evidence")
            trace_tokens = (hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
                            if tokens else None)
        return {"mode": label.rsplit("-", 1)[0], "process_ns": elapsed,
                "audio_sha256": sha(Path(str(base) + ".flac")), "trace_codebook0_sha256": trace_tokens,
                "trace_seeds": trace_seeds, "trace_bytes": trace_bytes, "stderr_bytes": len(proc.stderr),
                "timing": timing}

    # One unreported warmup, then rotate the mode order to reduce order and thermal bias.
    run("warmup-0", "error", None)
    rows = []
    for repeat in range(args.repeats):
        modes = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
        for mode, level, trace_level in modes:
            rows.append(run(mode + "-" + str(repeat), level, trace_level))
    hashes = {row["audio_sha256"] for row in rows}
    frames = {row["timing"]["frames_generated"] for row in rows}
    if len(hashes) != 1 or len(frames) != 1:
        raise RuntimeError("audio or generated frame count changed across modes")
    traced = [row for row in rows if row["trace_seeds"] is not None]
    if len({row["trace_codebook0_sha256"] for row in traced}) != 1 or any(
            row["trace_seeds"] != traced[0]["trace_seeds"] for row in traced):
        raise RuntimeError("traced modes changed codebook-0 tokens or draw seeds")
    summary = {}
    for mode, _, _ in MODES:
        runs = [row["timing"] for row in rows if row["mode"] == mode]
        for key in ("render_ms",):
            numbers = [r[key] for r in runs]
            summary.setdefault(mode, {})[key] = {"median": statistics.median(numbers),
                                                 "min": min(numbers), "max": max(numbers)}
        speeds = [r["stages"]["generated_frames_per_second"] for r in runs]
        if any(value is None for value in speeds):
            raise RuntimeError(mode + ": generation throughput is unavailable")
        summary[mode]["frames_per_second"] = {"median": statistics.median(speeds),
                                               "min": min(speeds), "max": max(speeds)}
    baseline = summary["error_no_trace"]["render_ms"]["median"]
    for data in summary.values():
        value = data["render_ms"]["median"]
        data["render_delta_ms_vs_error_no_trace"] = value - baseline
        data["render_delta_pct_vs_error_no_trace"] = ((value - baseline) / baseline * 100) if baseline else None
    output = {"format": "ggtts-logging-probe/1", "engine": args.engine, "device_requested": args.device,
              "threads": args.threads, "repeats": args.repeats, "inputs_sha256": {name: sha(getattr(args, name).resolve())
              for name in ("render", "story", "model", "codec", "voice") if getattr(args, name)},
              "summary": summary, "runs": rows,
              "limits": "One story and host; same-backend comparison only; GPU scheduling may use CPU fallback; "
                        "generation is host wall time including observers, not exclusive GPU compute. Timing "
                        "instrumentation is enabled in every mode. Token equality is checked between traced modes "
                        "only when the engine emits step tokens; untraced modes have audio/frame evidence only. "
                        "External child log costs are not isolated."}
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    (work / "summary.json").write_text(text, encoding="utf8")
    print(text, end="")


if __name__ == "__main__":
    main()
