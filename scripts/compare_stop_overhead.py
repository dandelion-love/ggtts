#!/usr/bin/env python3
"""Compare two ggtts-render binaries with identical Qwen inputs and a recorded stop trace."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "model", "codec", "story", "work"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3 or args.threads < 1:
        parser.error("at least three repeats and one thread are required")
    work = args.work.resolve()
    paths = {key: getattr(args, key).resolve() for key in ("old", "new", "model", "codec", "story")}
    if ROOT == work or ROOT in work.parents or any(p == work or work in p.parents for p in paths.values()):
        parser.error("work must be outside the checkout and must not contain an input")
    if not all(p.is_file() for p in paths.values()):
        parser.error("all binaries and inputs must exist")
    work.mkdir(parents=True, exist_ok=False)

    def run(which, label):
        base = work / label
        trace, report, audio = (Path(str(base) + suffix) for suffix in (".jsonl", ".timing.json", ".flac"))
        cmd = [str(paths[which]), "--engine", "qwen3-tts", "--device", args.device, "--threads", str(args.threads),
               "--model", str(paths["model"]), "--codec", str(paths["codec"]), "--story", str(paths["story"]),
               "--trace", str(trace), "--trace-level", "1", "--timing-report", str(report), "--out", str(audio),
               "--log-level", "error", "--log-format", "json"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode:
            raise RuntimeError(label + ": render failed, exit " + str(proc.returncode))
        try:
            logs = [json.loads(line) for line in proc.stderr.decode("utf8").splitlines()]
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(label + ": non-JSON process log") from error
        if any(log.get("format") != "ggtts-log/1" for log in logs):
            raise RuntimeError(label + ": unexpected log format")
        events = [json.loads(line) for line in trace.read_text(encoding="utf8").splitlines()]
        semantics = events[0].get("flags", {}).get("p_stop_semantics")
        if which == "new" and semantics != "qwen-sampler-weights/1":
            raise RuntimeError(label + ": corrected trace did not declare its stop-probability semantics")
        if which == "old" and semantics == "qwen-sampler-weights/1":
            raise RuntimeError(label + ": old binary is not a legacy baseline")
        steps = [(e["chunk"], e["frame"], e["token"]) for e in events if e.get("ev") == "step"]
        seeds = [e["seed"] for e in events if e.get("ev") == "chunk_begin"]
        if not steps or not seeds or events[-1].get("ev") != "end":
            raise RuntimeError(label + ": missing step, seed or end evidence")
        timing = json.loads(report.read_text(encoding="utf8"))
        return {"version": which, "p_stop_semantics": semantics, "audio_sha256": sha(audio),
                "steps_sha256": hashlib.sha256(json.dumps(steps, separators=(",", ":")).encode()).hexdigest(),
                "seeds": seeds, "render_ms": timing["render_ms"],
                "generate_ns": timing["stages"]["generate_ns"], "trace_sink_ns": timing["trace_sink"]["sink_ns"]}

    run("old", "warmup-old")
    run("new", "warmup-new")
    rows = []
    for i in range(args.repeats):
        for which in (("old", "new") if i % 2 == 0 else ("new", "old")):
            rows.append(run(which, f"{which}-{i}"))
    for key in ("audio_sha256", "steps_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise RuntimeError(key + " changed across versions or runs")
    if any(row["seeds"] != rows[0]["seeds"] for row in rows):
        raise RuntimeError("draw seeds changed")
    summary = {}
    for which in ("old", "new"):
        summary[which] = {}
        for key in ("render_ms", "generate_ns", "trace_sink_ns"):
            values = [row[key] for row in rows if row["version"] == which]
            summary[which][key] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
    output = {"format": "ggtts-stop-overhead/1", "device_requested": args.device, "threads": args.threads,
              "repeats": args.repeats, "inputs_sha256": {key: sha(path) for key, path in paths.items()},
              "summary": summary, "runs": rows,
              "limits": "One story and host, traced runs only. Generation includes synchronous trace writes; "
                        "wall time is not exclusive GPU compute and GPU scheduling may fall back to CPU. "
                        "Codebook-0 and FLAC identity are checked; other internal tokens are not compared."}
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    (work / "summary.json").write_text(text, encoding="utf8")
    print(text, end="")


if __name__ == "__main__":
    main()
