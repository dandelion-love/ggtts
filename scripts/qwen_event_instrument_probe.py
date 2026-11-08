#!/usr/bin/env python3
"""Compare real Qwen story output with tracing alone and with all shipped event instruments.

This is a partial no-change probe: step events expose codebook 0 only, not every codebook. The separate real
StopTrace test compares full codec-token vectors. Scratch files are created only in a new external directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

TALKER_SHA = "33502b2035f97d7f458248d91e965a6800026549d382c5e8e300176027c70b67"
CODEC_SHA = "7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def catalogue():
    # The template is generated from the Rust catalogue and held in sync by its unit test.
    names = []
    for line in (ROOT / "instruments/instruments.yaml").read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*# - (?:\{name:\s*)?([a-z][a-z0-9-]*)", line)
        if match and match.group(1) != "mine":
            names.append(match.group(1))
    if not names or len(set(names)) != len(names):
        raise ValueError("instrument catalogue template is empty or duplicated")
    return names


def selected_instruments(names):
    # This probe supplies no ASR model or reference. The excluded entry is not a pass.
    excluded = [name for name in names if name == "asr-qwen"]
    return [name for name in names if name not in excluded], excluded


def require_identical(audio_a, audio_b, tokens_a, tokens_b, seeds_a, seeds_b):
    if audio_a != audio_b or tokens_a != tokens_b or seeds_a != seeds_b:
        raise AssertionError("tracing plus event instruments changed FLAC, codebook-0 trace, or draw seeds")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("render", "talker", "codec", "work"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = ap.parse_args()
    talker, codec, binary, work = (p.resolve() for p in (args.talker, args.codec, args.render, args.work))
    if sha(talker) != TALKER_SHA or sha(codec) != CODEC_SHA:
        raise ValueError("model or codec does not match the published v0.0.1 hash")
    if work == ROOT or ROOT in work.parents or any(p == work or work in p.parents for p in (talker, codec, binary)):
        raise ValueError("work must be outside the checkout and must not contain an input")
    work.mkdir(parents=True, exist_ok=False)
    story = work / "story.txt"
    story.write_text("No-change fixture\n小朋友喺海邊砌咗一座好大嘅沙堡。\n", encoding="utf-8")
    names, excluded = selected_instruments(catalogue())
    env = dict(os.environ, GGTTS_INSTRUMENTS_DIR=str(ROOT / "instruments"))
    common = [str(binary), "--engine", "qwen3-tts", "--device", args.device, "--model", str(talker),
              "--codec", str(codec), "--story", str(story)]
    for label in ("baseline", "instruments"):
        cmd = common + ["--out", str(work / (label + ".flac")), "--trace", str(work / (label + ".jsonl"))]
        if label == "instruments":
            cmd += ["--report", str(work / "report")]
            for name in names:
                cmd += ["--instrument", name, "--require", name]
        with (work / (label + ".log")).open("w", encoding="utf-8") as log:
            log.write(json.dumps(cmd, ensure_ascii=False) + "\n")
            log.flush()
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def tokens(label):
        events = [json.loads(line) for line in (work / (label + ".jsonl")).read_text(encoding="utf-8").splitlines()]
        steps = [(e["chunk"], e["frame"], e["token"]) for e in events if e.get("ev") == "step"]
        seeds = [e["seed"] for e in events if e.get("ev") == "chunk_begin"]
        if not steps or not seeds or not any(e.get("ev") == "end" for e in events):
            raise ValueError(label + ": incomplete or empty trace")
        return steps, seeds

    a, seeds_a = tokens("baseline")
    b, seeds_b = tokens("instruments")
    audio_a, audio_b = (sha(work / (label + ".flac")) for label in ("baseline", "instruments"))
    require_identical(audio_a, audio_b, a, b, seeds_a, seeds_b)
    report = json.loads((work / "report/report.json").read_text(encoding="utf-8"))
    if report.get("measurement") != "complete":
        raise AssertionError("required instrument report is not complete")
    summary = {"device": args.device, "talker_sha256": TALKER_SHA, "codec_sha256": CODEC_SHA,
               "render_sha256": sha(binary), "instruments": names, "excluded_instruments": excluded,
               "seeds": seeds_a, "step_count": len(a),
               "codebook0_sha256": hashlib.sha256(json.dumps(a, separators=(",", ":")).encode()).hexdigest(),
               "flac_sha256": audio_a, "report_format": report.get("format"),
               "limits": "One fixture; unconfigured ASR excluded; codebook 0 only; event instruments execute after generation; "
                         "no timing or external-artifact identity; no cross-backend parity."}
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
