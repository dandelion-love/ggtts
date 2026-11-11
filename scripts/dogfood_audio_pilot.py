#!/usr/bin/env python3
"""Compare ggtts instruments with reference measures on retained audio and planted controls.

No input audio is committed. Reports and derived FLACs are written only below --output.
The reference implementation and the ggtts-measure binary are supplied explicitly: --reference is a directory put on
the import path, --levels-module names a module in it exporting `room_stats(samples)` (a dict with `peak_dbfs` and
`rms_dbfs`), and --silence-module one exporting `silence_report(samples, rate, top_db=, gap_min_s=)` (a dict with
`n_gaps`).
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ("reference", "measure", "qwen", "kitten", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("levels-module", "silence-module"):
        ap.add_argument("--" + name, required=True, help="dotted module name under --reference")
    args = ap.parse_args()
    out = args.output.resolve()
    reference = args.reference.resolve()
    root = Path(__file__).resolve().parents[1]
    if out == root or root in out.parents or out == reference or reference in out.parents:
        raise ValueError("--output must be outside both repositories")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(reference))
    levels_module = importlib.import_module(args.levels_module)
    silence_module = importlib.import_module(args.silence_module)
    room_stats, silence_report = levels_module.room_stats, silence_module.silence_report

    env = dict(os.environ, GGTTS_INSTRUMENTS_DIR=str(root / "instruments"), GGTTS_PYTHON=sys.executable)
    results = {}
    for engine in ("qwen", "kitten"):
        src = getattr(args, engine).resolve()
        samples, sr = sf.read(src, dtype="float64")
        if samples.ndim != 1 or sr != 24000:
            raise ValueError("pilot requires mono 24 kHz audio: " + str(src))
        files = {"original": src}
        files["noop"] = out / (engine + "-noop.flac")
        shutil.copyfile(src, files["noop"])
        for variant, data in (("half_gain", samples * 0.5),
                              ("two_second_gap", np.concatenate((samples[:len(samples)//2],
                                                                np.zeros(2 * sr), samples[len(samples)//2:])))):
            path = out / (engine + "-" + variant + ".flac")
            sf.write(path, data, sr, format="FLAC", subtype="PCM_16")
            files[variant] = path
        rows = {}
        for name, path in files.items():
            audio, rate = sf.read(path, dtype="float64")
            report = out / (engine + "-" + name + "-report")
            cmd = [str(args.measure.resolve()), "--audio", str(path), "--report", str(report),
                   "--instrument", "audio-levels", "--instrument", "internal-pauses,arg.min_pause_s=1.5",
                   "--require", "audio-levels", "--require", "internal-pauses"]
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
            if proc.returncode:
                raise RuntimeError("measurement failed: " + proc.stderr)
            def metrics(instrument):
                doc = json.loads((report / "instruments" / instrument / "envelope.json").read_text())
                return {m["name"]: m["value"] for m in doc["metrics"]}
            level = room_stats(audio)
            rows[name] = {"sha256": sha(path), "sample_count": len(audio), "rate": rate,
                          "ggtts_levels": metrics("audio-levels"), "ggtts_pauses": metrics("internal-pauses"),
                          "reference_levels": {k: level[k] for k in ("peak_dbfs", "rms_dbfs")},
                          "reference_silence": silence_report(audio, rate, top_db=40.0, gap_min_s=1.5)}
        results[engine] = rows
        original, noop, gain, gap = (rows[k] for k in ("original", "noop", "half_gain", "two_second_gap"))
        if any(original[k] != noop[k] for k in ("sha256", "ggtts_levels", "ggtts_pauses", "reference_levels", "reference_silence")):
            raise AssertionError(engine + ": no-op changed the measurement")
        for key in ("peak_dbfs", "rms_dbfs"):
            if abs(gain["ggtts_levels"][key] - original["ggtts_levels"][key] + 6.0206) > 0.02:
                raise AssertionError(engine + ": gain plant not detected by ggtts " + key)
            if abs(gain["reference_levels"][key] - original["reference_levels"][key] + 6.0206) > 0.02:
                raise AssertionError(engine + ": gain plant not detected by the reference " + key)
        if not (gap["ggtts_pauses"]["internal_pauses"] > original["ggtts_pauses"]["internal_pauses"] and
                gap["reference_silence"]["n_gaps"] > original["reference_silence"]["n_gaps"]):
            raise AssertionError(engine + ": planted gap not detected by both measures")
    missing = subprocess.run([str(args.measure.resolve()), "--audio", str(out / "missing.flac"),
                              "--report", str(out / "missing-report"), "--instrument", "audio-levels",
                              "--require", "audio-levels"], env=env, text=True, capture_output=True)
    results["missing_input"] = {"exit_code": missing.returncode, "stderr": missing.stderr.strip(),
                                "report_created": (out / "missing-report").exists()}
    if missing.returncode != 2 or (out / "missing-report").exists():
        raise AssertionError("missing audio was not refused before creating a report")
    results["versions"] = {"numpy": np.__version__, "soundfile": sf.__version__,
                           "reference_silence_module_sha256": sha(silence_module.__file__),
                           "reference_levels_module_sha256": sha(levels_module.__file__),
                           "measure_sha256": sha(args.measure)}
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
