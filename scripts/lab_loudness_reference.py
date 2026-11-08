#!/usr/bin/env python3
"""Run the lab's checkpoint-card loud/soft-end calculation on one external audio file.

This read-only probe loads pinned companion analysis code and writes aggregate values to stdout. It does not save
audio or text. NumPy, SciPy, soundfile and librosa are needed in the chosen Python environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--companion", type=Path, required=True, help="cantonese-speech-analysis checkout")
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--steps-sha256", required=True)
    ap.add_argument("--store-sha256", required=True)
    ap.add_argument("--export-voicing", type=Path, help="write the derived pYIN track to a new external JSON file")
    args = ap.parse_args()
    canto = args.companion.resolve() / "scripts/analysis/canto"
    for name, expected in (("steps.py", args.steps_sha256), ("store.py", args.store_sha256)):
        if digest(canto / name) != expected:
            raise ValueError(name + " does not match the supplied reference hash")
    sys.path.insert(0, str(canto))
    import numpy as np
    import steps
    import librosa
    import scipy
    import soundfile

    clip = {"path": str(args.audio.resolve())}
    energy, energy_arrays = steps.energy(clip, {}, {"hop_s": 0.01, "floor_db": -40.0,
                                                  "merge_s": 0.15, "min_s": 0.04})
    edges, _ = steps.edges(clip, {"energy": energy}, steps.EDGES_PARAMS)
    result = {"audio_sha256": digest(args.audio), "steps_sha256": args.steps_sha256,
              "store_sha256": args.store_sha256, "versions": {"numpy": np.__version__,
              "librosa": librosa.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__},
              "speech": edges["speech"], "loud_p90_db": None, "soft_p10_db": None, "voiced_frames": 0}
    if edges["speech"]:
        _, f0_arrays = steps.f0(clip, {}, {"hop_s": 0.01, "fmin": 65.0, "fmax": 500.0, "frame_length": 1024})
        core_start, core_end = edges["core_start_s"], edges["core_end_s"]
        a, b = int(core_start / 0.01), int(core_end / 0.01)
        f0, db = f0_arrays["f0"][a:b], energy_arrays["db"][a:b]
        n = min(len(f0), len(db))
        f0, db = f0[:n], db[:n]
        voiced = ~np.isnan(f0)
        result.update(core_start_s=core_start, core_end_s=core_end,
                      voiced_frames=int(voiced.sum()))
        if args.export_voicing:
            out = args.export_voicing.resolve()
            repo = Path(__file__).resolve().parents[1]
            companion = args.companion.resolve()
            if out == repo or repo in out.parents or out == companion or companion in out.parents:
                raise ValueError("voicing output must be outside both checkouts")
            track = {"format": "ggtts-voicing/1", "audio_sha256": result["audio_sha256"], "hop_s": 0.01,
                     "method": "librosa.pyin", "fmin_hz": 65.0, "fmax_hz": 500.0, "frame_length": 1024,
                     "core_method": "canto-edges/2", "steps_sha256": args.steps_sha256,
                     "store_sha256": args.store_sha256, "librosa_version": librosa.__version__,
                     "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                     "soundfile_version": soundfile.__version__,
                     "core_start_s": core_start, "core_end_s": core_end,
                     "f0_hz": [None if np.isnan(v) else float(v) for v in f0_arrays["f0"]]}
            with out.open("x", encoding="utf8") as dest:
                json.dump(track, dest, allow_nan=False)
        if voiced.sum() >= 20:
            levels = db[voiced] - np.median(db[voiced])
            result.update(loud_p90_db=float(np.percentile(levels, 90)),
                          soft_p10_db=float(np.percentile(levels, 10)))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
