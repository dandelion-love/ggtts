#!/usr/bin/env python3
"""Compare speech-core selection with a local lab checkout on deterministic synthetic level tracks."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ggtts_core_selection", ROOT / "instruments/examples/python/loudness-ends/core_selection.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path, required=True)
    args = parser.parse_args()
    reference = args.companion.resolve() / "scripts/analysis/canto"
    for name, expected in (("steps.py", "a0d3c0b265f51fb3e9d96fcf5ea6e746ed7862203bda87fb37033595f779ee41"),
                           ("store.py", "59619c36714fb00f3609c279da5dd877130f4e91ecd2494d032613f4ba6a75e7")):
        if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
            raise ValueError(name + " does not match the recorded lab reference")
    sys.path.insert(0, str(reference))
    import steps

    rng = random.Random(20261108)
    cases = [[], [-100.0] * 50, [-120.0] * 10 + [-10.0] * 60 + [-120.0] * 10]
    for _ in range(1000):
        levels = []
        for _ in range(rng.randint(1, 30)):
            levels.extend([rng.uniform(-120.0, 0.0)] * rng.randint(1, 80))
        cases.append(levels)
    for index, levels in enumerate(cases):
        db = np.asarray(levels, dtype=np.float64)
        initial = steps.segments(db, 0.01, -40.0, 0.15, 0.04)
        fine = steps.segments(db, 0.01, -40.0, 0.05, 0.04) if initial else []
        expected = steps.edge_core(fine, db, steps.EDGES_PARAMS) if fine else None
        actual = candidate.speech_core(levels)
        if actual != expected:
            raise AssertionError(f"case {index}: lab {expected}, ggtts {actual}")
    print(f"Matched lab edges/2 speech core for {len(cases)} synthetic level tracks.")


if __name__ == "__main__":
    main()
