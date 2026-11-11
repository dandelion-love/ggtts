"""Regenerate a synthetic fixture through a separately compiled ggml reference helper."""
import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("helper", help="C helper linked against the pinned ggml library")
p.add_argument("output", type=Path, help="new output path")
args = p.parse_args()
if args.output.exists():
    p.error("output already exists")
rng = np.random.default_rng(20261107)
rows = rng.uniform(-2, 2, (64, 256)).astype("<f4")
rows[0] = 0
rows[1] = np.resize(np.array([-1, 0, 1], dtype=np.float32), 256) * 0.75
data = subprocess.run([args.helper], input=rows.tobytes(), capture_output=True, check=True).stdout
if len(data) != 64 * (54 + 1024):
    raise ValueError("reference helper returned an unexpected length")
args.output.write_bytes(data)
print("NumPy", np.__version__, "SHA-256", hashlib.sha256(data).hexdigest())
