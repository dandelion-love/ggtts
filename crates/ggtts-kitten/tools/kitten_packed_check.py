#!/usr/bin/env python3
"""The converter's ternary matrices against Kitten's own packer, value for value and byte for byte.

Kitten publishes its language model as bf16 weights (`lm/model.safetensors`), not as a packed file. The packed form
is what Kitten's own exporter writes from those weights: kitten-tts-2-cpp's `tools/kitten-tts/export.py`, whose
default `--outtype tq2_1` quantizes through its `gguf-py` (`gguf/quants.py`, class `TQ2_1`: two fp16 scales per 256
weights, one per 128), and whose `--outtype ternary-q4_0` packs through `pack_ternary_q4` (the same `Q4_0` layout
this engine writes). This script runs both of Kitten's packers, from a kitten-tts-2-cpp checkout the user gives, on
the same weights the converter read, and compares them with the converter's GGUF:

  1. TQ2_1 values: every trit, and every per-128 scale's fp16 bits, equal to what Kitten's TQ2_1 packer writes.
  2. Q4_0 bytes: every 18-byte block equal to what Kitten's `pack_ternary_q4` writes. A block that differs is
     reported, and classified: an all-zero block of 32 whose scale field differs decodes to zeros either way.

It needs numpy and nothing else; no torch. It reads a Kitten model and a GGUF and writes nothing. Exit 0 only when
check 1 passes on every matrix and check 2 finds no block that differs in its codes.

  python tools/kitten_packed_check.py --kitten-cpp DIR --weights lm/model.safetensors --gguf LM.gguf [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_bf16(path: Path):
    """The safetensors header and a memory map; tensors are widened from bf16 to f32 on demand."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    base = 8 + n

    def get(name: str) -> np.ndarray:
        t = header[name]
        if t["dtype"] != "BF16":
            raise ValueError(f"{name}: {t['dtype']}, expected BF16")
        a, b = t["data_offsets"]
        raw = np.frombuffer(mm[base + a : base + b], dtype="<u2")
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(t["shape"])

    return header, get


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kitten-cpp", type=Path, required=True, help="a kitten-tts-2-cpp checkout (its gguf-py and export.py)")
    p.add_argument("--weights", type=Path, required=True, help="the safetensors the GGUF was converted from")
    p.add_argument("--gguf", type=Path, required=True, help="the converter's GGUF (Q4_0 layout)")
    p.add_argument("--limit", type=int, default=0, help="check only the first N ternary matrices (0: all)")
    a = p.parse_args()

    sys.path.insert(0, str(a.kitten_cpp / "gguf-py"))
    import gguf  # Kitten's fork, with TQ2_1

    spec = importlib.util.spec_from_file_location("kitten_export", a.kitten_cpp / "tools" / "kitten-tts" / "export.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)  # top level: stdlib imports and a sys.path change only

    reader = gguf.GGUFReader(a.gguf)
    ours = {t.name: t for t in reader.tensors}
    _, get = load_bf16(a.weights)
    # Kitten's rule for which tensors are ternary: model.layers.* with two dimensions (export.py, tensor_force_quant).
    names = sorted(
        n for n, t in ours.items() if n.startswith("model.layers.") and t.tensor_type == gguf.GGMLQuantizationType.Q4_0
    )
    if a.limit:
        names = names[: a.limit]
    if not names:
        print("no Q4_0 ternary matrices in the GGUF (convert with the default --layout q4_0)")
        return 2

    tot = dict(matrices=0, groups=0, trits=0, q4_blocks=0, q4_equal=0, q4_zero_scale_only=0)
    bad: list[str] = []
    for name in names:
        w = get(name).astype(np.float32)
        rows, cols = w.shape
        mine = np.asarray(ours[name].data, dtype=np.uint8).reshape(rows, cols // 32 * 18)

        # Ours, decoded: per 32 an fp16 scale and 32 nibbles (low halves first), code 8 meaning 0.
        blk = mine.reshape(-1, 18)
        d32 = blk[:, :2].copy().view("<u2").reshape(-1)
        nib = np.concatenate([blk[:, 2:] & 0x0F, blk[:, 2:] >> 4], axis=1).astype(np.int8) - 8
        our_trits = nib.reshape(rows, cols)
        d128 = d32.reshape(-1, 4)
        if not (d128 == d128[:, :1]).all():
            bad.append(f"{name}: the four Q4_0 scales of a 128-group differ")
        our_scale = d128[:, 0]

        # Kitten's TQ2_1, decoded as its own dequantize_blocks reads it.
        k = gguf.quants.quantize(w, gguf.GGMLQuantizationType.TQ2_1).reshape(-1, 68)
        k_scale = k[:, 64:].copy().view("<u2").reshape(-1)
        qs = k[:, :64].reshape(-1, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
        k_trits = ((qs & 3).reshape(rows, cols).astype(np.int8)) - 1

        if not np.array_equal(our_trits, k_trits):
            r, c = np.argwhere(our_trits != k_trits)[0]
            bad.append(f"{name}: trit at row {r} column {c} is {our_trits[r, c]}, Kitten's TQ2_1 has {k_trits[r, c]}")
        if not np.array_equal(our_scale, k_scale):
            i = int(np.argwhere(our_scale != k_scale)[0][0])
            bad.append(f"{name}: scale of group {i} is 0x{our_scale[i]:04x}, Kitten's TQ2_1 has 0x{k_scale[i]:04x}")

        # Kitten's ternary-q4_0 packer, byte for byte.
        kq = export.pack_ternary_q4(w).reshape(-1, 18)
        same = (kq == blk).all(axis=1)
        diff = ~same
        codes_same = (kq[:, 2:] == blk[:, 2:]).all(axis=1)
        zero_block = (blk[:, 2:] == 0x88).all(axis=1)
        scale_only_zero = diff & codes_same & zero_block
        if (diff & ~scale_only_zero).any():
            i = int(np.argwhere(diff & ~scale_only_zero)[0][0])
            bad.append(f"{name}: Q4_0 block {i} differs from Kitten's pack_ternary_q4 beyond an all-zero block's scale")

        tot["matrices"] += 1
        tot["groups"] += our_scale.size
        tot["trits"] += our_trits.size
        tot["q4_blocks"] += blk.shape[0]
        tot["q4_equal"] += int(same.sum())
        tot["q4_zero_scale_only"] += int(scale_only_zero.sum())

    print(
        f"TQ2_1 values: {tot['matrices']} matrices, {tot['trits']} trits and {tot['groups']} scales of 128 compared "
        f"with Kitten's TQ2_1 packer"
    )
    print(
        f"Q4_0 bytes: {tot['q4_equal']} of {tot['q4_blocks']} blocks byte-identical to Kitten's pack_ternary_q4; "
        f"{tot['q4_zero_scale_only']} differ only in the scale field of an all-zero block (both decode to zeros)"
    )
    for b in bad[:20]:
        print("  " + b)
    print(f"PACKED {'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
